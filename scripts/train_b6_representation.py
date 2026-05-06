#!/usr/bin/env python3
"""Train the B6 hierarchy model with weak pretraining + gold fine-tuning.

This script implements the full representation-learning workflow:
1) weak-label pretraining on large Reddit weak labels
2) gold fine-tuning on 3-annotator aggregates with hard/soft/hybrid loss
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import re
import sys
from datetime import datetime, timezone
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

# Make the repo root importable when executed as `python scripts/...`.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.b6_hierarchy import B6HierarchyModel


TASK_SPECS = {
    "relevance": {
        "values": ["0", "1"],
        "logits_key": "relevance_logits",
    },
    "target": {
        "values": [
            "government_institutions",
            "political_party_or_actor",
            "policy_or_legislation",
            "social_group_or_identity",
            "foreign_actor_or_geopolitics",
            "media_or_information",
            "other",
        ],
        "logits_key": "target_logits",
    },
    "stance": {
        "values": ["support", "oppose", "mixed_or_unclear"],
        "logits_key": "stance_logits",
    },
    "frame": {
        "values": [
            "economic_freedom",
            "economic_redistribution",
            "law_and_order",
            "civil_rights",
            "national_security",
            "environmental_protection",
            "public_health",
            "identity_and_values",
            "other",
        ],
        "logits_key": "frame_logits",
    },
    "economic_direction": {
        "values": ["-1", "0", "1"],
        "logits_key": "econ_logits",
    },
    "social_direction": {
        "values": ["-1", "0", "1"],
        "logits_key": "social_logits",
    },
    "intensity": {
        "values": ["0", "1", "2"],
        "logits_key": "intensity_logits",
    },
    "ambiguity": {
        "values": ["0", "1", "2"],
        "logits_key": "ambiguity_logits",
    },
}


STAGE1_TASKS = list(TASK_SPECS.keys())
STAGE2_TASKS = list(TASK_SPECS.keys())
IDEOLOGY_TASKS = ["economic_direction", "social_direction", "intensity", "ambiguity"]
IGNORE_INDEX = -100


class SimpleHashTokenizer:
    """Offline fallback tokenizer for quick local checks and restricted environments."""

    def __init__(self, vocab_size: int = 8192) -> None:
        self.vocab_size = max(1024, int(vocab_size))
        self.pad_id = 0
        self.cls_id = 1
        self.sep_id = 2

    def _token_to_id(self, token: str) -> int:
        return 3 + (hash(token) % (self.vocab_size - 3))

    def __call__(
        self,
        text: str,
        truncation: bool = True,
        padding: str = "max_length",
        max_length: int = 256,
        return_tensors: str = "pt",
    ) -> Dict[str, torch.Tensor]:
        tokens = re.findall(r"[A-Za-z0-9_]+|[^\sA-Za-z0-9_]", text.lower())
        token_ids = [self._token_to_id(t) for t in tokens]
        if truncation:
            token_ids = token_ids[: max(0, max_length - 2)]
        ids = [self.cls_id] + token_ids + [self.sep_id]
        ids = ids[:max_length]
        attn = [1] * len(ids)
        if padding == "max_length" and len(ids) < max_length:
            pad_n = max_length - len(ids)
            ids.extend([self.pad_id] * pad_n)
            attn.extend([0] * pad_n)
        if return_tensors != "pt":
            raise ValueError("SimpleHashTokenizer only supports return_tensors='pt'")
        return {
            "input_ids": torch.tensor([ids], dtype=torch.long),
            "attention_mask": torch.tensor([attn], dtype=torch.long),
        }


def read_csv(path: Path) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def parse_json_field(raw: str) -> Dict:
    if not raw:
        return {}
    return json.loads(raw)


def to_float(value: str, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def bound01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_created_at(value: str) -> float:
    raw = (value or "").strip()
    if not raw:
        return -1.0
    try:
        dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return float(dt.timestamp())
    except ValueError:
        return -1.0


def stable_user_hash(user_id: str) -> int:
    raw = (user_id or "").strip()
    if not raw:
        return -1
    digest = hashlib.md5(raw.encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def weak_relevance(low_quality: str, target: str, frame: str) -> str:
    likely_irrelevant = str(low_quality).strip() == "1" and target == "other" and frame == "other"
    return "0" if likely_irrelevant else "1"


def weak_direction(score: float, neutral_band: float = 0.10) -> str:
    if score > neutral_band:
        return "1"
    if score < -neutral_band:
        return "-1"
    return "0"


def weak_intensity(score_abs: float) -> str:
    if score_abs >= 0.60:
        return "2"
    if score_abs >= 0.25:
        return "1"
    return "0"


def weak_ambiguity(q_score: float) -> str:
    if q_score >= 0.65:
        return "0"
    if q_score >= 0.30:
        return "1"
    return "2"


def normalize_stance(raw: str) -> str:
    raw = (raw or "").strip()
    if raw in {"support_progressive", "support_conservative", "support"}:
        return "support"
    if raw == "oppose":
        return "oppose"
    return "mixed_or_unclear"


def encode_label(task: str, value: str) -> int:
    spec = TASK_SPECS[task]
    value_to_idx = {v: i for i, v in enumerate(spec["values"])}
    return value_to_idx.get(str(value).strip(), IGNORE_INDEX)


@dataclass
class TrainingRecord:
    post_id: str
    subreddit: str
    created_at: str
    user_id: str
    text: str
    hard_labels: Dict[str, int]
    soft_labels: Dict[str, List[float]]
    soft_mask: Dict[str, int]
    disagreement_weight: float
    quality_score: float


class RepresentationDataset(Dataset):
    def __init__(
        self,
        records: List[TrainingRecord],
        tokenizer,
        max_length: int,
        tasks: List[str],
    ) -> None:
        self.records = records
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.tasks = tasks

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        rec = self.records[idx]
        enc = self.tokenizer(
            rec.text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )

        item: Dict[str, torch.Tensor] = {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "disagreement_weight": torch.tensor(rec.disagreement_weight, dtype=torch.float32),
            "quality_score": torch.tensor(rec.quality_score, dtype=torch.float32),
            "user_hash": torch.tensor(stable_user_hash(rec.user_id), dtype=torch.long),
            "created_ts": torch.tensor(parse_created_at(rec.created_at), dtype=torch.float32),
        }
        for task in self.tasks:
            item[f"hard_{task}"] = torch.tensor(rec.hard_labels[task], dtype=torch.long)
            item[f"soft_{task}"] = torch.tensor(rec.soft_labels[task], dtype=torch.float32)
            item[f"soft_mask_{task}"] = torch.tensor(rec.soft_mask[task], dtype=torch.bool)
        return item


def split_records(
    records: List[TrainingRecord],
    val_fraction: float,
    seed: int,
) -> Tuple[List[TrainingRecord], List[TrainingRecord]]:
    if val_fraction <= 0 or len(records) < 2:
        return records, []
    local = list(records)
    random.Random(seed).shuffle(local)
    n_val = max(1, int(round(len(local) * val_fraction)))
    n_val = min(n_val, len(local) - 1)
    return local[n_val:], local[:n_val]


def build_weak_records(
    raw_rows: List[Dict[str, str]],
    weak_rows: List[Dict[str, str]],
    max_samples: int,
) -> List[TrainingRecord]:
    weak_by_post = {r["post_id"]: r for r in weak_rows}
    out: List[TrainingRecord] = []
    for raw in raw_rows:
        post_id = raw.get("post_id", "")
        weak = weak_by_post.get(post_id)
        if not weak:
            continue

        text = (raw.get("text") or "").strip()
        if not text:
            continue

        target = weak.get("target_hypothesis", "other")
        stance = normalize_stance(weak.get("stance_hypothesis", "mixed_or_unclear"))
        frame = weak.get("frame_hypothesis", "other")
        q_score = to_float(weak.get("q_score", "0"))
        zecon = to_float(weak.get("projected_zecon", "0"))
        zsoc = to_float(weak.get("projected_zsoc", "0"))
        intensity_signal = max(abs(zecon), abs(zsoc))

        weak_label_values = {
            "relevance": weak_relevance(weak.get("low_quality", ""), target, frame),
            "target": target,
            "stance": stance,
            "frame": frame,
            "economic_direction": weak_direction(zecon),
            "social_direction": weak_direction(zsoc),
            "intensity": weak_intensity(intensity_signal),
            "ambiguity": weak_ambiguity(q_score),
        }

        hard_labels = {task: encode_label(task, weak_label_values[task]) for task in STAGE1_TASKS}
        soft_labels = {}
        soft_mask = {}
        for task in STAGE1_TASKS:
            n = len(TASK_SPECS[task]["values"])
            soft_labels[task] = [0.0] * n
            soft_mask[task] = 0

        out.append(
            TrainingRecord(
                post_id=raw.get("post_id", ""),
                subreddit=raw.get("subreddit", ""),
                created_at=raw.get("created_at", ""),
                user_id=raw.get("user_id", ""),
                text=text,
                hard_labels=hard_labels,
                soft_labels=soft_labels,
                soft_mask=soft_mask,
                disagreement_weight=1.0,
                quality_score=bound01(q_score),
            )
        )
        if max_samples > 0 and len(out) >= max_samples:
            break
    return out


def soft_vector(task: str, dist: Dict[str, float]) -> Tuple[List[float], int]:
    values = TASK_SPECS[task]["values"]
    idx = {v: i for i, v in enumerate(values)}
    vec = [0.0] * len(values)
    total = 0.0

    for k, v in dist.items():
        kk = str(k).strip()
        if kk in idx:
            vv = float(v)
            vec[idx[kk]] += vv
            total += vv
    if total <= 0:
        return vec, 0
    vec = [x / total for x in vec]
    return vec, 1


def build_gold_records(
    raw_rows: List[Dict[str, str]],
    aggregate_rows: List[Dict[str, str]],
    max_samples: int,
) -> List[TrainingRecord]:
    raw_by_post = {r["post_id"]: r for r in raw_rows}
    out: List[TrainingRecord] = []

    json_keys = {
        "relevance": "q01_relevance",
        "target": "q02_target",
        "stance": "q03_stance",
        "frame": "q04_frame",
        "economic_direction": "q07_economic_direction",
        "social_direction": "q08_social_direction",
        "intensity": "q09_intensity",
        "ambiguity": "q10_ambiguity",
    }

    for agg in aggregate_rows:
        post_id = agg.get("post_id", "")
        raw = raw_by_post.get(post_id)
        if not raw:
            continue
        text = (raw.get("text") or "").strip()
        if not text:
            continue

        majority = parse_json_field(agg.get("majority_json", ""))
        soft = parse_json_field(agg.get("soft_distribution_json", ""))

        hard_labels = {}
        soft_labels = {}
        soft_mask = {}
        labeled_any = False

        for task in STAGE2_TASKS:
            question_key = json_keys[task]
            hard_value = str(majority.get(question_key, "")).strip()
            hard_idx = encode_label(task, hard_value) if hard_value else IGNORE_INDEX
            hard_labels[task] = hard_idx

            dist = soft.get(question_key, {})
            vec, has_soft = soft_vector(task, dist if isinstance(dist, dict) else {})
            soft_labels[task] = vec
            soft_mask[task] = has_soft
            if hard_idx != IGNORE_INDEX or has_soft:
                labeled_any = True

        if not labeled_any:
            continue

        disagreement = to_float(agg.get("disagreement_entropy", "0"), default=0.0)
        out.append(
            TrainingRecord(
                post_id=post_id,
                subreddit=raw.get("subreddit", ""),
                created_at=raw.get("created_at", ""),
                user_id=raw.get("user_id", ""),
                text=text,
                hard_labels=hard_labels,
                soft_labels=soft_labels,
                soft_mask=soft_mask,
                disagreement_weight=1.0 + max(0.0, disagreement),
                quality_score=1.0,
            )
        )
        if max_samples > 0 and len(out) >= max_samples:
            break
    return out


def masked_weighted_mean(
    values: torch.Tensor,
    mask: torch.Tensor,
    sample_weight: Optional[torch.Tensor] = None,
) -> Optional[torch.Tensor]:
    if mask.sum().item() == 0:
        return None
    vals = values[mask]
    if sample_weight is None:
        return vals.mean()
    weights = sample_weight[mask]
    denom = torch.clamp(weights.sum(), min=1e-8)
    return (vals * weights).sum() / denom


def compute_task_loss(
    task: str,
    logits: torch.Tensor,
    hard: torch.Tensor,
    soft: torch.Tensor,
    soft_mask: torch.Tensor,
    sample_weight: torch.Tensor,
    mode: str,
    hybrid_alpha: float,
) -> Optional[torch.Tensor]:
    hard_mask = hard != IGNORE_INDEX
    hard_loss = None
    if hard_mask.any():
        per = F.cross_entropy(logits, hard, reduction="none", ignore_index=IGNORE_INDEX)
        hard_loss = masked_weighted_mean(per, hard_mask, sample_weight=sample_weight)

    soft_loss = None
    if soft_mask.any():
        log_probs = F.log_softmax(logits, dim=-1)
        # Task 22: ideology heads use an explicit KL objective for soft labels.
        if task in IDEOLOGY_TASKS:
            per_soft = F.kl_div(log_probs, soft, reduction="none").sum(dim=-1)
        else:
            per_soft = -(soft * log_probs).sum(dim=-1)
        soft_loss = masked_weighted_mean(per_soft, soft_mask, sample_weight=sample_weight)

    if mode == "hard":
        return hard_loss
    if mode == "soft":
        if soft_loss is not None:
            return soft_loss
        return hard_loss
    if mode == "hybrid":
        if hard_loss is not None and soft_loss is not None:
            return hybrid_alpha * hard_loss + (1.0 - hybrid_alpha) * soft_loss
        return hard_loss if hard_loss is not None else soft_loss
    raise ValueError(f"Unknown loss mode: {mode}")


def curriculum_weights(quality: torch.Tensor, progress: float, enabled: bool) -> torch.Tensor:
    if not enabled:
        return torch.ones_like(quality)
    q = torch.clamp(quality, 0.0, 1.0)
    early = 0.15 + 0.85 * q
    late = torch.ones_like(q)
    w = (1.0 - progress) * early + progress * late
    return torch.clamp(w, 0.15, 1.0)


def temporal_consistency_loss(
    z: torch.Tensor,
    user_hash: torch.Tensor,
    created_ts: torch.Tensor,
    time_scale_days: float,
) -> Optional[torch.Tensor]:
    valid = user_hash >= 0
    if valid.sum().item() < 2:
        return None

    unique_users = torch.unique(user_hash[valid])
    penalties: List[torch.Tensor] = []
    denom_days = max(1e-6, float(time_scale_days))
    for uid in unique_users.tolist():
        idx = torch.nonzero(user_hash == uid, as_tuple=False).squeeze(-1)
        if idx.numel() < 2:
            continue
        times = created_ts[idx]
        if (times >= 0).any():
            order = torch.argsort(times)
        else:
            order = torch.arange(idx.numel(), device=idx.device)
        seq = idx[order]
        for j in range(seq.numel() - 1):
            a = seq[j]
            b = seq[j + 1]
            ts_a = created_ts[a].item()
            ts_b = created_ts[b].item()
            if ts_a < 0 or ts_b < 0:
                dt_seconds = 0.0
            else:
                dt_seconds = abs(ts_b - ts_a)
            decay = math.exp(-(dt_seconds / 86400.0) / denom_days)
            penalties.append(decay * ((z[a] - z[b]) ** 2).mean())
    if not penalties:
        return None
    return torch.stack(penalties).mean()


def train_one_epoch(
    model: B6HierarchyModel,
    loader: DataLoader,
    optimizer: AdamW,
    device: torch.device,
    tasks: List[str],
    loss_mode: str,
    hybrid_alpha: float,
    ortho_weight: float,
    consistency_weight: float,
    consistency_time_scale_days: float,
    curriculum_enabled: bool,
    epoch_progress: float,
) -> Dict[str, float]:
    model.train()
    running_total = 0.0
    running_batches = 0
    running_curriculum_weight = 0.0
    running_consistency = 0.0
    running_consistency_steps = 0
    task_sums = {t: 0.0 for t in tasks}
    task_counts = {t: 0 for t in tasks}

    for batch in loader:
        optimizer.zero_grad()
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        quality = batch["quality_score"].to(device)
        sample_weight = batch["disagreement_weight"].to(device)
        cweights = curriculum_weights(quality, epoch_progress, curriculum_enabled).to(device)
        sample_weight = sample_weight * cweights
        running_curriculum_weight += float(cweights.mean().detach().cpu())

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        total_loss = None

        for task in tasks:
            logits = outputs[TASK_SPECS[task]["logits_key"]]
            hard = batch[f"hard_{task}"].to(device)
            soft = batch[f"soft_{task}"].to(device)
            soft_mask = batch[f"soft_mask_{task}"].to(device)
            task_loss = compute_task_loss(
                task=task,
                logits=logits,
                hard=hard,
                soft=soft,
                soft_mask=soft_mask,
                sample_weight=sample_weight,
                mode=loss_mode,
                hybrid_alpha=hybrid_alpha,
            )
            if task_loss is None:
                continue
            total_loss = task_loss if total_loss is None else total_loss + task_loss
            task_sums[task] += float(task_loss.detach().cpu())
            task_counts[task] += 1

        if total_loss is None:
            continue

        if ortho_weight > 0:
            total_loss = total_loss + ortho_weight * outputs["orthogonality_loss"]

        if consistency_weight > 0:
            cons = temporal_consistency_loss(
                z=outputs["z"],
                user_hash=batch["user_hash"].to(device),
                created_ts=batch["created_ts"].to(device),
                time_scale_days=consistency_time_scale_days,
            )
            if cons is not None:
                total_loss = total_loss + consistency_weight * cons
                running_consistency += float(cons.detach().cpu())
                running_consistency_steps += 1

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        running_total += float(total_loss.detach().cpu())
        running_batches += 1

    out = {
        "total_loss": running_total / max(1, running_batches),
        "avg_curriculum_weight": running_curriculum_weight / max(1, running_batches),
        "avg_consistency_loss": running_consistency / max(1, running_consistency_steps),
    }
    for task in tasks:
        out[f"{task}_loss"] = task_sums[task] / max(1, task_counts[task])
    return out


@torch.no_grad()
def evaluate_hard_macro_f1(
    model: B6HierarchyModel,
    loader: DataLoader,
    device: torch.device,
    tasks: List[str],
) -> Dict[str, float]:
    model.eval()
    y_true: Dict[str, List[int]] = {t: [] for t in tasks}
    y_pred: Dict[str, List[int]] = {t: [] for t in tasks}

    for batch in loader:
        outputs = model(
            input_ids=batch["input_ids"].to(device),
            attention_mask=batch["attention_mask"].to(device),
        )
        for task in tasks:
            logits = outputs[TASK_SPECS[task]["logits_key"]]
            preds = logits.argmax(dim=-1).detach().cpu()
            hard = batch[f"hard_{task}"]
            mask = hard != IGNORE_INDEX
            for truth, pred, ok in zip(hard.tolist(), preds.tolist(), mask.tolist()):
                if ok:
                    y_true[task].append(truth)
                    y_pred[task].append(pred)

    metrics = {}
    macro_values = []
    for task in tasks:
        if not y_true[task]:
            continue
        f1 = macro_f1(y_true[task], y_pred[task], len(TASK_SPECS[task]["values"]))
        metrics[f"{task}_macro_f1"] = float(f1)
        macro_values.append(float(f1))
    if macro_values:
        metrics["macro_f1_mean"] = float(sum(macro_values) / len(macro_values))
    return metrics


@torch.no_grad()
def evaluate_ambiguity_heavy_subset(
    model: B6HierarchyModel,
    loader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    ambiguity_class_idx = TASK_SPECS["ambiguity"]["values"].index("2")
    counts = 0
    y_true: Dict[str, List[int]] = {t: [] for t in IDEOLOGY_TASKS}
    y_pred: Dict[str, List[int]] = {t: [] for t in IDEOLOGY_TASKS}

    for batch in loader:
        outputs = model(
            input_ids=batch["input_ids"].to(device),
            attention_mask=batch["attention_mask"].to(device),
        )
        ambiguity_hard = batch["hard_ambiguity"]
        heavy_mask = ambiguity_hard == ambiguity_class_idx
        counts += int(heavy_mask.sum().item())
        if heavy_mask.sum().item() == 0:
            continue

        for task in IDEOLOGY_TASKS:
            logits = outputs[TASK_SPECS[task]["logits_key"]]
            preds = logits.argmax(dim=-1).detach().cpu()
            hard = batch[f"hard_{task}"]
            valid_mask = (hard != IGNORE_INDEX) & heavy_mask
            for truth, pred, ok in zip(hard.tolist(), preds.tolist(), valid_mask.tolist()):
                if ok:
                    y_true[task].append(truth)
                    y_pred[task].append(pred)

    out: Dict[str, float] = {"ambiguity_heavy_n": float(counts)}
    macro_vals: List[float] = []
    for task in IDEOLOGY_TASKS:
        if not y_true[task]:
            continue
        f1 = macro_f1(y_true[task], y_pred[task], len(TASK_SPECS[task]["values"]))
        out[f"{task}_macro_f1_ambiguity_heavy"] = float(f1)
        macro_vals.append(float(f1))
    if macro_vals:
        out["macro_f1_mean_ambiguity_heavy"] = float(sum(macro_vals) / len(macro_vals))
    return out


@torch.no_grad()
def evaluate_ambiguity_heavy_soft_alignment(
    model: B6HierarchyModel,
    loader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    ambiguity_class_idx = TASK_SPECS["ambiguity"]["values"].index("2")
    task_sums = {t: 0.0 for t in IDEOLOGY_TASKS}
    task_counts = {t: 0 for t in IDEOLOGY_TASKS}
    heavy_n = 0

    for batch in loader:
        outputs = model(
            input_ids=batch["input_ids"].to(device),
            attention_mask=batch["attention_mask"].to(device),
        )
        heavy_mask = (batch["hard_ambiguity"] == ambiguity_class_idx).to(device)
        heavy_n += int(heavy_mask.sum().item())
        if heavy_mask.sum().item() == 0:
            continue

        for task in IDEOLOGY_TASKS:
            soft = batch[f"soft_{task}"].to(device)
            soft_mask = batch[f"soft_mask_{task}"].to(device)
            valid = heavy_mask & soft_mask
            if valid.sum().item() == 0:
                continue
            logits = outputs[TASK_SPECS[task]["logits_key"]]
            log_probs = F.log_softmax(logits, dim=-1)
            per = F.kl_div(log_probs, soft, reduction="none").sum(dim=-1)
            task_sums[task] += float(per[valid].sum().detach().cpu())
            task_counts[task] += int(valid.sum().item())

    out: Dict[str, float] = {"ambiguity_heavy_n": float(heavy_n)}
    means: List[float] = []
    for task in IDEOLOGY_TASKS:
        if task_counts[task] == 0:
            continue
        mean_kl = task_sums[task] / task_counts[task]
        out[f"{task}_soft_kl_ambiguity_heavy"] = float(mean_kl)
        means.append(float(mean_kl))
    if means:
        out["soft_kl_mean_ambiguity_heavy"] = float(sum(means) / len(means))
    return out


def macro_f1(y_true: List[int], y_pred: List[int], num_classes: int) -> float:
    f1s: List[float] = []
    for c in range(num_classes):
        tp = 0
        fp = 0
        fn = 0
        for t, p in zip(y_true, y_pred):
            if p == c and t == c:
                tp += 1
            elif p == c and t != c:
                fp += 1
            elif p != c and t == c:
                fn += 1
        denom = (2 * tp + fp + fn)
        f1s.append((2 * tp) / denom if denom > 0 else 0.0)
    return sum(f1s) / max(1, len(f1s))


@torch.no_grad()
def collect_task_logits_and_labels(
    model: B6HierarchyModel,
    loader: DataLoader,
    device: torch.device,
    tasks: List[str],
) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
    model.eval()
    per_task_logits: Dict[str, List[torch.Tensor]] = {t: [] for t in tasks}
    per_task_labels: Dict[str, List[torch.Tensor]] = {t: [] for t in tasks}
    for batch in loader:
        outputs = model(
            input_ids=batch["input_ids"].to(device),
            attention_mask=batch["attention_mask"].to(device),
        )
        for task in tasks:
            logits = outputs[TASK_SPECS[task]["logits_key"]].detach().cpu()
            labels = batch[f"hard_{task}"].detach().cpu()
            mask = labels != IGNORE_INDEX
            if mask.sum().item() == 0:
                continue
            per_task_logits[task].append(logits[mask])
            per_task_labels[task].append(labels[mask])
    out: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
    for task in tasks:
        if not per_task_logits[task]:
            continue
        out[task] = (torch.cat(per_task_logits[task], dim=0), torch.cat(per_task_labels[task], dim=0))
    return out


def fit_temperature(logits: torch.Tensor, labels: torch.Tensor, max_iter: int = 50) -> float:
    t = torch.tensor(1.0, requires_grad=True)
    labels = labels.long()
    optimizer = torch.optim.LBFGS([t], lr=0.2, max_iter=max_iter, line_search_fn="strong_wolfe")

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        temp = torch.clamp(t, 0.05, 10.0)
        loss = F.cross_entropy(logits / temp, labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(torch.clamp(t.detach(), 0.05, 10.0).item())


def multiclass_probs(logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    temp = max(0.05, float(temperature))
    return F.softmax(logits / temp, dim=-1)


def ece_multiclass(probs: torch.Tensor, labels: torch.Tensor, n_bins: int = 15) -> float:
    conf, pred = probs.max(dim=-1)
    correct = (pred == labels).float()
    bins = torch.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = probs.size(0)
    for i in range(n_bins):
        lo = bins[i]
        hi = bins[i + 1]
        in_bin = (conf > lo) & (conf <= hi) if i > 0 else (conf >= lo) & (conf <= hi)
        bcount = int(in_bin.sum().item())
        if bcount == 0:
            continue
        acc = correct[in_bin].mean().item()
        c = conf[in_bin].mean().item()
        ece += (bcount / max(1, n)) * abs(acc - c)
    return float(ece)


def brier_multiclass(probs: torch.Tensor, labels: torch.Tensor, n_classes: int) -> float:
    one_hot = F.one_hot(labels.long(), num_classes=n_classes).float()
    return float(((probs - one_hot) ** 2).sum(dim=-1).mean().item())


def risk_coverage_curve(
    probs: torch.Tensor,
    labels: torch.Tensor,
    coverages: List[float],
) -> List[Dict[str, float]]:
    conf, pred = probs.max(dim=-1)
    correct = (pred == labels).float()
    order = torch.argsort(conf, descending=True)
    conf_sorted = conf[order]
    corr_sorted = correct[order]
    n = len(conf_sorted)
    out: List[Dict[str, float]] = []
    for c in coverages:
        k = max(1, int(round(c * n)))
        selected = corr_sorted[:k]
        threshold = float(conf_sorted[k - 1].item())
        accuracy = float(selected.mean().item())
        out.append(
            {
                "coverage": float(c),
                "threshold": threshold,
                "risk": float(1.0 - accuracy),
                "accuracy": accuracy,
                "n_kept": float(k),
            }
        )
    return out


def run_calibration_and_abstention(
    model: B6HierarchyModel,
    loader: DataLoader,
    device: torch.device,
    tasks: List[str],
) -> Dict[str, object]:
    gathered = collect_task_logits_and_labels(model, loader, device, tasks)
    summary: Dict[str, object] = {}
    agg_n = 0
    agg_ece_before = 0.0
    agg_ece_after = 0.0
    agg_brier_before = 0.0
    agg_brier_after = 0.0
    abstention_improves_any = False
    for task in tasks:
        if task not in gathered:
            continue
        logits, labels = gathered[task]
        n_classes = len(TASK_SPECS[task]["values"])
        t_opt = fit_temperature(logits, labels)

        probs_before = multiclass_probs(logits, 1.0)
        probs_after_candidate = multiclass_probs(logits, t_opt)
        ece_before = ece_multiclass(probs_before, labels)
        brier_before = brier_multiclass(probs_before, labels, n_classes=n_classes)
        ece_candidate = ece_multiclass(probs_after_candidate, labels)
        brier_candidate = brier_multiclass(probs_after_candidate, labels, n_classes=n_classes)

        # Keep calibrated probs only if they improve a combined confidence quality score.
        if (ece_candidate + brier_candidate) <= (ece_before + brier_before):
            selected_temperature = t_opt
            probs_after = probs_after_candidate
            ece_after = ece_candidate
            brier_after = brier_candidate
        else:
            selected_temperature = 1.0
            probs_after = probs_before
            ece_after = ece_before
            brier_after = brier_before

        curve = risk_coverage_curve(
            probs_after,
            labels,
            coverages=[1.0, 0.95, 0.9, 0.8, 0.7, 0.6, 0.5],
        )
        full_risk = curve[0]["risk"] if curve else 1.0
        abstention_reduces_error = any(
            (row["coverage"] < 1.0) and (row["risk"] < full_risk) for row in curve
        )
        abstention_improves_any = abstention_improves_any or abstention_reduces_error
        n_eval = int(labels.numel())
        agg_n += n_eval
        agg_ece_before += n_eval * ece_before
        agg_ece_after += n_eval * ece_after
        agg_brier_before += n_eval * brier_before
        agg_brier_after += n_eval * brier_after
        summary[task] = {
            "n_eval": n_eval,
            "temperature": selected_temperature,
            "temperature_candidate": t_opt,
            "ece_before": ece_before,
            "ece_after": ece_after,
            "brier_before": brier_before,
            "brier_after": brier_after,
            "abstention_reduces_error": abstention_reduces_error,
            "risk_coverage": curve,
        }
    if agg_n > 0:
        agg_entry = {
            "n_eval_total": agg_n,
            "ece_before_weighted": agg_ece_before / agg_n,
            "ece_after_weighted": agg_ece_after / agg_n,
            "brier_before_weighted": agg_brier_before / agg_n,
            "brier_after_weighted": agg_brier_after / agg_n,
            "confidence_quality_improved": (agg_ece_after + agg_brier_after) < (agg_ece_before + agg_brier_before),
            "abstention_reduces_error_any_task": abstention_improves_any,
            "checkpoint_h_satisfied": ((agg_ece_after + agg_brier_after) < (agg_ece_before + agg_brier_before))
            and abstention_improves_any,
        }
        summary["_aggregate"] = agg_entry
    return summary


def save_checkpoint(path: Path, model: B6HierarchyModel, metadata: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state_dict": model.state_dict(),
        "metadata": metadata,
    }
    torch.save(payload, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-posts-csv", type=Path, default=Path("outputs/ingestion/raw_posts_60k.csv"))
    parser.add_argument("--weak-labels-csv", type=Path, default=Path("outputs/weak_labels_60k/post_weak_labels.csv"))
    parser.add_argument("--gold-aggregates-csv", type=Path, default=Path("outputs/annotation_60k/gold_aggregates.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/model_runs/b6_representation"))

    parser.add_argument("--encoder-name", type=str, default="roberta-base")
    parser.add_argument(
        "--offline-random-init",
        action="store_true",
        help=(
            "Run fully offline with random encoder initialization and hash tokenizer. "
            "Useful when pretrained model downloads are unavailable."
        ),
    )
    parser.add_argument("--offline-vocab-size", type=int, default=8192)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--weak-epochs", type=int, default=1)
    parser.add_argument("--gold-epochs", type=int, default=3)
    parser.add_argument("--weak-lr", type=float, default=2e-5)
    parser.add_argument("--gold-lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--ortho-weight", type=float, default=0.05)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument(
        "--consistency-weight",
        type=float,
        default=0.0,
        help="Task 23: weight for user-level temporal consistency loss during weak pretraining.",
    )
    parser.add_argument(
        "--consistency-time-scale-days",
        type=float,
        default=30.0,
        help="Task 23: temporal decay scale for consistency regularization.",
    )
    parser.add_argument(
        "--enable-curriculum",
        action="store_true",
        help="Task 24: confidence-based curriculum (high-q examples weighted more in early epochs).",
    )
    parser.add_argument(
        "--skip-calibration",
        action="store_true",
        help="Skip Task 25 calibration/abstention analysis.",
    )

    parser.add_argument("--gold-loss-mode", choices=["hard", "soft", "hybrid"], default="hybrid")
    parser.add_argument("--hybrid-alpha", type=float, default=0.5)

    parser.add_argument("--weak-max-samples", type=int, default=0)
    parser.add_argument("--gold-max-samples", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def jsonable_config(args: argparse.Namespace) -> Dict[str, object]:
    out: Dict[str, object] = {}
    for key, value in vars(args).items():
        if isinstance(value, Path):
            out[key] = str(value)
        else:
            out[key] = value
    return out


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = torch.device(args.device)

    if not args.raw_posts_csv.exists():
        raise SystemExit(f"Missing raw posts CSV: {args.raw_posts_csv}")
    if not args.weak_labels_csv.exists():
        raise SystemExit(f"Missing weak labels CSV: {args.weak_labels_csv}")
    if not args.gold_aggregates_csv.exists():
        raise SystemExit(f"Missing gold aggregates CSV: {args.gold_aggregates_csv}")

    raw_rows = read_csv(args.raw_posts_csv)
    weak_rows = read_csv(args.weak_labels_csv)
    gold_rows = read_csv(args.gold_aggregates_csv)

    if args.offline_random_init:
        tokenizer = SimpleHashTokenizer(vocab_size=args.offline_vocab_size)
        model = B6HierarchyModel(
            encoder_name=args.encoder_name,
            load_pretrained=False,
            vocab_size=args.offline_vocab_size,
        ).to(device)
    else:
        try:
            tokenizer = AutoTokenizer.from_pretrained(args.encoder_name)
            model = B6HierarchyModel(encoder_name=args.encoder_name, load_pretrained=True).to(device)
        except Exception as exc:
            raise SystemExit(
                "Failed to load pretrained tokenizer/model. "
                "Retry with --offline-random-init in restricted environments.\n"
                f"Original error: {exc}"
            )

    metadata: Dict[str, object] = {
        "config": jsonable_config(args),
        "stage1": {},
        "stage2": {},
    }

    weak_records = build_weak_records(
        raw_rows=raw_rows,
        weak_rows=weak_rows,
        max_samples=args.weak_max_samples,
    )
    if not weak_records:
        raise SystemExit("No weak pretraining records found. Check weak labels and raw posts overlap.")
    weak_train, weak_val = split_records(weak_records, args.val_fraction, args.seed)
    weak_train_ds = RepresentationDataset(weak_train, tokenizer, args.max_length, STAGE1_TASKS)
    weak_val_ds = RepresentationDataset(weak_val, tokenizer, args.max_length, STAGE1_TASKS)
    weak_train_loader = DataLoader(weak_train_ds, batch_size=args.batch_size, shuffle=True)
    weak_val_loader = DataLoader(weak_val_ds, batch_size=args.batch_size, shuffle=False) if weak_val else None

    metadata["stage1"]["records_total"] = len(weak_records)
    metadata["stage1"]["records_train"] = len(weak_train)
    metadata["stage1"]["records_val"] = len(weak_val)
    metadata["stage1"]["curriculum_enabled"] = args.enable_curriculum
    metadata["stage1"]["consistency_weight"] = args.consistency_weight
    metadata["stage1"]["consistency_time_scale_days"] = args.consistency_time_scale_days
    metadata["stage1"]["epochs"] = []

    if args.weak_epochs > 0:
        optimizer = AdamW(model.parameters(), lr=args.weak_lr, weight_decay=args.weight_decay)
        for epoch in range(1, args.weak_epochs + 1):
            weak_progress = (epoch - 1) / max(1, args.weak_epochs - 1)
            train_stats = train_one_epoch(
                model=model,
                loader=weak_train_loader,
                optimizer=optimizer,
                device=device,
                tasks=STAGE1_TASKS,
                loss_mode="hard",
                hybrid_alpha=args.hybrid_alpha,
                ortho_weight=args.ortho_weight,
                consistency_weight=args.consistency_weight,
                consistency_time_scale_days=args.consistency_time_scale_days,
                curriculum_enabled=args.enable_curriculum,
                epoch_progress=weak_progress,
            )
            epoch_result = {"epoch": epoch, "train": train_stats}
            if weak_val_loader is not None:
                epoch_result["val"] = evaluate_hard_macro_f1(model, weak_val_loader, device, STAGE1_TASKS)
            metadata["stage1"]["epochs"].append(epoch_result)
            print(f"[stage1][epoch {epoch}] train_loss={train_stats['total_loss']:.4f}")
    else:
        print("[stage1] skipped (weak-epochs=0)")

    save_checkpoint(args.output_dir / "stage1_weak_pretrained.pt", model, metadata)

    gold_records = build_gold_records(
        raw_rows=raw_rows,
        aggregate_rows=gold_rows,
        max_samples=args.gold_max_samples,
    )
    if not gold_records:
        raise SystemExit(
            "No gold records with labels found in gold_aggregates.csv. "
            "Complete annotation labels and rerun fold_annotations first."
        )

    gold_train, gold_val = split_records(gold_records, args.val_fraction, args.seed)
    gold_train_ds = RepresentationDataset(gold_train, tokenizer, args.max_length, STAGE2_TASKS)
    gold_val_ds = RepresentationDataset(gold_val, tokenizer, args.max_length, STAGE2_TASKS)
    gold_train_loader = DataLoader(gold_train_ds, batch_size=args.batch_size, shuffle=True)
    gold_val_loader = DataLoader(gold_val_ds, batch_size=args.batch_size, shuffle=False) if gold_val else None

    metadata["stage2"]["records_total"] = len(gold_records)
    metadata["stage2"]["records_train"] = len(gold_train)
    metadata["stage2"]["records_val"] = len(gold_val)
    metadata["stage2"]["gold_loss_mode"] = args.gold_loss_mode
    total_soft_rows = sum(
        1
        for rec in gold_records
        if any(rec.soft_mask.get(task, 0) == 1 for task in IDEOLOGY_TASKS)
    )
    metadata["stage2"]["records_with_soft_ideology_labels"] = total_soft_rows
    metadata["stage2"]["epochs"] = []

    if args.gold_loss_mode in {"soft", "hybrid"} and total_soft_rows == 0:
        raise SystemExit(
            "Gold loss mode requires soft label distributions, but none were found in gold_aggregates."
        )

    if args.gold_epochs > 0:
        optimizer = AdamW(model.parameters(), lr=args.gold_lr, weight_decay=args.weight_decay)
        for epoch in range(1, args.gold_epochs + 1):
            gold_progress = (epoch - 1) / max(1, args.gold_epochs - 1)
            train_stats = train_one_epoch(
                model=model,
                loader=gold_train_loader,
                optimizer=optimizer,
                device=device,
                tasks=STAGE2_TASKS,
                loss_mode=args.gold_loss_mode,
                hybrid_alpha=args.hybrid_alpha,
                ortho_weight=args.ortho_weight,
                consistency_weight=0.0,
                consistency_time_scale_days=args.consistency_time_scale_days,
                curriculum_enabled=False,
                epoch_progress=gold_progress,
            )
            epoch_result = {"epoch": epoch, "train": train_stats}
            if gold_val_loader is not None:
                epoch_result["val"] = evaluate_hard_macro_f1(model, gold_val_loader, device, STAGE2_TASKS)
                epoch_result["ambiguity_heavy_val"] = evaluate_ambiguity_heavy_subset(
                    model,
                    gold_val_loader,
                    device,
                )
                epoch_result["ambiguity_heavy_soft_alignment_val"] = evaluate_ambiguity_heavy_soft_alignment(
                    model,
                    gold_val_loader,
                    device,
                )
            metadata["stage2"]["epochs"].append(epoch_result)
            print(f"[stage2][epoch {epoch}] train_loss={train_stats['total_loss']:.4f}")
    else:
        print("[stage2] skipped (gold-epochs=0)")

    if (not args.skip_calibration) and gold_val_loader is not None:
        calibration_summary = run_calibration_and_abstention(
            model=model,
            loader=gold_val_loader,
            device=device,
            tasks=IDEOLOGY_TASKS,
        )
        metadata["stage2"]["calibration"] = calibration_summary
        (args.output_dir / "calibration_summary.json").write_text(
            json.dumps(calibration_summary, indent=2, ensure_ascii=True),
            encoding="utf-8",
        )
        print("[stage2] wrote calibration_summary.json")

    save_checkpoint(args.output_dir / "stage2_gold_finetuned.pt", model, metadata)
    (args.output_dir / "training_summary.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )
    print(f"Wrote checkpoints and summary to {args.output_dir}")


if __name__ == "__main__":
    main()
