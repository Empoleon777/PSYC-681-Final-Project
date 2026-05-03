#!/usr/bin/env python3
"""Train the B6 hierarchy model with weak pretraining + gold fine-tuning.

This script implements the full representation-learning workflow:
1) weak-label pretraining on large Reddit weak labels
2) gold fine-tuning on 3-annotator aggregates with hard/soft/hybrid loss
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import sys
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

from models.b6_ideology_model import B6HierarchyModel


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
    """Offline fallback tokenizer for smoke tests and restricted environments."""

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


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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
    text: str
    hard_labels: Dict[str, int]
    soft_labels: Dict[str, List[float]]
    soft_mask: Dict[str, int]
    disagreement_weight: float


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
                text=text,
                hard_labels=hard_labels,
                soft_labels=soft_labels,
                soft_mask=soft_mask,
                disagreement_weight=1.0,
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
                text=text,
                hard_labels=hard_labels,
                soft_labels=soft_labels,
                soft_mask=soft_mask,
                disagreement_weight=1.0 + max(0.0, disagreement),
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


def train_one_epoch(
    model: B6HierarchyModel,
    loader: DataLoader,
    optimizer: AdamW,
    device: torch.device,
    tasks: List[str],
    loss_mode: str,
    hybrid_alpha: float,
    ortho_weight: float,
) -> Dict[str, float]:
    model.train()
    running_total = 0.0
    running_batches = 0
    task_sums = {t: 0.0 for t in tasks}
    task_counts = {t: 0 for t in tasks}

    for batch in loader:
        optimizer.zero_grad()
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        sample_weight = batch["disagreement_weight"].to(device)

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

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        running_total += float(total_loss.detach().cpu())
        running_batches += 1

    out = {
        "total_loss": running_total / max(1, running_batches),
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
    metadata["stage1"]["epochs"] = []

    if args.weak_epochs > 0:
        optimizer = AdamW(model.parameters(), lr=args.weak_lr, weight_decay=args.weight_decay)
        for epoch in range(1, args.weak_epochs + 1):
            train_stats = train_one_epoch(
                model=model,
                loader=weak_train_loader,
                optimizer=optimizer,
                device=device,
                tasks=STAGE1_TASKS,
                loss_mode="hard",
                hybrid_alpha=args.hybrid_alpha,
                ortho_weight=args.ortho_weight,
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
            train_stats = train_one_epoch(
                model=model,
                loader=gold_train_loader,
                optimizer=optimizer,
                device=device,
                tasks=STAGE2_TASKS,
                loss_mode=args.gold_loss_mode,
                hybrid_alpha=args.hybrid_alpha,
                ortho_weight=args.ortho_weight,
            )
            epoch_result = {"epoch": epoch, "train": train_stats}
            if gold_val_loader is not None:
                epoch_result["val"] = evaluate_hard_macro_f1(model, gold_val_loader, device, STAGE2_TASKS)
                epoch_result["ambiguity_heavy_val"] = evaluate_ambiguity_heavy_subset(
                    model,
                    gold_val_loader,
                    device,
                )
            metadata["stage2"]["epochs"].append(epoch_result)
            print(f"[stage2][epoch {epoch}] train_loss={train_stats['total_loss']:.4f}")
    else:
        print("[stage2] skipped (gold-epochs=0)")

    save_checkpoint(args.output_dir / "stage2_gold_finetuned.pt", model, metadata)
    (args.output_dir / "training_summary.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )
    print(f"Wrote checkpoints and summary to {args.output_dir}")


if __name__ == "__main__":
    main()
