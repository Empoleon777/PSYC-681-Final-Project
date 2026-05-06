#!/usr/bin/env python3
"""Tasks 17-19: flat transformer B3/B4/B5 training pipeline."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer, RobertaConfig, RobertaModel


IGNORE_INDEX = -100

BASE_TASKS = {
    "relevance": ("q01_relevance", ["0", "1"]),
    "economic_direction": ("q07_economic_direction", ["-1", "0", "1"]),
    "social_direction": ("q08_social_direction", ["-1", "0", "1"]),
}

B5_EXTRA_TASKS = {
    "target": ("q02_target", [
        "government_institutions",
        "political_party_or_actor",
        "policy_or_legislation",
        "social_group_or_identity",
        "foreign_actor_or_geopolitics",
        "media_or_information",
        "other",
    ]),
    "stance": ("q03_stance", ["support", "oppose", "mixed_or_unclear"]),
    "frame": ("q04_frame", [
        "economic_freedom",
        "economic_redistribution",
        "law_and_order",
        "civil_rights",
        "national_security",
        "environmental_protection",
        "public_health",
        "identity_and_values",
        "other",
    ]),
    "intensity": ("q09_intensity", ["0", "1", "2"]),
    "ambiguity": ("q10_ambiguity", ["0", "1", "2"]),
}

IDEOLOGY_TASKS = ["economic_direction", "social_direction", "intensity", "ambiguity"]


@dataclass
class Rec:
    text: str
    hard: Dict[str, int]
    soft: Dict[str, List[float]]
    soft_mask: Dict[str, int]
    ambiguity_label: int = IGNORE_INDEX
    post_id: str = ""


class TaskDataset(Dataset):
    def __init__(self, rows: List[Rec], tokenizer, max_length: int, tasks: List[str]) -> None:
        self.rows = rows
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.tasks = tasks

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        r = self.rows[idx]
        enc = self.tokenizer(
            r.text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        out = {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "ambiguity_label": torch.tensor(r.ambiguity_label, dtype=torch.long),
        }
        for t in self.tasks:
            out[f"hard_{t}"] = torch.tensor(r.hard[t], dtype=torch.long)
            out[f"soft_{t}"] = torch.tensor(r.soft[t], dtype=torch.float32)
            out[f"soft_mask_{t}"] = torch.tensor(r.soft_mask[t], dtype=torch.bool)
        return out


class SimpleHashTokenizer:
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
        tokens = text.lower().split()
        token_ids = [self._token_to_id(t) for t in tokens]
        if truncation:
            token_ids = token_ids[: max(0, max_length - 2)]
        ids = [self.cls_id] + token_ids + [self.sep_id]
        ids = ids[:max_length]
        mask = [1] * len(ids)
        if padding == "max_length" and len(ids) < max_length:
            pad_n = max_length - len(ids)
            ids.extend([self.pad_id] * pad_n)
            mask.extend([0] * pad_n)
        if return_tensors != "pt":
            raise ValueError("SimpleHashTokenizer only supports return_tensors='pt'")
        return {
            "input_ids": torch.tensor([ids], dtype=torch.long),
            "attention_mask": torch.tensor([mask], dtype=torch.long),
        }


class FlatTransformer(nn.Module):
    def __init__(self, task_specs: Dict[str, Tuple[str, List[str]]], encoder_name: str, offline_random_init: bool):
        super().__init__()
        if offline_random_init:
            cfg = RobertaConfig(
                vocab_size=30522,
                hidden_size=256,
                num_hidden_layers=4,
                num_attention_heads=4,
                intermediate_size=1024,
                max_position_embeddings=514,
            )
            self.encoder = RobertaModel(cfg)
        else:
            self.encoder = AutoModel.from_pretrained(encoder_name)
        hidden = self.encoder.config.hidden_size
        self.task_heads = nn.ModuleDict({
            task: nn.Linear(hidden, len(spec[1])) for task, spec in task_specs.items()
        })
        self.task_specs = task_specs

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = out.last_hidden_state[:, 0]
        return {task: head(pooled) for task, head in self.task_heads.items()}


def read_csv(path: Path) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def parse_json(raw: str) -> Dict:
    if not raw:
        return {}
    return json.loads(raw)


def label_index(values: List[str], raw: str) -> int:
    raw = str(raw).strip()
    return values.index(raw) if raw in values else IGNORE_INDEX


def soft_vec(values: List[str], dist: Dict[str, float]) -> Tuple[List[float], int]:
    idx = {v: i for i, v in enumerate(values)}
    vec = [0.0] * len(values)
    total = 0.0
    for k, v in dist.items():
        if str(k) in idx:
            vv = float(v)
            vec[idx[str(k)]] += vv
            total += vv
    if total <= 0:
        return vec, 0
    return [x / total for x in vec], 1


def split_rows(rows: List[Rec], val_fraction: float, seed: int) -> Tuple[List[Rec], List[Rec]]:
    local = list(rows)
    random.Random(seed).shuffle(local)
    n_val = max(1, int(round(len(local) * val_fraction)))
    n_val = min(n_val, len(local) - 1)
    return local[n_val:], local[:n_val]


def build_weak_rows(
    raw_rows: List[Dict[str, str]],
    weak_rows: List[Dict[str, str]],
    tasks: Dict[str, Tuple[str, List[str]]],
    max_samples: int,
) -> List[Rec]:
    weak_by_post = {r["post_id"]: r for r in weak_rows}
    out: List[Rec] = []
    for r in raw_rows:
        post_id = r.get("post_id", "")
        weak = weak_by_post.get(post_id)
        if weak is None:
            continue
        text = (r.get("text") or "").strip()
        if not text:
            continue

        weak_value = {
            "relevance": "0" if weak.get("low_quality", "").strip() == "1" else "1",
            "economic_direction": "1" if float(weak.get("projected_zecon", 0)) > 0.1 else ("-1" if float(weak.get("projected_zecon", 0)) < -0.1 else "0"),
            "social_direction": "1" if float(weak.get("projected_zsoc", 0)) > 0.1 else ("-1" if float(weak.get("projected_zsoc", 0)) < -0.1 else "0"),
            "target": weak.get("target_hypothesis", "other"),
            "stance": "support" if weak.get("stance_hypothesis", "") in {"support", "support_progressive", "support_conservative"} else ("oppose" if weak.get("stance_hypothesis", "") == "oppose" else "mixed_or_unclear"),
            "frame": weak.get("frame_hypothesis", "other"),
            "intensity": "2" if max(abs(float(weak.get("projected_zecon", 0))), abs(float(weak.get("projected_zsoc", 0)))) >= 0.6 else ("1" if max(abs(float(weak.get("projected_zecon", 0))), abs(float(weak.get("projected_zsoc", 0)))) >= 0.25 else "0"),
            "ambiguity": "0" if float(weak.get("q_score", 0)) >= 0.65 else ("1" if float(weak.get("q_score", 0)) >= 0.3 else "2"),
        }

        hard: Dict[str, int] = {}
        soft: Dict[str, List[float]] = {}
        soft_mask: Dict[str, int] = {}
        for task, (_, values) in tasks.items():
            hard[task] = label_index(values, weak_value[task])
            soft[task] = [0.0] * len(values)
            soft_mask[task] = 0
        out.append(Rec(text=text, hard=hard, soft=soft, soft_mask=soft_mask, ambiguity_label=label_index(["0", "1", "2"], weak_value["ambiguity"])))
        if max_samples > 0 and len(out) >= max_samples:
            break
    return out


def build_gold_rows(
    raw_rows: List[Dict[str, str]],
    agg_rows: List[Dict[str, str]],
    tasks: Dict[str, Tuple[str, List[str]]],
    max_samples: int,
) -> List[Rec]:
    raw_by_post = {r["post_id"]: r for r in raw_rows}
    out: List[Rec] = []
    for agg in agg_rows:
        raw = raw_by_post.get(agg.get("post_id", ""))
        if raw is None:
            continue
        text = (raw.get("text") or "").strip()
        if not text:
            continue
        mj = parse_json(agg.get("majority_json", ""))
        sj = parse_json(agg.get("soft_distribution_json", ""))
        hard: Dict[str, int] = {}
        soft: Dict[str, List[float]] = {}
        soft_mask: Dict[str, int] = {}
        labeled_any = False
        ambiguity_label = label_index(["0", "1", "2"], mj.get("q10_ambiguity", ""))
        for task, (qkey, values) in tasks.items():
            hard_idx = label_index(values, mj.get(qkey, ""))
            soft_vec_value, has_soft = soft_vec(values, sj.get(qkey, {}) if isinstance(sj.get(qkey, {}), dict) else {})
            hard[task] = hard_idx
            soft[task] = soft_vec_value
            soft_mask[task] = has_soft
            if hard_idx != IGNORE_INDEX or has_soft:
                labeled_any = True
        if labeled_any:
            out.append(Rec(text=text, hard=hard, soft=soft, soft_mask=soft_mask, ambiguity_label=ambiguity_label, post_id=agg.get("post_id", "")))
        if max_samples > 0 and len(out) >= max_samples:
            break
    return out


def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> Optional[torch.Tensor]:
    if mask.sum().item() == 0:
        return None
    return values[mask].mean()


def task_loss(task: str, logits: torch.Tensor, hard: torch.Tensor, soft: torch.Tensor, soft_mask: torch.Tensor, mode: str) -> Optional[torch.Tensor]:
    hard_mask = hard != IGNORE_INDEX
    hard_loss = None
    if hard_mask.any():
        per = F.cross_entropy(logits, hard, reduction="none", ignore_index=IGNORE_INDEX)
        hard_loss = masked_mean(per, hard_mask)

    soft_loss = None
    if soft_mask.any():
        log_probs = F.log_softmax(logits, dim=-1)
        if task in IDEOLOGY_TASKS:
            per_soft = F.kl_div(log_probs, soft, reduction="none").sum(dim=-1)
        else:
            per_soft = -(soft * log_probs).sum(dim=-1)
        soft_loss = masked_mean(per_soft, soft_mask)

    if mode == "hard":
        return hard_loss
    if mode == "soft":
        return soft_loss if soft_loss is not None else hard_loss
    raise ValueError(mode)


@torch.no_grad()
def evaluate(model: FlatTransformer, loader: DataLoader, device: torch.device, tasks: Dict[str, Tuple[str, List[str]]]) -> Dict[str, float]:
    model.eval()
    y_true: Dict[str, List[int]] = {t: [] for t in tasks}
    y_pred: Dict[str, List[int]] = {t: [] for t in tasks}
    for batch in loader:
        out = model(
            input_ids=batch["input_ids"].to(device),
            attention_mask=batch["attention_mask"].to(device),
        )
        for task in tasks:
            preds = out[task].argmax(dim=-1).cpu().tolist()
            truth = batch[f"hard_{task}"].tolist()
            for t, p in zip(truth, preds):
                if t != IGNORE_INDEX:
                    y_true[task].append(t)
                    y_pred[task].append(p)
    metrics = {}
    vals = []
    for task, (_, values) in tasks.items():
        if not y_true[task]:
            continue
        f1 = float(f1_score(y_true[task], y_pred[task], average="macro", zero_division=0))
        metrics[f"{task}_macro_f1"] = f1
        vals.append(f1)
    if vals:
        metrics["macro_f1_mean"] = float(sum(vals) / len(vals))
    return metrics


@torch.no_grad()
def evaluate_ambiguity_heavy_subset(
    model: FlatTransformer,
    loader: DataLoader,
    device: torch.device,
    tasks: Dict[str, Tuple[str, List[str]]],
) -> Dict[str, float]:
    model.eval()
    heavy_idx = 2
    y_true: Dict[str, List[int]] = {t: [] for t in tasks}
    y_pred: Dict[str, List[int]] = {t: [] for t in tasks}
    heavy_n = 0
    for batch in loader:
        out = model(
            input_ids=batch["input_ids"].to(device),
            attention_mask=batch["attention_mask"].to(device),
        )
        heavy_mask = batch["ambiguity_label"] == heavy_idx
        heavy_n += int(heavy_mask.sum().item())
        if heavy_mask.sum().item() == 0:
            continue
        for task in tasks:
            preds = out[task].argmax(dim=-1).cpu()
            truth = batch[f"hard_{task}"]
            valid = (truth != IGNORE_INDEX) & heavy_mask
            for t, p, ok in zip(truth.tolist(), preds.tolist(), valid.tolist()):
                if ok:
                    y_true[task].append(t)
                    y_pred[task].append(p)

    metrics: Dict[str, float] = {"ambiguity_heavy_n": float(heavy_n)}
    vals: List[float] = []
    for task, (_, values) in tasks.items():
        if not y_true[task]:
            continue
        f1 = float(f1_score(y_true[task], y_pred[task], average="macro", zero_division=0))
        metrics[f"{task}_macro_f1_ambiguity_heavy"] = f1
        vals.append(f1)
    if vals:
        metrics["macro_f1_mean_ambiguity_heavy"] = float(sum(vals) / len(vals))
    return metrics


@torch.no_grad()
def evaluate_ambiguity_heavy_soft_alignment(
    model: FlatTransformer,
    loader: DataLoader,
    device: torch.device,
    tasks: Dict[str, Tuple[str, List[str]]],
) -> Dict[str, float]:
    model.eval()
    heavy_idx = 2
    ideology_tasks = [t for t in ["economic_direction", "social_direction", "intensity", "ambiguity"] if t in tasks]
    sums = {t: 0.0 for t in ideology_tasks}
    counts = {t: 0 for t in ideology_tasks}
    heavy_n = 0

    for batch in loader:
        out = model(
            input_ids=batch["input_ids"].to(device),
            attention_mask=batch["attention_mask"].to(device),
        )
        heavy_mask = (batch["ambiguity_label"] == heavy_idx).to(device)
        heavy_n += int(heavy_mask.sum().item())
        if heavy_mask.sum().item() == 0:
            continue
        for task in ideology_tasks:
            soft = batch[f"soft_{task}"].to(device)
            soft_mask = batch[f"soft_mask_{task}"].to(device)
            valid = heavy_mask & soft_mask
            if valid.sum().item() == 0:
                continue
            log_probs = F.log_softmax(out[task], dim=-1)
            per = F.kl_div(log_probs, soft, reduction="none").sum(dim=-1)
            sums[task] += float(per[valid].sum().detach().cpu())
            counts[task] += int(valid.sum().item())

    metrics: Dict[str, float] = {"ambiguity_heavy_n": float(heavy_n)}
    mean_vals: List[float] = []
    for task in ideology_tasks:
        if counts[task] == 0:
            continue
        val = sums[task] / counts[task]
        metrics[f"{task}_soft_kl_ambiguity_heavy"] = float(val)
        mean_vals.append(float(val))
    if mean_vals:
        metrics["soft_kl_mean_ambiguity_heavy"] = float(sum(mean_vals) / len(mean_vals))
    return metrics


@torch.no_grad()
def collect_logits_labels(
    model: FlatTransformer,
    loader: DataLoader,
    device: torch.device,
    task: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    logits_list: List[torch.Tensor] = []
    labels_list: List[torch.Tensor] = []
    for batch in loader:
        out = model(
            input_ids=batch["input_ids"].to(device),
            attention_mask=batch["attention_mask"].to(device),
        )
        logits = out[task].detach().cpu()
        labels = batch[f"hard_{task}"].detach().cpu()
        mask = labels != IGNORE_INDEX
        if mask.sum().item() == 0:
            continue
        logits_list.append(logits[mask])
        labels_list.append(labels[mask])
    return torch.cat(logits_list, dim=0), torch.cat(labels_list, dim=0)


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


def risk_coverage_curve(probs: torch.Tensor, labels: torch.Tensor) -> List[Dict[str, float]]:
    conf, pred = probs.max(dim=-1)
    correct = (pred == labels).float()
    order = torch.argsort(conf, descending=True)
    conf_sorted = conf[order]
    corr_sorted = correct[order]
    out: List[Dict[str, float]] = []
    n = len(conf_sorted)
    for c in [1.0, 0.95, 0.9, 0.8, 0.7, 0.6, 0.5]:
        k = max(1, int(round(c * n)))
        selected = corr_sorted[:k]
        out.append(
            {
                "coverage": float(c),
                "threshold": float(conf_sorted[k - 1].item()),
                "risk": float(1.0 - selected.mean().item()),
                "accuracy": float(selected.mean().item()),
            }
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=["b3", "b4", "b5"], required=True)
    parser.add_argument("--raw-posts-csv", type=Path, default=Path("outputs/ingestion/raw_posts_60k.csv"))
    parser.add_argument("--weak-labels-csv", type=Path, default=Path("outputs/weak_labels_60k/post_weak_labels.csv"))
    parser.add_argument("--gold-aggregates-csv", type=Path, default=Path("outputs/annotation_60k/gold_aggregates.csv"))
    parser.add_argument("--output-json", type=Path, default=Path("outputs/model_runs/flat_b3_b5_summary.json"))
    parser.add_argument("--encoder-name", type=str, default="roberta-base")
    parser.add_argument("--offline-random-init", action="store_true")
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--weak-epochs", type=int, default=1)
    parser.add_argument("--gold-epochs", type=int, default=2)
    parser.add_argument("--weak-max-samples", type=int, default=2000)
    parser.add_argument("--gold-max-samples", type=int, default=1200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-predictions-csv", type=Path, default=None,
                        help="Optional path to write per-post val-set predictions for downstream error analysis.")
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    tasks = dict(BASE_TASKS)
    if args.variant == "b5":
        tasks.update(B5_EXTRA_TASKS)

    raw_rows = read_csv(args.raw_posts_csv)
    weak_rows = read_csv(args.weak_labels_csv)
    gold_rows = read_csv(args.gold_aggregates_csv)

    weak_data = build_weak_rows(raw_rows, weak_rows, tasks, args.weak_max_samples)
    gold_data = build_gold_rows(raw_rows, gold_rows, tasks, args.gold_max_samples)
    if not weak_data or not gold_data:
        raise SystemExit("Insufficient data for B3/B4/B5 training.")

    gold_train, gold_val = split_rows(gold_data, 0.2, args.seed)
    mode = "hard" if args.variant == "b3" else "soft"

    if args.offline_random_init:
        tokenizer = SimpleHashTokenizer()
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.encoder_name)
    model = FlatTransformer(tasks, args.encoder_name, args.offline_random_init)
    device = torch.device(args.device)
    model = model.to(device)

    weak_loader = DataLoader(TaskDataset(weak_data, tokenizer, args.max_length, list(tasks.keys())), batch_size=args.batch_size, shuffle=True)
    gold_train_loader = DataLoader(TaskDataset(gold_train, tokenizer, args.max_length, list(tasks.keys())), batch_size=args.batch_size, shuffle=True)
    gold_val_loader = DataLoader(TaskDataset(gold_val, tokenizer, args.max_length, list(tasks.keys())), batch_size=args.batch_size, shuffle=False)

    weak_opt = AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    for _ in range(args.weak_epochs):
        model.train()
        for batch in weak_loader:
            weak_opt.zero_grad()
            out = model(batch["input_ids"].to(device), batch["attention_mask"].to(device))
            loss = None
            for task in tasks:
                l = task_loss(
                    task,
                    out[task],
                    batch[f"hard_{task}"].to(device),
                    batch[f"soft_{task}"].to(device),
                    batch[f"soft_mask_{task}"].to(device),
                    mode="hard",
                )
                if l is not None:
                    loss = l if loss is None else loss + l
            if loss is None:
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            weak_opt.step()

    gold_opt = AdamW(model.parameters(), lr=1e-5, weight_decay=0.01)
    for _ in range(args.gold_epochs):
        model.train()
        for batch in gold_train_loader:
            gold_opt.zero_grad()
            out = model(batch["input_ids"].to(device), batch["attention_mask"].to(device))
            loss = None
            for task in tasks:
                l = task_loss(
                    task,
                    out[task],
                    batch[f"hard_{task}"].to(device),
                    batch[f"soft_{task}"].to(device),
                    batch[f"soft_mask_{task}"].to(device),
                    mode=mode,
                )
                if l is not None:
                    loss = l if loss is None else loss + l
            if loss is None:
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            gold_opt.step()

    metrics = evaluate(model, gold_val_loader, device, tasks)
    ambiguity_heavy_metrics = evaluate_ambiguity_heavy_subset(model, gold_val_loader, device, tasks)
    ambiguity_heavy_soft_alignment = evaluate_ambiguity_heavy_soft_alignment(model, gold_val_loader, device, tasks)
    calibration: Dict[str, object] = {}
    if args.variant == "b5":
        for task in IDEOLOGY_TASKS:
            if task not in tasks:
                continue
            logits, labels = collect_logits_labels(model, gold_val_loader, device, task)
            probs = F.softmax(logits, dim=-1)
            calibration[task] = {
                "ece": ece_multiclass(probs, labels),
                "brier": brier_multiclass(probs, labels, n_classes=len(tasks[task][1])),
                "risk_coverage": risk_coverage_curve(probs, labels),
            }

    payload = {
        "variant": args.variant,
        "task_set": list(tasks.keys()),
        "gold_loss_mode": mode,
        "weak_records": len(weak_data),
        "gold_train_records": len(gold_train),
        "gold_val_records": len(gold_val),
        "metrics": metrics,
        "ambiguity_heavy_metrics": ambiguity_heavy_metrics,
        "ambiguity_heavy_soft_alignment": ambiguity_heavy_soft_alignment,
        "calibration": calibration,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=True))

    if args.output_predictions_csv is not None:
        write_val_predictions(model, gold_val, tokenizer, args.max_length, args.batch_size, device, tasks, args.variant, args.output_predictions_csv)


@torch.no_grad()
def write_val_predictions(
    model: "FlatTransformer",
    gold_val: List["Rec"],
    tokenizer,
    max_length: int,
    batch_size: int,
    device: torch.device,
    tasks: Dict[str, Tuple[str, List[str]]],
    variant: str,
    out_path: Path,
) -> None:
    model.eval()
    loader = DataLoader(
        TaskDataset(gold_val, tokenizer, max_length, list(tasks.keys())),
        batch_size=batch_size,
        shuffle=False,
    )
    all_preds: Dict[str, List[int]] = {t: [] for t in tasks}
    all_probs: Dict[str, List[List[float]]] = {t: [] for t in tasks}
    for batch in loader:
        out = model(
            input_ids=batch["input_ids"].to(device),
            attention_mask=batch["attention_mask"].to(device),
        )
        for task in tasks:
            logits = out[task]
            probs = F.softmax(logits, dim=-1).cpu().tolist()
            preds = logits.argmax(dim=-1).cpu().tolist()
            all_preds[task].extend(preds)
            all_probs[task].extend(probs)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        fieldnames = ["post_id", "split_name", "model_name"]
        for task in tasks:
            fieldnames += [f"pred_{task}", f"probs_{task}"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i, rec in enumerate(gold_val):
            row = {
                "post_id": rec.post_id,
                "split_name": "val",
                "model_name": variant,
            }
            for task, (_, values) in tasks.items():
                pred_idx = all_preds[task][i]
                row[f"pred_{task}"] = values[pred_idx] if 0 <= pred_idx < len(values) else ""
                row[f"probs_{task}"] = json.dumps({values[j]: float(p) for j, p in enumerate(all_probs[task][i])}, ensure_ascii=True)
            writer.writerow(row)


if __name__ == "__main__":
    main()
