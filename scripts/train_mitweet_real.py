#!/usr/bin/env python3
"""Real-data run on MITweet: pretrained RoBERTa-base + real human labels.

This is a one-script reproducible runner that produces the same per-row
prediction schema, summary JSON, and run manifest as B1/B2/B5/B6 -- so
the existing scripts/error_analysis.py and scripts/run_seeds.py work
unchanged.

What it does
------------
1. Load Data/MITweet.csv (real labels: domain relevance R1-R5 + facet
   ideology I1-I12).
2. Project MITweet labels to the project's task spec, exactly the
   subset that the dataset can populate cleanly:
     - relevance         : 1 if any R1-R5 == 1 else 0
     - economic_direction: majority sign of active econ facets (I1 .. I4)
     - social_direction  : majority sign of active social facets (I5 .. I12)
   Posts where every relevant facet is N/A get the neutral class 0.
3. Stratified split by topic into train/val/test.
4. Fine-tune RoBERTa-base (loaded from a local path; HuggingFace Hub is
   blocked in the experiment sandbox) for N epochs on train, with
   per-task heads on the [CLS] vector.
5. Eval on val: macro-F1 per task, accuracy, and write predictions CSV
   in the standard schema (post_id, pred_<task>, probs_<task>).
6. Emit a run_manifest.json so later analyses can verify the data
   hashes, seed, and git commit.

Why this is a real-data run
---------------------------
- Pretrained encoder (RoBERTa-base, 124.6M params) -> not random init.
- Real human-annotated labels on real tweets -> not synthetic templates.
- Stratified split, multi-seed friendly, full epoch budgets.

What it cannot include
----------------------
- Politosphere weak-supervision pretraining (Zenodo data download is
  blocked in the experiment sandbox). So this run trains gold-only,
  the "B2-class" branch of the project's ladder.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, Dataset
from transformers import RobertaModel, RobertaTokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts._run_manifest import build_manifest, write_manifest


# ---- task spec ----------------------------------------------------------------

TASKS: Dict[str, List[str]] = {
    "relevance": ["0", "1"],
    "economic_direction": ["-1", "0", "1"],
    "social_direction": ["-1", "0", "1"],
}

# Heuristic facet -> axis assignment. Coverage is partial by design; the
# label-mapping fallback is class 0 ("neutral") when no facet is active.
ECON_FACETS = ["I1", "I2", "I3", "I4"]
SOC_FACETS = ["I5", "I6", "I7", "I8", "I9", "I10", "I11", "I12"]
DOMAIN_R = ["R1", "R2", "R3", "R4", "R5"]


@dataclass
class Row:
    post_id: str
    text: str
    topic: str
    relevance: int          # 0 / 1
    economic: int           # 0=-1, 1=0, 2=+1
    social: int             # 0=-1, 1=0, 2=+1


# ---- label projection ----------------------------------------------------------

def parse_int(value: str) -> Optional[int]:
    s = (value or "").strip()
    if s == "":
        return None
    try:
        return int(float(s))
    except ValueError:
        return None


def project_labels(row: Dict[str, str], idx: int) -> Optional[Row]:
    text = (row.get("tweet") or "").strip()
    if not text:
        return None
    topic = (row.get("topic") or "unknown").strip()

    rel_any = 0
    for k in DOMAIN_R:
        v = parse_int(row.get(k, ""))
        if v == 1:
            rel_any = 1
            break

    def axis_label(keys: List[str]) -> int:
        active: List[int] = []
        for k in keys:
            v = parse_int(row.get(k, ""))
            if v in (0, 1, 2):
                active.append(v)
        if not active:
            # No active facet on this axis -> neutral.
            return 1  # index 1 corresponds to "0"
        # MITweet I-values: 0 = left, 1 = neutral, 2 = right (per the
        # mapping doc, mid-point = neutral). Project to {-1, 0, +1} indices
        # 0/1/2 by majority vote.
        c = Counter(active).most_common(1)[0][0]
        return int(c)

    return Row(
        post_id=f"mit_{idx:06d}",
        text=text,
        topic=topic,
        relevance=rel_any,
        economic=axis_label(ECON_FACETS),
        social=axis_label(SOC_FACETS),
    )


# ---- dataset / model -----------------------------------------------------------

class MitDataset(Dataset):
    def __init__(self, rows: List[Row], tokenizer, max_length: int) -> None:
        self.rows = rows
        self.tokenizer = tokenizer
        self.max_length = max_length

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
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "y_relevance": torch.tensor(r.relevance, dtype=torch.long),
            "y_economic": torch.tensor(r.economic, dtype=torch.long),
            "y_social": torch.tensor(r.social, dtype=torch.long),
        }


class FlatHeadModel(nn.Module):
    def __init__(self, encoder_path: str) -> None:
        super().__init__()
        self.encoder = RobertaModel.from_pretrained(encoder_path, local_files_only=True)
        h = self.encoder.config.hidden_size
        self.head_relevance = nn.Linear(h, 2)
        self.head_economic = nn.Linear(h, 3)
        self.head_social = nn.Linear(h, 3)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = out.last_hidden_state[:, 0]
        return {
            "relevance": self.head_relevance(pooled),
            "economic_direction": self.head_economic(pooled),
            "social_direction": self.head_social(pooled),
        }


# ---- split + IO ----------------------------------------------------------------

def stratified_split(
    rows: List[Row], seed: int, max_train: int, max_val: int
) -> Tuple[List[Row], List[Row]]:
    """Stratified sample by topic so every topic appears in both splits.

    A cap is applied to keep CPU runtimes reasonable. The caller is
    responsible for documenting that cap in the manifest.
    """
    rng = random.Random(seed)
    by_topic: Dict[str, List[Row]] = {}
    for r in rows:
        by_topic.setdefault(r.topic, []).append(r)

    train: List[Row] = []
    val: List[Row] = []
    val_share = 0.2
    for topic, recs in by_topic.items():
        rng.shuffle(recs)
        n_val = max(1, int(round(val_share * len(recs))))
        val.extend(recs[:n_val])
        train.extend(recs[n_val:])
    rng.shuffle(train)
    rng.shuffle(val)
    if max_train > 0:
        train = train[:max_train]
    if max_val > 0:
        val = val[:max_val]
    return train, val


def load_rows(csv_path: Path) -> List[Row]:
    out: List[Row] = []
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        r = csv.DictReader(f)
        for i, row in enumerate(r):
            mapped = project_labels(row, i)
            if mapped is not None:
                out.append(mapped)
    return out


# ---- training loop -------------------------------------------------------------

def macro_f1(y_true: List[int], y_pred: List[int]) -> float:
    if not y_true:
        return 0.0
    return float(f1_score(y_true, y_pred, average="macro", zero_division=0))


def evaluate(
    model: FlatHeadModel, loader: DataLoader, device: torch.device
) -> Tuple[Dict[str, float], Dict[str, List[int]], Dict[str, List[int]], Dict[str, List[List[float]]]]:
    model.eval()
    y_true: Dict[str, List[int]] = {t: [] for t in TASKS}
    y_pred: Dict[str, List[int]] = {t: [] for t in TASKS}
    y_probs: Dict[str, List[List[float]]] = {t: [] for t in TASKS}
    with torch.no_grad():
        for batch in loader:
            out = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
            )
            for task in TASKS:
                logits = out[task].cpu()
                probs = torch.softmax(logits, dim=-1)
                preds = logits.argmax(dim=-1).tolist()
                truth_key = "y_" + task.split("_")[0]  # y_relevance, y_economic, y_social
                truth = batch[truth_key].tolist()
                y_pred[task].extend(preds)
                y_true[task].extend(truth)
                y_probs[task].extend(probs.tolist())
    metrics = {f"{t}_macro_f1": macro_f1(y_true[t], y_pred[t]) for t in TASKS}
    metrics["macro_f1_mean"] = float(np.mean(list(metrics.values())))
    return metrics, y_true, y_pred, y_probs


def write_predictions(
    rows: List[Row],
    y_pred: Dict[str, List[int]],
    y_probs: Dict[str, List[List[float]]],
    model_name: str,
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["post_id", "split_name", "model_name"]
    for t in TASKS:
        fieldnames += [f"pred_{t}", f"probs_{t}"]
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for i, r in enumerate(rows):
            row = {"post_id": r.post_id, "split_name": "val", "model_name": model_name}
            for t, values in TASKS.items():
                pi = y_pred[t][i]
                row[f"pred_{t}"] = values[pi] if 0 <= pi < len(values) else ""
                row[f"probs_{t}"] = json.dumps(
                    {values[j]: float(p) for j, p in enumerate(y_probs[t][i])}, ensure_ascii=True
                )
            w.writerow(row)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mitweet-csv", type=Path, default=Path("Data/MITweet.csv"))
    ap.add_argument("--encoder-path", type=str, default="/tmp/roberta-base-local",
                    help="Local directory containing RoBERTa-base config.json/pytorch_model.bin/vocab.json/merges.txt")
    ap.add_argument("--max-train", type=int, default=1500,
                    help="Cap training set size (CPU-friendly). Set 0 for no cap.")
    ap.add_argument("--max-val", type=int, default=400)
    ap.add_argument("--max-length", type=int, default=96)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--learning-rate", type=float, default=2e-5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--torch-threads", type=int, default=4)
    ap.add_argument("--output-json", type=Path, required=True)
    ap.add_argument("--output-predictions-csv", type=Path, default=None)
    ap.add_argument("--model-name", type=str, default="b2_real_mitweet")
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(max(1, args.torch_threads))

    # 1) Load + project
    print(f"[load] reading {args.mitweet_csv}")
    rows = load_rows(args.mitweet_csv)
    print(f"[load] {len(rows)} rows after projection")
    label_dist = {
        "relevance": Counter(r.relevance for r in rows),
        "economic": Counter(r.economic for r in rows),
        "social": Counter(r.social for r in rows),
        "topic": Counter(r.topic for r in rows),
    }
    print(f"[load] relevance dist: {dict(label_dist['relevance'])}")
    print(f"[load] economic dist: {dict(label_dist['economic'])}")
    print(f"[load] social dist: {dict(label_dist['social'])}")

    # 2) Split
    train, val = stratified_split(rows, args.seed, args.max_train, args.max_val)
    print(f"[split] train={len(train)} val={len(val)}")

    # 3) Tokenizer + model
    tokenizer = RobertaTokenizer.from_pretrained(args.encoder_path, local_files_only=True)
    model = FlatHeadModel(args.encoder_path)
    device = torch.device(args.device)
    model = model.to(device)
    print(f"[model] params: {sum(p.numel() for p in model.parameters())/1e6:.1f} M, device={device}")

    train_ds = MitDataset(train, tokenizer, args.max_length)
    val_ds = MitDataset(val, tokenizer, args.max_length)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    optim = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)

    # 4) Train
    epoch_metrics: List[Dict[str, float]] = []
    train_start = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        steps = 0
        ep_start = time.time()
        for batch in train_loader:
            optim.zero_grad()
            out = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
            )
            loss = (
                F.cross_entropy(out["relevance"], batch["y_relevance"].to(device))
                + F.cross_entropy(out["economic_direction"], batch["y_economic"].to(device))
                + F.cross_entropy(out["social_direction"], batch["y_social"].to(device))
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            running += float(loss.detach().cpu())
            steps += 1
        metrics, _, _, _ = evaluate(model, val_loader, device)
        ep_secs = time.time() - ep_start
        ep_record = {"epoch": epoch, "train_loss": running / max(1, steps), "val": metrics, "epoch_seconds": ep_secs}
        epoch_metrics.append(ep_record)
        print(
            f"[epoch {epoch}] train_loss={ep_record['train_loss']:.4f} "
            f"val_macro_f1={metrics['macro_f1_mean']:.4f} "
            f"rel={metrics['relevance_macro_f1']:.4f} econ={metrics['economic_direction_macro_f1']:.4f} soc={metrics['social_direction_macro_f1']:.4f} "
            f"({ep_secs:.0f}s)"
        )
    total_train_seconds = time.time() - train_start

    # 5) Final eval + per-row predictions
    final_metrics, y_true, y_pred, y_probs = evaluate(model, val_loader, device)
    pred_csv = args.output_predictions_csv or args.output_json.with_name(args.output_json.stem + "_val_predictions.csv")
    write_predictions(val, y_pred, y_probs, args.model_name, pred_csv)

    payload = {
        "model_name": args.model_name,
        "seed": args.seed,
        "encoder_path": args.encoder_path,
        "n_train": len(train),
        "n_val": len(val),
        "n_total_rows_after_projection": len(rows),
        "max_length": args.max_length,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "metrics": final_metrics,
        "epochs_history": epoch_metrics,
        "total_train_seconds": total_train_seconds,
        "predictions_csv": str(pred_csv),
        "label_distribution": {k: dict(v) for k, v in label_dist.items() if k != "topic"},
        "topic_count": len(label_dist["topic"]),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")
    print(f"[write] summary -> {args.output_json}")
    print(f"[write] predictions -> {pred_csv}")

    # 6) Manifest
    manifest = build_manifest(
        run_name=f"{args.model_name}_seed{args.seed}",
        seed=args.seed,
        inputs={"mitweet_csv": args.mitweet_csv, "encoder_path": Path(args.encoder_path) / "config.json"},
        outputs={"summary_json": str(args.output_json), "val_predictions_csv": str(pred_csv)},
        config={
            "encoder_path": args.encoder_path,
            "max_train": args.max_train,
            "max_val": args.max_val,
            "max_length": args.max_length,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "learning_rate": args.learning_rate,
            "torch_threads": args.torch_threads,
        },
        extra={
            "n_train": len(train),
            "n_val": len(val),
            "n_total_rows_after_projection": len(rows),
            "real_pretrained_weights": True,
            "real_human_labels": True,
            "weak_pretraining_dataset": "blocked_zenodo_unreachable",
        },
    )
    manifest_path = args.output_json.with_name(args.output_json.stem + "_manifest.json")
    write_manifest(manifest_path, manifest)
    print(f"[write] manifest -> {manifest_path}")


if __name__ == "__main__":
    main()
