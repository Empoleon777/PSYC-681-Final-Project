#!/usr/bin/env python3
"""Task 16: B2 flat transformer baseline on gold hard labels."""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer, RobertaConfig, RobertaModel


TASKS = {
    "relevance": ("q01_relevance", ["0", "1"]),
    "economic_direction": ("q07_economic_direction", ["-1", "0", "1"]),
    "social_direction": ("q08_social_direction", ["-1", "0", "1"]),
}


@dataclass
class Row:
    text: str
    labels: Dict[str, int]
    post_id: str = ""


class GoldDataset(Dataset):
    def __init__(self, rows: List[Row], tokenizer, max_length: int) -> None:
        self.rows = rows
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.rows[idx]
        enc = self.tokenizer(
            row.text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        out = {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
        }
        for task in TASKS:
            out[f"label_{task}"] = torch.tensor(row.labels[task], dtype=torch.long)
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


class B2FlatModel(nn.Module):
    def __init__(self, encoder_name: str = "roberta-base", offline_random_init: bool = False) -> None:
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
        self.relevance_head = nn.Linear(hidden, 2)
        self.econ_head = nn.Linear(hidden, 3)
        self.social_head = nn.Linear(hidden, 3)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = out.last_hidden_state[:, 0]
        return {
            "relevance": self.relevance_head(pooled),
            "economic_direction": self.econ_head(pooled),
            "social_direction": self.social_head(pooled),
        }


def read_csv(path: Path) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def load_gold_rows(raw_posts_csv: Path, gold_aggregates_csv: Path) -> List[Row]:
    raw_rows = read_csv(raw_posts_csv)
    gold_rows = read_csv(gold_aggregates_csv)
    raw_by_post = {r["post_id"]: r for r in raw_rows}
    rows: List[Row] = []
    for g in gold_rows:
        raw = raw_by_post.get(g["post_id"])
        if raw is None:
            continue
        text = (raw.get("text") or "").strip()
        if not text:
            continue
        majority = json.loads(g["majority_json"])
        labels: Dict[str, int] = {}
        valid = True
        for task, (key, values) in TASKS.items():
            val = str(majority.get(key, "")).strip()
            if val not in values:
                valid = False
                break
            labels[task] = values.index(val)
        if valid:
            rows.append(Row(text=text, labels=labels, post_id=g["post_id"]))
    return rows


def split_rows(rows: List[Row], seed: int) -> Tuple[List[Row], List[Row]]:
    local = list(rows)
    random.Random(seed).shuffle(local)
    n_val = max(1, int(0.2 * len(local)))
    return local[n_val:], local[:n_val]


def macro_f1(y_true: List[int], y_pred: List[int]) -> float:
    return float(f1_score(y_true, y_pred, average="macro", zero_division=0))


def evaluate(model: B2FlatModel, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    model.eval()
    y_true = {task: [] for task in TASKS}
    y_pred = {task: [] for task in TASKS}
    with torch.no_grad():
        for batch in loader:
            logits = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
            )
            for task in TASKS:
                preds = logits[task].argmax(dim=-1).cpu().tolist()
                truth = batch[f"label_{task}"].tolist()
                y_true[task].extend(truth)
                y_pred[task].extend(preds)
    out = {}
    vals = []
    for task in TASKS:
        f1 = macro_f1(y_true[task], y_pred[task])
        out[f"{task}_macro_f1"] = f1
        vals.append(f1)
    out["macro_f1_mean"] = float(sum(vals) / len(vals))
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-posts-csv", type=Path, default=Path("outputs/ingestion/raw_posts_60k.csv"))
    parser.add_argument("--gold-aggregates-csv", type=Path, default=Path("outputs/annotation_60k/gold_aggregates.csv"))
    parser.add_argument("--encoder-name", type=str, default="roberta-base")
    parser.add_argument("--offline-random-init", action="store_true")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-json", type=Path, default=Path("outputs/model_runs/b2_summary.json"))
    parser.add_argument(
        "--output-predictions-csv",
        type=Path,
        default=None,
        help="Optional path to dump per-post val-set predictions for Task 35 error analysis.",
    )
    args = parser.parse_args()

    rows = load_gold_rows(args.raw_posts_csv, args.gold_aggregates_csv)
    if len(rows) < 50:
        raise SystemExit("Not enough rows for B2.")
    train_rows, val_rows = split_rows(rows, args.seed)

    if args.offline_random_init:
        tokenizer = SimpleHashTokenizer()
        model = B2FlatModel(encoder_name=args.encoder_name, offline_random_init=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.encoder_name)
        model = B2FlatModel(encoder_name=args.encoder_name, offline_random_init=False)

    train_ds = GoldDataset(train_rows, tokenizer, args.max_length)
    val_ds = GoldDataset(val_rows, tokenizer, args.max_length)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    device = torch.device(args.device)
    model = model.to(device)
    if args.learning_rate > 0:
        lr = args.learning_rate
    else:
        # Random-init offline mode needs a much larger LR than pretrained fine-tuning.
        lr = 3e-4 if args.offline_random_init else 2e-5
    opt = AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        steps = 0
        for batch in train_loader:
            opt.zero_grad()
            logits = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
            )
            loss = 0.0
            for task in TASKS:
                loss = loss + loss_fn(logits[task], batch[f"label_{task}"].to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            running += float(loss.detach().cpu())
            steps += 1
        metrics = evaluate(model, val_loader, device)
        print(f"[epoch {epoch}] train_loss={running/max(1,steps):.4f} val_macro_f1={metrics['macro_f1_mean']:.4f}")
        print(metrics)
    payload = {
        "variant": "b2",
        "learning_rate": lr,
        "train_rows": len(train_rows),
        "val_rows": len(val_rows),
        "metrics": metrics,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")

    if args.output_predictions_csv is not None:
        write_b2_val_predictions(model, val_rows, tokenizer, args.max_length, args.batch_size, device, args.output_predictions_csv)
        print(f"Wrote per-post val predictions to {args.output_predictions_csv}")

    # Reproducibility manifest.
    REPO_ROOT = Path(__file__).resolve().parents[1]
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from scripts._run_manifest import build_manifest, write_manifest

    manifest = build_manifest(
        run_name=f"b2_seed{args.seed}",
        seed=args.seed,
        inputs={
            "raw_posts_csv": args.raw_posts_csv,
            "gold_aggregates_csv": args.gold_aggregates_csv,
        },
        outputs={
            "summary_json": str(args.output_json),
            "val_predictions_csv": str(args.output_predictions_csv) if args.output_predictions_csv else None,
        },
        config={
            "encoder_name": args.encoder_name,
            "offline_random_init": args.offline_random_init,
            "batch_size": args.batch_size,
            "max_length": args.max_length,
            "epochs": args.epochs,
            "learning_rate": lr,
        },
        extra={"n_train": len(train_rows), "n_val": len(val_rows)},
    )
    manifest_path = args.output_json.with_name(args.output_json.stem + "_manifest.json")
    write_manifest(manifest_path, manifest)
    print(f"Wrote run manifest to {manifest_path}")


@torch.no_grad()
def write_b2_val_predictions(
    model: B2FlatModel,
    val_rows: List[Row],
    tokenizer,
    max_length: int,
    batch_size: int,
    device: torch.device,
    output_csv: Path,
) -> None:
    model.eval()
    val_ds = GoldDataset(val_rows, tokenizer, max_length)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    all_preds: Dict[str, List[int]] = {t: [] for t in TASKS}
    all_probs: Dict[str, List[List[float]]] = {t: [] for t in TASKS}
    for batch in val_loader:
        logits = model(
            input_ids=batch["input_ids"].to(device),
            attention_mask=batch["attention_mask"].to(device),
        )
        for task in TASKS:
            l = logits[task].detach().cpu()
            probs = torch.softmax(l, dim=-1)
            preds = l.argmax(dim=-1).tolist()
            all_preds[task].extend(preds)
            all_probs[task].extend(probs.tolist())
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["post_id", "split_name", "model_name"]
    for task in TASKS:
        fieldnames += [f"pred_{task}", f"probs_{task}"]
    with open(output_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i, rec in enumerate(val_rows):
            row: Dict[str, str] = {
                "post_id": rec.post_id,
                "split_name": "val",
                "model_name": "b2",
            }
            for task, (_, values) in TASKS.items():
                pred_idx = all_preds[task][i]
                row[f"pred_{task}"] = values[pred_idx] if 0 <= pred_idx < len(values) else ""
                row[f"probs_{task}"] = json.dumps(
                    {values[j]: float(p) for j, p in enumerate(all_probs[task][i])},
                    ensure_ascii=True,
                )
            writer.writerow(row)


if __name__ == "__main__":
    main()
