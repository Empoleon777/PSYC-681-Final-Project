#!/usr/bin/env python3
"""Task 26: continuous signal + vector-field analysis for B6 outputs."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from xml.sax.saxutils import escape as xml_escape

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

# Make repo root importable when run as `python scripts/...`.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.b6_hierarchy import B6HierarchyModel
from scripts.train_b6_representation import SimpleHashTokenizer


def read_csv(path: Path) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(
    path: Path,
    rows: List[Dict[str, object]],
    fieldnames: Optional[List[str]] = None,
) -> None:
    if not rows and not fieldnames:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        names = fieldnames or list(rows[0].keys())
        writer = csv.DictWriter(f, fieldnames=names)
        writer.writeheader()
        if rows:
            writer.writerows(rows)


def parse_timestamp(raw: str) -> Optional[datetime]:
    text = (raw or "").strip()
    if not text:
        return None
    try:
        dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except ValueError:
        return None


def month_key(raw: str) -> str:
    dt = parse_timestamp(raw)
    if dt is None:
        return "unknown"
    return dt.strftime("%Y-%m")


class TextDataset(Dataset):
    def __init__(self, rows: List[Dict[str, str]], tokenizer, max_length: int) -> None:
        self.rows = rows
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.rows[idx]
        enc = self.tokenizer(
            row.get("text", ""),
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "idx": torch.tensor(idx, dtype=torch.long),
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
        }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--raw-posts-csv", type=Path, default=Path("outputs/ingestion/raw_posts_60k.csv"))
    p.add_argument("--weak-labels-csv", type=Path, default=Path("outputs/weak_labels_60k/post_weak_labels.csv"))
    p.add_argument("--output-dir", type=Path, default=Path("outputs/task26_signal_analysis"))
    p.add_argument("--max-samples", type=int, default=8000)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--max-length", type=int, default=256)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--encoder-name", type=str, default="")
    p.add_argument("--offline-random-init", action="store_true")
    p.add_argument("--offline-vocab-size", type=int, default=8192)
    return p.parse_args()


def evenly_spaced_sample(rows: List[Dict[str, str]], n: int) -> List[Dict[str, str]]:
    if n <= 0 or n >= len(rows):
        return rows
    if n == 1:
        return [rows[0]]
    step = (len(rows) - 1) / float(n - 1)
    out: List[Dict[str, str]] = []
    for i in range(n):
        idx = int(round(i * step))
        out.append(rows[idx])
    return out


def load_model_and_tokenizer(args: argparse.Namespace) -> Tuple[B6HierarchyModel, object, Dict]:
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    metadata = ckpt.get("metadata", {})
    cfg = metadata.get("config", {})

    encoder_name = args.encoder_name or cfg.get("encoder_name", "roberta-base")
    offline = args.offline_random_init or bool(cfg.get("offline_random_init", False))
    vocab_size = int(cfg.get("offline_vocab_size", args.offline_vocab_size))

    if offline:
        tokenizer = SimpleHashTokenizer(vocab_size=vocab_size)
        model = B6HierarchyModel(
            encoder_name=encoder_name,
            load_pretrained=False,
            vocab_size=vocab_size,
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(encoder_name)
        model = B6HierarchyModel(encoder_name=encoder_name, load_pretrained=True)

    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    return model, tokenizer, metadata


def make_signal_rows(
    model: B6HierarchyModel,
    rows: List[Dict[str, str]],
    tokenizer,
    max_length: int,
    batch_size: int,
    device: torch.device,
) -> List[Dict[str, object]]:
    ds = TextDataset(rows, tokenizer, max_length=max_length)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    model = model.to(device)
    model.eval()

    w_econ = model.econ_head.weight.detach().cpu().numpy()
    w_soc = model.social_head.weight.detach().cpu().numpy()
    econ_axis = w_econ[2] - w_econ[0]
    soc_axis = w_soc[2] - w_soc[0]
    econ_axis = econ_axis / (np.linalg.norm(econ_axis) + 1e-8)
    soc_axis = soc_axis / (np.linalg.norm(soc_axis) + 1e-8)

    outputs: List[Optional[Dict[str, object]]] = [None] * len(rows)
    with torch.no_grad():
        for batch in loader:
            idx = batch["idx"].tolist()
            y = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
            )

            z = y["z"].detach().cpu().numpy()
            econ_probs = torch.softmax(y["econ_logits"].detach().cpu(), dim=-1).numpy()
            soc_probs = torch.softmax(y["social_logits"].detach().cpu(), dim=-1).numpy()
            intensity_probs = torch.softmax(y["intensity_logits"].detach().cpu(), dim=-1).numpy()
            ambiguity_probs = torch.softmax(y["ambiguity_logits"].detach().cpu(), dim=-1).numpy()

            for i, rec_idx in enumerate(idx):
                row = rows[rec_idx]
                econ_cont = float(econ_probs[i, 2] - econ_probs[i, 0])
                soc_cont = float(soc_probs[i, 2] - soc_probs[i, 0])
                intensity_cont = float((intensity_probs[i] * np.array([0.0, 0.5, 1.0])).sum())
                ambiguity_cont = float((ambiguity_probs[i] * np.array([0.0, 0.5, 1.0])).sum())
                ze = float(np.dot(z[i], econ_axis))
                zs = float(np.dot(z[i], soc_axis))

                outputs[rec_idx] = {
                    "post_id": row.get("post_id", ""),
                    "subreddit": row.get("subreddit", ""),
                    "created_at": row.get("created_at", ""),
                    "month": month_key(row.get("created_at", "")),
                    "econ_cont": round(econ_cont, 6),
                    "social_cont": round(soc_cont, 6),
                    "intensity_cont": round(intensity_cont, 6),
                    "ambiguity_cont": round(ambiguity_cont, 6),
                    "z_econ_projection": round(ze, 6),
                    "z_social_projection": round(zs, 6),
                    "z_norm": round(float(np.linalg.norm(z[i])), 6),
                }

    return [r for r in outputs if r is not None]


def aggregate_monthly(signals: List[Dict[str, object]]) -> List[Dict[str, object]]:
    groups: Dict[Tuple[str, str], Dict[str, float]] = defaultdict(
        lambda: {
            "count": 0.0,
            "econ_sum": 0.0,
            "social_sum": 0.0,
            "intensity_sum": 0.0,
            "ambiguity_sum": 0.0,
            "ze_sum": 0.0,
            "zs_sum": 0.0,
        }
    )
    for row in signals:
        key = (str(row["subreddit"]), str(row["month"]))
        g = groups[key]
        g["count"] += 1.0
        g["econ_sum"] += float(row["econ_cont"])
        g["social_sum"] += float(row["social_cont"])
        g["intensity_sum"] += float(row["intensity_cont"])
        g["ambiguity_sum"] += float(row["ambiguity_cont"])
        g["ze_sum"] += float(row["z_econ_projection"])
        g["zs_sum"] += float(row["z_social_projection"])

    out: List[Dict[str, object]] = []
    for (subreddit, month), g in groups.items():
        n = max(1.0, g["count"])
        out.append(
            {
                "subreddit": subreddit,
                "month": month,
                "n_posts": int(n),
                "econ_cont_mean": round(g["econ_sum"] / n, 6),
                "social_cont_mean": round(g["social_sum"] / n, 6),
                "intensity_cont_mean": round(g["intensity_sum"] / n, 6),
                "ambiguity_cont_mean": round(g["ambiguity_sum"] / n, 6),
                "z_econ_projection_mean": round(g["ze_sum"] / n, 6),
                "z_social_projection_mean": round(g["zs_sum"] / n, 6),
            }
        )
    out.sort(key=lambda r: (r["subreddit"], r["month"]))
    return out


def compute_velocity(monthly_rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    by_sub: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for row in monthly_rows:
        if row["month"] == "unknown":
            continue
        by_sub[str(row["subreddit"])].append(row)

    out: List[Dict[str, object]] = []
    for sub, rows in by_sub.items():
        rows = sorted(rows, key=lambda r: str(r["month"]))
        for i in range(len(rows) - 1):
            a = rows[i]
            b = rows[i + 1]
            x = float(a["econ_cont_mean"])
            y = float(a["social_cont_mean"])
            u = float(b["econ_cont_mean"]) - x
            v = float(b["social_cont_mean"]) - y
            out.append(
                {
                    "subreddit": sub,
                    "from_month": a["month"],
                    "to_month": b["month"],
                    "from_econ": round(x, 6),
                    "from_social": round(y, 6),
                    "delta_econ": round(u, 6),
                    "delta_social": round(v, 6),
                    "speed": round(float(np.sqrt(u * u + v * v)), 6),
                }
            )
    return out


def fit_divergence_curl(vel_rows: List[Dict[str, object]]) -> Dict[str, Dict[str, float]]:
    by_month: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for row in vel_rows:
        by_month[str(row["to_month"])].append(row)

    out: Dict[str, Dict[str, float]] = {}
    for month, rows in by_month.items():
        if len(rows) < 3:
            continue
        x = np.array([float(r["from_econ"]) for r in rows], dtype=float)
        y = np.array([float(r["from_social"]) for r in rows], dtype=float)
        u = np.array([float(r["delta_econ"]) for r in rows], dtype=float)
        v = np.array([float(r["delta_social"]) for r in rows], dtype=float)

        design = np.column_stack([x, y, np.ones_like(x)])
        a, b, _ = np.linalg.lstsq(design, u, rcond=None)[0]
        d, e, _ = np.linalg.lstsq(design, v, rcond=None)[0]
        divergence = float(a + e)
        curl = float(d - b)
        out[month] = {
            "n_vectors": float(len(rows)),
            "divergence": divergence,
            "curl": curl,
        }
    return out


def write_svg(
    vel_rows: List[Dict[str, object]],
    monthly_rows: List[Dict[str, object]],
    out_path: Path,
) -> None:
    rows: List[Dict[str, object]] = []
    latest = ""
    mode = "velocity"
    if vel_rows:
        latest = max(str(r["to_month"]) for r in vel_rows)
        rows = [r for r in vel_rows if str(r["to_month"]) == latest]
    else:
        mode = "snapshot"
        valid = [r for r in monthly_rows if str(r["month"]) != "unknown"]
        if not valid:
            return
        latest = max(str(r["month"]) for r in valid)
        rows = [r for r in valid if str(r["month"]) == latest]

    if not rows:
        return

    width, height = 980, 760
    margin = 70
    sx = (width - 2 * margin) / 2.0
    sy = (height - 2 * margin) / 2.0

    def map_xy(x: float, y: float) -> Tuple[float, float]:
        px = margin + (x + 1.0) * sx
        py = height - margin - (y + 1.0) * sy
        return px, py

    scale = 2.8
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        '<rect x="0" y="0" width="100%" height="100%" fill="#ffffff"/>',
        f'<line x1="{margin}" y1="{height/2}" x2="{width-margin}" y2="{height/2}" stroke="#cccccc" stroke-width="1"/>',
        f'<line x1="{width/2}" y1="{margin}" x2="{width/2}" y2="{height-margin}" stroke="#cccccc" stroke-width="1"/>',
        f'<text x="{margin}" y="{margin-20}" font-size="18" font-family="Helvetica">Task 26 {xml_escape(mode.title())} ({xml_escape(latest)})</text>',
    ]

    if mode == "velocity":
        for row in rows:
            x = float(row["from_econ"])
            y = float(row["from_social"])
            u = float(row["delta_econ"]) * scale
            v = float(row["delta_social"]) * scale
            x1, y1 = map_xy(x, y)
            x2, y2 = map_xy(max(-1.2, min(1.2, x + u)), max(-1.2, min(1.2, y + v)))
            sub = xml_escape(str(row["subreddit"]))
            lines.append(f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" stroke="#1f77b4" stroke-width="2"/>')
            lines.append(f'<circle cx="{x1:.2f}" cy="{y1:.2f}" r="3" fill="#d62728"/>')
            lines.append(f'<text x="{x1+5:.2f}" y="{y1-5:.2f}" font-size="10" font-family="Helvetica">{sub}</text>')
    else:
        for row in rows:
            x = float(row["econ_cont_mean"])
            y = float(row["social_cont_mean"])
            x1, y1 = map_xy(x, y)
            sub = xml_escape(str(row["subreddit"]))
            lines.append(f'<circle cx="{x1:.2f}" cy="{y1:.2f}" r="3.5" fill="#1f77b4"/>')
            lines.append(f'<text x="{x1+5:.2f}" y="{y1-5:.2f}" font-size="10" font-family="Helvetica">{sub}</text>')

    lines.append("</svg>")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    if not args.checkpoint.exists():
        raise SystemExit(f"Missing checkpoint: {args.checkpoint}")
    raw_rows = read_csv(args.raw_posts_csv)
    weak_ids = {r["post_id"] for r in read_csv(args.weak_labels_csv)} if args.weak_labels_csv.exists() else set()

    selected = [r for r in raw_rows if r.get("post_id", "") in weak_ids and (r.get("text") or "").strip()]
    if args.max_samples > 0:
        selected = evenly_spaced_sample(selected, args.max_samples)
    if not selected:
        raise SystemExit("No rows selected for signal analysis.")

    model, tokenizer, metadata = load_model_and_tokenizer(args)
    signals = make_signal_rows(
        model=model,
        rows=selected,
        tokenizer=tokenizer,
        max_length=args.max_length,
        batch_size=args.batch_size,
        device=torch.device(args.device),
    )
    monthly = aggregate_monthly(signals)
    velocity = compute_velocity(monthly)
    divcurl = fit_divergence_curl(velocity)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "continuous_signals.csv", signals)
    write_csv(args.output_dir / "community_monthly_means.csv", monthly)
    write_csv(
        args.output_dir / "community_velocity.csv",
        velocity,
        fieldnames=[
            "subreddit",
            "from_month",
            "to_month",
            "from_econ",
            "from_social",
            "delta_econ",
            "delta_social",
            "speed",
        ],
    )
    write_svg(velocity, monthly, args.output_dir / "vector_field.svg")

    summary = {
        "checkpoint": str(args.checkpoint),
        "rows_used": len(signals),
        "communities": len({r["subreddit"] for r in monthly}),
        "months": sorted({r["month"] for r in monthly}),
        "divergence_curl_by_month": divcurl,
        "metadata_snapshot": metadata.get("config", {}),
    }
    (args.output_dir / "vector_field_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )

    print(f"Wrote Task 26 outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
