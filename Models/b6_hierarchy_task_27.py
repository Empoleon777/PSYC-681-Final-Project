#!/usr/bin/env python3
"""Tasks 27/28 extension heads for B6: evidence + psychology."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.b6_hierarchy import B6HierarchyModel


def token_prf1(logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor, threshold: float = 0.5) -> Dict[str, float]:
    probs = torch.sigmoid(logits)
    preds = probs >= threshold
    valid = mask.bool()
    tp = ((preds == 1) & (labels == 1) & valid).sum().item()
    fp = ((preds == 1) & (labels == 0) & valid).sum().item()
    fn = ((preds == 0) & (labels == 1) & valid).sum().item()
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-8, precision + recall)
    return {"precision": precision, "recall": recall, "f1": f1}


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
        texts: List[str],
        return_tensors: str = "pt",
        padding: bool = True,
        truncation: bool = True,
        max_length: int = 64,
    ) -> Dict[str, torch.Tensor]:
        rows_ids: List[List[int]] = []
        rows_mask: List[List[int]] = []
        for text in texts:
            token_ids = [self._token_to_id(tok) for tok in text.lower().split()]
            if truncation:
                token_ids = token_ids[: max(0, max_length - 2)]
            ids = [self.cls_id] + token_ids + [self.sep_id]
            ids = ids[:max_length]
            mask = [1] * len(ids)
            if padding and len(ids) < max_length:
                pad_n = max_length - len(ids)
                ids.extend([self.pad_id] * pad_n)
                mask.extend([0] * pad_n)
            rows_ids.append(ids)
            rows_mask.append(mask)
        if return_tensors != "pt":
            raise ValueError("SimpleHashTokenizer only supports return_tensors='pt'")
        return {
            "input_ids": torch.tensor(rows_ids, dtype=torch.long),
            "attention_mask": torch.tensor(rows_mask, dtype=torch.long),
        }


class B6EvidencePsychModel(B6HierarchyModel):
    def __init__(
        self,
        encoder_name: str = "roberta-base",
        load_pretrained: bool = True,
        vocab_size: int = 30522,
        num_targets: int = 7,
        num_stances: int = 3,
        num_frames: int = 9,
        latent_dim: int = 128,
        enable_evidence: bool = True,
        enable_psych: bool = True,
    ) -> None:
        super().__init__(
            encoder_name=encoder_name,
            load_pretrained=load_pretrained,
            vocab_size=vocab_size,
            num_targets=num_targets,
            num_stances=num_stances,
            num_frames=num_frames,
            latent_dim=latent_dim,
        )
        self.enable_evidence = enable_evidence
        self.enable_psych = enable_psych
        hidden = self.encoder.config.hidden_size

        self.evidence_head = nn.Linear(hidden, 1)
        self.moralization_head = nn.Linear(hidden, 3)
        self.identity_signaling_head = nn.Linear(hidden, 3)
        self.moralization_repr = nn.Linear(3, latent_dim)
        self.identity_signaling_repr = nn.Linear(3, latent_dim)
        self.psych_scale = nn.Parameter(torch.tensor(1.0))
        self.evidence_scale = nn.Parameter(torch.tensor(1.0))

        in_features = self.ideology_projection[0].in_features
        extra = (hidden if enable_evidence else 0) + (2 * latent_dim if enable_psych else 0)
        self.ideology_projection[0] = nn.Linear(in_features + extra, latent_dim)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        return_intermediates: bool = True,
        labels: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        base = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = base.last_hidden_state[:, 0]
        token_emb = base.last_hidden_state

        relevance_logits = self.relevance_head(pooled)
        target_logits = self.target_head(pooled)
        stance_logits = self.stance_head(pooled)
        frame_logits = self.frame_head(pooled)

        target_h = self.target_repr(F.softmax(target_logits, dim=-1))
        stance_h = self.stance_repr(F.softmax(stance_logits, dim=-1))
        frame_h = self.frame_repr(F.softmax(frame_logits, dim=-1))

        z_parts = [pooled, target_h, stance_h, frame_h]
        out: Dict[str, torch.Tensor] = {}

        if self.enable_evidence:
            evidence_logits = self.evidence_head(token_emb).squeeze(-1)
            ev_mask = attention_mask.float()
            ev_probs = torch.sigmoid(evidence_logits) * ev_mask
            weighted = torch.bmm(ev_probs.unsqueeze(1), token_emb).squeeze(1)
            norm = ev_probs.sum(dim=1, keepdim=True).clamp(min=1e-6)
            evidence_repr = self.evidence_scale * (weighted / norm)
            z_parts.append(evidence_repr)
            out["evidence_logits"] = evidence_logits
            out["evidence_probs"] = ev_probs
            out["evidence_repr"] = evidence_repr

        if self.enable_psych:
            moralization_logits = self.moralization_head(pooled)
            identity_logits = self.identity_signaling_head(pooled)
            moralization_probs = F.softmax(moralization_logits, dim=-1)
            identity_probs = F.softmax(identity_logits, dim=-1)
            psych_repr = torch.cat(
                [
                    self.moralization_repr(moralization_probs),
                    self.identity_signaling_repr(identity_probs),
                ],
                dim=-1,
            )
            psych_repr = self.psych_scale * psych_repr
            z_parts.append(psych_repr)
            out["moralization_logits"] = moralization_logits
            out["identity_signaling_logits"] = identity_logits
            out["moralization_probs"] = moralization_probs
            out["identity_signaling_probs"] = identity_probs

        z_input = torch.cat(z_parts, dim=-1)
        z = self.ideology_projection(z_input)
        econ_logits = self.econ_head(z)
        social_logits = self.social_head(z)
        intensity_logits = self.intensity_head(z)
        ambiguity_logits = self.ambiguity_head(z)

        out.update(
            {
                "relevance_logits": relevance_logits,
                "target_logits": target_logits,
                "stance_logits": stance_logits,
                "frame_logits": frame_logits,
                "econ_logits": econ_logits,
                "social_logits": social_logits,
                "intensity_logits": intensity_logits,
                "ambiguity_logits": ambiguity_logits,
                "z": z,
                "orthogonality_loss": self.orthogonality_regularization(),
            }
        )
        if return_intermediates:
            out["target_h"] = target_h
            out["stance_h"] = stance_h
            out["frame_h"] = frame_h
        if labels is not None:
            out.update(self.compute_loss(out, labels))
            if self.enable_evidence and "evidence_labels" in labels:
                ev_labels = labels["evidence_labels"].float()
                ev_mask = labels["evidence_mask"].float()
                bce = F.binary_cross_entropy_with_logits(out["evidence_logits"], ev_labels, reduction="none")
                out["evidence_bce_loss"] = (bce * ev_mask).sum() / ev_mask.sum().clamp(min=1.0)
                out["loss"] = out["loss"] + 0.5 * out["evidence_bce_loss"]
            if self.enable_psych and "moralization_dist" in labels and "identity_signaling_dist" in labels:
                m_loss = F.kl_div(F.log_softmax(out["moralization_logits"], dim=-1), labels["moralization_dist"], reduction="batchmean")
                i_loss = F.kl_div(F.log_softmax(out["identity_signaling_logits"], dim=-1), labels["identity_signaling_dist"], reduction="batchmean")
                out["psych_loss"] = m_loss + i_loss
                out["loss"] = out["loss"] + 0.5 * out["psych_loss"]
        return out


def read_csv(path: Path) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def check_gold_psych_labels(gold_annotations_csv: Path) -> Dict[str, int]:
    rows = read_csv(gold_annotations_csv)
    moral = sum(1 for r in rows if (r.get("q05_moralization") or "").strip() != "")
    identity = sum(1 for r in rows if (r.get("q06_identity_signaling") or "").strip() != "")
    return {"rows_total": len(rows), "rows_with_moralization": moral, "rows_with_identity_signaling": identity}


def run_forward_check(enable_evidence: bool, enable_psych: bool, offline_random_init: bool) -> Dict[str, object]:
    model = B6EvidencePsychModel(
        load_pretrained=not offline_random_init,
        enable_evidence=enable_evidence,
        enable_psych=enable_psych,
    )
    tokenizer = SimpleHashTokenizer() if offline_random_init else AutoTokenizer.from_pretrained("roberta-base")
    batch = tokenizer(
        [
            "I support stronger labor protections and union rights.",
            "As a conservative, I favor stricter border enforcement.",
        ],
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=64,
    )
    out = model(batch["input_ids"], batch["attention_mask"])
    payload = {
        "enable_evidence": enable_evidence,
        "enable_psych": enable_psych,
        "keys": sorted(out.keys()),
        "z_shape": tuple(out["z"].shape),
        "has_evidence_head": "evidence_logits" in out,
        "has_psych_heads": ("moralization_logits" in out and "identity_signaling_logits" in out),
    }
    if "evidence_logits" in out:
        sampled_labels = torch.randint(0, 2, out["evidence_logits"].shape)
        metrics = token_prf1(out["evidence_logits"], sampled_labels, batch["attention_mask"])
        payload["token_prf1_sampled"] = metrics
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold-annotations-csv", type=Path, default=Path("outputs/annotation_60k/gold_annotations.csv"))
    parser.add_argument("--offline-random-init", action="store_true")
    parser.add_argument("--output-json", type=Path, default=Path("outputs/model_runs/task27_28_head_checks.json"))
    args = parser.parse_args()

    label_stats = check_gold_psych_labels(args.gold_annotations_csv)
    base = run_forward_check(enable_evidence=True, enable_psych=True, offline_random_init=args.offline_random_init)
    no_ev = run_forward_check(enable_evidence=False, enable_psych=True, offline_random_init=args.offline_random_init)
    no_ps = run_forward_check(enable_evidence=True, enable_psych=False, offline_random_init=args.offline_random_init)
    payload = {
        "gold_label_stats": label_stats,
        "full_model": base,
        "ablation_no_evidence": no_ev,
        "ablation_no_psych": no_ps,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
