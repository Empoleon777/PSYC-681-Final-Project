"""Draft B6 hierarchy model.

This is a working architecture draft, not a finished training implementation.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from transformers import AutoModel, AutoTokenizer
import os


class B6HierarchyDraft(nn.Module):
    def __init__(
        self,
        weighted = False,
        psych = False,
        encoder_name: str = "roberta-base",
        num_targets: int = 7,
        num_stances: int = 3,
        num_frames: int = 9,
        latent_dim: int = 128,
    ) -> None:
        super().__init__()
        self.weighted = weighted
        self.psych = psych        
        self.psych_scale = nn.Parameter(torch.tensor(1.0))
        self.evidence_scale = nn.Parameter(torch.tensor(1.0))
        self.encoder = AutoModel.from_pretrained(encoder_name)
        hidden = self.encoder.config.hidden_size

        # Discourse-level heads.
        self.relevance_head = nn.Linear(hidden, 2)
        self.target_head = nn.Linear(hidden, num_targets)
        self.stance_head = nn.Linear(hidden, num_stances)
        self.frame_head = nn.Linear(hidden, num_frames)

        # Evidence head
        self.evidence_head = nn.Linear(hidden, 1)

        # Psychological cue heads
        self.moralization_head = nn.Linear(hidden, 3)
        self.identity_signaling_head = nn.Linear(hidden, 3)

        # Intermediate representations used by the ideology projection layer.
        self.target_repr = nn.Linear(num_targets, latent_dim)
        self.stance_repr = nn.Linear(num_stances, latent_dim)
        self.frame_repr = nn.Linear(num_frames, latent_dim)

        self.moralization_repr = nn.Linear(3, latent_dim)
        self.identity_signaling_repr = nn.Linear(3, latent_dim)

        base_dim = (2 * hidden if self.weighted else hidden) + 3 * latent_dim
        input_dim = base_dim + (2 * latent_dim if self.psych else 0)
        # Projection into latent ideology vector z(x).
        self.ideology_projection = nn.Sequential(
            nn.Linear(input_dim, latent_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(latent_dim, latent_dim),
            nn.GELU(),
        )

        # Ideology-level heads.
        self.econ_head = nn.Linear(latent_dim, 3)
        self.social_head = nn.Linear(latent_dim, 3)
        self.intensity_head = nn.Linear(latent_dim, 3)
        self.ambiguity_head = nn.Linear(latent_dim, 1)

    def orthogonality_regularization(self) -> torch.Tensor:
        """Encourage decorrelated Level B head directions."""
        weights = torch.stack(
            [
                F.normalize(self.econ_head.weight.mean(dim=0), dim=0),
                F.normalize(self.social_head.weight.mean(dim=0), dim=0),
                F.normalize(self.intensity_head.weight.mean(dim=0), dim=0),
                F.normalize(self.ambiguity_head.weight.mean(dim=0), dim=0),
            ],
            dim=0,
        )
        gram = weights @ weights.T
        identity = torch.eye(gram.size(0), device=gram.device)
        return ((gram - identity) ** 2).mean()
    
    def compute_loss(self, outputs, labels):
        loss_dict = {}

        def kl_loss(logits, target_probs):
            log_probs = F.log_softmax(logits, dim=-1)
            return F.kl_div(log_probs, target_probs, reduction="batchmean")

        loss_econ = kl_loss(outputs["econ_logits"], labels["econ_dist"])
        loss_social = kl_loss(outputs["social_logits"], labels["social_dist"])
        loss_intensity = kl_loss(outputs["intensity_logits"], labels["intensity_dist"])
        
        if self.psych: 
            loss_moralization = kl_loss(outputs["moralization_logits"], labels["moralization_dist"])
            loss_identity_signaling = kl_loss(outputs["identity_signaling_logits"], labels["identity_signaling_dist"])
        else:
            loss_moralization = 0
            loss_identity_signaling = 0

        p = labels["econ_dist"]
        entropy = -(p * torch.log(p + 1e-8)).sum(dim=-1)

        bins = torch.bucketize(entropy, torch.tensor([0.5, 1.0], device=entropy.device))
        ambiguity_target = F.one_hot(bins, num_classes=3).float()

        loss_amb = kl_loss(outputs["ambiguity_logits"], ambiguity_target)

        loss_evidence = F.binary_cross_entropy_with_logits(
            outputs["evidence_logits"],
            labels["evidence_labels"].unsqueeze(-1)
        )

        total_loss = (
            loss_econ
            + loss_social
            + loss_intensity
            + 0.5 * loss_amb
            + 0.1 * outputs["orthogonality_loss"]
            + 0.5 * loss_evidence
        )

        if self.psych:
            total_loss += 0.5 * (loss_moralization + loss_identity_signaling)

        loss_dict.update({
            "loss": total_loss,
            "econ": loss_econ,
            "social": loss_social,
            "intensity": loss_intensity,
            "ambiguity": loss_amb,
            "moralization": loss_moralization,
            "identity_signaling": loss_identity_signaling
        })

        return loss_dict

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        return_intermediates: bool = True,
        labels: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state[:, 0]

        token_embeddings = outputs.last_hidden_state
        evidence_logits = self.evidence_head(token_embeddings)
        evidence_probs = torch.sigmoid(evidence_logits)
        mask = attention_mask.unsqueeze(-1).float()
        evidence_probs = evidence_probs * mask
        weighted_sum = (evidence_probs * token_embeddings).sum(dim=1)
        norm = evidence_probs.sum(dim=1).clamp(min=1e-6)
        evidence_repr = self.evidence_scale * weighted_sum/norm

        if self.weighted:
            encoder_features = torch.cat([pooled, evidence_repr], dim=-1)
        else:
            encoder_features = pooled

        # Discourse-level predictions.
        relevance_logits = self.relevance_head(pooled)
        target_logits = self.target_head(pooled)
        stance_logits = self.stance_head(pooled)
        frame_logits = self.frame_head(pooled)

        # Convert discourse logits into intermediate representations.
        target_probs = F.softmax(target_logits, dim=-1)
        stance_probs = F.softmax(stance_logits, dim=-1)
        frame_probs = F.softmax(frame_logits, dim=-1)

        target_h = self.target_repr(target_probs)
        stance_h = self.stance_repr(stance_probs)
        frame_h = self.frame_repr(frame_probs)

        # Project pooled encoder state and discourse representations into z.
        if self.psych:            
            moralization_logits = self.moralization_head(pooled)
            identity_signaling_logits = self.identity_signaling_head(pooled)
            moralization_probs = F.softmax(moralization_logits, dim=-1)
            identity_signaling_probs = F.softmax(identity_signaling_logits, dim=-1)
            moralization_h = self.moralization_repr(moralization_probs)
            identity_signaling_h = self.identity_signaling_repr(identity_signaling_probs)
            p = torch.cat([moralization_h, identity_signaling_h], dim=-1)
            p = self.psych_scale * p
            z_input = torch.cat([encoder_features, target_h, stance_h, frame_h, p], dim=-1)
        else:
            z_input = torch.cat([encoder_features, target_h, stance_h, frame_h], dim=-1)
        
        z = self.ideology_projection(z_input)

        # Ideology predictions from z.
        econ_logits = self.econ_head(z)
        social_logits = self.social_head(z)
        intensity_logits = self.intensity_head(z)
        ambiguity_logits = self.ambiguity_head(z)

        out = {
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
            "evidence_logits": evidence_logits,
            "evidence_probs": evidence_probs,
            "evidence_repr": evidence_repr,
        }

        if self.psych:
            out.update({
                "moralization_logits": moralization_logits,
                "identity_signaling_logits": identity_signaling_logits,
                "moralization_probs": moralization_probs,
                "identity_signaling_probs": identity_signaling_probs,
                "p": p
            })

        if return_intermediates:
            out["target_h"] = target_h
            out["stance_h"] = stance_h
            out["frame_h"] = frame_h

        if labels is not None:
            loss_dict = self.compute_loss(out, labels)
            out.update(loss_dict)

        return out

def mean_abs_diff(a, b):
    return (a - b).abs().mean().item()
    
def signed_diff(a, b):
    return (a - b).mean().item()

def rel_diff(a, b):
    return ((a - b).abs().mean() / (a.abs().mean() + 1e-6)).item()

def run_evidence_ablation(model_cls, enc, tokenizer=None, print_tokens=True):
    """
    model_cls: class (e.g., B6HierarchyDraft)
    enc: tokenizer output dict (input_ids, attention_mask)
    tokenizer: optional, for qualitative token print
    """

    torch.manual_seed(42)
    model_no = model_cls(weighted=False)
    torch.manual_seed(42)
    model_yes = model_cls(weighted=True)

    model_no.evidence_scale.data.fill_(0.0)
    model_yes.evidence_scale.data.fill_(1.0)

    model_no.eval()
    model_yes.eval()

    with torch.no_grad():
        y_no = model_no(**enc)
        y_yes = model_yes(**enc)

    print("\n=== Evidence Ablation ===")

    print("\n[Representation]")
    print("z diff:", mean_abs_diff(y_yes["z"], y_no["z"]))

    print("\n[Ideology logits]")
    print(f"Economic Difference: {mean_abs_diff(y_yes['econ_logits'], y_no['econ_logits'])}")
    print(f"Social Difference: {mean_abs_diff(y_yes['social_logits'], y_no['social_logits'])}")
    print(f"Intensity Difference: {mean_abs_diff(y_yes['intensity_logits'], y_no['intensity_logits'])}")

    print(f"Economic Relative Difference: {rel_diff(y_yes['econ_logits'], y_no['econ_logits'])}")
    print(f"Social Relative Difference: {rel_diff(y_yes['social_logits'], y_no['social_logits'])}")
    print(f"Intensity Relative Difference: {rel_diff(y_yes['intensity_logits'], y_no['intensity_logits'])}")

    print(f"Economic Signed Shift: {signed_diff(y_yes['econ_logits'], y_no['econ_logits'])}")
    print(f"Social Signed Shift: {signed_diff(y_yes['social_logits'], y_no['social_logits'])}")
    print(f"Intensity Signed Shift: {signed_diff(y_yes['intensity_logits'], y_no['intensity_logits'])}")

    probs = y_yes["evidence_probs"]
    print("\n[Evidence stats]")
    print("mean:", probs.mean().item())
    print("std :", probs.std().item())
    print("sum per example:", probs.sum(dim=1).squeeze(-1))

    if tokenizer is not None and print_tokens:
        print("\n[Token-level evidence (example 0)]")
        tokens = tokenizer.convert_ids_to_tokens(enc["input_ids"][0])
        scores = probs[0].squeeze(-1).cpu().numpy()

        for t, s in zip(tokens, scores):
            t_clean = t.replace("Ġ", "")
            print(f"{t_clean:15} {s:.3f}")

    return {
        "z_diff": mean_abs_diff(y_yes["z"], y_no["z"]),
        "econ_diff": mean_abs_diff(y_yes["econ_logits"], y_no["econ_logits"]),
        "social_diff": mean_abs_diff(y_yes["social_logits"], y_no["social_logits"]),
        "intensity_diff": mean_abs_diff(y_yes["intensity_logits"], y_no["intensity_logits"]),
        "evidence_mean": probs.mean().item(),
        "evidence_std": probs.std().item(),
    }

def run_psych_ablation(model_cls, enc, tokenizer=None, print_tokens=True):
    """
    model_cls: class (e.g., B6HierarchyDraft)
    enc: tokenizer output dict (input_ids, attention_mask)
    tokenizer: optional, for qualitative token print
    """

    torch.manual_seed(42)
    model_no = model_cls(psych=True)
    torch.manual_seed(42)
    model_yes = model_cls(psych=True)

    model_no.psych_scale.data.fill_(0.0)
    model_yes.psych_scale.data.fill_(1.0)

    model_no.eval()
    model_yes.eval()

    with torch.no_grad():
        y_no = model_no(**enc)
        y_yes = model_yes(**enc)

    print("\n=== Psychology Ablation ===")

    print("\n[Representation]")
    print("z diff:", mean_abs_diff(y_yes["z"], y_no["z"]))

    print("\n[Ideology logits]")
    print(f"Economic Difference: {mean_abs_diff(y_yes['econ_logits'], y_no['econ_logits'])}")
    print(f"Social Difference: {mean_abs_diff(y_yes['social_logits'], y_no['social_logits'])}")
    print(f"Intensity Difference: {mean_abs_diff(y_yes['intensity_logits'], y_no['intensity_logits'])}")

    print(f"Economic Relative Difference: {rel_diff(y_yes['econ_logits'], y_no['econ_logits'])}")
    print(f"Social Relative Difference: {rel_diff(y_yes['social_logits'], y_no['social_logits'])}")
    print(f"Intensity Relative Difference: {rel_diff(y_yes['intensity_logits'], y_no['intensity_logits'])}")

    print(f"Economic Signed Shift: {signed_diff(y_yes['econ_logits'], y_no['econ_logits'])}")
    print(f"Social Signed Shift: {signed_diff(y_yes['social_logits'], y_no['social_logits'])}")
    print(f"Intensity Signed Shift: {signed_diff(y_yes['intensity_logits'], y_no['intensity_logits'])}")

    m_probs = y_yes["moralization_probs"]
    id_probs = y_yes["identity_signaling_probs"]
    print("\n[Moralization stats]")
    print("mean:", m_probs.mean().item())
    print("std :", m_probs.std().item())
    print("sum per example:", m_probs.sum(dim=1).squeeze(-1))

    print("\n[Identity Signaling stats]")
    print("mean:", id_probs.mean().item())
    print("std :", id_probs.std().item())
    print("sum per example:", id_probs.sum(dim=1).squeeze(-1))

    # if tokenizer is not None and print_tokens:
    #     print("\n[Token-level evidence (example 0)]")
    #     tokens = tokenizer.convert_ids_to_tokens(enc["input_ids"][0])
    #     m_scores = m_probs[0].squeeze(-1).cpu().numpy()

    #     for t, s in zip(tokens, m_scores):
    #         t_clean = t.replace("Ġ", "")
    #         print(f"{t_clean:15} {s:.3f}")

    #     id_scores = id_probs[0].squeeze(-1).cpu().numpy()

    #     for t, s in zip(tokens, id_scores):
    #         t_clean = t.replace("Ġ", "")
    #         print(f"{t_clean:15} {s:.3f}")

    print("Moralization probs:", m_probs[0])
    print("Identity probs:", id_probs[0])

    return {
        "z_diff": mean_abs_diff(y_yes["z"], y_no["z"]),
        "econ_diff": mean_abs_diff(y_yes["econ_logits"], y_no["econ_logits"]),
        "social_diff": mean_abs_diff(y_yes["social_logits"], y_no["social_logits"]),
        "intensity_diff": mean_abs_diff(y_yes["intensity_logits"], y_no["intensity_logits"]),
        "moralization_mean": m_probs.mean().item(),
        "moralization_std": m_probs.std().item(),
        "identity_signaling_mean": id_probs.mean().item(),
        "identity_signaling_std": id_probs.std().item(),
    }

def graph_evidence_ablation(model_cls, enc):
    torch.manual_seed(42)
    model = model_cls(weighted=True)
    model.eval()

    results = []

    with torch.no_grad():
        # baseline (scale = 0)
        model.psych_scale.data.fill_(0.0)
        y_base = model(**enc)

        for s in torch.linspace(0, 1, steps=11):
            model.psych_scale.data.fill_(s.item())
            y = model(**enc)

            entry = {
                "scale": s.item(),
                "z_diff": (y["z"] - y_base["z"]).abs().mean().item(),
                "econ_diff": (y["econ_logits"] - y_base["econ_logits"]).abs().mean().item(),
                "social_diff": (y["social_logits"] - y_base["social_logits"]).abs().mean().item(),
                "intensity_diff": (y["intensity_logits"] - y_base["intensity_logits"]).abs().mean().item(),
                "econ_signed": (y["econ_logits"] - y_base["econ_logits"]).mean().item(),
                "social_signed": (y["social_logits"] - y_base["social_logits"]).mean().item(),
                "intensity_signed": (y["intensity_logits"] - y_base["intensity_logits"]).mean().item(),
                "econ_rel": rel_diff(y["econ_logits"], y_base["econ_logits"]),
                "social_rel": rel_diff(y["social_logits"], y_base["social_logits"]),
                "intensity_rel": rel_diff(y["intensity_logits"], y_base["intensity_logits"]),
            }

            results.append(entry)

    df = pd.Dataframe(results)

    os.makedirs('../Graphs', exist_ok=True)
    f = df.plot(x="scale", y=["econ_diff", "social_diff", "intensity_diff"])
    fig = f.get_figure()
    fig.savefig("Evidence Ablation.jpeg")

    f = df.plot(x="scale", y=["econ_signed", "social_signed", "intensity_signed"])
    fig = f.get_figure()
    fig.savefig("Evidence Ablation (Signed).jpeg")

    f = df.plot(x="scale", y=["econ_rel", "social_rel", "intensity_rel"])
    fig = f.get_figure()
    fig.savefig("Evidence Ablation (Relative).jpeg")

def graph_psych_ablation(model_cls, enc):
    torch.manual_seed(42)
    model = model_cls(psych=True)
    model.eval()

    results = []

    with torch.no_grad():
        # baseline (scale = 0)
        model.psych_scale.data.fill_(0.0)
        y_base = model(**enc)

        for s in torch.linspace(0, 1, steps=11):
            model.psych_scale.data.fill_(s.item())
            y = model(**enc)

            entry = {
                "scale": s.item(),
                "z_diff": (y["z"] - y_base["z"]).abs().mean().item(),
                "econ_diff": (y["econ_logits"] - y_base["econ_logits"]).abs().mean().item(),
                "social_diff": (y["social_logits"] - y_base["social_logits"]).abs().mean().item(),
                "intensity_diff": (y["intensity_logits"] - y_base["intensity_logits"]).abs().mean().item(),
                "econ_signed": (y["econ_logits"] - y_base["econ_logits"]).mean().item(),
                "social_signed": (y["social_logits"] - y_base["social_logits"]).mean().item(),
                "intensity_signed": (y["intensity_logits"] - y_base["intensity_logits"]).mean().item(),
                "econ_rel": rel_diff(y["econ_logits"], y_base["econ_logits"]),
                "social_rel": rel_diff(y["social_logits"], y_base["social_logits"]),
                "intensity_rel": rel_diff(y["intensity_logits"], y_base["intensity_logits"]),
            }

            results.append(entry)

    df = pd.Dataframe(results)

    os.makedirs('../Graphs', exist_ok=True)
    f = df.plot(x="scale", y=["econ_diff", "social_diff", "intensity_diff"])
    fig = f.get_figure()
    fig.savefig("Psychology Ablation.jpeg")

    f = df.plot(x="scale", y=["econ_signed", "social_signed", "intensity_signed"])
    fig = f.get_figure()
    fig.savefig("Psychology Ablation (Signed).jpeg")

    f = df.plot(x="scale", y=["econ_rel", "social_rel", "intensity_rel"])
    fig = f.get_figure()
    fig.savefig("Psychology Ablation (Relative).jpeg")

if __name__ == "__main__":
    model = B6HierarchyDraft()

    tokenizer = AutoTokenizer.from_pretrained("roberta-base")

    # batch = {
    #     "input_ids": torch.ones((2, 16), dtype=torch.long),
    #     "attention_mask": torch.ones((2, 16), dtype=torch.long),
    # }
    texts = [
        "The government should raise taxes on corporations.",
        "I think social programs are too expensive and inefficient."
    ]
    enc = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
    evidence_results = run_evidence_ablation(B6HierarchyDraft, enc, tokenizer)
    psych_results = run_psych_ablation(B6HierarchyDraft, enc, tokenizer)