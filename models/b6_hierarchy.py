"""B6 hierarchical ideology model."""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, RobertaConfig, RobertaModel


class B6HierarchyModel(nn.Module):
    def __init__(
        self,
        encoder_name: str = "roberta-base",
        load_pretrained: bool = True,
        vocab_size: int = 30522,
        num_targets: int = 7,
        num_stances: int = 3,
        num_frames: int = 9,
        latent_dim: int = 128,
    ) -> None:
        super().__init__()

        if load_pretrained:
            self.encoder = AutoModel.from_pretrained(encoder_name)
        else:
            cfg = RobertaConfig(
                vocab_size=vocab_size,
                hidden_size=256,
                num_hidden_layers=4,
                num_attention_heads=4,
                intermediate_size=1024,
                max_position_embeddings=514,
            )
            self.encoder = RobertaModel(cfg)
        hidden = self.encoder.config.hidden_size

        # Level A heads.
        self.relevance_head = nn.Linear(hidden, 2)
        self.target_head = nn.Linear(hidden, num_targets)
        self.stance_head = nn.Linear(hidden, num_stances)
        self.frame_head = nn.Linear(hidden, num_frames)

        # Level A intermediate representations for ideology projection.
        self.target_repr = nn.Linear(num_targets, latent_dim)
        self.stance_repr = nn.Linear(num_stances, latent_dim)
        self.frame_repr = nn.Linear(num_frames, latent_dim)

        # Latent ideology projection z(x).
        self.ideology_projection = nn.Sequential(
            nn.Linear(hidden + 3 * latent_dim, latent_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(latent_dim, latent_dim),
            nn.GELU(),
        )

        # Level B heads.
        self.econ_head = nn.Linear(latent_dim, 3)
        self.social_head = nn.Linear(latent_dim, 3)
        self.intensity_head = nn.Linear(latent_dim, 3)
        self.ambiguity_head = nn.Linear(latent_dim, 3)

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

    def compute_loss(self, outputs: Dict[str, torch.Tensor], labels: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        loss_dict: Dict[str, torch.Tensor] = {}

        def kl_loss(logits: torch.Tensor, target_probs: torch.Tensor) -> torch.Tensor:
            log_probs = F.log_softmax(logits, dim=-1)
            return F.kl_div(log_probs, target_probs, reduction="batchmean")

        loss_econ = kl_loss(outputs["econ_logits"], labels["econ_dist"])
        loss_social = kl_loss(outputs["social_logits"], labels["social_dist"])
        loss_intensity = kl_loss(outputs["intensity_logits"], labels["intensity_dist"])

        p = labels["econ_dist"]
        entropy = -(p * torch.log(p + 1e-8)).sum(dim=-1)
        bins = torch.bucketize(entropy, torch.tensor([0.5, 1.0], device=entropy.device))
        ambiguity_target = F.one_hot(bins, num_classes=3).float()
        loss_amb = kl_loss(outputs["ambiguity_logits"], ambiguity_target)

        total_loss = (
            loss_econ
            + loss_social
            + loss_intensity
            + 0.5 * loss_amb
            + 0.1 * outputs["orthogonality_loss"]
        )

        loss_dict.update(
            {
                "loss": total_loss,
                "econ": loss_econ,
                "social": loss_social,
                "intensity": loss_intensity,
                "ambiguity": loss_amb,
            }
        )
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

        # Level A predictions.
        relevance_logits = self.relevance_head(pooled)
        target_logits = self.target_head(pooled)
        stance_logits = self.stance_head(pooled)
        frame_logits = self.frame_head(pooled)

        # Level A to intermediate representations.
        target_probs = F.softmax(target_logits, dim=-1)
        stance_probs = F.softmax(stance_logits, dim=-1)
        frame_probs = F.softmax(frame_logits, dim=-1)

        target_h = self.target_repr(target_probs)
        stance_h = self.stance_repr(stance_probs)
        frame_h = self.frame_repr(frame_probs)

        # Ideology projection from pooled encoder + discourse intermediates.
        z_input = torch.cat([pooled, target_h, stance_h, frame_h], dim=-1)
        z = self.ideology_projection(z_input)

        # Level B predictions from z.
        econ_logits = self.econ_head(z)
        social_logits = self.social_head(z)
        intensity_logits = self.intensity_head(z)
        ambiguity_logits = self.ambiguity_head(z)

        out: Dict[str, torch.Tensor] = {
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

        if return_intermediates:
            out["target_h"] = target_h
            out["stance_h"] = stance_h
            out["frame_h"] = frame_h

        if labels is not None:
            out.update(self.compute_loss(out, labels))

        return out


# Backwards-compatible alias for older code paths.
B6HierarchyDraft = B6HierarchyModel


if __name__ == "__main__":
    model = B6HierarchyModel(load_pretrained=False)
    batch = {
        "input_ids": torch.ones((2, 16), dtype=torch.long),
        "attention_mask": torch.ones((2, 16), dtype=torch.long),
    }
    y = model(**batch)
    print("Forward pass keys:", sorted(y.keys()))
