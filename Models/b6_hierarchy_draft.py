"""Draft B6 hierarchy model.

This is a working architecture draft, not a finished training implementation.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel


class B6HierarchyDraft(nn.Module):
    def __init__(
        self,
        encoder_name: str = "roberta-base",
        num_targets: int = 7,
        num_stances: int = 3,
        num_frames: int = 9,
        latent_dim: int = 128,
    ) -> None:
        super().__init__()

        self.encoder = AutoModel.from_pretrained(encoder_name)
        hidden = self.encoder.config.hidden_size

        # Discourse-level heads.
        self.relevance_head = nn.Linear(hidden, 2)
        self.target_head = nn.Linear(hidden, num_targets)
        self.stance_head = nn.Linear(hidden, num_stances)
        self.frame_head = nn.Linear(hidden, num_frames)

        # Intermediate representations used by the ideology projection layer.
        self.target_repr = nn.Linear(num_targets, latent_dim)
        self.stance_repr = nn.Linear(num_stances, latent_dim)
        self.frame_repr = nn.Linear(num_frames, latent_dim)

        # Projection into latent ideology vector z(x).
        self.ideology_projection = nn.Sequential(
            nn.Linear(hidden + 3 * latent_dim, latent_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(latent_dim, latent_dim),
            nn.GELU(),
        )

        # Ideology-level heads.
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

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        return_intermediates: bool = True,
        labels: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state[:, 0]

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
        z_input = torch.cat([pooled, target_h, stance_h, frame_h], dim=-1)
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
        }

        if return_intermediates:
            out["target_h"] = target_h
            out["stance_h"] = stance_h
            out["frame_h"] = frame_h

        return out


if __name__ == "__main__":
    model = B6HierarchyDraft()
    batch = {
        "input_ids": torch.ones((2, 16), dtype=torch.long),
        "attention_mask": torch.ones((2, 16), dtype=torch.long),
    }
    y = model(**batch)
    print("Forward pass keys:", sorted(y.keys()))
