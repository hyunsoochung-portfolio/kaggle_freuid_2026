"""Loss functions beyond plain BCE.

All functions follow the sign convention: lower = better (same as BCE).
"""

from __future__ import annotations

import torch
import torch.nn as nn


def pairwise_soft_auc(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Pairwise sigmoid surrogate for 1 - AUC (lower = better ranking).

    For every (fraud, bona-fide) pair in the batch, applies a sigmoid to
    (score_bona - score_fraud) — the probability that the pair is mis-ranked.
    Minimising this term pushes fraud scores above bona-fide scores, directly
    optimising score *ordering* rather than calibration.

    Returns 0.0 (no gradient) when the batch has no cross-class pairs.
    """
    scores = torch.sigmoid(logits.squeeze(1))
    pos = scores[labels == 1]   # fraud
    neg = scores[labels == 0]   # bona-fide
    if pos.numel() == 0 or neg.numel() == 0:
        return torch.zeros(1, device=logits.device).squeeze()
    # diff[i,j] = score_fraud_i - score_bona_j; want this > 0 for every pair
    diff = pos.unsqueeze(1) - neg.unsqueeze(0)   # [P, N]
    return torch.sigmoid(-diff).mean()           # 0 = perfect ranking


def combined_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    bce: nn.BCEWithLogitsLoss,
    auc_weight: float = 0.0,
) -> torch.Tensor:
    """BCE + optional pairwise soft-AUC term.

    auc_weight=0.0 is bit-for-bit identical to plain BCE (no extra computation).
    labels must be integer (0/1) on the same device as logits.
    """
    targets = labels.float().unsqueeze(1)
    loss = bce(logits, targets)
    if auc_weight > 0.0:
        loss = loss + auc_weight * pairwise_soft_auc(logits, labels)
    return loss
