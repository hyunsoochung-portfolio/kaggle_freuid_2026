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


def pair_hinge_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    pair_ids: torch.Tensor,
    margin: float = 1.0,
) -> torch.Tensor:
    """Twin-pair hinge: for every ``pair_ids`` group present in this batch with BOTH a
    label==1 member (a photosub-tampered row) and a label==0 member (its own bona-fide
    source image -- see freuid.photosub.mixing.PhotosubTwinDataset), penalizes
    ``relu(margin - (tampered_logit - clean_logit))``. The pair differs ONLY in the
    substitution evidence (same source photo, before/after), so this term pushes directly on
    the feature the model is failing to use -- unlike ``pairwise_soft_auc``, which ranks
    across the whole batch regardless of provenance.

    Returns 0.0 (no gradient) when no group in this batch has both labels present -- expected
    whenever the batch sampler's ``twin_pair_prob`` didn't happen to land a pair together, or
    when photosub mixing is off entirely (every item then carries a distinct pair id).
    """
    scores = logits.squeeze(1)
    device = scores.device
    losses = []
    for g in torch.unique(pair_ids).tolist():
        mask = pair_ids == g
        if int(mask.sum()) < 2:
            continue
        group_labels = labels[mask]
        if not (bool((group_labels == 1).any()) and bool((group_labels == 0).any())):
            continue
        pos_logit = scores[mask][group_labels == 1].mean()
        neg_logit = scores[mask][group_labels == 0].mean()
        losses.append(torch.relu(torch.as_tensor(margin, device=device, dtype=scores.dtype) - (pos_logit - neg_logit)))
    if not losses:
        return torch.zeros((), device=device, dtype=scores.dtype)
    return torch.stack(losses).mean()


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
