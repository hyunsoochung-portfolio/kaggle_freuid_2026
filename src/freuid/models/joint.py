"""Joint attention-pool + patch-consistency model (model_type="joint").

Built ON TOP of the winning full-fine-tuned attention-pool baseline (build_model with
pool="map", public 0.03935), this adds a second, parallel branch that reads the SAME
backbone patch tokens and looks for local inconsistency (splice/tamper) anywhere on the
document -- then fuses the two logits:

    final_logit = logit_global  +  w_consist * logit_consist

Both branches consume the backbone's patch tokens (BEFORE pooling), in parallel:
  - global:  timm's attention-pool head (learned query -> summary -> logit)   [the 0.039 model]
  - consist: PatchConsistencyHead (learned [outlier] query + small TransformerEncoder over
             the patch tokens) -> Linear -> logit

Safe warm start: consist_fc is zero-init AND w_consist starts at 0, so at init the model
is *bit-for-bit* the attention-pool baseline (init BCE ~= ln2). The consistency branch only
"opens up" as training finds it useful, so it can help but never corrupts the known-good
baseline at step 1.

The whole backbone is fully fine-tuned (unlike the frozen ConsistencyNet in
consistency_model.py, which scored a much worse 0.307) -- full FT is what wins on this task.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from freuid.config import Config
from freuid.consistency_model import PatchConsistencyHead
from freuid.models.baseline import build_model


class JointConsistencyModel(nn.Module):
    def __init__(
        self,
        backbone: str,
        pretrained: bool = True,
        head_dropout: float = 0.0,
        patch_layers: int = 2,
        patch_heads: int = 8,
        patch_dropout: float = 0.1,
    ) -> None:
        super().__init__()
        # global branch = the winning attention-pool model (num_classes=1, zero-init head)
        self.net = build_model(backbone, pretrained, pool="map", head_dropout=head_dropout)
        dim = self.net.num_features
        self.num_prefix = int(getattr(self.net, "num_prefix_tokens", 1))

        # consistency branch on the patch tokens (prefix excluded)
        self.patch_head = PatchConsistencyHead(
            dim, num_layers=patch_layers, num_heads=patch_heads, dropout=patch_dropout
        )
        # Zero-init so logit_consist == 0 at init (model starts == the attention-pool
        # baseline, init BCE ~= ln2). Additive fusion (NOT logit_g + w*logit_c with w=0
        # too) so consist_fc still receives gradient from step 1 -- a learnable scale
        # multiplied by a zero-init logit would leave the whole branch dead (both factors
        # zero => zero grad to each). The branch grows only as far as it lowers the loss.
        self.consist_fc = nn.Linear(dim, 1)
        nn.init.zeros_(self.consist_fc.weight)
        nn.init.zeros_(self.consist_fc.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.net.forward_features(x)              # (B, N_all, D)
        logit_global = self.net.forward_head(tokens)       # (B, 1)  attn-pool + head
        patch = tokens[:, self.num_prefix:, :]             # (B, N, D) patch tokens only
        logit_consist = self.consist_fc(self.patch_head(patch))  # (B, 1)
        return logit_global + logit_consist


def build_joint_model(cfg: Config) -> JointConsistencyModel:
    model = JointConsistencyModel(
        cfg.backbone,
        cfg.pretrained,
        head_dropout=float(cfg.extra.get("head_dropout", 0.0)),
        patch_layers=int(cfg.extra.get("patch_consistency_layers", 2)),
        patch_heads=int(cfg.extra.get("patch_consistency_heads", 8)),
        patch_dropout=float(cfg.extra.get("patch_consistency_dropout", 0.1)),
    )
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(
        f"[joint] {cfg.backbone}: attn-pool(global) + PatchConsistency(patch), "
        f"w_consist init=0 -> starts == attn-pool baseline | trainable={n_train:,}"
    )
    return model
