"""Thin adapter: re-exports ConsistencyNet and build_consistency_model from
freuid.consistency_model so the existing models/__init__.py import path is unchanged.
"""

from freuid.consistency_model import (  # noqa: F401
    ConsistencyHead,
    ConsistencyNet,
    FaceRegionHead,
    FusionMLP,
    PatchConsistencyHead,
    build_consistency_model,
)

__all__ = [
    "ConsistencyHead",
    "ConsistencyNet",
    "FaceRegionHead",
    "FusionMLP",
    "PatchConsistencyHead",
    "build_consistency_model",
]
