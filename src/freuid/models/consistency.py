"""Thin adapter: re-exports ConsistencyNet and build_consistency_model from
freuid.consistency_model so the existing models/__init__.py import path is unchanged.
"""

from freuid.consistency_model import ConsistencyNet, build_consistency_model  # noqa: F401

__all__ = ["ConsistencyNet", "build_consistency_model"]
