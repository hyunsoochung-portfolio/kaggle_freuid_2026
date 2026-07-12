"""Train-time integration of offline photo-substitution rows (see freuid.photosub.pipeline):
mode-weighted + split-discipline row selection, a dataset that appends generated rows to the
normal train split, and a batch sampler that revives "twin-pair batching" in its correct form.

Twin-pair batching (see CLAUDE.md / photosub_v0.yaml): the offline generator records each
generated row's ``source_id`` -- the bona-fide TRAIN image it was derived from. That source
image is itself an ordinary row already in this same train split, so pairing a tampered row
with its own clean source is a *dataset/collate* concern, not a new augmentation: when a batch
happens to include a photosub-positive, ``TwinPairBatchSampler`` tries (with probability
``twin_pair_prob``) to also include its source image in the SAME batch. The two differ ONLY in
the substitution evidence, which is exactly what makes them useful for the pair-hinge loss term
(freuid.loss.pair_hinge_loss) -- a discriminative signal a same-batch-but-unrelated bona-fide
comparison can't offer.

Split-discipline invariant (pipeline.py's own docstring): a generated row's ``source_id`` MUST
resolve to a sample on the SAME (train) side as the row itself, or the synthetic row leaks
validation-set appearance into training. ``select_mixed_rows`` enforces this by filtering to
``source_id in train_ids`` before anything else; ``PhotosubTwinDataset`` re-asserts it (fails
loudly) against the actual base dataset it's given, since a caller could otherwise pass a
mismatched (rows, base) pair without the filtering step running first.

photosub_v1 additions:
  - ``select_mixed_rows``'s ``per_template_cap`` bounds any single document template's share of
    each mode's selection (see ``_select_with_template_cap``) -- fixes the disproportionate
    per-template concentration movement_census_part2 found in MODE_D specifically.
  - Twin pairing is now per-MODE, not global: ``is_pairable_mode``/``PhotosubTwinDataset.
    is_pairable``/``TwinPairBatchSampler.is_pairable`` restrict twin-seeking to A/B rows only
    (C/D rows still train normally, just never seek a twin or feed the pair-hinge loss) -- see
    ``_PAIRABLE_MODE_BUCKETS``'s comment for why C/D pairing is suspected of encouraging the
    model to fixate on generation-process pixel artifacts rather than real evidence.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, Sampler

# MODE_D's two row-level tags roll up to a single "D" bucket for mode-weighting purposes --
# MODE_D is off by default at generation time (see generators.generate_mode_d), so in practice
# this bucket is usually empty; kept here so a future run that does enable it doesn't need new
# mixing code, only a nonzero "D" weight.
_MODE_BUCKET = {"D_main": "D", "D_ghost": "D"}

# photosub_v1: twin pairing is ON for A/B, OFF for C/D. MODE_C/D's swap is a color- AND
# degradation-matched digital composite with no local statistical anomaly by design (see
# generators.py's module docstring) -- its bona-fide twin is therefore near-pixel-identical to
# the tampered row except for the exact substitution the pair-hinge loss is meant to teach.
# photosub_v0's postmortem flagged this as the standing suspect for encouraging the model to
# fixate on tiny, non-generalizable pixel artifacts from the generation process itself rather
# than real cross-region semantic evidence, rather than a stable, generalizable "twin"
# discriminative signal the way an A/B physical-paste pair (rim, shadow, style mismatch --
# genuinely different local statistics) offers. C/D rows still train normally via BCE +
# auc_loss; they simply never seek a twin in-batch nor contribute to pair_hinge_loss.
_PAIRABLE_MODE_BUCKETS = {"A", "B"}


def mode_bucket(mode: str) -> str:
    return _MODE_BUCKET.get(mode, mode)


def is_pairable_mode(mode: str) -> bool:
    """True for A/B (twin pairing ON), False for C/D (twin pairing OFF) -- see
    _PAIRABLE_MODE_BUCKETS' docstring comment above for why."""
    return mode_bucket(mode) in _PAIRABLE_MODE_BUCKETS


def load_photosub_rows(csv_path: str | Path) -> pd.DataFrame:
    """Loads a photosub generated-rows CSV (freuid.photosub.pipeline.append_rows_csv's
    schema). Empty (but correctly-columned) DataFrame if the file doesn't exist yet -- lets
    callers treat "no photosub data generated yet" as "zero rows available" rather than a
    separate branch."""
    cols = ["id", "image_path", "label", "is_digital", "type", "source_id", "mode", "mask_path", "params"]
    path = Path(csv_path)
    if not path.exists():
        return pd.DataFrame(columns=cols)
    return pd.read_csv(path, dtype={"id": str, "source_id": str})


def _select_with_template_cap(
    pool: pd.DataFrame, n_target: int, cap: float, rng: np.random.Generator, mode: str,
) -> pd.DataFrame:
    """Selects up to ``n_target`` rows from ``pool``, with no single ``type`` (document
    template) contributing more than ``ceil(n_target * cap)`` rows -- movement_census_part2's
    finding that MODE_D's ghost-mismatch generator concentrated 2.81x on MAURITIUS/ID vs 1.82x
    on EGYPT/DL (relative to their real-fraud share) is exactly the imbalance this caps. Falls
    back to filling remaining slots OVER the cap (with a loud warning) only if respecting the
    cap strictly would under-fill ``n_target`` -- a cap is a fairness ceiling, not license to
    silently ship fewer rows than the mode weight asked for.
    """
    max_per_template = int(np.ceil(n_target * cap))
    templates = list(pool["type"].dropna().unique())
    rng.shuffle(templates)  # no template systematically favored by iteration order

    chosen_parts: list[pd.DataFrame] = []
    remaining = n_target
    for t in templates:
        if remaining <= 0:
            break
        t_pool = pool[pool["type"] == t]
        n_take = min(max_per_template, len(t_pool), remaining)
        if n_take <= 0:
            continue
        idx = rng.choice(len(t_pool), size=n_take, replace=False)
        chosen_parts.append(t_pool.iloc[idx])
        remaining -= n_take

    if remaining > 0:
        chosen_idx = pd.concat(chosen_parts).index if chosen_parts else pd.Index([])
        leftover = pool.drop(index=chosen_idx, errors="ignore")
        if len(leftover) > 0:
            n_fill = min(remaining, len(leftover))
            print(
                f"[photosub_mix] WARNING: mode {mode!r} per_template_cap={cap} left {remaining} "
                f"slots unfilled after respecting the cap -- filling {n_fill} over-cap from "
                "remaining templates rather than under-shooting the mode target"
            )
            idx = rng.choice(len(leftover), size=n_fill, replace=False)
            chosen_parts.append(leftover.iloc[idx])

    return pd.concat(chosen_parts) if chosen_parts else pool.iloc[0:0]


def select_mixed_rows(
    rows_df: pd.DataFrame,
    train_ids: set[str],
    mode_weights: dict[str, float],
    share: float,
    base_n_positive: int,
    seed: int = 0,
    per_template_cap: float | None = None,
) -> pd.DataFrame:
    """Selects the photosub rows to mix into this run's train split.

    ``share`` is the target fraction of TOTAL positives (original fraud + photosub) that should
    be photosub-derived, i.e. ``n_target = base_n_positive * share / (1 - share)``. Rows are
    drawn per ``mode_weights`` (matching the dossier's confirmed A/B/C prevalence among the 9
    deep-miss ids -- see configs/photosub_v0.yaml's docstring for the exact numbers and the
    caveat that the 59 boundary ids carry no per-id mode label, so this weighting is derived
    from n=9, not n=9+59), not uniformly across modes.

    ``per_template_cap`` (photosub_v1, None = no cap = byte-identical to photosub_v0's
    selection): fraction of EACH mode's own selected count that any single document template
    may contribute -- see ``_select_with_template_cap``'s docstring for the imbalance this
    fixes. Applied independently per mode (not across the whole selection), since the modes
    that actually need it (C/D, ghost-restricted to 2 templates) are structurally different
    from A/B (all 5 templates eligible).

    Split-discipline: filters to ``source_id in train_ids`` FIRST, before any sampling, so a row
    whose source fell on the val side of THIS run's split is never eligible regardless of mode
    weight or availability elsewhere.
    """
    if not (0.0 <= share < 1.0):
        raise ValueError(f"share must be in [0, 1), got {share}")
    eligible = rows_df[rows_df["source_id"].isin(train_ids)]
    n_target = int(round(base_n_positive * share / (1.0 - share))) if share > 0.0 else 0
    if n_target <= 0 or eligible.empty:
        return eligible.iloc[0:0]

    weights = {k: v for k, v in mode_weights.items() if v > 0.0}
    total_w = sum(weights.values())
    if total_w <= 0.0:
        raise ValueError(f"mode_weights has no positive weight: {mode_weights!r}")

    rng = np.random.default_rng(seed)
    bucket_col = eligible["mode"].map(mode_bucket)
    selected = []
    for mode, w in weights.items():
        n_mode_target = int(round(n_target * w / total_w))
        if n_mode_target <= 0:
            continue
        pool = eligible[bucket_col == mode]
        if pool.empty:
            print(f"[photosub_mix] WARNING: mode {mode!r} has 0 eligible rows (wanted {n_mode_target})")
            continue
        n_take = min(n_mode_target, len(pool))
        if n_take < n_mode_target:
            print(
                f"[photosub_mix] WARNING: mode {mode!r} wanted {n_mode_target}, only {len(pool)} "
                f"eligible -- taking all {n_take}"
            )
        if per_template_cap is not None:
            selected.append(_select_with_template_cap(pool, n_take, per_template_cap, rng, mode))
        else:
            idx = rng.choice(len(pool), size=n_take, replace=False)
            selected.append(pool.iloc[idx])
    if not selected:
        return eligible.iloc[0:0]
    return pd.concat(selected, ignore_index=True)


class PhotosubTwinDataset(Dataset):
    """Wraps a plain ``FreuidDataset`` train split (model_type=baseline; no face_meta/
    face_crop -- photosub_v0 doesn't use those paths) and appends selected photosub rows.

    Returns ``(img, label, pair_group_id)``: a bare int, not a dict, so this stays orthogonal to
    freuid.data's face_meta/face_crop dict convention (those are extra MODEL inputs; a pair
    group id is loss-side bookkeeping, never passed to the model -- see train.py's
    photosub-aware run_epoch).

    ``pair_group_id`` is the BASE dataset index of the bona-fide source image: a plain base
    item's group id is its own index (globally unique, so it never collides with anything
    unless it IS some photosub row's source); a photosub row's group id is its source's base
    index. Two items in the same batch sharing a group id are therefore exactly a (tampered,
    clean-source) twin pair, with no separate sentinel needed.
    """

    def __init__(self, base: Dataset, photosub_rows: pd.DataFrame, transform) -> None:
        self.base = base
        self.transform = transform
        self.rows = photosub_rows.reset_index(drop=True)
        # `base` may be a plain FreuidDataset (has `.samples`) or a SynthTamperWrapper around
        # one (only `.base.samples`, same index space as the wrapper itself) -- finetune_v0's
        # recipe keeps synth_tamper_prob > 0, so photosub_v0 must accept either.
        samples = base.samples if hasattr(base, "samples") else base.base.samples  # type: ignore[attr-defined]
        id_to_base_idx = {s.id: i for i, s in enumerate(samples)}
        self._pair_base_idx: list[int] = []
        for source_id in self.rows["source_id"]:
            base_idx = id_to_base_idx.get(source_id)
            if base_idx is None:
                raise ValueError(
                    f"photosub row source_id {source_id!r} not found in the base train dataset "
                    "-- split-discipline violation (select_mixed_rows should have filtered this "
                    "out; check that the same train_ids set was used for both)"
                )
            self._pair_base_idx.append(base_idx)

    @property
    def n_base(self) -> int:
        return len(self.base)  # type: ignore[arg-type]

    def __len__(self) -> int:
        return self.n_base + len(self.rows)

    def is_photosub(self, idx: int) -> bool:
        return idx >= self.n_base

    def is_pairable(self, idx: int) -> bool:
        """True iff this is a photosub row whose mode is twin-pairable (A/B) -- see
        ``is_pairable_mode``'s docstring for the C/D exclusion rationale. False for any base
        (non-photosub) index; the sampler only ever consults this for photosub indices anyway,
        but a well-defined answer either way keeps this method total."""
        if idx < self.n_base:
            return False
        return is_pairable_mode(str(self.rows.iloc[idx - self.n_base]["mode"]))

    def pair_group_id(self, idx: int) -> int:
        if idx < self.n_base:
            return idx
        return self._pair_base_idx[idx - self.n_base]

    def __getitem__(self, idx: int):
        if idx < self.n_base:
            img, label = self.base[idx]  # type: ignore[index]
            return img, label, idx
        row = self.rows.iloc[idx - self.n_base]
        img = Image.open(row["image_path"]).convert("RGB")
        img = self.transform(img)
        return img, int(row["label"]), self._pair_base_idx[idx - self.n_base]


class TwinPairBatchSampler(Sampler):
    """Yields index batches where each photosub item, with probability ``twin_pair_prob``, is
    joined in the SAME batch by its own bona-fide source item (displacing a random other,
    non-forced item to keep batch size fixed -- the displaced item simply isn't seen this
    batch, same kind of coverage loss ``drop_last`` already accepts).

    This is a probabilistic UPPER BOUND on the per-item pairing rate, not an exact one: if the
    slot chosen to make room already IS another photosub item's forced twin, that other pairing
    is silently given up for this batch (rare unless photosub rows are a large share of the
    dataset). Verified by test to land close to ``twin_pair_prob`` in aggregate, not exactly.
    """

    def __init__(
        self,
        dataset: PhotosubTwinDataset,
        batch_size: int,
        twin_pair_prob: float,
        seed: int = 0,
        drop_last: bool = True,
    ) -> None:
        self.n = len(dataset)
        self.pair_group_id = np.array([dataset.pair_group_id(i) for i in range(self.n)], dtype=np.int64)
        self.is_photosub = np.array([dataset.is_photosub(i) for i in range(self.n)], dtype=bool)
        # photosub_v1: only A/B-mode photosub items ever seek a twin -- C/D rows still train
        # normally (BCE + auc_loss) but never trigger the batch-rearrangement below and never
        # feed pair_hinge_loss (see is_pairable_mode's docstring for why).
        self.is_pairable = np.array([dataset.is_pairable(i) for i in range(self.n)], dtype=bool)
        self.batch_size = batch_size
        self.twin_pair_prob = twin_pair_prob
        self.seed = seed
        self.drop_last = drop_last
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        if self.drop_last:
            return self.n // self.batch_size
        return -(-self.n // self.batch_size)

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        perm = rng.permutation(self.n)
        n_batches = len(self)
        for b in range(n_batches):
            batch = list(perm[b * self.batch_size : (b + 1) * self.batch_size])
            if not batch:
                continue
            batch_set = set(int(x) for x in batch)
            for pos in range(len(batch)):
                idx = int(batch[pos])
                if not self.is_photosub[idx]:
                    continue
                if not self.is_pairable[idx]:
                    continue
                if rng.random() >= self.twin_pair_prob:
                    continue
                twin_idx = int(self.pair_group_id[idx])
                if twin_idx in batch_set:
                    continue
                candidates = [j for j in range(len(batch)) if j != pos]
                non_forced = [j for j in candidates if not self.is_photosub[batch[j]]]
                replace_pos = int(rng.choice(non_forced)) if non_forced else int(rng.choice(candidates))
                batch_set.discard(int(batch[replace_pos]))
                batch[replace_pos] = twin_idx
                batch_set.add(twin_idx)
            yield [int(x) for x in batch]


def count_pairs_in_batch(labels, pair_ids) -> int:
    """Number of pair_ids groups in this batch with both a label==1 and a label==0 member --
    i.e. how many twin pairs the batch sampler actually landed together. Pure logging/
    verification helper (used by train.py's smoke-run output), not part of the loss."""
    import torch

    n = 0
    for g in torch.unique(pair_ids).tolist():
        mask = pair_ids == g
        if int(mask.sum()) < 2:
            continue
        group_labels = labels[mask]
        if bool((group_labels == 1).any()) and bool((group_labels == 0).any()):
            n += 1
    return n


def unpack_photosub_batch(batch, device):
    """(imgs, labels, pair_group_id) -> all moved to ``device``. Separate from
    freuid.data.unpack_and_move on purpose: pair_group_id is loss-side bookkeeping, never a
    model input, so it must never reach forward_with_extras."""
    imgs, labels, pair_ids = batch
    return imgs.to(device), labels.to(device), pair_ids.to(device)
