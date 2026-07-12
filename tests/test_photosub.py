"""Tests for the photo-substitution training-data generators (freuid.photosub.*).

All synthetic: tiny in-memory PIL images, no real dataset / regions cache / GPU needed. Covers
the exact checklist from the build request: mask correctness, determinism under seed,
frame-box exact alignment for MODE_C, overhang-exceeds-frame-area for MODE_B, the donor
probe-exclusion rule, and split-discipline bookkeeping (source_id on generated rows).
"""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from freuid.photosub.baseline_stats import build_natural_baseline, compute_p90_thresholds, natural_pair_delta
from freuid.photosub.degradation_match import match_blur, match_degradation, match_jpeg_grain, match_moire
from freuid.photosub.donor_pool import (
    DonorFace,
    build_donor_record,
    cosine_similarity,
    exclude_probe_overlap,
    sample_donor,
)
from freuid.photosub.donor_pool import _margined_crop
from freuid.photosub.face_embedding import arcface_available, best_available_embed_fn
from freuid.photosub.generators import generate_mode_a, generate_mode_b, generate_mode_c, generate_mode_d
from freuid.photosub.generators import _irregular_patch_alpha
from freuid.photosub.pipeline import save_tamper_result
from freuid.photosub.stats import laplacian_blur_score, region_stats
from freuid.photosub.template_regions import build_template_metadata, frame_box_from_face, ghost_box_px


def _solid_image(w: int, h: int, color=(120, 130, 140)) -> Image.Image:
    return Image.new("RGB", (w, h), color)


def _checker_image(w: int, h: int, block: int = 10) -> Image.Image:
    """An image with a background pattern crossing the whole canvas -- lets a test confirm a
    paste actually severs/occludes whatever was underneath (per MODE_A's spec)."""
    arr = np.zeros((h, w, 3), dtype=np.uint8)
    for y in range(h):
        for x in range(w):
            arr[y, x] = (255, 255, 255) if ((x // block) + (y // block)) % 2 == 0 else (30, 30, 30)
    return Image.fromarray(arr)


def _face_box(x1, y1, x2, y2, score=0.9) -> dict:
    return {"x1": x1, "y1": y1, "x2": x2, "y2": y2, "score": score}


CANVAS_W, CANVAS_H = 400, 300
FRAME_BOX = {"x1": 150, "y1": 100, "x2": 250, "y2": 200}  # comfortably inside the canvas
GHOST_BOX = {"x1": 20, "y1": 20, "x2": 60, "y2": 60}


def _donor_crop() -> Image.Image:
    return _solid_image(80, 100, color=(200, 150, 100))


def _noisy_image(w: int, h: int, seed: int = 0) -> Image.Image:
    """High-frequency random noise -- a sharp, high-Laplacian-variance, high-blockiness-free
    image, the opposite extreme from a solid color patch. Used to exercise degradation_match's
    "donor is sharper than the target" branch."""
    rng = np.random.default_rng(seed)
    arr = rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)
    return Image.fromarray(arr)


def _mild_texture_image(w: int, h: int, seed: int = 10) -> Image.Image:
    """Mild sinusoidal texture + a little noise -- low-but-NONZERO Laplacian variance. A
    perfectly flat solid color has an unreachable exact-zero blur score (no finite blur radius
    gets there), which isn't a realistic "smooth target" for match_blur tests -- real card
    regions always carry some texture."""
    rng = np.random.default_rng(seed)
    xx, yy = np.meshgrid(np.arange(w), np.arange(h))
    base = 150 + 8 * np.sin(xx / 6.0) * np.cos(yy / 6.0)
    arr = np.clip(base + rng.normal(0, 2.0, size=(h, w)), 0, 255).astype(np.uint8)
    return Image.fromarray(np.stack([arr, arr, arr], axis=-1))


# ---------------------------------------------------------------------------
# Degradation matching (freuid.photosub.degradation_match / .stats)
# ---------------------------------------------------------------------------

class TestDegradationMatch:
    def test_match_blur_softens_oversharp_donor_toward_target(self):
        sharp_donor = _noisy_image(80, 80, seed=1)
        smooth_target = _mild_texture_image(80, 80)
        out = match_blur(sharp_donor, smooth_target)
        donor_score = laplacian_blur_score(np.asarray(sharp_donor.convert("L")))
        target_score = laplacian_blur_score(np.asarray(smooth_target.convert("L")))
        out_score = laplacian_blur_score(np.asarray(out.convert("L")))
        assert out_score < donor_score
        # Bounded binary search over a finite radius range converges close to, not exactly
        # onto, the target score -- a loose multiplicative tolerance avoids test flakiness
        # while still proving real convergence (donor_score is ~2 orders of magnitude bigger).
        assert out_score <= target_score * 3 + 5

    def test_match_blur_noop_when_donor_already_blurrier(self):
        blurry_donor = _solid_image(80, 80, color=(100, 100, 100))
        sharp_target = _noisy_image(80, 80, seed=2)
        out = match_blur(blurry_donor, sharp_target)
        assert np.array_equal(np.asarray(out), np.asarray(blurry_donor))

    def test_match_moire_increases_periodicity_toward_low_target(self):
        smooth_donor = _solid_image(80, 80, color=(150, 140, 130))  # ~zero periodicity
        periodic_target = _checker_image(80, 80, block=4)  # strong periodicity
        donor_score = region_stats(smooth_donor)["moire_fft_score"]
        target_score = region_stats(periodic_target)["moire_fft_score"]
        out = match_moire(smooth_donor, periodic_target)
        out_score = region_stats(out)["moire_fft_score"]
        assert out_score > donor_score
        assert abs(out_score - target_score) <= abs(donor_score - target_score) + 1e-6

    def test_match_moire_adds_noise_to_reduce_overly_periodic_donor(self):
        """Real render-sheet data (docs/photosub_renders/stats_check_report.md) showed the
        mismatch runs in BOTH directions -- a donor can be MORE periodic than its target, not
        just less. Reducing it uses broadband noise, not blur -- blur was tried first and
        measured to make this donor's score WORSE (126 -> 656 at radius 3), since a low-pass
        filter doesn't reliably lower an FFT-annulus peak/median ratio the way it lowers
        Laplacian-variance sharpness. This must actively reduce the score, not no-op."""
        periodic_donor = _checker_image(80, 80, block=4)
        smooth_target = _mild_texture_image(80, 80)  # low but nonzero, like test_match_blur's target
        donor_score = region_stats(periodic_donor)["moire_fft_score"]
        target_score = region_stats(smooth_target)["moire_fft_score"]
        out = match_moire(periodic_donor, smooth_target)
        out_score = region_stats(out)["moire_fft_score"]
        assert out_score < donor_score
        assert abs(out_score - target_score) <= abs(donor_score - target_score) + 1e-6

    def test_match_moire_noop_when_already_equal(self):
        img = _mild_texture_image(80, 80)
        out = match_moire(img, img)
        assert np.array_equal(np.asarray(out), np.asarray(img))

    def test_match_jpeg_grain_moves_blockiness_toward_target(self):
        clean_donor = _solid_image(64, 64, color=(180, 120, 90))
        compressed_target = _noisy_image(64, 64, seed=3)
        donor_score = region_stats(clean_donor)["blockiness_score"]
        target_score = region_stats(compressed_target)["blockiness_score"]
        out = match_jpeg_grain(clean_donor, compressed_target)
        out_score = region_stats(out)["blockiness_score"]
        assert abs(out_score - target_score) <= abs(donor_score - target_score) + 1e-6

    def test_match_degradation_deterministic(self):
        donor = _noisy_image(80, 80, seed=4)
        target = _solid_image(80, 80, color=(200, 200, 200))
        out1 = match_degradation(donor, target)
        out2 = match_degradation(donor, target)
        assert np.array_equal(np.asarray(out1), np.asarray(out2))

    def test_region_stats_has_expected_keys(self):
        stats = region_stats(_solid_image(40, 40))
        assert set(stats) == {"blur_laplacian_var", "moire_fft_score", "blockiness_score"}

    def test_mode_c_composite_blur_closer_to_target_than_raw_donor(self):
        """Integration check: generate_mode_c's composited frame region should end up closer
        in blur score to the surrounding card than an unmatched donor paste would be -- the
        actual bug the render-sheet self-check caught (stats_check_report.md)."""
        image = _solid_image(CANVAS_W, CANVAS_H, color=(180, 180, 180))  # smooth "card"
        sharp_donor = _noisy_image(80, 100, seed=5)
        rng = np.random.default_rng(0)
        result = generate_mode_c(image, FRAME_BOX, sharp_donor, rng)

        fx1, fy1, fx2, fy2 = FRAME_BOX["x1"], FRAME_BOX["y1"], FRAME_BOX["x2"], FRAME_BOX["y2"]
        composited_score = laplacian_blur_score(np.asarray(result.image.crop((fx1, fy1, fx2, fy2)).convert("L")))
        raw_donor_score = laplacian_blur_score(np.asarray(sharp_donor.convert("L")))
        target_score = laplacian_blur_score(np.asarray(image.crop((fx1, fy1, fx2, fy2)).convert("L")))

        assert abs(composited_score - target_score) < abs(raw_donor_score - target_score)


# ---------------------------------------------------------------------------
# Natural intra-card baseline (freuid.photosub.baseline_stats)
# ---------------------------------------------------------------------------

class _FakeRow:
    def __init__(self, id_: str, path):
        self.id = id_
        self.path = path


class TestNaturalBaseline:
    def test_natural_pair_delta_zero_for_uniform_image(self):
        """Two random regions on a perfectly uniform image must have zero natural delta --
        establishes the floor of the natural-variation distribution."""
        img = _solid_image(200, 150, color=(100, 110, 120))
        rng = np.random.default_rng(0)
        delta = natural_pair_delta(img, box_w=40, box_h=30, rng=rng)
        assert delta["blur_laplacian_var"] == pytest.approx(0.0, abs=1e-9)
        assert delta["blockiness_score"] == pytest.approx(0.0, abs=1e-9)

    def test_build_natural_baseline_skips_rows_without_real_detection(self, tmp_path):
        regions_dir = tmp_path / "regions"
        images_dir = tmp_path / "images"
        images_dir.mkdir()

        rows = []
        for i in range(5):
            img_path = images_dir / f"img{i}.jpeg"
            _checker_image(200, 150).save(img_path)
            row_dir = regions_dir / f"id{i}"
            row_dir.mkdir(parents=True)
            score = 0.9 if i < 3 else 0.0  # last 2 are fallback (non-)detections
            (row_dir / "face.json").write_text(
                f'{{"x1": 60, "y1": 40, "x2": 120, "y2": 100, "score": {score}}}'
            )
            rows.append(_FakeRow(f"id{i}", img_path))

        baseline_df = build_natural_baseline(rows, regions_dir, n_cards=200, seed=0)
        assert len(baseline_df) == 3  # only the 3 real detections
        assert set(baseline_df.columns) >= {"blur_laplacian_var", "moire_fft_score", "blockiness_score", "id"}

    def test_compute_p90_thresholds_returns_all_stat_cols(self, tmp_path):
        regions_dir = tmp_path / "regions"
        images_dir = tmp_path / "images"
        images_dir.mkdir()
        rows = []
        for i in range(10):
            img_path = images_dir / f"img{i}.jpeg"
            _checker_image(200, 150, block=5 + i).save(img_path)
            row_dir = regions_dir / f"id{i}"
            row_dir.mkdir(parents=True)
            (row_dir / "face.json").write_text('{"x1": 60, "y1": 40, "x2": 120, "y2": 100, "score": 0.9}')
            rows.append(_FakeRow(f"id{i}", img_path))

        baseline_df = build_natural_baseline(rows, regions_dir, n_cards=200, seed=1)
        thresholds = compute_p90_thresholds(baseline_df)
        assert set(thresholds) == {"blur_laplacian_var", "moire_fft_score", "blockiness_score"}
        assert all(v >= 0 for v in thresholds.values())

    def test_build_natural_baseline_respects_n_cards_cap(self, tmp_path):
        regions_dir = tmp_path / "regions"
        images_dir = tmp_path / "images"
        images_dir.mkdir()
        rows = []
        for i in range(10):
            img_path = images_dir / f"img{i}.jpeg"
            _checker_image(200, 150).save(img_path)
            row_dir = regions_dir / f"id{i}"
            row_dir.mkdir(parents=True)
            (row_dir / "face.json").write_text('{"x1": 60, "y1": 40, "x2": 120, "y2": 100, "score": 0.9}')
            rows.append(_FakeRow(f"id{i}", img_path))

        baseline_df = build_natural_baseline(rows, regions_dir, n_cards=4, seed=0)
        assert len(baseline_df) == 4


# ---------------------------------------------------------------------------
# Mask correctness
# ---------------------------------------------------------------------------

class TestMaskCorrectness:
    def test_mode_a_mask_matches_image_size(self):
        image = _checker_image(CANVAS_W, CANVAS_H)
        rng = np.random.default_rng(0)
        result = generate_mode_a(image, FRAME_BOX, _donor_crop(), rng)
        assert result.mask.shape == (CANVAS_H, CANVAS_W)
        assert result.mask.dtype == np.uint8
        assert set(np.unique(result.mask)).issubset({0, 255})

    def test_mode_a_paste_occludes_background_pattern(self):
        """Opaque compositing must sever whatever card print falls under the paste -- verified
        over a background pattern (checkerboard) that crosses the frame, per the build spec."""
        image = _checker_image(CANVAS_W, CANVAS_H)
        rng = np.random.default_rng(1)
        result = generate_mode_a(image, FRAME_BOX, _donor_crop(), rng)
        out_arr = np.asarray(result.image)
        orig_arr = np.asarray(image)
        # Inside the mask, the output must differ from the original checkerboard (occluded).
        masked_pixels_changed = np.any(out_arr[result.mask == 255] != orig_arr[result.mask == 255])
        assert masked_pixels_changed
        assert result.mask.sum() > 0

    def test_mode_b_mask_matches_image_size(self):
        image = _solid_image(CANVAS_W, CANVAS_H)
        rng = np.random.default_rng(2)
        result = generate_mode_b(image, FRAME_BOX, _donor_crop(), rng)
        assert result.mask.shape == (CANVAS_H, CANVAS_W)

    def test_mode_c_mask_matches_image_size(self):
        image = _solid_image(CANVAS_W, CANVAS_H)
        rng = np.random.default_rng(3)
        result = generate_mode_c(image, FRAME_BOX, _donor_crop(), rng)
        assert result.mask.shape == (CANVAS_H, CANVAS_W)

    def test_mode_d_ghost_mask_confined_to_ghost_box(self):
        image = _solid_image(CANVAS_W, CANVAS_H)
        rng = np.random.default_rng(4)
        result = generate_mode_d(
            image, FRAME_BOX, GHOST_BOX, _donor_crop(), rng, swap_target="ghost", enabled=True,
        )
        ys, xs = np.where(result.mask == 255)
        assert ys.min() >= GHOST_BOX["y1"] and ys.max() < GHOST_BOX["y2"]
        assert xs.min() >= GHOST_BOX["x1"] and xs.max() < GHOST_BOX["x2"]
        # Main portrait region (frame_box) must be untouched by a ghost-only swap.
        assert result.mask[FRAME_BOX["y1"]:FRAME_BOX["y2"], FRAME_BOX["x1"]:FRAME_BOX["x2"]].sum() == 0


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------

class TestDeterminism:
    @pytest.mark.parametrize("mode_fn", [generate_mode_a, generate_mode_b, generate_mode_c])
    def test_same_seed_same_output(self, mode_fn):
        image = _checker_image(CANVAS_W, CANVAS_H)
        donor = _donor_crop()
        r1 = mode_fn(image, FRAME_BOX, donor, np.random.default_rng(42))
        r2 = mode_fn(image, FRAME_BOX, donor, np.random.default_rng(42))
        assert np.array_equal(np.asarray(r1.image), np.asarray(r2.image))
        assert np.array_equal(r1.mask, r2.mask)
        assert r1.params == r2.params

    def test_different_seed_generally_differs(self):
        image = _checker_image(CANVAS_W, CANVAS_H)
        donor = _donor_crop()
        r1 = generate_mode_a(image, FRAME_BOX, donor, np.random.default_rng(1))
        r2 = generate_mode_a(image, FRAME_BOX, donor, np.random.default_rng(2))
        assert not np.array_equal(np.asarray(r1.image), np.asarray(r2.image)) or r1.params != r2.params

    @pytest.mark.parametrize("mode_fn,kwargs", [
        (generate_mode_a, {"irregular_shape_prob": 1.0, "tear_prob": 1.0}),
        (generate_mode_b, {"irregular_shape_prob": 1.0}),
    ])
    def test_same_seed_same_output_on_irregular_shape_path(self, mode_fn, kwargs):
        image = _checker_image(CANVAS_W, CANVAS_H)
        donor = _donor_crop()
        r1 = mode_fn(image, FRAME_BOX, donor, np.random.default_rng(42), **kwargs)
        r2 = mode_fn(image, FRAME_BOX, donor, np.random.default_rng(42), **kwargs)
        assert np.array_equal(np.asarray(r1.image), np.asarray(r2.image))
        assert np.array_equal(r1.mask, r2.mask)
        assert r1.params == r2.params


# ---------------------------------------------------------------------------
# Frame-box alignment (MODE_C) / overhang (MODE_B)
# ---------------------------------------------------------------------------

class TestGeometry:
    def test_mode_c_seam_exactly_on_frame_edge(self):
        image = _solid_image(CANVAS_W, CANVAS_H)
        rng = np.random.default_rng(5)
        result = generate_mode_c(image, FRAME_BOX, _donor_crop(), rng)
        ys, xs = np.where(result.mask == 255)
        assert (xs.min(), ys.min(), xs.max() + 1, ys.max() + 1) == (
            FRAME_BOX["x1"], FRAME_BOX["y1"], FRAME_BOX["x2"], FRAME_BOX["y2"],
        )
        frame_area = (FRAME_BOX["x2"] - FRAME_BOX["x1"]) * (FRAME_BOX["y2"] - FRAME_BOX["y1"])
        assert result.mask.sum() // 255 == frame_area

    @pytest.mark.parametrize("seed", range(20))
    def test_mode_b_mask_area_exceeds_frame_area(self, seed):
        image = _solid_image(CANVAS_W, CANVAS_H)
        rng = np.random.default_rng(seed)
        result = generate_mode_b(image, FRAME_BOX, _donor_crop(), rng)
        frame_area = (FRAME_BOX["x2"] - FRAME_BOX["x1"]) * (FRAME_BOX["y2"] - FRAME_BOX["y1"])
        mask_area = int(result.mask.sum() // 255)
        assert mask_area > frame_area, f"seed={seed}: mask_area={mask_area} <= frame_area={frame_area}"


# ---------------------------------------------------------------------------
# Shape-realism fix: irregular hand-cut silhouette + tear effect (MODE_A/B)
# ---------------------------------------------------------------------------

class TestShapeRealismFix:
    def test_irregular_patch_alpha_is_deterministic(self):
        a1 = _irregular_patch_alpha(100, 120, np.random.default_rng(7))
        a2 = _irregular_patch_alpha(100, 120, np.random.default_rng(7))
        assert np.array_equal(a1, a2)

    def test_irregular_patch_alpha_is_binary_0_255(self):
        alpha = _irregular_patch_alpha(100, 120, np.random.default_rng(3))
        assert alpha.shape == (120, 100)
        assert alpha.dtype == np.uint8
        assert set(np.unique(alpha)).issubset({0, 255})

    @pytest.mark.parametrize("seed", range(10))
    def test_irregular_patch_alpha_roughly_sized(self, seed):
        box_w, box_h = 100, 120
        alpha = _irregular_patch_alpha(box_w, box_h, np.random.default_rng(seed))
        area_frac = float((alpha > 0).sum()) / (box_w * box_h)
        assert 0.4 < area_frac < 1.0, f"seed={seed}: area_frac={area_frac}"

    def test_mode_a_irregular_shape_recorded_in_params(self):
        image = _solid_image(CANVAS_W, CANVAS_H)
        result = generate_mode_a(
            image, FRAME_BOX, _donor_crop(), np.random.default_rng(9), irregular_shape_prob=1.0,
        )
        assert result.params["shape"] == "irregular"

    def test_mode_a_default_shape_is_rect(self):
        image = _solid_image(CANVAS_W, CANVAS_H)
        result = generate_mode_a(image, FRAME_BOX, _donor_crop(), np.random.default_rng(9))
        assert result.params["shape"] == "rect"
        assert result.params["tear"] is False

    @pytest.mark.parametrize("seed", range(10))
    def test_mode_a_irregular_shape_boundary_is_non_rectangular(self, seed):
        """A plain rectangle mask has area == its own bounding-box area exactly; a concave
        hand-cut silhouette must have strictly less area than its bounding box."""
        image = _solid_image(CANVAS_W, CANVAS_H)
        result = generate_mode_a(
            image, FRAME_BOX, _donor_crop(), np.random.default_rng(seed), irregular_shape_prob=1.0,
        )
        ys, xs = np.where(result.mask == 255)
        bbox_area = (xs.max() - xs.min() + 1) * (ys.max() - ys.min() + 1)
        mask_area = int(result.mask.sum() // 255)
        assert mask_area < bbox_area, f"seed={seed}: mask_area={mask_area} >= bbox_area={bbox_area}"

    def test_mode_a_tear_effect_visible(self):
        image = _solid_image(CANVAS_W, CANVAS_H)
        r_no_tear = generate_mode_a(image, FRAME_BOX, _donor_crop(), np.random.default_rng(11))
        r_tear = generate_mode_a(
            image, FRAME_BOX, _donor_crop(), np.random.default_rng(11), tear_prob=1.0,
        )
        assert r_tear.params["tear"] is True
        assert not np.array_equal(np.asarray(r_no_tear.image), np.asarray(r_tear.image))

    def test_mode_b_irregular_shape_recorded_in_params(self):
        image = _solid_image(CANVAS_W, CANVAS_H)
        result = generate_mode_b(
            image, FRAME_BOX, _donor_crop(), np.random.default_rng(13), irregular_shape_prob=1.0,
        )
        assert result.params["shape"] == "irregular"

    def test_mode_b_default_shape_is_rect(self):
        image = _solid_image(CANVAS_W, CANVAS_H)
        result = generate_mode_b(image, FRAME_BOX, _donor_crop(), np.random.default_rng(13))
        assert result.params["shape"] == "rect"

    # --- photosub_v1: arch shape + tape strips ---

    def test_mode_a_arch_shape_recorded_in_params(self):
        image = _solid_image(CANVAS_W, CANVAS_H)
        result = generate_mode_a(
            image, FRAME_BOX, _donor_crop(), np.random.default_rng(9), arch_shape_prob=1.0,
        )
        assert result.params["shape"] == "arch"

    @pytest.mark.parametrize("seed", range(10))
    def test_mode_a_arch_shape_boundary_is_non_rectangular(self, seed):
        image = _solid_image(CANVAS_W, CANVAS_H)
        result = generate_mode_a(
            image, FRAME_BOX, _donor_crop(), np.random.default_rng(seed), arch_shape_prob=1.0,
        )
        ys, xs = np.where(result.mask == 255)
        bbox_area = (xs.max() - xs.min() + 1) * (ys.max() - ys.min() + 1)
        mask_area = int(result.mask.sum() // 255)
        assert mask_area < bbox_area, f"seed={seed}: mask_area={mask_area} >= bbox_area={bbox_area}"

    def test_mode_a_arch_shape_flat_bottom(self):
        """The arch's defining feature vs. the general irregular silhouette: a flat bottom
        edge -- the mask's bottom row of the bounding box should be (near-)fully filled across
        its width, unlike the irregular silhouette's wobble on every edge."""
        from freuid.photosub.generators import _arch_patch_alpha
        alpha = _arch_patch_alpha(120, 140, np.random.default_rng(3))
        bottom_row = alpha[-1, :]
        assert (bottom_row > 0).mean() > 0.9

    def test_mode_a_arch_and_irregular_mutually_exclusive_and_sum_to_le_one(self):
        """arch_shape_prob=0.4, irregular_shape_prob=0.4 -> roughly 40/40/20 split (arch/
        irregular/rect) over many draws, never both at once for a single draw."""
        image = _solid_image(CANVAS_W, CANVAS_H)
        shapes = []
        for seed in range(200):
            result = generate_mode_a(
                image, FRAME_BOX, _donor_crop(), np.random.default_rng(seed),
                arch_shape_prob=0.4, irregular_shape_prob=0.4,
            )
            shapes.append(result.params["shape"])
        assert set(shapes) <= {"arch", "irregular", "rect"}
        arch_frac = shapes.count("arch") / len(shapes)
        irregular_frac = shapes.count("irregular") / len(shapes)
        assert 0.25 < arch_frac < 0.55
        assert 0.25 < irregular_frac < 0.55

    def test_mode_a_arch_shape_prob_zero_is_byte_identical_to_original(self):
        """arch_shape_prob=0.0 (the default) must reproduce the exact same output as calling
        without the kwarg at all -- the whole point of the additive-param discipline."""
        image = _solid_image(CANVAS_W, CANVAS_H)
        r_default = generate_mode_a(image, FRAME_BOX, _donor_crop(), np.random.default_rng(21))
        r_explicit_zero = generate_mode_a(
            image, FRAME_BOX, _donor_crop(), np.random.default_rng(21), arch_shape_prob=0.0,
        )
        assert np.array_equal(np.asarray(r_default.image), np.asarray(r_explicit_zero.image))
        assert r_default.params == r_explicit_zero.params

    def test_mode_a_tape_effect_visible(self):
        image = _solid_image(CANVAS_W, CANVAS_H)
        r_no_tape = generate_mode_a(image, FRAME_BOX, _donor_crop(), np.random.default_rng(11))
        r_tape = generate_mode_a(
            image, FRAME_BOX, _donor_crop(), np.random.default_rng(11), tape_prob=1.0,
        )
        assert r_tape.params["tape"] is True
        assert r_no_tape.params["tape"] is False
        assert not np.array_equal(np.asarray(r_no_tape.image), np.asarray(r_tape.image))

    def test_mode_a_tape_independent_of_shape(self):
        """tape_prob and arch_shape_prob/tear_prob are orthogonal -- can all fire together."""
        image = _solid_image(CANVAS_W, CANVAS_H)
        result = generate_mode_a(
            image, FRAME_BOX, _donor_crop(), np.random.default_rng(5),
            arch_shape_prob=1.0, tear_prob=1.0, tape_prob=1.0,
        )
        assert result.params["shape"] == "arch"
        assert result.params["tear"] is True
        assert result.params["tape"] is True

    def test_mode_a_tape_prob_zero_is_byte_identical_to_original(self):
        image = _solid_image(CANVAS_W, CANVAS_H)
        r_default = generate_mode_a(image, FRAME_BOX, _donor_crop(), np.random.default_rng(21))
        r_explicit_zero = generate_mode_a(
            image, FRAME_BOX, _donor_crop(), np.random.default_rng(21), tape_prob=0.0,
        )
        assert np.array_equal(np.asarray(r_default.image), np.asarray(r_explicit_zero.image))
        assert r_default.params == r_explicit_zero.params


# ---------------------------------------------------------------------------
# MODE_D gating
# ---------------------------------------------------------------------------

class TestModeDGating:
    def test_mode_d_disabled_by_default(self):
        image = _solid_image(CANVAS_W, CANVAS_H)
        rng = np.random.default_rng(6)
        with pytest.raises(RuntimeError, match="disabled by default"):
            generate_mode_d(image, FRAME_BOX, GHOST_BOX, _donor_crop(), rng)

    def test_mode_d_main_leaves_ghost_untouched(self):
        image = _solid_image(CANVAS_W, CANVAS_H)
        rng = np.random.default_rng(7)
        result = generate_mode_d(
            image, FRAME_BOX, GHOST_BOX, _donor_crop(), rng, swap_target="main", enabled=True,
        )
        assert result.mode == "D_main"
        assert result.mask[GHOST_BOX["y1"]:GHOST_BOX["y2"], GHOST_BOX["x1"]:GHOST_BOX["x2"]].sum() == 0

    def test_ghost_darken_prob_zero_never_darkens(self):
        image = _solid_image(CANVAS_W, CANVAS_H)
        rng = np.random.default_rng(9)
        for _ in range(10):
            result = generate_mode_d(
                image, FRAME_BOX, GHOST_BOX, _donor_crop(), rng, swap_target="ghost",
                enabled=True, ghost_darken_prob=0.0,
            )
            assert result.params["darkened"] is False
            assert result.params["darken_factor"] is None

    def test_ghost_darken_prob_one_always_darkens_and_dims_the_ghost_patch(self):
        image = _solid_image(CANVAS_W, CANVAS_H, color=(200, 200, 200))
        rng = np.random.default_rng(11)
        result = generate_mode_d(
            image, FRAME_BOX, GHOST_BOX, _donor_crop(), rng, swap_target="ghost",
            enabled=True, ghost_darken_prob=1.0, ghost_darken_factor_range=(0.2, 0.2),
        )
        assert result.params["darkened"] is True
        assert result.params["darken_factor"] == pytest.approx(0.2)
        ghost_patch = np.asarray(result.image)[
            GHOST_BOX["y1"]:GHOST_BOX["y2"], GHOST_BOX["x1"]:GHOST_BOX["x2"]
        ]
        # donor is a bright (200,150,100) patch; darkened by 0.2 it should read much darker than
        # an undarkened composite would (the target region itself is a mid-gray 120,130,140
        # background, so this isn't a trivial "any dark pixel" check).
        assert ghost_patch.mean() < 100

    def test_ghost_darken_only_applies_to_ghost_swap_not_main_swap(self):
        image = _solid_image(CANVAS_W, CANVAS_H)
        rng = np.random.default_rng(12)
        result = generate_mode_d(
            image, FRAME_BOX, GHOST_BOX, _donor_crop(), rng, swap_target="main",
            enabled=True, ghost_darken_prob=1.0,
        )
        assert result.mode == "D_main"
        assert "darkened" not in result.params  # darken machinery lives only in the ghost branch


# ---------------------------------------------------------------------------
# Template metadata (frame_box derivation, ghost_box lookup)
# ---------------------------------------------------------------------------

class TestTemplateRegions:
    def test_frame_box_expands_face_box_and_clamps_to_image(self):
        face = _face_box(190, 140, 210, 180)
        box = frame_box_from_face(face, img_w=CANVAS_W, img_h=CANVAS_H, margin_frac=0.6)
        assert box["x1"] < face["x1"] and box["y1"] < face["y1"]
        assert box["x2"] > face["x2"] and box["y2"] > face["y2"]
        assert 0 <= box["x1"] and box["x2"] <= CANVAS_W
        assert 0 <= box["y1"] and box["y2"] <= CANVAS_H

    def test_ghost_box_none_for_type_without_ghost(self):
        assert ghost_box_px("BENIN/DL", CANVAS_W, CANVAS_H) is None

    def test_ghost_box_defined_for_egypt_dl(self):
        box = ghost_box_px("EGYPT/DL", 1000, 1000)
        assert box is not None
        assert box["x2"] > box["x1"] and box["y2"] > box["y1"]

    def test_build_template_metadata_unknown_type_has_no_ghost(self):
        face = _face_box(190, 140, 210, 180)
        meta = build_template_metadata("id1", "ATLANTIS/ID", face, CANVAS_W, CANVAS_H)
        assert meta["ghost_box"] is None
        assert meta["frame_box"]["x1"] < face["x1"]


# ---------------------------------------------------------------------------
# Donor pool: exclusion rule + hard/broad sampling
# ---------------------------------------------------------------------------

def _donor(id_, type_, embedding, luminance=0.5) -> DonorFace:
    return DonorFace(
        id=id_, path=f"/fake/{id_}.jpeg", type=type_,
        face_box=_face_box(0, 0, 10, 10), is_grayscale=False,
        mean_luminance=luminance, embedding=embedding,
    )


class TestDonorExclusion:
    def test_planted_near_duplicate_is_excluded(self):
        base_vec = np.array([1.0, 0.0, 0.0])
        near_dup_vec = np.array([0.999, 0.001, 0.0])  # cosine sim ~= 0.999995
        different_vec = np.array([0.0, 1.0, 0.0])  # orthogonal, same type

        donors = [
            _donor("planted_dup", "EGYPT/DL", near_dup_vec),
            _donor("clearly_different", "EGYPT/DL", different_vec),
            _donor("other_type", "BENIN/DL", near_dup_vec),  # same embedding, different type
        ]
        probes = [_donor("probe_1", "EGYPT/DL", base_vec)]

        kept = exclude_probe_overlap(donors, probes, similarity_threshold=0.9)
        kept_ids = {d.id for d in kept}

        assert "planted_dup" not in kept_ids, "near-duplicate of a same-type probe must be excluded"
        assert "clearly_different" in kept_ids, "dissimilar same-type donor must be kept"
        assert "other_type" in kept_ids, "near-duplicate embedding but DIFFERENT type must be kept"

    def test_no_probes_keeps_all_donors(self):
        donors = [_donor("a", "EGYPT/DL", np.array([1.0, 0.0]))]
        assert exclude_probe_overlap(donors, [], similarity_threshold=0.9) == donors

    def test_cosine_similarity_identical_vectors_is_one(self):
        v = np.array([1.0, 2.0, 3.0])
        assert cosine_similarity(v, v) == pytest.approx(1.0)

    def test_cosine_similarity_orthogonal_is_zero(self):
        assert cosine_similarity(np.array([1.0, 0.0]), np.array([0.0, 1.0])) == pytest.approx(0.0)


class TestDonorSampling:
    def test_hard_fraction_zero_never_excludes_broad_pool(self):
        target = _donor("target", "EGYPT/DL", np.array([1.0, 0.0]), luminance=0.5)
        donors = [_donor(f"d{i}", "EGYPT/DL", np.array([0.0, 1.0]), luminance=0.9) for i in range(5)]
        rng = np.random.default_rng(0)
        picks = {sample_donor(donors, rng, target, hard_fraction=0.0).id for _ in range(20)}
        assert picks <= {d.id for d in donors}

    def test_hard_fraction_one_prefers_similar_embedding(self):
        target = _donor("target", "EGYPT/DL", np.array([1.0, 0.0]), luminance=0.5)
        close = _donor("close", "EGYPT/DL", np.array([0.99, 0.01]), luminance=0.5)
        far = _donor("far", "EGYPT/DL", np.array([0.0, 1.0]), luminance=0.5)
        rng = np.random.default_rng(0)
        picks = [sample_donor([close, far], rng, target, hard_fraction=1.0).id for _ in range(10)]
        assert all(p == "close" for p in picks)

    def test_excludes_target_itself(self):
        target = _donor("target", "EGYPT/DL", np.array([1.0, 0.0]))
        donors = [target, _donor("other", "EGYPT/DL", np.array([0.0, 1.0]))]
        rng = np.random.default_rng(0)
        picks = {sample_donor(donors, rng, target, hard_fraction=0.5).id for _ in range(20)}
        assert picks == {"other"}

    def test_empty_pool_after_exclusion_raises(self):
        target = _donor("only", "EGYPT/DL", np.array([1.0, 0.0]))
        with pytest.raises(ValueError, match="empty"):
            sample_donor([target], np.random.default_rng(0), target)


def test_build_donor_record_rejects_fallback_box(tmp_path):
    img_path = tmp_path / "x.jpeg"
    _solid_image(100, 100).save(img_path)
    fallback_box = _face_box(10, 10, 50, 50, score=0.0)  # score<=0 == center-square fallback
    assert build_donor_record("id1", img_path, "EGYPT/DL", fallback_box) is None


def test_build_donor_record_accepts_real_detection(tmp_path):
    img_path = tmp_path / "x.jpeg"
    _solid_image(100, 100).save(img_path)
    real_box = _face_box(10, 10, 50, 50, score=0.9)
    rec = build_donor_record("id1", img_path, "EGYPT/DL", real_box)
    assert rec is not None
    assert rec.id == "id1"
    assert rec.embedding.shape[0] > 0


def test_build_donor_record_rejects_when_embed_fn_returns_none(tmp_path):
    """A real face-recognition embedder (freuid.photosub.face_embedding.arcface_embedding) can
    fail to (re-)detect a face in the crop -- that must reject the donor, not silently fall back
    to a different embedding space for just this one record (which would make cosine similarity
    against the rest of the pool meaningless)."""
    img_path = tmp_path / "x.jpeg"
    _solid_image(100, 100).save(img_path)
    real_box = _face_box(10, 10, 50, 50, score=0.9)
    rec = build_donor_record("id1", img_path, "EGYPT/DL", real_box, embed_fn=lambda crop: None)
    assert rec is None


def test_margined_crop_is_superset_of_tight_crop(tmp_path):
    img = _checker_image(200, 150)
    box = _face_box(60, 40, 120, 100)
    tight = img.crop((box["x1"], box["y1"], box["x2"], box["y2"]))
    margined = _margined_crop(img, box, margin=0.3)
    assert margined.width >= tight.width and margined.height >= tight.height


def test_margined_crop_clamps_to_image_bounds(tmp_path):
    img = _solid_image(100, 100)
    box = _face_box(0, 0, 20, 20)  # near the corner -- margin would go negative without clamping
    margined = _margined_crop(img, box, margin=1.0)
    assert margined.width <= 100 and margined.height <= 100


# ---------------------------------------------------------------------------
# ArcFace embedding availability (freuid.photosub.face_embedding)
# ---------------------------------------------------------------------------

class TestFaceEmbeddingFallback:
    def test_best_available_embed_fn_matches_arcface_availability(self):
        """Whichever this environment has, best_available_embed_fn's choice must be consistent
        with arcface_available()'s own report -- not a hardcoded assumption about any one
        environment (this suite runs both in a bare dev env without insightface and, on VESSL,
        where ArcFace IS available)."""
        from freuid.photosub.donor_pool import cheap_face_embedding
        from freuid.photosub.face_embedding import arcface_embedding

        chosen = best_available_embed_fn()
        if arcface_available():
            assert chosen is arcface_embedding
        else:
            assert chosen is cheap_face_embedding


# ---------------------------------------------------------------------------
# Split-discipline bookkeeping (pipeline.save_tamper_result)
# ---------------------------------------------------------------------------

class TestPipelineRowBookkeeping:
    def test_generated_row_carries_source_id_and_mode(self, tmp_path):
        image = _solid_image(CANVAS_W, CANVAS_H)
        rng = np.random.default_rng(8)
        result = generate_mode_a(image, FRAME_BOX, _donor_crop(), rng)
        row = save_tamper_result(result, source_id="deadbeef1234", doc_type="EGYPT/DL", data_dir=tmp_path)

        assert row["source_id"] == "deadbeef1234"
        assert row["mode"] == "A"
        assert row["label"] == 1
        assert row["id"].startswith("deadbeef1234_photosub_A_")
        from pathlib import Path
        assert Path(row["image_path"]).exists()
        assert Path(row["mask_path"]).exists()

    def test_row_index_disambiguates_multiple_generations_from_same_source(self, tmp_path):
        image = _solid_image(CANVAS_W, CANVAS_H)
        donor = _donor_crop()
        r1 = generate_mode_a(image, FRAME_BOX, donor, np.random.default_rng(1))
        r2 = generate_mode_a(image, FRAME_BOX, donor, np.random.default_rng(2))
        row1 = save_tamper_result(r1, "src_id", "EGYPT/DL", tmp_path, row_index=0)
        row2 = save_tamper_result(r2, "src_id", "EGYPT/DL", tmp_path, row_index=1)
        assert row1["id"] != row2["id"]
        assert row1["source_id"] == row2["source_id"] == "src_id"

    def test_donor_id_recorded_when_passed(self, tmp_path):
        image = _solid_image(CANVAS_W, CANVAS_H)
        rng = np.random.default_rng(13)
        result = generate_mode_a(image, FRAME_BOX, _donor_crop(), rng)
        row = save_tamper_result(result, "src_id", "EGYPT/DL", tmp_path, donor_id="donor_abc123")
        assert row["donor_id"] == "donor_abc123"

    def test_donor_id_defaults_to_none(self, tmp_path):
        image = _solid_image(CANVAS_W, CANVAS_H)
        rng = np.random.default_rng(14)
        result = generate_mode_a(image, FRAME_BOX, _donor_crop(), rng)
        row = save_tamper_result(result, "src_id", "EGYPT/DL", tmp_path)
        assert row["donor_id"] is None

    def test_out_dir_override_keeps_two_generation_runs_from_colliding(self, tmp_path):
        """Regression test: ids are deterministic (source_id_photosub_mode_rowindex), so two
        generation runs sharing the same default directory would silently overwrite each other's
        files in place for any row with a matching (source_id, mode, row_index) -- exactly what
        happened to the photosub_v0 corpus during a v1 shape-realism regeneration pass before
        `out_dir` existed. Passing distinct `out_dir`s must keep both runs' files intact."""
        image = _solid_image(CANVAS_W, CANVAS_H)
        r1 = generate_mode_a(image, FRAME_BOX, _donor_crop(), np.random.default_rng(1))
        r2 = generate_mode_a(image, FRAME_BOX, _donor_crop(), np.random.default_rng(2))
        out_v0, out_v1 = tmp_path / "v0", tmp_path / "v1"

        row_v0 = save_tamper_result(r1, "same_src", "EGYPT/DL", tmp_path, row_index=0, out_dir=out_v0)
        row_v1 = save_tamper_result(r2, "same_src", "EGYPT/DL", tmp_path, row_index=0, out_dir=out_v1)

        assert row_v0["id"] == row_v1["id"]  # same deterministic id -- the whole point of the test
        assert row_v0["image_path"] != row_v1["image_path"]
        from pathlib import Path
        assert Path(row_v0["image_path"]).exists() and Path(row_v1["image_path"]).exists()
        assert not np.array_equal(
            np.asarray(Image.open(row_v0["image_path"])), np.asarray(Image.open(row_v1["image_path"])),
        )

    def test_out_dir_none_reproduces_default_location(self, tmp_path):
        image = _solid_image(CANVAS_W, CANVAS_H)
        rng = np.random.default_rng(15)
        result = generate_mode_a(image, FRAME_BOX, _donor_crop(), rng)
        row = save_tamper_result(result, "src_id", "EGYPT/DL", tmp_path)
        from freuid.photosub.pipeline import photosub_generated_dir
        assert row["image_path"].startswith(str(photosub_generated_dir(tmp_path)))
