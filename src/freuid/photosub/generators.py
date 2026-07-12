"""Photo-substitution generators: MODE_A/B/C/D.

Each generator is a deterministic function of (image, frame/ghost box, donor crop, rng, params)
-> TamperResult(image, mask, mode, params). See scripts/analysis/deep_miss_dossiers.py's module
docstring for the taxonomy (confirmed against 3 real deep-miss frauds, extended-by-eye to 6 more
-- see that script's CHECKLIST_RESULTS) these implement:

    MODE_A -- within-frame physical paste (style-mismatched print, paper rim, physical shadow,
              severed background print at the frame; recaptured elsewhere in the pipeline).
    MODE_B -- full-cover physical paste (overhangs the whole photo region, boundary coincides
              with the card's structural edge, rim + corner shadows + style mismatch).
    MODE_C -- frame-aligned digital swap (seam precisely on the frame edge; color- AND
              degradation-matched -- see freuid.photosub.degradation_match -- so it isn't a
              local-statistics outlier, only a cross-region semantic one). **v1: restricted to
              ghost-bearing templates only** (see scripts/generate_photosub_dataset.py's mode-
              selection logic) -- a "clean" digital swap on a template with no ghost/secondary-
              portrait feature has ZERO evidence this generator family can point to (no visible
              tell, no cross-region signature), which is exactly the "too clean" signature
              photosub_v0's deep-9 diagnostic suspects the model over-generalized from. On a
              ghost-bearing template, `generate_mode_c` never touches the ghost region, so
              swapping the main portrait automatically creates a real cross-region mismatch
              against the (untouched) ghost -- the function itself needs no code change for
              this restriction, only where it's called from. Note this makes ghost-restricted C
              and `generate_mode_d`'s `swap_target="main"` (D_main) branch produce the same
              rendering recipe -- an accepted, documented consequence of both being "evidence
              must be real" gates converging on the same non-ghostless-templates population, not
              a bug to resolve by collapsing the two mode labels.
    MODE_D -- ghost mismatch (swap main OR ghost, never both) -- only for ghost-bearing
              templates. Enabled at a modest weight in configs/photosub_v0.yaml, with a slice
              of ghost-swap rows deliberately darkened (``ghost_darken_prob``) to match the real
              illegible-ghost case documented in generate_mode_d's docstring -- still gated
              behind an explicit ``enabled=True`` at the function level.

Shape-realism fix: photosub_v0's deep-9 diagnostic (docs/technical_report.md) found MODE_A did
the OPPOSITE of its hypothesis -- 2 of 4 real deep-miss ids ended up more confidently bona-fide
than finetune_v0's own baseline -- most plausibly because this module originally only ever pasted
a rotated RECTANGLE, while the real exemplars show an arch-shaped cutout overlapping the crest
logo (`b5eebda1`), an irregular silhouette bulging past the hairline (`40dd1055`), a literal
diagonal tear exposing a lighter backing patch (`5542f45f`), and a visible tape strip
(`cd7ad569`, filed under MODE_B but not paste-mode-specific as a visual tell) -- see
CHECKLIST_RESULTS in deep_miss_dossiers.py. `generate_mode_a` offers `irregular_shape_prob`
(hand-cut silhouette via `_irregular_patch_alpha`), `arch_shape_prob` (arch/doorway silhouette
via `_arch_patch_alpha`), `tear_prob` (`_apply_tear_effect`), and `tape_prob`
(`_apply_tape_strips`); `generate_mode_b` offers `irregular_shape_prob` only. All default to
0.0 (byte-identical to the original rectangle-only behavior). Gated behind the same curated-
render-sheet human-review step already used for the MODE_D ghost-darkening decision before any
weight is chosen or the corpus is mass-regenerated -- **status as of photosub_v1**: irregular
+ tear reviewed and approved (docs/photosub_renders/mode_a_shape_variants_sheet.png,
docs/photosub_v1_spec.md has the review notes) at a real nonzero weight; arch + tape are
implemented and unit-tested but NOT yet rendered/reviewed (needs the regions cache, VESSL-only)
-- both stay at 0.0 in configs/photosub_v1.yaml until that review happens.

Mask convention: the returned mask marks pixels whose value came from the donor (the actual
photo swap), not pixels merely darkened by a rendered shadow -- a shadow alters original content
in place, it doesn't replace it with donor content.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from freuid.photosub.degradation_match import match_degradation
from freuid.photosub.print_style import PRINT_STYLES, apply_print_style

_ALL_SIDES = ("top", "right", "bottom", "left")
_TINT_COLORS = ((60, 160, 170), (150, 90, 170))  # teal / purple, per survey_templates_report.md


@dataclass
class TamperResult:
    image: Image.Image
    mask: np.ndarray  # uint8 HxW, 0 or 255, same size as the input image
    mode: str
    params: dict


# ---------------------------------------------------------------------------
# Shared geometry / compositing helpers
# ---------------------------------------------------------------------------

def _box_wh(box: dict) -> tuple[int, int]:
    w, h = int(box["x2"]) - int(box["x1"]), int(box["y2"]) - int(box["y1"])
    if w <= 0 or h <= 0:
        raise ValueError(f"degenerate box {box!r}")
    return w, h


def _fit_donor_to_box(donor: Image.Image, box_w: int, box_h: int) -> Image.Image:
    """Center-crop the donor to the box's aspect ratio, then resize exactly to (box_w, box_h)
    -- avoids the visible stretch a direct resize-to-arbitrary-aspect would cause."""
    dw, dh = donor.size
    target_ar = box_w / box_h
    src_ar = dw / dh
    if src_ar > target_ar:
        new_w = max(1, int(round(dh * target_ar)))
        x0 = (dw - new_w) // 2
        donor = donor.crop((x0, 0, x0 + new_w, dh))
    else:
        new_h = max(1, int(round(dw / target_ar)))
        y0 = (dh - new_h) // 2
        donor = donor.crop((0, y0, dw, y0 + new_h))
    return donor.resize((max(1, box_w), max(1, box_h)), Image.BILINEAR)


def _draw_rim(patch_rgba: Image.Image, rim_px: int, rim_color=(235, 235, 230, 255)) -> Image.Image:
    """A light border stroke just inside the patch's own edge, before rotation -- reads as a
    paper rim once composited (rotation carries it along with the patch)."""
    if rim_px <= 0:
        return patch_rgba
    out = patch_rgba.copy()
    draw = ImageDraw.Draw(out)
    w, h = out.size
    for i in range(rim_px):
        draw.rectangle((i, i, w - 1 - i, h - 1 - i), outline=rim_color)
    return out


def _shadow_layer(size: tuple[int, int], blur_radius: float, strength: int) -> Image.Image:
    """A soft dark RGBA silhouette meant to be pasted UNDER the patch at a small offset."""
    w, h = size
    shadow = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(shadow)
    draw.rectangle((0, 0, w - 1, h - 1), fill=(20, 20, 20, strength))
    return shadow.filter(ImageFilter.GaussianBlur(blur_radius))


def _composite_with_shadow(
    base: Image.Image,
    patch_rgba: Image.Image,
    center_xy: tuple[int, int],
    shadow_edges: tuple[str, ...],
    shadow_offset: int,
    shadow_blur: float,
    shadow_strength: int,
    shadow_alpha: np.ndarray | None = None,
) -> tuple[Image.Image, np.ndarray]:
    """Paste a soft shadow (offset toward `shadow_edges`) then the opaque patch, both centered
    at `center_xy`. Returns (composited image, mask marking the PATCH's own opaque footprint --
    i.e. donor content, not shadow-darkened original content). ``shadow_alpha`` (uint8 HxW, 0/255,
    same size as `patch_rgba`), when given, shapes the shadow to that silhouette instead of a
    plain rectangle -- pass the patch's own (irregular) alpha channel so the shadow doesn't leave
    a rectangular smudge past a hand-cut boundary; ``None`` (default) reproduces the original
    rectangle-shadow behavior exactly."""
    out = base.convert("RGB").copy()
    w, h = out.size
    pw, ph = patch_rgba.size
    cx, cy = center_xy

    dx = shadow_offset if "right" in shadow_edges else (-shadow_offset if "left" in shadow_edges else 0)
    dy = shadow_offset if "bottom" in shadow_edges else (-shadow_offset if "top" in shadow_edges else 0)
    if shadow_alpha is None:
        shadow = _shadow_layer((pw, ph), shadow_blur, shadow_strength)
    else:
        shadow = _shadow_layer_from_alpha(shadow_alpha, shadow_blur, shadow_strength)
    shadow_x, shadow_y = cx - pw // 2 + dx, cy - ph // 2 + dy
    out.paste(Image.new("RGB", (pw, ph), (0, 0, 0)), (shadow_x, shadow_y), shadow.split()[-1])

    px, py = cx - pw // 2, cy - ph // 2
    out.paste(patch_rgba, (px, py), patch_rgba.split()[-1])

    mask = np.zeros((h, w), dtype=np.uint8)
    alpha = np.asarray(patch_rgba.split()[-1])
    x0, y0 = max(0, px), max(0, py)
    x1, y1 = min(w, px + pw), min(h, py + ph)
    if x1 > x0 and y1 > y0:
        ax0, ay0 = x0 - px, y0 - py
        ax1, ay1 = ax0 + (x1 - x0), ay0 + (y1 - y0)
        mask[y0:y1, x0:x1] = np.where(alpha[ay0:ay1, ax0:ax1] > 127, 255, 0).astype(np.uint8)
    return out, mask


def _add_curl_highlight(patch_rgba: Image.Image, edge: str, width_frac: float = 0.12) -> Image.Image:
    """A soft bright gradient strip along one edge, simulating a lifted-corner reflection."""
    w, h = patch_rgba.size
    strip = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(strip)
    band = max(1, int((w if edge in ("left", "right") else h) * width_frac))
    if edge == "left":
        for i in range(band):
            draw.line((i, 0, i, h), fill=int(180 * (1 - i / band)))
    elif edge == "right":
        for i in range(band):
            draw.line((w - 1 - i, 0, w - 1 - i, h), fill=int(180 * (1 - i / band)))
    elif edge == "top":
        for i in range(band):
            draw.line((0, i, w, i), fill=int(180 * (1 - i / band)))
    else:  # bottom
        for i in range(band):
            draw.line((0, h - 1 - i, w, h - 1 - i), fill=int(180 * (1 - i / band)))
    strip = strip.filter(ImageFilter.GaussianBlur(band / 3.0 + 1))
    highlight = Image.new("RGBA", (w, h), (255, 255, 255, 0))
    highlight.putalpha(strip)
    return Image.alpha_composite(patch_rgba, highlight)


def _match_color(donor_arr: np.ndarray, target_arr: np.ndarray) -> np.ndarray:
    """Reinhard-style channel-wise mean/std transfer: matches the donor patch's color statistics
    to the surrounding card region it's being inserted into."""
    d_mean, d_std = donor_arr.mean(axis=(0, 1)), donor_arr.std(axis=(0, 1)) + 1e-6
    t_mean, t_std = target_arr.mean(axis=(0, 1)), target_arr.std(axis=(0, 1)) + 1e-6
    out = (donor_arr - d_mean) / d_std * t_std + t_mean
    return np.clip(out, 0, 255)


# ---------------------------------------------------------------------------
# Irregular hand-cut silhouette shape (shape-realism fix for MODE_A/B -- see module docstring)
# ---------------------------------------------------------------------------

def _irregular_patch_alpha(
    box_w: int,
    box_h: int,
    rng: np.random.Generator,
    jaggedness: float = 0.12,
    n_vertices: int = 14,
) -> np.ndarray:
    """A hand-cut-paper-style irregular silhouette (uint8 HxW alpha, 0/255) roughly filling a
    (box_w, box_h) canvas: ``n_vertices`` points placed around the box's own inscribed ellipse,
    each radially jittered by up to ``jaggedness * min(box_w, box_h)`` in EITHER direction (bulges
    out past the ellipse or notches in), approximating the arch-shaped cutouts and
    bulging-past-the-hairline silhouettes documented in deep_miss_dossiers.py's
    CHECKLIST_RESULTS -- not the plain rotated rectangle the rest of this module assumes. A light
    blur + rethreshold softens the polygon's straight edges into a hand-cut look."""
    cx, cy = box_w / 2.0, box_h / 2.0
    rx, ry = box_w / 2.0 * 0.94, box_h / 2.0 * 0.94  # slight inset so jitter stays mostly on-canvas
    jitter_px = jaggedness * min(box_w, box_h)
    angles = np.linspace(0, 2 * np.pi, n_vertices, endpoint=False) + rng.uniform(-0.15, 0.15, size=n_vertices)
    radial_jitter = rng.uniform(-jitter_px, jitter_px, size=n_vertices)
    pts = []
    for angle, jitter in zip(angles, radial_jitter):
        r_x, r_y = max(1.0, rx + jitter), max(1.0, ry + jitter)
        x = float(np.clip(cx + r_x * np.cos(angle), 0, box_w - 1))
        y = float(np.clip(cy + r_y * np.sin(angle), 0, box_h - 1))
        pts.append((x, y))

    canvas = Image.new("L", (box_w, box_h), 0)
    ImageDraw.Draw(canvas).polygon(pts, fill=255)
    softened = canvas.filter(ImageFilter.GaussianBlur(max(1.0, min(box_w, box_h) * 0.01)))
    return (np.asarray(softened) > 127).astype(np.uint8) * 255


def _erode_alpha(alpha_mask: np.ndarray, px: int) -> np.ndarray:
    """Minimal binary erosion by ``px`` pixels via iterative shifted-AND, pure numpy (no scipy
    dependency) -- good enough for a thin rim band at hand-cut-silhouette scale."""
    out = alpha_mask > 127
    for _ in range(max(0, px)):
        shifted = out.copy()
        shifted[1:, :] &= out[:-1, :]
        shifted[:-1, :] &= out[1:, :]
        shifted[:, 1:] &= out[:, :-1]
        shifted[:, :-1] &= out[:, 1:]
        out = shifted
    return out.astype(np.uint8) * 255


def _draw_rim_from_alpha(
    patch_rgba: Image.Image, alpha_mask: np.ndarray, rim_px: int, rim_color=(235, 235, 230, 255),
) -> Image.Image:
    """Rim for an arbitrary (e.g. irregular hand-cut) silhouette: paints ``rim_color`` into the
    band between ``alpha_mask`` and its own erosion by ``rim_px`` -- generalizes ``_draw_rim``'s
    rectangle-border logic to any shape, since a plain border stroke only makes sense on an
    axis-aligned rectangle."""
    if rim_px <= 0:
        return patch_rgba
    eroded = _erode_alpha(alpha_mask, rim_px) > 127
    band = (alpha_mask > 127) & ~eroded
    arr = np.asarray(patch_rgba).copy()
    arr[band] = rim_color
    return Image.fromarray(arr, mode="RGBA")


def _shadow_layer_from_alpha(alpha_mask: np.ndarray, blur_radius: float, strength: int) -> Image.Image:
    """Same as ``_shadow_layer`` but the silhouette follows ``alpha_mask`` (uint8 HxW, 0/255)
    instead of a plain filled rectangle -- needed so an irregular-shaped patch's shadow doesn't
    leave a rectangular smudge visible past its own hand-cut boundary."""
    h, w = alpha_mask.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[..., 0] = 20
    rgba[..., 1] = 20
    rgba[..., 2] = 20
    rgba[..., 3] = np.where(alpha_mask > 127, strength, 0).astype(np.uint8)
    return Image.fromarray(rgba, mode="RGBA").filter(ImageFilter.GaussianBlur(blur_radius))


def _arch_patch_alpha(
    box_w: int,
    box_h: int,
    rng: np.random.Generator,
    dome_height_frac_range: tuple[float, float] = (0.30, 0.48),
) -> np.ndarray:
    """A flat-bottomed, rounded-top silhouette (uint8 HxW alpha, 0/255) -- a paper cut in an
    arch/doorway shape, approximating deep-miss exemplar `b5eebda1`'s "arch-shaped cutout
    overlapping the crest logo" (deep_miss_dossiers.py's CHECKLIST_RESULTS). The lower
    ``1 - dome_height_frac`` of the box is a plain rectangle; the upper band is a half-ellipse
    whose apex touches the box's own top edge. Caveat: this compositing path pastes a patch
    sized to (box_w, box_h) centered on the frame box, so it cannot extend PAST the frame box's
    own bounds the way MODE_B's overhang mechanism does -- the dome apex reaching the box's own
    top edge approximates "pokes into the area above the slot" without an architecture change
    to box sizing. A small horizontal shear jitter avoids a perfectly symmetric, obviously-
    synthetic dome every time."""
    canvas = Image.new("L", (box_w, box_h), 0)
    draw = ImageDraw.Draw(canvas)
    dome_h = int(box_h * float(rng.uniform(*dome_height_frac_range)))
    draw.rectangle((0, dome_h, box_w - 1, box_h - 1), fill=255)
    draw.ellipse((0, 0, box_w - 1, dome_h * 2), fill=255)

    shear_px = int(rng.integers(-int(box_w * 0.06) - 1, int(box_w * 0.06) + 2))
    sheared = canvas.transform(
        canvas.size, Image.AFFINE, (1, 0, shear_px, 0, 1, 0), fillcolor=0,
    )
    softened = sheared.filter(ImageFilter.GaussianBlur(max(1.0, min(box_w, box_h) * 0.008)))
    return (np.asarray(softened) > 127).astype(np.uint8) * 255


def _apply_tape_strips(
    patch_rgba: Image.Image,
    rng: np.random.Generator,
    n_strips_range: tuple[int, int] = (1, 2),
    tape_color: tuple[int, int, int, int] = (214, 196, 140, 190),
) -> Image.Image:
    """Overlays 1-2 semi-transparent tan/yellow tape strips near a randomly-chosen edge --
    deep-miss exemplar `cd7ad569`'s "visible tan/yellow TAPE strip at the top of the photo"
    (deep_miss_dossiers.py's CHECKLIST_RESULTS; that exemplar is filed under MODE_B, but the
    visual tell itself -- tape holding a physical paste in place -- is not exclusive to a
    full-cover paste, so it's offered here as a MODE_A overlay too). Drawn on the patch's own
    RGBA before rotation, so it rotates along with the patch exactly like the rim/curl effects
    already do."""
    w, h = patch_rgba.size
    n = int(rng.integers(n_strips_range[0], n_strips_range[1] + 1))
    overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    edges = ("top", "bottom", "left", "right")
    for _ in range(n):
        edge = edges[int(rng.integers(len(edges)))]
        strip_span = w if edge in ("top", "bottom") else h
        strip_len = float(rng.uniform(0.30, 0.55)) * strip_span
        strip_w = max(2, int(0.06 * min(w, h)))
        pos = float(rng.uniform(0.15, 0.85))
        if edge == "top":
            cx = pos * w
            draw.rectangle((cx - strip_len / 2, 0, cx + strip_len / 2, strip_w), fill=tape_color)
        elif edge == "bottom":
            cx = pos * w
            draw.rectangle((cx - strip_len / 2, h - strip_w, cx + strip_len / 2, h), fill=tape_color)
        elif edge == "left":
            cy = pos * h
            draw.rectangle((0, cy - strip_len / 2, strip_w, cy + strip_len / 2), fill=tape_color)
        else:
            cy = pos * h
            draw.rectangle(
                (w - strip_w, cy - strip_len / 2, w, cy + strip_len / 2), fill=tape_color,
            )
    return Image.alpha_composite(patch_rgba, overlay)


def _apply_tear_effect(
    patch_rgba: Image.Image,
    rng: np.random.Generator,
    reveal_lighten: float = 0.55,
    tear_width_frac: float = 0.035,
    tear_length_frac_range: tuple[float, float] = (0.28, 0.5),
) -> Image.Image:
    """Overlays a SHORT jagged tear anchored at one randomly-chosen corner, running diagonally
    inward for only `tear_length_frac_range` of the patch's shorter side -- reproducing deep-miss
    id 5542f45f's "diagonal tear visible through the UPPER-RIGHT of the photo", not a full
    top-to-bottom crack bisecting the whole patch (an earlier version of this effect did exactly
    that and read as a stark, central lightning-bolt overlay on the render-sheet human-review
    gate -- rejected and reworked into this localized, corner-anchored version). The tear band
    LIGHTENS the patch's own local color by `reveal_lighten` (blended toward white) rather than
    pasting a fixed absolute color -- stays in the same tone family as whatever's under it instead
    of introducing a jarring, unrelated hue, closer to "a lighter patch of the same paper visible
    beneath a rip" than a flat-color scar. Works on the patch's own RGB channels before
    compositing, so it applies equally to a rectangular or irregular-silhouette patch -- pixels
    outside the patch's own alpha footprint stay invisible regardless, so no separate masking is
    needed here."""
    w, h = patch_rgba.size
    arr = np.asarray(patch_rgba).copy()

    corner = int(rng.integers(4))  # 0=top-left, 1=top-right, 2=bottom-left, 3=bottom-right
    tear_len = float(rng.uniform(*tear_length_frac_range)) * min(w, h)
    edge_bias = float(rng.uniform(0, 0.4)) * h
    if corner == 0:
        start, direction = (0.0, edge_bias), (1.0, 1.0)
    elif corner == 1:
        start, direction = (w - 1.0, edge_bias), (-1.0, 1.0)
    elif corner == 2:
        start, direction = (0.0, h - 1.0 - edge_bias), (1.0, -1.0)
    else:
        start, direction = (w - 1.0, h - 1.0 - edge_bias), (-1.0, -1.0)

    n_pts = 5
    pts = [start]
    for i in range(1, n_pts):
        frac = i / (n_pts - 1)
        jitter = float(rng.uniform(-0.15, 0.15)) * tear_len
        x = np.clip(start[0] + direction[0] * tear_len * frac + jitter, 0, w - 1)
        y = np.clip(start[1] + direction[1] * tear_len * frac, 0, h - 1)
        pts.append((float(x), float(y)))

    line_width = max(1, int(round(tear_width_frac * min(w, h))))
    tear_canvas = Image.new("L", (w, h), 0)
    ImageDraw.Draw(tear_canvas).line(pts, fill=255, width=line_width)
    tear_canvas = tear_canvas.filter(ImageFilter.GaussianBlur(max(0.5, line_width * 0.3)))
    tear_band = np.asarray(tear_canvas) > 100

    rgb = arr[..., :3].astype(np.float64)
    lightened = rgb + (255.0 - rgb) * reveal_lighten
    rgb[tear_band] = lightened[tear_band]
    arr[..., :3] = np.clip(rgb, 0, 255).astype(np.uint8)
    return Image.fromarray(arr, mode="RGBA")


# ---------------------------------------------------------------------------
# MODE_A -- within-frame physical paste
# ---------------------------------------------------------------------------

def generate_mode_a(
    image: Image.Image,
    frame_box: dict,
    donor_crop: Image.Image,
    rng: np.random.Generator,
    style: str | None = None,
    rotation_deg_range: tuple[float, float] = (-3.0, 3.0),
    offset_frac: float = 0.06,
    rim_px_range: tuple[int, int] = (1, 4),
    irregular_shape_prob: float = 0.0,
    arch_shape_prob: float = 0.0,
    tear_prob: float = 0.0,
    tape_prob: float = 0.0,
) -> TamperResult:
    """Within-frame physical paste. Style-mismatched print inset within the photo frame with a
    small random rotation/offset, a light paper rim, and a soft drop shadow along 2 adjacent
    edges. Opaque compositing severs whatever card print falls under the patch for free (verify
    with a render over a template whose background pattern crosses the frame -- see the
    render-sheet script).

    ``irregular_shape_prob`` / ``arch_shape_prob`` / ``tear_prob`` / ``tape_prob`` (all default
    0.0, exactly reproducing the original rotated-rectangle-only behavior -- every check below
    short-circuits before touching `rng` at all when its prob is 0.0, so the default path's
    random-draw sequence, and therefore its output, is byte-identical to before these params
    existed): the shape-realism fix flagged in this module's docstring and corroborated by
    photosub_v0's deep-9 diagnostic (docs/technical_report.md) -- MODE_A's real deep-miss
    exemplars show an arch-shaped cutout overlapping the crest logo (`b5eebda1`), an irregular
    silhouette bulging past the hairline (`40dd1055`), a literal diagonal tear exposing a lighter
    backing patch (`5542f45f`), and (filed under MODE_B but offered here too, since the tell
    itself isn't paste-mode-specific) a visible tape strip (`cd7ad569`) -- not a clean rotated
    rectangle.

    ``irregular_shape_prob`` and ``arch_shape_prob`` are mutually exclusive slices of one
    probability line (rect gets the remainder, so "rectangle remains one variant" as long as
    their sum is < 1): a single ``rng.random()`` draw picks arch first, then irregular, then
    falls through to rect -- see the code below for the exact order, which matters for
    reproducing draw sequences. ``_irregular_patch_alpha``/``_arch_patch_alpha`` both produce an
    alpha mask; rim and shadow follow whichever mask was chosen via
    ``_draw_rim_from_alpha``/``_composite_with_shadow``'s `shadow_alpha`. ``tear_prob``/
    ``tape_prob`` independently, and orthogonally, overlay ``_apply_tear_effect``/
    ``_apply_tape_strips`` regardless of shape.
    """
    style = style or PRINT_STYLES[int(rng.integers(len(PRINT_STYLES)))]
    fw, fh = _box_wh(frame_box)
    fitted = _fit_donor_to_box(donor_crop.convert("RGB"), fw, fh)
    styled = apply_print_style(fitted, style, rng)

    rim_px = int(rng.integers(rim_px_range[0], rim_px_range[1] + 1))
    shape_prob_total = arch_shape_prob + irregular_shape_prob
    shape_kind = "rect"
    if shape_prob_total > 0.0:
        shape_roll = rng.random()
        if shape_roll < arch_shape_prob:
            shape_kind = "arch"
        elif shape_roll < shape_prob_total:
            shape_kind = "irregular"

    if shape_kind == "arch":
        shape_alpha = _arch_patch_alpha(fw, fh, rng)
    elif shape_kind == "irregular":
        shape_alpha = _irregular_patch_alpha(fw, fh, rng)
    else:
        shape_alpha = None

    if shape_alpha is not None:
        rgba_arr = np.dstack([np.asarray(styled.convert("RGB")), shape_alpha])
        patch_rgba = _draw_rim_from_alpha(
            Image.fromarray(rgba_arr, mode="RGBA"), shape_alpha, rim_px,
        )
    else:
        patch_rgba = _draw_rim(styled.convert("RGBA"), rim_px)

    use_tear = tear_prob > 0.0 and rng.random() < tear_prob
    if use_tear:
        patch_rgba = _apply_tear_effect(patch_rgba, rng)

    use_tape = tape_prob > 0.0 and rng.random() < tape_prob
    if use_tape:
        patch_rgba = _apply_tape_strips(patch_rgba, rng)

    angle = float(rng.uniform(*rotation_deg_range))
    rotated = patch_rgba.rotate(angle, expand=True, resample=Image.BICUBIC, fillcolor=(0, 0, 0, 0))

    fx1, fy1, fx2, fy2 = frame_box["x1"], frame_box["y1"], frame_box["x2"], frame_box["y2"]
    center_x, center_y = (fx1 + fx2) // 2, (fy1 + fy2) // 2
    max_off_x, max_off_y = int(fw * offset_frac), int(fh * offset_frac)
    off_x = int(rng.integers(-max_off_x, max_off_x + 1)) if max_off_x > 0 else 0
    off_y = int(rng.integers(-max_off_y, max_off_y + 1)) if max_off_y > 0 else 0

    edge_pairs = (("top", "left"), ("top", "right"), ("bottom", "left"), ("bottom", "right"))
    shadow_edges = edge_pairs[int(rng.integers(len(edge_pairs)))]
    shadow_offset = max(1, int(round(min(fw, fh) * 0.02)))
    shadow_blur = float(rng.uniform(2.0, 5.0))
    shadow_strength = int(rng.integers(60, 140))

    out, mask = _composite_with_shadow(
        image, rotated, (center_x + off_x, center_y + off_y),
        shadow_edges, shadow_offset, shadow_blur, shadow_strength,
        shadow_alpha=np.asarray(rotated.split()[-1]) if shape_alpha is not None else None,
    )
    params = {
        "style": style, "rim_px": rim_px, "rotation_deg": angle,
        "offset_xy": [off_x, off_y], "shadow_edges": list(shadow_edges),
        "shadow_offset": shadow_offset, "shadow_blur": shadow_blur,
        "shadow_strength": shadow_strength,
        "shape": shape_kind, "tear": use_tear, "tape": use_tape,
    }
    return TamperResult(image=out, mask=mask, mode="A", params=params)


# ---------------------------------------------------------------------------
# MODE_B -- full-cover physical paste
# ---------------------------------------------------------------------------

def generate_mode_b(
    image: Image.Image,
    frame_box: dict,
    donor_crop: Image.Image,
    rng: np.random.Generator,
    style: str | None = None,
    rotation_deg_range: tuple[float, float] = (-3.0, 3.0),
    overhang_frac_range: tuple[float, float] = (0.02, 0.08),
    rim_px_range: tuple[int, int] = (1, 4),
    irregular_shape_prob: float = 0.0,
) -> TamperResult:
    """Full-cover physical paste. Sized to cover the WHOLE photo region with random overhang
    (1-3 sides, each up to ~8% beyond the frame), corner shadows, an optional slight curl
    highlight on one edge. Overhang guarantees the mask's area exceeds the frame box's own area
    (asserted in tests/test_photosub.py) -- the paste always severs surrounding print; merely
    filling the frame exactly is MODE_C's job, not this one's.

    ``irregular_shape_prob`` (default 0.0, byte-identical to the original rectangle-only
    behavior -- see ``generate_mode_a``'s docstring for the same short-circuit guarantee and the
    shape-realism motivation): unlike MODE_A, none of MODE_B's 3 confirmed deep-miss ids showed
    this gap (all converged fine in photosub_v0), so this is offered as a lower-weight, optional
    diversity variant rather than a required fix. When engaged, the curl-highlight flourish is
    skipped (it isn't motivated by any real exemplar and would bleed color past an irregular
    silhouette's transparent corners via plain alpha-compositing) -- the curl coin-flip itself
    still runs either way so the default path's random-draw sequence is unaffected."""
    style = style or PRINT_STYLES[int(rng.integers(len(PRINT_STYLES)))]
    fw, fh = _box_wh(frame_box)

    n_sides = int(rng.integers(1, 4))  # 1-3 sides
    sides = list(rng.choice(_ALL_SIDES, size=n_sides, replace=False))
    overhang = {s: float(rng.uniform(*overhang_frac_range)) for s in sides}
    ext_left = int(fw * overhang.get("left", 0.0))
    ext_right = int(fw * overhang.get("right", 0.0))
    ext_top = int(fh * overhang.get("top", 0.0))
    ext_bottom = int(fh * overhang.get("bottom", 0.0))
    box_w = fw + ext_left + ext_right
    box_h = fh + ext_top + ext_bottom

    fitted = _fit_donor_to_box(donor_crop.convert("RGB"), box_w, box_h)
    styled = apply_print_style(fitted, style, rng)

    rim_px = int(rng.integers(rim_px_range[0], rim_px_range[1] + 1))
    use_irregular = irregular_shape_prob > 0.0 and rng.random() < irregular_shape_prob
    if use_irregular:
        irregular_alpha = _irregular_patch_alpha(box_w, box_h, rng)
        rgba_arr = np.dstack([np.asarray(styled.convert("RGB")), irregular_alpha])
        patch_rgba = _draw_rim_from_alpha(Image.fromarray(rgba_arr, mode="RGBA"), irregular_alpha, rim_px)
    else:
        patch_rgba = _draw_rim(styled.convert("RGBA"), rim_px)

    add_curl = bool(rng.random() < 0.5)
    curl_edge = _ALL_SIDES[int(rng.integers(len(_ALL_SIDES)))] if add_curl else None
    if add_curl and not use_irregular:
        patch_rgba = _add_curl_highlight(patch_rgba, curl_edge)

    angle = float(rng.uniform(*rotation_deg_range))
    rotated = patch_rgba.rotate(angle, expand=True, resample=Image.BICUBIC, fillcolor=(0, 0, 0, 0))

    fx1, fy1, fx2, fy2 = frame_box["x1"], frame_box["y1"], frame_box["x2"], frame_box["y2"]
    frame_cx, frame_cy = (fx1 + fx2) // 2, (fy1 + fy2) // 2
    # Recenter the (possibly asymmetrically enlarged) box so its overhang lands on the chosen sides.
    center_x = frame_cx + (ext_right - ext_left) // 2
    center_y = frame_cy + (ext_bottom - ext_top) // 2

    if len(sides) >= 2:
        shadow_edges = tuple(sides[:2])
    else:
        shadow_edges = (sides[0], _ALL_SIDES[(_ALL_SIDES.index(sides[0]) + 1) % 4])
    shadow_offset = max(1, int(round(min(box_w, box_h) * 0.02)))
    shadow_blur = float(rng.uniform(2.0, 5.0))
    shadow_strength = int(rng.integers(80, 160))  # full-cover/corner shadow reads slightly stronger

    out, mask = _composite_with_shadow(
        image, rotated, (center_x, center_y),
        shadow_edges, shadow_offset, shadow_blur, shadow_strength,
        shadow_alpha=np.asarray(rotated.split()[-1]) if use_irregular else None,
    )
    params = {
        "style": style, "rim_px": rim_px, "rotation_deg": angle, "overhang_sides": sides,
        "overhang_frac": overhang, "curl_edge": curl_edge, "shadow_edges": list(shadow_edges),
        "shape": "irregular" if use_irregular else "rect",
    }
    return TamperResult(image=out, mask=mask, mode="B", params=params)


# ---------------------------------------------------------------------------
# MODE_C -- frame-aligned digital swap
# ---------------------------------------------------------------------------

def generate_mode_c(
    image: Image.Image,
    frame_box: dict,
    donor_crop: Image.Image,
    rng: np.random.Generator,
    extra_grain: bool | None = None,
) -> TamperResult:
    """Frame-aligned digital swap. Donor color-matched AND degradation-matched (blur + JPEG
    grain -- see freuid.photosub.degradation_match) to the target region's own local stats, then
    composited to EXACTLY the photo-frame box -- no rim, no shadow, no print-style transform,
    seam precisely on the frame edge. The evidence is deliberately only cross-region (ghost/
    fields untouched by this function -- generate_mode_d is the ghost-aware sibling).

    Degradation matching is MANDATORY here, not a toggle: the first version of this generator
    only color-matched the donor and left blur/JPEG-grain unmatched, and the render-sheet
    self-check (docs/photosub_renders/stats_check_report.md) confirmed that WAS locally
    distinguishable by cheap stats (blur_laplacian_var deltas to -685, moire_fft_score deltas
    almost always positive) -- exactly what MODE_C's design is supposed to avoid. ``extra_grain``
    optionally layers a second, slightly heavier noise pass on top of the matched baseline (a
    "well-worn/rescanned copy" variant, purely for render diversity) -- it does not replace the
    matching step.

    See generators.py's module docstring for the frame_box-accuracy caveat this mode is most
    sensitive to (frame_box here is a face-box-derived approximation, not a surveyed template
    annotation)."""
    fx1, fy1, fx2, fy2 = frame_box["x1"], frame_box["y1"], frame_box["x2"], frame_box["y2"]
    fw, fh = _box_wh(frame_box)

    fitted = _fit_donor_to_box(donor_crop.convert("RGB"), fw, fh)
    target_region = image.convert("RGB").crop((fx1, fy1, fx2, fy2))
    donor_arr = np.asarray(fitted, dtype=np.float64)
    matched_color = Image.fromarray(_match_color(donor_arr, np.asarray(target_region, dtype=np.float64)).astype(np.uint8))

    patch = match_degradation(matched_color, target_region)

    extra_grain = bool(rng.random() < 0.5) if extra_grain is None else extra_grain
    if extra_grain:
        arr = np.asarray(patch, dtype=np.float64)
        noise = rng.normal(0.0, 3.0, size=arr.shape[:2])
        patch = Image.fromarray(np.clip(arr + noise[..., None], 0, 255).astype(np.uint8))

    out = image.convert("RGB").copy()
    out.paste(patch, (fx1, fy1))

    w, h = out.size
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[fy1:fy2, fx1:fx2] = 255

    return TamperResult(image=out, mask=mask, mode="C", params={"extra_grain": extra_grain})


# ---------------------------------------------------------------------------
# MODE_D -- ghost mismatch (decision: ON at a modest weight, see configs/photosub_v0.yaml)
# ---------------------------------------------------------------------------

_GHOST_DARKEN_FACTOR_RANGE = (0.15, 0.45)  # severe dimming, matching id 40dd1055's illegible ghost


def generate_mode_d(
    image: Image.Image,
    frame_box: dict,
    ghost_box: dict,
    donor_crop: Image.Image,
    rng: np.random.Generator,
    ghost_style: Literal["grayscale", "tinted"] = "grayscale",
    swap_target: Literal["main", "ghost"] | None = None,
    enabled: bool = False,
    ghost_darken_prob: float = 0.0,
    ghost_darken_factor_range: tuple[float, float] = _GHOST_DARKEN_FACTOR_RANGE,
) -> TamperResult:
    """Ghost mismatch. Swaps EITHER the main portrait (ghost left untouched) OR the ghost region
    (main portrait left untouched) -- 50/50 by default -- for templates that carry a resolvable
    ghost/secondary-portrait security feature (``ghost_box`` must be non-None; see
    freuid.photosub.template_regions.GHOST_TEMPLATES).

    **Defaults to a hard failure unless ``enabled=True`` is passed explicitly** -- this stays a
    deliberate opt-in gate even though the legibility question below is now resolved, so a
    caller can't enable MODE_D by accident.

    Per ``scripts/analysis/deep_miss_dossiers_out/ghost_resolvability_report.md``, ghost regions
    measured so far stay >=~56px (short side) at every TTA scale -- so SIZE was never the
    concern -- but that report also flagged a real, separate EXPOSURE/legibility failure (id
    ``40dd1055fd``'s ghost is present but unreadably dark even at 4x zoom): training only on
    crisp, legible synthetic ghosts would teach a "ghost mismatch" signal the model then can't
    actually read at inference time on real, similarly-dark ghosts. Resolved by
    ``ghost_darken_prob``: for a SLICE of ghost-swap rows (only the ``swap_target == "ghost"``
    branch -- ``D_main`` doesn't touch the ghost region at all, so darkening it there wouldn't
    mean anything), the composited ghost patch is additionally dimmed by a random factor in
    ``ghost_darken_factor_range`` BEFORE degradation-matching, deliberately reproducing the real
    illegible case rather than only ever generating crisp ones. 0.0 (the default) reproduces the
    original crisp-only behavior exactly.
    """
    if not enabled:
        raise RuntimeError(
            "generate_mode_d is disabled by default -- pass enabled=True explicitly to use it "
            "(see this function's docstring for the ghost-legibility design decision already "
            "made for this project)."
        )
    swap_target = swap_target or ("main" if rng.random() < 0.5 else "ghost")

    if swap_target == "main":
        result = generate_mode_c(image, frame_box, donor_crop, rng)
        result.mode = "D_main"
        result.params["swap_target"] = "main"
        result.params["ghost_untouched"] = True
        return result

    gx1, gy1, gx2, gy2 = ghost_box["x1"], ghost_box["y1"], ghost_box["x2"], ghost_box["y2"]
    gw, gh = gx2 - gx1, gy2 - gy1
    fitted = _fit_donor_to_box(donor_crop.convert("RGB"), gw, gh)
    if ghost_style == "grayscale":
        styled = fitted.convert("L").convert("RGB")
    else:  # tinted -- rough approximation of the teal/purple tint in MAURITIUS/ID samples
        gray = np.asarray(fitted.convert("L"), dtype=np.float64)
        tint = np.array(_TINT_COLORS[int(rng.integers(len(_TINT_COLORS)))], dtype=np.float64)
        arr = (gray[..., None] / 255.0) * tint[None, None, :]
        styled = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))

    darkened = bool(rng.random() < ghost_darken_prob)
    darken_factor = None
    if darkened:
        darken_factor = float(rng.uniform(*ghost_darken_factor_range))
        arr = np.asarray(styled, dtype=np.float64) * darken_factor
        styled = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))

    # Same degradation-matching requirement as MODE_C -- ghost regions are typically a small,
    # already-degraded print (see ghost_resolvability_report.md), so match against ITS OWN local
    # region, not the frame_box's. Runs AFTER darkening so the final blur/moire/jpeg profile is
    # still matched to the target's own local stats regardless of the darken step.
    ghost_target_region = image.convert("RGB").crop((gx1, gy1, gx2, gy2))
    styled = match_degradation(styled, ghost_target_region)

    out = image.convert("RGB").copy()
    out.paste(styled, (gx1, gy1))

    w, h = out.size
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[gy1:gy2, gx1:gx2] = 255

    params = {
        "swap_target": "ghost", "ghost_style": ghost_style, "main_untouched": True,
        "darkened": darkened, "darken_factor": darken_factor,
    }
    return TamperResult(image=out, mask=mask, mode="D_ghost", params=params)


GENERATORS = {
    "A": generate_mode_a,
    "B": generate_mode_b,
    "C": generate_mode_c,
    "D": generate_mode_d,
}
