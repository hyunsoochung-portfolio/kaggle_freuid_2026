"""GPU feasibility dry-run: model x resolution x grad_checkpointing timing/memory matrix.

MUST run on the VESSL A100 workspace -- CPU/laptop timing numbers are meaningless for this.
Analysis only: builds real models via freuid.models.build_model and runs short synthetic
train loops (no real data, no checkpoint saved, never touches src/freuid/). Outputs under
reports/res_precheck/.

Usage (on freuid-hy):
    /opt/conda/bin/python3 scripts/analysis/vit_res_dryrun.py
"""

from __future__ import annotations

import gc
import sys
import time
import warnings
from pathlib import Path

import torch
import torch.nn as nn
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from freuid.models import build_model  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "reports" / "res_precheck"

MODELS = [
    "vit_base_patch14_reg4_dinov2.lvd142m",
    "vit_large_patch14_reg4_dinov2.lvd142m",
]
RESOLUTIONS = [518, 784, 1036]
PATCH = 14
NUM_REGISTER_TOKENS = 4
GRAD_CKPT_OPTIONS = [False, True]
WARMUP_STEPS = 5
TIMED_STEPS = 30
TTA_SCALES = {
    518: [476, 518, 560],
    784: [728, 784, 840],
    1036: [980, 1036, 1092],
}
TRAIN_N = None  # filled from the real dataset size minus val_fraction, at runtime
EPOCHS = 20


def token_counts() -> dict[int, tuple[int, int]]:
    out = {}
    for res in RESOLUTIONS:
        assert res % PATCH == 0, f"{res}px is not a multiple of patch size {PATCH}"
        side = res // PATCH
        patch_tokens = side * side
        total_tokens = patch_tokens + 1 + NUM_REGISTER_TOKENS  # + CLS + registers
        out[res] = (patch_tokens, total_tokens)
        print(f"[dryrun] {res}px -> {side}x{side}={patch_tokens} patch tokens, "
              f"+1 CLS +{NUM_REGISTER_TOKENS} reg = {total_tokens} total")
    return out


def read_finetune_v0_config() -> tuple[int, float]:
    cfg = yaml.safe_load((REPO_ROOT / "configs" / "finetune_v0.yaml").read_text())
    return int(cfg["batch_size"]), float(cfg.get("val_fraction", 0.1))


def check_pos_embed_interpolation(model_name: str, device) -> list[str]:
    """Load pretrained=True once, forward at 784 and 1036 (native is 518), and capture
    any warnings/errors from the pos-embed interpolation path. Returns a list of notes."""
    notes = []
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model = build_model(model_name, pretrained=True).to(device).eval()
        for res in (784, 1036):
            try:
                with torch.no_grad():
                    x = torch.randn(1, 3, res, res, device=device)
                    out = model(x)
                assert out.shape == (1, 1), f"unexpected output shape {out.shape}"
                notes.append(f"{model_name} @ {res}px pretrained forward OK, logits shape {tuple(out.shape)}")
            except Exception as e:
                notes.append(f"{model_name} @ {res}px pretrained forward FAILED: {e!r}")
        for w in caught:
            notes.append(f"WARNING [{model_name}]: {w.category.__name__}: {w.message}")
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return notes


def build_synthetic_batch(bs: int, res: int, device):
    imgs = torch.randn(bs, 3, res, res, device=device)
    labels = torch.randint(0, 2, (bs, 1), device=device, dtype=torch.float32)
    return imgs, labels


def time_steps(model, bs: int, res: int, device, criterion) -> tuple[float, float]:
    """Runs WARMUP_STEPS + TIMED_STEPS train steps; returns (s_per_it, peak_gb).
    Raises torch.cuda.OutOfMemoryError if it doesn't fit."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=5e-2)
    scaler = torch.cuda.amp.GradScaler(enabled=True)
    torch.cuda.reset_peak_memory_stats(device)
    model.train()
    t0 = None
    for step in range(WARMUP_STEPS + TIMED_STEPS):
        imgs, labels = build_synthetic_batch(bs, res, device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", enabled=True):
            logits = model(imgs)
            loss = criterion(logits, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        if step == WARMUP_STEPS - 1:
            torch.cuda.synchronize()
            t0 = time.perf_counter()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    peak_gb = torch.cuda.max_memory_allocated(device) / 1e9
    return elapsed / TIMED_STEPS, peak_gb


def fit_max_batch(model_name: str, res: int, grad_ckpt: bool, start_bs: int, device, criterion):
    """Halves start_bs until a full timed run fits. Returns dict with results, or an
    'error' key if even bs=1 doesn't fit."""
    bs = start_bs
    while bs >= 1:
        model = build_model(model_name, pretrained=False).to(device)
        ckpt_applied = False
        if grad_ckpt:
            if hasattr(model, "set_grad_checkpointing"):
                model.set_grad_checkpointing(True)
                ckpt_applied = True
            else:
                print(f"[dryrun] WARNING: {model_name} has no set_grad_checkpointing")
        try:
            s_per_it, peak_gb = time_steps(model, bs, res, device, criterion)
            del model
            gc.collect()
            torch.cuda.empty_cache()
            return {
                "model": model_name, "res": res, "grad_ckpt": grad_ckpt,
                "ckpt_applied": ckpt_applied, "start_bs": start_bs, "fit_bs": bs,
                "s_per_it": s_per_it, "peak_gb": peak_gb, "halved": bs != start_bs,
            }
        except torch.cuda.OutOfMemoryError:
            print(f"[dryrun] OOM: model={model_name} res={res} ckpt={grad_ckpt} bs={bs} -- halving")
            del model
            gc.collect()
            torch.cuda.empty_cache()
            bs //= 2
    return {
        "model": model_name, "res": res, "grad_ckpt": grad_ckpt,
        "ckpt_applied": grad_ckpt, "start_bs": start_bs, "fit_bs": 0,
        "s_per_it": None, "peak_gb": None, "halved": True, "error": "OOM even at bs=1",
    }


def tta_sanity_check(model_name: str, device) -> list[str]:
    notes = []
    model = build_model(model_name, pretrained=False).to(device).eval()
    for canvas, scales in TTA_SCALES.items():
        for res in scales:
            try:
                with torch.no_grad():
                    x = torch.randn(2, 3, res, res, device=device)
                    out = model(x)
                ok = out.shape == (2, 1)
                notes.append(f"{model_name}: TTA scale {res}px (canvas {canvas}) -> logits {tuple(out.shape)} {'OK' if ok else 'SHAPE MISMATCH'}")
            except Exception as e:
                notes.append(f"{model_name}: TTA scale {res}px (canvas {canvas}) FAILED: {e!r}")
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return notes


def df_to_md(rows: list[dict], cols: list[str], float_fmt: str = "{:.4f}") -> str:
    def fmt(v):
        if v is None:
            return "-"
        if isinstance(v, float):
            return float_fmt.format(v)
        return str(v)
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    body = "\n".join(
        "| " + " | ".join(fmt(r.get(c)) for c in cols) + " |" for r in rows
    )
    return "\n".join([header, sep, body])


def main() -> None:
    assert torch.cuda.is_available(), "this dry-run must run on a CUDA GPU (the VESSL A100 box)"
    device = torch.device("cuda")
    print(f"[dryrun] device: {torch.cuda.get_device_name(device)}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    start_bs, val_fraction = read_finetune_v0_config()
    train_n = round(69_352 * (1 - val_fraction))
    print(f"[dryrun] finetune_v0 batch_size={start_bs}, val_fraction={val_fraction}, "
          f"projected train_n={train_n}")

    tok = token_counts()
    criterion = nn.BCEWithLogitsLoss()

    # --- pos-embed interpolation check (pretrained=True, once per model) ---
    pos_embed_notes: list[str] = []
    for model_name in MODELS:
        pos_embed_notes.extend(check_pos_embed_interpolation(model_name, device))

    # --- main timing matrix ---
    results = []
    for model_name in MODELS:
        for res in RESOLUTIONS:
            for grad_ckpt in GRAD_CKPT_OPTIONS:
                print(f"[dryrun] === {model_name} @ {res}px, grad_ckpt={grad_ckpt} ===")
                r = fit_max_batch(model_name, res, grad_ckpt, start_bs, device, criterion)
                r["patch_tokens"], r["total_tokens"] = tok[res]
                if r.get("s_per_it") is not None:
                    steps_per_epoch = -(-train_n // r["fit_bs"])  # ceil
                    total_steps = steps_per_epoch * EPOCHS
                    r["proj_hours_20ep"] = total_steps * r["s_per_it"] / 3600
                else:
                    r["proj_hours_20ep"] = None
                results.append(r)
                print(f"[dryrun]   -> fit_bs={r['fit_bs']}, s_per_it={r['s_per_it']}, "
                      f"peak_gb={r['peak_gb']}, proj_h={r.get('proj_hours_20ep')}")

    # --- TTA sanity ---
    tta_notes: list[str] = []
    for model_name in MODELS:
        tta_notes.extend(tta_sanity_check(model_name, device))

    # --- quadratic-scaling check: normalize by batch size (time per sample) ---
    scaling_notes = []
    for model_name in MODELS:
        by_res = {r["res"]: r for r in results if r["model"] == model_name and not r["grad_ckpt"]}
        base = by_res.get(518)
        if base and base.get("s_per_it"):
            base_per_sample = base["s_per_it"] / base["fit_bs"]
            for res in (784, 1036):
                cell = by_res.get(res)
                if cell and cell.get("s_per_it"):
                    per_sample = cell["s_per_it"] / cell["fit_bs"]
                    observed_ratio = per_sample / base_per_sample
                    token_ratio = tok[res][1] / tok[518][1]
                    quad_pred = token_ratio ** 2
                    lin_pred = token_ratio
                    scaling_notes.append(
                        f"{model_name}: 518->{ res}px token_ratio={token_ratio:.2f}, "
                        f"observed per-sample time ratio={observed_ratio:.2f} "
                        f"(linear-in-tokens predicts {lin_pred:.2f}x, pure-quadratic predicts {quad_pred:.2f}x)"
                    )

    # --- write report ---
    lines = ["# ViT resolution/grad-checkpointing GPU feasibility dry-run\n"]
    lines.append(f"GPU: {torch.cuda.get_device_name(device)}. "
                 f"finetune_v0 batch_size={start_bs}, val_fraction={val_fraction}, "
                 f"projected train_n={train_n} (69,352 total).\n")

    lines.append("## Token counts (patch size 14)\n")
    lines.append("| resolution | side | patch tokens | +CLS+4 reg = total |")
    lines.append("| --- | --- | --- | --- |")
    for res in RESOLUTIONS:
        pt, tt = tok[res]
        lines.append(f"| {res}px | {res // PATCH}x{res // PATCH} | {pt} | {tt} |")
    lines.append("")

    lines.append("## Pos-embed interpolation check (pretrained=True, native=518px)\n")
    lines.extend(f"- {n}" for n in pos_embed_notes)
    lines.append("")

    lines.append("## Timing / memory matrix\n")
    cols = ["model", "res", "grad_ckpt", "ckpt_applied", "start_bs", "fit_bs", "halved",
            "s_per_it", "peak_gb", "proj_hours_20ep"]
    lines.append(df_to_md(results, cols))
    lines.append("")

    lines.append("## Token-count vs time scaling (grad_ckpt=off, normalized per-sample)\n")
    lines.extend(f"- {n}" for n in scaling_notes)
    lines.append("")

    lines.append("## TTA sanity forward passes\n")
    lines.extend(f"- {n}" for n in tta_notes)
    lines.append("")

    # ckpt-required flag: cases where grad_ckpt=off couldn't reach bs>=8 but ckpt=on could
    ckpt_required_lines = []
    for model_name in MODELS:
        for res in RESOLUTIONS:
            off = next((r for r in results if r["model"] == model_name and r["res"] == res and not r["grad_ckpt"]), None)
            on = next((r for r in results if r["model"] == model_name and r["res"] == res and r["grad_ckpt"]), None)
            off_bs = off["fit_bs"] if off else 0
            on_bs = on["fit_bs"] if on else 0
            required = off_bs < 8 and on_bs >= 8
            ckpt_required_lines.append(
                f"- {model_name} @ {res}px: off fit_bs={off_bs}, on fit_bs={on_bs} "
                f"-> ckpt required for bs>=8: {'YES' if required else 'no'}"
            )
    lines.append("## Checkpointing-required flags (bs >= 8 threshold)\n")
    lines.extend(ckpt_required_lines)
    lines.append("")

    def verdict_for(model_name: str, res: int) -> str:
        off = next((r for r in results if r["model"] == model_name and r["res"] == res and not r["grad_ckpt"]), None)
        on = next((r for r in results if r["model"] == model_name and r["res"] == res and r["grad_ckpt"]), None)
        best = off if (off and off.get("s_per_it") is not None) else on
        if best is None or best.get("s_per_it") is None:
            return f"VERDICT: {model_name.split('_')[1].upper()[0]}@{res} = DOES NOT FIT even at bs=1"
        tag = "B" if "base" in model_name else "L"
        ckpt_req = (off is None or off["fit_bs"] < 8) and (on is not None and on["fit_bs"] >= 8)
        return (f"VERDICT: {tag}@{res} = {best['s_per_it']:.3f} s/it, {best['peak_gb']:.1f} GB, "
                f"~{best['proj_hours_20ep']:.1f}h per 20-epoch run (fit_bs={best['fit_bs']}, "
                f"grad_ckpt={'on' if best is on else 'off'}), ckpt required: {'y' if ckpt_req else 'n'}")

    lines.append("## Verdicts\n")
    for model_name, res in [
        (MODELS[0], 784), (MODELS[1], 518), (MODELS[1], 784), (MODELS[0], 1036),
    ]:
        lines.append(verdict_for(model_name, res))
    lines.append("")

    lines.append("## Recommendation\n")
    lines.append(
        "_Fill in manually after reviewing the matrix above: which 1-2 cells fit a <=16h "
        "budget and which fit a <=40h budget, given the projected 20-epoch wall-clock times._\n"
    )

    report_path = OUT_DIR / "vit_dryrun.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[dryrun] wrote {report_path}")


if __name__ == "__main__":
    main()
