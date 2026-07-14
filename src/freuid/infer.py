"""Inference -> Kaggle submission csv.

    uv run python -m freuid.infer --checkpoint checkpoints/baseline.pt \
        --out submissions/baseline.csv

Model-defining params (``backbone``, ``image_size``) are read from the checkpoint itself
so preprocessing + architecture always match the trained weights -- no way to silently
mismatch them. ``--config`` is optional and only supplies runtime/environment params; the
CLI flags ``--data-dir`` / ``--batch-size`` / ``--num-workers`` override those per machine.

Submission format (from sample_submission.csv): columns ``id,label`` where label is the
predicted fraud score in [0, 1] (the DET metrics need a continuous score, not a hard 0/1).

This is a code competition: sample_submission.csv lists the FULL test set (~142.8k ids) but
only the public subset (~7.8k) of images ships in the download. We score every id whose image
is present locally and default the rest to ``extra.missing_id_score`` (default 0.5 -- never
0.0, which would silently tank AuDET if any genuinely-absent id happens to be fraud). On
Kaggle's grading run all images are present so every id gets a real score.
"""

from __future__ import annotations

import argparse
from dataclasses import fields
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from freuid.config import Config, load_config
from freuid.data import FreuidDataset, load_labels, unpack_batch
from freuid.models import build_model
from freuid.transforms import build_transforms, resolve_data_config
from freuid.utils import pick_device, seed_everything

ID_COLUMN = "id"
SCORE_COLUMN = "label"
_MISSING_FALLBACK = 0.5  # module-level fallback; overridden by extra.missing_id_score


# @torch.no_grad(): 이 함수 안에서는 autograd(자동미분)를 끈다. 추론만 할 때는 기울기가
# 필요 없으므로 계산 그래프를 안 만들어 메모리를 아끼고 속도를 높인다.
@torch.no_grad()
def predict_scores(model, loader, device) -> list[float]:
    """Fraud scores in dataset order (loader must be shuffle=False)."""
    scores: list[float] = []
    # leave=False: tqdm 진행바를 다 돌면 지운다(TTA로 여러 번 돌 때 화면이 안 지저분해짐).
    for batch in tqdm(loader, leave=False):
        # face_meta는 얼굴 위치 같은 부가 입력. 모델 종류에 따라 있을 수도(consistency) 없을 수도.
        imgs, _, face_meta = unpack_batch(batch)
        imgs = imgs.to(device)
        # face_meta가 있으면 2입력, 없으면 1입력으로 forward. logits는 [B, 1] 원시 점수(확률 아님).
        logits = model(imgs, face_meta.to(device)) if face_meta is not None else model(imgs)
        # sigmoid로 logit을 확률(0~1)로 변환. squeeze(1)로 [B,1]->[B], cpu로 옮겨 리스트화.
        scores.extend(torch.sigmoid(logits).squeeze(1).cpu().tolist())
    return scores


def _rank_normalize(arr) -> list[float]:
    """Convert a score array to fractional ranks in (0, 1).

    Uses average rank for ties. Result is in (0, 1) — never exactly 0.0
    (guarding the submission integrity check).
    """
    import numpy as np
    a = np.asarray(arr, dtype=np.float64)
    n = len(a)
    if n == 0:
        return []
    # argsort: 값을 오름차순으로 정렬했을 때의 "원래 인덱스" 배열. kind="stable"은 같은 값의
    # 순서를 입력 순서 그대로 유지(재현성). order[k] = k번째로 작은 원소의 원래 위치.
    order = np.argsort(a, kind="stable")
    ranks = np.empty(n, dtype=np.float64)
    # order 위치에 1..n을 뿌리면 각 원소의 순위(작을수록 1)가 원래 인덱스 자리에 들어간다.
    ranks[order] = np.arange(1, n + 1)
    # average ties: find runs of equal values and replace their ranks with the mean
    # 동점 처리: 값이 같은 원소들은 순위를 평균으로 통일한다. 안 그러면 우연한 정렬 순서에 따라
    # 동점끼리도 순위가 달라져 rank-average 결과가 흔들린다.
    sorted_a = a[order]
    i = 0
    while i < n:
        j = i + 1
        while j < n and sorted_a[j] == sorted_a[i]:
            j += 1
        if j > i + 1:
            avg = ranks[order[i:j]].mean()
            ranks[order[i:j]] = avg
        i = j
    # map to (epsilon, 1-epsilon) so no exact zeros reach the integrity check
    # 순위를 (1e-7, 1-1e-7) 범위로 min-max 스케일. 정확히 0.0을 피하는 이유는 모듈 docstring 참고:
    # 점수 0.0인 id가 실제로 fraud라면 AuDET(=1-ROC AUC)가 크게 나빠지므로 0을 원천 차단한다.
    lo, hi = ranks.min(), ranks.max()
    if hi > lo:
        ranks = (ranks - lo) / (hi - lo) * (1 - 2e-7) + 1e-7
    return ranks.tolist()


def predict_scores_tta(
    model,
    device,
    data_dir: str,
    present_ids: set,
    batch_size: int,
    num_workers: int,
    mean,
    std,
    scales: list[int],
    regions_dir: Path | None = None,
    return_face_meta: bool = False,
) -> list[tuple[str, float]]:
    """Multi-scale TTA: run inference at each scale, rank-average, return (id, score) pairs.

    No horizontal flip — documents carry orientation.
    Rank-averaging is used instead of score-averaging because AuDET is a rank
    metric; averaging ranks is invariant to per-scale score calibration differences.
    """
    # TTA(Test-Time Augmentation): 학습이 아니라 "추론할 때" 이미지를 여러 방식으로 변형해
    # 각각 예측한 뒤 합치는 기법. 여기서는 여러 해상도(scale)로 돌려 예측을 안정화한다.
    # 좌우 뒤집기(flip)는 안 쓴다 — 신분증 문서는 방향/글자 순서가 의미를 가지기 때문.
    per_scale_scores: list[list[float]] = []
    sample_ids: list[str] | None = None

    for scale in scales:
        tf = build_transforms(scale, False, mean, std)
        ds = FreuidDataset(
            data_dir, "public_test", tf, ids=present_ids, regions_dir=regions_dir,
            return_face_meta=return_face_meta,
        )
        # shuffle=False 필수: 아래에서 sample_ids와 scores를 "순서"로 짝짓기 때문에 순서가 고정돼야.
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        # id 순서는 scale마다 동일(같은 present_ids)하므로 첫 scale에서 한 번만 기록해 재사용한다.
        if sample_ids is None:
            sample_ids = [s.id for s in ds.samples]
        scores = predict_scores(model, loader, device)
        per_scale_scores.append(scores)
        # guard min()/max() on empty scores (no test images present locally) — the rest of the
        # TTA path already handles empty gracefully, so every id just falls back to missing_id_score
        if scores:
            print(f"[tta] scale={scale}  scores: min={min(scores):.4f} max={max(scores):.4f}")
        else:
            print(f"[tta] scale={scale}  0 present ids")

    # Rank-average across scales
    # 각 scale의 점수를 "순위"로 바꾼 뒤 평균낸다. 점수 자체를 평균하지 않는 이유: scale마다 sigmoid
    # 출력의 크기(캘리브레이션)가 미묘하게 달라서다. 순위로 바꾸면 그 차이에 무관해 AuDET에 맞다.
    import numpy as np
    n = len(per_scale_scores[0])
    avg_ranks = np.zeros(n, dtype=np.float64)
    for scores in per_scale_scores:
        ranked = _rank_normalize(scores)
        avg_ranks += np.array(ranked)
    avg_ranks /= len(per_scale_scores)

    return list(zip(sample_ids or [], avg_ranks.tolist(), strict=True))


def check_submission(path: str | Path) -> None:
    """Print an integrity report for a finished submission CSV.

    Catches the most dangerous silent failure: exact-zero scores for missing ids
    (a 0.0 fraud score on a genuine fraud sample collapses AuDET).
    """
    df = pd.read_csv(path)
    scores = df[SCORE_COLUMN]
    n = len(df)
    n_zeros = int((scores == 0.0).sum())
    pct_zeros = 100.0 * n_zeros / max(n, 1)
    print(
        f"[infer] integrity: rows={n} unique_scores={scores.nunique()} "
        f"exact_zeros={n_zeros} ({pct_zeros:.2f}%) "
        f"min={scores.min():.6f} max={scores.max():.6f}"
    )
    if n_zeros > 0:
        print(
            f"[WARNING] {n_zeros} exact-zero score(s) in submission -- "
            "if any of those ids are fraud, AuDET will be severely penalised."
        )


def resolve_config(args) -> tuple[Config, dict]:
    """Build the run config and return it alongside the loaded checkpoint.

    ``backbone`` and ``image_size`` always come from the checkpoint's stored config so the
    model matches its weights. ``--config`` (optional) seeds the rest; CLI flags override
    runtime/environment params (data dir, batch size, workers).
    """
    # map_location="cpu": GPU에서 저장한 체크포인트라도 일단 CPU로 로드(어느 장비에서든 열림).
    # weights_only=False: config 같은 가중치 외 파이썬 객체까지 함께 복원하려고 끈다.
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    ckpt_cfg = state.get("config", {})

    if args.config:
        cfg = load_config(args.config)
    elif ckpt_cfg:
        # --config가 없으면 체크포인트에 저장된 config로 재구성. 단, 지금 Config에 실제로 존재하는
        # 필드만 골라 넣어(known 교집합) 옛 체크포인트의 사라진 필드로 인한 오류를 막는다.
        known = {f.name for f in fields(Config)}
        cfg = Config(**{k: v for k, v in ckpt_cfg.items() if k in known})
    else:
        raise SystemExit("checkpoint has no stored config -- pass --config explicitly")

    # Model-defining params ALWAYS come from the checkpoint (guarantees the weights match).
    # backbone(모델 구조)과 image_size(입력 크기)는 반드시 체크포인트 값으로 덮어쓴다. 이 둘이
    # 학습 때와 다르면 가중치를 못 불러오거나 전처리가 어긋나 조용히 성능이 망가지기 때문.
    if ckpt_cfg:
        cfg.backbone = ckpt_cfg.get("backbone", cfg.backbone)
        cfg.image_size = ckpt_cfg.get("image_size", cfg.image_size)

    # Runtime / environment overrides.
    if args.data_dir is not None:
        cfg.data_dir = args.data_dir
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.num_workers is not None:
        cfg.num_workers = args.num_workers
    return cfg, state


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--config", default=None,
        help="optional; backbone/image_size still come from the checkpoint",
    )
    parser.add_argument("--out", default="submissions/submission.csv")
    parser.add_argument(
        "--data-dir", default=None,
        help="override data dir (default: from checkpoint config)",
    )
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    args = parser.parse_args()

    cfg, state = resolve_config(args)
    seed_everything(cfg.seed)
    device = pick_device()

    # backbone and image_size are sourced from the checkpoint -- logged here for audit.
    print(
        f"[infer] backbone={cfg.backbone} image_size={cfg.image_size} (from checkpoint) | "
        f"device={device} | data_dir={cfg.data_dir}"
    )

    # model_type dispatch: add new model types here (e.g. model_type="consistency")
    model_type = cfg.extra.get("model_type", "baseline")
    if model_type == "consistency":
        from freuid.models import build_consistency_model
        model = build_consistency_model(cfg).to(device)
    elif model_type == "joint":
        # pretrained=False: 학습 가중치를 아래에서 덮어쓰므로 백본 다운로드 불필요.
        from freuid.models.joint import JointConsistencyModel
        model = JointConsistencyModel(
            cfg.backbone, pretrained=False,
            head_dropout=float(cfg.extra.get("head_dropout", 0.0)),
            patch_layers=int(cfg.extra.get("patch_consistency_layers", 2)),
            patch_heads=int(cfg.extra.get("patch_consistency_heads", 8)),
            patch_dropout=float(cfg.extra.get("patch_consistency_dropout", 0.1)),
        ).to(device)
    else:
        # pretrained=False: 어차피 아래에서 우리가 학습한 가중치를 덮어쓰므로 ImageNet 사전학습을
        # 내려받을 필요가 없다(추론 시작 속도 향상).
        model = build_model(
            cfg.backbone, pretrained=False,
            pool=cfg.extra.get("pool"),
            head_dropout=float(cfg.extra.get("head_dropout", 0.0)),
        ).to(device)
    model.load_state_dict(state["model"])
    # eval() 모드로 전환: dropout을 끄고 BatchNorm을 학습 중 누적한 통계로 고정. 추론에서는 필수.
    model.eval()

    # Full test-id list from sample_submission.csv; score all present, fill the rest.
    # 제출은 전체 테스트 id를 담아야 함. 로컬엔 public 이미지 일부만 있어서,
    # 있는 id만 추론하고 나머지는 아래에서 missing_score로 채운다(docstring 참고).
    missing_score = cfg.extra.get("missing_id_score", _MISSING_FALLBACK)
    submission = load_labels(cfg.data_dir, "public_test")
    # 각 id의 이미지 파일이 로컬에 실제로 존재하는지 검사해 "present"인 id만 추린다.
    present_mask = submission["path"].map(lambda p: Path(p).exists())
    present_ids = set(submission.loc[present_mask, ID_COLUMN])
    n_missing = len(submission) - len(present_ids)
    print(
        f"[infer] {len(submission)} ids total | "
        f"{len(present_ids)} images present | "
        f"{n_missing} missing (will be scored {missing_score})"
    )
    if n_missing > 0:
        print(
            f"[WARNING] {n_missing} test id(s) have no local image -- "
            f"writing missing_id_score={missing_score} for those rows. "
            "On Kaggle's grading server all images are present; this is expected locally."
        )

    # backbone별 표준 전처리 값을 가져온다: 입력 크기, 정규화용 mean/std(사전학습 때와 같은 값이어야
    # 색 분포가 맞아 성능이 나온다). timm 백본은 각자 학습에 쓴 mean/std가 다르다.
    data_cfg = resolve_data_config(cfg.backbone, cfg.image_size)
    base_size = data_cfg["image_size"]
    mean, std = data_cfg["mean"], data_cfg["std"]

    # Regions cache: used when extra.use_rectify=True (card-rectification path).
    _rdir: Path | None = None
    if cfg.extra.get("use_rectify", False):
        from freuid.preprocess import regions_dir as _get_rdir
        _rdir = _get_rdir(cfg.data_dir)
        if not _rdir.exists():
            print(f"[infer] WARNING: use_rectify=True but cache not found "
                  f"at {_rdir}; using raw images")
            _rdir = None
        else:
            print(f"[infer] use_rectify=True → loading from {_rdir}")

    return_face_meta = model_type == "consistency" and bool(cfg.extra.get("use_face_region", False))

    # extra.tta: 리스트면 그 해상도들로, True면 아래 기본 3-scale로 TTA. 없으면(False) 단일 해상도.
    tta_cfg = cfg.extra.get("tta", False)
    if tta_cfg:
        if isinstance(tta_cfg, list):
            tta_scales = [int(s) for s in tta_cfg]
        else:
            # default: ±64px around the trained resolution (stays divisible by 32)
            # & ~31 은 32의 배수로 내림(하위 5비트 제거). CNN은 보통 입력을 32배수로 다운샘플하므로
            # 32의 배수라야 크기가 딱 나눠떨어져 안전하다. 학습 해상도 기준 ±step 3개 스케일 생성.
            step = max(32, (base_size // 6) & ~31)
            tta_scales = [base_size - step, base_size, base_size + step]
        print(f"[tta] scales={tta_scales}")
        id_score_pairs = predict_scores_tta(
            model, device,
            data_dir=cfg.data_dir,
            present_ids=present_ids,
            batch_size=cfg.batch_size,
            num_workers=cfg.num_workers,
            mean=mean, std=std,
            scales=tta_scales,
            regions_dir=_rdir,
            return_face_meta=return_face_meta,
        )
        id_to_score: dict[str, float] = dict(id_score_pairs)
    else:
        transform = build_transforms(base_size, False, mean, std)
        ds = FreuidDataset(
            cfg.data_dir, "public_test", transform, ids=present_ids, regions_dir=_rdir,
            return_face_meta=return_face_meta,
        )
        loader = DataLoader(
            ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers
        )
        scores = predict_scores(model, loader, device)
        # strict=True: id 개수와 score 개수가 다르면 조용히 잘리지 않고 에러를 낸다(짝 어긋남 방지).
        id_to_score = dict(zip((s.id for s in ds.samples), scores, strict=True))

    # 전체 제출 id에 점수 매핑. 추론한 id는 실제 점수를, 이미지 없던 id는 missing_score를 채운다.
    submission[SCORE_COLUMN] = submission[ID_COLUMN].map(
        lambda i: id_to_score.get(i, missing_score)
    )
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    submission[[ID_COLUMN, SCORE_COLUMN]].to_csv(out_path, index=False)
    print(f"[infer] wrote {len(submission)} rows -> {out_path}")

    check_submission(out_path)


if __name__ == "__main__":
    main()
