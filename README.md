# kaggle-freuid-2026

Team workspace for **[The FREUID Challenge 2026](https://www.kaggle.com/competitions/the-freuid-challenge-2026-ijcai-ecai/overview)**
— 1st Competition on Identity Documents Fraud Detection, **IJCAI-ECAI 2026** (Bremen, Aug 2026).

## The task

Binary detection: given an identity-document image, decide **bona-fide vs fraudulent**.
Fraud spans three modalities:

1. **Physical manipulations** on real document substrates
2. **GenAI-driven digital edits** (multimodal)
3. **Print-and-capture forgeries** (the "analog hole")

Output a continuous **fraud score** per image (higher = more likely fraud). The score
feeds DET-curve metrics, so calibration of the score matters, not just a hard label.

## Evaluation

- **Primary: AuDET** — area under the Detection Error Trade-off curve (APCER vs BPCER). **Lower is better.**
- **Secondary: APCER @ 1% BPCER** — attack pass-rate when only 1% of genuine docs are wrongly rejected.

Public leaderboard = validation subset; **final ranking = private held-out test set.**
Local implementations live in `src/freuid/metrics.py` so we can rank candidates offline
before spending submissions.

> Convention used throughout: **label 1 = fraud/attack, 0 = bona-fide**; score = P(fraud).

## Key dates (AoE)

- Full dataset release: **June 2026**
- Training phase: June–August 2026
- Workshops/tutorials: **Aug 15–17, 2026**
- Live award showdown: **Aug 18–21, 2026** (Bremen Exhibition Center)

## Winner obligations (drives this repo's design)

To stay eligible for prizes a team must ship, alongside final predictions:

- **Source code under an OSI-approved open-source license** (this repo: Apache-2.0).
- **All config + training + inference scripts**, reproducible enough for organizers to
  re-run the ranked result.
- A **technical report** (see `report/`).

So: deterministic seeds, pinned environment, config-driven runs, and a clean
`train → infer → submission` path are requirements, not nice-to-haves.

## Setup

```bash
# 1. Install uv (https://docs.astral.sh/uv/)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Sync the pinned environment
uv sync

# 3. Configure Kaggle API (put kaggle.json in ~/.config/kaggle/ — NOT in this repo)
#    https://www.kaggle.com/settings  →  Create New Token

# 4. Download competition data into ./data (gitignored)
bash scripts/download_data.sh
```

## Layout

```
configs/            # experiment configs (yaml) — one per run, committed
data/               # competition data — GITIGNORED, never commit (proprietary, non-commercial)
notebooks/          # exploration; clear outputs before committing
report/             # technical report (mandatory deliverable)
scripts/            # data download & helper scripts
src/freuid/
├── config.py       # config loading
├── data.py         # dataset / dataloaders
├── metrics.py      # AuDET + APCER@BPCER (offline ranking)
├── models/         # model definitions
├── train.py        # training entrypoint  (uv run python -m freuid.train --config ...)
├── infer.py        # inference → submission csv
└── utils.py        # seeding, reproducibility helpers
submissions/        # generated submission csvs — GITIGNORED
```

## Team workflow

- `main` stays runnable. Work on `feat/<name>` branches → PR → review → merge.
- One experiment = one `configs/*.yaml`. Log results in `report/experiments.md`.
- Gate every idea on the **local AuDET / APCER@1%BPCER** before burning a Kaggle submission.
- Merge Kaggle teams **before the team-merge deadline** (submission quota becomes shared).

## License

Apache-2.0 — see [LICENSE](LICENSE). Required by the competition; also gives a patent grant.
