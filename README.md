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
before spending submissions. Note: the local AuDET is a **linear-axis proxy (`1 − ROC AUC`)** —
treat it as a relative ranking signal and reconcile with the official Kaggle scorer before
trusting absolute values (see the caveat in `metrics.py`).

> Convention used throughout: **label 1 = fraud/attack, 0 = bona-fide**; score = P(fraud).

## Key dates

- Full dataset release: **June 2026** (released — data is live on Kaggle)
- **Submission deadline: 2026-07-14** — per the Kaggle competition page; confirm the exact
  time/zone there before the final push.
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

# 2. Sync the pinned environment (creates .venv with all deps, incl. the kaggle CLI)
uv sync
```

Then get the data — see the next section.

## Getting the data

The FREUID dataset is **proprietary, non-commercial, and gitignored** — it is never committed
to this repo. Every team member downloads their **own** copy from Kaggle. Three one-time steps,
then a single command.

### Step 1 — Kaggle account + accept the rules (in the browser)

1. Sign in (or sign up) at <https://www.kaggle.com>.
2. Open the competition:
   <https://www.kaggle.com/competitions/the-freuid-challenge-2026-ijcai-ecai>
3. Click **"Join Competition"** → on the **Rules** tab, **"I Understand and Accept"**.

> ⚠️ Without this, every download fails with **403 Forbidden**. The CLI *cannot* accept the
> rules for you — this step is manual and per-account.

### Step 2 — Authenticate the Kaggle CLI (per machine)

Credentials live **outside** this repo (`~/.kaggle/`) — never commit or paste them anywhere.
Your access token is a personal secret; do not share it or hardcode it. Pick one option:

**Option A — browser login (recommended):**
```bash
uv run kaggle auth login      # opens a browser, stores a token at ~/.kaggle/access_token
uv run kaggle config view     # verify — should print your Kaggle username
```

**Option B — write your own token manually** (get it from `kaggle auth print-access-token`
on a machine that's already logged in, or from your Kaggle account settings):
```bash
mkdir -p ~/.kaggle
echo "<YOUR_KAGGLE_ACCESS_TOKEN>" > ~/.kaggle/access_token   # your own token, not a teammate's
chmod 600 ~/.kaggle/access_token
```

**Option C — legacy API key:** at <https://www.kaggle.com/settings> → **Create New Token**,
download `kaggle.json`, then:
```bash
mkdir -p ~/.kaggle && mv ~/Downloads/kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json
```

### Step 3 — Download

```bash
bash scripts/download_data.sh     # downloads the competition archive into ./data and unzips it
```

Expected result (current Kaggle layout — note the double-nested image folders):

```
data/
├── train_labels.csv               # id, image_path, label, is_digital, type
├── train/train/<id>.jpeg          # training images
├── public_test/public_test/<id>.jpeg
├── sample_submission.csv          # id, label  (label = your fraud score in [0,1], all test ids)
└── train_sample/train_sample/...  # tiny labelled sample for smoke tests
```

> **Troubleshooting** — `403 Forbidden`: you skipped Step 1 (accept the rules).
> `401 Unauthorized` / no token: redo Step 2 (`uv run kaggle auth login`).

## Layout

```
configs/            # experiment configs (yaml) — one per run, committed
data/               # competition data — GITIGNORED, never commit (proprietary, non-commercial)
notebooks/          # exploration; clear outputs before committing
report/             # technical report (mandatory deliverable)
scripts/            # data download & helper scripts
src/freuid/
├── config.py       # config loading (one yaml per experiment)
├── data.py         # dataset / dataloaders (resolves the double-nested layout)
├── transforms.py   # image transforms / augmentation
├── metrics.py      # AuDET + APCER@BPCER (offline ranking)
├── models/         # model definitions (baseline.py: timm backbone + 1 fraud logit)
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

**→ Full end-to-end guide** (train → infer → submit, what goes in each directory, how to add
experiments/models, reproducibility checklist): [docs/workflow.md](docs/workflow.md).

## License

Apache-2.0 — see [LICENSE](LICENSE). Required by the competition; also gives a patent grant.
