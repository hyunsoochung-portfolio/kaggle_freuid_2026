# FREUID Challenge 2026 — Competition Brief

**1st Competition on Identity Documents Fraud Detection** · IJCAI-ECAI 2026 (Bremen, Germany)
Organized by **Microblink** (industrial organizer — dataset & prize pool) with academic partners
**UniZG FER** and **Politecnico di Torino (DBDM)**.

- Kaggle: <https://www.kaggle.com/competitions/the-freuid-challenge-2026-ijcai-ecai>
- Official site: <https://freuid2026.microblink.com/>
- Contact: freuid-challenge-2026@microblink.com · Discord: <https://discord.gg/8hKbNEKnT>

---

## 1. Motivation

- $47B lost by US adults to identity fraud in 2024 (+$4B YoY).
- +244% YoY surge in AI-driven digital document forgeries; deepfake identity-theft incidents now occur every ~5 min.
- Existing benchmarks have saturated on purely digital artifacts and underrepresent global document
  diversity, leaving production verification systems exposed to next-generation, GenAI-era attacks.

## 2. Task

Decide whether an identity-document image is **bona-fide** or **fraudulent** — a single detection task over a
realistic threat surface that mixes three attack vectors. Three open research problems drive it:

- **Cross-domain generalization** — perform accurately on highly underrepresented document types.
- **Physical vs. digital artifacts** — move beyond fragile GenAI pixel noise toward semantic inconsistencies
  in physically printed/captured forgeries.
- **Anti-fragility** — adapt to open-ended, evolving fraud strategies instead of overfitting to known attacks.

> Convention used in this repo: **label 1 = fraud/attack, 0 = bona-fide**; the model emits a continuous
> **fraud score** = P(fraud). Calibration matters because the metrics integrate over operating points.

## 3. Fraud types

| Type | Description |
|------|-------------|
| **Physical manipulations** | Real document substrates physically tampered, then captured digitally. |
| **GenAI multimodal edits** | Forgeries made with accessible text+image generative-AI editing tools. |
| **Print-and-capture** | Digital forgeries printed then re-captured to erase the digital noise SOTA detectors rely on; physical forgeries are also applied on top of printed documents. |

## 4. Dataset (FREUID)

- Proprietary collection from the **Microblink Fraud Lab**.
- **7 document types** from **Asian and African** regions, diverse scripts (**Latin, Arabic**) — deliberately
  underrepresented types to force cross-domain generalization.
- Composition: **synthetic + printed/captured physical plastic cards**; high-fidelity bona-fide and fraudulent docs.
- **License: non-commercial research use only.** Commercial licensing on request. **Never commit the data**
  to this (public) repo — see `.gitignore`.
- Release: sample available on Kaggle now; **full dataset early June 2026.**
- Split: public-leaderboard **validation subset** + private **held-out test set** for final ranking
  (exact percentages not yet published).

## 5. Evaluation

- **Primary — AuDET:** area under the Detection Error Trade-off (DET) curve; one scalar capturing the
  false-accept ↔ false-reject trade-off across operating points. **Lower is better.**
- **Operating point — APCER @ 1% BPCER:** attack pass-rate when bona-fide rejection (BPCER) is fixed at 1%
  — the production-relevant slice of the DET curve.
- Terms: **APCER** = Attack Presentation Classification Error Rate (fraud accepted as genuine);
  **BPCER** = Bona-Fide Presentation Classification Error Rate (genuine rejected as fraud).
- Leaderboards: **public** (validation subset, updates every submission) and **private** (held-out test, final).

Both metrics are implemented locally in `src/freuid/metrics.py` so we can rank candidates before spending submissions.

## 6. Timeline (23:59 AoE, UTC-12)

- Full dataset release: **early June 2026**
- Training phase: June–August 2026
- Workshops & tutorials: **Aug 15–17, 2026** (University of Bremen)
- Main conference + **live award showdown: Aug 18–21, 2026** (Bremen Exhibition Center)
- ⚠️ Exact milestone deadlines (submission close, team-merge) are **not yet published** — expected with the
  June data release. (A "June 1" date seen in search refers to the data release, not a submission deadline.)

## 7. Prizes

Total **$6,000** (Microblink): **1st $3,000 · 2nd $2,000 · 3rd $1,000**, plus conference entries for top
performers (subject to organizer confirmation).

## 8. Rules

**Eligibility & teams**
- Open to academia and industry worldwide; ~20 teams expected.
- Microblink employees / organizers / their immediate research groups are prize-ineligible (may participate informally).
- Up to **5 members per team**, **one team per person**; cross-affiliation teams encouraged.

**External data & models**
- Any architecture allowed. Any **public external data** and any **public pre-trained model** permitted,
  provided license compatibility and full citation in the technical report.
- **Proprietary/non-freely-accessible data is prohibited.**

**Submission**
- Predictions generated on your own infrastructure (no GPU quota). Final ranking uses the published test set.
- Exact submission format/flow is **to be announced.**

**Code & report (mandatory for ranked/prize eligibility)**
- Provide source code + all configuration, training, and inference scripts + instructions sufficient for
  organizers to **reasonably reproduce** the ranked result.
- Code must be released under an **OSI-approved open-source license** (this repo: Apache-2.0).
- Submit a short **technical report** (see `report/`).
- Top-3 teams strongly encouraged to attend the Aug 18–21 showdown.

## 9. Still TBD (confirm at June data release)

1. Exact submission deadline / team-merge deadline.
2. Train/val/test split ratios and on-disk file structure.
3. `sample_submission.csv` exact columns.

---

*Sources: [official site](https://freuid2026.microblink.com/), [Kaggle competition](https://www.kaggle.com/competitions/the-freuid-challenge-2026-ijcai-ecai), [IJCAI 2026 competitions](https://2026.ijcai.org/competitions/).*
