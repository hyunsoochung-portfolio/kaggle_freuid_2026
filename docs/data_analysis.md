# Training data summary — `data/raw/train_labels.csv`

69,352 rows total. Read-only analysis, computed directly from the CSV (`id, image_path, label,
is_digital, type`).

## Label distribution

| label | meaning   | count  | share |
| ----- | --------- | ------ | ----- |
| 0     | bona-fide | 40,005 | 57.7% |
| 1     | fraud     | 29,347 | 42.3% |

## Digital vs. non-digital

| is_digital | count  | share  |
| ---------- | ------ | ------ |
| True       | 69,332 | 99.97% |
| False      | 20     | 0.03%  |

Matches CLAUDE.md's "~99.97% digital" characterization almost exactly.

### Non-digital breakdown (the analog-hole case)

| label     | non-digital count | share of that label's total |
| --------- | ------------------ | ---------------------------- |
| fraud (1) | 14                  | 0.048% of 29,347 frauds      |
| bona-fide (0) | 6               | 0.015% of 40,005 bona-fide   |

Real print-and-capture examples are vanishingly rare in training — only 14 non-digital fraud
images and 6 non-digital bona-fide images exist in the entire 69,352-row set. This is why the
recapture augmentation pipeline (`src/freuid/augment.py`) is doing essentially all of the work
of teaching the model about the analog-hole case; there is almost no real signal to learn it
from directly.

## Document types

5 types total, identified by the `type` column (`COUNTRY/DOCTYPE`):

| type          | total  | bona-fide | fraud | fraud rate |
| ------------- | ------ | --------- | ----- | ---------- |
| EGYPT/DL      | 15,867 | 8,000     | 7,867 | 49.6%      |
| GUINEA/DL     | 13,389 | 8,001     | 5,388 | 40.2%      |
| BENIN/DL      | 13,369 | 8,001     | 5,368 | 40.2%      |
| MOZAMBIQUE/DL | 13,365 | 8,001     | 5,364 | 40.1%      |
| MAURITIUS/ID  | 13,362 | 8,002     | 5,360 | 40.1%      |

EGYPT/DL is the only near-50/50 type and has ~2,500 more rows than the others; the other four
types are consistently ~40% fraud with nearly identical bona-fide counts (8,001-8,002 each),
suggesting a deliberately balanced generation scheme per type, with EGYPT/DL having extra fraud
examples layered on top.
