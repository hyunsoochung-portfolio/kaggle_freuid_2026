# Per-slice metrics -- finetune_v0

Val set: n=6935, fraud_rate=0.4232 (plain eval transform, image_size=518, no TTA, no recapture degradation)

**Overall**: AuDET=0.000000  APCER@1%BPCER=0.000000

> **Val circularity**: val positives are NOT synthetic: train.py's build_loaders() only wraps train_ds in SynthTamperWrapper (synth_tamper_prob > 0); val_ds is always a plain FreuidDataset built directly from train_labels.csv. Every label=1 val sample is a genuine, dataset-provided fraud example -- val is not circular with respect to synth_tamper.

> **Attack-type slicing**: train_labels.csv has no attack/fraud-type column (columns: id, image_path, label, is_digital, type) -- a true per-attack-type (physical tamper / GenAI edit / print-capture) slice is not derivable from metadata alone. is_digital is reported as the closest available proxy for attack channel.

## By document type

| type | n | n_fraud | audet | apcer_at_1pct_bpcer | note |
| --- | --- | --- | --- | --- | --- |
| EGYPT/DL | 1587 | 787 | 0.000000 | 0.000000 |  |
| GUINEA/DL | 1339 | 539 | 0.000000 | 0.000000 |  |
| BENIN/DL | 1337 | 537 | 0.000000 | 0.000000 |  |
| MAURITIUS/ID | 1336 | 536 | 0.000000 | 0.000000 |  |
| MOZAMBIQUE/DL | 1336 | 536 | 0.000000 | 0.000000 |  |

## By is_digital (attack-channel proxy)

| is_digital | n | n_fraud | audet | apcer_at_1pct_bpcer | note |
| --- | --- | --- | --- | --- | --- |
| True | 6933 | 2933 | 0.000000 | 0.000000 |  |
| False | 2 | 2 | nan | nan | single-class slice, AuDET undefined |

