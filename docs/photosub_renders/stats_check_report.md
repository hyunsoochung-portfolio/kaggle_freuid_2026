# MODE_C/D statistical self-check: natural-baseline pass rate

Design intent (see freuid.photosub.generators' module docstring): MODE_C/D evidence should be SEMANTIC (a different person, a cross-region mismatch), not a local pixel-statistics anomaly. Eyeballing render sheets can't confirm that, so this replaces it with an empirical acceptance criterion (freuid.photosub.baseline_stats): the SAME 3 cheap stats (freuid.photosub.stats -- blur_laplacian_var, moire_fft_score, blockiness_score) measured between two random frame-sized regions on 200 UNTOUCHED bona-fide cards give the natural intra-card spread; a MODE_C/D composite PASSES when its tampered-vs-original delta, for every stat, falls within that natural distribution's 90th percentile.

## p90 thresholds (natural intra-card baseline)

- `blur_laplacian_var`: 1207.398
- `moire_fft_score`: 34.844
- `blockiness_score`: 1.687

## Overall pass rate: 91.7% (11/12)

At or above the ~90% bar -- composites are not, in aggregate, local-statistics outliers relative to how much two random regions on an ordinary untouched card naturally differ.

## Per-stat pass rate

- `blur_laplacian_var`: 100.0% within threshold
- `moire_fft_score`: 91.7% within threshold
- `blockiness_score`: 100.0% within threshold

## Per-example detail

| mode | type | hard | passed | delta_blur_laplacian_var | delta_moire_fft_score | delta_blockiness_score |
| --- | --- | --- | --- | --- | --- | --- |
| C | EGYPT/DL | True | True | 23.489000 | 16.052000 | 0.344000 |
| C | GUINEA/DL | True | True | 180.186000 | 18.387000 | 0.015000 |
| C | BENIN/DL | True | True | 21.029000 | 17.467000 | 0.118000 |
| C | MOZAMBIQUE/DL | True | True | 187.502000 | 7.361000 | 0.382000 |
| C | MAURITIUS/ID | True | False | 183.599000 | 36.417000 | 0.170000 |
| C | EGYPT/DL | True | True | 21.352000 | 23.471000 | 0.382000 |
| C | GUINEA/DL | False | True | 3.586000 | 23.262000 | 0.018000 |
| C | BENIN/DL | False | True | 16.543000 | 20.560000 | 0.156000 |
| C | MOZAMBIQUE/DL | False | True | 10.181000 | 1.433000 | 0.291000 |
| C | MAURITIUS/ID | False | True | 120.817000 | 17.990000 | 0.012000 |
| C | EGYPT/DL | False | True | 16.538000 | 18.914000 | 0.322000 |
| C | GUINEA/DL | False | True | 186.382000 | 21.873000 | 0.029000 |

