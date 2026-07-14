# Model weights (not in git)

The Docker build bakes in the trained checkpoint from **this directory**:

```
docker/model/synth_recapture_v1_ep11.pt
```

It is **git-ignored** (≈360 MB; exceeds GitHub's file limit) and distributed with the
built image / as a release asset (see the technical report). Before building:

```bash
cp checkpoints/synth_recapture_v1_ep11.pt docker/model/synth_recapture_v1_ep11.pt
docker build -t freuid-repro:local .
```

`synth_recapture_v1_ep11.pt` is **epoch 11** of the run defined by
`configs/synth_recapture_v1.yaml` — the checkpoint that produced our ranked Kaggle
submission `synth_recapture_v1_ep11.csv`. It stores its full config, so inference
rebuilds the exact model and preprocessing.
