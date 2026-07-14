# Model weights (not in git)

The Docker build bakes in the trained checkpoint from **this directory**:

```
docker/model/synth_tamper_v1_last.pt
```

It is **git-ignored** (≈360 MB; exceeds GitHub's file limit) and distributed with the
built image / via the release asset noted in the technical report. Before building:

```bash
cp checkpoints/synth_tamper_v1_last.pt docker/model/synth_tamper_v1_last.pt
docker build -t freuid-submission .
```

`synth_tamper_v1_last.pt` is the **last epoch (25)** of the run defined by
`configs/synth_tamper_v1.yaml` — the canonical winning recipe. The checkpoint stores
its full config, so inference rebuilds the exact model and preprocessing.
