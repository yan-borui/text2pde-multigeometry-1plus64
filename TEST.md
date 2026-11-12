# Ellipse Validation100 and Test100

Use the same Ellipse data directory, run root, environment and frozen dependencies as training. All 100 Validation trajectories select weights. Test uses the selected weights and performs no selection.

```bash
export RAW_DATA_DIR=/data/datasets/megaflow2d
export DATA_DIR=/data/datasets/megaflow2d_ellipse_stride15
export CUDA_VISIBLE_DEVICES=0,1,2,3
export RESULT_ROOT=/data/runs/ellipse_text2pde_seed0
bash scripts/ellipse_4gpu.sh validate
bash scripts/ellipse_4gpu.sh test
```

Set AUTOENCODER/ARTIFACTS_DIR for GLaDiT as in training. AROMA and Text2PDE automatically use their matching selected AE. Set EVAL_DIR for a new evaluation output location. The default output directories are `validation_ellipse_4gpu` and `test_ellipse_4gpu` under RESULT_ROOT.

Predictions use physical UVP `[65,N,3]` and the original high-resolution mesh. Only future64 enters forecast scores. Source frames are 1,16,…,961 and relative times are 0–9.6 s. VGAE output is explicitly reconstruction, with its selection loss retained. Four ranks partition cases without padding; merges verify complete unique coverage. Failures remain in reports and must accompany reported metrics. Test use must be recorded if its results guide later research choices.

The full conversion and four-GPU execution remain to be run in the production environment. See [ELLIPSE.md](ELLIPSE.md) for the complete protocol and method paths.

## Fixed-weight member and mean-field Test evaluation

See [Ellipse sampling and matched cost](ELLIPSE.md#ellipse-sampling-and-matched-cost)
for `SPLIT=test bash scripts/ellipse_sampling.sh quality`. The command evaluates
the existing selected checkpoint on all Test100 cases, retains individual draws
and scores decoded physical-field means. It does not overwrite native main-table
results or reselect weights. Use `SPLIT=validation` for the separate timing action.
All new entrypoints are locally prepared and statically checked; runtime results
are not part of this delivery.
