# Text2PDE on MegaFlow2D Ellipse

This branch implements the original high-resolution mesh UVP task: file frames 1,16,…,961, one observation and 64 predictions, Train782 / Validation100 / Test100. All training uses four GPUs and checkpoint selection uses full Validation100.

Start with [ELLIPSE.md](ELLIPSE.md) for data preparation, training, recovery, budgets and verification status. Use [TEST.md](TEST.md) for final evaluation. Architecture and optimizer settings are inherited from the Airfoil four-GPU source; [AIRFOIL.md](AIRFOIL.md) is historical reference, and its launch commands do not define this branch’s task.

```bash
export RAW_DATA_DIR=/data/datasets/megaflow2d
export DATA_DIR=/data/datasets/megaflow2d_ellipse_stride15
export CUDA_VISIBLE_DEVICES=0,1,2,3
export RESULT_ROOT=/data/runs/ellipse_text2pde_seed0
bash scripts/ellipse_4gpu.sh train
```

## AE → LDM handoff

The wrapper materializes four-device configs, trains AE to 250k updates, selects across four milestones using Validation100, trains LDM with that frozen AE and selects LDM on Validation100. Both configs set `max_epochs: -1`, so the native 250k update budget is authoritative. AE_CHECKPOINT and CHECKPOINT default to the two selected-checkpoint records under the run. Internal module names retain `cylinderflow_stride8` for compatibility; the active dataset contract is Ellipse stride15/65 frames.
