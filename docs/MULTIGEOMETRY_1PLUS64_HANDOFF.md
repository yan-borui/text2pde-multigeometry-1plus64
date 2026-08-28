# Text2PDE multigeometry 1→64 handoff

This package trains the matched Text2PDE baseline on the frozen OpenLB
multigeometry campaign. It receives one clean UVP frame on an irregular mesh and
generates the following 64 UVP frames. The physical data and split are fixed at
456 trajectories with train/validation/test counts `408/24/24`.

## Package layout

```text
handoff_v1/
  README.md
  source/
    text2pde_multigeometry_1plus64.bundle
  data/
    data_inventory.json
    prepared/
      train_windows.json
      validation_monitor_windows.json
      validation_full_windows.json
      test_full_windows_sealed.json
      train_normal_stats.pkl
      ...
    trajectories/
      train/<case_id>.h5
      validation/<case_id>.h5
      test/<case_id>.h5
```

The window manifests use paths relative to their own directory. Moving the
whole `data/` directory therefore preserves all references. The 456 HDF5 files
contain 30,895,327,209 bytes. `data_inventory.json` records the existing source
hash, byte size, split, family and portable path for every trajectory.

## Fixed experiment contract

- frames: post-ramp source frames `151..600`;
- sample: one 65-frame window, `field [65,N,3]` with channels `[u,v,p]`;
- position input: `pos_time [65,N,3]`, with trajectory-normalized `x,y` and
  window-relative time in `[0,1]`;
- train windows: `408 × 386 = 157,488`;
- normalization: unique train frames `151..600` only;
- pressure: original outlet-zero gauge is preserved; evaluation also reports
  gauge-free pressure error;
- optimization seed: `42`; evaluation seeds: `0,1,2`;
- sampling: DDIM-20;
- test fields remain sealed until validation checkpoint selection completes.

## Reference runtime

The exact environment used for adapter tests and the RTX 4090 resource smoke is
listed in `configs/multigeometry/reference_runtime.txt`. The target machine must
provide a CUDA build of PyTorch, Lightning, Open3D, torch-scatter, HDF5, NumPy
and PyYAML with the same APIs. No PyTorch Geometric dependency is used by this
pipeline.

The primary templates use FP16 mixed precision. BF16 failed in the reference
PyTorch 2.0.1 runtime because `upsample_nearest3d` did not support that dtype.
The templates keep the full `64^3` latent volume and use unchunked GNO queries.
The earlier 8,192/16,384 query chunks preserved outputs and gradients but did
not reduce the retained training workset, so they are resource diagnostics,
not the primary large-GPU configuration.

## Verify a transferred package

Clone the source bundle and run the CPU tests:

```bash
git clone source/text2pde_multigeometry_1plus64.bundle text2pde
cd text2pde
/path/to/python -m unittest discover -s tests
```

Verify the 456-file inventory and load one train and one validation window:

```bash
/path/to/python -m tools.multigeometry.verify_handoff_data \
  --data-dir /path/to/handoff_v1/data
```

This command resolves test metadata and its 72 fixed windows without opening a
test field.

## Launch the full pipeline

Choose a new result directory and an experiment identifier. The launch command
materializes the two path-free templates into the result directory, records the
code/runtime/data identity, and starts the AE stage in tmux:

```bash
CUDA_VISIBLE_DEVICES=0 bash tools/multigeometry/launch_pipeline.sh \
  /path/to/python \
  /path/to/handoff_v1/data \
  /path/to/results/text2pde_multigeometry_1plus64_RUN \
  RUN
```

The AE performs `118,116` optimizer updates with gradient accumulation 4 and
saves milestone checkpoints at updates `39,372`, `78,744` and `118,116`, plus a
rolling `last.ckpt`. The fixed validation monitor selects a finite,
non-constant AE checkpoint. The runner then starts the LDM stage in a second
tmux session. LDM training also performs `118,116` optimizer updates.

After LDM training, the runner:

1. selects a milestone with the 24-window validation monitor and seeds `0,1,2`;
2. evaluates all 72 fixed validation samples;
3. records the locked LDM checkpoint;
4. evaluates the 72 sealed test samples exactly once.

Training, selection and evaluation each write an exit marker. GPU samples,
resolved configs, logs, checkpoint identities and launch timestamps stay under
the chosen result root.

## Hardware evidence

The largest training window is `ellipse_37`, with `N=4365`. On the reference
RTX 4090, its finite FP16 update retained about 66--69 GB of PyTorch workset and
completed only through WSL managed-memory paging. This package is intended for
a GPU with at least 80 GB memory. The 4090 timing is not an estimate for the
target GPU; measure the maximum-window smoke on the destination before setting
a rental duration.

## Provenance boundary

The OpenLB data are project-generated. The Text2PDE code bundle is derived from
official compatibility commit `2d73a4ae7fb6688de1b1f4cb950a196ce09456f8` plus
the matched multigeometry adapter. The upstream GitHub repository did not carry
an explicit root license during this audit, so this handoff is for internal
research reproduction rather than source redistribution.
