# MegaFlow2D Ellipse: original mesh, UVP, stride 15

This branch runs the fixed Ellipse comparison. Use the existing method environment and the entrypoint below. It includes preparation, training, recovery and four-GPU evaluation. The model architecture, optimizer, learning-rate schedule, objective and seed come from the Airfoil four-GPU source; the data contract and complete Validation population change here.

## Data contract

- Source: [cmudrc/MegaFlow2D](https://huggingface.co/datasets/cmudrc/MegaFlow2D/tree/789ed4cab0a6d25858cc0653f695e3d1896f3241), pinned revision `789ed4cab0a6d25858cc0653f695e3d1896f3241`.
- Original high mesh `raw/mesh/has/ellipse_ID.npz`; original UVP `raw/has_original/ellipse_ID_has_FRAME.npz`. All original nodes and edges are retained. No interpolation or spatial subsampling.
- Exactly 65 file frames `1,16,31,...,961`. Stored index 0 is the observation; stored indices 1–64 are scored. Raw step 0.01 s; retained interval 0.15 s; relative times 0–9.6 s. The upstream files do not provide an absolute initial timestamp.
- [Published split lists](megaflow_data/splits.json): sort the 982 available source IDs, permute with NumPy `default_rng(42)`, assign Train782 / Validation100 / Test100. IDs remain the original, noncontiguous source IDs. All six repositories carry the same lists and preparation package.
- Train-only statistics, AE/VGAE fitting, dynamics fitting and latent caches use these same retained 65 frames. Complete Validation100 selects weights. Test100 uses those fixed weights.
- Predict all future nodes and all three fields. Boundary writeback is disabled. Conditions use static geometry and the observed initial frame only.

The converter traverses planar faces of the original directed edge graph, keeps triangular fluid cells and excludes the exterior and ellipse hole. It verifies equal recovered/original edge sets, full node coverage, positive triangle areas and the one-hole Euler characteristic. Every retained UVP file must be finite and match the high mesh node count. Upstream UVP contains no coordinates; node order is the released original-mesh convention, preserved without permutation.

Static geometry supplies labels: 0 interior, 2 ellipse, 4 inlet, 5 outlet, 6 top/bottom wall, with wall priority at rectangle corners. These are derived labels, not upstream annotations. GLaDiT keeps three condition categories (interior/outlet, inlet, solid); MGN uses four categories (interior, inlet, outlet, solid); EAGLE maps ellipse label 2 to solid label 6 and reserves 2 for padding. AROMA maps the fixed 20 × 10 physical rectangle to [-1,1] coordinates; Text2PDE retains its geometry/time normalization. Physical scoring uses the original coordinates.

## Shared preparation

```bash
export PYTHON=python                  # existing method environment
export RAW_DATA_DIR=/data/datasets/megaflow2d
export DATA_DIR=/data/datasets/megaflow2d_ellipse_stride15
bash scripts/ellipse_4gpu.sh data
```

The source directory may contain extracted `raw/...` files, the complete release parts `data.zip.001` through `data.zip.005`, or neither. Missing selected members are fetched with HTTP Range from the pinned release. The five-part ZIP totals 117,738,181,353 bytes; selective preparation fetches its 670,970,208-byte directory plus the required 982 meshes and 63,830 UVP files. It does not fetch the unselected field frames. Use `ELLIPSE_OFFLINE=1` for existing files or complete local parts. Set proxy environment variables normally when required.

Six repositories may share RAW_DATA_DIR and DATA_DIR. Directory locks serialize download and conversion. Downloaded members and completed HDF5 trajectory groups survive interruptions. Failures retain the case/member ID and logs; rerun the same command after correcting the cause. Never remove an occupied `.lock.d` directory before checking its recorded owner and workers. Retain failed cache attempts for inspection; use a fresh method-cache directory if an incomplete cache cannot be resumed.

Outputs are `ellipse_stride15_65frames.h5`, `ellipse_stride15_65frames_manifest.json` and `text2pde_normalizer.pkl`. Each trajectory stores UVP `[65,N,3]`, original coordinates/edges, recovered cells, derived labels and source frame IDs. The manifest binds the release, split lists, retained frames, time and normalization scope. Status and failures appear in `preparation_status.json`, `preparation.log`, `conversion_failure.json` or the source cache download-failure record. Test fields are read for conversion/format checks; no Test values enter normalization.

## Four-GPU operation

Use one Linux node with four allocated GPUs and the method's existing dependencies. The wrapper validates the GPU mask and uses `torch.distributed.run`/NCCL. It never allocates an unassigned device. Choose separate RESULT_ROOT values for every method and formal run.

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
export RESULT_ROOT=/data/runs/ellipse_METHOD_seed0
bash scripts/ellipse_4gpu.sh prepare
bash scripts/ellipse_4gpu.sh train
# After interruption, retain the same data, source, configuration and dependencies:
bash scripts/ellipse_4gpu.sh resume
bash scripts/ellipse_4gpu.sh validate
bash scripts/ellipse_4gpu.sh test
```

`train` automatically performs necessary preparation. `prepare` builds the method cache without training; GLaDiT representation preparation requires a completed VGAE export. `resume` requires an existing run and verifies saved identities. Final evaluation uses a fresh output folder, configurable with EVAL_DIR. Text2PDE also supports resuming its matching evaluation directory. Training logs record updates and consumed examples/windows; checkpoint metadata stores per-device batch, accumulation, global batch and EMA convention where applicable.

| Stage | Batch/rank × accumulation | Nominal global batch | Budget |
| --- | --- | --- | --- |
| GLaDiT VGAE | 4 × 1 | 16 | native schedule, at most 5,000 epochs |
| GLaDiT DiT | 1 × 1 | 4 | 250,000 updates |
| MGN | 1 × 1 | 4 | 25 epochs |
| EAGLE | 1 × 1 | 4 | 1,000 epochs |
| AROMA AE | 1 × 16 | 64 | 10,000 epochs |
| AROMA dynamics | 32 × 1 | 128 | 5,000 epochs |
| Text2PDE AE / LDM | each 1 × 1 | each 4 | each 250,000 updates |

VGAE, EAGLE and AROMA consume shorter final epoch batches with actual-count loss normalization: their tails are 14, 2, and 14/14 respectively. MGN enumerates all 782 × 64 adjacent retained-frame transitions, divisible by four. GLaDiT and Text2PDE continue a deterministic global cursor into the next shuffled epoch, maintaining global batch four without duplicated padding or discarded cases. Validation/Test partitions contain 25 distinct trajectories per rank; aggregation checks the exact case/seed population.

## Evaluation

Forecast bundles contain physical UVP `[65,N,3]`, with the observed first frame retained exactly. The main score is area-weighted UV relative RMSE over the future 64 frames, averaged equally across trajectories. Pressure and per-frame metrics remain available. Failed cases and sampling parameters are recorded. VGAE evaluation is posterior-mean reconstruction of future frames and is labelled separately from forecasting.

Selection retains each method's native frequency/objective while evaluating all Validation100: VGAE UVP total loss; GLaDiT raw/EMA physical selection; native baseline objectives; Text2PDE AE normalized UVP L1 and LDM physical selection across its four original checkpoint milestones. Final GLaDiT Validation/Test uses S6/K8 physical-field mean; AROMA uses its four-step schedule and sampling labels 0/1/2; Text2PDE uses DDIM20 and labels 0/1/2, preserving each method's native aggregation. [Test commands](TEST.md).

## Verification status

Implementation and static checks are delivered. The complete 982-case conversion and four-GPU train/save/resume/evaluate launch have not been executed for this branch. No matching production four-GPU environment was verified during delivery. Runtime verification must use the formal entrypoint on the assigned production-equivalent environment, with only workload reduced under the existing testing policy. No training results or model quality are claimed.

## AE → LDM handoff

The wrapper materializes four-device configs, trains AE to 250k updates, selects across four milestones using Validation100, trains LDM with that frozen AE and selects LDM on Validation100. Both configs set `max_epochs: -1`, so the native 250k update budget is authoritative. AE_CHECKPOINT and CHECKPOINT default to the two selected-checkpoint records under the run. Internal module names retain `cylinderflow_stride8` for compatibility; the active dataset contract is Ellipse stride15/65 frames.

## Six repositories

- [GLaDiT VGAE](https://github.com/yan-borui/vgae-cylinderflow-stride8/tree/feature/megaflow2d-ellipse-uvp-4gpu)
- [GLaDiT DiT](https://github.com/yan-borui/graph-dit-cylinderflow-stride8/tree/feature/megaflow2d-ellipse-uvp-4gpu)
- [MGN](https://github.com/yan-borui/meshgraphnets-cylinderflow-stride8/tree/feature/megaflow2d-ellipse-uvp-4gpu)
- [EAGLE](https://github.com/yan-borui/eagle-cylinderflow-stride8/tree/feature/megaflow2d-ellipse-uvp-4gpu)
- [AROMA](https://github.com/yan-borui/aroma-cylinderflow-stride8/tree/feature/megaflow2d-ellipse-uvp-4gpu)
- [Text2PDE](https://github.com/yan-borui/text2pde-multigeometry-1plus64/tree/feature/megaflow2d-ellipse-uvp-4gpu)

## Ellipse sampling and matched cost

`scripts/ellipse_sampling.sh quality` evaluates the complete requested split on
four GPUs. It supports K={1,2,4,8,16}, using nested prefixes of the same independent sample pool. The selected
checkpoint and its representation remain fixed. Physical UVP fields are averaged
before scoring; this is recorded separately from the native main-table aggregation.

```bash
export RESULT_ROOT=/shared/runs/ellipse_text2pde_seed0
export DATA_DIR=/shared/data/megaflow2d_ellipse_stride15
export OUTPUT_DIR=/shared/evaluation/ellipse_text2pde_test_members
export CUDA_VISIBLE_DEVICES=0,1,2,3
SPLIT=test bash scripts/ellipse_sampling.sh quality
export OUTPUT_DIR=/shared/evaluation/ellipse_text2pde_cost
export CUDA_VISIBLE_DEVICES=0
SPLIT=validation bash scripts/ellipse_sampling.sh benchmark
```

Each invocation requires a new output directory. Optional CHECKPOINT and
AE_CHECKPOINT select explicit matching weights; defaults use this run's native
Validation-selected files. PREPARED_DIR defaults to `$RESULT_ROOT/prepared` for
graph baselines. Text2PDE uses the run's frozen `config/ldm_1plus64.yaml`.
`--sampling-label-offset` selects a new sample pool without changing training seed.

Member archives, member metrics, mean-field archives and summaries for each K
retain the original case IDs and dt=0.15. Invalid members are retained; the
aggregate reports failure counts and the command exits unsuccessfully. Four-rank
merging checks every case/member pair exactly once. EAGLE geometry caches are
isolated by evaluation and rank.

The benchmark is a separate single-GPU FP32/batch1 invocation on the fixed 24
Validation positions `rint(linspace(0,99,24))`, with two warmups and three timed
repeats per case. It measures CPU initial field to CPU complete physical forecast,
including sampling and field averaging, and records device/runtime, latency,
memory and failures. Compare only matched hardware/runtime. Original Validation100
benchmark callers remain supported and are labelled separately.

The GLaDiT repository's `graph_dit.ellipse_report` accepts these summary files and
physical prediction bundles alongside CylinderFlow/Airfoil results. Native stage
`summary.json` records and pipeline timing receipts supply training cost; identify
AE, dynamics and preparation separately. These interfaces have static verification
only; formal four-GPU evaluation and single-GPU timing remain unexecuted.
