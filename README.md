# Text2PDE on raw MeshGraphNets CylinderFlow: `1 -> 24 -> rollout64`

This fork trains Text2PDE from random initialization on the original, untemporally-subsampled MeshGraphNets CylinderFlow trajectories. One Text2PDE call consumes one clean UVP frame and generates 24 future frames. Evaluation calls that model three times autoregressively and keeps 64 future frames for a frame-aligned comparison with a `1 -> 64` model.

The repository contains code and manifests only. It does not redistribute the MeshGraphNets data or newly trained checkpoints.

## Locked protocol

- Source: original MeshGraphNets CylinderFlow TFRecords, with `1000/100/100` Train/Validation/Test trajectories.
- Every trajectory has 600 frames at physical `dt=0.01`; frame stride is always 1.
- A training example is one contiguous 25-frame window. Every Train trajectory contributes starts `0..575`, for `1000 * 576 = 576,000` examples.
- UVP normalization is computed from every unique raw Train frame exactly once. Overlapping windows do not reweight the statistics, and Validation/Test fields never enter them.
- AE and LDM both start from random initialization. Microbatch is 1, gradient accumulation is 4, and one complete pass is 144,000 optimizer steps. Milestone checkpoints are written at steps 48,000, 96,000, and 144,000; `last.ckpt` is resumable at the next exact sample.
- Validation uses all 100 trajectories, starts `0/268/535`, and sampling seeds `0/1/2`: 900 rollout64 samples per selected checkpoint. Checkpoint selection is Validation-only.
- Test fields remain inaccessible to the main pipeline. A separate, explicit launcher materializes and evaluates Test only after the Validation checkpoint lock exists.

The rollout stitch is:

| Call | Condition available to the model | Kept local frames | Global future frames |
|---|---|---:|---:|
| 1 | true frame 0 | `1:25` | `1..24` |
| 2 | predicted frame 24 | `1:25` | `25..48` |
| 3 | predicted frame 48 | `1:17` | `49..64` |

The saved tensor is therefore `truth[0] + call1[1:25] + call2[1:25] + call3[1:17]`, with shape `[65, N, 3]`. The second and third calls receive predictions from the preceding call. Future reference fields are used only after sampling to compute metrics.

## Environment

The upstream environment remains in `environment.yml`. The raw reader is self-contained and does not require TensorFlow or an additional TFRecord package:

```bash
conda env create -n text2pde -f environment.yml
conda activate text2pde
```

Use `num_workers: 0` for this HDF5 workflow. The checked-in CylinderFlow configs already enforce it.

## 1. Prepare Train and Validation

Point the command at the official `meta.json`, `train.tfrecord`, and `valid.tfrecord`. It intentionally has no Test argument.

```bash
python -m tools.cylinderflow.prepare_data \
  --meta /data/mgn/cylinder_flow/meta.json \
  --train-tfrecord /data/mgn/cylinder_flow/train.tfrecord \
  --validation-tfrecord /data/mgn/cylinder_flow/valid.tfrecord \
  --output-dir /runs/cylinderflow_raw600_data

python -m tools.cylinderflow.verify_data \
  --data-dir /runs/cylinderflow_raw600_data
```

The prepared tree contains frame-chunked HDF5 storage, the 25-frame Train/AE monitor manifests, the rollout64 Validation manifests, Train-only normalization, a largest-graph smoke manifest, and a metadata-only sealed Test manifest.

## 2. GPU preflight

Run one AE optimizer update, one LDM optimizer update, and one real three-segment rollout on the largest graph in the prepared package. Released upstream CylinderFlow weights are used only to exercise the LDM/rollout code path; formal AE and LDM training still initialize randomly.

```bash
bash tools/cylinderflow/run_gpu_smoke.sh /path/to/python \
  --ae-config configs/cylinderflow/ae_1plus24_raw.yaml \
  --ldm-config configs/cylinderflow/ldm_1plus24_raw.yaml \
  --prepared-dir /runs/cylinderflow_raw600_data/prepared \
  --released-ae /models/ae_cylinder.ckpt \
  --released-ldm /models/ldm_DiTSmall_FF_cylinder.ckpt \
  --output-dir /runs/cylinderflow_raw600_preflight \
  --precision 16-mixed \
  --ddim-steps 20
```

An actual OOM, non-finite tensor, or process failure makes the preflight fail and leaves its evidence. Observed loss, gradient magnitude, temperature, memory, throughput, and runtime are recorded without heuristic stop thresholds.

## 3. Formal AE and LDM pipeline

The launcher verifies the data package and matching GPU preflight, materializes fixed configs, records source/runtime identity, and starts AE training in tmux. A successful AE run selects among the three AE milestones on a fixed 24-clip Validation reconstruction monitor and launches LDM training. A successful LDM run selects all three milestones by three-segment rollout64 over the full 100-trajectory, three-start, three-seed Validation protocol, evaluates the selected checkpoint again to save seed-0 arrays and shared-scale median/difficult/worst animations, then writes `validation_complete_awaiting_test` and stops.

```bash
bash tools/cylinderflow/launch_pipeline.sh \
  /path/to/python \
  /runs/cylinderflow_raw600_data \
  /runs/text2pde_raw_mgn_1plus24_rollout64 \
  20260901a \
  /runs/cylinderflow_raw600_preflight/summary.json
```

For a manual run, first bind the portable templates:

```bash
python -m tools.cylinderflow.materialize_config \
  --template configs/cylinderflow/ae_1plus24_raw.yaml \
  --prepared-dir /runs/cylinderflow_raw600_data/prepared \
  --result-root /runs/text2pde_raw_mgn_1plus24_rollout64 \
  --stage ae \
  --output /runs/text2pde_raw_mgn_1plus24_rollout64/config/ae_1plus24_raw.yaml

python -m tools.cylinderflow.materialize_config \
  --template configs/cylinderflow/ldm_1plus24_raw.yaml \
  --prepared-dir /runs/cylinderflow_raw600_data/prepared \
  --result-root /runs/text2pde_raw_mgn_1plus24_rollout64 \
  --stage ldm \
  --output /runs/text2pde_raw_mgn_1plus24_rollout64/config/ldm_1plus24_raw.yaml
```

Then train AE, select it, train LDM, select it by rollout64, and run full Validation:

```bash
python train_AE.py \
  --config /runs/text2pde_raw_mgn_1plus24_rollout64/config/ae_1plus24_raw.yaml

python -m tools.cylinderflow.select_ae \
  --config /runs/text2pde_raw_mgn_1plus24_rollout64/config/ae_1plus24_raw.yaml \
  --checkpoint-dir /runs/text2pde_raw_mgn_1plus24_rollout64/ae/formal/checkpoints \
  --manifest /runs/cylinderflow_raw600_data/prepared/validation_monitor_windows_25.json \
  --output-dir /runs/text2pde_raw_mgn_1plus24_rollout64/ae/selection_v1

python train_ldm.py \
  --config /runs/text2pde_raw_mgn_1plus24_rollout64/config/ldm_1plus24_raw.yaml \
  --first-stage-checkpoint /path/to/selected_ae.ckpt

python -m tools.cylinderflow.evaluate_rollout \
  --mode select \
  --config /runs/text2pde_raw_mgn_1plus24_rollout64/config/ldm_1plus24_raw.yaml \
  --ae-checkpoint /path/to/selected_ae.ckpt \
  --checkpoint-dir /runs/text2pde_raw_mgn_1plus24_rollout64/ldm/formal/checkpoints \
  --manifest /runs/cylinderflow_raw600_data/prepared/validation_full_rollout64.json \
  --output-dir /runs/text2pde_raw_mgn_1plus24_rollout64/evaluation/ldm_selection_v1 \
  --seeds 0 1 2 \
  --ddim-steps 20

python -m tools.cylinderflow.evaluate_rollout \
  --mode validation \
  --config /runs/text2pde_raw_mgn_1plus24_rollout64/config/ldm_1plus24_raw.yaml \
  --ae-checkpoint /path/to/selected_ae.ckpt \
  --checkpoint /path/to/selected_ldm.ckpt \
  --manifest /runs/cylinderflow_raw600_data/prepared/validation_full_rollout64.json \
  --output-dir /runs/text2pde_raw_mgn_1plus24_rollout64/evaluation/validation_v1 \
  --seeds 0 1 2 \
  --ddim-steps 20 \
  --save-first-seed \
  --render-representative-gifs
```

Resume either trainer by adding `--checkpoint .../checkpoints/last.ckpt`. The checkpoint stores model, optimizer, scheduler, exact examples-seen cursor, and Python/NumPy/CPU/CUDA RNG states. The deterministic sampler resumes the same fixed Train permutation.

## 4. Independent Test launcher

Run this command only after reviewing the full Validation report and accepting the locked checkpoint. This is the only provided workflow that accepts the raw Test TFRecord.

```bash
bash tools/cylinderflow/run_test_stage.sh \
  /path/to/python \
  /data/mgn/cylinder_flow/meta.json \
  /data/mgn/cylinder_flow/test.tfrecord \
  /runs/cylinderflow_raw600_data \
  /runs/text2pde_raw_mgn_1plus24_rollout64 \
  /runs/cylinderflow_raw600_test_data
```

The launcher requires a clean source tree at the training commit, verifies the locked AE/LDM/config/Train data identities, materializes Test into a separate tree, and evaluates exactly the locked checkpoint. It refuses an existing Test output path.

## Evaluation outputs

The evaluator reports:

- area-weighted future UV relative RMSE;
- raw and gauge-free pressure RMSE;
- piecewise-linear triangle vorticity and divergence RMSE;
- kinetic-energy and enstrophy errors/correlations;
- temporal velocity-spectrum error and energy phase lag;
- inlet, outlet, wall, and all-boundary UV errors using raw MGN node types;
- per-frame errors, the `24->25` and `48->49` rollout seams, failure count;
- end-to-end three-call time and peak allocated/reserved GPU memory.

Seed-0 `.npz` files record the raw frame indices, physical time, segment seeds, and whether each segment condition came from truth or a prior prediction. GIF truth/prediction panels use the same color limits, and the median, difficult (P90-ranked), and worst selected clips share one recorded scale set.

## Tests

```bash
python -m unittest discover -s tests
python -m compileall dataset modules tools train_AE.py train_ldm.py
bash -n tools/cylinderflow/launch_pipeline.sh
bash -n tools/cylinderflow/run_ae_stage.sh
bash -n tools/cylinderflow/run_ldm_stage.sh
bash -n tools/cylinderflow/run_test_stage.sh
bash -n tools/cylinderflow/run_gpu_smoke.sh
```

The focused tests cover raw 600-frame decoding, all 576 contiguous starts, Train-only statistics, split/Test isolation, three-segment information flow and `[65,N,3]` stitching, deterministic segment seeds, mesh diagnostics, and interrupted-versus-uninterrupted sample order, RNG draws, LR, next loss, and final weight.

## Legacy: OpenLB multi-geometry `1 -> 64`

The earlier user-generated OpenLB multi-geometry implementation remains unchanged as a legacy workflow:

- configs: `configs/multigeometry/ae_1plus64.yaml` and `configs/multigeometry/ldm_1plus64.yaml`;
- tools: `tools/multigeometry/`;
- handoff record: `docs/MULTIGEOMETRY_1PLUS64_HANDOFF.md`.

That path jointly models one clean frame plus 64 future frames from the OpenLB `408/24/24` split. Its checkpoints, manifests, and evaluation protocol are independent of the raw-MGN `1 -> 24`, three-call rollout64 experiment.

## Upstream Text2PDE

This repository is based on [Text2PDE: Latent Diffusion Models for Accessible Physics Simulation](https://arxiv.org/abs/2410.01153). Upstream datasets and pretrained models are linked from [ayz2/ldm_pdes](https://huggingface.co/datasets/ayz2/ldm_pdes). Existing cylinder, NS2D, baseline, captioning, and profiling code remains available under `configs/`, `dataset/`, `text/`, and the original training/validation entry points.

Open3D in the upstream stack may require PyTorch `<=2.0.1`; setting `use_open3d: false` uses the native PyTorch path at higher memory cost. The checked-in formal configs preserve the established Open3D setting and must be preflighted in the actual training environment.
