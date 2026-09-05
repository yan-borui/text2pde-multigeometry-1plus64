# CylinderFlow stride-8: staged training and common evaluation

Updated 2026-09-05. The three `cylinderflow` adapters use protocol `cylinderflow.stride8.train75.eval65.v2`; Text2PDE uses `text2pde.cylinderflow_stride8.ae75.joint65.v2`. All four use evaluator `cylinderflow.physical_mesh.v1` and prediction schema `cylinderflow.physical_prediction.v2`.

## Data and stage boundaries

Use [DingDong1921/mgn-cylinderflow-stride8-75frames](https://huggingface.co/datasets/DingDong1921/mgn-cylinderflow-stride8-75frames) at revision `8eae2c7a697e7d01f3b98f4d642ea476784df84a`. The published files are `data/cylinderflow_stride8_75frames.h5` and `metadata/cylinderflow_stride8_75frames_manifest.json`. Each trajectory contains physical `uvp [75,N,3]`, `mesh_pos [N,2]`, triangular `cells [C,3]`, and `node_type [N]`. Labels are normal=0, inlet=4, outlet=5, wall=6. Train is 0..999, Validation is 1000..1099; Test is sealed.

Stored frames 0..74 map to raw indices 0,8,...,592, with physical dt=0.08. Forecast evaluation always observes stored frame 0 and scores only frames 1..64: raw indices through 512, horizon 5.12. All archived forecasts are `[65,N,3]`. Expanding training never expands the evaluation horizon.

| Stage | Permitted training frames and samples | Frozen budget / effective batch |
| --- | --- | --- |
| MGN | All 74 adjacent pairs per Train trajectory | 25 epochs / 1 |
| EAGLE | One native 6-frame window per trajectory per epoch; start 0..69 | 1000 epochs / 2 |
| AROMA AE | One randomly selected frame among 0..74 per trajectory per epoch | 10000 epochs / 64 |
| AROMA dynamics | All 74 adjacent latent pairs per trajectory | 5000 epochs / 128 trajectories |
| Text2PDE AE | One complete 75-frame sequence per trajectory per epoch | 250000 updates, 1000 epochs / 4 |
| Text2PDE dynamics | One fixed first-65-frame sequence per trajectory per epoch; start 0 | 250000 updates, 1000 epochs / 4 |

These ranges are deliberate stage-specific choices. MGN and AROMA dynamics process 15.625% more transitions than the prior 64-pair recipe while retaining its epochs. Text2PDE dynamics has no sliding windows or rotating starts. The new protocol preserves native architectures, losses, optimizers, learning rates, schedules, and batch sizes; it introduces no hyperparameter search.

## Normalization and representations

The three `cylinderflow prepare` commands recompute statistics from all 75 unique Train frames and record the permitted range and dataset identity. AROMA/EAGLE use population moments pooled over nodes and time. EAGLE's internal channels are `[u,v,p,q]`, with q=0.5*(u*u+v*v). MGN retains equal-trajectory moments of current UV on frames 0..73, the 74 UV increments, next pressure on frames 1..74, and directed edge features; increment variance includes its native normal-node input noise. Each transform has an explicit zero-variance floor.

Text2PDE retains the published 75-frame Train normalizer. Its data config explicitly selects `stage: ae, sequence_length: 75` or `stage: ldm, sequence_length: 65`, both with sequence_start=0. The GINO latent grid and architecture remain unchanged. Each stage preserves the native sequence-relative time coordinate [0,1]; physical archive times remain dt=0.08. The LDM encodes only its first 65 physical frames, not a truncated encoding of all 75.

AROMA posterior-mean Train caches now have shape `[75,32,8]` and bind the selected AE, statistics, and data contract. Prefix65 normalizers, caches, and checkpoints fail new-protocol dependency checks. Text2PDE saves a unique checkpoint identifier and the stage/frame contract; LDM checkpoints also bind the selected AE identifier. Resume, evaluation, and Validation finalization check these dependencies, including AEs with identical frame contracts but different weights. Preserve prior runs with their original protocol identities and use new result directories.

## Prediction and boundary information

Inference conditions on the true initial frame, static mesh, and known node labels. Future reference values never enter the conditioning or recurrent predictor. MGN/EAGLE clamp inlet/wall velocity in physical recurrent state. AROMA advances latent state freely and clamps those velocity values after decoding. Text2PDE applies the same output writeback after its single 20-step joint DDIM call. Only inlet/wall UV uses first-frame values; pressure and outlet remain predictions.

Save both pre-writeback and post-writeback arrays, with separate boundary diagnostics. The common NPZ schema includes prediction, pre_boundary, target, points, cells, node_type, trajectory_index, seed, raw_frame_indices, physical_time, units, and JSON provenance containing checkpoint/configuration/normalization/training-seed information. `frame_indices` aliases raw_frame_indices for existing renderers. EAGLE additionally preserves its native `[65,N,4]` state and pre-writeback q predictions.

## Metrics, aggregation, and selection

`metrics.py` and `predictions.py` are maintained identically in all four repositories. In Text2PDE they live under `tools/cylinderflow_stride8/`; in the other three they live under `cylinderflow/`. The legacy Text2PDE raw-grid evaluator is unchanged.

Metrics consume physical UVP exactly once. UV relative RMSE sums triangle-derived node-area-weighted squared error over future times, nodes, and velocity components, then divides by the corresponding reference energy before taking a square root. Pressure is reported raw and after removing each frame's area-weighted pressure-error mean. Vorticity/divergence use piecewise linear triangle derivatives; energy/enstrophy use area means. Spectra are temporal velocity spectra. Per-frame and final-frame errors are retained. Boundary subset RMSE uses unweighted node/component averages and is labeled separately.

Use the same 24 uniformly covering Validation trajectories for dynamics selection. AROMA/Text2PDE sampling labels are 0/1/2; MGN/EAGLE use one deterministic prediction. Retain the actual derived PRNG seed. Score each generated field separately, average sampling metrics within each trajectory, then report mean/median/P90/min/max across trajectories. A trajectory with any failed clip has no finite trajectory mean; failure counts and both denominators remain visible. Duplicate trajectory/seed rows are rejected.

Rank dynamics candidates lexicographically by failed clip count, mean complete-trajectory UV relative RMSE, and optimizer update; earlier updates win exact ties. Native checkpoint production schedules remain unchanged: this change unifies the ranking rule, not the number of candidates. AROMA AE retains its reconstruction selection rule with representative frames 0/25/50/74; Text2PDE AE uses its native normalized reconstruction L1 on all 75 frames of the fixed monitor.

Full Validation evaluates one selected checkpoint on all 100 trajectories: 300 clips for AROMA/Text2PDE, 100 for MGN/EAGLE. Every evaluated sampling label is archived, including numerical failures; a failed prediction may be a NaN placeholder with an intact observed first frame and an explicit failure record. Text2PDE saves candidate arrays in `candidate_NNN/predictions/`, with incremental case JSONL, CSV, trajectory summaries, and failures.

Offline scoring validates frame count, raw indices, physical times, and mesh axes before computing the same metrics. The three adapters expose `python -m cylinderflow score`; Text2PDE exposes `python -m tools.cylinderflow_stride8.score --inputs ... --output-dir ...`. Files from any of the four are accepted. Offline results describe the explicitly supplied files, not a new checkpoint-selection opportunity.

## Execution and evidence

Existing native training budgets are retained. Keep independent method configurations, effective sample exposures, optimizer updates, training and sampling seeds, and inference timing semantics explicit; equal epochs or sample exposure do not imply equal GPU-hours.

Run the repository's CPU contract tests and bounded full-model smoke checks before a target-cluster run. Target GPU/mixed-precision/maximum-mesh acceptance remains a separate check. The software verification record for this change is `ALIGNMENT_VERIFICATION.json`; `ACCEPTANCE.json` in the three adapters retains the dated prefix65 evidence. Formal model quality requires new training and Validation evidence under the updated stage contract.
