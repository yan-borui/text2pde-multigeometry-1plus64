# Matched inference speed and cost

Protocol: `cylinderflow.initial_cpu_to_physical_cpu.fp32.v1`.

Run this benchmark once for the already-selected dynamics checkpoint. It uses the same fixed, uniformly covering Validation-24 registry in every repository. It does not select checkpoints or alter training recipes. Test remains sealed.

## Timing boundary

The model and its representation dependencies are resident on one device. The physical initial UVP, mesh, connectivity, node labels and prepared static caches are available in CPU memory. Start synchronized wall-clock timing immediately before the predictor and stop after the complete physical forecast returns to CPU.

Include input packing and normalization, CPU/device transfers, initial conditioning or encoding, every model call needed for the 64-frame forecast, decoding, inlet/wall velocity writeback, and the final CPU transfer. Output is float32 physical UVP `[65,N,3]`, including the unchanged observed frame. EAGLE clusters are loaded from disk before timing. Text2PDE reads and transfers only the initial physical field; future reference fields are absent from this benchmark.

Model/checkpoint loading, disk input, RNG reseeding, warmups, output checks, quality scoring, prediction-file writing and rendering are excluded from per-forecast latency. Loading and warmup durations are reported separately. Native stage timers and pre-boundary diagnostic forecasts are disabled during matched measurement. EAGLE also skips auxiliary training-output and target arrays in its forecast-only path. Existing quality-evaluation timing fields remain native diagnostics; use this benchmark's `inference_wall_seconds` for cross-method speed comparisons.

## Fixed execution

- One CPU or one explicitly assigned CUDA device; microbatch one.
- FP32 weights and arithmetic, autocast disabled, TF32 disabled, highest float32 matmul precision, cuDNN autotuning disabled.
- Two CPU intra-op threads by default in every entry point. Record inter-op and OMP/MKL/OpenBLAS thread settings.
- Each trajectory receives two full-forecast warmups and three measured full forecasts.
- Reseed before every call. Measured draw labels are 0/1/2, independent of warmups and training seeds; warmup labels are 100/101. The common seed mapping is recorded in the report.
- CUDA synchronization surrounds the predictor. GPU peak statistics reset before each call. Resident allocation, total allocated/reserved peaks and incremental allocated peak use bytes. CPU runs report CUDA quantities as null.
- Numerical and OOM failures retain their phase, trajectory, timing and error. A failed warmup or measured call makes that trajectory incomplete. All attempted measured calls remain in the record; failed reports cannot enter a matched comparison.

Use the same hardware and execution environment, without concurrent work on the measured device. The report records device/CPU details, package/runtime versions, precision and threading. Software collection alone cannot establish external GPU exclusivity; retain the target scheduler allocation with the run.

## Report and comparison

`samples.jsonl` and `samples.csv` contain all warmup and measured attempts. `summary.json` reports the mean of three repeats within each complete trajectory and then mean/median/P90/min/max across trajectories. `failures.json` retains unsuccessful attempts.

Common cost fields include seconds per generated frame, serial generated frames per second, measured inference GPU-hours, projected GPU-hours for 100 trajectories, model parameters/bytes, and peak GPU allocation/reservation. The projection uses the Validation-24 mean, excludes setup and warmup, and represents serial inference rather than batched throughput. Parameter storage includes the selected representation model. These are inference costs; training, AE fitting and cache preparation retain their separate run records.

The comparison command rejects incomplete or debug reports and differences in hardware/runtime identity, settings, input release or case registry. It does not mix native `model_seconds` with total `inference_wall_seconds`. A runtime mismatch should be resolved using compatible target environments before constructing the shared speed table; it is not silently ignored.

## Commands

For MGN, EAGLE and AROMA, use the corresponding repository:

```bash
python -m cylinderflow.benchmark \
  --dataset "$DATA" --manifest "$MANIFEST" --prepared runs/prepared \
  --checkpoint runs/main/best.pt --device cuda:0 --threads 2 \
  --output-dir runs/performance
```

For AROMA, use `--checkpoint runs/dynamics/best.pt --ae-checkpoint runs/ae/best.pt`. A benchmark against a standalone AE checkpoint is rejected. `--debug-data` is reserved for synthetic acceptance and produces a report excluded from formal comparisons.

For Text2PDE, use its materialized LDM configuration and the checkpoints selected by its Validation workflow:

```bash
python -m tools.cylinderflow_stride8.benchmark \
  --config "$LDM_CONFIG" --checkpoint "$SELECTED_LDM" \
  --ae-checkpoint "$SELECTED_AE" --device cuda:0 --threads 2 \
  --output-dir "$RESULT_ROOT/performance"
```

Compare the four complete reports from matched target runs:

```bash
python -m cylinderflow.performance \
  --reports mgn/summary.json eagle/summary.json aroma/summary.json text2pde/summary.json \
  --output comparison.json
```

In Text2PDE, the identical comparison implementation is `python -m tools.cylinderflow_stride8.performance`. Every destination must be new. Training precision, optimizer and sampling schedules remain those of the frozen method; this common FP32 execution applies to the separate inference benchmark.

The implementation verification record is `PERFORMANCE_VERIFICATION.json`. CPU tests validate protocol behavior; target-device latency and GPU memory measurements require actual target runs.
