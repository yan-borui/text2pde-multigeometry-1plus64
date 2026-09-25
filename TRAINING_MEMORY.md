# Training memory

The four-GPU Airfoil entrypoint records cumulative PyTorch CUDA memory peaks for
the AE and LDM stages. Each stage keeps all-rank observations, the largest
per-device allocation and reservation, hardware and precision, and the matching
update, epoch and checkpoint identity. The records support a training-cost table
when the corresponding run has produced them.

## Running and collecting

Use the existing [Airfoil training entrypoint](AIRFOIL.md). Its distributed
callback enables recording automatically. The AE and LDM run directories each
contain:

- `distributed_metrics.jsonl`: observations at training start, the existing
  logging interval, validation end, epoch end, checkpoint construction and
  training end before GPU teardown. Existing update, timing, GPU-hours and
  per-rank GiB fields are retained.
- `training_memory/<segment>.json`: the latest observation for each training
  process segment. A resumed process creates a new segment and preserves earlier
  summaries. The largest allocated/reserved peak across complete segment records
  gives the observed stage peak across those segments.
- The checkpoint's `training_memory` entry: all-rank memory observed during
  checkpoint construction. Its checkpoint identity matches the existing resume
  record. A checkpoint event records construction, and file existence establishes
  successful persistence.

The two peak fields are maxima over ranks, in bytes. Per-rank records include
the GPU model, physical memory, allocated/reserved peaks and device index.
The summary includes participating GPU count, world size, Lightning precision,
software versions, UTC observation time, update and zero-based epoch. AE and LDM
remain separate stages. For an overall per-device capacity figure, use the larger
stage peak and retain both stage records.

## Measurement scope

Peaks reset once in the existing training-start hook, after model setup and
validation sanity checks. Resident model and optimizer allocations at that point
remain part of the measured allocation. The counters then cover training,
in-training validation, checkpoint preparation and the recording operations.
They remain cumulative across epochs and validation calls. The stored scope
excludes transient setup and sanity-check allocations before this reset.

Allocated and reserved are PyTorch allocator measurements. Compare them with
measurements using the same precision, GPU topology, batch, validation and reset
scope. `nvidia-smi` process usage includes different components and requires its
own label. The timing fields retain the existing trainer convention; the memory
segment duration identifies the current process's observation window.

An exception records the surviving rank's local peak in its failure record.
Exception handling performs no all-rank collective, because another worker may
already have exited. An incomplete rank set supports a partial observation.

## Existing runs and evidence

Earlier four-GPU source already logged per-rank `allocated_gib` and
`reserved_gib` periodically in `distributed_metrics.jsonl`. Recover those logs
and the matching source/configuration before deciding whether historical memory
is missing. Those measurements can support the scope actually recorded.

The added metadata and checkpoint/validation observations require execution of
this source. Historical peaks before instrumentation or before a resumed process
are unavailable unless an existing record supplies them. A resume at a completed
budget creates no training segment because the training loop does not run.

The existing source-freeze contract remains active. Use this branch for a fresh
run; subsequent resumes require its identical frozen source. Existing runs keep
their original checkout and source snapshot. This delivery supplies code and
static validation; measured values require collection from the intended GPU run.

Hook signatures follow the project's
[Lightning 2.3 callback interface](https://lightning.ai/docs/pytorch/2.3.3/api/lightning.pytorch.callbacks.Callback.html).
