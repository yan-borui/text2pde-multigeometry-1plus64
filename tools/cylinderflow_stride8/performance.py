"""Matched single-device wall-clock latency and memory accounting."""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import random
import time
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
import torch

PERFORMANCE_PROTOCOL = "cylinderflow.initial_cpu_to_physical_cpu.fp32.v1"
WARMUP_REPEATS = 2
MEASURED_REPEATS = 3
FUTURE_FRAMES = 64
VALIDATION_TRAJECTORIES = tuple(
    int(value) + 1000 for value in np.rint(np.linspace(0, 99, 24))
)


def write_json(file_path: Path, value: Any) -> None:
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def configure(device: torch.device, threads: int = 2) -> None:
    """Apply the same inference execution settings before loading any model."""
    if device.type not in ("cpu", "cuda") or threads < 1:
        raise ValueError("benchmark requires cpu/cuda and a positive thread count")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    torch.set_num_threads(threads)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = False
    torch.use_deterministic_algorithms(False)


def runtime_identity(device: torch.device) -> dict[str, Any]:
    cpu_model = platform.processor()
    cpu_info = Path("/proc/cpuinfo")
    if cpu_info.is_file():
        for line in cpu_info.read_text().splitlines():
            if line.startswith("model name"):
                cpu_model = line.split(":", 1)[1].strip()
                break
    identity = {
        "device_type": device.type,
        "device_count": 1,
        "device_name": (
            cpu_model if device.type == "cpu" else torch.cuda.get_device_name(device)
        ),
        "cpu_model": cpu_model,
        "cpu_logical_count": os.cpu_count(),
        "platform_system": platform.system(),
        "platform_machine": platform.machine(),
        "python_version": platform.python_version(),
        "torch_version": str(torch.__version__),
        "numpy_version": str(np.__version__),
        "cuda_runtime": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "precision": "fp32",
        "autocast": False,
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
        "tf32_cudnn": torch.backends.cudnn.allow_tf32,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "torch_threads": torch.get_num_threads(),
        "torch_interop_threads": torch.get_num_interop_threads(),
        "thread_environment": {
            key: os.environ.get(key)
            for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")
        },
        "gpu_total_memory_bytes": None,
        "gpu_compute_capability": None,
    }
    if device.type == "cuda":
        properties = torch.cuda.get_device_properties(device)
        identity["gpu_total_memory_bytes"] = int(properties.total_memory)
        identity["gpu_compute_capability"] = [properties.major, properties.minor]
    return identity


def model_inventory(models: Iterable[torch.nn.Module]) -> dict[str, int]:
    parameters, buffers = {}, {}
    for model in models:
        model.eval().float()
        for parameter in model.parameters():
            parameters[id(parameter)] = parameter
        for buffer in model.buffers():
            buffers[id(buffer)] = buffer
    return {
        "parameter_count": sum(value.numel() for value in parameters.values()),
        "parameter_bytes": sum(
            value.numel() * value.element_size() for value in parameters.values()
        ),
        "buffer_bytes": sum(
            value.numel() * value.element_size() for value in buffers.values()
        ),
    }


def seed_draw(trajectory_index: int, draw: int) -> int:
    seed = ((draw + 1) * 1_000_003 + trajectory_index * 9_176) % (2**31 - 1)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return seed


def distribution(values: list[float]) -> dict[str, float | int | None]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": len(values),
        "mean": float(array.mean()) if len(array) else None,
        "median": float(np.median(array)) if len(array) else None,
        "p90": float(np.quantile(array, 0.9)) if len(array) else None,
        "min": float(array.min()) if len(array) else None,
        "max": float(array.max()) if len(array) else None,
    }


def measure_prediction(
    predict: Callable[[dict], np.ndarray], sample: dict, device: torch.device
) -> dict[str, Any]:
    """Time the complete predictor, with output validation outside the timer."""
    resident = None
    if device.type == "cuda":
        sync(device)
        torch.cuda.reset_peak_memory_stats(device)
        resident = int(torch.cuda.memory_allocated(device))
    sync(device)
    started = time.perf_counter()
    prediction, failure = None, None
    try:
        with torch.inference_mode(), torch.autocast(device.type, enabled=False):
            prediction = predict(sample)
    except (FloatingPointError, torch.cuda.OutOfMemoryError) as error:
        failure = f"{type(error).__name__}: {error}"
    finally:
        sync(device)
    elapsed = time.perf_counter() - started
    peak_allocated = (
        int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None
    )
    peak_reserved = (
        int(torch.cuda.max_memory_reserved(device)) if device.type == "cuda" else None
    )
    if failure is None:
        if not isinstance(prediction, np.ndarray) or prediction.dtype != np.float32:
            raise ValueError("predictor must return a CPU numpy float32 physical array")
        if prediction.shape != (65, len(sample["initial"]), 3):
            raise ValueError(
                "predictor must return all 65 states including the initial frame"
            )
        if not np.array_equal(prediction[0], sample["initial"]):
            raise ValueError("predictor changed the observed initial frame")
        if not np.isfinite(prediction).all():
            failure = "nonfinite physical prediction"
    del prediction
    if failure and device.type == "cuda":
        # Recovery is outside measured latency; failed warmups remain visible.
        torch.cuda.empty_cache()
    return {
        "inference_wall_seconds": elapsed,
        "finite": failure is None,
        "failure": failure,
        "cuda_resident_allocated_before_bytes": resident,
        "cuda_peak_allocated_bytes": peak_allocated,
        "cuda_peak_reserved_bytes": peak_reserved,
        "cuda_incremental_peak_allocated_bytes": (
            None if resident is None else max(0, peak_allocated - resident)
        ),
    }


def benchmark(
    *,
    method: str,
    indices: Iterable[int],
    load_case: Callable[[int], dict],
    predict: Callable[[dict], np.ndarray],
    device: torch.device,
    output_dir: Path,
    data_identity: dict,
    provenance: dict,
    models: Iterable[torch.nn.Module],
    model_load_seconds: float,
    formal: bool = True,
) -> dict[str, Any]:
    """Benchmark one already-selected checkpoint; produce no quality ranking."""
    output_dir.mkdir(parents=True, exist_ok=False)
    indices = tuple(indices)
    if len(set(indices)) != len(indices) or not indices:
        raise ValueError("benchmark case indices must be nonempty and unique")
    if formal and len(indices) != 24:
        raise ValueError("formal benchmark requires the fixed Validation-24 registry")
    inventory = model_inventory(models)
    sync(device)
    environment = runtime_identity(device)
    settings = {
        "protocol": PERFORMANCE_PROTOCOL,
        "registry": "validation24" if formal else "synthetic_debug",
        "warmup_repeats_per_trajectory": WARMUP_REPEATS,
        "measured_repeats_per_trajectory": MEASURED_REPEATS,
        "microbatch": 1,
        "future_frames": FUTURE_FRAMES,
        "physical_dt": 0.08,
        "sampling_labels": list(range(MEASURED_REPEATS)),
        "seed_rule": "((draw+1)*1000003 + trajectory_index*9176) mod (2**31-1)",
        "input": "CPU physical initial UVP and static geometry; disk-backed static caches resident in CPU memory",
        "output": "CPU numpy float32 physical UVP [65,N,3], including boundary writeback",
        "timed": [
            "input packing and normalization",
            "CPU-device transfers",
            "conditioning/encoding",
            "all 64-frame model calls",
            "decoding",
            "boundary writeback",
            "return to CPU",
        ],
        "excluded": [
            "checkpoint/model loading",
            "input disk IO and static-cache loading",
            "RNG reseeding",
            "warmups",
            "output validity checks",
            "pre-boundary diagnostic forecasts and auxiliary training targets",
            "quality scoring",
            "output archive IO",
            "rendering",
        ],
        "internal_stage_instrumentation": False,
    }
    write_json(
        output_dir / "protocol.json",
        {
            "settings": settings,
            "environment": environment,
            "data_identity": data_identity,
            "provenance": provenance,
            "model": inventory,
        },
    )
    started = time.perf_counter()
    rows, cases, case_ids = [], [], set()
    input_load_seconds = 0.0
    for position, index in enumerate(indices):
        load_started = time.perf_counter()
        sample = load_case(index)
        input_load_seconds += time.perf_counter() - load_started
        trajectory = int(sample["trajectory_index"])
        if formal and trajectory != VALIDATION_TRAJECTORIES[position]:
            raise ValueError(
                "benchmark must use the fixed, ordered Validation-24 registry"
            )
        if trajectory in case_ids:
            raise ValueError(
                "benchmark case loader returned duplicate trajectory identities"
            )
        case_ids.add(trajectory)
        initial = np.asarray(sample["initial"])
        if (
            initial.dtype != np.float32
            or initial.shape != (len(sample["points"]), 3)
            or not np.isfinite(initial).all()
        ):
            raise ValueError("benchmark initial state must be finite CPU float32 [N,3]")
        case = {
            "trajectory_index": trajectory,
            "node_count": len(initial),
            "cell_count": len(sample["cells"]),
        }
        cases.append(case)
        for phase, count in (("warmup", WARMUP_REPEATS), ("measure", MEASURED_REPEATS)):
            for repeat in range(count):
                draw = repeat + (100 if phase == "warmup" else 0)
                seed = seed_draw(trajectory, draw)
                result = measure_prediction(predict, sample, device)
                row = {
                    **case,
                    "phase": phase,
                    "repeat": repeat,
                    "sampling_seed": seed,
                    **result,
                }
                rows.append(row)
                with (output_dir / "samples.jsonl").open(
                    "a", encoding="utf-8"
                ) as stream:
                    stream.write(json.dumps(row, allow_nan=False) + "\n")
        del sample
    measured = [row for row in rows if row["phase"] == "measure"]
    warmups = [row for row in rows if row["phase"] == "warmup"]
    trajectory_rows = []
    for case in cases:
        samples = [
            row
            for row in measured
            if row["trajectory_index"] == case["trajectory_index"]
        ]
        preheated = [
            row
            for row in warmups
            if row["trajectory_index"] == case["trajectory_index"]
        ]
        complete = all(row["finite"] for row in samples + preheated)
        trajectory_rows.append(
            {
                **case,
                "complete": complete,
                "measured_repeats": len(samples),
                "failed_measurements": sum(not row["finite"] for row in samples),
                "failed_warmups": sum(not row["finite"] for row in preheated),
                "latency_seconds": (
                    distribution([row["inference_wall_seconds"] for row in samples])
                    if complete
                    else distribution([])
                ),
            }
        )
    latency = distribution(
        [row["latency_seconds"]["mean"] for row in trajectory_rows if row["complete"]]
    )
    mean_latency = latency["mean"]
    failed_measurements = sum(not row["finite"] for row in measured)
    failed_warmups = sum(not row["finite"] for row in warmups)
    measurement_seconds = sum(row["inference_wall_seconds"] for row in measured)
    summary = {
        "schema": "cylinderflow.performance_report.v1",
        "method": method,
        "formal": formal,
        "settings": settings,
        "environment": environment,
        "data_identity": data_identity,
        "provenance": provenance,
        "model": inventory,
        "case_registry": cases,
        "trajectory_metrics": trajectory_rows,
        "measured_samples": len(measured),
        "warmup_samples": len(warmups),
        "failed_measurements": failed_measurements,
        "failed_warmups": failed_warmups,
        "complete_trajectories": sum(row["complete"] for row in trajectory_rows),
        "latency_seconds": latency,
        "latency_aggregation": "mean of 3 measured repeats within each complete trajectory; equal trajectory weight",
        "latency_scope": "complete trajectories; failed measurements and warmups retained separately",
        "seconds_per_generated_frame": (
            None if mean_latency is None else mean_latency / FUTURE_FRAMES
        ),
        "serial_generated_frames_per_second": (
            None if not mean_latency else FUTURE_FRAMES / mean_latency
        ),
        "projected_gpu_hours_per_100_trajectories": (
            None
            if device.type != "cuda" or mean_latency is None
            else 100 * mean_latency / 3600
        ),
        "measured_inference_gpu_hours": (
            measurement_seconds / 3600 if device.type == "cuda" else None
        ),
        "cuda_peak_allocated_bytes": (
            max((row["cuda_peak_allocated_bytes"] for row in measured), default=None)
            if device.type == "cuda"
            else None
        ),
        "cuda_peak_reserved_bytes": (
            max((row["cuda_peak_reserved_bytes"] for row in measured), default=None)
            if device.type == "cuda"
            else None
        ),
        "model_load_seconds": model_load_seconds,
        "input_load_seconds": input_load_seconds,
        "warmup_inference_seconds": sum(
            row["inference_wall_seconds"] for row in warmups
        ),
        "measured_inference_seconds": measurement_seconds,
        "benchmark_wall_seconds": time.perf_counter() - started,
        "status": (
            "complete"
            if not failed_measurements and not failed_warmups
            else "completed_with_failures"
        ),
        "test_accessed": False,
        "training_cost_scope": "This measures inference cost; training/AE/cache preparation costs remain in their original run records.",
    }
    with (output_dir / "samples.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_json(
        output_dir / "failures.json",
        {"samples": [row for row in rows if not row["finite"]]},
    )
    write_json(output_dir / "summary.json", summary)
    return summary


def compare_reports(reports: list[dict]) -> dict:
    """Reject mixed hardware/runtime, registry or incomplete latency evidence."""
    if len(reports) < 2 or len({report["method"] for report in reports}) != len(
        reports
    ):
        raise ValueError(
            "comparison requires distinct methods and at least two reports"
        )
    reference = reports[0]
    mismatches = []
    for report in reports:
        if not report.get("formal") or report.get("status") != "complete":
            mismatches.append(f"{report['method']}: requires complete formal benchmark")
        for field in (
            "schema",
            "settings",
            "environment",
            "data_identity",
            "case_registry",
        ):
            if report.get(field) != reference.get(field):
                mismatches.append(f"{report['method']}: mismatched {field}")
    if mismatches:
        raise ValueError("; ".join(mismatches))
    return {
        "schema": "cylinderflow.performance_comparison.v1",
        "settings": reference["settings"],
        "environment": reference["environment"],
        "data_identity": reference["data_identity"],
        "case_registry": reference["case_registry"],
        "methods": [
            {
                key: report[key]
                for key in (
                    "method",
                    "provenance",
                    "model",
                    "latency_seconds",
                    "seconds_per_generated_frame",
                    "serial_generated_frames_per_second",
                    "projected_gpu_hours_per_100_trajectories",
                    "cuda_peak_allocated_bytes",
                    "cuda_peak_reserved_bytes",
                )
            }
            for report in reports
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare complete performance reports with matched hardware and runtime."
    )
    parser.add_argument("--reports", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    reports = [
        json.loads(file_path.read_text(encoding="utf-8")) for file_path in args.reports
    ]
    write_json(args.output, compare_reports(reports))


if __name__ == "__main__":
    main()
