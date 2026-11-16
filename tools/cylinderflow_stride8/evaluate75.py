"""Evaluate 1+74 forecasts using the existing joint64 window continuation."""

from __future__ import annotations

import argparse
import json
import time
import traceback
from copy import copy
from pathlib import Path

import numpy as np
import torch

from mgn_data.test_data import TestData
from .sliding_adapter import Predictor, add_arguments
from .evaluation_io import write_json, write_csv

from .metrics import compute_metrics, summarize_trajectories
from .sliding_window import rollout, source_snapshot


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_arguments(parser)
    parser.add_argument("--scope", choices=("validation", "test"), default="validation")
    parser.add_argument("--test-data-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--indices", nargs="+", type=int)
    parser.add_argument("--case-count", type=int, default=100)
    parser.add_argument("--stride", type=int, default=32)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--save-members", action="store_true")
    args = parser.parse_args()
    if (args.scope == "test") != (args.test_data_dir is not None):
        parser.error("only --scope test requires --test-data-dir")
    if not 1 <= args.stride <= 64 or min(args.case_count, args.threads) < 1:
        parser.error("stride must be 1..64 and counts/threads positive")
    device = torch.device(args.device)
    if device.type != "cuda":
        parser.error("use the assigned production CUDA environment")
    args.ensemble_size = 1
    args.output_dir.mkdir(parents=True, exist_ok=False)
    write_json(
        args.output_dir / "request.json",
        {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
    )
    try:
        run(args, device)
    except Exception as error:
        write_json(
            args.output_dir / "failure.json",
            {
                "error": f"{type(error).__name__}: {error}",
                "traceback": traceback.format_exc(),
            },
        )
        raise


def run(args: argparse.Namespace, device: torch.device) -> None:
    """Generate every requested case before opening its future reference fields."""
    torch.cuda.set_device(device)
    torch.set_num_threads(args.threads)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    # Load fixed model/AE using the training binding, then attach the separate Test75 reader.
    native_args = copy(args)
    native_args.scope = "validation"
    native_args.test_data_dir = None
    predictor = Predictor(native_args, device)
    dataset = predictor.data.dataset
    if args.scope == "test":
        test_data = TestData(args.test_data_dir, "cylinderflow", expected_frames=75)
        test_data.bind_training_manifest(dataset.manifest)
        population = list(test_data.splits["test"])
        load_case = test_data.initial

        def load_reference(index: int) -> np.ndarray:
            return test_data.read(index, 0, 75)["field"]

        evaluation_data = test_data.identity()
    else:
        population = list(dataset.trajectory_indices)
        load_case = predictor.load_case

        def load_reference(index: int) -> np.ndarray:
            # Native group validation preserves source/split identity; fields are read only here.
            position = predictor.data.positions[index]
            _, _, group = dataset._trajectory_group(position)
            return np.asarray(group["uvp"][:75], dtype=np.float32)

        evaluation_data = dataset.manifest
    labels = (0, 1, 2)
    identity = {**predictor.identity, "evaluation_data": evaluation_data}
    indices = (
        args.indices if args.indices is not None else population[: args.case_count]
    )
    if (
        not indices
        or len(indices) != len(set(indices))
        or not set(indices) <= set(population)
    ):
        raise ValueError("indices must be unique global IDs from the selected split")
    identity.update(
        {
            "protocol": "cylinderflow.joint64.evaluate75.v1",
            "source": source_snapshot(args.output_dir),
            "scope": args.scope,
            "test_accessed": args.scope == "test",
            "indices": indices,
            "population_count": len(population),
            "sampling_labels": list(labels),
            "ensemble_size": args.ensemble_size,
            "stride": args.stride,
            "future_frames": 74,
            "reference_frames": [0, 74],
            "frame_dt": 0.08,
            "duration_seconds": 5.92,
            "conditioning": "observed frame zero; later windows use merged generated UVP",
            "window_aggregation": "physical member mean, then equal overlapping-window mean",
            "score_aggregation": "mean scores across labels within trajectory, then equal trajectory weight",
            "runtime": {
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "device": str(device),
                "gpu": torch.cuda.get_device_name(device),
                "precision": "fp32",
            },
        }
    )
    write_json(args.output_dir / "identity.json", identity)
    rows = []
    for index in indices:
        for label in labels:
            output = args.output_dir / f"trajectory_{index:04d}_seed_{label}"
            output.mkdir()
            row = {"trajectory_index": index, "seed": label, "finite": False}
            begin = time.perf_counter()
            try:
                sample = load_case(index)
                prediction, first, counts, windows = rollout(
                    predictor,
                    sample,
                    output,
                    future_frames=74,
                    stride=args.stride,
                    ensemble_size=args.ensemble_size,
                    sampling_label=label,
                    save_members=args.save_members,
                )
                # Preserve completed predictions even if reference/scoring fails.
                np.savez_compressed(
                    output / "prediction.npz",
                    prediction_schema=np.asarray("cylinderflow.joint64.evaluate75.v1"),
                    units=np.asarray("physical_uvp"),
                    prediction=prediction,
                    points=sample["points"],
                    cells=sample["cells"],
                    node_type=sample["node_type"],
                    contribution_count=counts,
                    raw_frame_indices=np.arange(75) * 8,
                    physical_time=np.arange(75) * 0.08,
                    trajectory_index=index,
                    seed=label,
                )
                target = load_reference(index)
                if target.shape != prediction.shape or target.shape[0] != 75:
                    raise ValueError(
                        "evaluation requires complete matching [75,N,3] references"
                    )
                if not np.isfinite(target).all():
                    raise ValueError("nonfinite reference")
                np.savez_compressed(
                    output / "reference.npz",
                    target=target,
                    physical_time=np.arange(75) * 0.08,
                )
                geometry = (
                    sample["points"],
                    sample["cells"],
                    sample["node_type"],
                    0.08,
                )
                row["full74"] = compute_metrics(
                    prediction,
                    target,
                    *geometry,
                    expected_frames=75,
                )
                row["full74"]["frame74_uv_relative_rmse"] = row["full74"][
                    "per_frame_uv_relative_rmse"
                ][74]
                row["prefix64"] = compute_metrics(
                    prediction[:65], target[:65], *geometry
                )
                row["first_window64"] = compute_metrics(first, target[:65], *geometry)
                row["inference_seconds"] = sum(
                    window["inference_seconds"] for window in windows
                )
                row["window_count"] = len(windows)
                row["member_window_samples"] = len(windows) * args.ensemble_size
                row["finite"] = all(
                    row[name].get("finite")
                    and row[name].get("uv_relative_rmse") is not None
                    and np.isfinite(row[name]["uv_relative_rmse"])
                    for name in ("full74", "prefix64", "first_window64")
                )
                if not row["finite"]:
                    raise FloatingPointError("nonfinite scored metric")
            except Exception as error:
                row["finite"] = False
                row["error"] = f"{type(error).__name__}: {error}"
                write_json(
                    output / "failure.json",
                    {
                        "error": row["error"],
                        "traceback": traceback.format_exc(),
                    },
                )
            row["seconds"] = time.perf_counter() - begin
            write_json(output / "metrics.json", row)
            rows.append(row)
            aggregates = {}
            for name in ("full74", "prefix64", "first_window64"):
                metrics = [
                    {
                        **item.get(name, {}),
                        "trajectory_index": item["trajectory_index"],
                        "seed": item["seed"],
                        "finite": item["finite"],
                    }
                    for item in rows
                ]
                aggregates[name] = summarize_trajectories(metrics)
                write_csv(
                    args.output_dir / f"{name}_trajectory_metrics.csv",
                    aggregates[name]["trajectory_metrics"],
                )
            write_json(
                args.output_dir / "summary.json",
                {
                    "requested_indices": indices,
                    "sampling_labels": list(labels),
                    "requested_samples": len(indices) * len(labels),
                    "completed_samples": len(rows),
                    "failed_samples": sum(not item["finite"] for item in rows),
                    "metric_future_frames": {
                        "full74": 74,
                        "prefix64": 64,
                        "first_window64": 64,
                    },
                    "aggregate": aggregates,
                    "cases": rows,
                },
            )
            print(
                json.dumps(
                    {
                        "trajectory_index": index,
                        "seed": label,
                        "finite": row["finite"],
                        "completed_samples": len(rows),
                    }
                ),
                flush=True,
            )
            dataset._geometry_cache.clear()
    if any(not row["finite"] for row in rows):
        raise RuntimeError("one or more evaluation cases failed; records retained")


if __name__ == "__main__":
    main()
