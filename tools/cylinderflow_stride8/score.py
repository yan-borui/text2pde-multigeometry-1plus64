"""Recompute common physical metrics directly from saved complete predictions."""

import time

import numpy as np

from .metrics import compute_metrics, summarize_trajectories
from .predictions import boundary_metrics, validate_prediction
from .evaluation_io import append_json, write_csv, write_json

EVALUATOR_VERSION = "cylinderflow.physical_mesh.v1"


def score(inputs, output_dir):
    output_dir.mkdir(parents=True, exist_ok=False)
    rows, identities = [], set()
    for file_name in inputs:
        with np.load(file_name, allow_pickle=False) as bundle:
            validate_prediction(bundle)
            index, seed = int(bundle["trajectory_index"]), int(bundle["seed"])
            if (index, seed) in identities:
                raise ValueError("duplicate trajectory/sample in metric aggregation")
            identities.add((index, seed))
            started = time.perf_counter()
            metrics = compute_metrics(
                bundle["prediction"],
                bundle["target"],
                bundle["points"],
                bundle["cells"],
                bundle["node_type"],
                0.08,
            )
            metrics.update(
                boundary_metrics(
                    bundle["prediction"],
                    bundle["pre_boundary"],
                    bundle["target"][0],
                    bundle["node_type"],
                )
            )
            primary = metrics.get("uv_relative_rmse")
            metrics["finite"] = bool(
                metrics.get("finite") and primary is not None and np.isfinite(primary)
            )
            row = {
                "trajectory_index": index,
                "seed": seed,
                **metrics,
                "metrics_seconds": time.perf_counter() - started,
                "prediction_file": str(file_name),
            }
            rows.append(row)
            append_json(output_dir / "case_metrics.jsonl", row)
    summary = summarize_trajectories(rows)
    summary["evaluator"] = EVALUATOR_VERSION
    summary["scope"] = (
        "only the explicitly supplied prediction files; not checkpoint selection"
    )
    write_csv(output_dir / "case_metrics.csv", rows)
    write_csv(output_dir / "trajectory_metrics.csv", summary["trajectory_metrics"])
    write_json(output_dir / "summary.json", summary)
    write_json(
        output_dir / "failures.json",
        {"failures": [row for row in rows if not row["finite"]]},
    )
    return summary


def main() -> None:
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    score(args.inputs, args.output_dir)


if __name__ == "__main__":
    main()
