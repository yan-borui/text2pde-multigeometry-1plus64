"""Repeated physical predictions and nested ensemble evaluation on Validation100."""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import random
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch


def file_identity(source: Path) -> dict[str, Any]:
    info = source.stat()
    return {
        "file": str(source.resolve()),
        "bytes": info.st_size,
        "mtime_ns": info.st_mtime_ns,
    }


def load_adapter(method: str, args: argparse.Namespace, device: torch.device) -> dict:
    if method == "aroma":
        from cylinderflow.data import Dataset, DT
        from cylinderflow.engine import config_from_file, open_prepared, load_ae
        from cylinderflow.engine import validate_dependency
        from cylinderflow.models import make_model, rollout
        from cylinderflow.runtime import load_checkpoint

        if any(value is None for value in (args.dataset, args.manifest, args.prepared)):
            raise ValueError("AROMA requires --dataset, --manifest and --prepared")
        dataset = Dataset(args.dataset, args.manifest)
        config = config_from_file(args.config)
        if config["method"] != method:
            raise ValueError("configuration method mismatch")
        stats = open_prepared(dataset, config, args.prepared)
        checkpoint = load_checkpoint(args.checkpoint)
        if checkpoint["stage"] != "dynamics":
            raise ValueError("AROMA requires a dynamics checkpoint")
        validate_dependency(checkpoint, config, stats, "dynamics")
        model = make_model(checkpoint["config"], "dynamics", device)
        model.load_state_dict(checkpoint["model"], strict=True)
        model.eval().float()
        ae, _ = load_ae(
            config, stats, args.ae_checkpoint, device, checkpoint["ae_checkpoint_id"]
        )
        ae.eval().float()
        indices = list(dataset.splits["validation"])

        def predict(sample: dict) -> np.ndarray:
            return rollout(
                model,
                checkpoint["config"],
                sample,
                stats,
                args.prepared,
                device,
                autoencoder=ae,
                measure_stages=False,
                return_raw=False,
            )[0]

        def initial(index: int) -> dict:
            return {**dataset.initial(index), "trajectory_index": index}

        return dict(
            indices=indices,
            initial=initial,
            target=lambda index: dataset.evaluation(index)["field"],
            predict=predict,
            close=lambda: None,
            models=[model, ae],
            dt=DT,
            data_identity=dataset.identity(),
            provenance={
                "checkpoint_id": checkpoint["checkpoint_id"],
                "ae_checkpoint_id": checkpoint["ae_checkpoint_id"],
                "configuration": checkpoint["config"],
                "training_seed": checkpoint["settings"]["seed"],
                "normalization": stats,
            },
        )
    from dataset.datamodule import FluidsDataModule
    from dataset.cylinderflow_stride8 import CylinderFlowStride8TrajectoryDataset
    from modules.utils import get_yaml
    from tools.cylinderflow_stride8.benchmark import forecast_initial
    from tools.cylinderflow_stride8.evaluate_joint64 import (
        instantiate_model,
        make_sampler,
    )
    from tools.cylinderflow_stride8.protocol import validate_locked_config

    config = get_yaml(args.config)
    validate_locked_config(config, "ldm")
    datamodule = FluidsDataModule(config["data"])
    dataset = datamodule.val_dataset
    if (
        not isinstance(dataset, CylinderFlowStride8TrajectoryDataset)
        or dataset.split != "validation"
        or len(dataset) != 100
    ):
        raise ValueError("formal CylinderFlow Validation100 required")
    model, update = instantiate_model(
        config, args.checkpoint, args.ae_checkpoint, datamodule, device
    )
    model.eval().float()
    sampler = make_sampler(model)

    def close() -> None:
        datamodule.train_dataset.close()
        dataset.close()

    return dict(
        indices=list(range(100)),
        initial=dataset.initial,
        target=lambda index: dataset.__getitem__(index, eval=True)["field"].numpy(),
        predict=lambda sample: forecast_initial(model, sampler, sample),
        close=close,
        models=[model],
        dt=0.08,
        data_identity={
            "split": "validation",
            "trajectory_indices": list(dataset.trajectory_indices),
            "normalization": list(dataset.normalizer_values),
        },
        provenance={
            "checkpoint_id": model._stride8_checkpoint_id,
            "checkpoint_update": update,
            "configuration": config,
            "dependencies": model._stride8_dependencies,
            "training_seed": config["training"]["seed"],
        },
    )


def draw(adapter: dict, sample: dict, label: int) -> tuple[np.ndarray, int]:
    index = int(sample["trajectory_index"])
    seed = ((label + 1) * 1_000_003 + index * 9_176) % (2**31 - 1)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return adapter["predict"](sample), seed


def ensemble(adapter: dict, sample: dict, count: int) -> np.ndarray:
    total = None
    for label in range(count):
        values = draw(adapter, sample, label)[0].astype(np.float64)
        total = values if total is None else total + values
    return (total / count).astype(np.float32)


def scalar_statistics(rows: list[dict]) -> dict:
    """Describe variation across draws for one fixed trajectory."""
    result = {}
    for key in rows[0]:
        if key in {"seed", "trajectory_index", "finite", "label"}:
            continue
        values = [row.get(key) for row in rows]
        if all(
            isinstance(v, (int, float)) and not isinstance(v, bool) and np.isfinite(v)
            for v in values
        ):
            result[key] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values, ddof=1)),
                "count": len(values),
            }
    return result


def run(method: str) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("config", "checkpoint", "ae-checkpoint", "output-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in ("dataset", "manifest", "prepared"):
        parser.add_argument("--" + name, type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument(
        "--ensemble-sizes", type=int, nargs="+", default=[1, 2, 4, 8, 16]
    )
    parser.add_argument("--skip-timing", action="store_true")
    args = parser.parse_args()
    sizes = sorted(set(args.ensemble_sizes))
    if args.samples < 2 or not sizes or sizes[0] < 1 or sizes[-1] > args.samples:
        parser.error("samples >= 2 and 1 <= ensemble sizes <= samples required")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        parser.error("run in the verified production CUDA environment")
    prefix = "cylinderflow" if method == "aroma" else "tools.cylinderflow_stride8"
    performance = importlib.import_module(prefix + ".performance")
    metrics = importlib.import_module(prefix + ".metrics")
    performance.configure(device, args.threads)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    write = performance.write_json
    adapter = None
    inputs = {
        name: file_identity(getattr(args, name))
        for name in ("config", "checkpoint", "ae_checkpoint")
    }
    started = time.perf_counter()
    try:
        environment = performance.runtime_identity(device)
        with torch.inference_mode(), torch.autocast("cuda", enabled=False):
            adapter = load_adapter(method, args, device)
            performance.sync(device)
            load_seconds = time.perf_counter() - started
            revision = subprocess.run(
                ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
            ).stdout.strip()
            write(
                args.output_dir / "manifest.json",
                {
                    "schema": "sampling_ensemble.v1",
                    "method": method,
                    "inputs": inputs,
                    "source_commit": revision,
                    "environment": environment,
                    "provenance": adapter["provenance"],
                    "data_identity": adapter["data_identity"],
                    "samples": args.samples,
                    "ensemble_sizes": sizes,
                    "test_accessed": False,
                    "seed_rule": "((label+1)*1000003 + trajectory_index*9176) mod (2**31-1)",
                    "variance": "unbiased per-cell physical UVP variance across draws; ddof=1",
                    "pressure_variance": "raw physical pressure; score uses native gauge adjustment",
                    "training_seed_variation": False,
                },
            )
            member_rows = []
            means = {k: [] for k in sizes}
            variations = []
            seen = []
            for ordinal, index in enumerate(adapter["indices"]):
                sample = adapter["initial"](index)
                trajectory = int(sample["trajectory_index"])
                if trajectory not in range(1000, 1100) or trajectory in seen:
                    raise ValueError("unexpected or duplicate Validation trajectory")
                seen.append(trajectory)
                target = adapter["target"](index)
                case_dir = args.output_dir / "fields" / str(trajectory)
                case_dir.mkdir(parents=True)
                mean = second = None
                rows = []
                for label in range(args.samples):
                    prediction, seed = draw(adapter, sample, label)
                    if (
                        prediction.shape != target.shape
                        or not np.isfinite(prediction).all()
                    ):
                        raise FloatingPointError(
                            f"invalid prediction: {trajectory}, {label}"
                        )
                    if not np.array_equal(prediction[0], sample["initial"]):
                        raise ValueError("initial state changed")
                    values = prediction.astype(np.float64)
                    if mean is None:
                        mean = np.zeros_like(values)
                        second = np.zeros_like(values)
                    delta = values - mean
                    mean += delta / (label + 1)
                    second += delta * (values - mean)
                    row = dict(
                        metrics.compute_metrics(
                            prediction,
                            target,
                            sample["points"],
                            sample["cells"],
                            sample["node_type"],
                            adapter["dt"],
                        ),
                        trajectory_index=trajectory,
                        seed=seed,
                        label=label,
                    )
                    if not row["finite"]:
                        raise FloatingPointError(
                            f"nonfinite metrics: {trajectory}, {label}"
                        )
                    rows.append(row)
                    member_rows.append(row)
                    with (args.output_dir / "members.jsonl").open(
                        "a", encoding="utf-8"
                    ) as stream:
                        stream.write(json.dumps(row, allow_nan=False) + "\n")
                    count = label + 1
                    if count in means:
                        score = dict(
                            metrics.compute_metrics(
                                mean.astype(np.float32),
                                target,
                                sample["points"],
                                sample["cells"],
                                sample["node_type"],
                                adapter["dt"],
                            ),
                            trajectory_index=trajectory,
                            seed=0,
                        )
                        means[count].append(score)
                        np.savez_compressed(
                            case_dir / f"mean_k{count}.npz",
                            prediction=mean.astype(np.float32),
                            trajectory_index=trajectory,
                            count=count,
                        )
                variance = np.maximum(second / (args.samples - 1), 0)
                weights = metrics.node_area_weights(sample["points"], sample["cells"])
                weights /= weights.sum()
                variance_uvp = np.einsum("tnc,n->c", variance[1:], weights) / 64
                variation = {
                    "trajectory_index": trajectory,
                    "metrics": scalar_statistics(rows),
                    "area_time_mean_variance_uvp": variance_uvp.tolist(),
                }
                variations.append(variation)
                write(case_dir / "statistics.json", variation)
                np.savez_compressed(
                    case_dir / "variance.npz",
                    variance_uvp=variance.astype(np.float32),
                    mean_uvp=mean.astype(np.float32),
                    target=target,
                    points=sample["points"],
                    cells=sample["cells"],
                    node_type=sample["node_type"],
                    samples=args.samples,
                    ddof=1,
                    dt=adapter["dt"],
                )
                write(
                    args.output_dir / "progress.json",
                    {
                        "state": "sampling",
                        "trajectories": ordinal + 1,
                        "member_count": len(member_rows),
                    },
                )
                print(f"Validation {ordinal + 1}/100", flush=True)
            if sorted(seen) != list(range(1000, 1100)):
                raise ValueError("incomplete Validation100")
            label_scores = [
                metrics.summarize_trajectories(
                    [row for row in member_rows if row["label"] == label]
                )["selection_uv_relative_rmse"]
                for label in range(args.samples)
            ]
            summaries = {
                str(k): metrics.summarize_trajectories(rows)
                for k, rows in means.items()
            }
            result = {
                "method": method,
                "status": "quality_complete",
                "trajectory_count": 100,
                "single_draw": metrics.summarize_trajectories(member_rows),
                "uv_score_by_draw_label": label_scores,
                "uv_score_mean": float(np.mean(label_scores)),
                "uv_score_std_across_draw_labels": float(np.std(label_scores, ddof=1)),
                "trajectory_sampling_statistics": variations,
                "ensemble": summaries,
                "test_accessed": False,
            }
            write(args.output_dir / "summary.json", result)
            with (args.output_dir / "ensemble.csv").open(
                "w", newline="", encoding="utf-8"
            ) as stream:
                writer = csv.writer(stream)
                writer.writerow(
                    ["method", "ensemble_size", "validation100_uv_relative_rmse"]
                )
                for k in sizes:
                    writer.writerow(
                        [method, k, summaries[str(k)]["selection_uv_relative_rmse"]]
                    )
            if not args.skip_timing:
                monitor = set(performance.VALIDATION_TRAJECTORIES)
                selected = [
                    index
                    for index, trajectory in zip(adapter["indices"], seen)
                    if trajectory in monitor
                ]
                for count in sizes:
                    timing = performance.benchmark(
                        method=method,
                        indices=selected,
                        load_case=adapter["initial"],
                        predict=lambda sample, k=count: ensemble(adapter, sample, k),
                        device=device,
                        output_dir=args.output_dir / "timing" / f"k{count}",
                        data_identity=adapter["data_identity"],
                        provenance={
                            **adapter["provenance"],
                            "ensemble_size": count,
                            "aggregation": "physical UVP mean before scoring",
                        },
                        models=adapter["models"],
                        model_load_seconds=load_seconds,
                    )
                    if timing["status"] != "complete":
                        raise RuntimeError(f"timing k={count} has failed predictions")
            if performance.runtime_identity(device) != environment:
                raise ValueError("runtime settings changed")
            if inputs != {name: file_identity(getattr(args, name)) for name in inputs}:
                raise ValueError("input identity changed")
            result["status"] = "complete"
            result["timing_complete"] = not args.skip_timing
            write(args.output_dir / "summary.json", result)
            write(
                args.output_dir / "exit.json",
                {"exit_code": 0, "timing_complete": not args.skip_timing},
            )
    except Exception as error:
        write(
            args.output_dir / "exit.json",
            {"exit_code": 1, "error": f"{type(error).__name__}: {error}"},
        )
        raise
    finally:
        if adapter is not None:
            adapter["close"]()
