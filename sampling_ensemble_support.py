"""Repeated physical UVP predictions on Airfoil Validation100."""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import os
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
    if method == "dit":
        from graph_dit.data import DT
        from graph_dit.evaluate import Predictor, load_selected

        selected = json.loads((args.run / "selection.json").read_text(encoding="utf-8"))
        if not selected.get("complete_stage"):
            raise ValueError("finish the allocated stage before using its selection")
        model, checkpoint = load_selected(
            args.run / selected["checkpoint"], args.artifacts, selected["weights"]
        )
        if checkpoint["config"]["training"]["precision"] != "fp32":
            raise ValueError("Airfoil sampling requires the established FP32 model")
        model.to(device).eval().float()
        predictor = Predictor(
            model,
            args.artifacts,
            args.data_dir,
            device,
            "fp32",
            sampling_steps=args.sampling_steps,
        )
        return dict(
            indices=list(predictor.data.splits["validation"]),
            initial=predictor.load_case,
            target=lambda index: predictor.data.evaluation(index)["field"],
            predict=lambda sample, seed: predictor.predict(
                sample, seed, diagnostics=False
            )[0],
            close=lambda: None,
            models=[model, predictor.codec.autoencoder],
            dt=DT,
            data_identity=predictor.data.identity(),
            provenance={
                "checkpoint_id": checkpoint["checkpoint_id"],
                "weights": selected["weights"],
                "update": checkpoint["update"],
                "artifact_id": checkpoint["artifact_id"],
                "training_seed": checkpoint["config"]["seed"],
                "configuration": checkpoint["config"],
                "normalization": predictor.identity["normalization"],
                "sampling_steps": args.sampling_steps,
                "sampling_schedule": "DDIM",
            },
        )
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

        def predict(sample: dict, seed: int) -> np.ndarray:
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
                "sampling_steps_per_frame": 4,
                "autoregressive_steps": 64,
                "sampling_schedule": "DDPM",
            },
        )
    if method != "text2pde":
        raise ValueError(f"unsupported sampling method: {method}")
    from dataset.datamodule import FluidsDataModule
    from dataset.cylinderflow_stride8 import (
        CylinderFlowStride8TrajectoryDataset,
        FRAME_DT,
    )
    from modules.utils import get_yaml
    from tools.cylinderflow_stride8.benchmark import forecast_initial
    from tools.cylinderflow_stride8.evaluate_joint64 import (
        instantiate_model,
        make_sampler,
    )
    from tools.cylinderflow_stride8.protocol import DDIM_STEPS, validate_locked_config

    config = get_yaml(args.config)
    validate_locked_config(config, "ldm")
    normalizer = config["data"]["normalizer"]
    if not normalizer.get("use_norm") or normalizer.get("recalculate"):
        raise ValueError("reuse the frozen Train-only normalizer for sampling")
    Path(normalizer["stat_path"]).resolve(strict=True)
    datamodule = FluidsDataModule(config["data"])
    dataset = datamodule.val_dataset
    if (
        not isinstance(dataset, CylinderFlowStride8TrajectoryDataset)
        or dataset.split != "validation"
        or len(dataset) != 100
    ):
        raise ValueError("formal Airfoil Validation100 required")
    expected = np.asarray(dataset.normalizer_values, dtype=np.float64)
    actual = np.asarray(
        [
            float(getattr(datamodule.normalizer, name))
            for name in ("u_mean", "u_std", "v_mean", "v_std", "p_mean", "p_std")
        ]
    )
    if not np.allclose(actual, expected, rtol=1e-6, atol=1e-8):
        raise ValueError("normalizer differs from the Airfoil Train manifest")
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
        predict=lambda sample, seed: forecast_initial(model, sampler, sample),
        close=close,
        models=[model],
        dt=FRAME_DT,
        data_identity={
            "dataset_repository": "dm-meshgraphnets/airfoil",
            "dataset_revision": "airfoil.uvp.stride8.first75.v1",
            "dt": FRAME_DT,
            "evaluation_raw_frame_indices": list(range(0, 513, 8)),
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
            "sampling_steps": DDIM_STEPS,
            "sampling_schedule": "DDIM",
        },
    )


def draw(adapter: dict, sample: dict, label: int) -> tuple[np.ndarray, int]:
    index = int(sample["trajectory_index"])
    seed = ((label + 1) * 1_000_003 + index * 9_176) % (2**31 - 1)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return adapter["predict"](sample, seed), seed


def ensemble(adapter: dict, sample: dict, count: int) -> np.ndarray:
    total = None
    for label in range(count):
        values = draw(adapter, sample, label)[0].astype(np.float64)
        total = values if total is None else total + values
    mean = (total / count).astype(np.float32)
    mean[0] = sample["initial"]
    return mean


def input_files(method: str, args: argparse.Namespace) -> dict[str, Path]:
    """Resolve immutable model, normalization and data inputs without hashing."""
    if method == "dit":
        selection_file = args.run / "selection.json"
        selected = json.loads(selection_file.read_text(encoding="utf-8"))
        return {
            "selection": selection_file,
            "checkpoint": args.run / selected["checkpoint"],
            "autoencoder": args.artifacts / "autoencoder.pt",
            "artifacts": args.artifacts / "artifact.json",
            "latent_cache": args.artifacts / "train_latents.h5",
            "dataset": args.data_dir / "airfoil_stride8_75frames.h5",
            "manifest": args.data_dir / "airfoil_stride8_75frames_manifest.json",
        }
    files = {
        name: getattr(args, name) for name in ("config", "checkpoint", "ae_checkpoint")
    }
    if method == "aroma":
        files.update(
            dataset=args.dataset,
            manifest=args.manifest,
            normalization=args.prepared / "normalization.json",
            prepared=args.prepared / "ready.json",
        )
    else:
        from modules.utils import get_yaml

        config = get_yaml(args.config)
        manifest_file = Path(config["data"]["dataset"]["manifest"]).resolve()
        dataset_file = Path(config["data"]["dataset"]["data_path"])
        if not dataset_file.is_absolute():
            dataset_file = manifest_file.parent / dataset_file
        files.update(
            dataset=dataset_file,
            manifest=manifest_file,
            normalization=Path(config["data"]["normalizer"]["stat_path"]),
        )
    return files


def scalar_statistics(rows: list[dict]) -> dict:
    """Describe variation across draws for one fixed trajectory."""
    result = {}
    for key in rows[0]:
        if key in {"seed", "trajectory_index", "finite", "sample_seed"}:
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
    parser.add_argument("--output-dir", type=Path, required=True)
    if method == "dit":
        for name in ("run", "artifacts", "data-dir"):
            parser.add_argument("--" + name, type=Path, required=True)
        parser.add_argument("--sampling-steps", type=int, default=6)
    else:
        for name in ("config", "checkpoint", "ae-checkpoint"):
            parser.add_argument("--" + name, type=Path, required=True)
        if method == "aroma":
            for name in ("dataset", "manifest", "prepared"):
                parser.add_argument("--" + name, type=Path, required=True)
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
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        parser.error("use the standalone single-process evaluation entrypoint")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        parser.error("run in the verified production CUDA environment")
    prefix = {
        "dit": "graph_dit",
        "aroma": "cylinderflow",
        "text2pde": "tools.cylinderflow_stride8",
    }[method]
    performance = importlib.import_module(prefix + ".performance")
    metrics = importlib.import_module(prefix + ".metrics")
    predictions = importlib.import_module(prefix + ".predictions")
    if method == "dit":
        from graph_dit.train import configure_runtime

        configure_runtime(device, "fp32")
    performance.configure(device, args.threads)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    write = performance.write_json
    adapter = None
    started = time.perf_counter()
    current_case = None
    try:
        files = input_files(method, args)
        inputs = {name: file_identity(value) for name, value in files.items()}
        environment = performance.runtime_identity(device)
        with torch.inference_mode(), torch.autocast("cuda", enabled=False):
            adapter = load_adapter(method, args, device)
            if not np.isclose(adapter["dt"], 0.0016):
                raise ValueError("Airfoil requires stored dt=0.0016")
            performance.sync(device)
            load_seconds = time.perf_counter() - started
            revision = subprocess.run(
                [
                    "git",
                    "-C",
                    str(Path(__file__).resolve().parent),
                    "rev-parse",
                    "HEAD",
                ],
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
            write(
                args.output_dir / "manifest.json",
                {
                    "schema": "airfoil.sampling_ensemble.v1",
                    "method": method,
                    "physical_dt": adapter["dt"],
                    "initial_frame": "observed UVP",
                    "future_boundary": "predict all nodes and channels",
                    "pool": "nested prefixes of independent draws",
                    "inputs": inputs,
                    "source_commit": revision,
                    "environment": environment,
                    "provenance": adapter["provenance"],
                    "data_identity": adapter["data_identity"],
                    "samples": args.samples,
                    "ensemble_sizes": sizes,
                    "test_accessed": False,
                    "seed_rule": "((label+1)*1000003 + trajectory_index*9176) mod (2**31-1)",
                    "variance": "unbiased per-node physical UVP variance across draws; ddof=1",
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
                target = None
                member_seeds = []
                case_dir = args.output_dir / "fields" / str(trajectory)
                case_dir.mkdir(parents=True)
                mean = second = None
                rows = []
                for label in range(args.samples):
                    current_case = {
                        "trajectory_index": trajectory,
                        "sampling_label": label,
                    }
                    prediction, seed = draw(adapter, sample, label)
                    member_seeds.append(seed)
                    if target is None:
                        target = adapter["target"](index)
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
                        seed=label,
                        sample_seed=seed,
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
                        if not score["finite"]:
                            raise FloatingPointError(
                                f"nonfinite ensemble metrics: {trajectory}, {count}"
                            )
                        means[count].append(score)
                        predictions.save_prediction(
                            case_dir / f"mean_k{count}.npz",
                            prediction=mean.astype(np.float32),
                            pre_boundary=mean.astype(np.float32),
                            target=target,
                            points=sample["points"],
                            cells=sample["cells"],
                            node_type=sample["node_type"],
                            trajectory_index=trajectory,
                            seed=0,
                            provenance={
                                **adapter["provenance"],
                                "ensemble_size": count,
                                "aggregation": "physical_uvp_mean",
                                "member_prng_seeds": list(member_seeds),
                            },
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
                    "area_time_rms_std_uvp": np.sqrt(variance_uvp).tolist(),
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
                    trajectory_index=trajectory,
                    member_prng_seeds=np.asarray(member_seeds, dtype=np.int64),
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
                    [row for row in member_rows if row["seed"] == label]
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
                "mean_area_time_variance_uvp": np.mean(
                    [item["area_time_mean_variance_uvp"] for item in variations], axis=0
                ).tolist(),
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
                    current_case = {"stage": "timing", "ensemble_size": count}
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
                            "timing_includes": [
                                "member RNG reseeding",
                                "float64 mean reduction",
                            ],
                            "timing_member_labels": list(range(count)),
                        },
                        models=adapter["models"],
                        model_load_seconds=load_seconds,
                    )
                    if timing["status"] != "complete":
                        raise RuntimeError(f"timing k={count} has failed predictions")
            if performance.runtime_identity(device) != environment:
                raise ValueError("runtime settings changed")
            if inputs != {name: file_identity(value) for name, value in files.items()}:
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
            {
                "exit_code": 1,
                "case": current_case,
                "error": f"{type(error).__name__}: {error}",
            },
        )
        raise
    finally:
        if adapter is not None:
            adapter["close"]()
