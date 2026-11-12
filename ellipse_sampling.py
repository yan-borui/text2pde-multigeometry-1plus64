"""Ellipse member/ensemble evaluation on four GPUs, or matched single-GPU timing."""

from __future__ import annotations

import argparse
import importlib
import json
import os
from pathlib import Path
import random
import time

import numpy as np
import torch

from megaflow_data.contract import DT, FRAME_INDICES, REPOSITORY, REVISION, splits


def input_identity(args) -> dict:
    """Record local input identities without computing file hashes."""
    result = {}
    for name in ("config", "checkpoint", "ae_checkpoint", "dataset", "manifest"):
        file = getattr(args, name)
        if file is not None:
            stat = file.stat()
            result[name] = {
                "file": str(file.resolve()),
                "bytes": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
            }
    return result


def load_adapter(args, device, output):
    """Bind fixed dynamics and AE weights to the original Ellipse data contract."""
    if args.method in ("aroma", "meshgraphnets", "eagle"):
        from cylinderflow.data import Dataset
        from cylinderflow.engine import (
            config_from_file,
            open_prepared,
            load_ae,
            validate_dependency,
        )
        from cylinderflow.models import make_model, rollout, prepare_clusters
        from cylinderflow.runtime import load_checkpoint

        if any(value is None for value in (args.dataset, args.manifest, args.prepared)):
            raise ValueError(
                "graph baselines require --dataset, --manifest, --prepared"
            )
        data = Dataset(args.dataset, args.manifest)
        config = config_from_file(args.config)
        method = "mgn" if args.method == "meshgraphnets" else args.method
        if config["method"] != method:
            raise ValueError("configuration method mismatch")
        stats = open_prepared(data, config, args.prepared)
        checkpoint = load_checkpoint(args.checkpoint)
        stage = checkpoint["stage"]
        if stage != ("dynamics" if args.method == "aroma" else "main"):
            raise ValueError("forecast weights must match the declared method")
        validate_dependency(checkpoint, config, stats, stage)
        model = make_model(checkpoint["config"], stage, device)
        model.load_state_dict(checkpoint["model"], strict=True)
        model.eval().float()
        ae = None
        if args.method == "aroma":
            if args.ae_checkpoint is None:
                raise ValueError("AROMA needs its matching selected AE")
            ae, _ = load_ae(
                config,
                stats,
                args.ae_checkpoint,
                device,
                checkpoint["ae_checkpoint_id"],
            )
            ae.eval().float()
        prepared = args.prepared
        indices = list(data.splits[args.split])
        if args.method == "eagle":
            rank = int(os.environ.get("RANK", "0"))
            world = int(os.environ.get("WORLD_SIZE", "1"))
            prepared = output / f"geometry_rank_{rank:03d}"
            prepared.mkdir(parents=True)
            prepare_clusters(
                data,
                prepared / "clusters",
                checkpoint["settings"]["seed"],
                indices=indices[rank::world],
            )

        def initial(index):
            sample = {**data.initial(index), "trajectory_index": index}
            if args.method == "eagle":
                sample["clusters"] = np.load(
                    prepared / "clusters" / f"trajectory_{index:04d}.npy",
                    allow_pickle=False,
                )
            return sample

        return {
            "indices": indices,
            "ids": indices,
            "initial": initial,
            "target": lambda index: data.evaluation(index)["field"],
            "predict": lambda sample: rollout(
                model,
                checkpoint["config"],
                sample,
                stats,
                prepared,
                device,
                autoencoder=ae,
                measure_stages=False,
                return_raw=False,
            )[0],
            "models": [model] + ([ae] if ae else []),
            "close": lambda: None,
            "identity": data.identity(),
            "provenance": {
                "checkpoint_id": checkpoint["checkpoint_id"],
                "ae_checkpoint_id": checkpoint.get("ae_checkpoint_id"),
                "training_seed": checkpoint["settings"]["seed"],
                "configuration": checkpoint["config"],
                "normalization": stats,
                "sampling_steps_per_frame": 4 if args.method == "aroma" else 1,
                "autoregressive_steps": 64,
            },
        }
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
    if args.ae_checkpoint is None:
        raise ValueError("Text2PDE needs its matching selected AE")
    normalizer = config["data"]["normalizer"]
    if not normalizer.get("use_norm") or normalizer.get("recalculate"):
        raise ValueError("reuse the frozen Train-only normalizer")
    module = FluidsDataModule(config["data"])
    data = module.val_dataset
    if args.split == "test":
        data = CylinderFlowStride8TrajectoryDataset(
            config["data"]["dataset"]["manifest"],
            data_path=config["data"]["dataset"]["data_path"],
            split="test",
            stage="ldm",
            return_metadata=True,
        )
    if len(data) != 100 or data.split != args.split:
        raise ValueError("the complete declared split is required")
    actual = [
        float(getattr(module.normalizer, name))
        for name in ("u_mean", "u_std", "v_mean", "v_std", "p_mean", "p_std")
    ]
    if not np.allclose(actual, data.normalizer_values, rtol=1e-6, atol=1e-8):
        raise ValueError("normalizer differs from retained Train statistics")
    model, update = instantiate_model(
        config, args.checkpoint, args.ae_checkpoint, module, device
    )
    model.eval().float()
    sampler = make_sampler(model)

    def close():
        module.train_dataset.close()
        module.val_dataset.close()
        if args.split == "test":
            data.close()

    return {
        "indices": list(range(100)),
        "ids": list(data.trajectory_indices),
        "initial": data.initial,
        "target": lambda index: data.__getitem__(index, eval=True)["field"].numpy(),
        "predict": lambda sample: forecast_initial(model, sampler, sample),
        "models": [model],
        "close": close,
        "identity": {
            "repository": REPOSITORY,
            "revision": REVISION,
            "split": args.split,
            "trajectory_indices": list(data.trajectory_indices),
            "raw_frame_indices": list(FRAME_INDICES),
            "dt": DT,
            "normalization": list(data.normalizer_values),
        },
        "provenance": {
            "checkpoint_id": model._stride8_checkpoint_id,
            "update": update,
            "configuration": config,
            "dependencies": model._stride8_dependencies,
            "training_seed": config["training"]["seed"],
            "sampling_steps": 20,
        },
    }


def draw(adapter, sample, label):
    seed = ((label + 1) * 1000003 + int(sample["trajectory_index"]) * 9176) % (
        2**31 - 1
    )
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return adapter["predict"](sample), seed


def mean_prediction(adapter, sample, count, offset=0):
    total = None
    for label in range(count):
        values = draw(adapter, sample, offset + label)[0].astype(np.float64)
        total = values if total is None else total + values
    result = (total / count).astype(np.float32)
    result[0] = sample["initial"]
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--method",
        choices=("aroma", "text2pde", "meshgraphnets", "eagle"),
        required=True,
    )
    for name in ("config", "checkpoint", "output-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in ("dataset", "manifest", "prepared", "ae-checkpoint"):
        parser.add_argument("--" + name, type=Path)
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument(
        "--ensemble-sizes", type=int, nargs="+", default=[1, 2, 4, 8, 16]
    )
    parser.add_argument("--sampling-label-offset", type=int, default=0)
    parser.add_argument("--benchmark", action="store_true")
    args = parser.parse_args()
    sizes = sorted(set(args.ensemble_sizes))
    if not sizes or sizes[0] < 1 or args.sampling_label_offset < 0:
        parser.error("positive K and nonnegative sampling label offset required")
    if args.method in ("meshgraphnets", "eagle") and sizes != [1]:
        parser.error("deterministic baselines use --ensemble-sizes 1")
    if args.benchmark and args.split != "validation":
        parser.error("timing uses the fixed Validation24 registry")
    prefix = (
        "tools.cylinderflow_stride8" if args.method == "text2pde" else "cylinderflow"
    )
    context_class = importlib.import_module(prefix + ".distributed").Context
    performance = importlib.import_module(prefix + ".performance")
    metrics = importlib.import_module(prefix + ".metrics")
    predictions = importlib.import_module(prefix + ".predictions")
    ctx = context_class(
        1 if args.benchmark else 4, "cuda:0" if args.benchmark else "cuda"
    )
    adapter = None
    try:
        performance.configure(ctx.device)
        ctx.primary_call(lambda: args.output_dir.mkdir(parents=True, exist_ok=False))
        start = time.perf_counter()
        inputs = input_identity(args)
        adapter = load_adapter(args, ctx.device, args.output_dir)
        load_seconds = time.perf_counter() - start
        if tuple(adapter["ids"]) != splits()[args.split]:
            raise ValueError(
                "adapter trajectory ordering differs from published Ellipse split"
            )
        provenance = {
            **adapter["provenance"],
            "method": args.method,
            "split": args.split,
            "test_accessed": args.split == "test",
            "data_identity": adapter["identity"],
            "checkpoint_file": str(args.checkpoint.resolve()),
            "ae_checkpoint_file": str(args.ae_checkpoint.resolve())
            if args.ae_checkpoint
            else None,
            "world_size": ctx.world,
            "ensemble_sizes": sizes,
            "sampling_label_offset": args.sampling_label_offset,
            "aggregation": "physical_uvp_mean",
            "dt": DT,
            "environment": performance.runtime_identity(ctx.device),
            "inputs": inputs,
        }
        ctx.primary_call(
            lambda: performance.write_json(
                args.output_dir / "identity.json", provenance
            )
        )
        if args.benchmark:
            indices = [
                adapter["indices"][i]
                for i in np.rint(np.linspace(0, 99, 24)).astype(int)
            ]
            for count in sizes:
                result = performance.benchmark(
                    method=args.method,
                    indices=indices,
                    load_case=adapter["initial"],
                    predict=lambda sample, k=count: mean_prediction(
                        adapter, sample, k, args.sampling_label_offset
                    ),
                    device=ctx.device,
                    output_dir=args.output_dir / f"k{count}",
                    data_identity=adapter["identity"],
                    provenance={**provenance, "ensemble_size": count},
                    models=adapter["models"],
                    model_load_seconds=load_seconds,
                    validation_count=24,
                )
                if result["status"] != "complete":
                    raise RuntimeError("benchmark contains failed predictions")
        else:

            def evaluate_shard():
                member_rows, means = [], {k: [] for k in sizes}
                folder = args.output_dir / f"rank_{ctx.rank:03d}"
                folder.mkdir()
                for index in adapter["indices"][ctx.rank :: ctx.world]:
                    sample = adapter["initial"](index)
                    trajectory = int(sample["trajectory_index"])
                    total = None
                    target = None
                    for label in range(sizes[-1]):
                        failure = None
                        try:
                            with torch.inference_mode():
                                prediction, seed = draw(
                                    adapter, sample, args.sampling_label_offset + label
                                )
                        except (RuntimeError, FloatingPointError) as error:
                            failure = f"{type(error).__name__}: {error}"
                            seed = None
                            prediction = np.full(
                                (65, len(sample["points"]), 3), np.nan, dtype=np.float32
                            )
                            prediction[0] = sample["initial"]
                        if target is None:
                            target = adapter["target"](index)
                        score = metrics.compute_metrics(
                            prediction,
                            target,
                            sample["points"],
                            sample["cells"],
                            sample["node_type"],
                            DT,
                        )
                        if not np.isfinite(score.get("uv_relative_rmse", np.nan)):
                            score = {"finite": False}
                        filename = (
                            folder
                            / f"trajectory_{trajectory:04d}_member{label:03d}.npz"
                        )
                        predictions.save_prediction(
                            filename,
                            prediction=prediction,
                            pre_boundary=prediction,
                            target=target,
                            points=sample["points"],
                            cells=sample["cells"],
                            node_type=sample["node_type"],
                            trajectory_index=trajectory,
                            seed=label,
                            provenance={
                                **provenance,
                                "prng_seed": seed,
                                "failure": failure,
                            },
                        )
                        row = {
                            **score,
                            "trajectory_index": trajectory,
                            "seed": label,
                            "error": failure,
                            "prediction_file": str(
                                filename.relative_to(args.output_dir)
                            ),
                        }
                        member_rows.append(row)
                        with (folder / "members.jsonl").open(
                            "a", encoding="utf-8"
                        ) as stream:
                            stream.write(json.dumps(row, allow_nan=True) + "\n")
                        total = (
                            prediction.astype(np.float64)
                            if total is None
                            else total + prediction
                        )
                        k = label + 1
                        if k in means:
                            mean = (total / k).astype(np.float32)
                            filename = (
                                folder / f"trajectory_{trajectory:04d}_mean_k{k}.npz"
                            )
                            values = metrics.compute_metrics(
                                mean,
                                target,
                                sample["points"],
                                sample["cells"],
                                sample["node_type"],
                                DT,
                            )
                            if not np.isfinite(values.get("uv_relative_rmse", np.nan)):
                                values = {"finite": False}
                            predictions.save_prediction(
                                filename,
                                prediction=mean,
                                pre_boundary=mean,
                                target=target,
                                points=sample["points"],
                                cells=sample["cells"],
                                node_type=sample["node_type"],
                                trajectory_index=trajectory,
                                seed=0,
                                provenance={**provenance, "ensemble_size": k},
                            )
                            means[k].append(
                                {
                                    **values,
                                    "trajectory_index": trajectory,
                                    "seed": 0,
                                    "prediction_file": str(
                                        filename.relative_to(args.output_dir)
                                    ),
                                }
                            )
                performance.write_json(
                    folder / "records.json", {"members": member_rows, "means": means}
                )

            ctx.all_call(evaluate_shard)

            def merge():
                shards = [
                    json.loads(
                        (args.output_dir / f"rank_{r:03d}/records.json").read_text()
                    )
                    for r in range(ctx.world)
                ]
                members = [row for shard in shards for row in shard["members"]]
                expected = {
                    (i, label) for i in adapter["ids"] for label in range(sizes[-1])
                }
                actual = [(r["trajectory_index"], r["seed"]) for r in members]
                if len(actual) != len(expected) or set(actual) != expected:
                    raise ValueError("missing or duplicate member trajectories")
                for k in sizes:
                    rows = [row for shard in shards for row in shard["means"][str(k)]]
                    if len(rows) != 100 or {r["trajectory_index"] for r in rows} != set(
                        adapter["ids"]
                    ):
                        raise ValueError("incomplete mean-field population")
                    summary = metrics.summarize_trajectories(rows)
                    summary["provenance"] = {**provenance, "ensemble_size": k}
                    performance.write_json(
                        args.output_dir / f"k{k}/summary.json", summary
                    )
                    performance.write_json(
                        args.output_dir / f"k{k}/cases.json", {"rows": rows}
                    )
                performance.write_json(
                    args.output_dir / "members.json", {"rows": members}
                )
                failures = [r for r in members if not r["finite"]]
                performance.write_json(
                    args.output_dir / "failures.json", {"failures": failures}
                )
                return len(failures)

            if ctx.primary_call(merge):
                raise RuntimeError(
                    "evaluation retained failed members; inspect failures.json"
                )
        if input_identity(args) != inputs:
            raise ValueError("input files changed during evaluation")
        ctx.primary_call(
            lambda: performance.write_json(
                args.output_dir / "exit.json", {"exit_code": 0}
            )
        )
    except Exception as error:
        if args.output_dir.exists():
            performance.write_json(
                args.output_dir / f"failure_rank_{ctx.rank:03d}.json",
                {"error": str(error), "exit_code": 1},
            )
        raise
    finally:
        if adapter is not None:
            adapter["close"]()
        ctx.close()


if __name__ == "__main__":
    main()
