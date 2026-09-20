"""Historical Validation100 UV tables; copied unchanged into standalone repos."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import traceback

METHODS = ("mgn", "eagle", "aroma", "text2pde", "gladit")
LABELS = dict(zip(METHODS, ("MGN", "EAGLE", "AROMA", "Text2PDE", "GLaDiT")))


def write_json(destination: Path, value: object) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    temporary.replace(destination)


def read_json(source: Path) -> dict:
    return json.loads(source.read_text(encoding="utf-8"))


def stamp(source: Path) -> dict:
    info = source.stat()
    return {
        "file": str(source.resolve()),
        "bytes": info.st_size,
        "mtime_ns": info.st_mtime_ns,
    }


def metadata(source: Path, method: str) -> dict:
    import torch

    checkpoint = torch.load(source, map_location="cpu", weights_only=False)
    if method == "text2pde":
        from tools.cylinderflow_stride8.protocol import (
            checkpoint_identifier,
            validate_checkpoint_contract,
        )

        validate_checkpoint_contract(checkpoint, "ldm")
        update = checkpoint["global_step"]
        identifier = checkpoint_identifier(checkpoint)
        run = checkpoint.get("cylinderflow_resume_state", {}).get("run_id")
    else:
        if method != "gladit" and checkpoint.get("stage") == "ae":
            raise ValueError("AE checkpoint excluded: forecasting checkpoints only")
        update = checkpoint["update" if method == "gladit" else "updates"]
        identifier = checkpoint["checkpoint_id"]
        run = checkpoint.get("run_id")
        if method == "gladit" and not run:
            run = identifier.rsplit(":", 1)[0]
    if isinstance(update, bool) or int(update) != update or update < 0:
        raise ValueError("invalid checkpoint update")
    return {
        **stamp(source),
        "update": int(update),
        "checkpoint_id": identifier,
        "run_id": run,
    }


def reduced_summary(summary: dict) -> dict:
    rows = [
        {
            key: row[key]
            for key in ("trajectory_index", "finite", "uv_relative_rmse")
            if key in row
        }
        for row in summary["trajectory_metrics"]
    ]
    value = summary["selection_uv_relative_rmse"]
    valid = (
        summary["trajectory_count"] == 100
        and summary["failed_clips"] == 0
        and {r["trajectory_index"] for r in rows} == set(range(1000, 1100))
        and all(r["finite"] for r in rows)
        and value is not None
        and math.isfinite(value)
    )
    return {
        "status": "complete" if valid else "failed",
        "uv_relative_rmse": value if valid else None,
        "trajectory_count": summary["trajectory_count"],
        "failed_clips": summary["failed_clips"],
        "trajectory_metrics": rows,
    }


def evaluate_point(job: dict, destination: Path) -> None:
    import torch

    method = job["method"]
    device = torch.device(job["device"])
    if device.type != "cuda" or not torch.cuda.is_available():
        raise ValueError("Use the verified production CUDA evaluation environment")
    configure_module = (
        "graph_dit.performance"
        if method == "gladit"
        else "tools.cylinderflow_stride8.performance"
        if method == "text2pde"
        else "cylinderflow.performance"
    )
    from importlib import import_module

    performance = import_module(configure_module)
    performance.configure(device, 2)
    checkpoint_file = Path(job["checkpoint"]["file"])
    if metadata(checkpoint_file, method) != job["checkpoint"]:
        raise ValueError("checkpoint changed after discovery")
    inputs = job["inputs"]
    with torch.inference_mode(), torch.autocast("cuda", enabled=False):
        if method == "gladit":
            import inspect
            import numpy as np
            from graph_dit.evaluate import Predictor, load_selected
            from graph_dit.metrics import compute_metrics, summarize_trajectories

            load_options = (
                {"inference_only": True}
                if "inference_only" in inspect.signature(load_selected).parameters
                else {}
            )
            model, checkpoint = load_selected(
                checkpoint_file, Path(inputs["artifacts"]), "ema_0.9999", **load_options
            )
            model.to(device).eval().float()
            predictor_options = (
                {"inference_only": True}
                if "inference_only" in inspect.signature(Predictor).parameters
                else {}
            )
            predictor = Predictor(
                model,
                Path(inputs["artifacts"]),
                Path(inputs["data_dir"]),
                device,
                **predictor_options,
            )
            predictor.sampling_steps = 6
            indices = list(predictor.data.splits["validation"])
            if indices != list(range(1000, 1100)):
                raise ValueError("unexpected Validation100 indices")
            rows = []
            for index in indices:
                sample = predictor.load_case(index)
                total = None
                for label in range(8):
                    seed = ((label + 1) * 1_000_003 + index * 9_176) % (2**31 - 1)
                    prediction, _ = predictor.predict(sample, seed)
                    values = prediction.astype(np.float64)
                    if not np.isfinite(values).all():
                        raise FloatingPointError(
                            f"nonfinite prediction: {index}, {label}"
                        )
                    total = values if total is None else total + values
                target = predictor.data.evaluation(index)["field"]
                metrics = compute_metrics(
                    total / 8,
                    target,
                    sample["points"],
                    sample["cells"],
                    sample["node_type"],
                    0.08,
                )
                row = {
                    "trajectory_index": index,
                    "seed": 0,
                    "finite": bool(metrics["finite"]),
                    "uv_relative_rmse": metrics["uv_relative_rmse"],
                }
                rows.append(row)
                with (destination / "progress.jsonl").open(
                    "a", encoding="utf-8"
                ) as stream:
                    stream.write(json.dumps(row, allow_nan=False) + "\n")
                print(f"Validation {len(rows)}/100", flush=True)
            summary = summarize_trajectories(rows)
            data_identity = predictor.data.identity()
            dependency = {
                "artifact_id": checkpoint["artifact_id"],
                "representation_id": predictor.identity["representation_id"],
            }
        elif method == "text2pde":
            from modules.utils import get_yaml
            from dataset.datamodule import FluidsDataModule
            from tools.cylinderflow_stride8.evaluate_joint64 import (
                evaluate_one_checkpoint,
            )

            config = get_yaml(inputs["config"])
            datamodule = FluidsDataModule(config["data"])
            try:
                result, _ = evaluate_one_checkpoint(
                    checkpoint_file,
                    Path(inputs["ae_checkpoint"]),
                    config,
                    datamodule,
                    datamodule.val_dataset,
                    tuple(range(100)),
                    (0, 1, 2),
                    device,
                    None,
                    fail_on_runtime_error=True,
                )
            finally:
                datamodule.train_dataset.close()
                datamodule.val_dataset.close()
            summary = result["aggregate"]
            data_identity = result["data_contract"]
            dependency = result["dependencies"]
        else:
            from cylinderflow.engine import (
                config_from_file,
                open_prepared,
                validate_dependency,
                make_model,
                load_ae,
                evaluate_model,
            )
            from cylinderflow.data import Dataset

            config = config_from_file(inputs["config"])
            if config["method"] != method:
                raise ValueError("wrong repository/config method")
            dataset = Dataset(Path(inputs["dataset"]), Path(inputs["manifest"]))
            prepared = Path(inputs["prepared"])
            stats = open_prepared(dataset, config, prepared)
            checkpoint = torch.load(
                checkpoint_file, map_location="cpu", weights_only=False
            )
            stage = checkpoint["stage"]
            validate_dependency(checkpoint, config, stats, stage)
            model = make_model(checkpoint["config"], stage, device)
            model.load_state_dict(checkpoint["model"], strict=True)
            ae = None
            if stage == "dynamics":
                ae, _ = load_ae(
                    config,
                    stats,
                    Path(inputs["ae_checkpoint"]),
                    device,
                    checkpoint["ae_checkpoint_id"],
                )
            summary = evaluate_model(
                model,
                config,
                stage,
                dataset,
                stats,
                prepared,
                dataset.splits["validation"],
                device,
                checkpoint["settings"]["seed"],
                autoencoder=ae,
                fail_on_runtime_error=True,
            )
            data_identity = dataset.identity()
            dependency = {"ae_checkpoint_id": checkpoint.get("ae_checkpoint_id")}
    result = {
        **reduced_summary(summary),
        "checkpoint": job["checkpoint"],
        "protocol": job["protocol"],
        "data_identity": data_identity,
        "dependency": dependency,
        "environment": performance.runtime_identity(device),
        "test_accessed": False,
    }
    write_json(destination / "result.json", result)
    if result["status"] != "complete":
        raise RuntimeError("Validation100 incomplete; see result.json")


def render(reports: list[dict], output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output.mkdir(parents=True, exist_ok=True)
    mapping = {}
    for report in reports:
        method = report["method"]
        if method in mapping:
            raise ValueError("Merge accepts only one run per method")
        mapping[method] = report
    methods = [m for m in METHODS if m in mapping]
    coordinates = sorted({r["training_input"] for p in reports for r in p["rows"]})
    lookup = {m: {r["training_input"]: r for r in mapping[m]["rows"]} for m in methods}
    headers = [
        "Update"
        if methods == ["gladit"]
        else "Update x 4"
        if len(methods) == 1
        else "Training input*",
        *[LABELS[m] for m in methods],
    ]
    values = []
    for coordinate in coordinates:
        row = [str(coordinate)]
        for method in methods:
            entry = lookup[method].get(coordinate)
            row.append(
                format(entry["uv_relative_rmse"], ".6g")
                if entry and entry["status"] == "complete"
                else "—"
            )
        values.append(row)
    with (output / "table.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.writer(stream)
        writer.writerow(headers)
        writer.writerows(values)
    pages = max(1, math.ceil(len(values) / 15))
    image_files = []
    for page in range(pages):
        subset = values[page * 15 : (page + 1) * 15] or [["—"] * len(headers)]
        fig, ax = plt.subplots(
            figsize=(max(8, 2.1 * len(headers)), 3.1 + 0.4 * len(subset))
        )
        ax.axis("off")
        table = ax.table(
            cellText=subset, colLabels=headers, loc="center", cellLoc="center"
        )
        table.auto_set_font_size(False)
        table.set_fontsize(13)
        table.scale(1, 1.7)
        for (row, _), cell in table.get_celld().items():
            cell.set_edgecolor("#d6dce5")
            if row == 0:
                cell.set_facecolor("#17324d")
                cell.set_text_props(color="white", weight="bold")
            elif row % 2:
                cell.set_facecolor("#edf3f8")
        ax.set_title(
            "Validation100 | UV relative RMSE (lower is better)\n"
            + " / ".join(LABELS[m] for m in methods),
            fontsize=15,
            pad=18,
        )
        runs = "\n".join(f"{LABELS[m]} run: {mapping[m]['run_label']}" for m in methods)
        protocol = (
            "Baseline coordinate = update x 4; GLaDiT coordinate = update.\n"
            "Baselines: original sampling; GLaDiT: EMA0.9999, S6 / K8 physical mean.\n"
            "— = missing / excluded / failed / pending; see status.json."
        )
        fig.text(0.04, 0.02, f"{protocol}\n{runs}\nPage {page + 1}/{pages}", fontsize=9)
        fig.subplots_adjust(bottom=min(0.40, 0.17 + 0.018 * len(methods)), top=0.83)
        filename = f"table_{page + 1:02d}.png"
        fig.savefig(output / filename, dpi=180, facecolor="white")
        plt.close(fig)
        image_files.append(filename)
    write_json(output / "pages.json", {"images_to_return": image_files})


def run_collection(args: argparse.Namespace, method: str) -> None:
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    inputs = {
        key: str(value.resolve())
        for key, value in vars(args).items()
        if key
        in {
            "config",
            "dataset",
            "manifest",
            "prepared",
            "ae_checkpoint",
            "artifacts",
            "data_dir",
        }
        and value is not None
    }
    required = (
        {"artifacts", "data_dir"}
        if method == "gladit"
        else {"config", "ae_checkpoint"}
        if method == "text2pde"
        else {"config", "dataset", "manifest", "prepared"}
        | ({"ae_checkpoint"} if method == "aroma" else set())
    )
    if required - inputs.keys():
        raise ValueError(f"missing inputs: {sorted(required - inputs.keys())}")
    protocol = {
        "split": "validation100",
        "metric": "uv_relative_rmse",
        "version": 1,
        "multiplier": 1 if method == "gladit" else 4,
        "sampling": "ema_0.9999_s6_k8_physical_mean_float64"
        if method == "gladit"
        else "original_single_prediction_scores",
    }
    dependency_files = []
    for key in ("artifacts", "prepared"):
        if key in inputs:
            folder = Path(inputs[key])
            dependency_files.extend(
                stamp(p)
                for p in sorted(folder.iterdir())
                if p.is_file() and p.suffix in {".json", ".pt"}
            )
    identity = {
        "method": method,
        "run_dir": str(args.run_dir.resolve()),
        "inputs": inputs,
        "protocol": protocol,
        "device": args.device,
        "input_files": [stamp(Path(v)) for v in inputs.values() if Path(v).is_file()],
        "dependencies": dependency_files,
        "source": Path(__file__).read_text(encoding="utf-8"),
    }
    identity_file = output / "identity.json"
    if identity_file.exists() and read_json(identity_file) != identity:
        raise ValueError("output identity changed; use a new output directory")
    write_json(identity_file, identity)
    if args.checkpoints:
        files = args.checkpoints
    else:
        folder = args.run_dir / "checkpoints"
        if not folder.is_dir():
            folder = args.run_dir
        files = sorted(
            p for p in folder.iterdir() if p.suffix in {".pt", ".ckpt", ".pth"}
        )
    records, excluded, seen = [], [], set()
    for source in files:
        try:
            if not source.resolve().is_relative_to(args.run_dir.resolve()):
                raise ValueError(
                    "checkpoint is outside the explicitly selected run directory"
                )
            entry = metadata(source, method)
            if entry["checkpoint_id"] in seen:
                continue
            seen.add(entry["checkpoint_id"])
            records.append(entry)
        except Exception as error:
            excluded.append(
                {"file": str(source), "error": f"{type(error).__name__}: {error}"}
            )
    runs = {r["run_id"] for r in records if r["run_id"]}
    if len(runs) > 1:
        raise ValueError("multiple run identities; supply one run only")
    if method == "text2pde" and len({str(Path(r["file"]).parent) for r in records}) > 1:
        raise ValueError(
            "Text2PDE checkpoints must come from one run checkpoint directory"
        )
    duplicated = {
        r["update"]
        for r in records
        if sum(s["update"] == r["update"] for s in records) > 1
    }
    for entry in records:
        if entry["update"] in duplicated:
            excluded.append(
                {
                    **entry,
                    "error": "different checkpoints at same update; select one explicitly",
                }
            )
    records = sorted(
        (r for r in records if r["update"] not in duplicated), key=lambda r: r["update"]
    )
    run_label = next(iter(runs), args.run_dir.resolve().name)
    report = {
        "schema": "training_curve.uv.v1",
        "method": method,
        "run_label": run_label,
        "protocol": protocol,
        "excluded": excluded,
        "rows": [],
    }
    for entry in records:
        report["rows"].append(
            {
                "update": entry["update"],
                "training_input": entry["update"] * protocol["multiplier"],
                "checkpoint": entry,
                "status": "pending",
                "uv_relative_rmse": None,
            }
        )

    def refresh() -> None:
        write_json(output / "status.json", report)
        render([report], output)

    refresh()
    for row in report["rows"]:
        entry = row["checkpoint"]
        point = output / f"update_{entry['update']:09d}"
        job = {
            "method": method,
            "checkpoint": entry,
            "inputs": inputs,
            "protocol": protocol,
            "device": args.device,
        }
        attempts = sorted(point.glob("attempt_*/job.json"))
        completed = False
        for prior in attempts:
            if read_json(prior) != job:
                raise ValueError("checkpoint/inputs changed for an existing update")
            result_file = prior.parent / "result.json"
            exit_file = prior.parent / "exit.json"
            if (
                result_file.exists()
                and exit_file.exists()
                and read_json(exit_file)["exit_code"] == 0
            ):
                result = read_json(result_file)
                if result["status"] == "complete":
                    completed = True
                    break
        if not completed and (not attempts or args.retry_failed):
            attempt = point / f"attempt_{len(attempts) + 1:03d}"
            attempt.mkdir(parents=True)
            write_json(attempt / "job.json", job)
            print(f"Evaluating {method} update {entry['update']}", flush=True)
            with (attempt / "evaluation.log").open("w", encoding="utf-8") as stream:
                process = subprocess.run(
                    [
                        sys.executable,
                        str(Path(__file__).resolve()),
                        "--worker",
                        str(attempt / "job.json"),
                    ],
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                )
            write_json(attempt / "exit.json", {"exit_code": process.returncode})
            if process.returncode == 0:
                result = read_json(attempt / "result.json")
                completed = result["status"] == "complete"
        row["status"] = "complete" if completed else "failed"
        row["uv_relative_rmse"] = result["uv_relative_rmse"] if completed else None
        refresh()
        print(
            f"{method} update {entry['update']}: {row['status']} {row['uv_relative_rmse']}",
            flush=True,
        )
    failed = (
        bool(excluded)
        or not records
        or any(r["status"] != "complete" for r in report["rows"])
    )
    write_json(output / "exit.json", {"exit_code": int(failed)})
    raise SystemExit(int(failed))


def main(method: str) -> None:
    parser = argparse.ArgumentParser(
        description="Historical checkpoints -> Validation100 UV table"
    )
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--checkpoints", nargs="+", type=Path)
    for name in (
        "config",
        "dataset",
        "manifest",
        "prepared",
        "ae-checkpoint",
        "artifacts",
        "data-dir",
    ):
        parser.add_argument("--" + name, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument(
        "--merge",
        nargs="+",
        type=Path,
        help="status.json files; render without model execution",
    )
    args = parser.parse_args()
    if args.merge:
        reports = [read_json(p) for p in args.merge]
        for report in reports:
            if report.get("schema") != "training_curve.uv.v1":
                parser.error("unsupported report schema")
        render(reports, args.output_dir)
        write_json(args.output_dir / "merged.json", reports)
        return
    if args.run_dir is None:
        parser.error("--run-dir is required")
    lock = args.output_dir.resolve() / "collector.lock"
    lock.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        parser.error(
            "collector.lock exists; verify no collector is active before removing stale lock"
        )
    try:
        with os.fdopen(descriptor, "w") as stream:
            stream.write(str(os.getpid()))
        run_collection(args, method)
    finally:
        lock.unlink()


if __name__ == "__main__":
    if len(sys.argv) != 3 or sys.argv[1] != "--worker":
        raise SystemExit("Use training_curve.py")
    job_file = Path(sys.argv[2])
    try:
        evaluate_point(read_json(job_file), job_file.parent)
    except Exception:
        traceback.print_exc()
        raise SystemExit(1)
