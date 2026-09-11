"""Sequential AE -> selection -> LDM -> Validation in one four-GPU allocation."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from modules.utils import get_yaml, save_yaml
from .evaluation_io import write_json
from .materialize_config import materialize_config

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("data", "manifest", "normalizer", "result-root"):
        parser.add_argument("--" + key, type=Path, required=True)
    parser.add_argument(
        "--gpus", help="four allocated GPU IDs/UUIDs; omit inside Slurm"
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--preflight-stage", choices=("ae", "ldm"))
    parser.add_argument("--ae-checkpoint", type=Path)
    args = parser.parse_args()
    env = {
        **os.environ,
        "PYTHONUNBUFFERED": "1",
        "WANDB_MODE": "offline",
        "TOKENIZERS_PARALLELISM": "false",
    }
    if args.gpus:
        env["CUDA_VISIBLE_DEVICES"] = args.gpus
    assigned = env.get("CUDA_VISIBLE_DEVICES", "").split(",")
    if (
        len(assigned) != 4
        or any(not item.strip() for item in assigned)
        or len(set(assigned)) != 4
    ):
        raise ValueError("supply exactly four explicitly allocated visible GPUs")
    for file in (args.data, args.manifest, args.normalizer):
        if not file.is_file():
            raise FileNotFoundError(file)
    result = args.result_root.resolve()
    if result.exists() and not args.resume:
        raise FileExistsError("choose a new result root or use --resume")
    result.mkdir(parents=True, exist_ok=True)
    # One orchestrator owns both dependent stages for the lifetime of the allocation.
    import fcntl

    lock = (result / ".pipeline.lock").open("a")
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        logs = result / "logs"
        logs.mkdir(exist_ok=True)

        def execute(label, command):
            attempt = len(list(logs.glob(label + "_*.log"))) + 1
            log = logs / f"{label}_{attempt:03d}.log"
            started = time.time()
            with log.open("x", encoding="utf-8") as handle:
                code = subprocess.run(
                    command, cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT
                ).returncode
            write_json(
                log.with_suffix(".json"),
                {
                    "command": command,
                    "exit_code": code,
                    "started_unix": started,
                    "elapsed_seconds": time.time() - started,
                },
            )
            if code:
                write_json(
                    result / "status.json",
                    {
                        "state": "failed",
                        "phase": label,
                        "exit_code": code,
                        "log": str(log),
                    },
                )
                raise RuntimeError(f"{label} exited {code}; see {log}")

        def distributed(module, *arguments):
            return [
                sys.executable,
                "-m",
                "torch.distributed.run",
                "--standalone",
                "--nproc-per-node=4",
                "--max-restarts=0",
                "--module",
                module,
                *map(str, arguments),
            ]

        config_files = {}
        for stage in ("ae", "ldm"):
            template = get_yaml(
                ROOT / "configs/cylinderflow_stride8" / f"{stage}_1plus64.yaml"
            )
            config = materialize_config(
                template,
                args.data,
                args.manifest,
                args.normalizer,
                result,
                stage,
                world_size=4,
            )
            destination = result / "config" / f"{stage}_1plus64.yaml"
            destination.parent.mkdir(exist_ok=True)
            if destination.exists() and get_yaml(destination) != config:
                raise ValueError("materialized training configuration changed")
            save_yaml(config, destination)
            config_files[stage] = destination
        execute(
            "verify_data",
            [
                sys.executable,
                "-m",
                "tools.cylinderflow_stride8.verify_data",
                "--data",
                str(args.data.resolve()),
                "--manifest",
                str(args.manifest.resolve()),
            ],
        )
        stages = [args.preflight_stage] if args.preflight_stage else ["ae", "ldm"]
        selected_ae = args.ae_checkpoint.resolve() if args.ae_checkpoint else None
        for stage in stages:
            folder = result / stage / "formal"
            complete_marker = result / stage / "training_complete.json"
            checkpoint = folder / "checkpoints/last.ckpt"
            bounded_completion = folder / "training_completion.json"
            if args.preflight_stage and bounded_completion.exists():
                completion = json.loads(bounded_completion.read_text())
                if (
                    completion.get("state") == "preflight_complete"
                    and completion.get("update") == 8
                ):
                    write_json(
                        result / "status.json",
                        {
                            "state": "bounded_training_complete",
                            "stage": stage,
                            "updates": 8,
                            "world_size": 4,
                        },
                    )
                    return
            if not complete_marker.exists():
                command = distributed(
                    "train_AE" if stage == "ae" else "train_ldm",
                    "--config",
                    config_files[stage],
                )
                if args.resume and checkpoint.exists():
                    command += ["--checkpoint", str(checkpoint)]
                if stage == "ldm":
                    if selected_ae is None:
                        raise ValueError("LDM requires a selected AE checkpoint")
                    command += ["--first-stage-checkpoint", str(selected_ae)]
                if args.preflight_stage:
                    command += ["--stop-after-updates", "8"]
                execute(stage + "_train", command)
                completion = json.loads(
                    (folder / "training_completion.json").read_text()
                )
                expected = 8 if args.preflight_stage else 250000
                if (
                    completion["update"] != expected
                    or completion["examples_seen"] != expected * 4
                ):
                    raise RuntimeError(
                        "training completion does not match the allocated global batch/budget"
                    )
                if args.preflight_stage:
                    write_json(
                        result / "status.json",
                        {
                            "state": "bounded_training_complete",
                            "stage": stage,
                            "updates": 8,
                            "world_size": 4,
                        },
                    )
                    return
                write_json(complete_marker, completion)
            selection = (
                result / "ae/selection_v1"
                if stage == "ae"
                else result / "evaluation/ldm_selection_v1"
            )
            command = distributed(
                "tools.cylinderflow_stride8.evaluate_four_gpu",
                "--stage",
                stage,
                "--mode",
                "select",
                "--config",
                config_files[stage],
                "--checkpoint-dir",
                folder / "checkpoints",
                "--output-dir",
                selection,
            )
            if selected_ae:
                command += ["--ae-checkpoint", str(selected_ae)]
            execute(stage + "_selection", command)
            selected = Path((selection / "selected_checkpoint.txt").read_text().strip())
            if stage == "ae":
                selected_ae = selected
            else:
                execute(
                    "validation",
                    distributed(
                        "tools.cylinderflow_stride8.evaluate_four_gpu",
                        "--stage",
                        "ldm",
                        "--mode",
                        "validation",
                        "--config",
                        config_files["ldm"],
                        "--ae-checkpoint",
                        selected_ae,
                        "--checkpoint",
                        selected,
                        "--output-dir",
                        result / "evaluation/validation_v1",
                    ),
                )
                execute(
                    "finalize",
                    [
                        sys.executable,
                        "-m",
                        "tools.cylinderflow_stride8.finalize_validation",
                        "--result-root",
                        str(result),
                    ],
                )
        write_json(
            result / "status.json",
            {"state": "validation_complete_no_test_entry", "world_size": 4},
        )
    finally:
        lock.close()


if __name__ == "__main__":
    main()
