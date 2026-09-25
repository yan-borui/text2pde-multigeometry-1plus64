"""Repeated fixed-checkpoint ensemble scores with resumable group boundaries."""

from __future__ import annotations

import csv
import importlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
from typing import Any

from sampling_ensemble_support import build_parser, file_identity, input_files


TRAJECTORIES = list(range(1000, 1100))
SCORE = "selection_uv_relative_rmse"


def write_json(file_path: Path, value: Any) -> None:
    """Atomically publish a manifest or a completed-group marker."""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = file_path.with_suffix(file_path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(file_path)


def read_json(file_path: Path) -> Any:
    return json.loads(file_path.read_text(encoding="utf-8"))


def source_identity(root: Path) -> dict:
    """Require a committed implementation before starting or resuming evaluation."""
    revision = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    changes = subprocess.run(
        ["git", "-C", str(root), "diff", "--name-only", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if changes:
        raise ValueError("commit the evaluation implementation before sampling")
    return {"commit": revision, "tracked_worktree_clean": True}


def member_seed(label: int, trajectory: int) -> int:
    return ((label + 1) * 1_000_003 + trajectory * 9_176) % (2**31 - 1)


def validate_seed_plan(base: int, groups: int, samples: int) -> None:
    """Reject repeated PRNG seeds among any requested group/case/member."""
    seen = set()
    for label in range(base, base + groups * samples):
        for trajectory in TRAJECTORIES:
            seed = member_seed(label, trajectory)
            if seed in seen:
                raise ValueError("seed plan contains overlapping PRNG seeds")
            seen.add(seed)


def validate_group(
    directory: Path, identity: dict, group: int, offset: int, timing: bool
) -> tuple[dict, dict]:
    """Check complete native evidence before counting a group's score."""
    manifest = read_json(directory / "manifest.json")
    summary = read_json(directory / "summary.json")
    exit_record = read_json(directory / "exit.json")
    protocol = identity["protocol"]
    expected = {
        "method": identity["method"],
        "inputs": identity["inputs"],
        "source_commit": identity["source"]["commit"],
        "samples": protocol["samples"],
        "ensemble_sizes": protocol["ensemble_sizes"],
        "seed_offset": offset,
        "test_accessed": False,
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise ValueError(f"group {group}: manifest mismatch in {key}")
    if (
        exit_record.get("exit_code") != 0
        or summary.get("status") != "complete"
        or summary.get("trajectory_count") != 100
        or summary.get("test_accessed") is not False
        or summary.get("timing_complete") is not timing
    ):
        raise ValueError(f"group {group}: incomplete evaluation")
    order = [
        row["trajectory_index"] for row in summary["trajectory_sampling_statistics"]
    ]
    if sorted(order) != TRAJECTORIES:
        raise ValueError(f"group {group}: incomplete or duplicate Validation100")
    members = []
    with (directory / "members.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            members.append((row["trajectory_index"], row["seed"], row["sample_seed"]))
            if not row.get("finite"):
                raise ValueError(f"group {group}: failed member")
    expected_members = [
        (trajectory, label, member_seed(offset + label, trajectory))
        for trajectory in order
        for label in range(protocol["samples"])
    ]
    if members != expected_members:
        raise ValueError(f"group {group}: missing, duplicate or misordered members")
    for count in protocol["ensemble_sizes"]:
        result = summary["ensemble"][str(count)]
        rows = result["trajectory_metrics"]
        if (
            result["trajectory_count"] != 100
            or result["clip_count"] != 100
            or result["failed_clips"] != 0
            or result["failed_trajectories"] != 0
            or [row["trajectory_index"] for row in rows] != TRAJECTORIES
            or any(row.get("sample_count") != 1 for row in rows)
        ):
            raise ValueError(f"group {group}, K={count}: incomplete ensemble scores")
        score = result[SCORE]
        reference = statistics.mean(row["uv_relative_rmse"] for row in rows)
        if not math.isfinite(score) or not math.isclose(
            score, reference, rel_tol=1e-12, abs_tol=1e-15
        ):
            raise ValueError(f"group {group}, K={count}: score aggregation mismatch")
        for trajectory in order:
            saved = directory / "fields" / str(trajectory) / f"mean_k{count}.npz"
            if not saved.is_file() or saved.stat().st_size == 0:
                raise ValueError(f"group {group}, K={count}: missing physical field")
    return summary, {
        "provenance": manifest["provenance"],
        "data_identity": manifest["data_identity"],
        "environment": manifest["environment"],
        "trajectory_order": order,
    }


def statistics_for(values: list[float]) -> dict:
    variance = statistics.variance(values) if len(values) >= 2 else None
    return {
        "mean": statistics.mean(values),
        "variance": variance,
        "std": math.sqrt(variance) if variance is not None else None,
        "count": len(values),
        "ddof": 1,
        "scores": values,
    }


def publish(root: Path, identity: dict, groups: list[dict]) -> dict:
    """Reduce whole-Validation100 scores across independent ensemble groups."""
    result = {
        "schema": "airfoil.sampling_groups.summary.v1",
        "method": identity["method"],
        "status": (
            "complete" if len(groups) == identity["protocol"]["groups"] else "partial"
        ),
        "groups_completed": len(groups),
        "groups_requested": identity["protocol"]["groups"],
        "score": SCORE,
        "statistical_unit": "one K-member physical mean evaluated on Validation100",
        "training_seed_variation": False,
        "ensemble": {},
        "test_accessed": False,
    }
    for count in identity["protocol"]["ensemble_sizes"]:
        values = [item["summary"]["ensemble"][str(count)][SCORE] for item in groups]
        result["ensemble"][str(count)] = statistics_for(values)
    write_json(root / "summary.json", result)
    with (root / "score_curve.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            ["method", "K", "R", "score_mean", "score_variance", "score_std"]
        )
        for count, row in result["ensemble"].items():
            writer.writerow(
                [
                    result["method"],
                    count,
                    row["count"],
                    row["mean"],
                    row["variance"],
                    row["std"],
                ]
            )
    with (root / "group_scores.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["method", "group", "K", "validation100_uv_relative_rmse"])
        for item in groups:
            for count, row in item["summary"]["ensemble"].items():
                writer.writerow([result["method"], item["group"], count, row[SCORE]])
    return result


def plot_curve(root: Path, summary: dict) -> None:
    """Render group mean with standard deviation and unbiased score variance."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sizes = [int(count) for count in summary["ensemble"]]
    rows = list(summary["ensemble"].values())
    figure, axes = plt.subplots(1, 2, figsize=(9, 3.4), constrained_layout=True)
    axes[0].errorbar(
        sizes,
        [row["mean"] for row in rows],
        yerr=[row["std"] for row in rows],
        marker="o",
        capsize=3,
    )
    axes[0].set_ylabel("Validation100 UV relative RMSE")
    axes[0].set_title("Mean and sample standard deviation")
    axes[1].plot(sizes, [row["variance"] for row in rows], marker="o")
    axes[1].set_ylabel("Score variance (ddof=1)")
    axes[1].set_title("Variance across independent groups")
    for axis in axes:
        axis.set_xlabel("Ensemble size K")
        axis.set_xticks(sizes)
        axis.grid(alpha=0.25)
    figure.savefig(root / "score_curve.pdf")
    figure.savefig(root / "score_curve.png", dpi=200)
    plt.close(figure)


def run(method: str) -> None:
    parser = build_parser(method)
    parser.description = __doc__
    parser.add_argument("--groups", type=int, required=True)
    parser.add_argument("--seed-base", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()
    for name, value in vars(args).items():
        if isinstance(value, Path):
            setattr(args, name, value.resolve())
    args.ensemble_sizes = sorted(set(args.ensemble_sizes))
    if (
        args.groups < 2
        or args.samples < 2
        or not args.ensemble_sizes
        or args.ensemble_sizes[0] < 1
        or args.ensemble_sizes[-1] > args.samples
        or args.seed_base < 0
        or args.seed_offset != 0
        or args.seed_base + args.groups * args.samples >= 2**31 - 1
    ):
        parser.error(
            "require R>=2, samples>=2, 1<=K<=samples, and a valid --seed-base; "
            "--seed-offset is reserved for group members"
        )
    if os.name != "posix":
        parser.error("run in the verified Linux CUDA evaluation environment")
    if args.plot:
        importlib.import_module("matplotlib")
    validate_seed_plan(args.seed_base, args.groups, args.samples)
    code_root = Path(__file__).resolve().parent
    source = source_identity(code_root)
    files = input_files(method, args)
    inputs = {name: file_identity(value) for name, value in files.items()}
    native = {
        key: value
        for key, value in vars(args).items()
        if key not in {"groups", "seed_base", "resume", "plot", "output_dir"}
    }
    identity = json.loads(
        json.dumps(
            {
                "schema": "airfoil.sampling_groups.v1",
                "method": method,
                "source": source,
                "inputs": inputs,
                "native_arguments": native,
                "protocol": {
                    "groups": args.groups,
                    "samples": args.samples,
                    "ensemble_sizes": args.ensemble_sizes,
                    "seed_base": args.seed_base,
                    "seed_rule": "((seed_base + group*samples + member + 1)*1000003"
                    " + trajectory_index*9176) mod (2**31-1)",
                    "pool": "nested K prefixes within group; disjoint members across groups",
                    "aggregation": "physical UVP mean, native trajectory score, equal case mean",
                    "variance_unit": "whole-Validation100 score across groups; ddof=1",
                    "timing": "skipped" if args.skip_timing else "group 0 only",
                },
                "test_accessed": False,
            },
            default=str,
        )
    )
    root = args.output_dir.resolve()
    if args.resume:
        if read_json(root / "manifest.json") != identity:
            raise ValueError(
                "resume identity differs; use the original inputs or a new output"
            )
    else:
        root.mkdir(parents=True, exist_ok=False)

    # Advisory lock is released by the OS after interruption; evidence remains.
    import fcntl

    with (root / ".groups.lock").open("a", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        if not args.resume:
            write_json(root / "manifest.json", identity)
        completed = []
        shared = None
        current = None
        try:
            for group in range(args.groups):
                current = group
                offset = args.seed_base + group * args.samples
                timing = group == 0 and not args.skip_timing
                marker = root / "groups" / f"group{group:03d}.json"
                if marker.exists():
                    record = read_json(marker)
                    relative = Path(record["directory"])
                    directory = (root / relative).resolve()
                    if relative.is_absolute() or root not in directory.parents:
                        raise ValueError(
                            "group marker points outside the output directory"
                        )
                    if record["group"] != group or record["seed_offset"] != offset:
                        raise ValueError("group marker identity mismatch")
                else:
                    group_root = root / "groups" / f"group{group:03d}"
                    group_root.mkdir(parents=True, exist_ok=True)
                    attempt = 0
                    while any(
                        (group_root / f"attempt{attempt:03d}{suffix}").exists()
                        for suffix in ("", ".log", ".exit.json")
                    ):
                        attempt += 1
                    directory = group_root / f"attempt{attempt:03d}"
                    log_file = group_root / f"attempt{attempt:03d}.log"
                    command = [sys.executable, str(code_root / "sampling_ensemble.py")]
                    options = dict(native, output_dir=directory, seed_offset=offset)
                    options["skip_timing"] = not timing
                    for name, value in options.items():
                        option = "--" + name.replace("_", "-")
                        if isinstance(value, bool):
                            if value:
                                command.append(option)
                        elif isinstance(value, list):
                            command.extend([option, *map(str, value)])
                        else:
                            command.extend([option, str(value)])
                    print(
                        f"Group {group + 1}/{args.groups}; log: {log_file}", flush=True
                    )
                    with log_file.open("w", encoding="utf-8") as stream:
                        child = subprocess.run(
                            command,
                            cwd=code_root,
                            stdout=stream,
                            stderr=subprocess.STDOUT,
                            check=False,
                            pass_fds=(lock.fileno(),),
                        )
                    write_json(
                        group_root / f"attempt{attempt:03d}.exit.json",
                        {
                            "returncode": child.returncode,
                            "group": group,
                            "seed_offset": offset,
                        },
                    )
                    if child.returncode:
                        raise RuntimeError(
                            f"group {group} exited {child.returncode}; inspect {log_file}"
                        )
                summary, shared_identity = validate_group(
                    directory, identity, group, offset, timing
                )
                if shared is None:
                    shared = shared_identity
                elif shared_identity != shared:
                    raise ValueError(
                        "weights, normalization, case order or runtime changed"
                    )
                if inputs != {
                    name: file_identity(value) for name, value in files.items()
                }:
                    raise ValueError("input files changed during grouped evaluation")
                if source_identity(code_root) != source:
                    raise ValueError("source changed during grouped evaluation")
                write_json(
                    marker,
                    {
                        "group": group,
                        "seed_offset": offset,
                        "directory": str(directory.relative_to(root)),
                    },
                )
                completed.append({"group": group, "summary": summary})
                result = publish(root, identity, completed)
            if args.plot:
                plot_curve(root, result)
            write_json(
                root / "exit.json",
                {
                    "exit_code": 0,
                    "groups_complete": len(completed),
                    "timing_complete": not args.skip_timing,
                },
            )
        except BaseException as error:
            write_json(
                root / "exit.json",
                {
                    "exit_code": 1,
                    "group": current,
                    "groups_complete": len(completed),
                    "error": f"{type(error).__name__}: {error}",
                },
            )
            raise
