"""Four-GPU standalone AE/checkpoint selection and final joint64 Validation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import traceback

from dataset.datamodule import FluidsDataModule
from modules.utils import get_yaml
from .distributed import Context
from .evaluation_io import write_json
from .metrics import selection_key, summarize_trajectories
from .protocol import (
    AE_MILESTONES,
    LDM_MILESTONES,
    checkpoint_identifier,
    stage_data_contract,
    validate_ae_dependency,
    validate_locked_config,
    validation_monitor_indices,
)


def evaluate(args):
    ctx = Context(4, args.device)
    try:
        config = get_yaml(args.config)
        validate_locked_config(config, args.stage)
        datamodule = FluidsDataModule(config["data"])
        dataset = datamodule.val_dataset
        if dataset.split != "validation" or len(dataset) != 100:
            raise ValueError(
                "four-GPU evaluation requires the fixed 100-trajectory Validation split"
            )
        if args.stage == "ae":
            from .select_ae import checkpoint_identity, evaluate_checkpoint

            milestones, prefix = AE_MILESTONES, "ae"
        else:
            from .evaluate_joint64 import checkpoint_identity, evaluate_one_checkpoint

            milestones, prefix = LDM_MILESTONES, "ldm"
            if args.ae_checkpoint is None or not args.ae_checkpoint.is_file():
                raise ValueError("joint64 requires the frozen AE checkpoint")
        if args.mode == "select":
            if args.checkpoint_dir is None or args.checkpoint is not None:
                raise ValueError("selection requires only --checkpoint-dir")
            candidates = sorted(args.checkpoint_dir.glob(f"{prefix}-epoch*.ckpt"))
            if len(candidates) != 4:
                raise ValueError(
                    "selection must contain the four original checkpoint milestones"
                )
            indices = validation_monitor_indices()
        else:
            if args.stage != "ldm" or args.checkpoint is None:
                raise ValueError(
                    "final Validation requires one selected LDM checkpoint"
                )
            candidates, indices = [args.checkpoint], tuple(range(100))
        ctx.primary_call(lambda: args.output_dir.mkdir(parents=True, exist_ok=True))

        def checkpoint_metadata():
            ae = None
            if args.stage == "ldm":
                from .select_ae import checkpoint_identity as ae_identity

                ae = {
                    "cylinderflow_resume_state": ae_identity(args.ae_checkpoint)[1][
                        "cylinderflow_resume_state"
                    ]
                }
            identifiers, steps = [], []
            for candidate in candidates:
                step, checkpoint = checkpoint_identity(candidate)
                if ae is not None:
                    validate_ae_dependency(checkpoint, ae)
                identifiers.append(checkpoint_identifier(checkpoint))
                steps.append(step)
                del checkpoint
            return {
                "ids": identifiers,
                "steps": steps,
                "ae_id": checkpoint_identifier(ae) if ae is not None else None,
            }

        metadata = ctx.primary_call(checkpoint_metadata)
        if args.mode == "select" and tuple(sorted(metadata["steps"])) != milestones:
            raise ValueError(
                "selection must contain the four original checkpoint milestones"
            )
        identity = {
            "stage": args.stage,
            "mode": args.mode,
            "config": config,
            "checkpoints": [str(item.resolve()) for item in candidates],
            "checkpoint_ids": metadata["ids"],
            "ae_checkpoint_id": metadata["ae_id"],
            "ae_checkpoint": str(args.ae_checkpoint.resolve())
            if args.ae_checkpoint
            else None,
        }

        def check_identity():
            file = args.output_dir / "identity.json"
            if file.exists() and json.loads(file.read_text()) != identity:
                raise ValueError("evaluation directory belongs to a different task")
            write_json(file, identity)

        ctx.primary_call(check_identity)

        def assigned():
            results = []
            work = (
                list(enumerate(candidates))[ctx.rank :: ctx.world]
                if args.mode == "select"
                else list(enumerate(candidates))
            )
            for number, checkpoint in work:
                folder = args.output_dir / f"candidate_{number:03d}"
                if args.mode == "validation":
                    folder = folder / f"rank_{ctx.rank:03d}"
                completed = folder / "completed.json"
                if completed.exists():
                    record = json.loads(completed.read_text())
                else:
                    if args.stage == "ae":
                        result = evaluate_checkpoint(
                            checkpoint, config, datamodule, dataset, indices, ctx.device
                        )
                        rows = []
                    else:
                        local_indices = (
                            indices
                            if args.mode == "select"
                            else indices[ctx.rank :: ctx.world]
                        )
                        result, rows = evaluate_one_checkpoint(
                            checkpoint,
                            args.ae_checkpoint,
                            config,
                            datamodule,
                            dataset,
                            local_indices,
                            (0, 1, 2),
                            ctx.device,
                            folder / "predictions",
                            resume=True,
                            fail_on_runtime_error=True,
                        )
                    record = {"number": number, "result": result, "rows": rows}
                    write_json(completed, record)
                results.append(record)
            return results

        shards = ctx.all_call(assigned)

        def finish():
            records = sorted(
                [row for shard in shards for row in shard], key=lambda r: r["number"]
            )
            if args.mode == "validation":
                rows = [row for record in records for row in record["rows"]]
                expected = {(index, seed) for index in indices for seed in (0, 1, 2)}
                if (
                    len(rows) != len(expected)
                    or {(row["sample_index"], row["seed"]) for row in rows} != expected
                ):
                    raise ValueError(
                        "final Validation has missing or duplicate samples"
                    )
                selected = dict(records[0]["result"])
                inference = sum(row["inference_seconds"] for row in rows)
                selected.update(
                    aggregate=summarize_trajectories(rows),
                    total_sequences=len(rows),
                    finite_sequences=sum(bool(row["finite"]) for row in rows),
                    validation_indices=list(indices),
                    total_inference_seconds=inference,
                    elapsed_seconds=max(
                        record["result"]["elapsed_seconds"] for record in records
                    ),
                    sequences_per_inference_second=len(rows) / inference,
                    future_frames_per_inference_second=sum(
                        bool(row["finite"]) for row in rows
                    )
                    * 64
                    / inference,
                )
                results = [selected]
                rows_by_checkpoint = {selected["checkpoint"]: rows}
            else:
                results = [record["result"] for record in records]
                if args.stage == "ae":
                    selected = min(
                        results,
                        key=lambda r: (r["mean_normalized_uvp_l1"], r["global_step"]),
                    )
                else:
                    selected = min(
                        results,
                        key=lambda r: selection_key(r["aggregate"], r["global_step"]),
                    )
                rows_by_checkpoint = {
                    record["result"]["checkpoint"]: record["rows"] for record in records
                }
            summary = {
                "schema": "text2pde.cylinderflow_stride8.distributed_evaluation.v1",
                "stage": args.stage,
                "mode": args.mode,
                "data_contract": stage_data_contract(args.stage),
                "selection_metric": "mean_normalized_uvp_l1"
                if args.stage == "ae"
                else "failed clips, then mean trajectory UV relative RMSE, then earlier update",
                "test_accessed": False,
                "test_entry_available": False,
                "candidates": results,
                "selected": selected,
                "world_size": ctx.world,
            }
            if args.stage == "ldm":
                summary.update(ddim_steps=20, sampling_seeds=[0, 1, 2])
            write_json(
                args.output_dir
                / ("selection.json" if args.stage == "ae" else "summary.json"),
                summary,
            )
            write_json(args.output_dir / "rows.json", rows_by_checkpoint)
            (args.output_dir / "selected_checkpoint.txt").write_text(
                selected["checkpoint"] + "\n", encoding="utf-8"
            )
            if args.mode == "validation":
                from .evaluate_joint64 import render_representatives

                if not (args.output_dir / "representative_samples.json").exists():
                    render_representatives(
                        args.output_dir,
                        rows_by_checkpoint[selected["checkpoint"]],
                        resume=True,
                    )
            return summary

        return ctx.primary_call(finish)
    except BaseException as error:
        write_json(
            args.output_dir / f"rank_{ctx.rank:03d}_failure.json",
            {
                "error": str(error),
                "traceback": traceback.format_exc(),
            },
        )
        raise
    finally:
        ctx.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("ae", "ldm"), required=True)
    parser.add_argument("--mode", choices=("select", "validation"), default="select")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--ae-checkpoint", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--checkpoint-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()
