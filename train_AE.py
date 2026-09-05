import argparse
from datetime import datetime
import torch
import os

from modules.utils import get_yaml, save_yaml
from modules.modules.callbacks import GridPlottingCallback, MeshPlottingCallback
from dataset.datamodule import FluidsDataModule

import lightning as L
from lightning.pytorch import seed_everything
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from modules.modules.reproducible_resume import ExactResumeCallback, load_resume_record

torch.set_float32_matmul_precision("high")
torch.multiprocessing.set_sharing_strategy("file_system")


def main(args):
    config = get_yaml(args.config)
    aeconfig = config["model"]["aeconfig"]
    lossconfig = config["model"]["lossconfig"]
    trainconfig = config["training"]
    dataconfig = config["data"]

    if args.run_dir is not None:
        trainconfig["run_dir"] = args.run_dir
    if args.checkpoint is not None:
        trainconfig["checkpoint"] = args.checkpoint

    data_contract = None
    if dataconfig["mode"] == "cylinderflow_stride8":
        from tools.cylinderflow_stride8.protocol import (
            stage_data_contract,
            validate_locked_config,
        )

        validate_locked_config(config, "ae")
        data_contract = stage_data_contract("ae")

    seed = trainconfig["seed"]
    start_examples_seen = 0
    exact_resume_modes = ("cylinderflow_windows", "cylinderflow_stride8")
    if dataconfig["mode"] in exact_resume_modes and trainconfig["checkpoint"]:
        resume_record = load_resume_record(trainconfig["checkpoint"])
        start_examples_seen = int(resume_record["examples_seen"])
        if dataconfig["mode"] == "cylinderflow_stride8":
            dataconfig["train_examples_seen"] = start_examples_seen
        else:
            dataconfig["train_start_offset"] = start_examples_seen
    now = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    seed_everything(seed)

    name = config["wandb"]["name"] + now
    configured_run_dir = trainconfig.get("run_dir")
    if configured_run_dir:
        path = os.path.abspath(configured_run_dir)
        name = os.path.basename(path)
    else:
        path = os.path.join(trainconfig["default_root_dir"], name)

    if trainconfig.get("logger", "wandb") == "csv":
        experiment_logger = CSVLogger(save_dir=path, name="lightning", version="")
    elif trainconfig.get("logger", "wandb") == "none":
        experiment_logger = False
    else:
        experiment_logger = WandbLogger(
            project=config["wandb"]["project"],
            name=name,
            offline=trainconfig.get("wandb_offline", False),
        )

    if torch.cuda.current_device() == 0:
        os.makedirs(path, exist_ok=True)
        save_yaml(config, os.path.join(path, "config.yml"))
        print("Making folder on rank 0")

    checkpoint_dir = os.path.join(path, "checkpoints")
    milestone_steps = trainconfig.get("milestone_every_n_steps")
    checkpoint_callbacks = []
    if dataconfig["mode"] in exact_resume_modes:
        checkpoint_callbacks.append(
            ExactResumeCallback(start_examples_seen, data_contract=data_contract)
        )
    checkpoint_callbacks.extend(
        [
            ModelCheckpoint(
                filename="ae-epoch{epoch:02d}-step{step}",
                dirpath=checkpoint_dir,
                every_n_train_steps=milestone_steps,
                save_top_k=-1,
            ),
            ModelCheckpoint(
                dirpath=checkpoint_dir,
                every_n_train_steps=trainconfig.get("last_every_n_steps", 5000),
                save_top_k=0,
                save_last=True,
            ),
            LearningRateMonitor(logging_interval="step"),
        ]
    )

    datamodule = FluidsDataModule(dataconfig)

    if dataconfig["mode"] in (
        "cylinder",
        "multigeometry",
        "cylinderflow_windows",
        "cylinderflow_stride8",
    ):
        from modules.models.ae.ae_mesh import AutoencoderKL, Autoencoder

        if "loss" in lossconfig.keys():  # use more complex autoencoder w/ GAN and LPIPS
            ae = Autoencoder
        else:
            ae = AutoencoderKL
        eval_callback = MeshPlottingCallback()
    else:
        from modules.models.ae.ae_grid import Autoencoder, AutoencoderKL

        if "loss" in lossconfig.keys():  # use more complex autoencoder w/ GAN and LPIPS
            ae = Autoencoder
        else:
            ae = AutoencoderKL
        eval_callback = GridPlottingCallback()

    model = ae(
        aeconfig,
        lossconfig,
        trainconfig,
        normalizer=datamodule.normalizer,
        batch_size=dataconfig["batch_size"],
        accumulation_steps=trainconfig["accumulate_grad_batches"],
    )

    callbacks = checkpoint_callbacks
    if trainconfig.get("enable_plot_callback", True):
        callbacks.append(eval_callback)

    trainer = L.Trainer(
        devices=trainconfig["devices"],
        accelerator=trainconfig["accelerator"],
        check_val_every_n_epoch=trainconfig["check_val_every_n_epoch"],
        log_every_n_steps=trainconfig["log_every_n_steps"],
        max_epochs=trainconfig["max_epochs"],
        max_steps=trainconfig.get("max_steps", -1),
        default_root_dir=path,
        callbacks=callbacks,
        logger=experiment_logger,
        strategy=trainconfig["strategy"],
        accumulate_grad_batches=trainconfig["accumulate_grad_batches"],
        precision=trainconfig.get("precision", "32-true"),
        deterministic=trainconfig.get("deterministic", False),
        num_sanity_val_steps=trainconfig.get("num_sanity_val_steps", 2),
        limit_train_batches=int(trainconfig["limit_train_batches"])
        if "limit_train_batches" in trainconfig
        else 1.0,
        limit_val_batches=int(trainconfig["limit_val_batches"])
        if "limit_val_batches" in trainconfig
        else 1.0,
    )

    if trainconfig["checkpoint"] is not None:
        trainer.fit(
            model=model, datamodule=datamodule, ckpt_path=trainconfig["checkpoint"]
        )
    else:
        trainer.fit(model=model, datamodule=datamodule)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a AE")
    parser.add_argument("--config", default=None)
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--checkpoint", default=None)
    args = parser.parse_args()

    main(args)
