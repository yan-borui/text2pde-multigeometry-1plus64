import lightning as L
from torch.utils.data import DataLoader
from modules.modules.normalizer import Normalizer
from modules.utils import Struct
import copy


class FluidsDataModule(L.LightningDataModule):
    def __init__(
        self,
        dataconfig,
    ) -> None:

        super().__init__()
        dataset_config = copy.deepcopy(dataconfig["dataset"])
        normalizer_config = dataconfig["normalizer"]
        self.batch_size = dataconfig["batch_size"]
        self.num_workers = dataconfig["num_workers"]
        self.mode = dataconfig["mode"]
        self.normalizer = None
        self.sampler_seed = int(dataconfig.get("sampler_seed", 0))
        self.train_start_offset = int(dataconfig.get("train_start_offset", 0))
        self.train_examples_seen = int(dataconfig.get("train_examples_seen", 0))

        if "drop_last" in dataconfig.keys():
            self.drop_last = dataconfig["drop_last"]
        else:
            self.drop_last = False

        if self.mode == "ns2D":
            from dataset.smoke_data import (
                train_datapipe_ns_cond,
                valid_datapipe_ns_cond,
            )

            self.train_dataset = train_datapipe_ns_cond(
                Struct(**dataset_config)
            )  # change dict to object to support dot notation
            self.val_dataset = valid_datapipe_ns_cond(Struct(**dataset_config))

        elif self.mode == "cylinder":
            from dataset.cylinder import CylinderMeshDataset

            data_dir = copy.copy(dataconfig["dataset"]["data_dir"])
            dataset_config["data_dir"] = data_dir + "/train_downsampled_labeled.h5"
            self.train_dataset = CylinderMeshDataset(**dataset_config)
            dataset_config["data_dir"] = data_dir + "/valid_downsampled_labeled.h5"
            self.val_dataset = CylinderMeshDataset(**dataset_config)

        elif self.mode == "multigeometry":
            from dataset.multigeometry import MultiGeometryWindowDataset

            self.train_dataset = MultiGeometryWindowDataset(
                manifest_path=dataset_config["train_manifest"],
                max_open_files=dataset_config.get("max_open_files", 8),
            )
            self.val_dataset = MultiGeometryWindowDataset(
                manifest_path=dataset_config["validation_manifest"],
                max_open_files=dataset_config.get("max_open_files", 8),
            )

        elif self.mode == "cylinderflow_windows":
            from dataset.cylinderflow import CylinderFlowWindowDataset

            if self.num_workers != 0:
                raise ValueError(
                    "cylinderflow_windows requires num_workers=0 for one safe lazy "
                    "HDF5 handle per training process"
                )
            self.train_dataset = CylinderFlowWindowDataset(
                manifest_path=dataset_config["train_manifest"],
            )
            self.val_dataset = CylinderFlowWindowDataset(
                manifest_path=dataset_config["validation_manifest"],
            )

        elif self.mode == "cylinderflow_stride8":
            from dataset.cylinderflow_stride8 import (
                CylinderFlowStride8TrajectoryDataset,
            )

            if self.num_workers != 0:
                raise ValueError(
                    "cylinderflow_stride8 requires num_workers=0 for a safe lazy "
                    "HDF5 handle"
                )
            shared = {
                "manifest_path": dataset_config["manifest"],
                "data_path": dataset_config["data_path"],
                "strict_formal_counts": dataset_config.get(
                    "strict_formal_counts", True
                ),
                "stage": dataset_config.get("stage", "ldm"),
                "sequence_start": dataset_config.get("sequence_start", 0),
                "sequence_length": dataset_config.get("sequence_length"),
            }
            self.train_dataset = CylinderFlowStride8TrajectoryDataset(
                split="train", **shared
            )
            self.val_dataset = CylinderFlowStride8TrajectoryDataset(
                split="validation", **shared
            )

        else:
            raise ValueError(f"Unsupported data mode: {self.mode}")

        self.normalizer = Normalizer(dataset=self.train_dataset, **normalizer_config)

    def prepare_data(self):
        # download, split, etc...
        # only called on 1 GPU/TPU in distributed
        pass

    def setup(self, stage: str):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit":
            pass

        # Assign test dataset for use in dataloader(s)
        if stage == "test":
            pass

        if stage == "predict":
            pass

    def train_dataloader(self):
        self.pin_memory = False if self.num_workers == 0 else True
        sampler = None
        shuffle = True
        if self.mode == "cylinderflow_windows":
            from modules.modules.reproducible_resume import (
                DeterministicPermutationSampler,
            )

            sampler = DeterministicPermutationSampler(
                self.train_dataset,
                seed=self.sampler_seed,
                start_offset=self.train_start_offset,
            )
            shuffle = False
        elif self.mode == "cylinderflow_stride8":
            from modules.modules.reproducible_resume import (
                DeterministicEpochPermutationSampler,
            )

            sampler = DeterministicEpochPermutationSampler(
                self.train_dataset,
                seed=self.sampler_seed,
                start_examples_seen=self.train_examples_seen,
            )
            shuffle = False
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=self.drop_last,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=self.drop_last,
        )

    def test_dataloader(self):
        return None

    def predict_dataloader(self):
        return None
