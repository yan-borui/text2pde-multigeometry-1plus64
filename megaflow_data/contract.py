"""Published source identity, independent-case split and fixed retained frames."""

from __future__ import annotations

import json
from pathlib import Path

REPOSITORY = "cmudrc/MegaFlow2D"
SOURCE_REVISION = "789ed4cab0a6d25858cc0653f695e3d1896f3241"
REVISION = "megaflow2d.ellipse.high_uvp.stride15.first65.seed42.v1"
FORMAT = "dgn4cfd.megaflow2d_ellipse_uvp.v1"
PROTOCOL = "megaflow2d.ellipse.uvp.stride15.train65.eval65.v1"
EVALUATOR = "megaflow2d.ellipse.uvp.physical_mesh.v1"
DATA_FILE = "ellipse_stride15_65frames.h5"
MANIFEST_FILE = "ellipse_stride15_65frames_manifest.json"
FRAMES = 65
DT = 0.15
RAW_DT = 0.01
FRAME_INDICES = tuple(range(1, 962, 15))
SPLIT_COUNTS = {"train": 782, "validation": 100, "test": 100}


def splits() -> dict[str, tuple[int, ...]]:
    """Use the published lists, never independently resample a method's split."""
    payload = json.loads(Path(__file__).with_name("splits.json").read_text())
    result = {name: tuple(payload["splits"][name]) for name in SPLIT_COUNTS}
    values = [case for members in result.values() for case in members]
    if len(values) != len(set(values)) or any(
        len(result[name]) != size for name, size in SPLIT_COUNTS.items()
    ):
        raise ValueError("invalid published Ellipse split lists")
    return result


def validate_manifest(manifest: dict) -> None:
    """Reject mixed resolutions, frame scopes, source releases and split identities."""
    expected = {
        "format": FORMAT,
        "dataset_revision": REVISION,
        "frames": FRAMES,
        "temporal_stride": 15,
        "frame_dt": DT,
        "raw_frame_dt": RAW_DT,
        "source_frame_indices": list(FRAME_INDICES),
        "splits": {key: list(value) for key, value in splits().items()},
        "source_revision": SOURCE_REVISION,
        "trajectory_count": 982,
        "train_count": 782,
        "validation_count": 100,
        "test_count": 100,
        "boundary_policy": "predict_all_uvp_nodes",
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise ValueError(f"Ellipse manifest mismatch: {key}")
    stats = manifest["train_only_normalization"]
    if stats.get("split") != "train" or stats.get("frame_range") != [0, 64]:
        raise ValueError("normalization must use only the retained Train65 frames")
