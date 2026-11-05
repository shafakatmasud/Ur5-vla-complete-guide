"""Locations of project artifacts relative to this central repository."""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT / "ros2_ws"
PACKAGE = WORKSPACE / "src" / "ur5-vla-dataset-recorder"
DATA = ROOT / "data" / "dataset"
MODELS = ROOT / "data" / "models"
DOWNLOADED_MODELS = ROOT / "dump" / "downloaded_models"
HF_CACHE = ROOT / "data" / "huggingface_cache"
LEROBOT_CACHE = ROOT / "data" / "hf_lerobot_cache"


def inference_environment() -> dict[str, str]:
    """Launch the model server with only this repository's project paths."""
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(PACKAGE) + os.pathsep + str(ROOT / "lerobot/src")
    environment["PYTHONNOUSERSITE"] = "1"
    environment["CUDA_VISIBLE_DEVICES"] = "0"
    environment["HF_HUB_OFFLINE"] = "1"
    environment["OMP_NUM_THREADS"] = "4"
    environment["HF_HOME"] = str(HF_CACHE)
    environment["HF_LEROBOT_HOME"] = str(LEROBOT_CACHE)
    return environment
