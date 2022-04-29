"""Entrypoint of the package."""
import json

from core.config import DATASET
from core.calibration import Calibration

if __name__ == "__main__":
  config: dict = {}

  with open(DATASET, "r") as fh:
    config = json.load(fh)

  calib = Calibration(config["calibration"])
