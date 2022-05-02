"""Record.

A record is a collection of measurement from different sensors
(lidar, radar, imu, etc.)
"""
import sys
import os

from core.config import ROOTDIR, DATASET
from core.lidar import Lidar
from core.radar import SCRadar, CCRadar

from .utils.common import error


class Record:
    """Record.

    Attributes:
        calibration: Calibration parameters
        lidar: Velodyne samples of the dataset record
    """

    def __init__(self, descriptor: dict[str, dict[str, str]],
                 calibration, codename: str, index: int) -> None:
        """Init.

        Arguments:
            descriptor: Holds the paths that describe the dataset
            calibration: Calibration object
                Provide all the calibration parameters of all sensors
            codename: Subdirectory codename
            index: Order number indicating the entry of the dataset in interest
        """
        self.calibration = calibration
        subdir: str = ""
        for dataset in descriptor["folders"]:
            if dataset["codename"] == codename:
                subdir = dataset["path"]
                break
        if not subdir:
            error(f"Dataset codename '{codename}' not defined in '{DATASET}")
        descriptor["paths"]["rootdir"] = os.path.join(ROOTDIR, subdir)
        self.lidar = Lidar(descriptor, self.calibration, index)
        self.scradar = SCRadar(descriptor, self.calibration, index)
        self.ccradar = CCRadar(descriptor, self.calibration, index)
