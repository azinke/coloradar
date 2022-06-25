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

    Class describing records in the dataset

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
        self.descriptor = descriptor
        self.index = index
        subdir: str = ""
        for dataset in descriptor["folders"]:
            if dataset["codename"] == codename:
                subdir = dataset["path"]
                break
        if not subdir:
            error(f"Dataset codename '{codename}' not defined in '{DATASET}")
        self.descriptor["paths"]["rootdir"] = os.path.join(ROOTDIR, subdir)
        self.lidar = None
        self.scradar = None
        self.ccradar = None

    def load(self, sensor: str) -> None:
        """Load the data file for a given sensor.

        Arguments:
            sensor: The sensor considered so that only the data of that sensor
                    would be loaded.
                    Possible Values: lidar, scradar, ccradar
        """

        if sensor == "lidar":
            self.lidar = Lidar(self.descriptor, self.calibration, self.index)
        elif sensor == "scradar":
            self.scradar = SCRadar(self.descriptor, self.calibration, self.index)
        elif sensor == "ccradar":
            self.ccradar = CCRadar(self.descriptor, self.calibration, self.index)
