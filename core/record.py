"""Record.

A record is a collection of measurement from different sensors
(lidar, radar, imu, etc.)
"""
from glob import glob
import sys
import os
import multiprocessing

import matplotlib.pyplot as plt
import matplotlib.animation as anim
import cv2 as cv

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
        self.codename: str = codename
        subdir: str = ""
        for dataset in descriptor["folders"]:
            if dataset["codename"] == codename:
                subdir = dataset["path"]
                break
        if not subdir:
            error(f"Dataset codename '{codename}' not defined in '{DATASET}")
            sys.exit(1)
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

    def process_and_save(self, sensor: str, **kwargs) -> None:
        """Process and save the result into an output folder.

        Arguments:
            sensor (str): Name of sensor of interest
                          Values: "lidar", "scradar", "ccradar"
            kwargs (dict): Keyword argument
                "threshold": Threshold value to be used for rendering
                             radar heatmap
                "no_sidelobe": Ignore closest recording in each frame
                "velocity_view": Enable the rendering of radial velocity
                                 as fourth dimention
                "heatmap_3d": Save 3D heatmap when true. Otherwise, a
                              2D heatmap is generated
        """
        # Dot per inch
        self._dpi: int = 400
        self._kwargs = kwargs
        self._sensor = sensor

        # Output directory path
        output_dir: str = kwargs.get("output", "output")
        output_dir = f"{output_dir}/{self.codename}/{sensor}"
        os.makedirs(output_dir, exist_ok=True)
        self._output_dir = output_dir
        cpu_count: int = multiprocessing.cpu_count()
        print(f"Please wait! Processing on {cpu_count} CPU(s)")

        if sensor == "lidar":
            dataset_path: str = os.path.join(
                self.descriptor["paths"]["rootdir"],
                self.descriptor["paths"][sensor]["data"]
            )
            nb_files: int = len(os.listdir(dataset_path)) - 1
            with multiprocessing.Pool(cpu_count) as pool:
                pool.map(
                    self._process_lidar,
                    range(1, nb_files + 1),
                    chunksize=10
                )
        elif (sensor == "ccradar") or (sensor == "scradar"):
            dataset_path: str = os.path.join(
                self.descriptor["paths"]["rootdir"],
                self.descriptor["paths"][sensor]["raw"]["data"]
            )
            nb_files: int = len(os.listdir(dataset_path)) - 1
            with multiprocessing.Pool(cpu_count) as pool:
                pool.map(
                    self._process_radar,
                    range(1, nb_files + 1),
                    chunksize=10
                )

    def _process_radar(self, idx: int) -> int:
        """Handler of radar data processing.

        Used as the handler for parallel processing. The context attributes
        needed by this method are only defined in the method `process_and_save`
        As so, only that method is supposed to call this one.

        NOTE: THIS METHOD IS NOT EXPECTED TO BE CALLED FROM OUTSIDE OF THIS
        CLASS

        Argument:
            idx: Index of the file to process
        """
        self.index = idx
        self.load(self._sensor)
        SIZE: int = 20   # inch
        plt.figure(1, clear=True, dpi=self._dpi, figsize=(SIZE, SIZE))
        if self._kwargs.get("heatmap_3d") == False:
            self.ccradar.show2dHeatmap(False, False)
        elif self._kwargs.get("heatmap_3d"):
            self.ccradar.showHeatmapFromRaw(
                self._kwargs.get("threshold"),
                self._kwargs.get("no_sidelobe"),
                self._kwargs.get("velocity_view"),
                self._kwargs.get("polar"),
                show=False,
            )
        elif self._kwargs.get("pointcloud"):
            self.ccradar.showPointcloudFromRaw(
                self._kwargs.get("velocity_view"),
                self._kwargs.get("bird_eye_view"),
                self._kwargs.get("polar"),
                show=False,
            )
        plt.savefig(f"{self._output_dir}/radar_{idx:04}.jpg", dpi=self._dpi)
        return idx

    def _process_lidar(self, idx: int) -> int:
        """Handler of lidar data processing.

        Used as the handler for parallel processing. The context attributes
        needed by this method are only defined in the method `process_and_save`
        As so, only that method is supposed to call this one.

        NOTE: THIS METHOD IS NOT EXPECTED TO BE CALLED FROM OUTSIDE OF THIS
        CLASS

        Argument:
            idx: Index of the file to process
        """
        self.index = idx
        self.load(self._sensor)
        bev = self.lidar.getBirdEyeView(
            self._kwargs.get("resolution", 0.05),
            self._kwargs.get("srange"),
            self._kwargs.get("frange"),
        )
        plt.imsave(f"{self._output_dir}/lidar_bev_{idx:04}.jpg", bev)

    def make_video(self, inputdir: str, ext: str = "jpg") -> None:
        """Make video out of pictures"""
        files = glob(inputdir + f"/*.{ext}")
        files = sorted(files)
        height, width, _ = plt.imread(files[0]).shape
        fourcc = cv.VideoWriter_fourcc(*'MJPG')
        video = cv.VideoWriter(inputdir + f"/{self.codename}.avi", fourcc, 5, (width, height))
        for idx, img in enumerate(files):
            print(
                f"[ ========= {100 * idx/len(files): 2.2f}% ========= ]\r",
                end=""
            )
            video.write(cv.imread(img))
        cv.destroyAllWindows()
        video.release()
