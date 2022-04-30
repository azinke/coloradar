"""Radar.

SCRadar: Single Chip Radar Sensor
CCRadar: Cascade Chip Radar Sensor
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

from core.calibration import Calibration, SCRadarCalibration
from core.utils.common import error
from .lidar import Lidar

class SCRadar(Lidar):
    """Radar.

    Attrinutes:
        NUMBER_RECORDING_ATTRIBUTES: Number of 32-bit integer packed
        to form a single recording measurement
    """

    # The recorded attributes are:
    # x, y, z, I (Intensity of the reflections), Vr (Radial velocity)
    NUMBER_RECORDING_ATTRIBUTES: int = 5

    def __init__(self, config: dict[str, str],
                 calib: Calibration, index: int) -> None:
        """Init.

        Arguments:
            config (dict): Paths to access the dataset
            calib: Calibration object (See calibration.py)
            index (int): Index of the lidar record to load
        """
        sensor: str = self.__class__.__name__.lower()
        self.calibration: SCRadarCalibration = getattr(calib, sensor)

        # Read pointcloud
        filename: str = self._filename(
            config["paths"][sensor]["pointcloud"]["filename_prefix"],
            index,
            "bin"
        )
        self.filepath = os.path.join(
            config["paths"]["rootdir"],
            config["paths"][sensor]["pointcloud"]["data"],
            filename
        )
        try:
            cld = np.fromfile(self.filepath, np.float32)
            self.cld = np.reshape(cld, (-1, self.NUMBER_RECORDING_ATTRIBUTES))
        except FileNotFoundError:
            error(f"File '{self.filepath}' not found.")
            sys.exit(1)

        # Read heatmap
        filename = self._filename(
            config["paths"][sensor]["heatmap"]["filename_prefix"],
            index,
            "bin"
        )
        self.heatmap_filepath = os.path.join(
            config["paths"]["rootdir"],
            config["paths"][sensor]["heatmap"]["data"],
            filename
        )
        try:
            heatmap = np.fromfile(self.heatmap_filepath, np.float32)
            self.heatmap = np.reshape(
                heatmap,
                (
                    self.calibration.heatmap.num_elevation_bins,
                    self.calibration.heatmap.num_azimuth_bins,
                    self.calibration.heatmap.num_range_bins,
                    2 # Number of value per bin (intensity and location)
                )
            )
        except FileNotFoundError:
            error(f"File '{self.heatmap_filepath}' not found.")
            sys.exit(1)

    def showHeatmap(self, threshold: float = 0.15, render: bool = True) -> None:
        """Render heatmap.

        Argument:
            threshold: Value used to filter the pointcloud
            render: Flag triggering the rendering of the heatmap when 'true'.
        """
        ax = plt.axes(projection="3d")
        ax.set_title("Heatmap")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        pcl = self._heatmap_to_pointcloud(threshold)
        plot = ax.scatter(
            pcl[:, 0],
            pcl[:, 1],
            pcl[:, 2],
            c=pcl[:, 3],
            cmap=plt.cm.get_cmap(),
        )
        colorbar = plt.colorbar(plot)
        colorbar.set_label("Reflection Intensity")
        if render:
            plt.show()

    def showHeatmapBirdEyeView(self, threshold: float) -> None:
        """Render the bird eye view of the heatmp pointcloud.

        Argument:
            threshold (float): Threshold to filter the pointcloud
        """
        pointcloud = self._heatmap_to_pointcloud(threshold)
        x = pointcloud[:, 0]
        y = pointcloud[:, 1]
        z = pointcloud[:, 2]

        _range_bin_width: float = self.calibration.heatmap.range_bin_width
        _num_r_num: int = self.calibration.heatmap.num_range_bins
        max_range = np.ceil(_range_bin_width * _num_r_num)

        resolution: float = _range_bin_width/10
        srange: tuple[float, float] = (-max_range, max_range)
        frange: tuple[float, float] = (0, max_range)

        ximg = (-y / resolution).astype(np.int32)
        yimg = (-x / resolution).astype(np.int32)

        ximg -= int(np.floor(srange[0] / resolution))
        yimg += int(np.floor(frange[1] / resolution))

        # Prepare the three channels of the bird eye view image
        pixels = np.zeros((len(z), 3), dtype=np.uint8)

        # Encode distance
        norm = np.sqrt(x **2 + y **2 + z**2)
        pixels[:, 0] = (255.0 / (1.0 + np.exp(-norm))).astype(np.uint8)

        # Encode radial velosity of the radar
        pixels[:, 1] = (
            255.0 / (1.0 + np.exp(-np.abs(pointcloud[:, 4])))
        ).astype(np.uint8)

        # Encode reflection intensity information
        pixels[:, 2] = (255.0 * pointcloud[:, 3]).astype(np.uint8)

        # Create the frame for the bird eye view
        # Note: the "+1" to estimate the width and height of the image is
        # to count for the (0, 0) position in the center of the pointcloud
        img_width: int = 1 + int((srange[1] - srange[0])/resolution)
        img_height: int = 1 + int((frange[1] - frange[0])/resolution)
        bev_img = np.zeros([img_height, img_width, 3], dtype=np.uint8)

        # Set the height information in the created image frame
        bev_img[yimg, ximg] = pixels

        plt.imshow(bev_img)
        plt.title(f"Heatmap BEV - ~{resolution:.4} m/pixel")
        plt.show()

    def _polar_to_cartesian(self, r_idx: int,
            az_idx: int, el_idx: int) -> np.array:
        """Convert polar coordinate to catesian coordinate."""
        _range_bin_width: float = self.calibration.heatmap.range_bin_width
        _el_bin = self.calibration.heatmap.elevation_bins[el_idx]
        _az_bin = self.calibration.heatmap.azimuth_bins[az_idx]

        point = np.zeros(3)
        point[0] = r_idx * _range_bin_width * np.cos(_el_bin) * np.cos(_az_bin)
        point[1] = r_idx * _range_bin_width * np.cos(_el_bin) * np.sin(_az_bin)
        point[2] = r_idx * _range_bin_width * np.sin(_el_bin)
        return point

    def _heatmap_to_pointcloud(self, threshold: float = 0.15) -> np.array:
        """Compute pointcloud from heatmap.

        calculates point locations in the sensor frame for plotting heatmaps

        Argument:
            threshold (float): Threshold to filter the pointcloud

        Return
            pcl (np.array): the heatmap point locations
        """
        _num_el_bin: int = self.calibration.heatmap.num_elevation_bins
        _num_az_bin: int = self.calibration.heatmap.num_azimuth_bins
        _num_r_num: int = self.calibration.heatmap.num_range_bins
        # transform range-azimuth-elevation heatmap to pointcloud
        pcl = np.zeros((_num_el_bin, _num_az_bin, _num_r_num, 5))

        for range_idx in range(1, _num_r_num):
            for az_idx in range(1, _num_az_bin):
                for el_idx in range(1, _num_el_bin):
                    pcl[el_idx, az_idx, range_idx, :3] = self._polar_to_cartesian(
                        range_idx - 1, az_idx - 1, el_idx - 1
                    )
        pcl = pcl.reshape(-1,5)
        pcl[:,3:] = self.heatmap.reshape(-1, 2)
        # Normalise the radar reflection intensity
        pcl[:, 3] -= np.min(pcl[:, 3])
        pcl[:, 3] /= np.max(pcl[:, 3])
        if threshold:
            pcl = pcl[pcl[:, 3] > threshold]
            # Re-Normalise the radar reflection intensity after filtering
            pcl[:, 3] -= np.min(pcl[:, 3])
            pcl[:, 3] /= np.max(pcl[:, 3])
        return pcl
