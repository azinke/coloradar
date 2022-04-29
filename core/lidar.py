"""Lidar."""
import sys
import os
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt

from .utils.common import error


class Lidar(object):
    """Lidar.

    Attributes:
        filepah: Path of the lidar recording
        cld: Lidar pointcloud
    """

    def __init__(self, config: dict[str, str], index: int) -> None:
        """Init.

        Argument:
            filepath (str): Path to the lidar recording
            index (int): Index of the lidar record to load
        """
        # The recorded attributes are:
        # x, y, z, I (Intensity of the reflections)
        NUMBER_LIDAR_RECORDING_ATTRIBUTES: int = 4

        filename: str = self._filename(
            config["paths"]["lidar"]["filename_prefix"],
            index,
            "bin"
        )
        self.filepath = os.path.join(
            config["paths"]["rootdir"],
            config["paths"]["lidar"]["data"],
            filename
        )
        try:
            cld = np.fromfile(self.filepath, np.float32)
            self.cld = np.reshape(cld, (-1, NUMBER_LIDAR_RECORDING_ATTRIBUTES))
        except FileNotFoundError:
            error(f"File '{self.filepath}' not found.")
            sys.exit(1)

    def show(self, render: bool = True) -> None:
        """Render the lidar pointcloud."""
        ax = plt.axes(projection="3d")
        plot = ax.scatter(
            self.cld[:, 0],
            self.cld[:, 1],
            self.cld[:, 2],
            c=self.cld[:, 3],
            cmap=plt.cm.get_cmap(),
        )
        plt.colorbar(plot)
        if render:
            plt.show()

    def getPointlcoud(self) -> np.matrix:
        """Return the lidar pointcloud."""
        return np.matrix(self.cld)

    def getPointCloud(self, **kwargs) -> np.array:
        """Get the x, y, z coordinate from the lidar dataset.
        Return
            Format:
                X001 Y001 Z001
                X002 Y002 Z002
                X003 Y003 Z003
                ...
            Shape: (n, 3)
        """
        return self.cld[:, 0:3]

    def getPointCloudSample(self,
            x_filter: Optional[tuple[float, float]] = None,
            y_filter: Optional[tuple[float, float]] = None,
            z_filter: Optional[tuple[float, float]] = None,
            full: bool = False) -> np.array:
        """Extract a part of the pointcloud based on the filters.

        Arguments:
            x_filter: tuple containing the min and max values defining
                      the x-axis boundary
            y_filter: tuple containing the min and max values defining
                      the y-axis boundary
            z_filter: tuple containing the min and max values defining
                      the z-axis boundary
                      If z_filter is "None", the pointcloud is not filtered
                      in regard of the z-axis.
            full: Boolean flag. When it's "True" all the entries of the pointcould
                  are returned. If not, only the (x, y, z) columns of the dataset
                  are returned.
        Note: Each filtering tuple is compose of min value and max value (in
              that order)
        """
        if full:
            pointcloud = self.cld
        else:
            pointcloud = self.getPointCloud()
        filtering_mask = True
        if x_filter:
            x_min, x_max = x_filter
            filtering_mask = filtering_mask & (pointcloud[:, 0] >= x_min)
            filtering_mask = filtering_mask & (pointcloud[:, 0] <= x_max)
        if y_filter:
            y_min, y_max = y_filter
            filtering_mask = filtering_mask & (pointcloud[:, 1] >= y_min)
            filtering_mask = filtering_mask &  (pointcloud[:, 1] <= y_max)
        if z_filter:
            z_min, z_max = z_filter
            filtering_mask = filtering_mask & (pointcloud[:, 2] >= z_min)
            filtering_mask = filtering_mask &  (pointcloud[:, 2] <= z_max)
        return pointcloud[filtering_mask]


    def getBirdEyeView(self, resolution: float,
                       srange: tuple[float, float], # side range
                       frange: tuple[float, float], # forward range
                       ) -> None:
        """Generate the bird eye view of the pointcloud.
        Arguments:
            resoluton: The pixel resolution of the image to generate
            srange: Side range to cover.
                Format: (srange_min, srange_max)
            frange: Forward range to cover.
                Format (frange_min, frange_max)
        Note: The ranges are expected to be provided the minimum first and then
        the maxnimum
                -----------------   <-- frange_max
                |               |
                |        ^      |
                |        |      |
                |    <-- 0      |
                |               |
                |               |
                -----------------   <-- frange_min
                ^               ^
            srange_min      srange_max
        """
        pointcloud = self.getPointCloudSample(frange, srange, z_filter=None, full=True)
        x = pointcloud[:, 0]
        y = pointcloud[:, 1]
        z = pointcloud[:, 2]

        ximg = (-y / resolution).astype(np.int32)
        yimg = (-x / resolution).astype(np.int32)

        ximg -= int(np.floor(srange[0] / resolution))
        yimg += int(np.floor(frange[1] / resolution))

        # Prepare the three channels of the bird eye view image
        pixels = np.zeros((len(z), 3), dtype=np.uint8)

        # Encode distance
        norm = np.sqrt(x **2 + y **2 + z**2)
        pixels[:, 0] = (255.0 / (1.0 + np.exp(-norm))).astype(np.uint8)

        # Encode height information
        pixels[:, 1] = (255.0 / (1.0 + np.exp(-z))).astype(np.uint8)

        # Encode intensity (for lidar) and radial velosity (for radar)
        pixels[:, 2] = pointcloud[:, 3]
        # Scale pixels between 0 and 2555
        minv = min(pointcloud[:, 3])
        maxv = max(pointcloud[:, 3])
        pixels[:, 2] = (
            ((pixels[:, 2] - minv) / np.abs(maxv - minv)) * 255
        ).astype(np.uint8)

        # Create the frame for the bird eye view
        # Note: the "+1" to estimate the width and height of the image is
        # to count for the (0, 0) position in the center of the pointcloud
        img_width: int = 1 + int((srange[1] - srange[0])/resolution)
        img_height: int = 1 + int((frange[1] - frange[0])/resolution)
        bev_img = np.zeros([img_height, img_width, 3], dtype=np.uint8)

        # Set the height information in the created image frame
        bev_img[yimg, ximg] = pixels
        return bev_img

    def _filename(self, basename: str, index: int, ext: str) -> str:
        """Filename builder.

        Generate a typical filename in the Coloradar dataset given its index

        Arguments:
            basename: Filename prefix
            index: Order number of the file in the dataset
            ext: Extension of the file
        Return:
            name of the file in the dataset
        """
        return basename + str(index) + "." + ext
