"""Lidar."""
import sys
import os

import numpy as np
import matplotlib.pyplot as plt

from .utils.common import error


class Lidar(object):
    """Lidar.

    Attributes:
        filepah: Path of the lidar recording
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