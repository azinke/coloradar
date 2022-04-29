"""Tranform.

Handle the projection of data from the base coordinate system to
a sensor-specific coordinate system.
"""

import os
import numpy as np

from core.config import ROOTDIR
from .utils.quaternion import q2r, q2er


class Transform:
    """Transform.

    Attributes:
        t: Translation vector
        q: Rotation quaternion
        r: Rotation matrix
        rt: Extended rotation matrix
            The extended rotation matrix contain both rotation matrix and
            translation vecteor
    """

    def __init__(self, filepath: str) -> None:
        """Tranform constructor.

        Argument:
            filepath: Path to access the transform file
        """

        with open(os.path.join(ROOTDIR, filepath), "r") as fh:
            line_number: int = 0
            for line in fh:
                line_number += 1
                if line_number == 1:
                    self.t = np.array([float(x) for x in line.split(" ")])
                else:
                    self.q = np.array([float(x) for x in line.split(" ")])
        self.r = q2r(self.q)
        self.rt = q2er(self.q, self.t)


class BaseToCCRadar(Transform):
    """Base to Cascade Chip Radar tranform."""

    pass


class BaseToSCRadar(Transform):
    """Base to Single Chip Radar tranform."""

    pass


class BaseToLidar(Transform):
    """Base to Lidar tranform."""

    pass


class BaseToVicon(Transform):
    """Base to Vicon tranform."""

    pass


class BaseToImu(Transform):
    """Base to IMU tranform."""

    pass
