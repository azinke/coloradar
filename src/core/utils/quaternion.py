"""Quaternion.

Collection of function for manipulating quaternions
"""
import numpy as np


def quaternion_to_rotation_matrix(q: np.array) -> np.matrix:
    """Convert a quaternion into a rotation matrix.

                r00     r01     r02
        Rt =    r10     r11     r12
                r20     r21     r22
    """
    q1, q2, q3, q0 = q

    # 1rst row of the rotation matrix
    r00 = 2 * (q0 * q0 + q1 * q1) - 1
    r01 = 2 * (q1 * q2 - q0 * q3)
    r02 = 2 * (q1 * q3 + q0 * q2)

    # 2nd row of the rotation matrix
    r10 = 2 * (q1 * q2 + q0 * q3)
    r11 = 2 * (q0 * q0 + q2 * q2) - 1
    r12 = 2 * (q2 * q3 - q0 * q1)

    # 3rd row of the rotation matrix
    r20 = 2 * (q1 * q3 - q0 * q2)
    r21 = 2 * (q2 * q3 + q0 * q1)
    r22 = 2 * (q0 * q0 + q3 * q3) - 1

    return np.matrix([[r00, r01, r02], [r10, r11, r12], [r20, r21, r22]])

# Alias name of the function above
q2r = quaternion_to_rotation_matrix


def quaternion_to_extended_rotation_matrix(q: np.array, t: np.array) -> np.matrix:
    """Quaternion and translation vector to rotation matrix.

    Convert a quaternion and a translation matrix into an extended rotation matrix.

                r00     r01     r02     t0
        Rt =    r10     r11     r12     t1
                r20     r21     r22     t2
                 0       0       0       1
    """
    r = quaternion_to_rotation_matrix(q)
    r = np.column_stack((r, t))
    return np.row_stack((r, [0., 0., 0., 1.0]))

# Alias name for the function above
q2er = quaternion_to_extended_rotation_matrix
