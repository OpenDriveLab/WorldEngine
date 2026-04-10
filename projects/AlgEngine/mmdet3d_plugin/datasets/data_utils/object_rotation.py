"""Utility functions related to object rotation and orientation computation"""
import math
from typing import List

import numpy as np
import pyquaternion
from numba import jit


def get_yaw_vector(rotation: np.ndarray) -> np.ndarray:
    """Convert yaw from rotation to a vector"""

    yaw_deg = rotation[1]
    yaw_rad = np.radians(yaw_deg)
    return np.array([np.cos(yaw_rad), np.sin(yaw_rad)])


def forward_vec2rot(forward_vector: np.ndarray) -> np.ndarray:
    """Convert a forward vector to rotation vector in degree

    Note: only yaw is considered now

    Args
        forward_vector: (3, ) or (2, ) with z ignored
    """

    yaw: float = np.rad2deg(math.atan2(forward_vector[1], forward_vector[0]))
    return np.array([0, yaw, 0])


def compute_heading_diff(yaw_vec1: np.ndarray, yaw_vec2: np.ndarray) -> float:
    """Compute the difference of two headings, e.g., ego and obj

    Returns:
        heading_diff_deg: float in degree, range [-180, 180] degree
    """

    # make sure that the input dot product is within [-1, 1]
    dot_product: float = yaw_vec1.dot(yaw_vec2)
    dot_product: float = min(1.0, max(-1.0, dot_product))
    heading_diff_rad = np.arccos(dot_product)

    # convert to degrees within [-180, 180]
    heading_diff_deg = np.degrees(heading_diff_rad)
    heading_diff_deg = min(heading_diff_deg, 360.0 - heading_diff_deg)

    return heading_diff_deg


def convert_mat2qua(rotation: np.ndarray) -> np.ndarray:
    """Convert a 3x3 rotation matrix to a (4, ) np array for quaternion"""

    rotation: np.ndarray = pyquaternion.Quaternion(matrix=rotation).elements  # (4, )
    return rotation


# @jit
def rotx(t):
    """Rotation about the x-axis."""
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


# @jit
def roty(t):
    """Rotation about the y-axis."""
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


# @jit
def rotz(t):
    """Rotation about the z-axis."""
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


# @jit
def transform_from_rot_trans(R, t):
    """Transforation matrix from rotation matrix and translation vector."""
    R = R.reshape(3, 3)
    t = t.reshape(3, 1)
    return np.vstack((np.hstack([R, t]), [0, 0, 0, 1]))
