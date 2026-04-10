from typing import List, Union
from shapely.geometry import Polygon

import math
import numpy as np
import numpy.typing as npt

number_pos_inf = float("inf")
number_neg_inf = float("-inf")

def not_zero(x: float, eps: float = 1e-2) -> float:
    if abs(x) > eps:
        return x
    elif x > 0:
        return eps
    else:
        return -eps


def wrap_to_pi(x: float) -> float:
    """Wrap the input radian to (-pi, pi]. Note that -pi is exclusive and +pi is inclusive.

    Args:
        x (float): radian.

    Returns:
        The radian in range (-pi, pi].
    """
    angles = x
    angles %= 2 * np.pi
    angles -= 2 * np.pi * (angles > np.pi)
    return angles


def angle_diff(x: float, y: float, period: float) -> float:
    """
    Wrapped from https://github.com/motional/nuplan-devkit/blob/master/nuplan/database/utils/measure.py#L107
    Get the smallest angle difference between 2 angles: the angle from y to x.
    :param x: To angle.
    :param y: From angle.
    :param period: Periodicity for assessing angle difference.
    :return: Signed smallest between-angle difference in range (-pi, pi).
    """
    # calculate angle difference, modulo to [0, 2*pi]
    diff = (x - y + period / 2) % period - period / 2
    if diff > math.pi:
        diff = diff - (2 * math.pi)  # shift (pi, 2*pi] to (-pi, 0]

    return diff


def principal_value(
    angle: Union[float, int, npt.NDArray[np.float32]], min_: float = -np.pi
) -> Union[float, npt.NDArray[np.float32]]:
    """
    Wrap heading angle in to specified domain (multiples of 2 pi alias),
    ensuring that the angle is between min_ and min_ + 2 pi. This function raises an error if the angle is infinite
    :param angle: rad
    :param min_: minimum domain for angle (rad)
    :return angle wrapped to [min_, min_ + 2 pi).
    """
    assert np.all(np.isfinite(angle)), "angle is not finite"

    lhs = (angle - min_) % (2 * np.pi) + min_

    return lhs


def get_vertical_vector(vector: np.array):
    length = norm(vector[0], vector[1])
    # return (vector[1] / length, -vector[0] / length), (-vector[1] / length, vector[0] / length)
    return (-vector[1] / length, vector[0] / length), (vector[1] / length, -vector[0] / length)


def norm(x, y):
    return math.sqrt(x**2 + y**2)


def clip(a, low, high):
    return min(max(a, low), high)


def point_distance(x, y):
    return norm(x[0] - y[0], x[1] - y[1])


def rotate_points(vec, origin, heading):
    """
    Translate and rotate <vec> according to <origin> and <heading>.

    Args:
        vec: ndarray with shape [N, 2]
        origin: [2] x/y
        heading: A float indicating the direction.
    """
    new_vec = vec - origin
    rot_matrix = np.array([
        [math.cos(heading), math.sin(heading)],
        [-math.sin(heading), math.cos(heading)],
    ])
    new_vec = np.matmul(new_vec, rot_matrix)
    return new_vec


def rotate_multi_points(vec, origin, heading):
    """
    Translate and rotate <vec> according to a list of <origin> and <heading>.

    Args:
        vec: ndarray with shape [N, 2]
        origin: [N, 2] x/y
        heading: [N] directions.
    """
    new_vec = vec - origin
    rot_matrix = np.array([
        [np.cos(heading), np.sin(heading)],
        [-np.sin(heading), np.cos(heading)],
    ])  # 2, 2, N
    rot_matrix = np.transpose(rot_matrix, [2, 0, 1])
    new_vec = np.matmul(new_vec[:, np.newaxis, :], rot_matrix)
    return new_vec[:, 0, :]


def translate_longitudinally(pose, heading, distance):
    """
    Translate a pose longitudinally (along heading direction)
    :param pose: SE2 pose to be translated, in global coordinates.
    :param distance: [m] distance by which point (x, y, heading) should be translated longitudinally
    :return translated se2
    """
    local_tensor = np.array([distance, 0]).reshape(1, 2)
    global_tensor = rotate_points(local_tensor, 0, heading)
    return pose + global_tensor

def check_polygon_overlap(poly1: np.ndarray, poly2: np.ndarray) -> bool:
    """
    Use shapely to check if two convex quadrilaterals (as np.ndarray of shape (4, 2)) overlap.
    """
    polygon1 = Polygon(poly1)
    polygon2 = Polygon(poly2)
    return polygon1.intersects(polygon2)  # True if overlap or touch
