import numpy as np
import scipy.interpolate as sp_interp

from worldengine.components.maps.lanes.center_lane import CenterLane
from worldengine.utils import math_utils


INITIAL_CURVATURE_PENALTY = 1e-10


def _make_banded_difference_matrix(number_rows: int):
    """
    Returns a banded difference matrix with specified number_rows.
    When applied to a vector [x_1, ..., x_N], it returns [x_2 - x_1, ..., x_N - x_{N-1}].
    :param number_rows: The row dimension of the banded difference matrix (e.g. N-1 in the example above).
    :return: A banded difference matrix with shape (number_rows, number_rows+1).
    """
    banded_matrix = -1.0 * np.eye(number_rows + 1, dtype=np.float64)[:-1, :]
    for ind in range(len(banded_matrix)):
        banded_matrix[ind, ind + 1] = 1.0

    return banded_matrix


def _generate_profile_from_initial_condition_and_derivatives(
     initial_condition: float,
     derivatives: np.ndarray,
     discretization_time: float):
    """
    Returns the corresponding profile (i.e. trajectory) given an initial condition and derivatives at
    multiple timesteps by integration.
    :param initial_condition: The value of the variable at the initial timestep.
    :param derivatives: The trajectory of time derivatives of the variable at timesteps 0,..., N-1.
    :param discretization_time: [s] Time discretization used for integration.
    :return: The trajectory of the variable at timesteps 0,..., N.
    """
    assert discretization_time > 0.0, "Discretization time must be positive."

    profile = initial_condition + np.insert(np.cumsum(derivatives * discretization_time), 0, 0.0)

    return profile  # type: ignore


def get_interpolated_reference_trajectory_poses(
    planning_trajectory: CenterLane,
    planning_waypoints: np.ndarray,
    planning_interval: float,
    discretization_time: float,
    start_time: float,
    end_time: float):
    """
    Resamples the reference trajectory at discretization_time resolution.
    It will return N times and poses, where N is a function of the trajectory duration and the discretization time.

    Args:
        planning_trajectory: A centerlane object with trackable orientation and position.
        planning_waypoints: An ndarray object with shape [N, 2], indicating the planned
            position at each planning interval.
        planning_interval: [s], A float number indicating the actual time_iterval
            of different planning_waypoints.
        discretization_time: [s], A float number for resampling the trajectory.

    Return:
        An array of times in seconds (N) and an array of associated poses (N,3), sampled at the discretization time.
    """
    assert planning_waypoints.ndim == 2, 'The planning waypoints should have a shape as [N, 2].'
    interpolation_time = np.arange(start_time, end_time, discretization_time)

    # Get global coordinates each timestamp.
    index_list, position_list, heading_list = [], [], []
    for segment in planning_trajectory.lane_segments:
        start_idx = segment['start_idx']
        if start_idx not in index_list:
            index_list.append(start_idx)
            position_list.append(segment['start_point'])
            heading_list.append(segment['heading'])

        end_idx = segment['end_idx']
        if end_idx not in index_list:
            index_list.append(end_idx)
            position_list.append(segment['end_point'])
            heading_list.append(segment['heading'])

    time_series = np.array(index_list) * planning_interval

    # position interpolation function.
    linear_pos_interp_func = sp_interp.interp1d(time_series, np.array(position_list), axis=0)
    interpolation_position = linear_pos_interp_func(interpolation_time)

    # heading interpolation function.
    heading_list = np.unwrap(np.array(heading_list))
    linear_heading_interp_func = sp_interp.interp1d(time_series, heading_list, axis=0)
    interpolation_heading = math_utils.principal_value(linear_heading_interp_func(interpolation_time))

    return interpolation_time, interpolation_position, interpolation_heading


def _fit_initial_velocity_and_acceleration_profile(
    xy_displacements: np.ndarray,
    heading_s: np.ndarray,
    discretization_time: float,
    jerk_penalty: float):
    """
    According to the xy-displacements, estimates:
        - initial velocity (v_0)
        - and associated acceleration ({a_0, ...})
    by least square optimization.

    Args:
        xy_displacements: A float32 ndarray with shape as [n-1, 2] given n positions.
        heading_s: [rad] A float32 ndarray with shape as [n-1], indicating the heading of ego-vehicle
            at each timestamp.
        discretization_time: [s] Time discretization used for integration.
        jerk_penalty: penalty for acceleration differences.  Should be positive.

    Return:
        Least squares solution for initial velocity (v_0) and acceleration profile ({a_0, ..., a_M-1})
             for M displacement values.
    """
    assert discretization_time > 0.0, "Discretization time must be positive."
    assert jerk_penalty > 0, "Should have a positive jerk_penalty."

    assert len(xy_displacements.shape) == 2, "Expect xy_displacements to be a matrix."
    assert xy_displacements.shape[1] == 2, "Expect xy_displacements to have 2 columns."

    num_displacements = len(xy_displacements)  # aka M in the docstring

    assert heading_s.shape == (
        num_displacements,
    ), "Expect the length of heading to match that of xy_displacements."

    # Core problem: minimize_x ||y-Ax||_2
    y = xy_displacements.flatten()  # Flatten to a vector, [delta x_0, delta y_0, ...]

    # build the discretization_time and heading matrix.
    #  Thus, A @ [v_0, a_0, a_1, a_2, ...] = xy_displacements
    A = np.zeros((2 * num_displacements, num_displacements), dtype=np.float32)
    for idx, heading in enumerate(heading_s):
        cur_row = 2 * idx

        # dist_x = cos(heading) * time * v + cos(heading) * time^2 * a
        A[cur_row:(cur_row + 2), 0] = np.array([
            np.cos(heading) * discretization_time,
            np.sin(heading) * discretization_time
        ])

        if idx > 0:
            A[cur_row:(cur_row + 2), 1:(1 + idx)] = np.array([
                np.cos(heading) * discretization_time ** 2,
                np.sin(heading) * discretization_time ** 2
            ])[:, np.newaxis]

    # Regularization using jerk penalty, i.e. difference of acceleration values.
    # If there are M displacements, then we have M - 1 acceleration values.
    # That means we have M - 2 jerk values, thus we make a banded difference matrix of that size.
    banded_matrix = _make_banded_difference_matrix(num_displacements - 2)
    R = np.block([np.zeros((len(banded_matrix), 1)), banded_matrix])

    # Compute regularized least squares solution.
    x = np.linalg.pinv(A.T @ A + jerk_penalty * R.T @ R) @ A.T @ y

    # Extract profile from solution.
    initial_velocity = x[0]
    acceleration_profile = x[1:]

    return initial_velocity, acceleration_profile


def _fit_initial_curvature_and_curvature_rate_profile(
    heading_displacements: np.ndarray,
    velocity_profile: np.ndarray,
    discretization_time: float,
    curvature_rate_penalty: float,
    initial_curvature_penalty: float = INITIAL_CURVATURE_PENALTY):
    """
    According to the heading-displacements, estimates:
        - initial curvature (curvature_0)
        - and associated curvature_rate ({a_0, ...})
    by least square optimization.

    Note that, the curvature is something like 1/R where R is the radius of curvation circle.

    Args:
        heading_displacements: [rad] A float32 ndarray with shape as [n-1],
            Angular deviations in heading occuring between timesteps.
        velocity_profile: [m/s] A float32 ndarray with shape as [n-1, 2]
            Estimated or actual velocities at the timesteps matching displacements.
        discretization_time: [s] Time discretization used for integration.
        curvature_rate_penalty: penalty for curvature_rate.  Should be positive.
        initial_curvature_penalty: A regularization parameter to handle zero initial speed.  Should be positive and small.

    Return:
        Least squares solution for initial curvature (curvature_0) and curvature rate profile
             (curvature_rate_0, ..., curvature_rate_{M-1}) for M heading displacement values.
    """
    assert discretization_time > 0.0, "Discretization time must be positive."
    assert curvature_rate_penalty > 0.0, "Should have a positive curvature_rate_penalty."
    assert initial_curvature_penalty > 0.0, "Should have a positive initial_curvature_penalty."

    # Core problem: minimize_x ||y-Ax||_2
    y = heading_displacements
    num_displacements = len(y)

    # build the discretization_time and velocity matrix.
    #  Thus, A @ [curvature_0, curvature_rate_0, ...,] = heading_displacement.
    #  heading_displacement = velocity x discretization_time x curvature.
    A = np.tri(num_displacements, dtype=np.float32)  # lower triangular matrix
    A[:, 0] = velocity_profile * discretization_time

    # for later frame:
    #  heading_displacement =
    #   discretization_time x velocity x curvature_0 (v0t) +
    #   discretization_time ** 2 x velocity x curvature_rate ... (at**2)
    for idx, velocity in enumerate(velocity_profile):
        if idx == 0:
            continue
        A[idx, 1:] *= velocity * discretization_time**2

    # Regularization on curvature rate.  We add a small but nonzero weight on initial curvature too.
    # This is since the corresponding row of the A matrix might be zero if initial speed is 0, leading to singularity.
    # We guarantee that Q is positive definite such that the minimizer of the least squares problem is unique.
    Q = curvature_rate_penalty * np.eye(num_displacements)
    Q[0, 0] = initial_curvature_penalty

    # Compute regularized least squares solution.
    x = np.linalg.pinv(A.T @ A + Q) @ A.T @ y

    # Extract profile from solution.
    initial_curvature = x[0]
    curvature_rate_profile = x[1:]

    return initial_curvature, curvature_rate_profile


def get_velocity_curvature_profiles_with_derivatives_from_poses(
    discretization_time: float,
    pos_s: np.ndarray,
    heading_s: np.ndarray,
    jerk_penalty: float,
    curvature_rate_penalty: float):
    """
    Main function for joint estimation of velocity, acceleration, curvature, and curvature rate given N poses
    sampled at discretization_time.  This is done by solving two least squares problems with the given penalty weights.
    :param discretization_time: [s] Time discretization used for integration.
    :param poses: <np.ndarray: num_poses, 3> A trajectory of N poses (x, y, heading).
    :param jerk_penalty: A regularization parameter used to penalize acceleration differences.  Should be positive.
    :param curvature_rate_penalty: A regularization parameter used to penalize curvature_rate.  Should be positive.
    :return: Profiles for velocity (N-1), acceleration (N-2), curvature (N-1), and curvature rate (N-2).
    """

    assert pos_s.shape[0] > 1, "Cannot get position displacements given an empty or single " \
                               "element pose trajectory."
    assert pos_s.shape[1] == 2, "Expect positions to have two elements (x, y)."
    xy_displacements = np.diff(pos_s, axis=0)

    assert heading_s.shape[0] > 1, "Cannot get heading displacements given an empty or single " \
                                   "element pose trajectory."
    heading_displacements = math_utils.principal_value(np.diff(heading_s, axis=0))

    # Compute initial velocity + acceleration least squares solution and extract results.
    # Note: If we have M displacements, we require the M associated heading values.
    #       Therefore, we exclude the last heading in the call below.
    initial_velocity, acceleration_profile = _fit_initial_velocity_and_acceleration_profile(
        xy_displacements=xy_displacements,
        heading_s=heading_s[:-1],
        discretization_time=discretization_time,
        jerk_penalty=jerk_penalty,
    )

    velocity_profile = _generate_profile_from_initial_condition_and_derivatives(
        initial_condition=initial_velocity,
        derivatives=acceleration_profile,
        discretization_time=discretization_time,
    )

    # Compute initial curvature + curvature rate least squares solution and extract results.  It relies on velocity fit.
    initial_curvature, curvature_rate_profile = _fit_initial_curvature_and_curvature_rate_profile(
        heading_displacements=heading_displacements,
        velocity_profile=velocity_profile,
        discretization_time=discretization_time,
        curvature_rate_penalty=curvature_rate_penalty,
    )

    curvature_profile = _generate_profile_from_initial_condition_and_derivatives(
        initial_condition=initial_curvature,
        derivatives=curvature_rate_profile,
        discretization_time=discretization_time,
    )

    return velocity_profile, acceleration_profile, curvature_profile, curvature_rate_profile