#   Heavily borrowed from:
#   https://github.com/autonomousvision/tuplan_garage (Apache License 2.0)
# & https://github.com/motional/nuplan-devkit (Apache License 2.0)

from typing import Tuple
import torch
import math
import numpy as np
import numpy.typing as npt
import warnings
# from nuplan.common.geometry.compute import principal_value

# Default regularization weight for initial curvature fit.  Users shouldn't really need to modify this,
# we just want it positive and small for improved conditioning of the associated least squares problem.
INITIAL_CURVATURE_PENALTY = 1e-10

# helper function to apply matrix multiplication over a batch-dim
batch_matmul = lambda a, b: np.einsum("bij, bjk -> bik", a, b)


def is_symmetric_positive_definite(A: torch.Tensor) -> bool:
    if not torch.allclose(A, A.transpose(-2, -1), atol=1e-6):
        return False
    try:
        _ = torch.linalg.cholesky(A)
        return True
    except RuntimeError:
        return False
    
    
def principal_value(angle: np.ndarray, min_: float = -np.pi) -> np.ndarray:
    """
    Wrap heading angle into [min_, min_ + 2pi)
    """
    return (angle - min_) % (2 * np.pi) + min_


def torch_principal_value(angle: torch.Tensor) -> torch.Tensor:
    """
    Map a angle in range [-π, π] using arctan2(sin, cos)
    :param angle: Tensor of any shape, angle in radians
    :return: Tensor of same shape with angles normalized to [-π, π]
    """
    return torch.atan2(torch.sin(angle), torch.cos(angle))


def _generate_profile_from_initial_condition_and_derivatives(
    initial_condition: torch.Tensor,     # shape (B, N)
    derivatives: torch.Tensor,           # shape (B, N, T)
    discretization_time: float,
) -> torch.Tensor:                       # shape (B, N, T+1)
    """
    Integrates derivatives over time to generate a trajectory profile.

    :param initial_condition: Tensor of shape (B, N), the starting values.
    :param derivatives: Tensor of shape (B, N, T), the derivative values at each step.
    :param discretization_time: Time step delta for integration.
    :return: Tensor of shape (B, N, T+1), the integrated profile over time.
    """
    assert discretization_time > 0.0, "discretization_time must be positive"
    
    cumsum = torch.cumsum(derivatives * discretization_time, dim=-1)  # (B, N, T)
    
    # Pad a zero at t=0 to match shape (B, N, T+1)
    zero_pad = torch.zeros_like(cumsum[..., :1])  # (B, N, 1)
    padded_cumsum = torch.cat([zero_pad, cumsum], dim=-1)  # (B, N, T+1)

    profile = initial_condition.unsqueeze(-1) + padded_cumsum  # (B, N, T+1)
    return profile


def _get_xy_heading_displacements_from_poses(
    poses: torch.Tensor  # shape: (B, N, T, 3)
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns position and heading displacements given a pose trajectory.

    :param poses: <torch.Tensor: (B, N, T, 3)> A trajectory of poses (x, y, heading).
    :return: Tuple of:
        - xy_displacements: shape (B, N, T-1, 2)
        - heading_displacements: shape (B, N, T-1)
    """
    assert poses.ndim == 4 and poses.shape[-1] == 3, "Expected input shape (B, N, T, 3)"
    B, N, T, _ = poses.shape
    assert T > 1, "Need at least 2 poses to compute displacements"

    pose_deltas = poses[:, :, 1:, :] - poses[:, :, :-1, :]      # (B, N, T-1, 3)
    xy_displacements = pose_deltas[..., :2]                     # (B, N, T-1, 2)
    heading_displacements = torch_principal_value(pose_deltas[..., 2])  # (B, N, T-1)

    return xy_displacements, heading_displacements


def _make_banded_difference_matrix(
        number_rows: int, 
        device=torch.device("cuda"), 
        dtype=torch.float32
) -> torch.Tensor:
    """
    Returns a banded difference matrix with specified number_rows.
    When applied to a vector [x_1, ..., x_N], it returns [x_2 - x_1, ..., x_N - x_{N-1}].

    :param number_rows: The row dimension of the banded difference matrix (e.g. N-1 in the example above).
    :param device: The device to place the tensor on (e.g. 'cuda' or 'cpu').
    :param dtype: The data type of the tensor (default: torch.float32).
    :return: A banded difference matrix with shape (number_rows, number_rows+1).
    """
    banded_matrix = torch.zeros((number_rows, number_rows + 1), device=device, dtype=dtype)
    eye = torch.eye(number_rows, device=device, dtype=dtype)
    banded_matrix[:, 1:] = eye
    banded_matrix[:, :-1] = -eye
    return banded_matrix  # (number_rows, number_rows + 1)


def _fit_initial_velocity_and_acceleration_profile(
    xy_displacements: torch.Tensor,      # (B, N, 80, 2)
    heading_profile: torch.Tensor,       # (B, N, 80)
    discretization_time: float,          # 0.1
    jerk_penalty: float,                 # 0.0001
) -> Tuple[torch.Tensor, torch.Tensor]:
    assert discretization_time > 0.0, "Discretization time must be positive."
    assert jerk_penalty > 0, "Should have a positive jerk_penalty."
    assert xy_displacements.ndim == 4 and xy_displacements.shape[-1] == 2
    assert heading_profile.shape[1] == xy_displacements.shape[1]

    device = xy_displacements.device
    dtype = xy_displacements.dtype

    B, N, M, _ = xy_displacements.shape
    
    y = xy_displacements.reshape(B, N, -1)  # (B, N, 160)
    
    A_column = torch.zeros((B, N, 2 * M), device=device, dtype=dtype)  # (B, N, 160)
    A_column[:, :, 0::2] = torch.cos(heading_profile)
    A_column[:, :, 1::2] = torch.sin(heading_profile)
    
    A = A_column.unsqueeze(-1) * discretization_time**2  # (B, N, 160, 1)
    A = A.repeat(1, 1, 1, M)  # (B, N, 160, 80)
    A[..., 0] = A_column * discretization_time  # initial condition term
    
    upper_mask = torch.triu(torch.ones((M, M), dtype=torch.bool, device=device), diagonal=1)
    upper_mask = upper_mask.repeat_interleave(2, dim=0)  # (160, 80)
    A = torch.where(upper_mask.unsqueeze(0).unsqueeze(0), torch.zeros_like(A), A)
    
    banded_matrix = _make_banded_difference_matrix(number_rows=M - 2, device=device, dtype=dtype)  # (78, 79)
    R = torch.cat([torch.zeros((banded_matrix.shape[0], 1), device=device, dtype=dtype), banded_matrix], dim=1)
    R = R.expand(B, N, -1, -1)  # (B, N, 78, 80)

    A_T = A.transpose(2, 3)  # (B, N, 80, 160)
    R_T = R.transpose(2, 3)  # (B, N, 80, 78)

    AtA = torch.matmul(A_T, A)  # (B, N, 80, 80)
    RtR = torch.matmul(R_T, R)  # (B, N, 80, 80)

    reg_matrix = AtA + jerk_penalty * RtR     
    if is_symmetric_positive_definite(reg_matrix):
        rhs = torch.matmul(A_T, y.unsqueeze(-1))                  # (B, N, M, 1)
        L = torch.linalg.cholesky(reg_matrix)                     # (B, N, M, M)
        x = torch.cholesky_solve(rhs, L).squeeze(-1)              # (B, N, M)
    else:
        warnings.warn("reg_matrix is not SPD, fallback to SVD-based pinv (slower).", category=UserWarning)
        print(f"> WARNING: reg_matrix is not SPD, fallback to SVD-based pinv (slower).")
        reg_inv = torch.linalg.pinv(AtA + jerk_penalty * RtR)    
        intermediate = torch.matmul(reg_inv, A_T)  # (B, N, 80, 160)
        x = torch.matmul(intermediate, y.unsqueeze(-1)).squeeze(-1)  # (B, N, 80)

    initial_velocity = x[..., 0]       # (B, N)
    acceleration_profile = x[..., 1:]  # (B, N, 79)
    
    return initial_velocity, acceleration_profile


def _fit_initial_curvature_and_curvature_rate_profile(
    heading_displacements: torch.Tensor,     # (B, N, T)
    velocity_profile: torch.Tensor,          # (B, N, T)
    discretization_time: float,              # scalar
    curvature_rate_penalty: float,           # scalar
    initial_curvature_penalty: float = 1e-10 # scalar
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Vectorized batch version: supports shape (B, N, T) with torch.Tensor input
    """
    assert discretization_time > 0.0
    assert curvature_rate_penalty > 0.0
    assert initial_curvature_penalty > 0.0

    device = heading_displacements.device
    dtype = heading_displacements.dtype

    B, N, T = heading_displacements.shape
    y = heading_displacements.reshape(B * N, T)       # (B*N, T)
    v = velocity_profile.reshape(B * N, T)            # (B*N, T)

    # Lower-triangular matrix
    tril = torch.tril(torch.ones(T, T, device=device, dtype=dtype))  # 下三角矩阵 (T, T)
    A = tril.unsqueeze(0).repeat(B * N, 1, 1)            # (B*N, T, T)
    A[:, :, 0] = v * discretization_time  # (B*N, T)
    v_scaled = (v * discretization_time ** 2)[:, None, 1:]  # (B*N, 1, T-1)
    A[:, 1:, 1:] *= v_scaled.transpose(1, 2)                # (B*N, T-1, T-1)
    
    A_T = A.transpose(1, 2)  # (B*N, T, T)

    # Regularization Q
    Q = torch.eye(T, dtype=dtype, device=device) * curvature_rate_penalty
    Q[0, 0] = initial_curvature_penalty
    Q = Q.unsqueeze(0).expand(B * N, T, T)  # (B*N, T, T)

    # Solve: x = pinv(A^T A + Q) A^T y
    AtA = torch.bmm(A_T, A) + Q                          # (B*N, T, T)
    AtY = torch.bmm(A_T, y.unsqueeze(2)).squeeze(2)      # (B*N, T)

    if is_symmetric_positive_definite(AtA):
        L = torch.linalg.cholesky(AtA)                      # (B*N, T, T)
        x = torch.cholesky_solve(AtY.unsqueeze(-1), L).squeeze(-1)  # (B*N, T)
    else:
        warnings.warn("reg_matrix is not SPD, fallback to SVD-based pinv (slower).", category=UserWarning)
        x = torch.linalg.pinv(AtA).bmm(AtY.unsqueeze(2)).squeeze(2)  # (B*N, T)
        
    initial_curvature = x[:, 0].reshape(B, N)              # (B, N)
    curvature_rate_profile = x[:, 1:].reshape(B, N, T - 1) # (B, N, T-1)
    
    return initial_curvature, curvature_rate_profile


def get_velocity_curvature_profiles_with_derivatives_from_poses(
    discretization_time: float,
    poses: torch.Tensor,  # (B, N, 81, 3)
    jerk_penalty: float,
    curvature_rate_penalty: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Estimate velocity, acceleration, curvature, and curvature rate profiles using torch.Tensor.
    """
    device = poses.device
    dtype = poses.dtype

    xy_displacements, heading_displacements = _get_xy_heading_displacements_from_poses(poses)

    initial_velocity, acceleration_profile = _fit_initial_velocity_and_acceleration_profile(
        xy_displacements=xy_displacements,
        heading_profile=poses[:, :, :-1, 2],
        discretization_time=discretization_time,
        jerk_penalty=jerk_penalty,
    )

    velocity_profile = _generate_profile_from_initial_condition_and_derivatives(
        initial_condition=initial_velocity,
        derivatives=acceleration_profile,
        discretization_time=discretization_time,
    )

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

    return (
        velocity_profile,
        acceleration_profile,
        curvature_profile,
        curvature_rate_profile,
    )