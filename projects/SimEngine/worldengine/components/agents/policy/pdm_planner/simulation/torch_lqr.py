#   Heavily borrowed from:
#   https://github.com/autonomousvision/tuplan_garage (Apache License 2.0)
# & https://github.com/motional/nuplan-devkit (Apache License 2.0)

from enum import IntEnum
from typing import Any, Dict, List, Set, Tuple, Type, Union, Optional
import torch
import numpy as np
import numpy.typing as npt
from nuplan.common.actor_state.vehicle_parameters import (
    VehicleParameters,
    get_pacifica_parameters,
)
# from nuplan.common.geometry.compute import principal_value
from nuplan.planning.simulation.simulation_time_controller.simulation_iteration import (
    SimulationIteration,
)
import time
from .batch_kinematic_bicycle import DynamicStateIndex, StateIndex
from .torch_lqr_utils import (
    _generate_profile_from_initial_condition_and_derivatives,
    get_velocity_curvature_profiles_with_derivatives_from_poses,principal_value, torch_principal_value
)


class LateralStateIndex(IntEnum):
    """
    Index mapping for the lateral dynamics state vector.
    """

    LATERAL_ERROR = 0  # [m] The lateral error with respect to the planner centerline at the vehicle's rear axle center.
    HEADING_ERROR = 1  # [rad] The heading error "".
    STEERING_ANGLE = (
        2  # [rad] The wheel angle relative to the longitudinal axis of the vehicle.
    )


class TorchLQRTracker:
    """
    Implements an LQR tracker for a kinematic bicycle model.

    Tracker operates on a batch of proposals. Implementation directly based on the nuplan-devkit
    Link: https://github.com/motional/nuplan-devkit

    We decouple into two subsystems, longitudinal and lateral, with small angle approximations for linearization.
    We then solve two sequential LQR subproblems to find acceleration and steering rate inputs.

    Longitudinal Subsystem:
        States: [velocity]
        Inputs: [acceleration]
        Dynamics (continuous time):
            velocity_dot = acceleration

    Lateral Subsystem (After Linearization/Small Angle Approximation):
        States: [lateral_error, heading_error, steering_angle]
        Inputs: [steering_rate]
        Parameters: [velocity, curvature]
        Dynamics (continuous time):
            lateral_error_dot  = velocity * heading_error
            heading_error_dot  = velocity * (steering_angle / wheelbase_length - curvature)
            steering_angle_dot = steering_rate

    The continuous time dynamics are discretized using Euler integration and zero-order-hold on the input.
    In case of a stopping reference, we use a simplified stopping P controller instead of LQR.

    The final control inputs passed on to the motion model are:
        - acceleration
        - steering_rate
    """

    def __init__(
        self,
        q_longitudinal: Union[list, torch.Tensor] = [10.0],
        r_longitudinal: Union[list, torch.Tensor] = [1.0],
        q_lateral: Union[list, torch.Tensor] = [1.0, 10.0, 0.0],
        r_lateral: Union[list, torch.Tensor] = [1.0],
        discretization_time: float = 0.1,
        tracking_horizon: int = 10,
        jerk_penalty: float = 1e-4,
        curvature_rate_penalty: float = 1e-2,
        stopping_proportional_gain: float = 0.5,
        stopping_velocity: float = 0.2,
        vehicle: VehicleParameters = get_pacifica_parameters(),
        estop: bool = False,
        soft_brake: bool = False,
        device: torch.device = torch.device("cuda"),
        dtype: torch.dtype = torch.float64,
    ):
        """
        Constructor for LQR controller using torch.Tensor

        :param device: Torch device for all tensors
        :param dtype: Torch dtype for all tensors
        """
        # Longitudinal LQR Parameters
        assert len(q_longitudinal) == 1, "q_longitudinal should have 1 element (velocity)."
        assert len(r_longitudinal) == 1, "r_longitudinal should have 1 element (acceleration)."
        self._q_longitudinal: float = q_longitudinal[0]
        self._r_longitudinal: float = r_longitudinal[0]

        # Lateral LQR Parameters
        assert len(q_lateral) == 3, "q_lateral should have 3 elements (lateral_error, heading_error, steering_angle)."
        assert len(r_lateral) == 1, "r_lateral should have 1 element (steering_rate)."

        self._q_lateral: torch.Tensor = torch.diag(torch.tensor(q_lateral, dtype=dtype, device=device))
        self._r_lateral: torch.Tensor = torch.diag(torch.tensor(r_lateral, dtype=dtype, device=device))

        # Common LQR Parameters
        assert discretization_time > 0.0, "The discretization_time should be positive."
        assert tracking_horizon > 1, "tracking_horizon must be greater than 1."

        self._discretization_time = discretization_time
        self._tracking_horizon = tracking_horizon
        self._wheel_base = vehicle.wheel_base

        # Velocity/Curvature Estimation Parameters
        assert jerk_penalty > 0.0, "jerk_penalty must be positive."
        assert curvature_rate_penalty > 0.0, "curvature_rate_penalty must be positive."

        self._jerk_penalty = jerk_penalty
        self._curvature_rate_penalty = curvature_rate_penalty

        # Stopping Controller Parameters
        assert stopping_proportional_gain > 0, "stopping_proportional_gain must be > 0."
        assert stopping_velocity > 0, "stopping_velocity must be > 0."

        self._stopping_proportional_gain = stopping_proportional_gain
        self._stopping_velocity = stopping_velocity

        # Lazy-loaded internal states
        self._proposal_states: Optional[torch.Tensor] = None
        self._initialized: bool = False
        self._estop = estop
        self._soft_brake = soft_brake

        # For use in downstream methods
        self.device = device
        self.dtype = dtype

    def update(self, proposal_states: torch.Tensor) -> None:
        """
        Loads proposal state tensor and resets velocity and curvature profiles.

        :param proposal_states: Tensor representation of proposals, shape (B, N, 81, 3)
        """
        assert isinstance(proposal_states, torch.Tensor), "proposal_states must be a torch.Tensor"
        
        self._proposal_states: torch.Tensor = proposal_states.to(dtype=self.dtype, device=self.device)
        self._velocity_profile, self._curvature_profile = None, None
        self._initialized = True

    def track_trajectory(
        self,
        current_iteration: SimulationIteration, 
        next_iteration: SimulationIteration, 
        initial_states: torch.Tensor,  # (B, N, 11)
    ) -> torch.Tensor:
        """
        Calculates the command values given the proposals to track.
        :param current_iteration: current simulation iteration.
        :param next_iteration: desired next simulation iteration.
        :param initial_states: array representation of current ego states.
        :return: command values for motion model.
        """
        assert (
            self._initialized
        ), "BatchLQRTracker: Run update first to load proposal states!"

        B, N, _ = initial_states.shape
        (
            initial_velocity,               # (B, N)
            initial_lateral_state_vector,   # (B, N, 3)
        ) = self._compute_initial_velocity_and_lateral_state(
            current_iteration, initial_states
        )

        (
            reference_velocities,   # (B, N)
            curvature_profiles,     # (B, N, 10)
        ) = self._compute_reference_velocity_and_curvature_profile(
            current_iteration
        )
        
        if self._estop:
            reference_velocities = torch.zeros_like(reference_velocities)

        # create output arrays
        accel_cmds = torch.zeros((B, N), dtype=self.dtype, device=self.device)         # (B, N)
        steering_rate_cmds = torch.zeros((B, N), dtype=self.dtype, device=self.device) # (B, N)

        # 1. Stopping Controller
        should_stop_mask = torch.logical_and(
            reference_velocities <= self._stopping_velocity,
            initial_velocity <= self._stopping_velocity,
        )  # shape: (B, N)
        
        # vectorized calculation of acceleration commands
        stopping_accel_cmds = -self._stopping_proportional_gain * (
            initial_velocity - reference_velocities
        )  # shape: (B, N)
        
        accel_cmds = torch.where(should_stop_mask, stopping_accel_cmds, accel_cmds)  # (B, N)
        steering_rate_cmds = torch.where(should_stop_mask, torch.tensor(0.0, device=self.device, dtype=self.dtype), steering_rate_cmds)  # (B, N)
        
        # 2. Regular Controller
        accel_cmds_lqr = self._longitudinal_lqr_controller(initial_velocity, reference_velocities)  # shape: (B, N)
        accel_cmds = torch.where(should_stop_mask, stopping_accel_cmds, accel_cmds_lqr)  # shape: (B, N)

        accel_profile = accel_cmds.unsqueeze(-1).repeat(1, 1, self._tracking_horizon)  # (B, N, T)
        
        # call the profile generation function that supports batch input (B, N, T)
        velocity_profiles = _generate_profile_from_initial_condition_and_derivatives(
            initial_condition=initial_velocity,  # (B, N)
            derivatives=accel_profile,           # (B, N, T)
            discretization_time=self._discretization_time
        )
        # take the first tracking_horizon steps (usually T)
        velocity_profiles = velocity_profiles[..., : self._tracking_horizon]  # shape: (B, N, T)

        # lateral_lqr
        mask = ~should_stop_mask  # shape: (B, N)
        # extract data from masked positions
        selected_lateral_state = initial_lateral_state_vector[mask]    # (B*N', 3)
        selected_velocity_profile = velocity_profiles[mask]            # (B*N', T)
        selected_curvature_profile = curvature_profiles[mask]          # (B*N', T)
        
        # execute LQR controller, get steering_rate (B*N',)
        steering_rate_selected = self._lateral_lqr_controller(
            selected_lateral_state,
            selected_velocity_profile,
            selected_curvature_profile
        )  # shape: (B*N',)
        
        steering_rate_cmds[mask] = steering_rate_selected
        
        if self._estop:
            print("warning: self._estop not check!")
            if self._soft_brake or should_stop_mask[0].any():
                accel_cmds = torch.clamp(accel_cmds, min=-4.0, dtype=self.dtype, device=self.device)
            else:
                accel_cmds = torch.full_like(accel_cmds, -4.5, dtype=self.dtype, device=self.device)

        command_states = torch.zeros(
            (B, N, len(DynamicStateIndex)), dtype=self.dtype, device=self.device
        )
        
        command_states[:, :, DynamicStateIndex.ACCELERATION_X] = accel_cmds
        command_states[:, :, DynamicStateIndex.STEERING_RATE] = steering_rate_cmds

        return command_states

    def _compute_initial_velocity_and_lateral_state(
        self,
        current_iteration: SimulationIteration,
        initial_values: torch.Tensor,  # (B, N, 11)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        This method projects the initial tracking error into vehicle/Frenet frame.  It also extracts initial velocity.
        :param current_iteration: Used to get the current time.
        :param initial_state: The current state for ego.
        :param trajectory: The reference trajectory we are tracking.
        :return: Initial velocity [m/s] and initial lateral state.
        """
        # Get initial trajectory state.
        initial_trajectory_values = self._proposal_states[:, :, current_iteration.index]  
        # self._proposal_states(B, N, 81, 3) initial_trajectory_values(B, N, 3)
        # Determine initial error state.
        x_errors = (
            initial_values[:, :, StateIndex.X] - initial_trajectory_values[:, :, StateIndex.X]
        )  # (B, N)
        y_errors = (
            initial_values[:, :, StateIndex.Y] - initial_trajectory_values[:, :, StateIndex.Y]
        )  # (B, N)
        heading_references = initial_trajectory_values[:, :, StateIndex.HEADING]  # (B, N)

        lateral_errors = -x_errors * torch.sin(heading_references) + y_errors * torch.cos(
            heading_references
        )  # (B, N)
        heading_errors = torch_principal_value(
            initial_values[:, :, StateIndex.HEADING] - heading_references
        )  # (B, N)

        # Return initial velocity and lateral state vector.
        initial_velocities = initial_values[:, :, StateIndex.VELOCITY_X]  # (B, N)

        initial_lateral_state_vector = torch.stack(
            [
                lateral_errors,
                heading_errors,
                initial_values[:, :, StateIndex.STEERING_ANGLE],
            ],
            dim=-1,
        )  # (B, N, 3)
        # initial_velocities(B, N), initial_lateral_state_vector(B, N, 3)
        return initial_velocities, initial_lateral_state_vector  

    def _compute_reference_velocity_and_curvature_profile(
        self,
        current_iteration: SimulationIteration,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        This method computes reference velocity and curvature profile based on the reference trajectory.
        We use a lookahead time equal to self._tracking_horizon * self._discretization_time.
        :param current_iteration: Used to get the current time.
        :param trajectory: The reference trajectory we are tracking.
        :return: The reference velocity [m/s] and curvature profile [rad] to track.
        """

        poses = self._proposal_states[..., StateIndex.STATE_SE2]  # (B, N, 81, 3)

        if self._velocity_profile is None or self._curvature_profile is None:
            (
                self._velocity_profile,     # (B, N, 80)
                acceleration_profile,       # (B, N, 79)
                self._curvature_profile,    # (B, N, 80)
                curvature_rate_profile,     # (B, N, 79)
            ) = get_velocity_curvature_profiles_with_derivatives_from_poses(
                discretization_time=self._discretization_time,
                poses=poses,
                jerk_penalty=self._jerk_penalty,
                curvature_rate_penalty=self._curvature_rate_penalty,
            )

        B, N, num_poses = self._velocity_profile.shape
        reference_idx = min(
            current_iteration.index + self._tracking_horizon, num_poses - 1
        )  # reference_idx=10
        reference_velocities = self._velocity_profile[:, :, reference_idx]  # (B, N)

        reference_curvature_profiles = torch.zeros(
            (B, N, self._tracking_horizon), dtype=self.dtype, device=self.device
        )

        reference_length = reference_idx - current_iteration.index
        reference_curvature_profiles[:, :, 0:reference_length] = self._curvature_profile[
            :, :, current_iteration.index : reference_idx
        ]

        if reference_length < self._tracking_horizon:
            reference_curvature_profiles[
                :, :, reference_length:
            ] = self._curvature_profile[:, :, reference_idx, None]
        # reference_curvature_profiles (B, N, self._tracking_horizon)
        
        return reference_velocities, reference_curvature_profiles

    def _stopping_controller(
        self,
        initial_velocities: torch.Tensor, 
        reference_velocities: torch.Tensor, 
    ) -> Tuple[torch.Tensor, float]:
        """
        Apply proportional controller when at near-stop conditions.
        :param initial_velocity: [m/s] The current velocity of ego.
        :param reference_velocity: [m/s] The reference velocity to track.
        :return: Acceleration [m/s^2] and zero steering_rate [rad/s] command.
        """
        accel = -self._stopping_proportional_gain * (
            initial_velocities - reference_velocities
        )
        return accel, 0.0

    def _longitudinal_lqr_controller(
        self,
        initial_velocities: torch.Tensor,       # (B, N)
        reference_velocities: torch.Tensor,     # (B, N)
    ) -> torch.Tensor:                          # (B, N)
        """
        This longitudinal controller determines an acceleration input to minimize velocity error at a lookahead time.
        :param initial_velocities: [m/s] The current velocity of ego for each agent.
        :param reference_velocities: [m/s] The reference velocity to track at a lookahead time.
        :return: Acceleration [m/s^2] command based on LQR, shape (B, N).
        """
        B_scale = self._tracking_horizon * self._discretization_time
        A = torch.ones_like(initial_velocities, dtype=self.dtype, device=self.device)      # (B, N)
        B_mat = torch.full_like(initial_velocities, B_scale, dtype=self.dtype, device=self.device)  # (B, N)
        g = torch.zeros_like(initial_velocities, dtype=self.dtype, device=self.device)     # (B, N)
    
        accel_cmds = self._solve_one_step_longitudinal_lqr(
            initial_state=initial_velocities,         # (B, N)
            reference_state=reference_velocities,     # (B, N)
            A=A, 
            B=B_mat, 
            g=g,                         # All shape (B, N)
        )

        return accel_cmds  # (B, N)

    def _lateral_lqr_controller(
        self,
        initial_lateral_state_vector: torch.Tensor,
        velocity_profile: torch.Tensor,
        curvature_profile: torch.Tensor,
    ) -> float:
        """
        This lateral controller determines a steering_rate input to minimize lateral errors at a lookahead time.
        It requires a velocity sequence as a parameter to ensure linear time-varying lateral dynamics.
        :param initial_lateral_state_vector: The current lateral state of ego.
        :param velocity_profile: [m/s] The velocity over the entire self._tracking_horizon-step lookahead.
        :param curvature_profile: [rad] The curvature over the entire self._tracking_horizon-step lookahead..
        :return: Steering rate [rad/s] command based on LQR.
        """
        assert velocity_profile.shape[-1] == self._tracking_horizon, (
            f"The linearization velocity sequence should have length {self._tracking_horizon} "
            f"but is {len(velocity_profile)}."
        )
        assert curvature_profile.shape[-1] == self._tracking_horizon, (
            f"The linearization curvature sequence should have length {self._tracking_horizon} "
            f"but is {len(curvature_profile)}."
        )

        batch_dim = velocity_profile.shape[0]

        # Set up the lateral LQR problem using the constituent linear time-varying (affine) system dynamics.
        # Ultimately, we'll end up with the following problem structure where N = self._tracking_horizon:
        # lateral_error_N = A @ lateral_error_0 + B @ steering_rate + g
        n_lateral_states = len(LateralStateIndex)

        I = torch.eye(n_lateral_states, dtype=self.dtype, device=self.device)
        
        in_matrix = torch.zeros((n_lateral_states, 1), dtype=self.dtype, device=self.device)
        in_matrix[LateralStateIndex.STEERING_ANGLE] = self._discretization_time

        states_matrix_at_step = I.repeat(self._tracking_horizon, batch_dim, 1, 1)  # (T, B, 3, 3)
        
        states_matrix_at_step[
            :, :, LateralStateIndex.LATERAL_ERROR, LateralStateIndex.HEADING_ERROR
        ] = (velocity_profile.T * self._discretization_time)

        states_matrix_at_step[
            :, :, LateralStateIndex.HEADING_ERROR, LateralStateIndex.STEERING_ANGLE
        ] = (velocity_profile.T * self._discretization_time / self._wheel_base)

        affine_terms = torch.zeros((self._tracking_horizon, batch_dim, n_lateral_states), dtype=self.dtype, device=self.device)
        
        affine_terms[:, :, LateralStateIndex.HEADING_ERROR] = (
            -velocity_profile.T * curvature_profile.T * self._discretization_time
        )

        A = I.repeat(batch_dim, 1, 1)  # (B, 3, 3)
        B = torch.zeros((batch_dim, n_lateral_states, 1), dtype=self.dtype, device=self.device)
        g = torch.zeros((batch_dim, n_lateral_states), dtype=self.dtype, device=self.device)
        
        for index_step in range(self._tracking_horizon):
            A = torch.bmm(states_matrix_at_step[index_step], A)
            B = torch.bmm(states_matrix_at_step[index_step], B) + in_matrix
            g = torch.bmm(states_matrix_at_step[index_step], g.unsqueeze(-1)).squeeze(-1) + affine_terms[index_step]

        steering_rate_cmd = self._solve_one_step_lateral_lqr(
            initial_state=initial_lateral_state_vector,  
            A=A,
            B=B,
            g=g,
        )

        return torch.squeeze(steering_rate_cmd, dim=-1)

    def _solve_one_step_longitudinal_lqr(
        self,
        initial_state: torch.Tensor,    # (B, N)
        reference_state: torch.Tensor,   # (B, N)
        A: torch.Tensor,                 # (B, N)
        B: torch.Tensor,                 # (B, N)
        g: torch.Tensor,                 # (B, N)
    ) -> torch.Tensor:
        """
        This function uses LQR to find an optimal input to minimize tracking error in one step of dynamics.
        The dynamics are next_state = A @ initial_state + B @ input + g and our target is the reference_state.
        :param initial_state: The current state.
        :param reference_state: The desired state in 1 step (according to A,B,g dynamics).
        :param A: The state dynamics matrix.
        :param B: The input dynamics matrix.
        :param g: The offset/affine dynamics term.
        :return: LQR optimal input for the 1-step longitudinal problem.
        """
        A = A.to(dtype=self.dtype, device=self.device)
        B = B.to(dtype=self.dtype, device=self.device)
        g = g.to(dtype=self.dtype, device=self.device)
        initial_state = initial_state.to(dtype=self.dtype, device=self.device)
        reference_state = reference_state.to(dtype=self.dtype, device=self.device)
        
        state_error_zero_input = A * initial_state + g - reference_state
        inverse = -1 / (B * self._q_longitudinal * B + self._r_longitudinal)
        lqr_input = inverse * B * self._q_longitudinal * state_error_zero_input

        return lqr_input

    def _solve_one_step_lateral_lqr(
        self,
        initial_state: torch.Tensor,  # (B, 3)
        A: torch.Tensor,               # (B, 3, 3)
        B:  torch.Tensor,              # (B, 3, 1)
        g:  torch.Tensor,              # (B, 3)
    ) -> torch.Tensor:
        """
        This function uses LQR to find an optimal input to minimize tracking error in one step of dynamics.
        The dynamics are next_state = A @ initial_state + B @ input + g and our target is the reference_state.
        :param initial_state: The current state.
        :param A: The state dynamics matrix.
        :param B: The input dynamics matrix.
        :param g: The offset/affine dynamics term.
        :return: LQR optimal input for the 1-step lateral problem.
        """         
        Q = self._q_lateral.clone().detach().to(device=self.device, dtype=self.dtype)  # (3, 3)
        R = self._r_lateral.clone().detach().to(device=self.device, dtype=self.dtype)  # (1, 1)
        
        angle_diff_indices = [
            LateralStateIndex.HEADING_ERROR.value,
            LateralStateIndex.STEERING_ANGLE.value,
        ]
        BT = B.transpose(1, 2)

        state_error_zero_input = torch.einsum("bij,bj->bi", A, initial_state) + g  # (B, 3)
        
        angle = state_error_zero_input[..., angle_diff_indices]
        state_error_zero_input[..., angle_diff_indices] = torch.atan2(
            torch.sin(angle), torch.cos(angle)
        )

        BT_x_Q = torch.matmul(BT, Q)  # (B, 1, 3)
        denom = torch.matmul(BT_x_Q, B).squeeze(-1) + R  # (B, 1)
        Inv = -1.0 / denom  # (B, 1)
        # BT * Q * error
        Tail = torch.einsum("bij,bj->bi", BT_x_Q, state_error_zero_input)  # (B, 1)
        lqr_input = Inv * Tail  # (B, 1)
        
        return lqr_input
