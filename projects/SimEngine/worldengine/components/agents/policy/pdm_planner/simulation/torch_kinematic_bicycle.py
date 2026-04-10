#   Heavily borrowed from:
#   https://github.com/autonomousvision/tuplan_garage (Apache License 2.0)
# & https://github.com/motional/nuplan-devkit (Apache License 2.0)

import torch
import copy
import time
import numpy as np
import numpy.typing as npt
from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.actor_state.state_representation import TimePoint
from nuplan.common.actor_state.vehicle_parameters import (
    VehicleParameters,
    get_pacifica_parameters,
)
# from nuplan.common.geometry.compute import principal_value
from worldengine.components.agents.policy.pdm_planner.utils.pdm_enums import (
    DynamicStateIndex,
    StateIndex,
)
from worldengine.components.agents.policy.pdm_planner.simulation.torch_lqr_utils import torch_principal_value


def forward_integrate(
    init: torch.Tensor,
    delta: torch.Tensor,
    sampling_time: TimePoint,
) -> torch.Tensor:
    """
    Performs a simple Euler integration.
    :param init: Initial state tensor
    :param delta: The rate of change of the state tensor
    :param sampling_time: The time duration to propagate for
    :return: The result of integration (tensor)
    """
    return init + delta * sampling_time.time_s


class TorchKinematicBicycleModel:
    """
    A batch-wise operating class describing the kinematic motion model where the rear axle is the point of reference.
    """

    def __init__(
        self,
        vehicle: VehicleParameters = get_pacifica_parameters(),
        max_steering_angle: float = np.pi / 3,
        accel_time_constant: float = 0.2,
        steering_angle_time_constant: float = 0.05,
    ):
        """
        Construct BatchKinematicBicycleModel.
        :param vehicle: Vehicle parameters.
        :param max_steering_angle: [rad] Maximum absolute value steering angle allowed by model.
        :param accel_time_constant: low pass filter time constant for acceleration in s
        :param steering_angle_time_constant: low pass filter time constant for steering angle in s
        """
        self._vehicle = vehicle
        self._max_steering_angle = max_steering_angle
        self._accel_time_constant = accel_time_constant
        self._steering_angle_time_constant = steering_angle_time_constant
        
    def get_state_dot(self, states: torch.Tensor) -> torch.Tensor:
        """
        Calculates the changing rate of state tensor representation.
        :param states: Tensor describing the state of the ego-vehicle.  # shape: (B, N, 11)
        :return: Tensor of change rates across several state values      # shape: (B, N, 11)
        """
        state_dots = torch.zeros_like(states)  # (B, N, 11)

        longitudinal_speeds = states[:, :, StateIndex.VELOCITY_X]  # (B, N)

        state_dots[:, :, StateIndex.X] = longitudinal_speeds * torch.cos(
            states[:, :, StateIndex.HEADING]
        )
        state_dots[:, :, StateIndex.Y] = longitudinal_speeds * torch.sin(
            states[:, :, StateIndex.HEADING]
        )
        state_dots[:, :, StateIndex.HEADING] = (
            longitudinal_speeds
            * torch.tan(states[:, :, StateIndex.STEERING_ANGLE])
            / self._vehicle.wheel_base
        )

        state_dots[:, :, StateIndex.VELOCITY_2D] = states[:, :, StateIndex.ACCELERATION_2D]
        state_dots[:, :, StateIndex.ACCELERATION_2D] = 0.0

        state_dots[:, :, StateIndex.STEERING_ANGLE] = states[:, :, StateIndex.STEERING_RATE]

        return state_dots  # (B, N, 11)

    def _update_commands(
        self,
        states: torch.Tensor,          # (B, N, 11)
        command_states: torch.Tensor,  # (B, N, 2)
        sampling_time: TimePoint,
    ) -> torch.Tensor:                # returns (B, N, 11)
        """
        Apply first-order control delay (low-pass filter) on acceleration and steering for batched inputs.
        """

        dt_control = sampling_time.time_s  # float

        # 使用 clone 替代 deepcopy（更快且满足 tensor 语义）
        propagating_state = states.clone()  # (B, N, 11)

        # 取出当前状态量
        accel = states[:, :, StateIndex.ACCELERATION_X]             # (B, N)
        steering_angle = states[:, :, StateIndex.STEERING_ANGLE]    # (B, N)

        # 指令输入
        ideal_accel_x = command_states[:, :, DynamicStateIndex.ACCELERATION_X]  # (B, N)
        ideal_steering_angle = (
            dt_control * command_states[:, :, DynamicStateIndex.STEERING_RATE] + steering_angle
        )  # (B, N)

        # 低通滤波更新
        updated_accel_x = (
            dt_control / (dt_control + self._accel_time_constant)
            * (ideal_accel_x - accel) + accel
        )

        updated_steering_angle = (
            dt_control / (dt_control + self._steering_angle_time_constant)
            * (ideal_steering_angle - steering_angle) + steering_angle
        )

        updated_steering_rate = (updated_steering_angle - steering_angle) / dt_control

        # 更新状态张量
        propagating_state[:, :, StateIndex.ACCELERATION_X] = updated_accel_x
        propagating_state[:, :, StateIndex.ACCELERATION_Y] = 0.0
        propagating_state[:, :, StateIndex.STEERING_RATE] = updated_steering_rate

        return propagating_state  # (B, N, 11)

    def propagate_state(
        self,
        states: torch.Tensor,  # (N ,11)
        command_states: torch.Tensor,  # (N, 2)
        sampling_time: TimePoint,  # TimePoint(time_us=100000)
    ) -> torch.Tensor:
        """
        Propagates ego state array forward with motion model.
        :param states: state array representation of the ego-vehicle
        :param command_states: command array representation of controller
        :param sampling_time: time to propagate [s]
        :return: updated tate array representation of the ego-vehicle
        """

        assert len(states) == len(
            command_states
        ), "Batch size of states and command_states does not match!"

        propagating_state = self._update_commands(states, command_states, sampling_time)  # (B, N, 11)

        output_state = states.clone()  # (B, N, 11)
        
        # Compute state derivatives
        state_dot = self.get_state_dot(propagating_state)  # state_dot(B, N, 11)

        output_state[:, :, StateIndex.X] = forward_integrate(
            states[:, :, StateIndex.X], state_dot[:, :, StateIndex.X], sampling_time
        )
        output_state[:, :, StateIndex.Y] = forward_integrate(
            states[:, :, StateIndex.Y], state_dot[:, :, StateIndex.Y], sampling_time
        )

        output_state[:, :, StateIndex.HEADING] = torch_principal_value(
            forward_integrate(
                states[:, :, StateIndex.HEADING],
                state_dot[:, :, StateIndex.HEADING],
                sampling_time,
            )
        )

        output_state[:, :, StateIndex.VELOCITY_X] = forward_integrate(
            states[:, :, StateIndex.VELOCITY_X],
            state_dot[:, :, StateIndex.VELOCITY_X],
            sampling_time,
        )

        # Lateral velocity is always zero in kinematic bicycle model
        output_state[:, :, StateIndex.VELOCITY_Y] = 0.0

        # Integrate steering angle and clip to bounds
        steering_angle_next = forward_integrate(
            propagating_state[:, :, StateIndex.STEERING_ANGLE],
            state_dot[:, :, StateIndex.STEERING_ANGLE],
            sampling_time,
        )
        output_state[:, :, StateIndex.STEERING_ANGLE] = torch.clamp(
            steering_angle_next,
            min=-self._max_steering_angle,
            max=self._max_steering_angle,
        )

        output_state[:, :, StateIndex.ANGULAR_VELOCITY] = (
            output_state[:, :, StateIndex.VELOCITY_X]
            * torch.tan(output_state[:, :, StateIndex.STEERING_ANGLE])
            / self._vehicle.wheel_base
        )

        output_state[:, :, StateIndex.ACCELERATION_2D] = state_dot[
            :, :, StateIndex.VELOCITY_2D
        ]

        output_state[:, :, StateIndex.ANGULAR_ACCELERATION] = (
            output_state[:, :, StateIndex.ANGULAR_VELOCITY]
            - states[:, :, StateIndex.ANGULAR_VELOCITY]
        ) / sampling_time.time_s

        output_state[:, :, StateIndex.STEERING_RATE] = state_dot[
            :, :, StateIndex.STEERING_ANGLE
        ]

        return output_state
