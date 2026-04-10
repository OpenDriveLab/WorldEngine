import numpy as np
from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.actor_state.state_representation import TimeDuration, TimePoint
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from nuplan.planning.simulation.simulation_time_controller.simulation_iteration import (
    SimulationIteration,
)

import torch
from typing import Any, Dict, List, Set, Tuple, Type, Union
from worldengine.components.agents.policy.pdm_planner.utils.pdm_array_representation import (
    ego_state_to_state_array,
)
from worldengine.components.agents.policy.pdm_planner.simulation.torch_lqr import TorchLQRTracker
from worldengine.components.agents.policy.pdm_planner.simulation.torch_kinematic_bicycle import TorchKinematicBicycleModel


class TorchSimulator:
    def __init__(
        self, 
        proposal_sampling: TrajectorySampling,
        dtype: torch.dtype = torch.float64,
        device: torch.device = torch.device("cuda")
    ) -> None:
        # time parameters
        self._proposal_sampling = proposal_sampling
        self._dt = proposal_sampling.interval_length
        self._num_poses = proposal_sampling.num_poses

        # simulation objects
        self._motion_model = TorchKinematicBicycleModel()
        self._tracker = TorchLQRTracker(dtype=dtype, device=device)
        
        self.dtype = dtype
        self.device = device

    def simulate_proposals(
        self, 
        states: torch.Tensor, 
        initial_ego_state: EgoState, 
        batch_sim: bool = False
    ) -> torch.Tensor:
        """
        Simulate all proposals over batch-dim
        Args:
            states: proposal states as tensor (B, N, T, 3)
            initial_ego_state: ego-vehicle state at current iteration
            batch_sim: whether to use batch simulation
        Returns:
            simulated_states: simulated proposal states as tensor
        """
        # set parameters of motion model and tracker
        self._motion_model._vehicle = initial_ego_state.car_footprint.vehicle_parameters
        self._tracker._discretization_time = self._dt

        # process input states
        proposal_states = states[:, :, :self._num_poses + 1]
        if batch_sim:
            self._tracker.update(proposal_states[..., :3])
        else:
            self._tracker.update(proposal_states)

        # initialize simulated states
        if batch_sim:
            simulated_states = torch.zeros(
                (proposal_states.shape[0], proposal_states.shape[1], proposal_states.shape[2], 11),
                dtype=self.dtype,
                device=self.device
            )
        else:
            simulated_states = torch.zeros(
                proposal_states.shape,
                dtype=self.dtype,
                device=self.device
            )

        # set initial state
        simulated_states[:, :, 0] = torch.tensor(
            ego_state_to_state_array(initial_ego_state),
            dtype=self.dtype,
            device=self.device
        )

        # set time parameters
        current_time_point = initial_ego_state.time_point
        delta_time_point = TimeDuration.from_s(self._dt)

        current_iteration = SimulationIteration(current_time_point, 0)
        next_iteration = SimulationIteration(current_time_point + delta_time_point, 1)

        # simulation loop
        for time_idx in range(1, self._num_poses + 1):
            sampling_time = next_iteration.time_point - current_iteration.time_point

            # get control commands
            command_states = self._tracker.track_trajectory(
                current_iteration,
                next_iteration,
                simulated_states[:, :, time_idx - 1],
            )

            # state propagation
            simulated_states[:, :, time_idx] = self._motion_model.propagate_state(
                states=simulated_states[:, :, time_idx - 1],
                command_states=command_states,
                sampling_time=sampling_time,
            )

            # update iteration
            current_iteration = next_iteration
            next_iteration = SimulationIteration(
                current_iteration.time_point + delta_time_point, 
                1 + time_idx
            )

        return simulated_states
