import logging
import numpy as np
import torch
import time
import os

from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.actor_state.state_representation import StateSE2, TimePoint, StateVector2D
from nuplan.common.geometry.transform import translate_longitudinally
from worldengine.components.agents.vehicle_model.pacifica_vehicle import get_pacifica_parameters
from worldengine.components.agents.policy.base_policy import BasePolicy
from worldengine.scenario.scenarios.scenario_description import ScenarioDescription as SD
from worldengine.common.dataclasses import Trajectory

WE_root = os.environ.get("WORLDENGINE_ROOT")

class HumanPolicy(BasePolicy):

    def __init__(self, agent, random_seed=None, config=None):
        super(HumanPolicy, self).__init__(agent=agent, random_seed=random_seed, config=config)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._traj_info = self.agent.object_track
        self._valid_indicator = self._traj_info[SD.VALID] == 1
        self._traj_length = len(self._valid_indicator)
        self.raw_human_traj = np.load(
            os.path.join(
                WE_root,
                "data/alg_engine/test_8192_kmeans.npy"
            )
        )
        self.rear_axle_to_center = 1.461

    @property
    def is_current_step_valid(self):

        index = max(int(self.traj_step), 0)
        if index >= self._traj_length:
            return False
        
        return self._valid_indicator[self.traj_step]

    def act(self, *args, **kwargs):
        """
        Return a set of waypoints.
        """
        index = max(int(self.traj_step), 0)
        start_time = time.time()
        self._convert_trajectory()
        end_time = time.time()
        print(f"Conversion time: {end_time - start_time} seconds")
        target_traj = self._traj_info[SD.POSITION][index + 1, :2]
        all_traj_points = self.center_traj[:, ::5, :2]
        all_traj_points = all_traj_points[:,1,:2]
        losses = np.sum((all_traj_points - target_traj) ** 2, axis=1)  # shape: [8192]
        best_idx = np.argmin(losses)

        if not self.is_current_step_valid:
            return None  # Return None action so the base vehicle will not overwrite the steering & throttle

        waypoints = self.center_traj[best_idx] # shape: [41, 2]
        velocities = np.zeros((waypoints.shape[0], 2))
        velocities[1:] = (waypoints[1:, :2] - waypoints[:-1, :2]) / 0.1
        velocities[0] = velocities[1]
        headings = self.local_heading[best_idx].cpu().numpy()
        angular_velocities = np.zeros(headings.shape[0])
        angular_velocities[1:] = (headings[1:] - headings[:-1]) / 0.1
        angular_velocities[0] = angular_velocities[1]
        
        return Trajectory(
            waypoints=waypoints[::5],
            velocities=velocities[::5], 
            headings=headings[::5],
            angular_velocities=angular_velocities[::5],
        )
    
    def _convert_trajectory(self):
        """
        Convert 8192 raw trajectories to waypoints format using PyTorch
        """
        dtype = torch.float64
        human_traj = torch.cat([
            torch.zeros((8192, 1, 3), device=self.device, dtype=dtype),
            torch.from_numpy(self.raw_human_traj).to(self.device).to(dtype)
        ], dim=1)
        
        human_heading = human_traj[:, :, 2]
        human_xy = human_traj[:, :, :2]
        
        ego2local_translation = torch.tensor(self.agent.rear_vehicle.current_position, device=self.device, dtype=dtype)
        ego2local_rotation = torch.tensor([
            [np.cos(self.agent.current_heading), -np.sin(self.agent.current_heading)],
            [np.sin(self.agent.current_heading), np.cos(self.agent.current_heading)]
        ], device=self.device, dtype=dtype)
        
        local_traj = torch.matmul(human_xy, ego2local_rotation.T) + ego2local_translation
        self.local_heading = human_heading + self.agent.current_heading

        rear_2_center_translation = torch.stack([
            self.rear_axle_to_center * torch.cos(self.local_heading),
            self.rear_axle_to_center * torch.sin(self.local_heading)
        ], dim=-1)  # shape: [8192, 41, 2]
        self.center_traj = (local_traj + rear_2_center_translation).cpu().numpy()  # shape: [8192, 41, 2]

        
