import logging

from worldengine.components.agents.policy.base_policy import BasePolicy
from worldengine.scenario.scenarios.scenario_description import ScenarioDescription as SD
from worldengine.common.dataclasses import Trajectory


class TrajectoryPolicy(BasePolicy):

    def __init__(self, agent, random_seed=None, config=None):
        super(TrajectoryPolicy, self).__init__(agent=agent, random_seed=random_seed, config=config)

        self._traj_info = self.agent.object_track
        self._valid_indicator = self._traj_info[SD.VALID] == 1
        self._traj_length = len(self._valid_indicator)

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
        # if index >= self._traj_length:
        #     return None

        if not self.is_current_step_valid:
            return None  # Return None action so the base vehicle will not overwrite the steering & throttle

        return Trajectory(
            waypoints=self._traj_info[SD.POSITION][index:index + 5, :2],
            velocities=self._traj_info['velocity'][index:index + 5],
            headings=self._traj_info[SD.HEADING][index:index + 5],
            angular_velocities=self._traj_info["angular_velocity"][index:index + 5],
        )
