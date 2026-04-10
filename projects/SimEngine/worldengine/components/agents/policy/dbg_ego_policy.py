"""
The debugging Ego policy is used to
transfer the GT trajectory into the planning trajectory.

Starting from the current Ego position, to the following waypoints.
"""

import logging

from worldengine.components.agents.policy.base_policy import BasePolicy
from worldengine.scenario.scenarios.scenario_description import ScenarioDescription as SD
from worldengine.common.dataclasses import Trajectory


class DebugEgoPolicy(BasePolicy):

    def __init__(self, agent, random_seed=None, config=None):
        super(DebugEgoPolicy, self).__init__(agent=agent, random_seed=random_seed, config=config)

        self._traj_info = self.agent.object_track
        self._valid_indicator = self._traj_info[SD.VALID] == 1
        self._traj_length = len(self._valid_indicator)

    @property
    def is_current_step_valid(self):
        return True

    def act(self, *args, **kwargs):
        """
        Return a set of waypoints.
        """

        import numpy as np
        from worldengine.components.maps.lanes.center_lane import CenterLane
        # First, generate the lane according to the route.
        waypoints_1 = self._traj_info[SD.POSITION][:39, :2]
        waypoints_2 = np.tile(self._traj_info[SD.POSITION][39:40, :2], [100, 1])
        waypoints = np.concatenate([waypoints_1, waypoints_2], axis=0)
        target_lane = CenterLane(waypoints, width=2)

        cur_pos_longtitude = target_lane.local_coordinates(self.agent.current_position)[0]

        # Then index the cur_pos_longtitude.
        import numpy as np
        valid_longtitude = np.arange(cur_pos_longtitude, target_lane.length, 1).tolist()
        valid_longtitude.append(target_lane.length)

        waypoints = np.array([
            target_lane.position(p, 0) for p in valid_longtitude
        ]).reshape(-1, 2)

        headings = np.array([
            target_lane.heading_theta_at(p) for p in valid_longtitude
        ])

        index = max(int(self.traj_step), 0)
        return Trajectory(
            waypoints=waypoints,
            velocities=self._traj_info['velocity'][index:],
            headings=headings,
            angular_velocities=self._traj_info["angular_velocity"][index:],
        )
