"""
Used for other agent cars, following a given trajectory.
"""

from collections import deque
import numpy as np

from worldengine.engine.engine_utils import get_engine

from worldengine.scenario.scenarios.parse_scenario_state import parse_full_trajectory
from worldengine.scenario.scenarios.scenario_description import ScenarioDescription as SD
from worldengine.components.maps.lanes.center_lane import CenterLane

from worldengine.utils import math_utils


class TrajectoryNavigation:
    WAY_POINTS_INTERVAL = 2  # m, interval between waypoints.

    # Next waypoints for controller.
    NUM_WAY_POINTS = 10  # 10 is the same as MetaDrive.
    CHECK_POINT_INFO_DIM = 2  # 2 for x && y coordinates in the global coordinates.

    #  maximum navigation point distance,
    #   used to clip value, should be greater than WAY_POINTS_INTERVAL * MAX_NUM_WAY_POINT
    MAXIMUM_NAVI_POINT_DIST = 30  # m

    # maximum difference between the controlled trajectory
    #  and target route.
    MAX_LATERAL_DIST = 3  # m

    # route related properties.
    ROUTE_WIDTH = 2.0

    def __init__(self, agent):
        self.agent = agent

        self.full_traj, self.valid_indicator = parse_full_trajectory(self.agent.object_track)
        self.reference_route = CenterLane(self.full_traj, width=self.ROUTE_WIDTH)

        # agent trajectory information.
        self.last_and_current_long = deque([0.0, 0.0], maxlen=2)
        self.last_and_current_lat = deque([0.0, 0.0], maxlen=2)
        self.last_and_current_heading = deque([0.0, 0.0], maxlen=2)

        # navigation information.
        self._navi_info = np.zeros((self.get_navigation_info_dim(),), dtype=np.float32)

    def reset(self):
        self.current_lane = self.reference_route
        
        possible_lanes = self.map.road_network.get_closest_lane_index(
            self.agent.current_position, return_all=True)
        possible_lane_indexes = [lane_index for dist, lane_index, lane in possible_lanes]
        _, dest_lane_index, _ = self.map.road_network.get_closest_lane_index(self.full_traj[-1])

        self._checkpoints_lane_indexes = []
        for point in self.full_traj:
            _, lane_index, _ = self.map.road_network.get_closest_lane_index(point)
            if lane_index not in self._checkpoints_lane_indexes:
                self._checkpoints_lane_indexes.append(lane_index)
        
        self.set_route()
        self.update_localization()

    def set_route(self):
        """
        Find the shortest path from current_lane_index to destination_lane_index.
        """
        self._checkpoints = self.discretize_reference_trajectory()

    def discretize_reference_trajectory(self):
        ret = []
        length = self.reference_route.length
        num = int(length / self.WAY_POINTS_INTERVAL)
        for i in range(num):
            ret.append(self.reference_route.position(i * self.WAY_POINTS_INTERVAL, 0))
        ret.append(self.reference_route.end)
        return ret

    def update_localization(self):
        """
        The method for updating route / checkpoints information
            according to the associated ego_vehicle motion.
        """
        assert self.reference_route is not None

        # Update ckpt index
        agent_position = self.agent.current_position
        long, lat = self.reference_route.local_coordinates(agent_position)
        route_heading = self.reference_route.heading_theta_at(long)
        self.last_and_current_long.append(long)
        self.last_and_current_lat.append(lat)
        self.last_and_current_heading.append(route_heading)

        # update checkpoints_lane_indexes
        if hasattr(self.agent, 'trajectory') and self.agent.trajectory is not None:
            self._checkpoints_lane_indexes = []
            _, current_lane_index, _ = self.map.road_network.get_closest_lane_index(agent_position)
            self._checkpoints_lane_indexes.append(current_lane_index)

            for waypoint in self.agent.trajectory.waypoints:
                _, waypoint_lane_index, _ = self.map.road_network.get_closest_lane_index(
                    waypoint)
            
                if waypoint_lane_index not in self._checkpoints_lane_indexes:
                    self._checkpoints_lane_indexes.append(waypoint_lane_index)
            
        # find the next target goal points,
        #  and <NUM_WAY_POINTS> of target goal points.
        next_idx = max(int(long / self.WAY_POINTS_INTERVAL) + 1, 0)
        next_idx = min(next_idx, len(self.checkpoints) - 1)
        end_idx = min(next_idx + self.NUM_WAY_POINTS, len(self.checkpoints))
        ckpts = self.checkpoints[next_idx:end_idx]
        diff = self.NUM_WAY_POINTS - len(ckpts)
        assert diff >= 0, "Number of Navigation points error!"
        if diff > 0:  # not enough waypoints.
            ckpts += [self.checkpoints[-1] for _ in range(diff)]

        # update the navi information.
        # The navi_information includes:
        #  (NUM_WAY_POINTS * 2)
        #  the heading and rhs difference between current position to the next NUM_WAY_POINTS target points.
        #  (2)
        #  The lateral difference and angle difference between ego_car and associated lane.
        self._navi_info.fill(0.0)
        for index, ckpt in enumerate(ckpts):
            start = index * self.CHECK_POINT_INFO_DIM
            end = (index + 1) * self.CHECK_POINT_INFO_DIM
            self._navi_info[start:end] = self._get_info_for_checkpoint(ckpt)

        # finally add relative information of current position / heading
        # to the target route.
        self._navi_info[end] = math_utils.clip(
            (lat / self.MAX_LATERAL_DIST + 1) / 2, 0.0, 1.0)
        self._navi_info[end + 1] = math_utils.clip(
            (math_utils.wrap_to_pi(route_heading - self.agent.current_heading) / np.pi + 1) / 2, 0.0, 1.0
        )

        self._route_completion = long / self.reference_route.length

    def _get_info_for_checkpoint(self, ckpt):
        """ Get navigation information of agent from the target checkpoints.

        Args:
            ckpt: A numpy array with shape of [2], indicating the
                global position of the checkpoint.
        """

        navi_information = []
        # Project the checkpoint position into the target vehicle's coordination, where
        # +x is the heading and +y is the right hand side.
        dir_vec = ckpt - self.agent.current_position  # get the vector from center of vehicle to checkpoint
        dir_norm = math_utils.norm(dir_vec[0], dir_vec[1])
        # if the checkpoint is too far then crop the direction vector
        if dir_norm > self.MAXIMUM_NAVI_POINT_DIST:
            dir_vec = dir_vec / dir_norm * self.MAXIMUM_NAVI_POINT_DIST

        ckpt_in_heading, ckpt_in_rhs = self.agent.convert_to_local_coordinates(dir_vec)  # project to vehicle's coordination

        # Dim 1: the relative position of the checkpoint in the target vehicle's heading direction.
        navi_information.append(
            math_utils.clip(
                (ckpt_in_heading / self.MAXIMUM_NAVI_POINT_DIST + 1) / 2, 0.0, 1.0)
        )

        # Dim 2: the relative position of the checkpoint in the target vehicle's right hand side direction.
        navi_information.append(
            math_utils.clip(
                (ckpt_in_rhs / self.MAXIMUM_NAVI_POINT_DIST + 1) / 2, 0.0, 1.0)
        )

        return navi_information

    def get_navigation_info_dim(self):
        # The additional 2 is relative heading distance and relative latitude distance.
        return self.NUM_WAY_POINTS * self.CHECK_POINT_INFO_DIM + 2

    def destroy(self):
        self._checkpoints = None
        self._route_completion = 0

    @property
    def engine(self):
        return get_engine()

    @property
    def map(self):
        return self.engine.current_map

    @property
    def checkpoints(self):
        return self._checkpoints
    
    @property
    def checkpoints_lane_indexes(self):
        return self._checkpoints_lane_indexes

    @property
    def checkpoint_lanes(self):
        return [self.map.road_network.get_lane(ckpt) for ckpt in self.checkpoints_lane_indexes]

    @property
    def current_ref_lanes(self):
        return [self.reference_route]

    @property
    def navi_info(self):
        return self._navi_info

    def get_current_lateral_range(self) -> float:
        return self.current_lane.width * 2

    def get_current_lane_width(self) -> float:
        return self.current_lane.width

    def get_current_lane_num(self) -> float:
        return 1
