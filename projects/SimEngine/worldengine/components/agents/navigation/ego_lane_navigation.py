"""
Used for Ego-car.
Navigation helper for localizing vehicles on road maps.
"""


from collections import deque
import numpy as np

from worldengine.engine.engine_utils import get_engine
from worldengine.components.maps.road_networks.edge_road_network import EdgeRoadNetwork
from worldengine.utils import math_utils, agent_utils


class EgoLaneNavigation:

    # Next waypoints for controller.
    NUM_WAY_POINTS = 2  # 2 means the current lane and next lane.
    CHECK_POINT_INFO_DIM = 2  # 2 for x && y coordinates in the global coordinates.

    #  maximum navigation point distance,
    #   used to clip value, should be greater than WAY_POINTS_INTERVAL * MAX_NUM_WAY_POINT
    MAXIMUM_NAVI_POINT_DIST = 30  # m

    def __init__(self, agent):
        self.agent = agent

        # used for filter current_lane.
        self.lane_valid_range = self.agent.length

        assert isinstance(self.map.road_network, EdgeRoadNetwork)

        # navigation information.
        self._navi_info = np.zeros((self.get_navigation_info_dim(),), dtype=np.float32)

    def reset(self):

        possible_lanes = self.map.road_network.get_closest_lane_index(
            self.agent.current_position, return_all=True)
        possible_lane_indexes = [lane_index for dist, lane_index, lane in possible_lanes]

        if len(possible_lanes) == 0:
            raise ValueError("Can't find valid lane for navigation.")

        spawn_lane_index = self.agent.config.get('spawn_lane_index', None)
        if spawn_lane_index is not None and spawn_lane_index in possible_lane_indexes:
            dist, lane_index, lane = possible_lanes[spawn_lane_index]
        else:
            dist, lane_index, lane = possible_lanes[0]

        dest = self.agent.destination
        dest_dist, dest_lane_index, dest_lane = self.map.road_network.get_closest_lane_index(dest)

        current_lane = lane
        assert current_lane is not None, "spawn place is not on road!"

        self.current_lane = current_lane
        self.set_route(current_lane.index, dest_lane_index)

    def set_route(self, current_lane_index, dest_lane_index):
        """
        Find the shortest path from current_lane_index to destination_lane_index.
        """

        self._checkpoints_lane_indexes = self.map.road_network.shortest_path(current_lane_index, dest_lane_index)

        self._checkpoints = []
        for lane_index in self._checkpoints_lane_indexes:
            lane = self.map.road_network.get_lane(lane_index)
            self._checkpoints.append(lane.end)  # global coordinates

        # target_checkpoints_index:
        #  used to index the current lane and next lane in the _checkpoints.
        #  Initialized as [0, 1]
        #  and will be updated accordingly during the moving of ego vehicle.
        self._target_checkpoints_index = [0, 1]
        # update routing info
        assert len(self.checkpoints_lane_indexes) > 0, "Can not find a route from {} to {}".format(
            current_lane_index, dest_lane_index)

        self.final_lane = self.map.road_network.get_lane(self.checkpoints_lane_indexes[-1])
        self._navi_info.fill(0.0)
        self.current_ref_lanes = self.map.road_network.get_peer_lanes_from_index(self.current_checkpoint_lane_index)
        self.next_ref_lanes = self.map.road_network.get_peer_lanes_from_index(self.next_checkpoint_lane_index)

        # update navigation information.
        half = self.CHECK_POINT_INFO_DIM
        self._navi_info[:half] = self._get_info_for_checkpoint(
            ref_lane=self.map.road_network.get_lane(self.current_checkpoint_lane_index),
        )
        self._navi_info[half:] = self._get_info_for_checkpoint(
            ref_lane=self.map.road_network.get_lane(self.next_checkpoint_lane_index),
        )

    def update_localization(self):
        lane, lane_index = self._update_current_lane()
        updated = self._update_target_checkpoints(lane_index)

        if updated:  # update current_ref_lanes, and next_ref_lanes.
            self.current_ref_lanes = self.map.road_network.get_peer_lanes_from_index(
                self.current_checkpoint_lane_index)

            self.next_ref_lanes = self.map.road_network.get_peer_lanes_from_index(
                self.next_checkpoint_lane_index)

            if self.current_checkpoint_lane_index == self.next_checkpoint_lane_index:
                # When we are in the final road segment that there is no further road to drive on
                self.next_ref_lanes = None

        # update the _navi_information.
        # The navi_information includes:
        self._navi_info.fill(0.0)
        half = self.CHECK_POINT_INFO_DIM
        self._navi_info[:half] = self._get_info_for_checkpoint(
            ref_lane=self.map.road_network.get_lane(self.current_checkpoint_lane_index),
        )
        self._navi_info[half:] = self._get_info_for_checkpoint(
            ref_lane=self.map.road_network.get_lane(self.next_checkpoint_lane_index),
        )

    def _update_current_lane(self):
        dist, lane_index, lane = agent_utils.get_current_lane(self.agent, self.map)
        self.current_lane = lane
        assert lane_index == lane.index, "lane index mismatch!"
        return lane, lane_index

    def _update_target_checkpoints(self, current_lane_index) -> bool:
        """
        update the checkpoint, return True if updated else False
        """
        # At the end of the route,
        if self.current_checkpoint_lane_index == self.next_checkpoint_lane_index:  # on last road
            return False

        # At the middle of route.
        if current_lane_index in self.checkpoints_lane_indexes[self._target_checkpoints_index[1]:]:
            idx = self.checkpoints_lane_indexes.index(current_lane_index,
                                         self._target_checkpoints_index[1])
            self._target_checkpoints_index = [idx]
            if idx + 1 == len(self.checkpoints_lane_indexes):  # the last one lane.
                self._target_checkpoints_index.append(idx)
            else:  # else, update as the next lane.
                self._target_checkpoints_index.append(idx + 1)
            return True
        return False

    def _get_info_for_checkpoint(self, ref_lane):
        """ Get navigation information of agent from the target checkpoints.

        ref_lane: the lane object.
        """

        navi_information = []
        # Project the checkpoint position into the target vehicle's coordination, where
        # +x is the heading and +y is the left hand side.
        check_point = ref_lane.end
        dir_vec = check_point - self.agent.current_position  # get the vector from center of vehicle to checkpoint
        dir_norm = math_utils.norm(dir_vec[0], dir_vec[1])
        if dir_norm > self.MAXIMUM_NAVI_POINT_DIST:  # if the checkpoint is too far then crop the direction vector
            dir_vec = dir_vec / dir_norm * self.MAXIMUM_NAVI_POINT_DIST
        ckpt_in_heading, ckpt_in_rhs = self.agent.convert_to_local_coordinates(dir_vec)

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
        return self.NUM_WAY_POINTS * self.CHECK_POINT_INFO_DIM

    def destroy(self):
        self._checkpoints = None
        self._checkpoints_lane_indexes = None
        self._route_completion = 0
    
    @property
    def route_roadblocks_ids(self):
        route = self.checkpoint_lanes
        route_roadblocks_ids = []
        for lane in route:
            roadblock_id = lane.roadblock_id
            if roadblock_id:
                route_roadblocks_ids.append(roadblock_id)
        return route_roadblocks_ids

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

    def get_current_lateral_range(self) -> float:
        return self.current_lane.width * len(self.current_ref_lanes)

    def get_left_lateral_range(self) -> float:
        # Range from current lane to its left lane.
        left_ref_lanes = self.map.road_network.get_left_peer_lanes_from_index(
            self.current_checkpoint_lane_index)
        return len(left_ref_lanes) * self.current_lane.width

    def get_current_lane_width(self) -> float:
        return self.current_lane.width

    def get_current_lane_num(self) -> float:
        return 1

    @property
    def current_checkpoint_lane_index(self):
        return self.checkpoints_lane_indexes[self._target_checkpoints_index[0]]

    @property
    def next_checkpoint_lane_index(self):
        return self.checkpoints_lane_indexes[self._target_checkpoints_index[1]]

    @property
    def navi_info(self):
        return self._navi_info