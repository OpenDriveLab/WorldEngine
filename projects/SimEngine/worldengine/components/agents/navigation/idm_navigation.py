"""
Used for other agent cars, following an IDM-controller.
Follow a given reference trajectory given a map
"""

from collections import deque
import numpy as np

from worldengine.engine.engine_utils import get_engine

from worldengine.scenario.scenarios.parse_scenario_state import parse_full_trajectory
from worldengine.scenario.scenarios.scenario_description import ScenarioDescription as SD
from worldengine.components.maps.lanes.center_lane import CenterLane

from worldengine.utils import math_utils

class IDMNavigation:
    """
    Used for IDM-controlled agents, combining original trajectory and map lanes.
    First follows the original trajectory, then continues on map lanes.
    """
    WAY_POINTS_INTERVAL = 2  # m, interval between waypoints
    NUM_WAY_POINTS = 10
    CHECK_POINT_INFO_DIM = 2  # 2 for x && y coordinates
    MAXIMUM_NAVI_POINT_DIST = 30  # m
    MAX_LATERAL_DIST = 3  # m
    ROUTE_WIDTH = 2.0

    def __init__(self, agent):
        self.agent = agent
        
        # Parse original trajectory
        self.full_traj, self.valid_indicator = parse_full_trajectory(self.agent.object_track)
        self.original_route = CenterLane(self.full_traj, width=self.ROUTE_WIDTH)
        self.original_traj_length = self.original_route.length
        
        # Current state
        self.following_original_traj = True
        self.current_lane = None
        self.next_ref_lanes = None
        
        # Navigation tracking
        self.last_and_current_long = deque([0.0, 0.0], maxlen=2)
        self.last_and_current_lat = deque([0.0, 0.0], maxlen=2)
        self.last_and_current_heading = deque([0.0, 0.0], maxlen=2)
        
        # Navigation info
        self._navi_info = np.zeros((self.get_navigation_info_dim(),), dtype=np.float32)
        self._route_completion = 0
        
        # Checkpoint lanes
        self._checkpoints_lane_indexes = []

    def reset(self):
        """Initialize navigation state"""
        # Start with original trajectory
        self.current_lane = self.original_route
        self.following_original_traj = True
        
        # Get map lane info for later use
        possible_lanes = self.map.road_network.get_closest_lane_index(
            self.agent.current_position, return_all=True)
        possible_lane_indexes = [lane_index for dist, lane_index, lane in possible_lanes]
        _, self.dest_lane_index, self.dest_lane = self.map.road_network.get_closest_lane_index(self.full_traj[-1])
        
        _, self.map_lane_index, self.map_lane = possible_lanes[0]
        
        # Initialize references
        self.current_ref_lanes = [self.current_lane]
        self.next_ref_lanes_idx, self.next_ref_lanes = self._get_next_available_lanes(self.map_lane)
        
        # Initialize checkpoint lanes
        self._checkpoints_lane_indexes = []
        for point in self.full_traj:
            _, lane_index, _ = self.map.road_network.get_closest_lane_index(point)
            if lane_index not in self._checkpoints_lane_indexes:
                self._checkpoints_lane_indexes.append(lane_index)
        
        # Set up route and update location
        self.update_localization()

    def update_localization(self):
        """Update route and navigation information"""
        # Get current position info
        agent_position = self.agent.current_position
        long, lat = self.current_lane.local_coordinates(agent_position)
        route_heading = self.current_lane.heading_theta_at(long)
        
        # Update tracking queues
        self.last_and_current_long.append(long)
        self.last_and_current_lat.append(lat)
        self.last_and_current_heading.append(route_heading)

        # Update checkpoint lanes
        current_lane_index = self.map.road_network.get_closest_lane_index(agent_position)[1]
        if current_lane_index not in self._checkpoints_lane_indexes:
            self._checkpoints_lane_indexes.append(current_lane_index)

        # Check for route transition
        if self.following_original_traj:
            if long > self.original_traj_length * 0.9:  # Near end of original trajectory
                self.following_original_traj = False
                self.current_lane = self.dest_lane
                self.current_ref_lanes = self.map.road_network.get_peer_lanes_from_index(self.dest_lane_index)
                self.next_ref_lanes_idx, self.next_ref_lanes = self._get_next_available_lanes(self.current_lane)     
        else:
            if long > self.current_lane.length * 0.9 and self.next_ref_lanes:  # Near end of current lane
                self.current_lane = self.next_ref_lanes[0]
                self.current_ref_lanes = self.next_ref_lanes
                self.next_ref_lanes_idx, self.next_ref_lanes = self._get_next_available_lanes(self.current_lane)
               
        # Update route completion
        if self.following_original_traj:
            self._route_completion = long / self.original_traj_length
        else:
            self._route_completion = long / self.current_lane.length

    def _select_best_lane(self, next_ref_lanes, checkpoints_lane_indexes):
        for lane in next_ref_lanes:
            if lane.index in checkpoints_lane_indexes:
                return lane
        return next_ref_lanes[0]

    def _get_next_available_lanes(self, current_lane):
        """Get available lane to continue on"""
        if hasattr(current_lane, 'exit_lanes'):
            next_lane_idxs = current_lane.exit_lanes
            next_lanes = [self.map.road_network.get_lane(idx) for idx in next_lane_idxs]
            return next_lane_idxs, next_lanes
        else:
            return [], []

    def get_navigation_info_dim(self):
        return self.NUM_WAY_POINTS * self.CHECK_POINT_INFO_DIM + 2

    def destroy(self):
        self._route_completion = 0

    @property
    def engine(self):
        return get_engine()

    @property
    def map(self):
        return self.engine.current_map

    @property
    def navi_info(self):
        return self._navi_info

    def get_current_lateral_range(self) -> float:
        return self.current_lane.width * len(self.current_ref_lanes)

    def get_current_lane_width(self) -> float:
        return self.current_lane.width

    def get_current_lane_num(self) -> float:
        return len(self.current_ref_lanes)

    @property
    def checkpoints_lane_indexes(self):
        return self._checkpoints_lane_indexes

    @property
    def checkpoint_lanes(self):
        return [self.map.road_network.get_lane(ckpt) for ckpt in self.checkpoints_lane_indexes]
