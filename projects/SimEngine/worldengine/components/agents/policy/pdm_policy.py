import os
import logging
import numpy as np
from typing import Any, Dict, Optional, Tuple

from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from nuplan.planning.simulation.planner.abstract_planner import PlannerInitialization, PlannerInput
from nuplan.planning.simulation.simulation_time_controller.simulation_iteration import SimulationIteration
from nuplan.common.actor_state.state_representation import TimePoint
from nuplan.planning.simulation.history.simulation_history_buffer import SimulationHistoryBuffer
from worldengine.components.agents.policy.base_policy import BasePolicy
from nuplan.common.maps.nuplan_map.map_factory import get_maps_api
from worldengine.components.agents.policy.pdm_planner.pdm_closed_planner import PDMClosedPlanner
from worldengine.components.agents.policy.pdm_planner.proposal.batch_idm_policy import BatchIDMPolicy
from collections import defaultdict
from shapely.geometry import Polygon

from worldengine.common.dataclasses import Trajectory
from worldengine.components.agents.policy.pdm_planner.utils.WE2pdm_utils import WE2NuPlanConverter
class PDMPolicy(BasePolicy):
    """
    PDM Policy class.
    """ 

    def __init__(self, config: dict):
        """
        Constructor for PDMPolicy
        :param config: Configuration dictionary
        """
        super(PDMPolicy, self).__init__(config)
        self._future_sampling = TrajectorySampling(num_poses=50, interval_length=0.5)
        self._proposal_sampling = TrajectorySampling(num_poses=40, interval_length=0.5)
        self._map_radius = 100
        self.scenario_manager = self.engine.managers['scenario_manager']
        self._pdm_closed = PDMClosedPlanner(
            trajectory_sampling=self._future_sampling,
            proposal_sampling=self._proposal_sampling,
           idm_policies=BatchIDMPolicy(
                speed_limit_fraction=[0.15, 0.25, 0.5, 0.75, 1.0],
                fallback_target_velocity=30.0,
                min_gap_to_lead_agent=10.0,
                headway_time=1.5,
                accel_max=4.5,
                decel_max=3.0,
            ),
            lateral_offsets=[-3.5, 3.5],
            map_radius=self._map_radius,
        ) # register planner

        self.current_scene = self.scenario_manager.current_scene
        self.converter = WE2NuPlanConverter(self.current_scene, self.agent, self.engine)
        nuplan_map_root = self.engine.global_config['nuplan_map_root']
        map_location = self.scenario_manager.current_scene.get("map")
        self.map_api = get_maps_api(nuplan_map_root, "nuplan-maps-v1.0", map_location)

        self.ego_states_list = []
        self.observations_list = []
        self._logger = logging.getLogger(__name__)

    def act(self):
        """
        Run the PDM policy.

        """
        self.current_step = self.engine.episode_step 
        if self.current_scene is None:
            self._logger.error("No scene available for PDM policy.")
            return None
        else:
            current_ego_state = self.converter.convert_to_current_ego_state(self.current_step)
            self.ego_states_list.append(current_ego_state)
        
        planner_input, planner_initialization = self._get_planner_inputs(self.current_scene)
        
        self._pdm_closed.initialize(planner_initialization)
        
        planned_trajectory = self._pdm_closed.compute_planner_trajectory(planner_input)
        
        trajectory = self.converter.convert_to_trajectory(planned_trajectory)
        
        return trajectory
    
    def _get_planner_inputs(self, scene) -> Tuple[PlannerInput, PlannerInitialization]:
        """
        Creates planner input arguments from scenario object.
        :param scenario: scenario object of WE
        :return: tuple of planner input and initialization objects
        """
        # observation = self.converter.convert_to_detections_tracks_from_scene(self.current_step) 
        observation = self.converter.convert_to_detections_tracks_from_agent_input(self.current_step)
        self.observations_list.append(observation)
        base_timestamp = scene.get("base_timestamp", 0.0)
        time_stamp = base_timestamp + self.current_step * scene["sample_rate"] * 0.05 * 1e6
        
        route_roadblocks_ids = self.get_route_roadblocks_ids()
        
        # Initialize Planner
        planner_initialization = {
            "route_roadblock_dict_ids": route_roadblocks_ids,
            "map_api": self.map_api
        }   

        if self.current_step <= 5:
            buffer_size = self.current_step + 1
        else:
            buffer_size = 5

        history = SimulationHistoryBuffer.initialize_from_list(
            buffer_size=buffer_size,
            ego_states=self.ego_states_list, 
            observations=self.observations_list
        )

        traffic_light_data = self.converter.convert_to_traffic_lights(self.current_step)
        planner_input = PlannerInput(
            iteration = SimulationIteration(index=self.current_step, time_point=TimePoint(time_stamp)),
            history = history,
            traffic_light_data = traffic_light_data,
        )

        return planner_input, planner_initialization
    
    def get_route_roadblocks_ids(self):
        
        route = self.agent.navigation.checkpoint_lanes
        route_roadblocks_ids = []
        for lane in route:
            roadblock_id = lane.roadblock_id
            if roadblock_id:
                route_roadblocks_ids.append(roadblock_id)
        return route_roadblocks_ids

    
    @property
    def is_current_step_valid(self):
        return True
