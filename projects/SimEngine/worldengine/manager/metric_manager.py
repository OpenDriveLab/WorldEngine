import copy
import os
import pandas as pd
import numpy as np
import numpy.typing as npt
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import asdict

from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from nuplan.planning.simulation.planner.abstract_planner import PlannerInitialization, PlannerInput
from nuplan.planning.simulation.history.simulation_history_buffer import SimulationHistoryBuffer
from nuplan.planning.simulation.observation.observation_type import DetectionsTracks
from nuplan.common.actor_state.state_representation import TimePoint
from nuplan.common.maps.nuplan_map.map_factory import get_maps_api
from nuplan.common.maps.maps_datatypes import TrafficLightStatusData, SemanticMapLayer
from nuplan.planning.simulation.simulation_time_controller.simulation_iteration import SimulationIteration

from worldengine.common.dataclasses import NavsimTrajectory, PDMResults
from worldengine.manager.base_manager import BaseManager
from worldengine.utils.type import WorldEngineObjectType
from worldengine.utils import math_utils
from worldengine.scenario.scenarios.scenario_description import ScenarioDescription as SD
from worldengine.components.agents.policy.pdm_planner.pdm_closed_planner import PDMClosedPlanner
from worldengine.components.agents.policy.pdm_planner.proposal.batch_idm_policy import BatchIDMPolicy
from worldengine.components.agents.policy.pdm_planner.utils.WE2pdm_utils import WE2NuPlanConverter
from worldengine.components.agents.policy.pdm_planner.utils.pdm_path import PDMPath
from worldengine.components.agents.policy.pdm_planner.utils.pdm_enums import MultiMetricIndex, WeightedMetricIndex
from worldengine.components.agents.policy.pdm_planner.observation.pdm_observation import PDMObservation
from worldengine.components.agents.policy.pdm_planner.observation.pdm_occupancy_map import PDMDrivableMap
from worldengine.components.agents.policy.pdm_planner.abstract_pdm_closed_planner import transform_trajectory, get_trajectory_as_array


import logging
logger = logging.getLogger(__name__)


class MetricManager(BaseManager):
    """Manager for evaluating autonomous driving system performance metrics"""
    
    PRIORITY = 20 

    def __init__(self):
        super(MetricManager, self).__init__()
        self._future_sampling = TrajectorySampling(num_poses=self.engine.global_config['sampling_poses'] + 1, interval_length=0.5)
        self._proposal_sampling = TrajectorySampling(num_poses=self.engine.global_config['sampling_poses'], interval_length=0.5)
        self._map_radius = 100
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

        self._ttc_threshold = 4.0  # TTC threshold in seconds
        self._collision_distance_threshold = 0.3  # Collision distance threshold in meters
        self.num_history = self.engine.global_config['num_history'] # history frames
        self.num_future = self.engine.global_config['num_future'] # future frames
        
        self.current_scene = None
        self.converter = None
        self.ego_states_list = []
        self.observations_list = []
        self.detection_tracks_list = []
        self.traffic_light_data_list = []
        self.pdm_scores_list = []
        self.all_scene_averages = []

        self._cached_ego_poses = np.array([0.00, 0.00, 0.00])
        self._cached_ego_velocities = np.array([0.00, 0.00])
        self._cached_roadblock_ids = []
        self.buffer_size = self.engine.global_config['buffer_size']

        self.route_roadblock_dict = {}
        
        self.first_NC_DAC_step = None

    def before_reset(self):
        """Reset all metrics"""
        reset_info = super().before_reset()
        self.agent = None
        return reset_info

    def reset(self):
        """Reset the planner"""
        self.current_step = self.engine.episode_step
        self.current_scene = self.engine.managers['scenario_manager'].current_scene
        self.agent = self.engine.managers['agent_manager'].ego_agent
        self.converter = WE2NuPlanConverter(self.current_scene, self.agent, self.engine)
        nuplan_map_root = self.engine.global_config['nuplan_map_root']
        map_location = self.current_scene.get("map")
        self.map_api = get_maps_api(nuplan_map_root, "nuplan-maps-v1.0", map_location)
        self.ego_states_list = []
        self.observations_list = []
        self.detection_tracks_list = []
        self.traffic_light_data_list = []
        self._cached_roadblock_ids = []
        self.route_roadblock_dict = {}  
        self.first_NC_DAC_step = None


    def after_reset(self):
        """Get ego vehicle reference after reset"""
        reset_info = super().after_reset()
        return reset_info
    
    def before_step(self):
        self.current_step = self.engine.episode_step
        if self.current_scene is None:
            self._logger.error("No scene available for PDM policy.")
            return None
        elif self.current_step == 0:
            current_ego_state = self.converter.convert_to_current_ego_state(self.current_step)
            self.ego_states_list.append(current_ego_state)
            observation = self.converter.convert_to_detections_tracks_from_agent_input(self.current_step)
            self.observations_list.append(observation)
    
    def step(self):
        """Update all metrics"""
        self.current_step = self.engine.episode_step
        score_row: Dict[str, Any] = {
            "token": self.current_scene["id"], 
            "step": self.current_step,
        }
        
        if self.current_scene is None:
            self._logger.error("No scene available for PDM policy.")
            return None
        else:
            current_ego_state = self.converter.convert_to_current_ego_state(self.current_step)
            self.ego_states_list.append(current_ego_state)
            observation = self.converter.convert_to_detections_tracks_from_agent_input(self.current_step)
            self.observations_list.append(observation)
        
        # init and run PDM-Closed
        if self.current_step == self.num_history - 1: 
            planner_input, planner_initialization = self._get_planner_inputs(self.current_scene)
            self._pdm_closed.initialize(planner_initialization) # TODO: fix centerline bug
            self._map_api = planner_initialization["map_api"]
        
        if self.num_history - 1 <= self.current_step < self.num_history + self.buffer_size - 1:
            # update roadblock ids when coming to current frame
            current_roadblock_ids = self.get_current_roadblock_ids() 
            for current_roadblock_id in current_roadblock_ids:
                current_roadblock = self._map_api.get_map_object(current_roadblock_id, SemanticMapLayer.ROADBLOCK)
                if not current_roadblock:
                    current_roadblock = self._map_api.get_map_object(current_roadblock_id, SemanticMapLayer.ROADBLOCK_CONNECTOR)
                if current_roadblock_id not in self._cached_roadblock_ids:
                    self._cached_roadblock_ids.append(current_roadblock_id)
                    self.route_roadblock_dict[current_roadblock_id] = current_roadblock
            
        if self.current_step >= self.num_history - 1:
            # update detection tracks and traffic light data
            detection_tracks = self.converter.convert_to_detections_tracks_from_agent_input(self.current_step)
            self.detection_tracks_list.append(detection_tracks)
            traffic_light_data = []
            self.traffic_light_data_list.append(traffic_light_data)
        
        # cache agent trajectory
        agent_pos = self.agent.trajectory.waypoints
        agent_headings = self.agent.trajectory.headings
        agent_velocities = self.agent.trajectory.velocities
        
        # convert velocity global2ego
        ego_velocities = np.zeros_like(agent_velocities)
        for i in range(len(agent_velocities)):
            heading = agent_headings[i]
            cos_heading = np.cos(heading)
            sin_heading = np.sin(heading)
            # rotation matrix: [cos_heading, sin_heading; -sin_heading, cos_heading]
            ego_velocities[i, 0] = agent_velocities[i, 0] * cos_heading + agent_velocities[i, 1] * sin_heading
            ego_velocities[i, 1] = -agent_velocities[i, 0] * sin_heading + agent_velocities[i, 1] * cos_heading
        
        agent_poses = np.concatenate([agent_pos, agent_headings[:, None]], axis=1)
        
        if self.current_step == self.num_history - 1:
            expert_pos = self.agent.object_track['position'][self.current_step:self.current_step + self.buffer_size,:2] 
            expert_headings = self.agent.object_track['heading'][self.current_step:self.current_step + self.buffer_size]
            # center to real-axle
            expert_pos_rear = expert_pos.copy()
            for i in range(len(expert_pos)):
                expert_pos_rear[i] = math_utils.translate_longitudinally(
                    expert_pos_rear[i], 
                    expert_headings[i], 
                    -self.agent.rear_vehicle.rear_axle_to_center_dist
                ).reshape(2)
            reference_pos = expert_pos_rear[0].copy()
            for i in range(len(expert_pos)):
                expert_pos_rear[i] = expert_pos_rear[i] - reference_pos
            self._cached_expert_traj = np.concatenate([expert_pos_rear, expert_headings[:, None]], axis=1)
        
        if self.current_step == 1:
            self._cached_ego_velocities = ego_velocities[0:2]
        elif self.current_step < self.num_history + self.buffer_size - 1:
            self._cached_ego_velocities = np.concatenate([self._cached_ego_velocities, [ego_velocities[1]]])
            
        if len(self._cached_ego_velocities) > self.buffer_size:
            self._cached_ego_velocities = self._cached_ego_velocities[1:]
        
        if self.current_step == self.num_history + self.num_future - 1: 
            expert_navsim_trajectory = NavsimTrajectory(
                poses=self._cached_expert_traj, 
                trajectory_sampling=TrajectorySampling(num_poses=len(self._cached_expert_traj), interval_length=0.5)
            )
            pdm_results = self.compute_pdm_scores(expert_navsim_trajectory, self._cached_ego_velocities)
            score_row.update(asdict(pdm_results))
            score_row["first_violation_step"] = self.first_NC_DAC_step
            self.pdm_scores_list.append(score_row)
            self.save_pdm_scores()

    def compute_pdm_scores(self,expert_trajectory: NavsimTrajectory, input_trajectory_velocities: npt.NDArray[np.float64]):
        """Compute PDM scores"""
        # 1. Build PDM observation
        self._pdm_closed._load_route_dicts(self._cached_roadblock_ids)
        observation = self._build_pdm_observation(
            interpolated_detection_tracks=self.detection_tracks_list,
            interpolated_traffic_light_data=self.traffic_light_data_list,
            route_lane_dict=self._pdm_closed._route_lane_dict,
        )
        
        initial_ego_state = self.ego_states_list[self.num_history - 1]
        self._pdm_closed._drivable_area_map = PDMDrivableMap.from_simulation(
            self._map_api, initial_ego_state, self._map_radius
        )

        # 2. Set Trajectory
        expert_traj = transform_trajectory(expert_trajectory, initial_ego_state)
        pred_states = self.get_pred_states()
        expert_states = get_trajectory_as_array(expert_traj, expert_trajectory.trajectory_sampling, initial_ego_state.time_point)
        trajectory_states = np.concatenate([expert_states[None, ...], pred_states[None, ...]], axis=0)
        trajectory_states[1,:,3] = input_trajectory_velocities[:,0]
        trajectory_states[1,:,4] = input_trajectory_velocities[:,1]

        # 3. Centerline extraction
        centerline_discrete_path = []
        for i in expert_traj._trajectory:
            centerline_discrete_path.append(i.rear_axle)
        self._centerline = PDMPath(centerline_discrete_path)

        # 4. Score proposals
        scores = self._pdm_closed._scorer.score_proposals(
            trajectory_states,
            initial_ego_state,
            observation,
            self._centerline,
            self._pdm_closed._route_lane_dict,
            self._pdm_closed._drivable_area_map,
            self._map_api,
        )
        
        pred_idx = 1 # only pred_states

        no_at_fault_collisions = self._pdm_closed._scorer._multi_metrics[MultiMetricIndex.NO_COLLISION, pred_idx]
        drivable_area_compliance = self._pdm_closed._scorer._multi_metrics[MultiMetricIndex.DRIVABLE_AREA, pred_idx]

        ego_progress = self._pdm_closed._scorer._weighted_metrics[WeightedMetricIndex.PROGRESS, pred_idx]
        time_to_collision_within_bound = self._pdm_closed._scorer._weighted_metrics[WeightedMetricIndex.TTC, pred_idx]
        comfort = self._pdm_closed._scorer._weighted_metrics[WeightedMetricIndex.COMFORTABLE, pred_idx]
        driving_direction_compliance = self._pdm_closed._scorer._weighted_metrics[WeightedMetricIndex.DRIVING_DIRECTION, pred_idx]

        score = scores[pred_idx]
        # Pick the first violation frame
        if drivable_area_compliance == 0.0:
            batch_nondrivable_area_mask = self._pdm_closed._scorer._batch_nondrivable_area_mask
            if batch_nondrivable_area_mask[pred_idx].any():
                first_violation_frame = np.where(batch_nondrivable_area_mask[pred_idx])[0][0]
                dac_frame = first_violation_frame + self.num_history - 1
                self.first_NC_DAC_step = dac_frame
        if no_at_fault_collisions == 0.0:
            nc_time = self._pdm_closed._scorer.time_to_at_fault_collision(pred_idx)
            nc_frame = int(nc_time / self._proposal_sampling.interval_length) + self.num_history - 1
            if self.first_NC_DAC_step is None:
                self.first_NC_DAC_step = nc_frame
            else:
                self.first_NC_DAC_step = min(nc_frame, self.first_NC_DAC_step)

        return PDMResults(
            no_at_fault_collisions,
            drivable_area_compliance,
            ego_progress,
            time_to_collision_within_bound,
            comfort,
            driving_direction_compliance,
            score,
        )

    def _build_pdm_observation(
        self,
        interpolated_detection_tracks: List[DetectionsTracks],
        interpolated_traffic_light_data: List[List[TrafficLightStatusData]],
        route_lane_dict,
    ):
        # convert to pdm observation
        pdm_observation = PDMObservation(
            self._proposal_sampling,
            self._proposal_sampling,
            self._map_radius,
            observation_sample_res=1,
        )
        pdm_observation.update_detections_tracks(
            interpolated_detection_tracks,
            interpolated_traffic_light_data,
            route_lane_dict,
            compute_traffic_light_data=True,
        )
        return pdm_observation
    
    
    def _get_planner_inputs(self, scene) -> Tuple[PlannerInput, PlannerInitialization]:
        """
        Creates planner input arguments from scenario object.
        :param scenario: scenario object of WE
        :return: tuple of planner input and initialization objects
        """
        # observation = self.converter.convert_to_detections_tracks_from_scene(self.current_step) 
        base_timestamp = scene.get("base_timestamp", 0.0)
        time_stamp = base_timestamp + self.current_step * scene["sample_rate"] * 0.05 * 1e6
        
        route_roadblocks_ids = self.get_route_roadblocks_ids()
        
        # Initialize Planner
        planner_initialization = {
            "route_roadblock_dict_ids": route_roadblocks_ids,
            "map_api": self.map_api
        }   

        if self.current_step < 3: # history buffer size
            buffer_size = self.current_step
        else:
            buffer_size = 4

        history = SimulationHistoryBuffer.initialize_from_list(
            buffer_size=buffer_size,
            ego_states=self.ego_states_list, 
            observations=self.observations_list,
            sample_interval=0.5
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

    def get_current_roadblock_ids(self):
        """Get current roadblock ids"""
        route_roadblock_ids = self.get_route_roadblocks_ids()
        return route_roadblock_ids

    def get_pred_states(self):
        """Get pred states"""
        pred_ego_states_list = self.ego_states_list[self.num_history - 1:self.num_history + self.buffer_size - 1]
        pred_states = []
        for ego_state in pred_ego_states_list:
            state = np.zeros(11, dtype=np.float64)

            state[0] = ego_state.rear_axle.x
            state[1] = ego_state.rear_axle.y
            state[2] = ego_state.rear_axle.heading
            
            pred_states.append(state)

        return np.array(pred_states)
    
    def save_pdm_scores(self):
        """Save PDM scores to CSV file with resume support"""
        if not self.pdm_scores_list:
            logger.warning("No PDM scores to save.")
            return

        # Determine CSV path
        if self.engine.global_config['agent_policy'] == 'idm_policy':
            averages_csv_path = os.path.join(self.engine.global_config['data_output_dir'], 
                                            'all_scenes_pdm_averages_R.csv')
        else:
            averages_csv_path = os.path.join(self.engine.global_config['data_output_dir'], 
                                            'all_scenes_pdm_averages_NR.csv')

        # Check if CSV already exists for resume functionality
        new_score_row = self.pdm_scores_list[-1]
        if os.path.exists(averages_csv_path):
            existing_df = pd.read_csv(averages_csv_path)
            # Remove the last row (overall_average)
            if len(existing_df) > 0 and existing_df.iloc[-1]['token'] == 'overall_average':
                existing_df = existing_df.iloc[:-1]

            averages_df = pd.concat([existing_df, pd.DataFrame([new_score_row]).dropna(axis=1, how='all')], ignore_index=True)
        else:
            # No existing file, create new DataFrame
            averages_df = pd.DataFrame(self.pdm_scores_list)

        # Calculate overall average
        overall_averages = averages_df.drop('token', axis=1).mean()
        overall_averages_dict = overall_averages.to_dict()
        overall_averages_dict['token'] = 'overall_average'
        averages_df = pd.concat([averages_df, pd.DataFrame([overall_averages_dict]).dropna(axis=1, how='all')], ignore_index=True)

        averages_df.to_csv(averages_csv_path, float_format='%.5f', index=False)
        logger.info(f"Saved metrics to {averages_csv_path} successfully.")
