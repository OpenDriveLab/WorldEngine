import os
import pickle
import pandas as pd
import numpy as np
import torch
import time
from multiprocessing import Pool
from functools import partial
from typing import Any, Dict, List, Tuple

from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from nuplan.planning.simulation.planner.abstract_planner import PlannerInitialization, PlannerInput
from nuplan.planning.simulation.history.simulation_history_buffer import SimulationHistoryBuffer
from nuplan.planning.simulation.observation.observation_type import DetectionsTracks
from nuplan.common.actor_state.agent import Agent
from nuplan.common.actor_state.static_object import StaticObject
from nuplan.common.actor_state.scene_object import SceneObjectMetadata
from nuplan.common.actor_state.tracked_objects import TrackedObjects
from nuplan.common.actor_state.state_representation import TimePoint, StateSE2, StateVector2D
from nuplan.common.actor_state.oriented_box import OrientedBox
from nuplan.common.maps.nuplan_map.map_factory import get_maps_api
from nuplan.common.maps.maps_datatypes import TrafficLightStatusData, SemanticMapLayer
from nuplan.planning.simulation.simulation_time_controller.simulation_iteration import SimulationIteration

from worldengine.manager.base_manager import BaseManager
from worldengine.scenario.scenarios.scenario_description import ScenarioDescription as SD
from worldengine.components.agents.policy.pdm_planner.pdm_closed_planner import PDMClosedPlanner
from worldengine.components.agents.policy.pdm_planner.proposal.batch_idm_policy import BatchIDMPolicy
from worldengine.components.agents.policy.pdm_planner.utils.WE2pdm_utils import WE2NuPlanConverter
from worldengine.components.agents.policy.pdm_planner.utils.pdm_path import PDMPath
from worldengine.components.agents.policy.pdm_planner.utils.pdm_enums import MultiMetricIndex, WeightedMetricIndex
from worldengine.components.agents.policy.pdm_planner.observation.pdm_observation import PDMObservation
from worldengine.components.agents.policy.pdm_planner.observation.pdm_occupancy_map import PDMDrivableMap
from worldengine.components.agents.policy.pdm_planner.simulation.pdm_simulator import PDMSimulator
from worldengine.components.agents.policy.pdm_planner.simulation.torch_simulator import TorchSimulator


import logging
logger = logging.getLogger(__name__)
WE_root = os.environ.get("WORLDENGINE_ROOT")

def wrap_to_pi(theta):
    return (theta+np.pi) % (2*np.pi) - np.pi

class DenseRewardManager(BaseManager):
    """Manager for evaluating autonomous driving system performance metrics"""
    
    PRIORITY = 20 

    def __init__(self):
        super(DenseRewardManager, self).__init__()

        self._num_scoring_workers = 8
        self._scoring_pool = Pool(processes=self._num_scoring_workers)

        self._future_sampling = TrajectorySampling(num_poses=self.engine.global_config['reward_sampling_poses'] * 5 + 1, interval_length=0.1)
        self._proposal_sampling = TrajectorySampling(num_poses=self.engine.global_config['reward_sampling_poses'] * 5, interval_length=0.1)
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
        self.openloop_pdm_scores_list = []
        self.all_scene_averages = []


        self._cached_roadblock_ids = []
        self.buffer_size = self.engine.global_config['reward_buffer_size']

        self.route_roadblock_dict = {}
        gt_array = np.load(os.path.join(WE_root, 'data/alg_engine/test_8192_kmeans.npy'))
        self.gt_array = gt_array[:,:self.engine.global_config['reward_sampling_poses'] * 5,:]
        self.pkl_paths_df = pd.DataFrame(columns=["token", "step", "pkl_path"])
        # check cuda availability
        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # IL_plan_idx related
        self.raw_human_traj = np.load(
            os.path.join(
                WE_root,
                "data/alg_engine/test_8192_kmeans.npy"
            )
        )
        self.rear_axle_to_center = 1.461

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
        self._traj_info = self.agent.object_track
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


    def after_reset(self):
        """Get ego vehicle reference after reset"""
        reset_info = super().after_reset()
        return reset_info
    
    def before_step(self):
        self.current_step = self.engine.episode_step
        if self.current_scene is None:
            logger.error("No scene available for PDM policy.")
            return None
        elif self.current_step == 0:
            current_ego_state = self.converter.convert_to_current_ego_state(self.current_step)
            self.ego_states_list.append(current_ego_state)
            obs_len = self.engine.global_config['num_history'] + self.engine.global_config['num_future'] + self.engine.global_config['reward_sampling_poses']
            for i in range(obs_len): # TODO: NR or R
                observation = self.converter.convert_to_detections_tracks_from_scene(i)
                self.observations_list.append(observation)
                traffic_light_data = []
                self.traffic_light_data_list.append(traffic_light_data)
    
    def step(self):
        """Update all metrics"""
        self.current_step = self.engine.episode_step
        score_row: Dict[str, Any] = {"token": self.current_scene["token"], "step": self.current_step}
        
        if self.current_scene is None:
            logger.error("No scene available for PDM policy.")
            return None
        else:
            current_ego_state = self.converter.convert_to_current_ego_state(self.current_step)
            self.ego_states_list.append(current_ego_state)
        
        # init and run PDM-Closed
        if self.current_step >= self.num_history - 1: 
            self.planner_input, planner_initialization = self._get_planner_inputs(self.current_scene)
            self._pdm_closed.initialize(planner_initialization) 
            self._map_api = planner_initialization["map_api"]
            pdm_closed_trajectory = self._pdm_closed.compute_planner_trajectory(self.planner_input)
            if self.current_step == self.num_history - 1:
                self.openloop_pdm_closed_trajectory = pdm_closed_trajectory
            
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
        
        if self.current_step >= self.num_history - 1: 
            pdm_results = self.compute_pdm_scores(pdm_closed_trajectory)
            pkl_filename = f"{self.current_scene['id']}_step_{self.current_step}_scores.pkl"
            pkl_path = os.path.join(self.engine.global_config['data_output_dir'], "pdms_pkl", pkl_filename)

            os.makedirs(os.path.dirname(pkl_path), exist_ok=True)
            # IL_plan_idx
            self._convert_trajectory()
            target_traj = self._traj_info[SD.POSITION][self.current_step + 1 : self.current_step + self.engine.global_config['reward_sampling_poses'] + 1, :2] # TODO: transform to current ego state
            target_traj = target_traj - self.agent.current_position
            target_point = self._traj_info[SD.POSITION][self.current_step + 1, :2]
            all_traj_points = self.center_traj[:, ::5, :2]
            all_traj_points = all_traj_points[:,1,:2]
            losses = np.sum((all_traj_points - target_point) ** 2, axis=1)  # shape: [8192]
            best_idx = np.argmin(losses)
            pdm_results.update({
                'IL_plan_idx': best_idx,
                'target_traj': target_traj
            })

            with open(pkl_path, 'wb') as f:
                pickle.dump(pdm_results, f)
            
            if not hasattr(self, 'pkl_paths_list'):
                self.pkl_paths_list = []
            self.pkl_paths_list.append(pkl_path)
            self.save_pkl_paths()

            # # IL_plan_idx
            # self._convert_trajectory()
            # target_traj = self._traj_info[SD.POSITION][self.current_step + 1, :2]
            # all_traj_points = self.center_traj[:, ::5, :2]
            # all_traj_points = all_traj_points[:,1,:2]
            # losses = np.sum((all_traj_points - target_traj) ** 2, axis=1)  # shape: [8192]
            # best_idx = np.argmin(losses)
            # score_row.update({
            #     'IL_plan_idx': best_idx
            # })
            # self.openloop_pdm_scores_list.append(score_row)
            # self.save_openloop_pdm_scores()


    def compute_pdm_scores(self, pdm_closed_trajectory):
        """Compute PDM scores"""
        initial_ego_state = self.ego_states_list[self.current_step]
        initial_observation = self.observations_list[self.current_step]

        # 1. Build PDM observation
        self._pdm_closed._load_route_dicts(self._cached_roadblock_ids)
        self._pdm_closed._observation.update(
            initial_ego_state,
            initial_observation,
            self.planner_input.traffic_light_data,
            self._pdm_closed._route_lane_dict,
        )
        self._pdm_closed._drivable_area_map = PDMDrivableMap.from_simulation(
            self._map_api, initial_ego_state, self._map_radius
        )

        # Interpolate observations and traffic light data
        interpolated_observations = self._interpolate_observations(
            self.observations_list[self.current_step:self.current_step + self.buffer_size]
        )
        interpolated_traffic_light_data = [[] for _ in range(len(interpolated_observations))]

        self._pdm_closed._observation.update_detections_tracks(
            interpolated_observations,
            interpolated_traffic_light_data,
            self._pdm_closed._route_lane_dict,
            compute_traffic_light_data=True,
        )

        # 2. Centerline extraction
        current_lane = self._pdm_closed._get_starting_lane(initial_ego_state)
        centerline_discrete_path = self._pdm_closed._get_discrete_centerline(current_lane)
        self._centerline = PDMPath(centerline_discrete_path)


        # 3. Set Trajectory
        n = self.gt_array.shape[0]
        curr = np.zeros((n, 1, 3))
        curr[:, 0, :2] = initial_ego_state.rear_axle.array
        curr[:, 0, 2] = initial_ego_state.rear_axle.heading
        pred_trajectory = self.batched_global_transform(self.gt_array, initial_ego_state.rear_axle.array, initial_ego_state.rear_axle.heading)
        pred_states = np.concatenate([curr, pred_trajectory], axis=1)
        
        if self.use_cuda:
            torch_pred_states = torch.from_numpy(pred_states).double()  # convert to torch.Tensor with float64
            torch_pred_states = torch_pred_states.unsqueeze(0)  # add batch dimension, shape becomes [1, 8192, 41, 3]
            simulator = TorchSimulator(proposal_sampling = TrajectorySampling(num_poses=self.engine.global_config['reward_sampling_poses'] * 5, interval_length=0.1)) 
            new_trajectory_states = simulator.simulate_proposals(torch_pred_states, initial_ego_state, batch_sim=True)
            new_trajectory_states = new_trajectory_states.squeeze(0)
            new_trajectory_states = new_trajectory_states.cpu().numpy()
        else:
            simulator = PDMSimulator(proposal_sampling = TrajectorySampling(num_poses=self.engine.global_config['reward_sampling_poses'] * 5, interval_length=0.1))
            new_trajectory_states = simulator.simulate_proposals(pred_states, initial_ego_state, batch_sim=True)

        simulate_states = new_trajectory_states

        # 4. Score proposals
        time_start = time.time()
        simulate_states_chunks = np.array_split(simulate_states, self._num_scoring_workers)

        chunks = [
            np.concatenate([[pdm_closed_trajectory], chunk], axis=0)
            for chunk in simulate_states_chunks
        ]

        worker_fn = partial(
            _score_proposals_impl,
            initial_ego_state=initial_ego_state,
            observation=self._pdm_closed._observation,
            centerline=self._centerline,
            route_lane_dict=self._pdm_closed._route_lane_dict,
            drivable_area_map=self._pdm_closed._drivable_area_map,
            map_api=self._map_api,
            scorer=self._pdm_closed._scorer,
        )

        results = self._scoring_pool.map(worker_fn, chunks)

        # merge results
        merged_metrics = {
            'no_at_fault_collisions': np.concatenate([r['no_at_fault_collisions'] for r in results]),
            'drivable_area_compliance': np.concatenate([r['drivable_area_compliance'] for r in results]),
            'lane_keeping': np.concatenate([r['lane_keeping'] for r in results]),
            'ego_progress': np.concatenate([r['ego_progress'] for r in results]),
            'time_to_collision_within_bound': np.concatenate([r['time_to_collision_within_bound'] for r in results]),
            'comfort': np.concatenate([r['comfort'] for r in results]),
            'driving_direction_compliance': np.concatenate([r['driving_direction_compliance'] for r in results]),
            'score': np.concatenate([r['score'] for r in results])
        }
        time_end = time.time()
        logger.info(f"Time taken: {time_end - time_start:.3f} seconds")
        return merged_metrics
    
    def save_openloop_pdm_scores(self):
        """Save PDM scores to CSV file"""
        if not self.openloop_pdm_scores_list:
            logger.warning("No PDM scores to save.")
            return
        
        averages_df = pd.DataFrame(self.openloop_pdm_scores_list)
        # overall_averages = averages_df.drop('token', axis=1).mean()
        # overall_averages_dict = overall_averages.to_dict()
        # overall_averages_dict['token'] = 'overall_average'
        # averages_df = pd.concat([averages_df, pd.DataFrame([overall_averages_dict])], ignore_index=True)
        averages_csv_path = os.path.join(self.engine.global_config['data_output_dir'], 
                                        'IL_plan_idx.csv')
        averages_df.to_csv(averages_csv_path, index=False)
        logger.info(f"Saved IL_plan_idx for {len(self.openloop_pdm_scores_list)} idx")
    
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
        """Get current roadblock id"""
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
    
    def batched_global_transform(self, input_array, origin_xy, theta):
        '''
        batched global transform:
        input_array [N, T, 3] (x, y, theta)
        origin_xy: [2], theta []
        '''
        x, y = input_array[:, :, 0], input_array[:, :, 1]
        curr_x, curr_y = np.array(origin_xy[0]).reshape(1, 1), np.array(origin_xy[1]).reshape(1, 1)
        curr_angle = np.array(-theta).reshape(1, 1)
        cos, sin = np.cos(curr_angle), np.sin(curr_angle)
        
        new_x = cos * x + sin * y + curr_x
        new_y = -sin * x + cos * y + curr_y 
        new_theta = wrap_to_pi(input_array[:, :, 2] - curr_angle)

        return np.stack([new_x, new_y, new_theta], axis=2)
    
    def save_pkl_paths(self):
        """ Save 8192 traj PDM scores info to CSV file """
        if not hasattr(self, 'pkl_paths_list') or not self.pkl_paths_list:
            logger.info("No 8192 trsj PDM scores to save.")
            return

        data = {
        "token": self.current_scene["token"],
        "step": self.current_step,
        "pkl_path": self.pkl_paths_list[-1]
        }
        new_df = pd.DataFrame(data, index=[0])
        self.pkl_paths_df = pd.concat([self.pkl_paths_df, new_df], ignore_index=True)

        if self.engine.global_config['agent_policy'] == 'idm_policy':
            pkl_paths_csv_path = os.path.join(self.engine.global_config['data_output_dir'], 
                                         "all_scenes_pdm_pkl_paths_R.csv")
        else:
            pkl_paths_csv_path = os.path.join(self.engine.global_config['data_output_dir'], 
                                         "all_scenes_pdm_pkl_paths_NR.csv")
    
        self.pkl_paths_df.to_csv(pkl_paths_csv_path, index=False)
    
    def _interpolate_observations(self, observations_list):
        """Interpolate observations from 0.5s interval to 0.1s interval
        
        Args:
            observations_list: original 0.5s interval observations list
            
        Returns:
            interpolated_observations: interpolated observations list with 0.1s interval
        """
        if not observations_list:
            return []
            
        # calculate the number of points to interpolate
        num_original = len(observations_list)
        num_interpolated = (num_original - 1) * 5 + 1  # insert 4 points between each interval
        
        interpolated_observations = []
        
        for i in range(num_original - 1):
            # add original points
            interpolated_observations.append(observations_list[i])
            
            # interpolate between adjacent points
            for j in range(1, 5):
                # linear interpolation
                alpha = j / 5.0
                interpolated_observation = self._interpolate_single_observation(
                    observations_list[i], 
                    observations_list[i + 1], 
                    alpha
                )
                interpolated_observations.append(interpolated_observation)
                
        # add the last point
        interpolated_observations.append(observations_list[-1])
        
        return interpolated_observations
        
    def _interpolate_single_observation(self, obs1, obs2, alpha):
        """Interpolate a single observation
        
        Args:
            obs1: first observation (DetectionsTracks)
            obs2: second observation (DetectionsTracks)
            alpha: interpolation coefficient (0-1)
            
        Returns:
            interpolated_obs: interpolated observation (DetectionsTracks)
        """
        # Create new TrackedObjects list
        interpolated_tracked_objects = []
        
        # Get tracked_objects from two observations
        tracked_objects1 = obs1.tracked_objects.tracked_objects
        tracked_objects2 = obs2.tracked_objects.tracked_objects
        
        # Create tracked_objects mapping
        tracked_objects2_dict = {obj.track_token: obj for obj in tracked_objects2}
        
        for obj1 in tracked_objects1:
            obj2 = tracked_objects2_dict.get(obj1.track_token)
            if obj2 is None:
                # If object does not exist in second observation, use first observation
                interpolated_tracked_objects.append(obj1)
                continue
            
            # Interpolate position
            pos1 = obj1.center.array
            pos2 = obj2.center.array
            interpolated_pos = pos1 + alpha * (pos2 - pos1)
            
            # Interpolate heading (using angle interpolation)
            heading1 = obj1.center.heading
            heading2 = obj2.center.heading
            # Handle angle crossing 2π
            heading_diff = (heading2 - heading1 + np.pi) % (2 * np.pi) - np.pi
            interpolated_heading = heading1 + alpha * heading_diff
            
            # Update center and box
            interpolated_obj_center = StateSE2(
                x=interpolated_pos[0],
                y=interpolated_pos[1],
                heading=interpolated_heading
            )
            interpolated_obj_box = OrientedBox(
                center=interpolated_obj_center,
                length=obj1.box.length,
                width=obj1.box.width,
                height=obj1.box.height
            )
            
            # If dynamic object (Agent), also interpolate velocity
            if hasattr(obj1, 'velocity'):
                vel1 = obj1.velocity.array
                vel2 = obj2.velocity.array
                interpolated_vel = vel1 + alpha * (vel2 - vel1)
                interpolated_obj_velocity = StateVector2D(
                    x=interpolated_vel[0],
                    y=interpolated_vel[1]
                )
            timestamp = obj1.metadata.timestamp_us + alpha * 5.0 * 0.1 * 1e6
            # Create new TrackedObject
            if isinstance(obj1, Agent):
                interpolated_obj = Agent(
                    tracked_object_type=obj1.tracked_object_type,
                    oriented_box=interpolated_obj_box,
                    velocity=interpolated_obj_velocity,
                    metadata=SceneObjectMetadata(
                        timestamp_us = timestamp,
                        token = obj1.metadata.token,
                        track_id = obj1.metadata.track_id,
                        track_token = obj1.metadata.track_token
                    )
                )
            else:
                interpolated_obj = StaticObject(
                    tracked_object_type=obj1.tracked_object_type,
                    oriented_box=interpolated_obj_box,
                    metadata=SceneObjectMetadata(
                        timestamp_us = timestamp,
                        token = obj1.metadata.token,
                        track_id = obj1.metadata.track_id,
                        track_token = obj1.metadata.track_token
                    )
                )
            # Append to the tracked objects list
            interpolated_tracked_objects.append(interpolated_obj)
        
        # Wrap in TrackedObjects
        interpolated_tracked_objects_container = TrackedObjects(tracked_objects=interpolated_tracked_objects)
        
        # Wrap in DetectionsTracks
        interpolated_obs = DetectionsTracks(tracked_objects=interpolated_tracked_objects_container)
        
        return interpolated_obs

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

def _score_proposals_impl(
    simulate_states_chunk,
    initial_ego_state,
    observation,
    centerline,
    route_lane_dict,
    drivable_area_map,
    map_api,
    scorer,
) -> Dict:

    scores = scorer.score_proposals(
        simulate_states_chunk,
        initial_ego_state,
        observation,
        centerline,
        route_lane_dict,
        drivable_area_map,
        map_api,
        batch=True
    )
    
    pred_idx = 1
    metrics = {
        'no_at_fault_collisions': scorer._multi_metrics[MultiMetricIndex.NO_COLLISION, pred_idx:].astype(bool),
        'drivable_area_compliance': scorer._multi_metrics[MultiMetricIndex.DRIVABLE_AREA, pred_idx:].astype(bool),
        'lane_keeping': scorer._weighted_metrics[WeightedMetricIndex.LANE_KEEPING, pred_idx:],
        'ego_progress': scorer._weighted_metrics[WeightedMetricIndex.PROGRESS, pred_idx:],
        'time_to_collision_within_bound': scorer._weighted_metrics[WeightedMetricIndex.TTC, pred_idx:],
        'comfort': scorer._weighted_metrics[WeightedMetricIndex.COMFORTABLE, pred_idx:],
        'driving_direction_compliance': scorer._weighted_metrics[WeightedMetricIndex.DRIVING_DIRECTION, pred_idx:],
        'score': scores[pred_idx:]
    }
    return metrics
