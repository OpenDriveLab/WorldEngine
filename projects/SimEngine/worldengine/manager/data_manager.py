import os
from typing import Dict, List
import uuid
import pickle
import numpy as np
from pathlib import Path
from pyquaternion import Quaternion
import cv2
import scipy
import copy

from worldengine.manager.base_manager import BaseManager
from worldengine.scenario.scenarios.scenario_description import ScenarioDescription as SD
from worldengine.utils.agent_utils import Video_visualizer, SceneDataSaver

import logging
logger = logging.getLogger(__name__)

FPS = 2
FRAME_INTERVAL = 1
FPS_KEYFRAME = FPS / FRAME_INTERVAL
WORLDENGINE_ROOT = os.getenv('WORLDENGINE_ROOT', os.path.abspath('.'))

class DataManager(BaseManager):
    PRIORITY = 8

    def __init__(self):
        super().__init__()
        self.output_dir = Path(self.engine.global_config.get('data_output_dir', f'{WORLDENGINE_ROOT}/data_output'))
        self.episode_data = {}
        self.episode_data_processed = {}
        self.seq_index = []
        self._video_visualizer = None
        self._data_saver = None

    def _get_current_frame_data(self):
        """
        getting current frame data
        Returns:
            frame_data: dict of navsim metadata.
        """

        agent_manager = self.engine.agent_manager
        current_scene = self.engine.current_scene
        ego_vehicle = self.engine.agent_manager.ego_agent
        current_scene_id = current_scene[SD.ID]
        if len(current_scene_id.split('-')[-1]) == 3:
            suffix = current_scene_id.split('-')[-1]
        else:
            suffix = '000'

        # get original frame data
        openscene_info_dict = current_scene["metadata"]["openscene_data_infos_dict"]
        frame_idx = self.engine.episode_step
        frame_data = None
        for i, (_, original_frame_data) in enumerate(openscene_info_dict.items()):
            if frame_idx == i:
                frame_data = copy.deepcopy(original_frame_data)
                break
        
        # Token related fields
        if frame_data is not None:
            original_token = frame_data['token']
            # generate new token: original_token-reproduction_idx
            new_token = f"{original_token}-{suffix}" # 2cbf505c735c5c34-000
            parts = current_scene_id.split('-')
            if len(parts[-1]) == 3:
                new_scene_token = '-'.join(parts[-2:]) # for synthetic data, e.g. bb4f37403cea5b0e-001
            else:
                new_scene_token = parts[-1] # for original data, e.g. bb4f37403cea5b0e

        if 'synthetic_scene_info' in current_scene[SD.METADATA]:
            synthetic_frame = current_scene[SD.METADATA]['synthetic_scene_info']['frames'][frame_idx]
            new_token = synthetic_frame['token']
            new_scene_token = current_scene[SD.METADATA]['synthetic_scene_info']['scene_metadata']['scene_token']

        # time_stamp related
        base_timestamp = current_scene.get("base_timestamp", 0)
        time_stamp = base_timestamp + int(frame_idx * current_scene["sample_rate"] * 0.05 * 1e6)

        render_results = self.engine.managers['render_manager'].rendering_results[frame_idx]
        ego2global = render_results['ego2global']
        ego2global_translation = ego2global[:3, 3]
        ego2global_rotation = Quaternion(matrix=ego2global[:3, :3]).elements

        frame_data['ego2global_translation'] = ego2global_translation
        frame_data['ego2global_rotation'] = ego2global_rotation
        frame_data['ego2global'] = ego2global
        frame_data['lidar2global'] = ego2global @ frame_data['lidar2ego']

        # loc: [0:3], quat: [3:7], accel: [7:10], velocity: [10:13], rotation_rate: [13:16]
        acc_global = ego_vehicle.rear_vehicle.current_acceleration
        velo_global = ego_vehicle.current_velocity
        ego_heading = ego_vehicle.rear_vehicle.current_heading
        ego2global_rotation_2d = np.array([
            [np.cos(ego_heading), -np.sin(ego_heading)],
            [np.sin(ego_heading), np.cos(ego_heading)]
        ])
        global2ego_rotation_2d = ego2global_rotation_2d.T

        acc_ego = acc_global @ global2ego_rotation_2d.T
        velo_ego = velo_global @ global2ego_rotation_2d.T

        ego_dynamic_state = [
            velo_ego[0],  # velocity_x
            velo_ego[1],  # velocity_y
            acc_ego[0],  # acceleration_x
            acc_ego[1]   # acceleration_y
        ]

        can_bus = frame_data['can_bus']
        can_bus[0:3] = ego2global_translation
        can_bus[3:7] = ego2global_rotation
        can_bus[7:9] = acc_ego
        can_bus[10:12] = velo_ego
        can_bus[15] = ego_vehicle.rear_vehicle.current_angular_velocity


        # update all related token and name fields
        frame_data.update({
            'token': new_token,  # new frame token
            'frame_idx': frame_idx,  # new frame_idx
            'timestamp': time_stamp, # new timestamp
            'log_name': current_scene_id,  # new log name
            'log_token': new_scene_token,  # new log token
            'scene_name': current_scene_id,  # new scene name
            'scene_token': new_scene_token,  # new scene token
            'lidar_path': None,
            'ego_dynamic_state': ego_dynamic_state,
            'sample_prev': None,
            'sample_next': None,
            'can_bus': can_bus,
        })

        # update cams
        for cam_name, cam_data in frame_data['cams'].items():
            if 'synthetic_scene_info' in current_scene[SD.METADATA]:
                synthetic_frame = current_scene[SD.METADATA]['synthetic_scene_info']['frames'][frame_idx]
                data_path = synthetic_frame['camera_dict'][cam_name.lower()]['data_path']
            else:
                original_path = cam_data['data_path']
                filename = original_path.split('/')[-1].split('.')[0]
                data_path = f"{new_scene_token}/{cam_name}/{filename}-{suffix}.jpg"

            if isinstance(data_path, Path):
                data_path = data_path.as_posix()
            cam_data['data_path'] = data_path
            # get image from render_results
            img = render_results['cameras'][cam_name]['image']
            output_path = self.output_dir / 'sensor_blobs' / data_path
            output_path.parent.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(output_path.as_posix(), img):
                logger.error("problem in saving")

        # update anns
        frame_data['anns'] = self._get_annotations(
            agent_manager, 
            agent_manager.ego_agent,
            frame_data
        )

        return frame_data

    def _get_annotations(self, agent_manager, ego_vehicle, original_frame_data):
        """
        getting annotations for all agents in the scene
        Args:
            agent_manager: agent manager
            ego_vehicle: ego vehicle
            original_frame_data: original frame data for getting instance_tokens and track_tokens
        Returns:
            dict: dict with gt_boxes, gt_names, gt_velocity_3d etc.
        """

        if len(agent_manager.all_agents) <= 1:  # only ego vehicle
            return {
                'gt_boxes': np.zeros((0, 7), dtype=np.float32),
                'gt_names': np.array([], dtype=str),
                'gt_velocity_3d': np.zeros((0, 3), dtype=np.float32),
                'instance_tokens': [],
                'track_tokens': [],
                'original_track_tokens': []
            }

        gt_boxes = []
        gt_names = []
        gt_velocity_3d = []
        track_tokens = []
        # getting tokens from original data
        original_anns = original_frame_data.get('anns', {})
        original_track_tokens = original_anns.get('track_tokens', [])
        
        # getting ego position and heading for coordinate transformation
        ego_pos = ego_vehicle.rear_vehicle.current_position
        ego_heading = ego_vehicle.rear_vehicle.current_heading

        for agent_id, agent in agent_manager.all_agents.items():
            if agent.id == "ego":
                continue
            # calculating relative position for coordinate transformation
            rel_pos = agent.current_position - ego_pos
            
            # calculating relative heading for coordinate transformation
            rel_heading = agent.current_heading - ego_heading
            
            # coordinate transformation (global -> ego)
            cos_heading = np.cos(-ego_heading)
            sin_heading = np.sin(-ego_heading)
            x_ego = rel_pos[0] * cos_heading - rel_pos[1] * sin_heading
            y_ego = rel_pos[0] * sin_heading + rel_pos[1] * cos_heading
            
            # building box info (x,y,z,l,w,h,yaw)
            box = [
                x_ego,                  # x in ego frame
                y_ego,                  # y in ego frame
                0.0,                    # z 
                agent.length,           # length
                agent.width,            # width
                agent.height,           # height
                rel_heading            # relative heading
            ]
            
            # getting velocity for coordinate transformation
            if hasattr(agent, 'current_velocity'):
                vel_x = agent.current_velocity[0] * cos_heading - agent.current_velocity[1] * sin_heading
                vel_y = agent.current_velocity[0] * sin_heading + agent.current_velocity[1] * cos_heading
                velocity_3d = [vel_x, vel_y, 0.0]  # z direction velocity set to 0
            else:
                velocity_3d = [0.0, 0.0, 0.0]
            
            # get agent type
            agent_type = agent_manager.current_agent_data[agent.id]['type']
            agent_type = agent_type.lower()
            if agent_type == "traffic_barrier":
                agent_type = "barrier"
            elif agent_type == "traffic_object":
                agent_type = "generic_object"

            gt_boxes.append(box)
            gt_names.append(agent_type)
            gt_velocity_3d.append(velocity_3d)
            track_tokens.append(agent.id)
            
        gt_boxes = np.array(gt_boxes)
        # convert heading to [-pi, pi] range
        gt_boxes[:, -1] = gt_boxes[:, -1] % (2 * np.pi)
        gt_boxes[:, -1][gt_boxes[:, -1] > np.pi] -= 2 * np.pi

        return {
            'gt_boxes': gt_boxes,           # Ground truth boxes (x,y,z,l,w,h,yaw)
            'gt_names': np.array(gt_names),           # Class names
            'gt_velocity_3d': np.array(gt_velocity_3d),  # 3D velocity
            'instance_tokens': track_tokens,  # keep original instance tokens
            'track_tokens': track_tokens,      # current track tokens, aligning with gt_boxes
            'original_track_tokens': original_track_tokens       # keep original track tokens
        }

    def save_current_frame_data(self):
        if self.engine.episode_step < self.engine.current_scene['log_length']:
            frame_data = self._get_current_frame_data()
            if frame_data:
                token = frame_data['token']
                self.episode_data[token] = frame_data
                if self.engine.global_config.use_planner_actions:
                    ego_client = self.engine.managers['agent_manager'].ego_agent.client
                    ego_client.process_frame(frame_data, self.engine.episode_step)

    def after_step(self):
        return {}

    def reset(self):
        self.save_data()
        self.episode_data = {}
        self.episode_data_processed = {}
        self.seq_index = []
        if self._video_visualizer is None:
            if self.engine.global_config.visualize_video:
                self._video_visualizer = Video_visualizer()
            else:
                self._video_visualizer = None
        if self._data_saver is None:
            if self.engine.global_config.save_data:
                self._data_saver = SceneDataSaver()
            else:
                self._data_saver = None
        self.original_frame_data = copy.deepcopy(list(self.engine.current_scene["metadata"]["openscene_data_infos_dict"].values()))
        return {}

    def after_reset(self):
        return {}

    def save_data(self):
        if len(self.episode_data) == 0:
            return

        data_infos = list(self.episode_data.values())
        data_infos[0]['sample_prev'] = None
        data_infos[-1]['sample_next'] = None

        for i in range(len(data_infos)):
            data_infos[i]['sample_prev'] = data_infos[i-1]['token'] if i > 0 else None
            data_infos[i]['sample_next'] = data_infos[i+1]['token'] if i < len(data_infos) - 1 else None

        filename = data_infos[0]['log_name'] + '.pkl'
        filepath = self.output_dir / 'meta_datas' / filename
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(data_infos, f)
        if self._video_visualizer is not None and self.engine.global_config.NC_DAC_video_capture:
            metric_manager = self.engine.managers['metric_manager']
            if metric_manager.first_NC_DAC_step is not None:
                self._video_visualizer.generate_video()
            else:
                logger.info("Skipping video generation as first_NC_DAC_step is None")
        elif self._video_visualizer is not None:
            self._video_visualizer.generate_video()
        if self._data_saver is not None:
            self._data_saver.save_scene_data(data_infos[0]['scene_token'], self.original_frame_data)
