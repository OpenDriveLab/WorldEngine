from typing import Dict, List
from pathlib import Path
import numpy as np
import os
import time
import scipy.spatial.transform
import pickle
import logging
from pyquaternion import Quaternion

from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters
from nuplan.common.actor_state.state_representation import (
    StateSE2,
    StateVector2D,
    TimePoint,
)

from worldengine.engine.engine_utils import get_engine
from worldengine.common.dataclasses import Trajectory
from worldengine.scenario.scenarios.scenario_description import (
    ScenarioDescription as SD,
)
from worldengine.components.agents.client.base_client import BaseClient

logger = logging.getLogger(__name__)

FPS = 2
FRAME_INTERVAL = 1
FPS_KEYFRAME = FPS / FRAME_INTERVAL
WE_root = os.environ.get("WORLDENGINE_ROOT")


def find_unique_common_from_lists(input_list1, input_list2, only_com=False):
    """
    find common items from 2 lists, the returned elements are unique. repetitive items will be ignored
    if the common items in two elements are not in the same order, the outputs follows the order in the first list

    parameters:
            input_list1, input_list2:		two input lists
            only_com:		True if only need the common list, i.e., the first output, saving computational time

    outputs:
            list_common:	a list of elements existing both in list_src1 and list_src2
            index_list1:	a list of index that list 1 has common items
            index_list2:	a list of index that list 2 has common items
    """
    set1 = set(input_list1)
    set2 = set(input_list2)
    common_set = set1 & set2

    if only_com:
        return list(common_set)

    index_list1 = [i for i, item in enumerate(input_list1) if item in common_set]
    index_list2 = [i for i, item in enumerate(input_list2) if item in common_set]

    return list(common_set), index_list1, index_list2


class NAVFormerClient(BaseClient):
    def __init__(self, agent):
        self.agent = agent
        self.config = self.engine.global_config
        self.episode_data_processed = {}
        self.seq_index = []
        self.data_folder = Path(self.config["planner_client_folder"])
        self._traj_info = self.agent.object_track

    def get_trajectory(self, step: int):
        # for history frames, return log replay trajectory
        if step < self.config['num_history'] - 1:
            return Trajectory(
                waypoints=self._traj_info[SD.POSITION][step : step + 9, :2],
                velocities=self._traj_info["velocity"][step : step + 9],
                headings=self._traj_info[SD.HEADING][step : step + 9],
                angular_velocities=self._traj_info["angular_velocity"][step : step + 9],
            )

        current_scene = self.engine.current_scene
        parts = current_scene["id"].split("-")
        if len(parts[-1]) == 3:
            prefix = "-".join(
                parts[-2:]
            )  # for synthetic data, e.g. bb4f37403cea5b0e-001
        else:
            prefix = parts[-1]  # for original data, e.g. bb4f37403cea5b0e

        traj_file_path = os.path.join(
            self.config["planner_data_path"], f"{prefix}_{step + 1}.npy"
        )

        # Start waiting NAVFormer output
        while not os.path.exists(traj_file_path):
            time.sleep(0.2)

        while True:
            try:
                ego_traj = np.load(traj_file_path)
                break
            except Exception:    # in case the file is not ready yet
                time.sleep(0.1)

        ego_traj = np.concatenate([np.zeros((1, 3)), ego_traj], axis=0)
        ego_heading = ego_traj[:, 2]
        ego_xy = ego_traj[:, :2]
        ego2local_translation = self.agent.rear_vehicle.current_position
        ego2local_rotation = np.array(
            [
                [
                    np.cos(self.agent.current_heading),
                    -np.sin(self.agent.current_heading),
                ],
                [
                    np.sin(self.agent.current_heading),
                    np.cos(self.agent.current_heading),
                ],
            ]
        )
        local_traj = ego_xy @ ego2local_rotation.T + ego2local_translation  # rear_axle

        local_heading = ego_heading + self.agent.current_heading

        local_velocities = (local_traj[1:] - local_traj[:-1]) / 0.1
        local_velocities = np.vstack([local_velocities, local_velocities[-1]])

        local_angular_velocities = (local_heading[1:] - local_heading[:-1]) / 0.1
        local_angular_velocities = np.append(
            local_angular_velocities, local_angular_velocities[-1]
        )

        waypoints = []

        for i, local_traj_point in enumerate(local_traj):
            ego_state = EgoState.build_from_rear_axle(
                StateSE2(local_traj_point[0], local_traj_point[1], local_heading[i]),
                tire_steering_angle=0.0,
                vehicle_parameters=get_pacifica_parameters(),
                time_point=TimePoint(0.0),
                rear_axle_velocity_2d=StateVector2D(
                    local_velocities[i, 0], local_velocities[i, 1]
                ),
                rear_axle_acceleration_2d=StateVector2D(x=0.0, y=0.0),
            )
            waypoint = [ego_state.waypoint.x, ego_state.waypoint.y]
            waypoints.append(waypoint)

        waypoints = np.array(waypoints)
        local_velocities = (waypoints[1:] - waypoints[:-1]) / 0.1
        local_velocities = np.vstack([local_velocities, local_velocities[-1]])

        local_angular_velocities = (local_heading[1:] - local_heading[:-1]) / 0.1
        local_angular_velocities = np.append(
            local_angular_velocities, local_angular_velocities[-1]
        )

        return Trajectory(
            waypoints=waypoints[::5],
            velocities=local_velocities[::5],
            headings=local_heading[::5],
            angular_velocities=local_angular_velocities[::5],
        )

    def process_frame(self, frame_data, step: int):
        frame_data = self._postprocess_frame_data(frame_data)
        self.episode_data_processed[step] = frame_data
        self.seq_index.append(step)
        self.episode_data_processed = self.parse_clip(
            self.episode_data_processed, self.seq_index
        )
        filename = frame_data["log_token"] + "_" + str(step) + ".pkl"
        filepath = self.data_folder / filename
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "wb") as f:
            pickle.dump(
                self.episode_data_processed[step], f, protocol=pickle.HIGHEST_PROTOCOL
            )

    @property
    def engine(self):
        return get_engine()

    def _postprocess_frame_data(self, frame_data):
        """
        Postprocess frame data
        """
        global_track_id = 1
        mapping_tracktoken2globalid = dict()

        frame_data, global_track_id, mapping_tracktoken2globalid = self.parse_frame(
            frame_data,
            global_track_id=global_track_id,
            mapping_tracktoken2globalid=mapping_tracktoken2globalid,
        )

        return frame_data

    def parse_frame(
        self,
        anno: Dict,
        global_track_id: int,
        mapping_tracktoken2globalid: Dict[str, int],
    ) -> Dict:
        """Convert carla data into the collection of path needed by mmdetection3D dataloader"""

        anno: Dict = self.parse_ego_sensor_calib(anno)

        anno, global_track_id, mapping_tracktoken2globalid = self.parse_bbox(
            anno,
            global_track_id=global_track_id,
            mapping_tracktoken2globalid=mapping_tracktoken2globalid,
        )

        return anno, global_track_id, mapping_tracktoken2globalid

    def parse_ego_sensor_calib(self, anno: Dict) -> Dict:
        """Parse calibration between world, ego, lidar and camera"""

        # from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters
        x, y, z = 0.0, 0.0, 0.0
        l, w, h = 5.176, 2.297, 1.777
        sdc_vel_ego = anno["can_bus"][10:13]
        sdc_vel_global = anno["ego2global"][:3, :3] @ sdc_vel_ego
        sdc_vel_lidar = np.linalg.inv(anno["lidar2ego"][:3, :3]) @ sdc_vel_ego

        gt_sdc_bbox_lidar = np.array([
                x, y, z,
                l, w, h,
                0.,
                sdc_vel_lidar[0], sdc_vel_lidar[1],
            ], dtype=np.float64)

        # load camera calib
        cams = anno["cams"]
        for cam_type, cam_dict in cams.items():
            sensor2lidar = np.identity(4)
            sensor2lidar[:3, :3] = cam_dict["sensor2lidar_rotation"]
            sensor2lidar[:3, 3] = cam_dict["sensor2lidar_translation"]
            sensor2ego: np.ndarray = anno["lidar2ego"] @ sensor2lidar

            # construct nuScenes-like camera record
            cams[cam_type].update(
                {
                    "type": cam_type,
                    "sensor2ego_translation": sensor2ego[:3, 3],
                    "sensor2ego_rotation": Quaternion(matrix=sensor2ego[:3, :3]).elements,
                }
            )

        command = np.argmax(anno["driving_command"])  # int
        # update dictionary
        anno.update(
            {
                "sweeps": [],
                "cams": cams,
                "sdc_vel_global": sdc_vel_global,
                # sdc's information in lidar coordinate
                "gt_sdc_bbox_lidar": gt_sdc_bbox_lidar,  # 9
                "command": command,  # int, convert 1-indexed -> 0-indexed
            }
        )

        return anno

    def parse_bbox(
        self,
        anno: Dict,
        global_track_id: int,
        mapping_tracktoken2globalid: Dict[str, int],
    ):
        """Parse each one of the bounding boxes and compute properties"""

        gt_boxes = np.array(anno["anns"].get("gt_boxes", []))
        if len(gt_boxes.shape) == 1:
            gt_boxes = gt_boxes.reshape(0, 7)
        num_obj = gt_boxes.shape[0]

        gt_velocity_3d = np.array(anno["anns"].get("gt_velocity_3d", []))
        if len(gt_velocity_3d.shape) == 1:
            gt_velocity_3d = gt_velocity_3d.reshape(0, 3)

        gt_names = np.array(anno["anns"].get("gt_names", []))
        anno.update(
            {
                "gt_velocity": gt_velocity_3d[:, :2],  # lidar coordinate
                "gt_boxes": gt_boxes,  # lidar coordinate (x,y,z,l,w,h,yaw)
                "gt_bboxes_global": np.zeros((num_obj, 9), dtype=np.float64),
                "gt_names": gt_names if num_obj > 0 else np.array([], dtype=str),
                "gt_inds": np.zeros((num_obj,), dtype=int),
                "num_lidar_pts": np.zeros((num_obj,), dtype=int),
                "valid_flag": np.ones((num_obj,), dtype=bool),
            }
        )

        ego_yaw = scipy.spatial.transform.Rotation.from_matrix(
            anno["lidar2global"][:3, :3]
        ).as_euler("zyx", degrees=False)[0]
        lidar2global = anno["lidar2global"]

        box_7dof_lidar = anno["anns"]["gt_boxes"]
        box_velo_3d = anno["anns"]["gt_velocity_3d"]

        bbox_center_global = (
            box_7dof_lidar[:, :3] @ lidar2global[:3, :3].T + lidar2global[:3, 3]
        )
        bbox_yaw_world = box_7dof_lidar[:, -1] + ego_yaw
        bbox_velocity_world = box_velo_3d @ lidar2global[:3, :3].T
        anno["gt_bboxes_global"][:, :3] = bbox_center_global
        anno["gt_bboxes_global"][:, 3:6] = box_7dof_lidar[:, 3:6]
        anno["gt_bboxes_global"][:, 6] = bbox_yaw_world
        anno["gt_bboxes_global"][:, 7:] = bbox_velocity_world[:, :2]

        for bbox_index in range(num_obj):
            track_token = anno["anns"]["track_tokens"][bbox_index]
            # assign global ID
            if track_token not in mapping_tracktoken2globalid:
                mapping_tracktoken2globalid[track_token] = global_track_id
                anno["gt_inds"][bbox_index] = global_track_id

                # update track_id for the next object
                global_track_id += 1
            else:
                anno["gt_inds"][bbox_index] = mapping_tracktoken2globalid[track_token]

        return anno, global_track_id, mapping_tracktoken2globalid

    def parse_clip(
        self,
        annos: List[Dict],
        seq_index: List[int],
        pre_sec: float = 1.5,
        fut_sec: float = 4,
    ) -> List[Dict]:
        """Add past/future trajectory into data at each frame to allow prediction"""

        anno_seq: List[Dict] = [annos[index] for index in seq_index]

        # sort data by frame ID, as the original data might not be sorted based
        # on frame index
        def sort_data(data_dict: Dict):
            return data_dict["frame_idx"]

        anno_seq.sort(key=sort_data)

        # check if the sequence has non-continuous frames, if so, remove the data
        min_frame_ID: int = anno_seq[0]["frame_idx"]
        max_frame_ID: int = anno_seq[-1]["frame_idx"]
        if len(anno_seq) < max_frame_ID - min_frame_ID + 1:
            for index in seq_index:
                annos[index] = None
            return annos

        # loop through each frame in this sequence
        for index in range(len(seq_index)):
            anno: Dict = anno_seq[index]  # data for the frame
            frame_ID: int = anno["frame_idx"]  # the frame ID
            # print("current frame is", frame_ID)

            # retrieve info for other objects
            # gt_velocity: np.ndarray = anno["gt_velocity"]  # N x 2
            gt_bboxes_lidar: np.ndarray = np.array(anno["anns"]["gt_boxes"])  # N x 7
            num_obj: int = gt_bboxes_lidar.shape[0]

            # convert the str ID to global ID
            obj_ids: List[int] = anno["gt_inds"].tolist()

            # retrieve info for ego and calib
            l, w, h = anno["gt_sdc_bbox_lidar"][3:6]
            lidar2global: np.ndarray = anno["lidar2global"]  # 4 x 4
            global2lidar: np.ndarray = np.eye(4)
            global2lidar[:3, :3] = lidar2global[:3, :3].T
            global2lidar[:3, 3] = -lidar2global[:3, :3].T @ lidar2global[:3, 3]

            lidar2global_yaw = scipy.spatial.transform.Rotation.from_matrix(
                lidar2global[:3, :3]
            ).as_euler("zyx", degrees=False)[0]

            ########### get the target data we need for GT
            pre_frames: int = int(pre_sec * FPS_KEYFRAME)
            fut_frames: int = int(fut_sec * FPS_KEYFRAME)

            # the past & future trajectories for objects existing in the current frame
            # so we know the exact number of objects
            gt_fut_bbox_lidar: np.ndarray = np.zeros((num_obj, fut_frames, 9), dtype=np.float64)
            gt_pre_bbox_lidar: np.ndarray = np.zeros((num_obj, pre_frames, 9), dtype=np.float64)

            # initially set all mask as invalid until we identify valid frames
            gt_fut_bbox_mask: np.ndarray = np.zeros((num_obj, fut_frames, 1), dtype=bool)
            gt_pre_bbox_mask: np.ndarray = np.zeros((num_obj, pre_frames, 1), dtype=bool)

            # sdc's temporal info
            gt_pre_bbox_sdc_lidar: np.ndarray = np.zeros((1, pre_frames, 9), dtype=np.float64)
            gt_fut_bbox_sdc_lidar: np.ndarray = np.zeros((1, fut_frames, 9), dtype=np.float64)
            gt_pre_bbox_sdc_global: np.ndarray = np.zeros((1, pre_frames, 9), dtype=np.float64)
            gt_fut_bbox_sdc_global: np.ndarray = np.zeros((1, fut_frames, 9), dtype=np.float64)
            gt_pre_bbox_sdc_mask: np.ndarray = np.zeros((1, pre_frames, 1), dtype=bool)
            gt_fut_bbox_sdc_mask: np.ndarray = np.zeros((1, fut_frames, 1), dtype=bool)
            gt_pre_command_sdc: np.ndarray = np.zeros((1, pre_frames, 1), dtype=int)
            gt_fut_command_sdc: np.ndarray = np.zeros((1, fut_frames, 1), dtype=int)

            # get past trajectory, backward order, i.e., frame 4, 3, 2, 1
            # start with 1 to not include the current frame
            for pre_frame_index in range(1, pre_frames + 1):
                pre_frame_ID = int(frame_ID - pre_frame_index * FRAME_INTERVAL)
                pre_frame_index_in_seq: int = index - pre_frame_index

                if pre_frame_ID < min_frame_ID:
                    continue

                # retrieve the data in the global coordinate
                anno_pre_tmp: Dict = anno_seq[pre_frame_index_in_seq]
                gt_pre_bbox_global_tmp: np.ndarray = anno_pre_tmp[
                    "gt_bboxes_global"
                ]  # N x 9
                num_obj_pre: int = gt_pre_bbox_global_tmp.shape[0]
                gt_pre_bbox_lidar_tmp = np.zeros((num_obj_pre, 9))

                ########## convert the global coordinate to lidar coordinate in current frame
                # i.e., ego motion compensation

                # location
                gt_pre_bbox_lidar_tmp[:, :3] = gt_pre_bbox_global_tmp[:, :3] @ global2lidar[:3, :3].T + global2lidar[:3, 3]
                # size
                gt_pre_bbox_lidar_tmp[:, 3:6] = gt_pre_bbox_global_tmp[:, 3:6]
                # rotation (yaw)
                gt_pre_bbox_lidar_tmp[:, 6] = gt_pre_bbox_global_tmp[:, 6] - lidar2global_yaw
                # velocity
                gt_pre_bbox_global_velo3d = np.concatenate([gt_pre_bbox_global_tmp[:, 7:9], np.zeros((num_obj_pre, 1))], axis=1)
                gt_pre_bbox_lidar_tmp[:, 7:] = (gt_pre_bbox_global_velo3d @ global2lidar[:3, :3].T)[:, :2]  # N x 2

                # now check the IDs that exist in the current frame
                # in order to produce the mask for future frames
                obj_ids_pre: List[int] = anno_pre_tmp["gt_inds"].tolist()
                (
                    obj_ids_common,
                    index_cur,
                    index_past,
                ) = find_unique_common_from_lists(obj_ids, obj_ids_pre)

                # possible that we get 0 object in future frame matching with GT object currint
                if len(index_cur) > 0:
                    gt_pre_bbox_mask[np.array(index_cur), pre_frame_index - 1] = 1

                    # also assign the actual gt boxes into the array
                    gt_pre_bbox_lidar[
                        np.array(index_cur), pre_frame_index - 1, :
                    ] = gt_pre_bbox_lidar_tmp[np.array(index_past), :]

                ############### get past of the sdc box in lidar coordinate
                sdc_loc_global = anno_pre_tmp["ego2global"][:3, 3]
                sdc_loc_lidar = global2lidar[:3, :3] @ sdc_loc_global + global2lidar[:3, 3]
                # velocity
                sdc_vel_global = anno_pre_tmp["sdc_vel_global"]  # 3
                sdc_vel_lidar = global2lidar[:3, :3] @ sdc_vel_global   # 3
                # rotation
                sdc_trans_lidar = global2lidar @ anno_pre_tmp["ego2global"]  # 4 x 4
                yaw_lidar = scipy.spatial.transform.Rotation.from_matrix(
                    sdc_trans_lidar[:3, :3]
                ).as_euler("zyx", degrees=False)[0]
                yaw_global = scipy.spatial.transform.Rotation.from_matrix(
                    anno_pre_tmp["ego2global"][:3, :3]
                ).as_euler("zyx", degrees=False)[0]
                gt_pre_bbox_sdc_mask[0, pre_frame_index - 1] = 1

                # put things into bbox, 9 dof
                gt_sdc_bbox_lidar = np.array([
                        sdc_loc_lidar[0], sdc_loc_lidar[1], sdc_loc_lidar[2],
                        l, w, h,
                        yaw_lidar,
                        sdc_vel_lidar[0], sdc_vel_lidar[1],
                    ], dtype=np.float64)
                gt_pre_bbox_sdc_lidar[0, pre_frame_index - 1] = gt_sdc_bbox_lidar

                gt_sdc_bbox_global = np.array([
                        sdc_loc_global[0], sdc_loc_global[1], sdc_loc_global[2],
                        l, w, h,
                        yaw_global,
                        sdc_vel_global[0], sdc_vel_global[1],
                    ], dtype=np.float64)
                gt_pre_bbox_sdc_global[0, pre_frame_index - 1] = gt_sdc_bbox_global

                # get command
                gt_pre_command_sdc[0, pre_frame_index - 1]: int = anno_pre_tmp["command"]

            # flip array to allow time in forward order for the past trajectory
            gt_pre_bbox_lidar = np.flip(gt_pre_bbox_lidar, axis=1)
            gt_pre_bbox_mask = np.flip(gt_pre_bbox_mask, axis=1)
            gt_pre_bbox_sdc_lidar = np.flip(gt_pre_bbox_sdc_lidar, axis=1)
            gt_pre_bbox_sdc_global = np.flip(gt_pre_bbox_sdc_global, axis=1)
            gt_pre_bbox_sdc_mask = np.flip(gt_pre_bbox_sdc_mask, axis=1)
            gt_pre_command_sdc = np.flip(gt_pre_command_sdc, axis=1)

            # assign the value back into the dictionary
            # in-place assignment to the sorted anno
            anno["gt_pre_bbox_lidar"] = gt_pre_bbox_lidar.astype(np.float32)  # N x 3 x 9
            anno["gt_fut_bbox_lidar"] = gt_fut_bbox_lidar.astype(np.float32)  # N x 8 x 9
            anno["gt_pre_bbox_mask"] = gt_pre_bbox_mask.astype(bool)  # N x 3 x 1
            anno["gt_fut_bbox_mask"] = gt_fut_bbox_mask.astype(bool)  # N x 8 x 1
            anno["gt_pre_bbox_sdc_lidar"] = gt_pre_bbox_sdc_lidar.astype(np.float32)  # 1 x 3 x 9
            anno["gt_fut_bbox_sdc_lidar"] = gt_fut_bbox_sdc_lidar.astype(np.float32)  # 1 x 8 x 9
            anno["gt_pre_bbox_sdc_global"] = gt_pre_bbox_sdc_global.astype(np.float64)  # 1 x 3 x 9
            anno["gt_fut_bbox_sdc_global"] = gt_fut_bbox_sdc_global.astype(np.float64)  # 1 x 8 x 9
            anno["gt_pre_bbox_sdc_mask"] = gt_pre_bbox_sdc_mask.astype(bool)  # 1 x 3 x 1
            anno["gt_fut_bbox_sdc_mask"] = gt_fut_bbox_sdc_mask.astype(bool)  # 1 x 8 x 1
            anno["gt_pre_command_sdc"] = gt_pre_command_sdc.astype(int)  # 1 x 3 x 1
            anno["gt_fut_command_sdc"] = gt_fut_command_sdc.astype(int)  # 1 x 8 x 1

        return annos
