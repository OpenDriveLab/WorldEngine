import copy
import os
import pickle
from typing import Dict, List
from pathlib import Path
import lzma
import numpy as np
import pandas as pd
import yaml
from pyquaternion import Quaternion
import cv2
import torch
import mmcv
from mmcv.parallel import DataContainer as DC
from mmdet.datasets import DATASETS
from mmdet.datasets.pipelines import to_tensor
from mmdet3d.core.bbox import LiDARInstance3DBoxes
from mmdet3d.datasets.custom_3d import Custom3DDataset
from nuscenes.eval.common.utils import quaternion_yaw
from nuplan.common.maps.maps_datatypes import SemanticMapLayer

from mmdet3d_plugin.datasets.data_utils.nuplan_vector_map import VectorizedLocalMap


from mmdet3d_plugin.utils import get_logger
logger = get_logger(__name__)
WORLDENGINE_ROOT = os.getenv('WORLDENGINE_ROOT', os.path.abspath('.'))

@DATASETS.register_module()
class NavSimOpenSceneE2E(Custom3DDataset):
    r"""OpenScene E2E Dataset"""

    CLASSES = (
        "vehicle",
        "bicycle",
        "pedestrian",
        "traffic_cone",
        "barrier",
        "czone_sign",
        "generic_object",
    )
    
    def __init__(
        self,
        ann_file,
        pipeline=None,
        data_root=None,
        classes=None,
        modality=None,
        box_type_3d='LiDAR',
        filter_empty_gt=True,
        test_mode=False,
        # E2E specific parameters
        queue_length=4,
        bev_size=(200, 200),
        # tracking/planning
        past_steps=4,
        fut_steps=4,
        planning_steps=6,
        # map-related
        patch_size=(102.4, 102.4),
        canvas_size=(200, 200),
        # NavSim specific parameters
        process_perception=False,
        nav_filter_path=None,
        train_pdm_path = os.path.join(WORLDENGINE_ROOT,'data/alg_engine/pdms_cache/pdm_8192_gt_cache_navtrain'),
        test_pdm_path = os.path.join(WORLDENGINE_ROOT,'data/alg_engine/pdms_cache/pdm_8192_gt_cache_navtest'),
        history_frame_num=3,
        future_frame_num=8,
        fix_can_bus_rotation=False,
        map_root=None,
        with_velocity=True,
        use_valid_flag=False,
        file_client_args=dict(backend="disk"),
        **kwargs
    ):
        # NavSim specific parameters
        self.process_perception = process_perception
        self.future_frame_num = future_frame_num
        self.history_frame_num = history_frame_num
        if os.environ.get("NUPLAN_MAPS_ROOT") is not None:
            self.map_root = os.environ.get("NUPLAN_MAPS_ROOT")
        else:
            self.map_root = map_root
        self.fix_can_bus_rotation = fix_can_bus_rotation
        self.nav_filter_path = nav_filter_path

        if "navtrain" in os.path.basename(self.nav_filter_path):
            self.pdm_path = train_pdm_path
        elif "navtest" in os.path.basename(self.nav_filter_path):
            self.pdm_path = test_pdm_path
        else:
            self.pdm_path = None
            logger.warning(f"No pdm cache found for {os.path.basename(self.nav_filter_path)}")

        # E2E specific parameters (previously from NuScenesE2EDataset)
        self.queue_length = queue_length
        self.bev_size = bev_size
        self.with_velocity = with_velocity
        self.use_valid_flag = use_valid_flag

        # File client for data loading
        self.file_client_args = file_client_args
        self.file_client = mmcv.FileClient(**file_client_args)

        # Initialize the parent class (Custom3DDataset)
        super(NavSimOpenSceneE2E, self).__init__(
            data_root=data_root,
            ann_file=ann_file,
            pipeline=pipeline,
            classes=classes,
            modality=modality,
            box_type_3d=box_type_3d,
            filter_empty_gt=filter_empty_gt,
            test_mode=test_mode,
        )
        
        # Dataset-specific initialization
        self.init_dataset()
        
        # Task-specific initialization
        self.init_trackpredplan(
            planning_steps=planning_steps,
            past_steps=past_steps,
            fut_steps=fut_steps,
        )
        if self.process_perception:
            self.init_mapping(
                canvas_size=canvas_size, 
                patch_size=patch_size, 
                lane_ann_file=None
            )

        # Load PDM infos after data_infos is loaded
        self.load_pdm_infos()


    def load_annotations(self, ann_file):

        nav_filter = self.nav_filter_path
        with open(nav_filter, 'r') as file:
            nav_filter = yaml.safe_load(file)

        scene_filter = nav_filter.get('tokens')
        if scene_filter is None:
            scene_filter = nav_filter['scenario_tokens']
        self.scene_filter = set(scene_filter)

        # load annotations:
        logger.info('loading dataset!')
        data_infos = mmcv.load(ann_file, file_format="pkl")
        if isinstance(data_infos, dict):
            data_infos = data_infos['infos']
        else:
            assert isinstance(data_infos, list)
        logger.info(f'dataset loaded, length: {len(data_infos)}')

        self.index_map = []
        for i, info in enumerate(data_infos):
            if info['token'] in self.scene_filter:
                self.index_map.append(i)

        logger.info(f'filtering {len(data_infos)} frames to {len(self.index_map)}...')

        # use log_token in OpenScene as scene_token.
        for info in data_infos:
            info['scene_name'] = info['log_name']
            info['scene_token'] = info['log_token']

        return data_infos

    def load_pdm_infos(self):
        with open(f'{self.pdm_path}.pkl', 'rb') as f:
            self.pdm_dict = pickle.load(f)
        logger.info(f'loaded PDM score cache of {len(self.pdm_dict)} tokens.')

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        # map nav filter idx to the data_infos idx.
        new_idx = self.index_map[idx]

        if self.test_mode:
            return self.prepare_test_data(new_idx)
        while True:
            data = self.prepare_train_data(new_idx)
            if data is None:
                idx = self._rand_another(idx)
                new_idx = self.index_map[idx]
                continue
            return data

    def init_dataset(self):
        pass
    
    def prepare_train_data(self, index):
        return self.prepare_test_data(index)

    def prepare_test_data(self, index):
        """
        Training data preparation.
        Args:
            index (int): Index for accessing the target data.
        Returns:
            dict: Training data dict of the corresponding index.
                img: queue_length, 6, 3, H, W
                img_metas: img_metas of each frame (list)
                gt_globals_3d: gt_globals of each frame (list)
                gt_bboxes_3d: gt_bboxes of each frame (list)
                gt_inds: gt_inds of each frame (list)
        """
        data_queue = []
        self.enbale_temporal_aug = False
        # ensure the first and final frame in same scene
        final_index = index
        first_index = index - self.queue_length + 1
        if first_index < 0:
            return None
        if (
            self.data_infos[first_index]["scene_token"]
            != self.data_infos[final_index]["scene_token"]
        ):
            return None

        input_dict = self.get_data_info(final_index)
        if input_dict is None:
            return None

        prev_indexs_list = list(reversed(range(first_index, final_index)))
        timestamp = input_dict["timestamp"]
        scene_token = input_dict["scene_token"]
        self.pre_pipeline(input_dict)
        example = self.pipeline(input_dict)
        data_queue.insert(0, example)

        ########## retrieve previous infos, frame by frame
        for i in prev_indexs_list:

            input_dict = self.get_data_info(i, prev_frame=True)
            if input_dict is None:
                return None

            if (input_dict["timestamp"] < timestamp and input_dict["scene_token"] == scene_token):
                self.pre_pipeline(input_dict)
                example = self.pipeline(input_dict)
                timestamp = input_dict["timestamp"]
                data_queue.insert(0, copy.deepcopy(example))
            else:
                break

        # merge a sequence of data into one dictionary, only for temporal data
        if self.test_mode:
            data_queue = self.union2one_test(data_queue)
        else:
            data_queue = self.union2one(data_queue, self.data_infos[index]["token"])
        return data_queue

    def get_metric_cache(self, token):
        with lzma.open(self.metric_cache_dict[token], "rb") as f:
            metric_cache = pickle.load(f)
        return metric_cache

    def union2one(self, queue, token):
        """
        convert a sequence of the sample dict into one single sample.
        """

        # sensor data and transform
        imgs_list: List[torch.Tensor] = [
            each["img"].data for each in queue
        ]  # L, C x 3 x H x W
        l2g_r_mat_list: List[torch.Tensor] = [
            to_tensor(each["l2g_r_mat"]) for each in queue
        ]  # L, 3 x 3
        l2g_t_list: List[torch.Tensor] = [
            to_tensor(each["l2g_t"]) for each in queue
        ]  # L, 3

        # BUG: do not convert to fp32!
        # timestamp_list = [to_tensor(each["timestamp"]) for each in queue]  # L, 1
        timestamp_list = [torch.tensor([each["timestamp"]], dtype=torch.float64) for each in queue]  # L, 1

        # converting the global absolute coordinate into relative position and orientation change
        if self.process_perception:
            # class category labels for classification, e.g., 0 for Car
            gt_labels_3d_list: List[torch.Tensor] = [
                each["gt_labels_3d"].data for each in queue
            ]  # L, N
            gt_sdc_label_list: List[torch.Tensor] = [
                each["gt_sdc_label"].data for each in queue
            ]  # L, 1

            # global ID for tracking
            gt_inds_list = [to_tensor(each["gt_inds"]) for each in queue]  # L, N

            # detection
            gt_bboxes_3d_list: List[LiDARInstance3DBoxes] = [
                each["gt_bboxes_3d"].data for each in queue
            ]  # L, N x 9

            gt_sdc_bbox_list: List[LiDARInstance3DBoxes] = [
                each["gt_sdc_bbox"].data for each in queue
            ]  # L, 1 x 9

            gt_past_traj_list = [
                to_tensor(each["gt_past_traj"]) for each in queue
            ]  # L, N x 8 x 2
            gt_past_traj_mask_list = [
                to_tensor(each["gt_past_traj_mask"]) for each in queue
            ]  # L, N x 8 x 2
            gt_fut_traj = to_tensor(queue[-1]["gt_fut_traj"])  # L, N x 12 x 2
            gt_fut_traj_mask = to_tensor(queue[-1]["gt_fut_traj_mask"])  # L, N x 12 x 2

        metas_map = {}
        prev_pos = None
        prev_quaternion = None
        prev_angle = None
        prev_angle_radian = None
        for i, each in enumerate(queue):
            metas_map[i] = each["img_metas"].data
            if i == 0:
                metas_map[i]["prev_bev"] = False
                prev_pos = copy.deepcopy(metas_map[i]["can_bus"][:3])
                prev_quaternion = copy.deepcopy(metas_map[i]["can_bus"][3:7])
                prev_angle_radian = copy.deepcopy(metas_map[i]["can_bus"][-2])
                prev_angle = copy.deepcopy(metas_map[i]["can_bus"][-1])
                metas_map[i]["can_bus"][:3] = 0
                metas_map[i]["can_bus"][-1] = 0
                if self.fix_can_bus_rotation:
                    metas_map[i]["can_bus"][3:7] = np.array([1, 0, 0, 0])   # quaternion for 0 degree
                    metas_map[i]["can_bus"][-2] = 0
            else:
                metas_map[i]["prev_bev"] = True
                tmp_pos = copy.deepcopy(metas_map[i]["can_bus"][:3])
                tmp_quaternion = copy.deepcopy(metas_map[i]["can_bus"][3:7])
                tmp_angle_radian = copy.deepcopy(metas_map[i]["can_bus"][-2])
                tmp_angle = copy.deepcopy(metas_map[i]["can_bus"][-1])
                rel_quaternion = Quaternion(tmp_quaternion) * Quaternion(prev_quaternion).inverse
                metas_map[i]["can_bus"][:3] -= prev_pos
                metas_map[i]["can_bus"][-1] -= prev_angle
                if self.fix_can_bus_rotation:
                    metas_map[i]["can_bus"][3:7] = rel_quaternion.elements
                    metas_map[i]["can_bus"][-2] -= prev_angle_radian
                prev_pos = copy.deepcopy(tmp_pos)
                prev_quaternion = copy.deepcopy(tmp_quaternion)
                prev_angle = copy.deepcopy(tmp_angle)
                prev_angle_radian = copy.deepcopy(tmp_angle_radian)

        queue[-1]["img"] = DC(
            torch.stack(imgs_list), cpu_only=False, stack=True
        )  # T x N x 3 x H x W
        queue[-1]["img_metas"] = DC(metas_map, cpu_only=True)

        # inherent all the labels for the current frame only
        queue = queue[-1]

        # merge all other labels that requires temporal and merge from queue
        queue["l2g_r_mat"] = DC(l2g_r_mat_list)
        queue["l2g_t"] = DC(l2g_t_list)
        queue["timestamp"] = DC(timestamp_list)

        if self.process_perception:
            # merge all other labels that requires temporal and merge from queue
            queue["gt_labels_3d"] = DC(gt_labels_3d_list)
            queue["gt_sdc_label"] = DC(gt_sdc_label_list)
            queue["gt_inds"] = DC(gt_inds_list)
            queue["gt_bboxes_3d"] = DC(gt_bboxes_3d_list, cpu_only=True)
            queue["gt_sdc_bbox"] = DC(gt_sdc_bbox_list, cpu_only=True)
            queue["gt_fut_traj"] = DC(gt_fut_traj)
            queue["gt_fut_traj_mask"] = DC(gt_fut_traj_mask)
            queue["gt_past_traj"] = DC(gt_past_traj_list)
            queue["gt_past_traj_mask"] = DC(gt_past_traj_mask_list)

            # queue["gt_future_boxes"] = DC(gt_future_boxes_list, cpu_only=True)
            # queue["gt_future_labels"] = DC(gt_future_labels_list)

        return queue

    def union2one_test(self, queue):
        """
        convert a sequence of the sample dict into one single sample.
        """

        # sensor data and transform
        imgs_list: List[torch.Tensor] = [
            each["img"][0].data for each in queue
        ]  # L, C x 3 x H x W
        l2g_r_mat_list: List[torch.Tensor] = [
            to_tensor(each["l2g_r_mat"]) for each in queue
        ]  # L, 3 x 3
        l2g_t_list: List[torch.Tensor] = [
            to_tensor(each["l2g_t"]) for each in queue
        ]  # L, 3
        # BUG: do not convert to fp32!
        # timestamp_list = [to_tensor(each["timestamp"]) for each in queue]  # L, 1
        timestamp_list = [torch.tensor([each["timestamp"]], dtype=torch.float64) for each in queue]  # L, 1


        # converting the global absolute coordinate into relative position and orientation change
        metas_map = {}
        prev_pos = None
        prev_quaternion = None
        prev_angle = None
        prev_angle_radian = None
        for i, each in enumerate(queue):
            metas_map[i] = each["img_metas"][0].data
            if i == 0:
                metas_map[i]["prev_bev"] = False
                prev_pos = copy.deepcopy(metas_map[i]["can_bus"][:3])
                prev_quaternion = copy.deepcopy(metas_map[i]["can_bus"][3:7])
                prev_angle_radian = copy.deepcopy(metas_map[i]["can_bus"][-2])
                prev_angle = copy.deepcopy(metas_map[i]["can_bus"][-1])
                metas_map[i]["can_bus"][:3] = 0
                metas_map[i]["can_bus"][-1] = 0
                if self.fix_can_bus_rotation:
                    metas_map[i]["can_bus"][3:7] = np.array([1, 0, 0, 0])   # quaternion for 0 degree
                    metas_map[i]["can_bus"][-2] = 0
            else:
                metas_map[i]["prev_bev"] = True
                tmp_pos = copy.deepcopy(metas_map[i]["can_bus"][:3])
                tmp_quaternion = copy.deepcopy(metas_map[i]["can_bus"][3:7])
                tmp_angle_radian = copy.deepcopy(metas_map[i]["can_bus"][-2])
                tmp_angle = copy.deepcopy(metas_map[i]["can_bus"][-1])
                rel_quaternion = Quaternion(tmp_quaternion) * Quaternion(prev_quaternion).inverse
                metas_map[i]["can_bus"][:3] -= prev_pos
                metas_map[i]["can_bus"][-1] -= prev_angle
                if self.fix_can_bus_rotation:
                    metas_map[i]["can_bus"][3:7] = rel_quaternion.elements
                    metas_map[i]["can_bus"][-2] -= prev_angle_radian
                prev_pos = copy.deepcopy(tmp_pos)
                prev_quaternion = copy.deepcopy(tmp_quaternion)
                prev_angle = copy.deepcopy(tmp_angle)
                prev_angle_radian = copy.deepcopy(tmp_angle_radian)

        queue[-1]["img"] = DC(
            torch.stack(imgs_list), cpu_only=False, stack=True
        )  # T x N x 3 x H x W
        queue[-1]["img_metas"] = DC(metas_map, cpu_only=True)

        # inherent all the labels for the current frame only
        queue = queue[-1]

        queue["l2g_r_mat"] = DC(l2g_r_mat_list)
        queue["l2g_t"] = DC(l2g_t_list)
        queue["timestamp"] = DC(timestamp_list)
        return queue

    def init_trackpredplan(
        self,
        planning_steps,
        past_steps,
        fut_steps,
    ):
        # trajectory APIs for tracking/prediction/planning
        self.planning_steps = planning_steps
        self.past_steps = past_steps
        self.fut_steps = fut_steps


    def init_occupancy(
        self,
        occ_receptive_field,
        occ_n_future,
        occ_filter_invalid_sample,
        occ_filter_by_valid_flag,
    ):
        self.occ_receptive_field = occ_receptive_field  # past + current
        self.occ_n_future = occ_n_future  # future only
        self.occ_filter_invalid_sample = occ_filter_invalid_sample
        self.occ_filter_by_valid_flag = occ_filter_by_valid_flag
        self.occ_only_total_frames = 7  # NOTE: hardcode, not influenced by planning
        assert self.occ_filter_by_valid_flag is False

    def init_mapping(self, canvas_size, patch_size, lane_ann_file):
        self.map_num_classes = 3
        if canvas_size[0] == 50 or canvas_size[0] == 60:
            self.thickness = 1
        elif canvas_size[0] == 200:
            self.thickness = 2
        else:
            assert False
        self.patch_size = patch_size
        self.canvas_size = canvas_size

        self.vector_map = VectorizedLocalMap(
            map_root=self.map_root,
            patch_size=patch_size,
            map_classes={
                'ped_crossing': [SemanticMapLayer.CROSSWALK],
                'road_boundary': [SemanticMapLayer.ROADBLOCK, SemanticMapLayer.INTERSECTION, SemanticMapLayer.CARPARK_AREA],
            }
        )
        return

    def get_pdm_score_info(self, input_dict, index=None, info=None):
        if input_dict['sample_idx'] not in self.pdm_dict:
            logger.warning(f"PDM score not found for token: {input_dict['sample_idx']}")
            return self.get_zero_pdm(input_dict)
        data = self.pdm_dict[input_dict['sample_idx']]
        input_dict.update(data)
        return input_dict

    def get_zero_pdm(self, input_dict):
        pdm_list = [
            "no_at_fault_collisions",
            "drivable_area_compliance",
            "ego_progress",
            "time_to_collision_within_bound",
            "comfort",
            "score",
        ]
        for k in pdm_list:
            input_dict[k] = np.zeros((8192,))
        return input_dict

    def update_transform(self, input_dict, index):
        """Update transformation matrices from lidar to ego to global coordinates.
        
        Adapted from NuScenesE2EDataset for OpenScene/nuPlan data format.
        """
        info = self.data_infos[index]

        def inverse_trans(trans_matrix):
            R = trans_matrix[:3, :3]  # rot
            t = trans_matrix[:3, 3]   # trans

            # check whether R is a valid rotation matrix
            if not np.allclose(R @ R.T, np.eye(3), atol=1e-3):
                raise ValueError("R is not a valid rotation matrix")

            R_inv = R.T
            t_inv = -R.T @ t

            T_inv = np.eye(4)
            T_inv[:3, :3] = R_inv
            T_inv[:3, 3] = t_inv
            return T_inv
        
        # lidar to ego (from OpenScene format)
        if "lidar2ego" in info:
            l2e = info["lidar2ego"]
            l2e_r_mat = l2e[:3, :3]
            l2e_t = l2e[:3, 3]
        else:
            l2e_r = info.get("lidar2ego_rotation", [1, 0, 0, 0])
            l2e_t = info.get("lidar2ego_translation", [0, 0, 0])
            l2e_r_mat = Quaternion(l2e_r).rotation_matrix
            l2e = np.identity(4)
            l2e[:3, :3] = l2e_r_mat
            l2e[:3, 3] = l2e_t
        
        e2l = inverse_trans(l2e)

        # ego to global (from OpenScene format)
        if "ego2global" in info:
            e2g = info["ego2global"]
            e2g_r_mat = e2g[:3, :3]
            e2g_t = e2g[:3, 3]
        else:
            e2g_r = info.get("ego2global_rotation", [1, 0, 0, 0])
            e2g_t = info.get("ego2global_translation", [0, 0, 0])
            e2g_r_mat = Quaternion(e2g_r).rotation_matrix
            e2g = np.identity(4)
            e2g[:3, :3] = e2g_r_mat
            e2g[:3, 3] = e2g_t
        g2e = inverse_trans(e2g)

        # lidar to global
        if "lidar2global" in info:
            l2g = info["lidar2global"]
            l2g_r_mat = l2g[:3, :3]
            l2g_t = l2g[:3, 3]
        else:
            l2g_r_mat = l2e_r_mat.T @ e2g_r_mat.T
            l2g_t = l2e_t @ e2g_r_mat.T + e2g_t
            l2g = np.identity(4)
            l2g[:3, :3] = l2g_r_mat
            l2g[:3, 3] = l2g_t
        g2l = inverse_trans(l2g)

        # output dictionary
        input_dict.update(
            dict(
                # lidar to global
                l2g_r_mat=l2g_r_mat,
                l2g_t=l2g_t,
                l2g=l2g,
                g2l=g2l,
                # lidar to ego
                l2e_r_mat=np.array(l2e_r_mat),
                l2e_t=np.array(l2e_t),
                l2e=l2e,
                e2l=e2l,
                # ego to global
                e2g_r_mat=np.array(e2g_r_mat),
                e2g_t=np.array(e2g_t),
                e2g=e2g,
                g2e=g2e,
            )
        )
        return input_dict

    def update_sensor(self, input_dict, index):
        """Update sensor information including camera images and transforms."""
        info = self.data_infos[index]

        if self.modality is not None and self.modality.get("use_camera", False):
            image_paths = []
            lidar2img_rts = []
            lidar2cam_rts = []
            cam_intrinsics = []
            cam_distortions = []
            cam_optim_intrinsics = []

            # loop through all cameras
            for cam_type, cam_info in info["cams"].items():
                image_paths.append(cam_info["data_path"])

                # obtain lidar to image transformation matrix
                lidar2cam_r = cam_info["sensor2lidar_rotation"].T
                lidar2cam_t = -lidar2cam_r @ cam_info["sensor2lidar_translation"]
                lidar2cam_rt = np.eye(4)
                lidar2cam_rt[:3, :3] = lidar2cam_r
                lidar2cam_rt[:3, 3] = lidar2cam_t

                intrinsic = cam_info["cam_intrinsic"]
                viewpad = np.eye(4)
                viewpad[: intrinsic.shape[0], : intrinsic.shape[1]] = intrinsic
                lidar2img_rt = viewpad @ lidar2cam_rt
                lidar2img_rts.append(lidar2img_rt)

                cam_intrinsics.append(intrinsic)
                lidar2cam_rts.append(lidar2cam_rt)

                distortion = np.array(cam_info['distortion'])
                optim_intrinsic, _ = cv2.getOptimalNewCameraMatrix(
                    intrinsic, distortion, (1920, 1080), 1
                )
                cam_distortions.append(distortion)
                cam_optim_intrinsics.append(optim_intrinsic)

            input_dict.update(
                dict(
                    img_filename=image_paths,
                    lidar2img=lidar2img_rts,
                    cam_intrinsic=cam_intrinsics,
                    lidar2cam=lidar2cam_rts,
                    cam_distortion=cam_distortions,
                    cam_optim_intrinsic=cam_optim_intrinsics,
                )
            )
        return input_dict

    def update_canbus(self, input_dict, index):
        """Update CAN bus information including ego position, rotation, velocity, etc."""
        info = self.data_infos[index]
        can_bus = info.get("can_bus", np.zeros(18))
        if can_bus is None:
            can_bus = np.zeros(18)
        can_bus = np.array(can_bus).copy()

        # Location - global coordinate
        if "ego2global" in info:
            translation = info["ego2global"][:3, 3]
        else:
            translation = info.get("ego2global_translation", [0, 0, 0])
        
        sdc_loc_global = np.array([[0, 0, 0, 1.0]]).transpose()
        sdc_loc_global[:3, 0] = translation
        sdc_loc_ego = input_dict["g2e"] @ sdc_loc_global
        sdc_loc_lidar = input_dict["e2l"] @ sdc_loc_ego

        # Rotation
        if "ego2global_rotation" in info:
            e2g_r = info["ego2global_rotation"]
            sdc_rot_global_quat = Quaternion(e2g_r)
        else:
            e2g_r_mat = info["ego2global"][:3, :3]
            sdc_rot_global_quat = Quaternion(matrix=e2g_r_mat)
        
        sdc_rot_global_ele = sdc_rot_global_quat.elements
        yaw_angle_global = quaternion_yaw(sdc_rot_global_quat) / np.pi * 180
        if yaw_angle_global < 0:
            yaw_angle_global += 360

        sdc_rot_global = sdc_rot_global_quat.yaw_pitch_roll
        sdc_rot_global_mat = sdc_rot_global_quat.rotation_matrix
        sdc_rot_ego_mat = input_dict["g2e"][:3, :3] @ sdc_rot_global_mat
        sdc_rot_ego = Quaternion(matrix=sdc_rot_ego_mat).yaw_pitch_roll
        sdc_rot_lidar_mat = input_dict["e2l"][:3, :3] @ sdc_rot_ego_mat
        sdc_rot_lidar = Quaternion(matrix=sdc_rot_lidar_mat).yaw_pitch_roll

        # Velocity, acceleration and angular velocity
        sdc_acc_ego = can_bus[7:10] if len(can_bus) > 10 else np.zeros(3)
        sdc_vel_ego = can_bus[10:13] if len(can_bus) > 13 else np.zeros(3)
        sdc_ang_ego = can_bus[13:16] if len(can_bus) > 16 else np.zeros(3)

        sdc_acc_lidar = input_dict["e2l"][:3, :3] @ sdc_acc_ego
        sdc_acc_global = input_dict["e2g_r_mat"] @ sdc_acc_ego
        sdc_ang_lidar = input_dict["e2l"][:3, :3] @ sdc_ang_ego
        sdc_ang_global = input_dict["e2g_r_mat"] @ sdc_ang_ego
        sdc_vel_lidar = input_dict["e2l"][:3, :3] @ sdc_vel_ego
        sdc_vel_global = input_dict["e2g_r_mat"] @ sdc_vel_ego

        # Assign values to can_bus
        can_bus[:3] = sdc_loc_global[:3, 0]
        can_bus[3:7] = sdc_rot_global_ele
        can_bus[-2] = yaw_angle_global / 180 * np.pi
        can_bus[-1] = yaw_angle_global

        # output dictionary
        input_dict.update(
            dict(
                can_bus=can_bus,
                sdc_loc_global=np.array(sdc_loc_global),
                sdc_loc_ego=np.array(sdc_loc_ego),
                sdc_loc_lidar=np.array(sdc_loc_lidar),
                sdc_rot_global=np.array(sdc_rot_global),
                sdc_rot_ego=np.array(sdc_rot_ego),
                sdc_rot_lidar=np.array(sdc_rot_lidar),
                sdc_acc_global=np.array(sdc_acc_global),
                sdc_acc_ego=np.array(sdc_acc_ego),
                sdc_acc_lidar=np.array(sdc_acc_lidar),
                sdc_vel_global=np.array(sdc_vel_global),
                sdc_vel_ego=np.array(sdc_vel_ego),
                sdc_vel_lidar=np.array(sdc_vel_lidar),
                sdc_ang_global=np.array(sdc_ang_global),
                sdc_ang_ego=np.array(sdc_ang_ego),
                sdc_ang_lidar=np.array(sdc_ang_lidar),
            )
        )
        return input_dict

    def update_detection(self, annotation, index, use_mask=True):

        info = self.data_infos[index]

        # get mask for filtering
        if self.use_valid_flag:
            mask = info.get("valid_flag", np.ones(len(info.get("gt_boxes", [])), dtype=bool))
        else:
            mask = info.get("num_lidar_pts", np.ones(len(info.get("gt_boxes", [])))) > 0

        # GT for detection/tracking
        gt_bboxes_3d = info.get("gt_boxes", np.zeros((0, 7))).copy()
        gt_names_3d = info.get("gt_names", np.array([])).copy()
        gt_inds = info.get("gt_inds", np.array([])).copy()
        gt_vis_tokens = info.get("visibility_tokens", None)
        if gt_vis_tokens is not None:
            gt_vis_tokens = gt_vis_tokens.copy()

        # filter out bbox containing no points
        if use_mask and len(mask) > 0 and len(gt_bboxes_3d) > 0:
            gt_bboxes_3d = gt_bboxes_3d[mask]
            gt_names_3d = gt_names_3d[mask]
            gt_inds = gt_inds[mask] if len(gt_inds) > 0 else gt_inds
            if gt_vis_tokens is not None:
                gt_vis_tokens = gt_vis_tokens[mask]

        # cls_name to cls_id
        gt_labels_3d = []
        for cat in gt_names_3d:
            if cat in self.CLASSES:
                gt_labels_3d.append(self.CLASSES.index(cat))
            else:
                gt_labels_3d.append(-1)
        gt_labels_3d = np.array(gt_labels_3d)

        if self.with_velocity and len(gt_bboxes_3d) > 0:
            gt_velocity = info.get("gt_velocity", np.zeros((len(gt_bboxes_3d), 2)))
            if use_mask and len(mask) > 0:
                gt_velocity = gt_velocity[mask]
            nan_mask = np.isnan(gt_velocity[:, 0])
            gt_velocity[nan_mask] = [0.0, 0.0]
            gt_bboxes_3d = np.concatenate([gt_bboxes_3d, gt_velocity], axis=-1)

        # Create LiDARInstance3DBoxes
        if len(gt_bboxes_3d) > 0:
            gt_bboxes_3d = LiDARInstance3DBoxes(
                gt_bboxes_3d, box_dim=gt_bboxes_3d.shape[-1], origin=(0.5, 0.5, 0.5)
            ).convert_to(self.box_mode_3d)
        else:
            gt_bboxes_3d = LiDARInstance3DBoxes(
                np.zeros((0, 9)), box_dim=9, origin=(0.5, 0.5, 0.5)
            ).convert_to(self.box_mode_3d)

        # output dictionary
        anno_tmp = dict(
            gt_bboxes_3d=gt_bboxes_3d,
            gt_labels_3d=gt_labels_3d,
            gt_names=gt_names_3d,
            gt_inds=gt_inds,
            mask=mask,
            gt_vis_tokens=gt_vis_tokens,
        )
        if annotation is None:
            return anno_tmp

        annotation.update(anno_tmp)
        return annotation

    def get_ann_info(self, index):
        """Get annotation info according to the given index.

        Returns:
            dict: Annotation information consists of the following keys:
                - gt_bboxes_3d (:obj:`LiDARInstance3DBoxes`): 3D ground truth bboxes
                - gt_labels_3d (np.ndarray): Labels of ground truths.
                - gt_names (list[str]): Class names of ground truths.
                - gt_inds (np.ndarray): Instance ids of ground truths.
                - gt_fut_traj (np.ndarray): Future trajectories.
                - gt_fut_traj_mask (np.ndarray): Future trajectory masks.
        """
        annotation = dict()
        annotation = self.update_detection(annotation, index, use_mask=True)
        annotation = self.update_tracking_prediction(annotation, index)
        return annotation

    def get_data_info(self, index, info=None, debug=False, prev_frame=False):
        """Get data info according to the given index.

        Args:
            index (int): Index of the sample data to get.
        Returns:
            dict: Data information that will be passed to the data \
                preprocessing pipelines. It includes the following keys:

                - sample_idx (str): Sample index.
                - pts_filename (str): Filename of point clouds.
                - sweeps (list[dict]): Infos of sweeps.
                - timestamp (float): Sample timestamp.
                - img_filename (str, optional): Image filename.
                - lidar2img (list[np.ndarray], optional): Transformations \
                    from lidar to different cameras.
                - ann_info (dict): Annotation info.
        """

        if info is None:
            info = self.data_infos[index]
        # standard protocal modified from SECOND.Pytorch
        input_dict = dict(
            sample_idx=info["token"],  # str: OpenScene unique sample token
            frame_idx=info["frame_idx"],  # int: 0-indexed frame IDs
            timestamp=info["timestamp"] / 1e6,  # int: OpenScene unique time index
            log_name=info["log_name"],  # str: OpenScene unique sample token
            log_token=info["log_token"],  # str: OpenScene unique sample token
            scene_name=info["scene_name"],  # str: OpenScene sequence name
            scene_token=info["scene_token"],  # str: OpenScene sequence name
            pts_filename=info["lidar_path"],  # str: relative path for the lidar data
            prev=info["sample_prev"],  # str: OpenScene unique sample token
            next=info["sample_next"],  # str: OpenScene unique sample token
            lidar2global_rotation=info["lidar2global"][:3, :3],
        )

        input_dict = self.update_transform(input_dict=input_dict, index=index)
        input_dict = self.update_sensor(input_dict=input_dict, index=index)
        input_dict = self.update_canbus(input_dict=input_dict, index=index)

        if not prev_frame:
            # only count for curr frame PDMScore
            input_dict = self.get_pdm_score_info(input_dict, index, info=info)
        else:
            input_dict = self.get_zero_pdm(input_dict)

        if self.process_perception:
            input_dict["ann_info"] = self.get_ann_info(index)
            input_dict = self.update_mapping(input_dict=input_dict, index=index)

        input_dict = self.update_ego_prediction(input_dict=input_dict, index=index)
        input_dict = self.update_ego_planning(input_dict=input_dict, index=index)

        return input_dict

    def update_anno_token(self, annotation, index):
        # add OpenScene-specific token information

        return annotation

    def update_tracking_prediction(self, annotation, index):
        info = self.data_infos[index]

        # get trajectories for surrounding agents, lidar coordinate, 2Hz
        gt_fut_bbox_lidar = info["gt_fut_bbox_lidar"][annotation["mask"]]  # N x 12 x 9
        gt_fut_bbox_mask = info["gt_fut_bbox_mask"][annotation["mask"]]  # N x 12 x 1
        gt_pre_bbox_lidar = info["gt_pre_bbox_lidar"][annotation["mask"]]  # N x 4 x 9
        gt_pre_bbox_mask = info["gt_pre_bbox_mask"][annotation["mask"]]  # N x 4 x 1

        # change the past traj into backward order to satisfy UniAD
        # e.g., -4, -3, -2, -1 changed to -1, -2, -3, -4
        gt_pre_bbox_lidar = np.flip(gt_pre_bbox_lidar, axis=1)  # N x 4 x 9
        gt_pre_bbox_mask = np.flip(gt_pre_bbox_mask, axis=1)  # N x 4 x 1

        # repeat redundant channels for masks
        gt_fut_bbox_mask = np.repeat(gt_fut_bbox_mask, 2, axis=2)  # N x 12 x 2
        gt_pre_bbox_mask = np.repeat(gt_pre_bbox_mask, 2, axis=2)  # N x 4 x 2

        # append 4 future frames of trajectories
        # so now, it contains frames in the order of -1, -2, -3, -4, 1, 2, 3, 4
        gt_pre_bbox_lidar = np.concatenate(
            (gt_pre_bbox_lidar, gt_fut_bbox_lidar[:, :4]), axis=1
        )  # N x 8 x 9
        gt_pre_bbox_mask = np.concatenate(
            (gt_pre_bbox_mask, gt_fut_bbox_mask[:, :4]), axis=1
        )  # N x 8 x 2

        # update output dictionary
        annotation.update(
            dict(
                gt_fut_traj=gt_fut_bbox_lidar[:, :, :2],  # N x 12 x 2, lidar coordinate
                gt_fut_traj_mask=gt_fut_bbox_mask,  # N x 12 x 2
                gt_past_traj=gt_pre_bbox_lidar[:, :, :2],  # N x 8 x 2
                gt_past_traj_mask=gt_pre_bbox_mask,  # N x 8 x 2
            )
        )
        return annotation

    def update_mapping(self, input_dict, index):
        # get polyline in the ego coordinate
        polyline_dict, drivable_area = self.update_map_vectorized(input_dict, index)

        input_dict = self.update_map_rasterized(input_dict, index, polyline_dict, drivable_area)

        return input_dict

    def update_map_vectorized(self, input_dict, index):
        info = self.data_infos[index]

        map_location = info['map_location']
        e2g_translation = info['ego2global_translation']
        e2g_rotation = info['ego2global_rotation']
        if len(e2g_rotation) != 4:
            e2g_rotation = Quaternion._from_matrix(e2g_rotation).elements
        polyline_dict = self.vector_map.gen_vectorized_samples(e2g_translation, e2g_rotation, map_location)

        drivable_area = self.vector_map.gen_drivable_area(e2g_translation, e2g_rotation, map_location)

        return polyline_dict, drivable_area

    def update_map_rasterized(self, input_dict, index, polyline_dict, drivable_area, debug=False):
        gt_labels = []
        gt_bboxes = []
        gt_masks = []

        origin = np.array([self.canvas_size[1] // 2, self.canvas_size[0] // 2])
        scale = np.array([self.canvas_size[1] / self.patch_size[0], self.canvas_size[0] / self.patch_size[1]])

        for polyline in polyline_dict:
            if polyline["type"] == 0:
                continue
            # skip centerline
            map_mask = np.zeros(self.canvas_size, np.uint8)
            draw_coor = (polyline["pts"] * scale + origin).round().astype(np.int32)
            gt_mask = cv2.polylines(map_mask, [draw_coor], False, color=1, thickness=self.thickness) / 255

            ys, xs = np.where(gt_mask)
            try:
                gt_bbox = [min(xs), min(ys), max(xs), max(ys)]
            except ValueError:
                gt_bbox = [0, 0, 1, 1]
                continue

            cls = polyline["type"]

            gt_labels.append(cls)
            gt_bboxes.append(gt_bbox)
            gt_masks.append(gt_mask)

        # for stuff class, drivable area
        map_mask = np.zeros(self.canvas_size, np.uint8)
        exteriors = []
        interiors = []
        for p in drivable_area.geoms:
            exteriors.append(
                (np.array(p.exterior.coords) * scale + origin).round().astype(np.int32)
            )
            for inter in p.interiors:
                interiors.append(
                    (np.array(inter.coords) * scale + origin).round().astype(np.int32)
                )
        cv2.fillPoly(map_mask, exteriors, 255)
        cv2.fillPoly(map_mask, interiors, 0)
        map_mask = map_mask / 255
        try:
            ys, xs = np.where(map_mask)
            gt_bbox = [min(xs), min(ys), max(xs), max(ys)]
            gt_labels.append(self.map_num_classes)
            gt_bboxes.append(gt_bbox)
            gt_masks.append(map_mask)
        except ValueError:
            pass

        gt_labels = torch.tensor(gt_labels)
        gt_bboxes = torch.from_numpy(np.stack(gt_bboxes))
        gt_masks = torch.from_numpy(np.stack(gt_masks))  # N x H x W

        # update output dictionary
        input_dict.update(
            dict(
                gt_lane_labels=gt_labels,
                gt_lane_bboxes=gt_bboxes,
                gt_lane_masks=gt_masks,
            )
        )
        return input_dict


    def update_ego_prediction(self, input_dict, index):
        info = self.data_infos[index]

        # retrieve ego bounding box in the current frame in lidar coordinate
        sdc_vel_ego = info["can_bus"][10:13]
        sdc_vel_lidar = info["lidar2ego"][:3, :3].T @ sdc_vel_ego

        # NOTE: from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters
        # All ego box here is not the center of the box, but the rear-axle position.
        x, y, z = 0., 0., 0.
        l, w, h = 5.176, 2.297, 1.777
        gt_sdc_bbox = np.array([
            x, y, z,
            l, w, h,
            0.,
            sdc_vel_lidar[0], sdc_vel_lidar[1],
        ]).reshape((1, 9)) # 9
        sdc_status = gt_sdc_bbox.reshape(9)

        # retrieve ego class label, default it's car
        gt_sdc_label = np.array([0])
        gt_sdc_label = DC(to_tensor(gt_sdc_label))
        gt_sdc_bbox = LiDARInstance3DBoxes(
            gt_sdc_bbox, box_dim=gt_sdc_bbox.shape[-1], origin=(0.5, 0.5, 0)
        ).convert_to(self.box_mode_3d)
        gt_sdc_bbox = DC(gt_sdc_bbox, cpu_only=True)

        # ego trajectory in lidar coordinate
        gt_pre_bbox_sdc_lidar = info["gt_pre_bbox_sdc_lidar"]  # 1 x 4 x 9
        gt_fut_bbox_sdc_lidar = info["gt_fut_bbox_sdc_lidar"]  # 1 x 12 x 9

        # ego trajectory in global coordinate
        gt_pre_bbox_sdc_global = info["gt_pre_bbox_sdc_global"]  # 1 x 4 x 9
        gt_fut_bbox_sdc_global = info["gt_fut_bbox_sdc_global"]  # 1 x 12 x 9

        # ego trajectory mask
        gt_fut_bbox_sdc_mask = info["gt_fut_bbox_sdc_mask"]  # 1 x future_frame_num x 1
        gt_fut_bbox_sdc_mask = np.repeat(gt_fut_bbox_sdc_mask, 2, axis=2)  # 1 x future_frame_num x 2

        gt_pre_bbox_sdc_mask = info["gt_pre_bbox_sdc_mask"]  # 1 x history_frame_num x 1
        gt_pre_bbox_sdc_mask = np.repeat(gt_pre_bbox_sdc_mask, 4, axis=2)  # 1 x history_frame_num x 4
        gt_pre_command_sdc = info["gt_pre_command_sdc"]

        sdc_planning = gt_fut_bbox_sdc_lidar[
            :, :self.planning_steps, [0, 1, 6]
        ]  # 1 x planning_steps x 3, lidar coordinate, x,y,yaw
        sdc_planning_mask = gt_fut_bbox_sdc_mask[:, :self.planning_steps]

        # update output dictionary for ego's prediction
        input_dict.update(
            dict(
                # sdc_vel_lidar_calculated=sdc_vel_lidar_calculated,  # (2, )
                gt_sdc_bbox=gt_sdc_bbox,  # DC (LiDARInstance3DBoxes), xyz, lwh, yaw, vel_x, vel_y, lidar coordinate
                gt_sdc_label=gt_sdc_label,  # DC (tensor[0])
                gt_sdc_fut_traj=gt_fut_bbox_sdc_lidar[
                    :, :, :2
                ],  # 1 x 12 x 2, lidar coordinate
                gt_sdc_fut_traj_mask=gt_fut_bbox_sdc_mask,  # 1 x 12 x 2
                # planning labels
                command=np.argmax(info["driving_command"]),  # int, change from 1-indexed to 0-indexed
                sdc_planning_world=gt_fut_bbox_sdc_global[
                    :, : self.planning_steps, [0, 1, 2]
                ],
                sdc_planning=sdc_planning,
                sdc_planning_mask=sdc_planning_mask,
                sdc_planning_past=gt_pre_bbox_sdc_lidar[:, :, [0, 1, 6]],
                sdc_planning_mask_past=gt_pre_bbox_sdc_mask,
                gt_pre_command_sdc=gt_pre_command_sdc,
                sdc_status=sdc_status[[0, 1, 6]]
            )
        )

        return input_dict

    def update_ego_planning(self, input_dict, index):
        # sdc_planning already added in update_ego_prediction
        return input_dict

    def evaluate(
        self,
        results,
        metric="bbox",
        logger=None,
        jsonfile_prefix=None,
        result_names=["pts_bbox"],
        show=False,
        out_dir=None,
        pipeline=None,
    ):
        results_df = pd.DataFrame(results)
        results_df = results_df.drop_duplicates(subset=['token'], keep='first')

        essential_columns = [
            'token', 'ade_4s', 'fde_4s', 'no_at_fault_collisions', 
            'drivable_area_compliance', 'ego_progress', 
            'time_to_collision_within_bound', 'comfort', 'score'
        ]
        results_df = results_df[essential_columns]

        average_row = results_df.drop(columns=["token"]).mean(skipna=True)
        average_row['token'] = 'average'
        results_df.loc[len(results_df)] = average_row

        save_path = Path(jsonfile_prefix + ".csv")
        save_path.parent.mkdir(parents=True, exist_ok=True)
        results_df.to_csv(save_path, float_format='%.4f', index=False)

        # save&export navsim submission pickle
        from navsim.common.dataclasses import Trajectory
        output: Dict[str, Trajectory] = {}
        for result in results:
            token = result['token']
            trajectory = Trajectory(result['trajectory'][4::5])
            output[token] = trajectory

        if "navtest.yaml" in self.nav_filter_path:
            submission = {
                "team_name": "PLACEHOLDER",
                "authors": ["PLACEHOLDER"],
                "email": "PLACEHOLDER@gmail.com",
                "institution": "PLACEHOLDER",
                "country / region": "PLACEHOLDER",
                "predictions": [output],
            }
            filename = Path(jsonfile_prefix + "_navsim_submission.pkl")
            with filename.open("wb") as file:
                pickle.dump(submission, file)

        if "navtest_failures" not in self.nav_filter_path and "navtest.yaml" in self.nav_filter_path:
            with open("configs/navsim_splits/navtest_split/navtest_failures_filtered.yaml", 'r') as file:
                nav_filter = yaml.safe_load(file)
            navtest_failures_tokens = nav_filter['tokens']
            results_df_navtest_failures = results_df[results_df['token'].isin(navtest_failures_tokens)]
            average_row_navtest_failures = results_df_navtest_failures.drop(columns=["token"]).mean(skipna=True)
            average_row_navtest_failures['token'] = 'average'
            results_df_navtest_failures.loc[len(results_df_navtest_failures)] = average_row_navtest_failures

            save_path_navtest_failures = Path(jsonfile_prefix + "_navtest_failures.csv")
            save_path_navtest_failures.parent.mkdir(parents=True, exist_ok=True)
            results_df_navtest_failures.to_csv(save_path_navtest_failures, float_format='%.4f', index=False)

        return_dict = {
            'ade_4s': average_row['ade_4s'],
            'fde_4s': average_row['fde_4s'],
            'no_at_fault_collisions': average_row['no_at_fault_collisions'],
            'drivable_area_compliance': average_row['drivable_area_compliance'],
            'ego_progress': average_row['ego_progress'],
            'time_to_collision_within_bound': average_row['time_to_collision_within_bound'],
            'comfort': average_row['comfort'],
            'pdm_score': average_row['score'],
        }
        return return_dict
