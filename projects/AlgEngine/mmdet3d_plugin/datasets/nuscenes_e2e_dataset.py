import copy
import json
import os
import pickle
import pprint
import random
import tempfile
from os import path as osp
from typing import Dict, List

import sys
import mmcv
import numpy as np
import scipy
import torch

from mmcv.parallel import DataContainer as DC
from mmdet3d.core.bbox import LiDARInstance3DBoxes
from mmdet3d.datasets import NuScenesDataset
import mmdet3d_plugin.datasets.data_utils.object_rotation as object_rotation
from mmdet3d_plugin.datasets.data_utils.rasterize import preprocess_map
from mmdet3d_plugin.datasets.data_utils.trajectory_api import NuScenesTraj
from mmdet3d_plugin.datasets.data_utils.vector_map import VectorizedLocalMap
from mmdet3d_plugin.datasets.eval_utils.map_api import NuScenesMap
from mmdet3d_plugin.uniad.dense_heads.planning_head_plugin.compute_metrics import (
    PredictionConfig, compute_metrics)
from mmdet.datasets import DATASETS
from mmdet.datasets.pipelines import to_tensor
from nuscenes import NuScenes
from nuscenes.eval.common.config import config_factory
from nuscenes.eval.common.utils import Quaternion, quaternion_yaw
from nuscenes.eval.tracking.evaluate import TrackingEval
from nuscenes.prediction import convert_local_coords_to_global
from PIL import Image
from prettytable import PrettyTable

from .data_utils.data_utils import (lidar_nusc_box_to_global, obtain_map_info,
                                    output_to_nusc_box, output_to_nusc_box_det)
from .eval_utils.nuscenes_eval import NuScenesEval_custom
from .eval_utils.nuscenes_eval_motion import MotionEval

# np.set_printoptions(suppress=True, precision=3)
# torch.set_printoptions(sci_mode=False, precision=3)

from mmdet3d_plugin.utils import get_logger
logger = get_logger(__name__)
WORLDENGINE_ROOT = os.getenv('WORLDENGINE_ROOT', os.path.abspath('.'))

@DATASETS.register_module()
class NuScenesE2EDataset(NuScenesDataset):
    r"""NuScenes E2E Dataset"""

    def __init__(
        self,
        #split for sub-set
        use_num_split=0,
        queue_length=4,
        bev_size=(200, 200),
        # map-related
        patch_size=(102.4, 102.4),
        canvas_size=(200, 200),
        lane_ann_file=None,
        # tracking
        past_steps=4,
        fut_steps=4,
        # prediction
        predict_steps=12,
        # planning
        planning_steps=6,
        use_nonlinear_optimizer=False,
        # Occ dataset
        occ_receptive_field=3,
        occ_n_future=4,
        occ_filter_invalid_sample=False,
        occ_filter_by_valid_flag=False,
        # eval and eval
        enbale_temporal_aug=False,
        eval_mod=None,
        overlap_test=False,
        # debug & miscell
        is_debug=False,
        len_debug=10,
        file_client_args=dict(backend="disk"),
        *args,
        **kwargs,
    ):
        # init before super init since it is called in parent class
        self.file_client_args = file_client_args
        self.file_client = mmcv.FileClient(**file_client_args)

        # debugging
        self.is_debug = is_debug
        self.len_debug = len_debug
        self.old_canbus = False  # using the old canbus setting from UniAD, is wrong

        self.use_num_split = use_num_split

        # only check canbus for the new setting
        if self.old_canbus:
            self.check_canbus = False
        else:
            self.check_canbus = True
        super().__init__(*args, **kwargs)

        # key hyper-parameters
        self.queue_length = queue_length
        self.bev_size = bev_size

        # train & eval
        self.enbale_temporal_aug = enbale_temporal_aug
        assert self.enbale_temporal_aug is False
        self.overlap_test = overlap_test
        self.eval_mod = eval_mod

        # nuScenes
        self.init_dataset()

        # task-specific init
        self.init_trackpredplan(
            predict_steps=predict_steps,
            planning_steps=planning_steps,
            past_steps=past_steps,
            fut_steps=fut_steps,
            use_nonlinear_optimizer=use_nonlinear_optimizer,
        )
        self.init_occupancy(
            occ_receptive_field=occ_receptive_field,
            occ_n_future=occ_n_future,
            occ_filter_invalid_sample=occ_filter_invalid_sample,
            occ_filter_by_valid_flag=occ_filter_by_valid_flag,
        )
        self.init_mapping(
            canvas_size=canvas_size, patch_size=patch_size, lane_ann_file=lane_ann_file
        )

    def __len__(self):
        if not self.is_debug:
            return len(self.data_infos)
        else:
            return self.len_debug

    def init_dataset(self):
        # nuScenes APIs
        self.nusc = NuScenes(
            version=self.version, dataroot=self.data_root, verbose=True
        )
        self.default_attr = NuScenesDataset.DefaultAttribute
        self.sdc_acc_ego_fix = True

    def init_trackpredplan(
        self,
        predict_steps,
        planning_steps,
        past_steps,
        fut_steps,
        use_nonlinear_optimizer,
    ):
        # trajectory APIs for tracking/prediction/planning
        self.predict_steps = predict_steps
        self.planning_steps = planning_steps
        self.past_steps = past_steps
        self.fut_steps = fut_steps
        self.use_nonlinear_optimizer = use_nonlinear_optimizer
        self.traj_api = NuScenesTraj(
            self.nusc,
            self.predict_steps,
            self.planning_steps,
            self.past_steps,
            self.fut_steps,
            self.with_velocity,
            self.CLASSES,
            self.box_mode_3d,
            self.use_nonlinear_optimizer,
        )

        # eval for planning
        self.helper = self.traj_api.predict_helper
        # self.config = load_prediction_config(self.helper, "./planning.json")
        # Load config file and deserialize it.

        if self.test_mode:
            planning_map_config_file = "./sources/eval/planning_all.json"
        else:
            planning_map_config_file = "./sources/eval/planning_trainval.json"
        with open(planning_map_config_file, "r") as f:
            planning_map_config = json.load(f)
        self.planning_map_config = PredictionConfig.deserialize(
            planning_map_config, self.helper
        )

        # manually labelled route navigation signal
        route_info_path = os.path.join(WORLDENGINE_ROOT, "data/raw/nuscenes/infos/e2e_av/route_label.json")
        # route_info_path = None
        if route_info_path is not None:
            with open(route_info_path) as f:
                self.route_info_dict: Dict[str, str] = json.load(f)

            # update the number of commands after checking manual human labels
            all_commands = set(self.route_info_dict.values())

            # map all commands to integer
            self.command_str2int = {
                "TURN RIGHT": 0,
                "TURN LEFT": 1,
                "KEEP FORWARD": 2,
                "TURN RIGHT AT THE NEXT INTERSECTION": 3,
                "TURN LEFT AT THE NEXT INTERSECTION": 4,
                "PREPARE TO STOP ON THE LEFT": 5,
                "ENTER AND DRIVE IN THE ROUNDABOUT": 6,
                "EXIT THE ROUNDABOUT": 7,
                "UTURN": 8,
            }
        else:
            self.route_info_dict = None

    def init_occupancy(
        self,
        occ_receptive_field,
        occ_n_future,
        occ_filter_invalid_sample,
        occ_filter_by_valid_flag,
    ):
        self.occ_receptive_field = (
            occ_receptive_field  # past + current, if=3, means 2 past frames
        )
        self.occ_n_future = occ_n_future  # future only
        self.occ_filter_invalid_sample = occ_filter_invalid_sample
        self.occ_filter_by_valid_flag = occ_filter_by_valid_flag
        self.occ_only_total_frames = 7  # NOTE: hardcode, not influenced by planning
        assert self.occ_filter_by_valid_flag is False

    def init_mapping(self, canvas_size, patch_size, lane_ann_file):
        # initialize the map-related APIs in nuScenes

        self.map_root = os.path.join(WORLDENGINE_ROOT, "data/raw/nuscenes")

        # map rasterization parameters
        # 0 -> lane divider, 1 -> crossing, 2 -> contour
        self.map_num_classes = 3
        if canvas_size[0] == 50 or canvas_size[0] == 60:
            self.thickness = 1
        elif canvas_size[0] == 200:
            self.thickness = 2
        else:
            assert False
        self.angle_class = 36  # total 10 different colors to cover 360 degrees
        # actual meter range, [-102.4, 102.4]
        self.patch_size = patch_size
        self.canvas_size = canvas_size  # 200 x 200, BEV space
        assert canvas_size[0] == canvas_size[1], "not square"
        assert patch_size[0] == patch_size[1], "not square"
        self.pixel_per_meter = canvas_size[0] / patch_size[0]

        ####### map rasterized data
        # preload the data file
        self.lane_infos = (
            self.load_annotations(lane_ann_file) if lane_ann_file else None
        )
        # load the data online
        self.nusc_maps = {
            "boston-seaport": NuScenesMap(
                dataroot=self.map_root, map_name="boston-seaport"
            ),
            "singapore-hollandvillage": NuScenesMap(
                dataroot=self.map_root, map_name="singapore-hollandvillage"
            ),
            "singapore-onenorth": NuScenesMap(
                dataroot=self.map_root, map_name="singapore-onenorth"
            ),
            "singapore-queenstown": NuScenesMap(
                dataroot=self.map_root, map_name="singapore-queenstown"
            ),
        }

        ####### map vectorized data
        self.vector_map = VectorizedLocalMap(
            self.map_root, patch_size=self.patch_size, canvas_size=self.canvas_size
        )

    def load_annotations(self, ann_file):
        """Load annotations from ann_file.
        Args:
            ann_file (str): Path of the annotation file.

        Returns:
            list[dict]: List of annotations sorted by timestamps.
        """
        if self.file_client_args["backend"] == "disk":
            # data_infos = mmcv.load(ann_file)
            data = pickle.loads(self.file_client.get(ann_file))
            data_infos = list(sorted(data["infos"], key=lambda e: e["timestamp"]))
            data_infos = data_infos[:: self.load_interval]
            self.metadata = data["metadata"]
            self.version = self.metadata["version"]
        elif self.file_client_args["backend"] == "petrel":
            data = pickle.loads(self.file_client.get(ann_file))
            data_infos = list(sorted(data["infos"], key=lambda e: e["timestamp"]))
            data_infos = data_infos[:: self.load_interval]
            self.metadata = data["metadata"]
            self.version = self.metadata["version"]
        else:
            assert False, "Invalid file_client_args!"
        return data_infos
    

    def prepare_train_data(self, index):
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
        if self.enbale_temporal_aug:
            # temporal aug
            prev_indexs_list = list(range(index - self.queue_length, index))
            random.shuffle(prev_indexs_list)
            prev_indexs_list = sorted(prev_indexs_list[1:], reverse=True)
            input_dict = self.get_data_info(index)
        else:
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
            # current timestamp
            input_dict = self.get_data_info(final_index)
            prev_indexs_list = list(reversed(range(first_index, final_index)))
        if input_dict is None:
            return None
        frame_idx = input_dict["frame_idx"]
        scene_token = input_dict["scene_token"]
        self.pre_pipeline(input_dict)
        example = self.pipeline(input_dict)

        assert example["gt_labels_3d"].data.shape[0] == example["gt_fut_traj"].shape[0]
        assert example["gt_labels_3d"].data.shape[0] == example["gt_past_traj"].shape[0]

        if self.filter_empty_gt and (
            example is None or ~(example["gt_labels_3d"]._data != -1).any()
        ):
            return None
        data_queue.insert(0, example)

        ########## retrieve previous infos, frame by frame
        for i in prev_indexs_list:
            if self.enbale_temporal_aug:
                i = max(0, i)
            input_dict = self.get_data_info(i, prev_frame=True)
            if input_dict is None:
                return None
            if (
                input_dict["frame_idx"] < frame_idx
                and input_dict["scene_token"] == scene_token
            ):
                self.pre_pipeline(input_dict)
                example = self.pipeline(input_dict)
                if self.filter_empty_gt and (
                    example is None or ~(example["gt_labels_3d"]._data != -1).any()
                ):
                    return None
                frame_idx = input_dict["frame_idx"]
            assert (
                example["gt_labels_3d"].data.shape[0] == example["gt_fut_traj"].shape[0]
            )
            assert (
                example["gt_labels_3d"].data.shape[0]
                == example["gt_past_traj"].shape[0]
            )
            data_queue.insert(0, copy.deepcopy(example))

        # merge a sequence of data into one dictionary, only for temporal data
        data_queue = self.union2one(data_queue)
        return data_queue

    def prepare_test_data(self, index):
        """
        Training data preparation.
        Args:
            index (int): Index for accessing the target data.
        Returns:
            dict: Training data dict of the corresponding index.
                img: queue_length, 6, 3, H, W
                img_metas: img_metas of each frame (list)
                gt_labels_3d: gt_labels of each frame (list)
                gt_bboxes_3d: gt_bboxes of each frame (list)
                gt_inds: gt_inds of each frame(list)
        """

        input_dict = self.get_data_info(index)
        self.pre_pipeline(input_dict)
        example = self.pipeline(input_dict)
        data_dict = {}
        for key, value in example.items():
            if "l2g" in key:
                data_dict[key] = to_tensor(value[0])
            else:
                data_dict[key] = value

        return data_dict

    def union2one(self, queue):
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
        timestamp_list = [to_tensor(each["timestamp"]) for each in queue]  # L, 1

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

        # trajectory prediction
        # gt_past_traj_list = [
        #     to_tensor(each["gt_past_traj"]) for each in queue
        # ]  # L, N x 8 x 2
        # gt_past_traj_mask_list = [
        #     to_tensor(each["gt_past_traj_mask"]) for each in queue
        # ]  # L, N x 8 x 2
        gt_fut_traj = to_tensor(queue[-1]["gt_fut_traj"])  # L, N x 12 x 2
        gt_fut_traj_mask = to_tensor(queue[-1]["gt_fut_traj_mask"])  # L, N x 12 x 2

        # planning
        gt_sdc_bbox_list: List[LiDARInstance3DBoxes] = [
            each["gt_sdc_bbox"].data for each in queue
        ]  # L, 1 x 9
        gt_sdc_fut_traj = to_tensor(queue[-1]["gt_sdc_fut_traj"])  # L, 1 x 12 x 2
        gt_sdc_fut_traj_mask = to_tensor(
            queue[-1]["gt_sdc_fut_traj_mask"]
        )  # L, 1 x 12 x 2

        # gt_future_boxes_list = queue[-1]["gt_future_boxes"]  # 7, N x 9

        # gt_future_labels_list = [
        #     to_tensor(each) for each in queue[-1]["gt_future_labels"]
        # ]  # 7, N* (could be larger than N), only those valid are >=0, others are -1

        # converting the global absolute coordinate into relative position and orientation change
        metas_map = {}
        prev_pos = None
        prev_angle = None
        for i, each in enumerate(queue):
            metas_map[i] = each["img_metas"].data
            if i == 0:
                metas_map[i]["prev_bev"] = False
                prev_pos = copy.deepcopy(metas_map[i]["can_bus"][:3])
                prev_angle = copy.deepcopy(metas_map[i]["can_bus"][-1])
                metas_map[i]["can_bus"][:3] = 0
                metas_map[i]["can_bus"][-1] = 0
            else:
                metas_map[i]["prev_bev"] = True
                tmp_pos = copy.deepcopy(metas_map[i]["can_bus"][:3])
                tmp_angle = copy.deepcopy(metas_map[i]["can_bus"][-1])
                metas_map[i]["can_bus"][:3] -= prev_pos
                metas_map[i]["can_bus"][-1] -= prev_angle
                prev_pos = copy.deepcopy(tmp_pos)
                prev_angle = copy.deepcopy(tmp_angle)
            # print(metas_map[i]["can_bus"][:3])

        queue[-1]["img"] = DC(
            torch.stack(imgs_list), cpu_only=False, stack=True
        )  # T x N x 3 x H x W
        queue[-1]["img_metas"] = DC(metas_map, cpu_only=True)

        # inherent all the labels for the current frame only
        queue = queue[-1]

        # merge all other labels that requires temporal and merge from queue
        queue["gt_labels_3d"] = DC(gt_labels_3d_list)
        queue["gt_sdc_label"] = DC(gt_sdc_label_list)
        queue["gt_inds"] = DC(gt_inds_list)
        queue["gt_bboxes_3d"] = DC(gt_bboxes_3d_list, cpu_only=True)
        queue["gt_sdc_bbox"] = DC(gt_sdc_bbox_list, cpu_only=True)
        queue["l2g_r_mat"] = DC(l2g_r_mat_list)
        queue["l2g_t"] = DC(l2g_t_list)
        queue["timestamp"] = DC(timestamp_list)
        queue["gt_fut_traj"] = DC(gt_fut_traj)
        queue["gt_fut_traj_mask"] = DC(gt_fut_traj_mask)
        # queue["gt_past_traj"] = DC(gt_past_traj_list)
        # queue["gt_past_traj_mask"] = DC(gt_past_traj_mask_list)

        # queue["gt_future_boxes"] = DC(gt_future_boxes_list, cpu_only=True)
        # queue["gt_future_labels"] = DC(gt_future_labels_list)
        return queue

    def get_ann_info(self, index):
        """Get annotation info according to the given index.

        Args:
            index (int): Index of the annotation data to get.

        Returns:
            dict: Annotation information consists of the following keys:

                - gt_bboxes_3d (:obj:`LiDARInstance3DBoxes`): \
                    3D ground truth bboxes
                - gt_labels_3d (np.ndarray): Labels of ground truths.
                - gt_names (list[str]): Class names of ground truths.
                - gt_inds (np.ndarray): Instance ids of ground truths.
                - gt_fut_traj (np.ndarray): .
                - gt_fut_traj_mask (np.ndarray): .
        """
        annotation = dict()
        annotation = self.update_detection(annotation, index, use_mask=True)
        annotation = self.update_anno_token(annotation, index)
        annotation = self.update_tracking_prediction(annotation, index)
        return annotation

    def get_data_info(self, index, info=None, debug=False):
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
            sample_idx=info["token"],  # str: nuScenes unique sample token
            prev_idx=info["prev"],  # str: nuScenes unique sample token
            next_idx=info["next"],  # str: nuScenes unique sample token
            scene_token=info["scene_token"],  # str: nuScenes sequence name
            frame_idx=info["frame_idx"],  # int: 0-indexed frame IDs
            pts_filename=info["lidar_path"],  # str: relative path for the lidar data
            sweeps=info["sweeps"],  # List[Dict]: list of infos for sweep data
            timestamp=info["timestamp"] / 1e6,  # int: nuScenes unique time index
        )

        input_dict = self.update_transform(input_dict=input_dict, index=index)
        input_dict = self.update_sensor(input_dict=input_dict, index=index)
        input_dict = self.update_canbus(input_dict=input_dict, index=index)

        # generate annotation for detection/tracking, putting them in the annotation so that
        # we could do range filtering altogether in the transform_3d.py
        input_dict["ann_info"] = self.get_ann_info(index)

        # all data ultimately needs to be in the input_dict for downstream heads
        input_dict = self.update_mapping(input_dict=input_dict, index=index)
        input_dict = self.update_occupancy(input_dict=input_dict, index=index)
        input_dict = self.update_ego_prediction(input_dict=input_dict, index=index)
        input_dict = self.update_ego_planning(input_dict=input_dict, index=index)

        return input_dict

    def update_transform(self, input_dict, index):
        # nuScenes lidar x (right), y (front), z (up), main coordinate used for training

        # lidar to ego
        info = self.data_infos[index]
        l2e_r = info["lidar2ego_rotation"]  # List[float], (4, ),
        l2e_t = info["lidar2ego_translation"]  # List[float], (3, ),
        l2e_r_mat: np.array = Quaternion(l2e_r).rotation_matrix  # 3 x 3
        l2e = np.identity(4)
        l2e[:3, :3] = l2e_r_mat
        l2e[:3, 3] = l2e_t  # 4 x 4
        e2l: np.ndarray = np.linalg.inv(l2e)  # 4 x 4

        # ego to global
        e2g_r = info[
            "ego2global_rotation"
        ]  # List[float], (4, ) quaternion, in global coordinate
        e2g_t = info[
            "ego2global_translation"
        ]  # List[float], (3, ), in global coordinate
        e2g_r_mat = Quaternion(e2g_r).rotation_matrix
        e2g = np.identity(4)
        e2g[:3, :3] = e2g_r_mat
        e2g[:3, 3] = e2g_t  # 4 x 4
        g2e: np.ndarray = np.linalg.inv(e2g)  # 4 x 4

        # lidar to global
        l2g_r_mat = l2e_r_mat.T @ e2g_r_mat.T
        l2g_t = l2e_t @ e2g_r_mat.T + e2g_t
        l2g = np.identity(4)
        l2g[:3, :3] = l2g_r_mat
        l2g[:3, 3] = l2g_t  # 4 x 4
        g2l: np.ndarray = np.linalg.inv(l2g)  # 4 x 4

        # output dictionary
        input_dict.update(
            dict(
                # lidar to global
                l2g_r_mat=l2g_r_mat,  # np.ndarray: 3 x 3
                l2g_t=l2g_t,  # np.ndarray: 3
                l2g=l2g,  # np.ndarray: 4 x 4
                g2l=g2l,  # np.ndarray: 4 x 4
                # lidar to ego
                l2e_r_mat=np.array(l2e_r_mat),  # np.ndarray: 3 x 3
                l2e_t=np.array(l2e_t),  # np.ndarray: 3
                l2e=l2e,  # np.ndarray: 4 x 4
                e2l=e2l,  # np.ndarray: 4 x 4
                # ego to global
                e2g_r_mat=np.array(e2g_r_mat),  # np.ndarray: 3 x 3
                e2g_t=np.array(e2g_t),  # np.ndarray: 3
                e2g=e2g,  # np.ndarray: 4 x 4
                g2e=g2e,  # np.ndarray: 4 x 4
            )
        )

        return input_dict

    def update_sensor(self, input_dict, index):
        info = self.data_infos[index]

        if self.modality["use_camera"]:
            image_paths = []
            lidar2img_rts = []
            lidar2cam_rts = []
            cam_intrinsics = []

            # loop through all cameras
            for cam_type, cam_info in info["cams"].items():
                image_paths.append(cam_info["data_path"])

                # obtain lidar to image transformation matrix
                lidar2cam_r = np.linalg.inv(cam_info["sensor2lidar_rotation"])  # 3 x 3
                lidar2cam_t = (
                    cam_info["sensor2lidar_translation"] @ lidar2cam_r.T
                )  # (3, )
                lidar2cam_rt = np.eye(4)
                lidar2cam_rt[:3, :3] = lidar2cam_r.T
                lidar2cam_rt[3, :3] = -lidar2cam_t
                intrinsic = cam_info["cam_intrinsic"]
                viewpad = np.eye(4)
                viewpad[: intrinsic.shape[0], : intrinsic.shape[1]] = intrinsic
                lidar2img_rt = viewpad @ lidar2cam_rt.T
                lidar2img_rts.append(lidar2img_rt)

                cam_intrinsics.append(viewpad)
                lidar2cam_rts.append(lidar2cam_rt.T)

            input_dict.update(
                dict(
                    img_filename=image_paths,  # List[str]: list of relative paths towards camera images
                    lidar2img=lidar2img_rts,  # List[np.ndarray]: C, 4 x 4
                    cam_intrinsic=cam_intrinsics,  # List[np.ndarray]: C, 4 x 4
                    lidar2cam=lidar2cam_rts,  # List[np.ndarray]: C, 4 x 4
                )
            )
        return input_dict

    def update_canbus(self, input_dict, index):
        # TODO: to add steering angle from can_bus here

        info = self.data_infos[index]
        can_bus = info["can_bus"]  # (18, )
        # 0-3: global position
        # 3-7: quarternion for rotation, global coordinate
        # 7-10: acceleration, ego vehicle frame
        # 10-13: velocity, ego vehicle frame, always 0 in yz
        # 13-16: angular velocity, rad/s, ego vehicle frame
        # 16-18: 0 are assigned, nothing meaningful

        ##### location
        sdc_loc_global = np.array([[0, 0, 0, 1.0]]).transpose()  # 4 x 1
        translation = info["ego2global_translation"]
        sdc_loc_global[:3, 0] = translation  # global coordinate
        sdc_loc_ego = input_dict["g2e"] @ sdc_loc_global
        if self.check_canbus:
            assert (
                abs(scipy.linalg.norm(sdc_loc_ego) - 1) < 1e-4
            ), "sdc location in ego coordinate is wrong"
        sdc_loc_lidar = input_dict["e2l"] @ sdc_loc_ego
        # print('\n\nprint out can_bus now\n')
        # print('can_bus_loc\n', can_bus[:3])
        # print("sdc_loc_global\n", sdc_loc_global)
        # print("sdc_loc_ego\n", sdc_loc_ego)  # should be always [0, 0, 0, 1]
        # print("sdc_loc_lidar\n", sdc_loc_lidar)  # should be [0, -0.9, -1.86]

        ##### rotation
        sdc_rot_global_quat = Quaternion(info["ego2global_rotation"])
        sdc_rot_global_ele = sdc_rot_global_quat.elements
        yaw_angle_global = quaternion_yaw(sdc_rot_global_quat) / np.pi * 180
        if yaw_angle_global < 0:
            yaw_angle_global += 360

        # not 100% necessary below
        sdc_rot_global = sdc_rot_global_quat.yaw_pitch_roll
        sdc_rot_global_mat = sdc_rot_global_quat.rotation_matrix
        sdc_rot_ego_mat = (
            input_dict["g2e"][:3, :3] @ sdc_rot_global_mat
        )  # ego coordinate
        sdc_rot_ego = Quaternion(
            object_rotation.convert_mat2qua(sdc_rot_ego_mat)
        ).yaw_pitch_roll
        if self.check_canbus:
            assert (
                scipy.linalg.norm(sdc_rot_ego) < 1e-4
            ), "sdc rotation in ego coordinate is wrong"
        sdc_rot_lidar_mat = (
            input_dict["e2l"][:3, :3] @ sdc_rot_ego_mat
        )  # lidar coordinate
        sdc_rot_lidar = Quaternion(
            object_rotation.convert_mat2qua(sdc_rot_lidar_mat)
        ).yaw_pitch_roll

        ##### velocity, acceleration and angular velocity from the ego coordinate
        # acceleration has gravity in the last dim
        sdc_acc_ego: np.ndarray = can_bus[7:10]  # ego coordinate
        # velocity in the ego coordinate
        sdc_vel_ego: np.ndarray = can_bus[10:13]  # ego coordinate
        # angular velocity is the yaw change per second
        sdc_ang_ego: np.ndarray = can_bus[13:16]  # ego coordinate
        # clear the gravity and irrelevant axis to avoid bad things in coordinatr transform
        # accleration still not perfectly match with velocity, possibly due to measurement noise

        # TODO: change this would change the training loss, possibly prior models would not work
        if not self.old_canbus and self.sdc_acc_ego_fix:
            sdc_acc_ego[1:] = 0  # this will change can_bus value
        # sdc_acc_ego = copy.copy(sdc_acc_ego)

        sdc_acc_lidar = input_dict["e2l"][:3, :3] @ sdc_acc_ego
        sdc_acc_global = input_dict["e2g_r_mat"] @ sdc_acc_ego
        sdc_ang_lidar = input_dict["e2l"][:3, :3] @ sdc_ang_ego  # last dim for yaw
        sdc_ang_global = input_dict["e2g_r_mat"] @ sdc_ang_ego  # last dim for yaw
        sdc_vel_lidar = input_dict["e2l"][:3, :3] @ sdc_vel_ego
        sdc_vel_global = input_dict["e2g_r_mat"] @ sdc_vel_ego

        ##### assign values to can_bus
        # 0-3: global position
        # 3-7: quarternion for rotation, global frame
        # 7-10: acceleration, ego vehicle frame
        # 10-13: velocity, ego vehicle frame
        # 13-16: angular velocity, rad/s, ego vehicle frame
        # 17: global yaw in radian
        # 18: global yaw in degree
        can_bus[:3] = sdc_loc_global[:3, 0]  # global coordinate
        can_bus[3:7] = sdc_rot_global_ele
        can_bus[-2] = yaw_angle_global / 180 * np.pi  # in radian
        can_bus[-1] = yaw_angle_global  # in degree

        # output dictionary
        input_dict.update(
            dict(
                can_bus=can_bus,  # (18, ), List of 18 elements
                sdc_loc_global=np.array(sdc_loc_global),  # (3, ), tuple
                sdc_loc_ego=np.array(sdc_loc_ego),  # (3, ), tuple
                sdc_loc_lidar=np.array(sdc_loc_lidar),  # (3, ), tuple
                sdc_rot_global=np.array(sdc_rot_global),  # (3, ), tuple
                sdc_rot_ego=np.array(sdc_rot_ego),  # (3, ), tuple
                sdc_rot_lidar=np.array(sdc_rot_lidar),  # (3, ), tuple
                sdc_acc_global=np.array(sdc_acc_global),  # (3, ), tuple
                sdc_acc_ego=np.array(sdc_acc_ego),  # (3, ), tuple
                sdc_acc_lidar=np.array(sdc_acc_lidar),  # (3, ), tuple
                sdc_vel_global=np.array(sdc_vel_global),  # (3, ), tuple
                sdc_vel_ego=np.array(sdc_vel_ego),  # (3, ), tuple
                sdc_vel_lidar=np.array(sdc_vel_lidar),  # (3, ), tuple
                sdc_ang_global=np.array(sdc_ang_global),  # (3, ), tuple
                sdc_ang_ego=np.array(sdc_ang_ego),  # (3, ), tuple
                sdc_ang_lidar=np.array(sdc_ang_lidar),  # (3, ), tuple
            )
        )
        return input_dict

    def update_detection(self, annotation, index, use_mask=True):
        # update keys for detection/tracking in the annotation dictionary

        info = self.data_infos[index]

        # get mask for filtering
        if self.use_valid_flag:
            mask = info["valid_flag"]
        else:
            mask = info["num_lidar_pts"] > 0

        ################ GT for detection/tracking
        # retrieve information
        gt_bboxes_3d: np.ndarray = info["gt_boxes"].copy()  # N x 7
        gt_names_3d: np.ndarray = info["gt_names"].copy()  # N
        gt_inds: np.ndarray = info["gt_inds"].copy()  # N (int)
        gt_vis_tokens = info.get("visibility_tokens", None)
        gt_vis_tokens = gt_vis_tokens.copy() if gt_vis_tokens is not None else None

        # print(gt_names_3d)

        # filter out bbox containing no points
        if use_mask:
            gt_bboxes_3d = gt_bboxes_3d[mask]
            gt_names_3d = gt_names_3d[mask]
            gt_inds = gt_inds[mask]
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

        if self.with_velocity:
            gt_velocity = info["gt_velocity"]
            if use_mask:
                gt_velocity = gt_velocity[mask]
            nan_mask = np.isnan(gt_velocity[:, 0])
            gt_velocity[nan_mask] = [0.0, 0.0]
            gt_bboxes_3d = np.concatenate([gt_bboxes_3d, gt_velocity], axis=-1)

        # the nuscenes box center is [0.5, 0.5, 0.5], we change it to be
        # the same as KITTI (0.5, 0.5, 0)
        gt_bboxes_3d = LiDARInstance3DBoxes(
            gt_bboxes_3d, box_dim=gt_bboxes_3d.shape[-1], origin=(0.5, 0.5, 0.5)
        ).convert_to(self.box_mode_3d)
        # xyz,lwh,yaw, lidar coordinate
        # print(gt_bboxes_3d.tensor[:10])

        # output dictionary
        anno_tmp = dict(
            gt_bboxes_3d=gt_bboxes_3d,  # LiDARInstance3DBoxes, Nx9, xyz, lwh, yaw, vel_xy, lidar coordinate
            gt_labels_3d=gt_labels_3d,  # (N, ): int
            gt_names=gt_names_3d,  # (N, ): (str, no capital)
            gt_inds=gt_inds,  # (N, ): int
            mask=mask,  # (N, ): binary
            gt_vis_tokens=gt_vis_tokens,
        )
        if annotation is None:
            return anno_tmp

        # update dictionary if existing
        else:
            annotation.update(anno_tmp)
        return annotation

    def update_anno_token(self, annotation, index):
        # add nuScenes-specific token information, require dataset-specific APIs

        info = self.data_infos[index]

        # retrieve sample
        sample = self.nusc.get("sample", info["token"])
        ann_tokens = np.array(sample["anns"])[annotation["mask"]]
        assert ann_tokens.shape[0] == annotation["gt_bboxes_3d"].tensor.shape[0]

        # output dictionary
        annotation.update(
            dict(
                ann_tokens=ann_tokens,  # str
            )
        )
        return annotation

    def update_tracking_prediction(self, annotation, index):
        info = self.data_infos[index]

        # get trajectories for surrounding agents, lidar coordinate
        (
            gt_fut_traj,  # N x 12 x 2
            gt_fut_traj_mask,  # N x 12 x 2
            gt_past_traj,  # N x 8 x 2
            gt_past_traj_mask,  # N x 8 x 2
        ) = self.traj_api.get_traj_label(info["token"], annotation["ann_tokens"])
        assert gt_fut_traj.shape[0] == annotation["gt_labels_3d"].shape[0]
        assert gt_past_traj.shape[0] == annotation["gt_labels_3d"].shape[0]
        # gt_past_traj is used for tracking, including 4 steps of future and 4 steps of past
        # t-1, t-2, t-3, t-4, t+1, t+2, t+3, t+4
        # gt_fut_traj is used for end-to-end prediction
        # t+1, t+2, t+3, ..., t+12

        # update output dictionary
        annotation.update(
            dict(
                gt_fut_traj=gt_fut_traj,  # N x 12 x 2, lidar coordinate
                gt_fut_traj_mask=gt_fut_traj_mask,  # N x 12 x 2
                gt_past_traj=gt_past_traj,  # N x 8 x 2
                gt_past_traj_mask=gt_past_traj_mask,  # N x 8 x 2
            )
        )
        return annotation

    def update_ego_prediction(self, input_dict, index):
        info = self.data_infos[index]

        # velocity derived from differences of two positions over time
        # NOT the same compared to measured velocity using the sensor
        sdc_vel_lidar_calculated = self.traj_api.sdc_vel_info[info["token"]]  # (2, )
        # print("sdc_vel_lidar_calculated", sdc_vel_lidar_calculated)

        # ego bounding box
        gt_sdc_bbox, gt_sdc_label = self.traj_api.generate_sdc_info(
            sdc_vel_lidar_calculated,
            old_canbus=self.old_canbus,
        )  # 1 x 9, 1
        if self.check_canbus:
            assert (
                scipy.linalg.norm(
                    input_dict["sdc_rot_lidar"][0] - gt_sdc_bbox.data.tensor[0, 6]
                )
                < 1e-2
            ), "sdc rotation in lidar coordinate is wrong"
        # print("gt_sdc_bbox\n", gt_sdc_bbox.data)

        # ego's future trajectory for prediction
        gt_sdc_fut_traj, gt_sdc_fut_traj_mask = self.traj_api.get_sdc_traj_label(
            info["token"]
        )  # 1 x 12 x 2, lidar coordinate
        # print('gt_sdc_fut_traj\n', gt_sdc_fut_traj)

        # update output dictionary for ego's prediction
        input_dict.update(
            dict(
                sdc_vel_lidar_calculated=sdc_vel_lidar_calculated,  # (2, )
                gt_sdc_bbox=gt_sdc_bbox,  # DC (LiDARInstance3DBoxes), xyz, lwh, yaw, vel_x, vel_y, lidar coordinate
                gt_sdc_label=gt_sdc_label,  # DC (tensor[0])
                gt_sdc_fut_traj=gt_sdc_fut_traj,  # 1 x 12 x 2, lidar coordinate
                gt_sdc_fut_traj_mask=gt_sdc_fut_traj_mask,  # 1 x 12 x 2
            )
        )
        return input_dict

    def update_command(self, command: int, info: Dict) -> int:
        # correct some of the auto-labelled navigation commands with manual labels

        if self.route_info_dict is not None:
            sample_token = info["token"]
            # print("\n\nnew frame")
            # print("scene_token", self.scene_token)
            # print("frame_idx", self.frame_idx)
            # print('sample_token: ', sample_token)
            # print('old command: ', command)
            if sample_token in self.route_info_dict:
                command: str = self.route_info_dict[sample_token]
                command: int = self.command_str2int[command]
                # command = torch.tensor([new_command]).to(command.device)
                # print('new command: ', command)

        return command

    def update_ego_planning(self, input_dict, index):
        info = self.data_infos[index]

        # get planning for ego sdc in the lidar coordinate
        (
            sdc_planning,
            sdc_planning_mask,
            command,
            sdc_planning_past,
            sdc_planning_mask_past,
            sdc_planning_world,
        ) = self.traj_api.get_sdc_planning_label(info["token"])
        # sdc_planning: 1 x 6 x 3, lidar coordinate, xy-yaw
        # sdc_planning_past: 1 x 4 x 3, lidar coordinate, xy-yaw
        # sdc_planning_mask_past: 1 x 4 x 2
        # t-4, t-3, t-2, t-1, reserve one more frame to help compute the diff
        # print("sdc_planning\n", sdc_planning)
        # print("sdc_planning_world\n", sdc_planning_world)
        # print("sdc_planning_past\n", sdc_planning_past)

        # correct the input navigation command
        command: int = self.update_command(command, info)

        # compute the difference between frames, add the current frame for the diff
        # from xy-yaw, to vel-xy, angular velocity
        # multiple by 2 due to the 2Hz to get the velocity
        gt_sdc_xyyaw = input_dict["gt_sdc_bbox"].data.tensor[0, [0, 1, 6]]
        sdc_planning_cur = np.array(gt_sdc_xyyaw).reshape((1, 1, 3))
        sdc_planning_past_diff = np.concatenate(
            (sdc_planning_past, sdc_planning_cur), axis=1
        )  # 1 x 5 x 3
        # normalize the yaw to be -pi + pi
        for time_index in range(sdc_planning_past_diff.shape[1]):
            if sdc_planning_past_diff[0, time_index, -1] < -np.pi:
                sdc_planning_past_diff[0, time_index, -1] += 2 * np.pi
            if sdc_planning_past_diff[0, time_index, -1] > np.pi:
                sdc_planning_past_diff[0, time_index, -1] -= 2 * np.pi
        # print("sdc_planning_past_diff\n", sdc_planning_past_diff)
        sdc_planning_past_diff = 2 * np.diff(
            sdc_planning_past_diff, axis=1
        )  # 1 x 4 x 3
        # print("sdc_planning_past_diff\n", sdc_planning_past_diff)

        # build new past states in lidar coordinate
        # dim 1-2 are the xy in lidar coordinate
        # dim 3-4 are the vel in lidar coordinate
        # dim 5 is the yaw in lidar coordinate
        # dim 6-7 are the sin/cos of the yaw
        # dim 8 is the angular velocity
        sdc_planning_past_all = np.zeros(
            (1, sdc_planning_past.shape[1], 8)
        )  # 1 x 4 x 8
        sdc_planning_past_all[:, :, :2] = sdc_planning_past[:, :, :2]
        sdc_planning_past_all[:, :, 2:4] = sdc_planning_past_diff[:, :, :2]
        sdc_planning_past_all[:, :, 4] = sdc_planning_past[:, :, 2]
        sdc_planning_past_all[:, :, 5] = np.sin(sdc_planning_past[:, :, 2])
        sdc_planning_past_all[:, :, 6] = np.cos(sdc_planning_past[:, :, 2])
        sdc_planning_past_all[:, :, 7] = sdc_planning_past_diff[:, :, 2]
        # print("sdc_planning_past_all\n", sdc_planning_past_all)

        # sanity check
        sdc_vel_lidar_delta = scipy.linalg.norm(
            sdc_planning_past_all[0, -1, 2:4] - input_dict["sdc_vel_lidar"][:2]
        )
        sdc_ang_lidar_delta = abs(
            sdc_planning_past_all[0, -1, -1] - input_dict["sdc_ang_lidar"][-1]
        )
        # try:
        #     assert sdc_vel_lidar_delta < 2, f"sdc velocity {sdc_vel_lidar_delta} is wrong"
        #     assert (
        #         sdc_ang_lidar_delta < 0.1
        #     ), f"sdc angular velocity {sdc_ang_lidar_delta} is wrong"
        # except AssertionError:
        #     print(sdc_vel_lidar_delta)
        #     print(sdc_ang_lidar_delta)

        # update output dictionary
        input_dict.update(
            dict(
                command=command,  # int
                sdc_planning_world=sdc_planning_world,  # 1 x 6 x 3, global coordinate, xyz
                sdc_planning=sdc_planning,  # 1 x 6 x 3, lidar coordinate, xy-yaw
                sdc_planning_mask=sdc_planning_mask,  # 1 x 6 x 2
                sdc_planning_past=sdc_planning_past_all,  # 1 x 4 x 8, lidar_coordinate, xy, v_x, v_y, yaw, sin/cos yaw, ang_vel
                sdc_planning_mask_past=sdc_planning_mask_past.copy(),  # 1 x 4 x 2
            )
        )
        return input_dict

    def update_mapping(self, input_dict, index):
        # get polyline in the ego coordinate
        polyline_dict = self.update_map_vectorized(input_dict, index)

        input_dict = self.update_map_rasterized(input_dict, index, polyline_dict)

        return input_dict

    def update_map_vectorized(self, input_dict, index):
        info = self.data_infos[index]

        # retrieve the city string
        location: str = self.nusc.get(
            "log", self.nusc.get("scene", info["scene_token"])["log_token"]
        )["location"]

        # retrieve vector polylines from nuScenes given the city and ego's location
        polyline_dict: List[Dict] = self.vector_map.gen_vectorized_samples(
            location, info["ego2global_translation"], info["ego2global_rotation"]
        )
        # Dict: pts: num_pts x 2, pts_num: int, type: int, 0/1/2, ego coordinate

        return polyline_dict

    def update_map_rasterized(self, input_dict, index, polyline_dict, debug=False):
        ##### get rasterize maps for instances in the lidar coordinate
        gt_labels, gt_bboxes, gt_masks = self.update_map_rasterize_instance(
            index,
            polyline_dict,  # ego coordinate
        )

        ##### get semantic mask in the lidar coordinate
        map_mask = self.update_map_rasterize_semantic(
            index,
            input_dict,
            polyline_dict,  # ego coordinate
        )

        map_mask = np.flip(map_mask, axis=1)  # 2 x H x W
        map_mask = np.rot90(map_mask, k=-1, axes=(1, 2))  # 2 x H x W, 0-1 clockwise

        # merge instance and semantic results
        # add one other semantic channel for the driveable area
        # not using the 2nd channel of the road/lane dividers
        map_mask = torch.tensor(map_mask)  # 2 x H x W, 0-1
        for i, gt_mask in enumerate(map_mask[:1]):
            ys, xs = np.where(gt_mask)
            # some areas
            try:
                gt_bbox = [min(xs), min(ys), max(xs), max(ys)]
            except ValueError:
                gt_bbox = [0, 0, 1, 1]

            gt_labels.append(i + self.map_num_classes)
            gt_bboxes.append(gt_bbox)
            gt_masks.append(gt_mask)

        gt_labels = torch.tensor(gt_labels)
        gt_bboxes = torch.tensor(np.stack(gt_bboxes))
        gt_masks = torch.stack(gt_masks)  # N x H x W

        # update output dictionary
        input_dict.update(
            dict(
                # map_filename=lane_info["maps"]["map_mask"] if lane_info else None,
                gt_lane_labels=gt_labels,  # (7, ), not used
                gt_lane_bboxes=gt_bboxes,  # (7, 4), not used
                gt_lane_masks=gt_masks,  # (7, 60, 60), not used
            )
        )
        return input_dict

    def update_map_rasterize_instance(
        self,
        index,
        polyline_dict,
        debug=False,
    ):
        info = self.data_infos[index]

        # semantic format, current not used
        # lane_info = self.lane_infos[index] if self.lane_infos else None

        # rasterization of the map polylines, here considering road/lane divider,
        # ped crossing, contours, has multiple instances of three types
        # for contour, it has many segmented portions and it does not have the lane-level
        # information, all dividers are separated already

        _, instance_masks, _, _ = preprocess_map(
            polyline_dict,
            self.patch_size,
            self.canvas_size,
            self.map_num_classes,
            self.thickness,
            self.angle_class,
        )

        # ego to lidar coordinate
        instance_masks = np.rot90(
            instance_masks, k=-1, axes=(1, 2)
        )  # 3 x H x W, not 0/1, but actual instance ID
        instance_masks = torch.tensor(instance_masks.copy())

        # convert 3-channel instances masks to multi-channel instance masks
        # for individual instances, and also get its bbox
        gt_labels = []
        gt_bboxes = []
        gt_masks = []
        image_total_vis = np.zeros((self.canvas_size[0], self.canvas_size[1], 3))
        for cls in range(self.map_num_classes):
            # loop through instances falling into the same category
            for i in np.unique(instance_masks[cls]):
                if i == 0:
                    continue
                gt_mask = (instance_masks[cls] == i).to(torch.uint8)
                ys, xs = np.where(gt_mask)
                gt_bbox = [min(xs), min(ys), max(xs), max(ys)]
                gt_labels.append(cls)
                gt_bboxes.append(gt_bbox)
                gt_masks.append(gt_mask)

        return gt_labels, gt_bboxes, gt_masks

    def update_map_rasterize_semantic(
        self, index, input_dict, polyline_dict
    ) -> np.ndarray:
        info = self.data_infos[index]

        # get semantic masks for driveable area, lane divider, road divider
        # 1st channel is driveable area, binary mask
        # 2nd channel is the merge of lane divider and road divider
        # in the lidar coordinate
        map_mask: np.ndarray = obtain_map_info(
            self.nusc,
            self.nusc_maps,
            info,
            patch_size=self.patch_size,
            canvas_size=self.canvas_size,
            layer_names=["lane_divider", "road_divider"],
        )  # 2 x H x W, uint8, 0-1

        return map_mask

    def update_occupancy(self, input_dict, index):
        # data index, past 2, future 6, not including the current
        prev_indices, future_indices = self.occ_get_temporal_indices(
            index, self.occ_receptive_field, self.occ_n_future
        )

        # ego motions of all frames are needed
        all_frames = prev_indices + [index] + future_indices
        # print("occ_n_future", self.occ_n_future)
        # print("prev_indices", prev_indices)
        # print("future_indices", future_indices)
        # print("all_frames", all_frames)

        # whether invalid frames is present
        has_invalid_frame = -1 in all_frames[: self.occ_only_total_frames]
        # NOTE: This can only represent 7 frames in total as it influence evaluation
        # print("has_invalid_frame", has_invalid_frame)

        # might have None if not in the same sequence
        curfut_frames = [index] + future_indices

        # get lidar to ego to global transforms for each curr and fut index
        occ_transforms = self.occ_get_transforms(curfut_frames)  # might have None
        input_dict.update(occ_transforms)
        # DERIVED: occ_l2e_r_mats: 7, 3x3
        # DERIVED: occ_l2e_t_vecs: 7, 3
        # DERIVED: occ_e2g_r_mats: 7, 3x3
        # DERIVED: occ_e2g_t_vecs: 7, 3

        # for (current and) future frames, detection labels are needed
        # generate detection labels for current + future frames
        occ_future_ann_infos = []
        for curfut_frame in curfut_frames:
            if curfut_frame >= 0:
                occ_future_ann_infos.append(
                    # self.occ_get_detection_ann_info(curfut_frame),
                    self.update_detection(
                        index=curfut_frame,
                        annotation=None,
                        use_mask=self.occ_filter_by_valid_flag,
                    ),
                )
            else:
                occ_future_ann_infos.append(None)

        # update output dictionary
        input_dict.update(
            dict(
                occ_has_invalid_frame=has_invalid_frame,
                occ_img_is_valid=np.array(all_frames) >= 0,  # (9, )
                occ_future_ann_infos=occ_future_ann_infos,  # List of Dict,
            )
        )
        return input_dict

    def occ_get_temporal_indices(self, index, receptive_field, n_future):
        current_scene_token = self.data_infos[index]["scene_token"]

        # generate the past
        previous_indices = []

        for t in range(-receptive_field + 1, 0):
            index_t = index + t
            if (
                index_t >= 0
                and self.data_infos[index_t]["scene_token"] == current_scene_token
            ):
                previous_indices.append(index_t)
            else:
                previous_indices.append(-1)  # for invalid indices

        # generate the future
        future_indices = []

        for t in range(1, n_future + 1):
            index_t = index + t
            if (
                index_t < len(self.data_infos)
                and self.data_infos[index_t]["scene_token"] == current_scene_token
            ):
                future_indices.append(index_t)
            else:
                # NOTE: How to deal the invalid indices???
                future_indices.append(-1)

        return previous_indices, future_indices

    def occ_get_transforms(self, indices, data_type=torch.float32):
        """
        get l2e, e2g rotation and translation for each valid frame
        """
        l2e_r_mats = []
        l2e_t_vecs = []
        e2g_r_mats = []
        e2g_t_vecs = []

        for index in indices:
            if index == -1:
                l2e_r_mats.append(None)
                l2e_t_vecs.append(None)
                e2g_r_mats.append(None)
                e2g_t_vecs.append(None)
            else:
                info = self.data_infos[index]
                l2e_r = info["lidar2ego_rotation"]
                l2e_t = info["lidar2ego_translation"]
                e2g_r = info["ego2global_rotation"]
                e2g_t = info["ego2global_translation"]

                l2e_r_mat = torch.from_numpy(Quaternion(l2e_r).rotation_matrix)
                e2g_r_mat = torch.from_numpy(Quaternion(e2g_r).rotation_matrix)

                l2e_r_mats.append(l2e_r_mat.to(data_type))
                l2e_t_vecs.append(torch.tensor(l2e_t).to(data_type))
                e2g_r_mats.append(e2g_r_mat.to(data_type))
                e2g_t_vecs.append(torch.tensor(e2g_t).to(data_type))

        res = {
            "occ_l2e_r_mats": l2e_r_mats,
            "occ_l2e_t_vecs": l2e_t_vecs,
            "occ_e2g_r_mats": e2g_r_mats,
            "occ_e2g_t_vecs": e2g_t_vecs,
        }

        return res

    def __getitem__(self, idx):
        """Get item from infos according to the given index.
        Returns:
            dict: Data dictionary of the corresponding index.
        """
        if self.test_mode:
            print('test mode')
            return self.prepare_test_data(idx)
        while True:
            print('train mode')
            data = self.prepare_train_data(idx)
            if data is None:
                idx = self._rand_another(idx)
                continue
            return data

    def _format_bbox(self, results, jsonfile_prefix=None):
        """Convert the results to the standard format.
        Args:
            results (list[dict]): Testing results of the dataset.
            jsonfile_prefix (str): The prefix of the output jsonfile.
                You can specify the output directory/filename by
                modifying the jsonfile_prefix. Default: None.
        Returns:
            str: Path of the output json file.
        """
        nusc_annos = {}
        nusc_map_annos = {}
        mapped_class_names = self.CLASSES

        print("_format_bbox, Start to convert detection format...")
        for sample_id, det in enumerate(mmcv.track_iter_progress(results)):
            annos = []
            sample_token = self.data_infos[sample_id]["token"]

            if "map" in self.eval_mod:
                map_annos = {}
                for key, value in det["ret_iou"].items():
                    map_annos[key] = float(value.numpy()[0])
                    nusc_map_annos[sample_token] = map_annos

            if "boxes_3d" not in det:
                nusc_annos[sample_token] = annos
                continue

            boxes = output_to_nusc_box(det)
            boxes_ego = copy.deepcopy(boxes)
            boxes, keep_idx = lidar_nusc_box_to_global(
                self.data_infos[sample_id],
                boxes,
                mapped_class_names,
                self.eval_detection_configs,
                self.eval_version,
            )
            for i, box in enumerate(boxes):
                name = mapped_class_names[box.label]

                if np.sqrt(box.velocity[0] ** 2 + box.velocity[1] ** 2) > 0.2:
                    if name in [
                        "car",
                        "construction_vehicle",
                        "bus",
                        "truck",
                        "trailer",
                        "Car",
                    ]:
                        attr = "vehicle.moving"
                    elif name in ["bicycle", "motorcycle", "Motorcycle", "Cyclist"]:
                        attr = "cycle.with_rider"
                    else:
                        attr = self.default_attr[name]
                else:
                    if name in ["pedestrian", "Pedestrian"]:
                        attr = "pedestrian.standing"
                    elif name in ["bus"]:
                        attr = "vehicle.stopped"
                    else:
                        attr = self.default_attr[name]

                # center_ = box.center.tolist()
                # change from ground height to center height
                # center_[2] = center_[2] + (box.wlh.tolist()[2] / 2.0)
                if name not in [
                    "car",
                    "truck",
                    "bus",
                    "trailer",
                    "motorcycle",
                    "bicycle",
                    "pedestrian",
                    "Car",
                    "Motorcycle",
                    "Cyclist",
                    "Pedestrian",
                ]:
                    continue

                box_ego = boxes_ego[keep_idx[i]]
                trans = box_ego.center
                if "traj" in det:
                    traj_local = det["traj"][keep_idx[i]].numpy()[..., :2]
                    traj_scores = det["traj_scores"][keep_idx[i]].numpy()
                else:
                    traj_local = np.zeros((0,))
                    traj_scores = np.zeros((0,))
                traj_ego = np.zeros_like(traj_local)
                rot = Quaternion(axis=np.array([0, 0.0, 1.0]), angle=np.pi / 2)
                for kk in range(traj_ego.shape[0]):
                    traj_ego[kk] = convert_local_coords_to_global(
                        traj_local[kk], trans, rot
                    )

                nusc_anno = dict(
                    sample_token=sample_token,
                    translation=box.center.tolist(),
                    size=box.wlh.tolist(),
                    rotation=box.orientation.elements.tolist(),
                    velocity=box.velocity[:2].tolist(),
                    detection_name=name,
                    detection_score=box.score,
                    attribute_name=attr,
                    tracking_name=name,
                    tracking_score=box.score,
                    tracking_id=box.token,
                    predict_traj=traj_ego,
                    predict_traj_score=traj_scores,
                )
                annos.append(nusc_anno)
            nusc_annos[sample_token] = annos
        nusc_submissions = {
            "meta": self.modality,
            "results": nusc_annos,
            "map_results": nusc_map_annos,
        }

        mmcv.mkdir_or_exist(jsonfile_prefix)
        res_path = os.path.join(jsonfile_prefix, "results_nusc.json")
        print("Results writes to", res_path)
        mmcv.dump(nusc_submissions, res_path)
        return res_path

    def format_results(self, results, jsonfile_prefix=None):
        """Format the results to json (standard format for COCO evaluation).

        Args:
            results (list[dict]): Testing results of the dataset.
            jsonfile_prefix (str | None): The prefix of json files. It includes
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.

        Returns:
            tuple: Returns (result_files, tmp_dir), where `result_files` is a \
                dict containing the json filepaths, `tmp_dir` is the temporal \
                directory created for saving json files when \
                `jsonfile_prefix` is not specified.
        """
        assert isinstance(results, list), "results must be a list"
        assert len(results) == len(
            self
        ), "The length of results is not equal to the dataset len: {} != {}".format(
            len(results), len(self)
        )

        if jsonfile_prefix is None:
            tmp_dir = tempfile.TemporaryDirectory()
            jsonfile_prefix = osp.join(tmp_dir.name, "results")
        else:
            tmp_dir = None

        result_files = self._format_bbox(results, jsonfile_prefix)

        return result_files, tmp_dir

    def _format_bbox_det(self, results, jsonfile_prefix=None):
        """Convert the results to the standard format.
        Args:
            results (list[dict]): Testing results of the dataset.
            jsonfile_prefix (str): The prefix of the output jsonfile.
                You can specify the output directory/filename by
                modifying the jsonfile_prefix. Default: None.
        Returns:
            str: Path of the output json file.
        """
        nusc_annos = {}
        mapped_class_names = self.CLASSES

        print("_format_bbox_det, Start to convert detection format...")
        for sample_id, det in enumerate(mmcv.track_iter_progress(results)):
            annos = []
            sample_token = self.data_infos[sample_id]["token"]

            if det is None:
                nusc_annos[sample_token] = annos
                continue

            boxes = output_to_nusc_box_det(det)
            boxes_ego = copy.deepcopy(boxes)
            boxes, keep_idx = lidar_nusc_box_to_global(
                self.data_infos[sample_id],
                boxes,
                mapped_class_names,
                self.eval_detection_configs,
                self.eval_version,
            )
            for i, box in enumerate(boxes):
                name = mapped_class_names[box.label]
                if np.sqrt(box.velocity[0] ** 2 + box.velocity[1] ** 2) > 0.2:
                    if name in [
                        "car",
                        "construction_vehicle",
                        "bus",
                        "truck",
                        "trailer",
                        "Car",
                    ]:
                        attr = "vehicle.moving"
                    elif name in ["bicycle", "motorcycle", "Motorcycle", "Cyclist"]:
                        attr = "cycle.with_rider"
                    else:
                        attr = self.default_attr[name]
                else:
                    if name in ["pedestrian", "Pedestrian"]:
                        attr = "pedestrian.standing"
                    elif name in ["bus"]:
                        attr = "vehicle.stopped"
                    else:
                        attr = self.default_attr[name]

                nusc_anno = dict(
                    sample_token=sample_token,
                    translation=box.center.tolist(),
                    size=box.wlh.tolist(),
                    rotation=box.orientation.elements.tolist(),
                    velocity=box.velocity[:2].tolist(),
                    detection_name=name,
                    detection_score=box.score,
                    attribute_name=attr,
                )
                annos.append(nusc_anno)
            nusc_annos[sample_token] = annos
        nusc_submissions = {
            "meta": self.modality,
            "results": nusc_annos,
        }

        mmcv.mkdir_or_exist(jsonfile_prefix)
        res_path = osp.join(jsonfile_prefix, "results_nusc_det.json")
        print("Results writes to", res_path)
        mmcv.dump(nusc_submissions, res_path)
        return res_path

    def format_results_det(self, results, jsonfile_prefix=None):
        """Format the results to json (standard format for COCO evaluation).

        Args:
            results (list[dict]): Testing results of the dataset.
            jsonfile_prefix (str | None): The prefix of json files. It includes
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.

        Returns:
            tuple: Returns (result_files, tmp_dir), where `result_files` is a \
                dict containing the json filepaths, `tmp_dir` is the temporal \
                directory created for saving json files when \
                `jsonfile_prefix` is not specified.
        """
        assert isinstance(results, list), "results must be a list"
        assert len(results) == len(
            self
        ), "The length of results is not equal to the dataset len: {} != {}".format(
            len(results), len(self)
        )

        if jsonfile_prefix is None:
            tmp_dir = tempfile.TemporaryDirectory()
            jsonfile_prefix = osp.join(tmp_dir.name, "results_det")
        else:
            tmp_dir = None

        result_files = self._format_bbox_det(results, jsonfile_prefix)
        return result_files, tmp_dir

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
        """Evaluation in nuScenes protocol.
        Args:
            results (list[dict]): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated.
            logger (logging.Logger | str | None): Logger used for printing
                related information during evaluation. Default: None.
            jsonfile_prefix (str | None): The prefix of json files. It includes
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.
            show (bool): Whether to visualize.
                Default: False.
            out_dir (str): Path to save the visualization results.
                Default: None.
            pipeline (list[dict], optional): raw data loading for showing.
                Default: None.
        Returns:
            dict[str, float]: Results of each evaluation metric.
            bbox_results
            occ_results_computed
            planning_results_computed
        """
        if isinstance(results, dict):
            if "occ_results_computed" in results.keys():
                occ_results_computed = results["occ_results_computed"]
                out_metrics = ["iou"]

                # pan_eval
                if occ_results_computed.get("pq", None) is not None:
                    out_metrics = ["iou", "pq", "sq", "rq"]

                print("Occ-flow Val Results:")
                for panoptic_key in out_metrics:
                    print(panoptic_key)
                    # HERE!! connect
                    print(
                        " & ".join(
                            [f"{x:.1f}" for x in occ_results_computed[panoptic_key]]
                        )
                    )

                if (
                    "num_occ" in occ_results_computed.keys()
                    and "ratio_occ" in occ_results_computed.keys()
                ):
                    print(f"num occ evaluated:{occ_results_computed['num_occ']}")
                    print(
                        f"ratio occ evaluated: {occ_results_computed['ratio_occ'] * 100:.1f}%"
                    )
            if "planning_results_computed" in results.keys():
                planning_results_computed = results["planning_results_computed"]
                planning_tab = PrettyTable()

                # check the length of the first key
                if len(list(planning_results_computed.values())[0]) == 10:
                    planning_tab.field_names = [
                        "metrics",
                        "0.5s",
                        "1.0s",
                        "1.5s",
                        "2.0s",
                        "2.5s",
                        "3.0s",
                        "3.5s",
                        "4.0s",
                        "Ave_1-4s",
                        "Ave_all",
                    ]
                else:
                    planning_tab.field_names = [
                        "metrics",
                        "0.5s",
                        "1.0s",
                        "1.5s",
                        "2.0s",
                        "2.5s",
                        "3.0s",
                        "Ave_123s",
                        "Ave_all",
                    ]

                for key in planning_results_computed.keys():
                    value = planning_results_computed[key]
                    row_value = []
                    row_value.append(key)
                    for i in range(len(value)):
                        row_value.append("%.4f" % float(value[i]))
                    planning_tab.add_row(row_value)
                print(planning_tab)

            # planning subset evaluation by auto mode
            if "planning_results_subset_auto_computed" in results.keys():
                planning_results_subset_auto_computed = results[
                    "planning_results_subset_auto_computed"
                ]
                planning_tab = PrettyTable()
                if len(list(planning_results_subset_auto_computed.values())[0]) == 10:
                    planning_tab.field_names = [
                        "metrics",
                        "0.5s",
                        "1.0s",
                        "1.5s",
                        "2.0s",
                        "2.5s",
                        "3.0s",
                        "3.5s",
                        "4.0s",
                        "Ave_1-4s",
                        "Ave_all",
                    ]
                else:
                    planning_tab.field_names = [
                        "metrics",
                        "0.5s",
                        "1.0s",
                        "1.5s",
                        "2.0s",
                        "2.5s",
                        "3.0s",
                        "Ave_123s",
                        "Ave_all",
                    ]
                for key in planning_results_subset_auto_computed.keys():
                    value = planning_results_subset_auto_computed[key]
                    row_value = []
                    row_value.append(key)
                    for i in range(len(value)):
                        row_value.append("%.4f" % float(value[i]))
                    planning_tab.add_row(row_value)
                print("\nsubset evaluation -- auto mode")
                print(planning_tab)

            # evaluate planning results with nuscenes API for lane-level eval
            results = results["bbox_results"]  # get bbox_results
            if "planning_map" in self.eval_mod:
                results_planning = compute_metrics(results, self.planning_map_config)
                print("\n\nmapping-level planning evaluation")
                pprint.pprint(results_planning)
                print("\n\n")

        # if we ever need to evaluate detection, tracking, mapping or motion
        if (
            "map" in self.eval_mod
            or "det" in self.eval_mod
            or "track" in self.eval_mod
            or "motion" in self.eval_mod
        ):
            result_files, tmp_dir = self.format_results(results, jsonfile_prefix)
            result_files_det, tmp_dir = self.format_results_det(
                results, jsonfile_prefix
            )

            if isinstance(result_files, dict):
                results_dict = dict()
                for name in result_names:
                    print("Evaluating bboxes of {}".format(name))
                    ret_dict = self._evaluate_single(
                        result_files[name], result_files_det[name]
                    )
                results_dict.update(ret_dict)
            elif isinstance(result_files, str):
                results_dict = self._evaluate_single(result_files, result_files_det)
        else:
            tmp_dir = None
            results_dict = dict()

        if "map" in self.eval_mod:
            drivable_intersection = 0
            drivable_union = 0
            lanes_intersection = 0
            lanes_union = 0
            divider_intersection = 0
            divider_union = 0
            crossing_intersection = 0
            crossing_union = 0
            contour_intersection = 0
            contour_union = 0
            for i in range(len(results)):
                drivable_intersection += results[i]["ret_iou"]["drivable_intersection"]
                drivable_union += results[i]["ret_iou"]["drivable_union"]
                lanes_intersection += results[i]["ret_iou"]["lanes_intersection"]
                lanes_union += results[i]["ret_iou"]["lanes_union"]
                divider_intersection += results[i]["ret_iou"]["divider_intersection"]
                divider_union += results[i]["ret_iou"]["divider_union"]
                crossing_intersection += results[i]["ret_iou"]["crossing_intersection"]
                crossing_union += results[i]["ret_iou"]["crossing_union"]
                contour_intersection += results[i]["ret_iou"]["contour_intersection"]
                contour_union += results[i]["ret_iou"]["contour_union"]
            results_dict.update(
                {
                    "drivable_iou": float(drivable_intersection / drivable_union),
                    "lanes_iou": float(lanes_intersection / lanes_union),
                    "divider_iou": float(divider_intersection / divider_union),
                    "crossing_iou": float(crossing_intersection / crossing_union),
                    "contour_iou": float(contour_intersection / contour_union),
                }
            )

            pprint.pprint(results_dict)

        if tmp_dir is not None:
            tmp_dir.cleanup()

        if show:
            self.show(results, out_dir, pipeline=pipeline)
        return results_dict

    def _evaluate_single(
        self,
        result_path,
        result_path_det,
        logger=None,
        metric="bbox",
        result_name="pts_bbox",
    ):
        """Evaluation for a single model in nuScenes protocol.

        Args:
            result_path (str): Path of the result file.
            logger (logging.Logger | str | None): Logger used for printing
                related information during evaluation. Default: None.
            metric (str): Metric name used for evaluation. Default: 'bbox'.
            result_name (str): Result name in the metric prefix.
                Default: 'pts_bbox'.

        Returns:
            dict: Dictionary of evaluation details.
        """

        # TODO: fix the evaluation pipelines

        output_dir = osp.join(*osp.split(result_path)[:-1])
        output_dir_det = osp.join(output_dir, "det")
        output_dir_track = osp.join(output_dir, "track")
        output_dir_motion = osp.join(output_dir, "motion")
        mmcv.mkdir_or_exist(output_dir_det)
        mmcv.mkdir_or_exist(output_dir_track)
        mmcv.mkdir_or_exist(output_dir_motion)
        print(output_dir)
        print(output_dir_det)
        print(output_dir_track)
        print(output_dir_motion)

        eval_set_map = {
            "v1.0-mini": "mini_val",
            "v1.0-trainval": "val",
        }
        detail = dict()

        if "det" in self.eval_mod:
            self.nusc_eval = NuScenesEval_custom(
                self.nusc,
                config=self.eval_detection_configs,
                result_path=result_path_det,
                eval_set=eval_set_map[self.version],
                output_dir=output_dir_det,
                verbose=True,
                overlap_test=self.overlap_test,
                data_infos=self.data_infos,
            )
            self.nusc_eval.main(plot_examples=0, render_curves=False)
            # record metrics
            metrics = mmcv.load(osp.join(output_dir_det, "metrics_summary.json"))
            metric_prefix = f"{result_name}_NuScenes"
            for name in self.CLASSES:
                for k, v in metrics["label_aps"][name].items():
                    val = float("{:.4f}".format(v))
                    detail["{}/{}_AP_dist_{}".format(metric_prefix, name, k)] = val
                for k, v in metrics["label_tp_errors"][name].items():
                    val = float("{:.4f}".format(v))
                    detail["{}/{}_{}".format(metric_prefix, name, k)] = val
                for k, v in metrics["tp_errors"].items():
                    val = float("{:.4f}".format(v))
                    detail["{}/{}".format(metric_prefix, self.ErrNameMapping[k])] = val
            detail["{}/NDS".format(metric_prefix)] = metrics["nd_score"]
            detail["{}/mAP".format(metric_prefix)] = metrics["mean_ap"]

        if "track" in self.eval_mod:
            cfg = config_factory("tracking_nips_2019")
            self.nusc_eval_track = TrackingEval(
                config=cfg,
                result_path=result_path,
                eval_set=eval_set_map[self.version],
                output_dir=output_dir_track,
                verbose=True,
                nusc_version=self.version,
                nusc_dataroot=self.data_root,
            )
            self.nusc_eval_track.main()
            # record metrics
            metrics = mmcv.load(osp.join(output_dir_track, "metrics_summary.json"))
            keys = [
                "amota",
                "amotp",
                "recall",
                "motar",
                "gt",
                "mota",
                "motp",
                "mt",
                "ml",
                "faf",
                "tp",
                "fp",
                "fn",
                "ids",
                "frag",
                "tid",
                "lgd",
            ]
            for key in keys:
                detail["{}/{}".format(metric_prefix, key)] = metrics[key]

        # if 'map' in self.eval_mod:
        #     for i, ret_iou in enumerate(ret_ious):
        #         detail['iou_{}'.format(i)] = ret_iou

        if "motion" in self.eval_mod:
            self.nusc_eval_motion = MotionEval(
                self.nusc,
                config=self.eval_detection_configs,
                result_path=result_path,
                eval_set=eval_set_map[self.version],
                output_dir=output_dir,
                verbose=True,
                overlap_test=self.overlap_test,
                data_infos=self.data_infos,
                category_convert_type="motion_category",
            )
            print("-" * 50)
            print(
                "Evaluate on motion category, merge class for vehicles and pedestrians..."
            )
            print("evaluate standard motion metrics...")
            self.nusc_eval_motion.main(
                plot_examples=0, render_curves=False, eval_mode="standard"
            )
            print("evaluate motion mAP-minFDE metrics...")
            self.nusc_eval_motion.main(
                plot_examples=0, render_curves=False, eval_mode="motion_map"
            )
            print("evaluate EPA motion metrics...")
            self.nusc_eval_motion.main(
                plot_examples=0, render_curves=False, eval_mode="epa"
            )
            print("-" * 50)
            print("Evaluate on detection category...")
            self.nusc_eval_motion = MotionEval(
                self.nusc,
                config=self.eval_detection_configs,
                result_path=result_path,
                eval_set=eval_set_map[self.version],
                output_dir=output_dir,
                verbose=True,
                overlap_test=self.overlap_test,
                data_infos=self.data_infos,
                category_convert_type="detection_category",
            )
            print("evaluate standard motion metrics...")
            self.nusc_eval_motion.main(
                plot_examples=0, render_curves=False, eval_mode="standard"
            )
            print("evaluate EPA motion metrics...")
            self.nusc_eval_motion.main(
                plot_examples=0, render_curves=False, eval_mode="motion_map"
            )
            print("evaluate EPA motion metrics...")
            self.nusc_eval_motion.main(
                plot_examples=0, render_curves=False, eval_mode="epa"
            )

        return detail

    def vis_dataset_specific(self, data: Dict):
        parent_dir = "./"
        pts_filename: str = os.path.join(parent_dir, data["pts_filename"])
        data_root = os.path.join(WORLDENGINE_ROOT, "data/raw/nuscenes/")

        if data_root not in pts_filename:
            pts_filename = pts_filename.split("nuscenes/")[-1]
            pts_filename = os.path.join(data_root, pts_filename)
            # print(pts_filename)

        points = np.fromfile(pts_filename, dtype=np.float32)
        points = points.reshape(-1, 5)  # N x 5

        image_file_list = []
        for cam_name in range(6):
            # FRONT, FRONT_RIGHT, FRONT_LEFT, BACK, BACK_LEFT, BACK_RIGHT
            image_file: str = os.path.join(parent_dir, data["img_filename"][cam_name])
            image_file_list.append(image_file)

        return {
            "lidar": points,
            "command_dict": {
                0: "TURN RIGHT",
                1: "TURN LEFT",
                2: "KEEP FORWARD",
                3: "TURN RIGHT AT THE NEXT INTERSECTION",
                4: "TURN LEFT AT THE NEXT INTERSECTION",
                5: "PREPARE TO STOP ON THE LEFT",
                6: "ENTER AND DRIVE IN THE ROUNDABOUT",
                7: "EXIT THE ROUNDABOUT",
                8: "UTURN",
            },
            "image_file_list": image_file_list,
            "left_hand": False,
        }
