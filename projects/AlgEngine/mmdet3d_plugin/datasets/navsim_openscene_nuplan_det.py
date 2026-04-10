import os
import copy
from typing import List
import numpy as np
import torch
import yaml
from pyquaternion import Quaternion

import mmcv
from mmcv.parallel import DataContainer as DC
from mmdet.datasets import DATASETS
from mmdet.datasets.pipelines import to_tensor
from mmdet3d.core.bbox import LiDARInstance3DBoxes

from nuscenes.eval.common.data_classes import EvalBoxes
from nuscenes.utils.data_classes import Box as NuScenesBox

from mmdet3d_plugin.datasets.data_utils.data_utils import output_to_nusc_box_det, lidar_nusc_box_to_global
from mmdet3d_plugin.datasets.eval_utils.nuplan_eval import CustomDetectionBox, NuScenesEval_NuPlan
from mmdet3d_plugin.datasets.navsim_openscene_nuplan import NavSimOpenSceneE2E
from mmdet3d_plugin.eval.detection.config import config_factory_nuPlan

from mmdet3d_plugin.utils import get_logger
logger = get_logger(__name__)

@DATASETS.register_module()
class NavSimOpenSceneE2EDet(NavSimOpenSceneE2E):
    r"""OpenScene detection Dataset"""

    def __init__(
        self, 
        *args,
        process_perception=True,
        eval_mod=None,
        **kwargs):

        self.eval_mod = eval_mod
        super(NavSimOpenSceneE2EDet, self).__init__(*args, process_perception=process_perception, **kwargs)

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
                # including history frames for test_mode
                if self.test_mode:
                    for j in range(self.history_frame_num):
                        self.index_map.append(i - j - 1)

        self.index_map = sorted(list(set(self.index_map)))
        self.index_map = [idx for idx in self.index_map if idx >= 0]
        logger.info(f'filtering {len(data_infos)} frames to {len(self.index_map)}...')

        # use log_token in OpenScene as scene_token.
        for info in data_infos:
            info['scene_name'] = info['log_name']
            info['scene_token'] = info['log_token']

        return data_infos

    def init_dataset(self):
        # update detection config to handle evaluation with different classes
        self.eval_detection_configs = config_factory_nuPlan()
        self.eval_version = "detection_cvpr_2019"
        self.default_attr = {
            "vehicle": "vehicle.parked",
            "bicycle": "cycle.without_rider",
            "pedestrian": "pedestrian.moving",
            "traffic_cone": "",
            "barrier": "",
            "czone_sign": "",
            "generic_object": "",
        }

        return

    def load_pdm_infos(self):
        return

    def prepare_test_data(self, index):
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

        final_index = index
        first_index = index - self.queue_length + 1
        first_index = max(first_index, 0)
        prev_indexs_list = list(reversed(range(first_index, final_index)))

        # insufficient history frames
        if first_index < 0:
            return None
        if self.data_infos[first_index]["scene_token"] != self.data_infos[final_index]["scene_token"]:
            return None

        input_dict = self.get_data_info(final_index)
        # empty detection gt
        if self.filter_empty_gt and (input_dict["ann_info"]["gt_labels_3d"] >= 0).sum() == 0:
            return None

        timestamp = input_dict["timestamp"]
        scene_token = input_dict["scene_token"]
        self.pre_pipeline(input_dict)
        example = self.pipeline(input_dict)
        # empty detection gt after filtering
        gt_labels = example['gt_labels_3d'].data
        if self.filter_empty_gt and gt_labels.numel() == 0:
            return None
        data_queue.insert(0, example)

        ########## retrieve previous infos, frame by frame
        for i in prev_indexs_list:
            input_dict = self.get_data_info(i, prev_frame=True)
            if (input_dict["timestamp"] < timestamp and input_dict["scene_token"] == scene_token):
                self.pre_pipeline(input_dict)
                example = self.pipeline(input_dict)
                timestamp = input_dict["timestamp"]
                gt_labels = example['gt_labels_3d'].data
                if self.filter_empty_gt and gt_labels.numel() == 0:
                    return None
                data_queue.insert(0, copy.deepcopy(example))
            else:
                break

        # insufficient history frames
        if len(data_queue) < self.queue_length:
            return None

        # merge a sequence of data into one dictionary, only for temporal data
        data_queue = self.union2one(data_queue)
        return data_queue

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

        # BUG: do not convert to fp32!
        timestamp_list = [torch.tensor([each["timestamp"]], dtype=torch.float64) for each in queue]  # L, 1

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
        gt_past_traj_list = [
            to_tensor(each["gt_past_traj"]) for each in queue
        ]  # L, N x 8 x 2
        gt_past_traj_mask_list = [
            to_tensor(each["gt_past_traj_mask"]) for each in queue
        ]  # L, N x 8 x 2
        gt_fut_traj = to_tensor(queue[-1]["gt_fut_traj"])  # L, N x 12 x 2
        gt_fut_traj_mask = to_tensor(queue[-1]["gt_fut_traj_mask"])  # L, N x 12 x 2

        # planning
        gt_sdc_bbox_list: List[LiDARInstance3DBoxes] = [
            each["gt_sdc_bbox"].data for each in queue
        ]  # L, 1 x 9

        # converting the global absolute coordinate into relative position and orientation change
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
        queue["gt_past_traj"] = DC(gt_past_traj_list)
        queue["gt_past_traj_mask"] = DC(gt_past_traj_mask_list)
        return queue

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

        # generate annotation for detection/tracking, putting them in the annotation so that
        # we could do range filtering altogether in the transform_3d.py
        input_dict["ann_info"] = self.get_ann_info(index)
        input_dict = self.update_mapping(input_dict=input_dict, index=index)
        input_dict = self.update_ego_prediction(input_dict=input_dict, index=index)
        input_dict = self.update_ego_planning(input_dict=input_dict, index=index)

        return input_dict

    def format_bbox_results(self, results):
        assert isinstance(results, list), 'results must be a list'
        assert len(results) == len(self), (
            'The length of results is not equal to the dataset len: {} != {}'.
            format(len(results), len(self)))

        if 'pts_bbox' in results[0]:
            results_ = [out['pts_bbox'] for out in results]
        else:
            results_ = results

        result_boxes = self._format_bbox(results_)
        return result_boxes

    def _format_bbox(self, results):
        all_preds = EvalBoxes()
        mapped_class_names = (
            "vehicle",
            "bicycle",
            "pedestrian",
            "traffic_cone",
            "barrier",
            "czone_sign",
            "generic_object",
        )
        data_infos = [info for info in self.data_infos if info['token'] in self.scene_filter]
        token_to_info = {info['token']: info for info in data_infos}

        logger.info('Start to convert detection format...')
        for det in mmcv.track_iter_progress(results):
            sample_preds = []
            boxes = output_to_nusc_box_det(det)
            sample_token = det['token']
            if sample_token not in token_to_info:
                continue
            frame_info = token_to_info[sample_token]
            boxes, keep_idx = lidar_nusc_box_to_global(frame_info, boxes,
                                             mapped_class_names,
                                             self.eval_detection_configs,
                                             self.eval_version)
            for i, box in enumerate(boxes):
                name = mapped_class_names[box.label]
                if name not in self.CLASSES:
                    continue

                if np.sqrt(box.velocity[0]**2 + box.velocity[1]**2) > 0.2:
                    if name == 'vehicle':
                        attr = 'vehicle.moving'
                    elif name == 'bicycle':
                        attr = 'cycle.with_rider'
                    else:
                        attr = self.default_attr[name]
                else:
                    if name in ['pedestrian']:
                        attr = 'pedestrian.standing'
                    else:
                        attr = self.default_attr[name]

                ego_translation = box.center - frame_info['ego2global_translation']

                box = CustomDetectionBox(
                    sample_token=sample_token,
                    translation=box.center.tolist(),
                    size=box.wlh.tolist(),
                    rotation=box.orientation.elements.tolist(),
                    velocity=box.velocity[:2].tolist(),
                    ego_translation=ego_translation.tolist(),
                    detection_name=name,
                    detection_score=box.score,  # GT samples do not have a score.
                    attribute_name=attr)
                sample_preds.append(box)

            all_preds.add_boxes(sample_token, sample_preds)
        
        pred_tokens = set(all_preds.sample_tokens)
        gt_tokens = set(token_to_info.keys())
        diff_tokens = gt_tokens - pred_tokens
        if len(diff_tokens) != 0:
            logger.warning(f"{len(diff_tokens)} tokens are not predicted.")
            for token in diff_tokens:
                all_preds.add_boxes(token, [])

        return all_preds

    def format_gt_bbox(self):

        all_annotations = EvalBoxes()
        data_infos = [info for info in self.data_infos if info['token'] in self.scene_filter]

        for sample_idx, info in enumerate(data_infos):
            sample_token = info['token']

            if self.use_valid_flag:
                mask = info["valid_flag"]
            else:
                mask = info["num_lidar_pts"] > 0

            gt_bboxes_3d = info['gt_boxes'][mask]
            gt_names_3d = info['gt_names'][mask]
            gt_velocity = info['gt_velocity'][mask]

            sample_boxes = []
            for box_idx in range(len(gt_bboxes_3d)):
                if gt_names_3d[box_idx] not in self.CLASSES:
                    continue
                quat = Quaternion(axis=[0, 0, 1], radians=gt_bboxes_3d[box_idx][6])
                box = NuScenesBox(
                    center=gt_bboxes_3d[box_idx][:3],
                    size=gt_bboxes_3d[box_idx][[4, 3, 5]],
                    orientation=quat,
                    velocity=(*gt_velocity[box_idx], 0),
                    name=gt_names_3d[box_idx]
                )
                box.rotate(Quaternion(info['lidar2ego_rotation']))
                box.translate(np.array(info['lidar2ego_translation']))

                det_range = self.eval_detection_configs.class_range[box.name]
                if np.linalg.norm(box.center[:2], 2) > det_range:
                    continue

                box.rotate(Quaternion(info['ego2global_rotation']))
                ego_translation = box.center.tolist()
                box.translate(np.array(info['ego2global_translation']))

                if np.sqrt(box.velocity[0]**2 + box.velocity[1]**2) > 0.2:
                    if box.name == 'vehicle':
                        attr = 'vehicle.moving'
                    elif box.name == 'bicycle':
                        attr = 'cycle.with_rider'
                    else:
                        attr = self.default_attr[box.name]
                else:
                    if box.name in ['pedestrian']:
                        attr = 'pedestrian.standing'
                    else:
                        attr = self.default_attr[box.name]

                box = CustomDetectionBox(
                    sample_token=sample_token,
                    translation=box.center.tolist(),
                    size=box.wlh.tolist(),
                    rotation=box.orientation.elements.tolist(),
                    velocity=box.velocity[:2].tolist(),
                    ego_translation=ego_translation,
                    num_pts=1,
                    detection_name=box.name,
                    detection_score=-1.0,  # GT samples do not have a score.
                    attribute_name=attr
                )
                sample_boxes.append(box)
            all_annotations.add_boxes(sample_token, sample_boxes)

        return all_annotations

    def _evaluate_single_bbox(self,
                              gt_boxes,
                              result_boxes,
                              jsonfile_prefix,
                              logger=None):

        output_dir = jsonfile_prefix

        self.nusc_eval = NuScenesEval_NuPlan(
            gt_boxes,
            result_boxes,
            config=self.eval_detection_configs,
            output_dir=output_dir,
            verbose=True
        )
        self.nusc_eval.main(plot_examples=0, render_curves=False)
        # record metrics
        metrics = mmcv.load(os.path.join(output_dir, 'metrics_summary.json'))
        detail = dict()
        metric_prefix = 'NuPlan'
        for name in self.CLASSES:
            for k, v in metrics['label_aps'][name].items():
                val = float('{:.4f}'.format(v))
                detail['{}/{}_AP_dist_{}'.format(metric_prefix, name, k)] = val
            for k, v in metrics['label_tp_errors'][name].items():
                val = float('{:.4f}'.format(v))
                detail['{}/{}_{}'.format(metric_prefix, name, k)] = val
            for k, v in metrics['tp_errors'].items():
                val = float('{:.4f}'.format(v))
                detail['{}/{}'.format(metric_prefix,
                                      self.ErrNameMapping[k])] = val
        detail['{}/NDS'.format(metric_prefix)] = metrics['nd_score']
        detail['{}/mAP'.format(metric_prefix)] = metrics['mean_ap']
        return detail

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
            bbox_results, map_results
        """
        # if we ever need to evaluate detection, tracking, mapping or motion
        if "det" in self.eval_mod:
            gt_boxes = self.format_gt_bbox()
            result_boxes = self.format_bbox_results(results)
            tmp_dir = None
            results_dict = self._evaluate_single_bbox(gt_boxes, result_boxes, jsonfile_prefix, logger=logger)
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

        if tmp_dir is not None:
            tmp_dir.cleanup()

        return results_dict
