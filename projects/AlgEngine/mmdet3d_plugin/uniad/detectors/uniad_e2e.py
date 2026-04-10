# ---------------------------------------------------------------------------------#
# UniAD: Planning-oriented Autonomous Driving (https://arxiv.org/abs/2212.10156)  #
# Source code: https://github.com/OpenDriveLab/UniAD                              #
# Copyright (c) OpenDriveLab. All rights reserved.                                #
# ---------------------------------------------------------------------------------#

import torch
import torch.nn as nn
from mmcv.runner import auto_fp16
from mmdet.models import DETECTORS
import copy
import os
import numpy as np
from ..dense_heads.seg_head_plugin import IOU
from .uniad_track import UniADTrack
from mmdet.models.builder import build_head
from mmdet3d_plugin.models.utils.functional import pos2posemb2d
from mmdet3d.core.bbox.structures.lidar_box3d import LiDARInstance3DBoxes


@DETECTORS.register_module()
class UniAD(UniADTrack):
    """
    UniAD: Unifying Detection, Tracking, Segmentation, Motion Forecasting, Occupancy Prediction and Planning for Autonomous Driving
    """

    def __init__(
        self,
        seg_head=None,
        motion_head=None,
        occ_head=None,
        planning_head=None,
        task_loss_weight=dict(track=1.0, map=1.0, motion=1.0, occ=1.0, planning=1.0),
        **kwargs,
    ):
        super(UniAD, self).__init__(**kwargs)
        if seg_head:
            self.seg_head = build_head(seg_head)
        if occ_head:
            self.occ_head = build_head(occ_head)
        if motion_head:
            self.motion_head = build_head(motion_head)
        else:
            # used to get track query positional embedding in the case of no motion module
            self.boxes_query_embedding_layer = nn.Sequential(
                nn.Linear(self.embed_dims, self.embed_dims * 2),
                nn.ReLU(),
                nn.Linear(self.embed_dims * 2, self.embed_dims),
            )
        if planning_head:
            self.planning_head = build_head(planning_head)

        self.task_loss_weight = task_loss_weight
        assert set(task_loss_weight.keys()) == {
            "track",
            "occ",
            "motion",
            "map",
            "planning",
        }

        # self.frame_idx = 0
        # self.scene_token = None

    @property
    def with_planning_head(self):
        return hasattr(self, "planning_head") and self.planning_head is not None

    @property
    def with_occ_head(self):
        return hasattr(self, "occ_head") and self.occ_head is not None

    @property
    def with_motion_head(self):
        return hasattr(self, "motion_head") and self.motion_head is not None

    @property
    def with_seg_head(self):
        return hasattr(self, "seg_head") and self.seg_head is not None

    def forward_dummy(self, img):
        dummy_metas = {
            'scene_token': 'dummy',
            'frame_idx': 0,
            'sample_idx': 0,
            'can_bus': np.array([0 for i in range(18)]),
            'lidar2img': [np.zeros((4, 4)) for i in range(6)],   # 6, 4x4
            'img_shape': [(928, 1600, 3) for i in range(6)],
            'box_type_3d': LiDARInstance3DBoxes,
            'pts_filename': 'dummy',
        }
        return self.forward_test(img=img, img_metas=[[dummy_metas]])

    def forward(self, return_loss=True, **kwargs):
        """Calls either forward_train or forward_test depending on whether
        return_loss=True.
        Note this setting will change the expected inputs. When
        `return_loss=True`, img and img_metas are single-nested (i.e.
        torch.Tensor and list[dict]), and when `resturn_loss=False`, img and
        img_metas should be double nested (i.e.  list[torch.Tensor],
        list[list[dict]]), with the outer list indicating test time
        augmentations.
        """
        if return_loss:
            return self.forward_train(**kwargs)
        else:
            return self.forward_test(**kwargs)
        
    def _extract_tracking_centers(self, bbox_results, bev_range):
        """
        extract the bboxes centers and normized according to the bev range

        Args:
            bbox_results (List[Tuple[torch.Tensor]]): A list of tuples containing the bounding box results for each image in the batch.
            bev_range (List[float]): A list of float values representing the bird's eye view range.

        Returns:
            torch.Tensor: A tensor representing normized centers of the detection bounding boxes.
        """
        batch_size = len(bbox_results)
        det_bbox_posembed = []
        for i in range(batch_size):
            bboxes, scores, labels, bbox_index, mask = bbox_results[i]
            xy = bboxes.gravity_center[:, :2]
            x_norm = (xy[:, 0] - bev_range[0]) / (bev_range[3] - bev_range[0])
            y_norm = (xy[:, 1] - bev_range[1]) / (bev_range[4] - bev_range[1])
            det_bbox_posembed.append(
                torch.cat([x_norm[:, None], y_norm[:, None]], dim=-1)
            )
        return torch.stack(det_bbox_posembed)

    def generate_out_motion(self, outs_track, gt_labels_3d=None, require_matching=True):
        # used in the case of not using motion head but to mimick the motion head outputs, 
        # but actually only from the track head to give it to occupancy

        outs_motion = dict()

        ####### get track query from tracking
        track_query = outs_track["track_query_embeddings"][
            None, None, ...
        ]  # num_dec, B, A, C          
        # track_query = torch.cat(
        #     [track_query, outs_track["sdc_embedding"][None, None, None, :]], dim=2
        # )  # num_dec, B, A+1, C          
        # # print(track_query.size())  
        track_query = track_query[:, -1]    # B, A, C   

        ####### get positional embedding from the tracked boxes
        device = track_query.device
        track_boxes = outs_track["track_bbox_results"]      # .tensor has A x 9

        reference_points_track = self._extract_tracking_centers(
            track_boxes, self.pc_range
        )              # 1 x A x 2 
        # print(reference_points_track.size())

        # TODO: shift this module to the tracking module
        track_query_pos = self.boxes_query_embedding_layer(
            pos2posemb2d(reference_points_track.to(device))
        )  # B, A, D
        
        outs_motion['track_query'] = track_query
        outs_motion['track_query_pos'] = track_query_pos

        ######## get matched indexes with GT

        # for training
        if require_matching:
            all_matched_idxes = [outs_track["track_query_matched_idxes"]]  # len(0), A
            class_labels = gt_labels_3d
        
        # for testing
        else:
            all_matched_idxes = None
            class_labels = track_boxes[0][2]
            scores = track_boxes[0][1]
            outs_motion["track_scores"] = scores[None, :]

        def filter_vehicle_query(
            outs_motion, all_matched_idxes, class_labels, vehicle_id_list
        ):
            
            # during training, get the label for matched object only
            if require_matching:
                query_label = class_labels[0][-1][all_matched_idxes[0]]
            # during testing, we get all tracked objects
            else:
                query_label = class_labels

            # select vehicle query according to vehicle_id_list
            vehicle_mask = torch.zeros_like(query_label)
            for veh_id in vehicle_id_list:
                vehicle_mask |= query_label == veh_id

            # no need
            # outs_motion["traj_query"] = outs_motion["traj_query"][
            #     :, :, vehicle_mask > 0
            # ]
            # print(vehicle_mask)

            outs_motion["track_query"] = outs_motion["track_query"][:, vehicle_mask > 0]
            outs_motion["track_query_pos"] = outs_motion["track_query_pos"][
                :, vehicle_mask > 0
            ]
            
            if 'track_scores' in outs_motion:
                outs_motion["track_scores"] = outs_motion["track_scores"][
                    :, vehicle_mask > 0
                ]

            # filter for training
            if require_matching:
                all_matched_idxes[0] = all_matched_idxes[0][vehicle_mask > 0]
            
            return outs_motion, all_matched_idxes
        
        # generate output dict
        outs_motion, all_matched_idxes = filter_vehicle_query(
            outs_motion, all_matched_idxes, class_labels, self.vehicle_id_list
        )

        if require_matching:
            outs_motion["all_matched_idxes"] = all_matched_idxes

        return outs_motion

    # Add the subtask loss to the whole model loss
    @auto_fp16(apply_to=("img", "points"))
    def forward_train(
        self,
        img=None,
        img_metas=None,
        gt_bboxes_3d=None,
        gt_labels_3d=None,
        gt_inds=None,
        l2g_t=None,
        l2g_r_mat=None,
        timestamp=None,
        gt_lane_labels=None,
        gt_lane_bboxes=None,
        gt_lane_masks=None,
        gt_fut_traj=None,
        gt_fut_traj_mask=None,
        gt_past_traj=None,
        gt_past_traj_mask=None,
        gt_sdc_bbox=None,
        gt_sdc_label=None,
        gt_sdc_fut_traj=None,
        gt_sdc_fut_traj_mask=None,
        # Occ_gt
        gt_segmentation=None,
        gt_instance=None,
        gt_occ_img_is_valid=None,
        # planning
        sdc_planning=None,
        sdc_planning_mask=None,
        command=None,
        sdc_planning_past=None,
        sdc_planning_mask_past=None,
        # fut gt for planning
        gt_future_boxes=None,
        **kwargs,  # [1, 9]
    ):
        """Forward training function for the model that includes multiple tasks, such as tracking, segmentation, motion prediction, occupancy prediction, and planning.

        Args:
        img (torch.Tensor, optional): Tensor containing images of each sample with shape (N, C, H, W). Defaults to None.
        img_metas (list[dict], optional): List of dictionaries containing meta information for each sample. Defaults to None.
        gt_bboxes_3d (list[:obj:BaseInstance3DBoxes], optional): List of ground truth 3D bounding boxes for each sample. Defaults to None.
        gt_labels_3d (list[torch.Tensor], optional): List of tensors containing ground truth labels for 3D bounding boxes. Defaults to None.
        gt_inds (list[torch.Tensor], optional): List of tensors containing indices of ground truth objects. Defaults to None.
        l2g_t (list[torch.Tensor], optional): List of tensors containing translation vectors from local to global coordinates. Defaults to None.
        l2g_r_mat (list[torch.Tensor], optional): List of tensors containing rotation matrices from local to global coordinates. Defaults to None.
        timestamp (list[float], optional): List of timestamps for each sample. Defaults to None.
        gt_bboxes_ignore (list[torch.Tensor], optional): List of tensors containing ground truth 2D bounding boxes in images to be ignored. Defaults to None.
        gt_lane_labels (list[torch.Tensor], optional): List of tensors containing ground truth lane labels. Defaults to None.
        gt_lane_bboxes (list[torch.Tensor], optional): List of tensors containing ground truth lane bounding boxes. Defaults to None.
        gt_lane_masks (list[torch.Tensor], optional): List of tensors containing ground truth lane masks. Defaults to None.
        gt_fut_traj (list[torch.Tensor], optional): List of tensors containing ground truth future trajectories. Defaults to None.
        gt_fut_traj_mask (list[torch.Tensor], optional): List of tensors containing ground truth future trajectory masks. Defaults to None.
        gt_past_traj (list[torch.Tensor], optional): List of tensors containing ground truth past trajectories. Defaults to None.
        gt_past_traj_mask (list[torch.Tensor], optional): List of tensors containing ground truth past trajectory masks. Defaults to None.
        gt_sdc_bbox (list[torch.Tensor], optional): List of tensors containing ground truth self-driving car bounding boxes. Defaults to None.
        gt_sdc_label (list[torch.Tensor], optional): List of tensors containing ground truth self-driving car labels. Defaults to None.
        gt_sdc_fut_traj (list[torch.Tensor], optional): List of tensors containing ground truth self-driving car future trajectories. Defaults to None.
        gt_sdc_fut_traj_mask (list[torch.Tensor], optional): List of tensors containing ground truth self-driving car future trajectory masks. Defaults to None.
        gt_segmentation (list[torch.Tensor], optional): List of tensors containing ground truth segmentation masks. Defaults to
        gt_instance (list[torch.Tensor], optional): List of tensors containing ground truth instance segmentation masks. Defaults to None.
        gt_occ_img_is_valid (list[torch.Tensor], optional): List of tensors containing binary flags indicating whether an image is valid for occupancy prediction. Defaults to None.
        sdc_planning (list[torch.Tensor], optional): List of tensors containing self-driving car planning information. Defaults to None.
        sdc_planning_mask (list[torch.Tensor], optional): List of tensors containing self-driving car planning masks. Defaults to None.
        command (list[torch.Tensor], optional): List of tensors containing high-level command information for planning. Defaults to None.
        gt_future_boxes (list[torch.Tensor], optional): List of tensors containing ground truth future bounding boxes for planning. Defaults to None.
        gt_future_labels (list[torch.Tensor], optional): List of tensors containing ground truth future labels for planning. Defaults to None.

        Returns:
            dict: Dictionary containing losses of different tasks, such as tracking, segmentation, motion prediction, occupancy prediction, and planning. Each key in the dictionary
                is prefixed with the corresponding task name, e.g., 'track', 'map', 'motion', 'occ', and 'planning'. The values are the calculated losses for each task.
        """


        losses = dict()
        len_queue = img.size(1)


        losses_track, outs_track = self.forward_track_train(
            img,
            gt_bboxes_3d,
            gt_labels_3d,
            gt_past_traj,
            gt_past_traj_mask,
            gt_inds,
            gt_sdc_bbox,
            gt_sdc_label,
            l2g_t,
            l2g_r_mat,
            img_metas,
            timestamp,
        )
        losses_track = self.loss_weighted_and_prefixed(losses_track, prefix="track")
        losses.update(losses_track)

        # Upsample bev for tiny version
        outs_track = self.upsample_bev_if_tiny(outs_track)
        bev_embed = outs_track["bev_embed"]
        bev_pos = outs_track["bev_pos"]
        img_metas = [each[len_queue - 1] for each in img_metas]

        # mapping Head
        outs_seg = dict()
        if self.with_seg_head:
            losses_seg, outs_seg = self.seg_head.forward_train(
                bev_embed, img_metas, gt_lane_labels, gt_lane_bboxes, gt_lane_masks
            )
            losses_seg = self.loss_weighted_and_prefixed(losses_seg, prefix="map")
            losses.update(losses_seg)

        # Motion prediction Head
        outs_motion = dict()
        if self.with_motion_head:
            ret_dict_motion = self.motion_head.forward_train(
                bev_embed,
                gt_bboxes_3d,
                gt_labels_3d,
                gt_fut_traj,
                gt_fut_traj_mask,
                gt_sdc_fut_traj,
                gt_sdc_fut_traj_mask,
                outs_track=outs_track,
                outs_seg=outs_seg,
            )
            losses_motion = ret_dict_motion["losses"]
            outs_motion = ret_dict_motion["outs_motion"]
            losses_motion = self.loss_weighted_and_prefixed(
                losses_motion, prefix="motion"
            )
            losses.update(losses_motion)
        
        # below code originally coming from the prediction module, copy it here when
        # when we do not include the prediction head
        else:
            outs_motion = self.generate_out_motion(outs_track, gt_labels_3d)
        outs_motion["bev_pos"] = bev_pos

        # Occupancy Head
        if self.with_occ_head:
            if outs_motion["track_query"].shape[1] == 0:        # 1 x 4 x 256
                # TODO: rm hard code
                outs_motion["track_query"] = torch.zeros((1, 1, 256)).to(bev_embed)
                outs_motion["track_query_pos"] = torch.zeros((1, 1, 256)).to(bev_embed)
                outs_motion["traj_query"] = torch.zeros((3, 1, 1, 6, 256)).to(bev_embed)
                outs_motion["all_matched_idxes"] = [[-1]]
            losses_occ, outs_occ = self.occ_head.forward_train(
                bev_embed,
                outs_motion,
                gt_inds_list=gt_inds,   # List of List, gt_inds[0][0].size() is 55
                gt_segmentation=gt_segmentation,    # 1 x 7 x H x W
                gt_instance=gt_instance,        # 1 x 7 x H x W
                gt_img_is_valid=gt_occ_img_is_valid,    # 1 x 9
                img_metas=img_metas,
            )
            losses_occ = self.loss_weighted_and_prefixed(losses_occ, prefix="occ")
            losses.update(losses_occ)
        else:
            outs_occ = dict()

        # Planning Head
        if self.with_planning_head:
            outs_planning = self.planning_head.forward_train(
                bev_embed=bev_embed,
                outs_motion=outs_motion,
                sdc_planning=sdc_planning,
                sdc_planning_mask=sdc_planning_mask,
                command=command,
                gt_future_boxes=gt_future_boxes,
                img_metas=img_metas,
                outs_seg=outs_seg,
                outs_occ=outs_occ,
                sdc_planning_past=sdc_planning_past,
                sdc_planning_mask_past=sdc_planning_mask_past,
            )
            losses_planning = outs_planning["losses"]
            losses_planning = self.loss_weighted_and_prefixed(
                losses_planning, prefix="planning"
            )
            losses.update(losses_planning)

        for k, v in losses.items():
            losses[k] = torch.nan_to_num(v[None])
        return losses

    def loss_weighted_and_prefixed(self, loss_dict, prefix=""):
        loss_factor = self.task_loss_weight[prefix]
        loss_dict = {f"{prefix}.{k}": v * loss_factor for k, v in loss_dict.items()}
        return loss_dict

    def forward_test(
        self,
        img=None,
        img_metas=None,
        l2g_t=None,
        l2g_r_mat=None,
        timestamp=None,
        gt_lane_labels=None,
        gt_lane_masks=None,
        rescale=False,
        # planning gt (for evaluation only)
        sdc_planning=None,
        sdc_planning_mask=None,
        command=None,
        sdc_planning_past=None,
        sdc_planning_mask_past=None,        
        # Occ_gt (for evaluation only)
        gt_segmentation=None,
        gt_instance=None,
        gt_occ_img_is_valid=None,
        # higher-resolution segmentations
        segmentations_finegrained=None,
        vehicles_finegrained=None,
        pedestrains_finegrained=None,
        **kwargs,
    ):
        """Test function"""
        for var, name in [(img_metas, "img_metas")]:
            if not isinstance(var, list):
                raise TypeError("{} must be a list, but got {}".format(name, type(var)))
        img = [img] if img is None else img

        # reset the frame id for this sequence
        # try:
        scene_token: str = img_metas[0][0]["scene_token"]
        frame_idx: int = img_metas[0][0]["frame_idx"]
        # except TypeError:
        #     scene_token = ''
        #     print('Invalid input')

        if (scene_token != self.prev_frame_info["scene_token"] or
            (timestamp[0].item() - self.prev_frame_info["timestamp"]) > 1.1):
            # the first sample of each scene is truncated
            self.prev_frame_info["prev_bev"] = None
        # update idx
        self.prev_frame_info["scene_token"] = scene_token
        self.prev_frame_info["timestamp"] = timestamp[0].item()

        # do not use temporal information
        if not self.video_test_mode:
            self.prev_frame_info["prev_bev"] = None

        # Get the delta of ego position and angle between two timestamps.
        tmp_pos = copy.deepcopy(img_metas[0][0]["can_bus"][:3])
        tmp_angle = copy.deepcopy(img_metas[0][0]["can_bus"][-1])

        # first frame
        if self.prev_frame_info["scene_token"] is None:
            img_metas[0][0]["can_bus"][:3] = 0
            img_metas[0][0]["can_bus"][-1] = 0
        # following frames
        else:
            try:
                img_metas[0][0]["can_bus"][:3] -= self.prev_frame_info["prev_pos"]
                img_metas[0][0]["can_bus"][-1] -= self.prev_frame_info["prev_angle"]
            except TypeError:
                print('dummy input')
        self.prev_frame_info["prev_pos"] = tmp_pos
        self.prev_frame_info["prev_angle"] = tmp_angle

        img = img[0]
        img_metas = img_metas[0]
        timestamp = timestamp[0] if timestamp is not None else None

        result = [dict() for i in range(len(img_metas))]
        result_track = self.simple_test_track(
            img, l2g_t, l2g_r_mat, img_metas, timestamp
        )

        # Upsample bev for tiny model
        result_track[0] = self.upsample_bev_if_tiny(result_track[0])

        bev_embed = result_track[0]["bev_embed"]  # HW x 1 x 256

        # segmentation head
        if self.with_seg_head:
            result_seg = self.seg_head.forward_test(
                bev_embed, gt_lane_labels, gt_lane_masks, img_metas, rescale
            )

            # copy the soft results over to save the results
            # also convert it to cpu to not explode the GPU memory and cause out-of-memory error
            result_seg[0]["soft"] = {
                "drivable": result_seg[0]["pts_bbox"]["score_list"][-1]
                .cpu()
                .detach()
                .numpy(),  # H x W
                "lanes": result_seg[0]["pts_bbox"]["lane_score"]
                .cpu()
                .detach()
                .numpy(),  # 3 x H x W
            }

        # handle bug in the motion head below without seg head
        else:
            result_seg = [None]

        # motion prediction head
        if self.with_motion_head:
            result_motion, outs_motion = self.motion_head.forward_test(
                bev_embed, 
                outs_track=result_track[0], 
                outs_seg=result_seg[0],
            )
            # outs_motion["bev_pos"] = result_track[0]["bev_pos"]

        # below code originally coming from the prediction module, copy it here when
        # when we do not include the prediction head
        else:
            if 'track_query_embeddings' in result_track:
                outs_motion = self.generate_out_motion(result_track[0], None, require_matching=False)
            else:
                outs_motion = {}
            # print(gt_labels_3d)
        outs_motion["bev_pos"] = result_track[0]["bev_pos"]

        # occupancy prediction head
        outs_occ = dict()
        if self.with_occ_head:

            # if no agent at all detected in this frame
            try:
                occ_no_query = outs_motion["track_query"].shape[1] == 0
            except KeyError:
                occ_no_query = True

            outs_occ = self.occ_head.forward_test(
                bev_embed,      # HW x 1 x 256
                outs_motion,
                no_query=occ_no_query,
                gt_segmentation=gt_segmentation,
                gt_instance=gt_instance,
                gt_img_is_valid=gt_occ_img_is_valid,
                img_metas=img_metas,                
            )
            result[0]["occ"] = outs_occ
            # seg_gt: 1 x 5 x 1 x H x W

        if self.with_planning_head:
            planning_gt = dict(
                segmentation=gt_segmentation,
                sdc_planning=sdc_planning,
                sdc_planning_mask=sdc_planning_mask,
                command=command,
                segmentations_finegrained=segmentations_finegrained,
                vehicles_finegrained=vehicles_finegrained,
                pedestrains_finegrained=pedestrains_finegrained,
            )
            result_planning = self.planning_head.forward_test(
                bev_embed=bev_embed, 
                outs_motion=outs_motion, 
                command=command, 
                img_metas=img_metas,
                outs_seg=result_seg[0],
                outs_occ=outs_occ,
                sdc_planning_past=sdc_planning_past,
                sdc_planning_mask_past=sdc_planning_mask_past,                
            )
            result[0]["planning"] = dict(
                planning_gt=planning_gt,
                result_planning=result_planning,
            )

        if self.with_seg_head:
            del result_seg[0]["args_tuple"]

        pop_track_list = [
            "prev_bev",
            "bev_pos",
            "bev_embed",
            "track_query_embeddings",
            "sdc_embedding",
        ]
        result_track[0] = pop_elem_in_result(result_track[0], pop_track_list)

        # not removing the pts_bbox from online mapping results for visualization
        if self.with_seg_head:
            result_seg[0] = pop_elem_in_result(result_seg[0], pop_list=["pts_bbox"])

        if self.with_motion_head:
            result_motion[0] = pop_elem_in_result(result_motion[0])
        if self.with_occ_head:
            result[0]["occ"] = pop_elem_in_result(
                result[0]["occ"],
                pop_list=[
                    "seg_out_mask",
                    "flow_out",
                    "future_states_occ",
                    "pred_ins_masks",
                    "pred_raw_occ",
                    "pred_ins_logits",
                    "pred_ins_sigmoid",
                ],
            )

        for i, res in enumerate(result):
            res["token"] = img_metas[i]["sample_idx"]
            res["scene_token"] = img_metas[i]["scene_token"]
            res.update(result_track[i])
            if self.with_motion_head:
                res.update(result_motion[i])
            if self.with_seg_head:
                res.update(result_seg[i])

        # result dictionary content
        # occ
            # seg_gt
        # planning
        # token
        # track_bbox_results
        # boxes_3d
        # scores_3d
        # labels_3d
        # track_scores
        # track_ids
        # sdc_boxes_3d
        # sdc_scores_3d
        # sdc_track_scores
        # sdc_track_bbox_results
        # boxes_3d_det
        # scores_3d_det
        # labels_3d_det
        # traj_0
        # traj_scores_0
        # traj_1
        # traj_scores_1
        # traj
        # traj_scores
        # ret_iou
        # soft

        return result


def pop_elem_in_result(task_result: dict, pop_list: list = None):
    all_keys = list(task_result.keys())
    for k in all_keys:
        if k.endswith("query") or k.endswith("query_pos") or k.endswith("embedding"):
            task_result.pop(k)

    if pop_list is not None:
        for pop_k in pop_list:
            task_result.pop(pop_k, None)
    return task_result
