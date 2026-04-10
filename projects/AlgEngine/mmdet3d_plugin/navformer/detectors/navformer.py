import random
import copy
import math
from typing import List
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.nn.utils.stateless import functional_call
from torch.cuda.amp import autocast
from einops import rearrange

from mmcv.runner import auto_fp16
from mmdet.models import DETECTORS, build_loss
from mmdet.models.builder import build_head
from mmdet.models.utils.transformer import inverse_sigmoid
from mmdet3d.core.bbox.coders import build_bbox_coder
from mmdet3d.models.detectors.mvx_two_stage import MVXTwoStageDetector
from mmdet3d_plugin.models.utils.grid_mask import GridMask
from mmdet3d_plugin.core.bbox.util import normalize_bbox

from mmdet3d_plugin.uniad.dense_heads.track_head_plugin import (
    MemoryBank,
    QueryInteractionModule,
    Instances,
    RuntimeTrackerBase,
)
from mmdet3d_plugin.utils import get_logger
logger = get_logger(__name__)

def print_cuda_memory(tag=""):
    print(f"[{tag}] Allocated: {torch.cuda.memory_allocated() / 1e6:.2f} MB, "
          f"Cached: {torch.cuda.memory_reserved() / 1e6:.2f} MB")

def trace_parameters(output, model):
    """Traces all parameters contributing to a model output."""
    visited = set()
    parameters_involved = set()

    def trace_fn(node):
        if node in visited:
            return
        visited.add(node)

        if hasattr(node, 'variable'):  # If it's a leaf node (parameter)
            parameters_involved.add(node.variable)
        if hasattr(node, 'next_functions'):  # Traverse computation graph
            for next_fn in node.next_functions:
                if next_fn[0] is not None:
                    trace_fn(next_fn[0])

    trace_fn(output.grad_fn)
    print('=====')
    param_name_map = {param: name for name, param in model.named_parameters()}
    print(len(parameters_involved))
    param_names_used = {param_name_map[p] for p in parameters_involved if p in param_name_map}
    for name in list(param_names_used):
        if 'lora' in name:
            print(name+'='*10+'lora!')
        else:
            print(name)

import pickle
@DETECTORS.register_module()
class NAVFormer(MVXTwoStageDetector):
    """NAVFormer"""

    def __init__(
        self,
        use_grid_mask=False,
        img_backbone=None,
        img_neck=None,
        pts_bbox_head=None,
        train_cfg=None,
        test_cfg=None,
        pretrained=None,
        video_test_mode=False,
        loss_cfg=None,
        planning_head=None,
        qim_args=dict(
            qim_type="QIMBase",
            merger_dropout=0,
            update_query_pos=False,
            fp_ratio=0.3,
            random_drop=0.1,
        ),
        mem_args=dict(
            memory_bank_type="MemoryBank",
            memory_bank_score_thresh=0.0,
            memory_bank_len=4,
        ),
        bbox_coder=dict(
            type="DETRTrack3DCoder",
            post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            pc_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
            max_num=300,
            num_classes=10,
            score_threshold=0.0,
            with_nms=False,
            iou_thres=0.3,
        ),
        pc_range=None,
        embed_dims=256,
        num_query=900,
        num_classes=10,
        vehicle_id_list=None,
        score_thresh=0.2,
        filter_score_thresh=0.1,
        miss_tolerance=5,
        gt_iou_threshold=0.0,
        freeze_img_backbone=False,
        freeze_img_neck=False,
        freeze_bn=False,
        freeze_bev_encoder=False,
        lora_finetuning=False,
        queue_length=3,
        process_perception=False,
    ):
        super(NAVFormer, self).__init__(
            img_backbone=img_backbone,
            img_neck=img_neck,
            pts_bbox_head=pts_bbox_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained,
        )

        self.process_perception = process_perception
        self.planning_head = build_head(planning_head)

        self.grid_mask = GridMask(
            True, True, rotate=1, offset=False, ratio=0.5, mode=1, prob=0.7
        )
        self.use_grid_mask = use_grid_mask
        self.fp16_enabled = False
        self.embed_dims = embed_dims
        self.num_query = num_query
        self.num_classes = num_classes
        self.vehicle_id_list = vehicle_id_list
        self.pc_range = pc_range
        self.queue_length = queue_length
        self.lora_finetuning = lora_finetuning
        self.freeze_bn = freeze_bn
        self.freeze_img_backbone = freeze_img_backbone
        self.freeze_img_neck = freeze_img_neck
        if freeze_img_backbone:
            if freeze_bn:
                self.img_backbone.eval()
            for param in self.img_backbone.parameters():
                param.requires_grad = False

        if freeze_img_neck:
            if freeze_bn:
                self.img_neck.eval()
            for param in self.img_neck.parameters():
                param.requires_grad = False

        # temporal
        self.video_test_mode = video_test_mode
        assert self.video_test_mode

        self.prev_frame_info = {
            "prev_bev": None,
            "scene_token": None,
            "prev_pos": 0,
            "prev_angle": 0,
        }
        
        # add one additional learnable query for the ego vehicle
        self.query_embedding = nn.Embedding(self.num_query + 1, self.embed_dims * 2)

        self.reference_points = nn.Linear(self.embed_dims, 3)
        self.bbox_size_fc = nn.Linear(self.embed_dims, 3)

        self.mem_bank_len = mem_args["memory_bank_len"]
        self.memory_bank = None
        self.track_base = RuntimeTrackerBase(
            score_thresh=score_thresh,
            filter_score_thresh=filter_score_thresh,
            miss_tolerance=miss_tolerance,
        )  # hyper-param for removing inactive queries

        self.query_interact = QueryInteractionModule(
            qim_args,
            dim_in=embed_dims,
            hidden_dim=embed_dims,
            dim_out=embed_dims,
        )

        self.bbox_coder = build_bbox_coder(bbox_coder)

        self.memory_bank = MemoryBank(
            mem_args,
            dim_in=embed_dims,
            hidden_dim=embed_dims,
            dim_out=embed_dims,
        )
        self.mem_bank_len = (
            0 if self.memory_bank is None else self.memory_bank.max_his_length
        )
        if self.process_perception:
            self.criterion = build_loss(loss_cfg)
        self.test_track_instances = None
        self.l2g_r_mat = None
        self.l2g_t = None
        self.gt_iou_threshold = gt_iou_threshold
        self.bev_h, self.bev_w = self.pts_bbox_head.bev_h, self.pts_bbox_head.bev_w
        self.freeze_bev_encoder = freeze_bev_encoder

        if self.freeze_bev_encoder:
            if freeze_bn:
                self.pts_bbox_head.eval()
            for param in self.pts_bbox_head.parameters():
                param.requires_grad = False

        # used to speed up inference
        self.skip_tracking = False if self.process_perception else True

        if self.lora_finetuning:
            self.img_backbone.eval()
            self.pts_bbox_head.eval()

    def extract_img_feat(self, img, len_queue=None):

        """Extract features of images."""
        if img is None:
            return None
        assert img.dim() == 5, f'image dim is {img.dim()}'
        B, N, C, H, W = img.size()    # 1 x 6 x 3 x 928 x 1600, 1 x 4 x 3 x 320 x 416
        img = img.reshape(B * N, C, H, W)
        if self.use_grid_mask:
            img = self.grid_mask(img)  

        # get multi-stage backbone features
        img_feats = self.img_backbone(img)
        if isinstance(img_feats, dict):
            img_feats = list(img_feats.values())
        if self.with_img_neck:
            img_feats = self.img_neck(img_feats)

        img_feats_reshaped = []
        for img_feat in img_feats:
            _, c, h, w = img_feat.size()
            if len_queue is not None:
                img_feat_reshaped = img_feat.view(B // len_queue, len_queue, N, c, h, w)
            else:
                img_feat_reshaped = img_feat.view(B, N, c, h, w)
            img_feats_reshaped.append(img_feat_reshaped)

        return img_feats_reshaped

    def _generate_empty_tracks(self):
        # create class for empty tracks in the first frame, same as MUTR3D

        track_instances = Instances((1, 1))
        num_queries, dim = self.query_embedding.weight.shape  # (N, 256 * 2)
        device = self.query_embedding.weight.device
        query = self.query_embedding.weight  # N x 512

        # convert the query embedding to a point in 3D
        track_instances.ref_pts = self.reference_points(query[..., : dim // 2])  # N x 3

        # init boxes: xy, wl, z, h, sin, cos, vx, vy, vz
        box_sizes = self.bbox_size_fc(query[..., : dim // 2])  # N x 3
        pred_boxes_init = torch.zeros(
            (len(track_instances), 10), dtype=torch.float, device=device
        )  # N x 10
        pred_boxes_init[..., 2:4] = box_sizes[..., 0:2]
        pred_boxes_init[..., 5:6] = box_sizes[..., 2:3]

        # track instance class, add other properties
        track_instances.query = query
        track_instances.output_embedding = torch.zeros(
            (num_queries, dim >> 1), device=device
        )  # N x 256
        track_instances.obj_idxes = torch.full(
            (len(track_instances),), -1, dtype=torch.long, device=device
        )  # N x 1
        track_instances.matched_gt_idxes = torch.full(
            (len(track_instances),), -1, dtype=torch.long, device=device
        )  # N x 1
        track_instances.disappear_time = torch.zeros(
            (len(track_instances),), dtype=torch.long, device=device
        )  # N x 1
        track_instances.iou = torch.zeros(
            (len(track_instances),), dtype=torch.float, device=device
        )  # N x 1
        track_instances.scores = torch.zeros(
            (len(track_instances),), dtype=torch.float, device=device
        )  # N x 1
        track_instances.track_scores = torch.zeros(
            (len(track_instances),), dtype=torch.float, device=device
        )
        # xy, wl, z, h, sin, cos, vx, vy, vz
        track_instances.pred_boxes = pred_boxes_init
        track_instances.pred_logits = torch.zeros(
            (len(track_instances), self.num_classes), dtype=torch.float, device=device
        )
        mem_bank_len = self.mem_bank_len
        track_instances.mem_bank = torch.zeros(
            (len(track_instances), mem_bank_len, dim // 2),
            dtype=torch.float32,
            device=device,
        )
        track_instances.mem_padding_mask = torch.ones(
            (len(track_instances), mem_bank_len), dtype=torch.bool, device=device
        )
        track_instances.save_period = torch.zeros(
            (len(track_instances),), dtype=torch.float32, device=device
        )

        return track_instances.to(self.query_embedding.weight.device)

    def velo_update(
        self, ref_pts, velocity, l2g_r1, l2g_t1, l2g_r2, l2g_t2, time_delta
    ):
        """
        Args:
            ref_pts (Tensor): (num_query, 3).  in inevrse sigmoid space
            velocity (Tensor): (num_query, 2). m/s
                in lidar frame. vx, vy
            global2lidar (np.Array) [4,4].
        Outs:
            ref_pts (Tensor): (num_query, 3).  in inevrse sigmoid space

            same as MUTR3D
        """
        
        # print(l2g_r1.type(), l2g_t1.type(), ref_pts.type())
        # change data type
        l2g_r1 = l2g_r1.float()
        l2g_t1 = l2g_t1.float()
        l2g_r2 = l2g_r2.float()
        l2g_t2 = l2g_t2.float()
        
        time_delta = time_delta.type(torch.float)
        num_query = ref_pts.size(0)
        velo_pad_ = velocity.new_zeros((num_query, 1))
        velo_pad = torch.cat((velocity, velo_pad_), dim=-1)

        # unnormalize
        reference_points = ref_pts.sigmoid().clone()
        pc_range = self.pc_range
        reference_points[..., 0:1] = (
            reference_points[..., 0:1] * (pc_range[3] - pc_range[0]) + pc_range[0]
        )
        reference_points[..., 1:2] = (
            reference_points[..., 1:2] * (pc_range[4] - pc_range[1]) + pc_range[1]
        )
        reference_points[..., 2:3] = (
            reference_points[..., 2:3] * (pc_range[5] - pc_range[2]) + pc_range[2]
        )

        # motion model
        reference_points = reference_points + velo_pad * time_delta

        # coordinate transform
        try:
            ref_pts = reference_points @ l2g_r1 + l2g_t1 - l2g_t2
        except RuntimeError:
            print("l2g_r1", l2g_r1.type())
            print("l2g_t1", l2g_t1.type())
            print("l2g_t2", l2g_t2.type())
            print("reference_points", reference_points.type())
            print("ref_pts", ref_pts.type())
            ref_pts = reference_points.float() @ l2g_r1.float() + l2g_t1.float() - l2g_t2.float()

        g2l_r = torch.linalg.inv(l2g_r2).type(torch.float)
        ref_pts = ref_pts @ g2l_r

        # unnormalize
        ref_pts[..., 0:1] = (ref_pts[..., 0:1] - pc_range[0]) / (
            pc_range[3] - pc_range[0]
        )
        ref_pts[..., 1:2] = (ref_pts[..., 1:2] - pc_range[1]) / (
            pc_range[4] - pc_range[1]
        )
        ref_pts[..., 2:3] = (ref_pts[..., 2:3] - pc_range[2]) / (
            pc_range[5] - pc_range[2]
        )
        ref_pts = inverse_sigmoid(ref_pts)

        return ref_pts

    def _copy_tracks_for_loss(self, tgt_instances):
        device = self.query_embedding.weight.device
        track_instances = Instances((1, 1))

        track_instances.obj_idxes = copy.deepcopy(tgt_instances.obj_idxes)

        track_instances.matched_gt_idxes = copy.deepcopy(tgt_instances.matched_gt_idxes)
        track_instances.disappear_time = copy.deepcopy(tgt_instances.disappear_time)

        track_instances.scores = torch.zeros(
            (len(track_instances),), dtype=torch.float, device=device
        )
        track_instances.track_scores = torch.zeros(
            (len(track_instances),), dtype=torch.float, device=device
        )
        track_instances.pred_boxes = torch.zeros(
            (len(track_instances), 10), dtype=torch.float, device=device
        )
        track_instances.iou = torch.zeros(
            (len(track_instances),), dtype=torch.float, device=device
        )
        track_instances.pred_logits = torch.zeros(
            (len(track_instances), self.num_classes), dtype=torch.float, device=device
        )

        track_instances.save_period = copy.deepcopy(tgt_instances.save_period)
        return track_instances.to(device)

    def get_history_bev(self, imgs_queue, img_metas_list):
        """
        Get history BEV features iteratively. To save GPU memory, gradients are not calculated.
        """
        training = False
        if self.training:
            training = True
            self.eval()
        with torch.no_grad():
            prev_bev = None
            bs, len_queue, num_cams, C, H, W = imgs_queue.shape   # 1 x 1 x num_cams
            imgs_queue = imgs_queue.reshape(bs * len_queue, num_cams, C, H, W)
            img_feats_list = self.extract_img_feat(img=imgs_queue, len_queue=len_queue)
            for i in range(len_queue):
                img_metas = [each[i] for each in img_metas_list]
                img_feats = [each_scale[:, i] for each_scale in img_feats_list]
                prev_bev, _ = self.pts_bbox_head.get_bev_features(
                    mlvl_feats=img_feats, img_metas=img_metas, prev_bev=prev_bev
                )    # 1 x (HW) x 256
        # restore training status
        if training:
            self.train()
            if self.freeze_bn:
                if self.freeze_img_backbone:
                    self.img_backbone.eval()
                if self.freeze_img_neck:
                    self.img_neck.eval()
                if self.freeze_bev_encoder:
                    self.pts_bbox_head.eval()
        return prev_bev

    # Generate bev using bev_encoder in BEVFormer
    def get_bevs(
        self, imgs, img_metas, prev_img=None, prev_img_metas=None, prev_bev=None
    ):
        # get features for past frames
        # prev_img: B x T x N
        if prev_img is not None and prev_img_metas is not None:
            assert prev_bev is None
            prev_bev = self.get_history_bev(prev_img, prev_img_metas)   # 1 x (HW) x 256

        # get features for the current frame
        img_feats: List[torch.Tensor] = self.extract_img_feat(img=imgs)
        # L of 4, 1 x N(6) x C(256) x H x W
        # stage1: 111 x 200, stage 2: 50 x 100, stage 3: 29 x 50, stage 4: 15 x 25

        if self.freeze_bev_encoder:
            with torch.no_grad():
                bev_embed, bev_pos = self.pts_bbox_head.get_bev_features(
                    mlvl_feats=img_feats, img_metas=img_metas, prev_bev=prev_bev
                )
        else:
            bev_embed, bev_pos = self.pts_bbox_head.get_bev_features(
                mlvl_feats=img_feats, img_metas=img_metas, prev_bev=prev_bev
            )
        # bev_embed: 1 x (HW) x 256
        # bev_pos:  1 x 256 x H x W

        if bev_embed.shape[1] == self.bev_h * self.bev_w:
            bev_embed = bev_embed.permute(1, 0, 2)   # HW x 1 x 256

        assert bev_embed.shape[0] == self.bev_h * self.bev_w
        return bev_embed, bev_pos

    @auto_fp16(apply_to=("img", "prev_img"))
    def _forward_single_frame(
        self,
        img,
        img_metas,
        track_instances,
        prev_img,
        prev_img_metas,
        l2g_r1=None,
        l2g_t1=None,
        l2g_r2=None,
        l2g_t2=None,
        time_delta=None,
        all_query_embeddings=None,
        all_matched_indices=None,
        all_instances_pred_logits=None,
        all_instances_pred_boxes=None,
    ):
        """
        Perform forward only on one frame. Called in  forward_train
        Warnning: Only Support BS=1
        Args:
            img: shape [B, num_cam, 3, H, W]
            if l2g_r2 is None or l2g_t2 is None:
                it means this frame is the end of the training clip,
                so no need to call velocity update
        """

        # NOTE: You can replace BEVFormer with other BEV encoder and provide bev_embed here
        bev_embed, bev_pos = self.get_bevs(
            img,
            img_metas,
            prev_img=prev_img,
            prev_img_metas=prev_img_metas,
        )

        if not self.process_perception:
            out = {
                "bev_embed": bev_embed,
                "bev_pos": bev_pos,
            }

            return out

        det_output = self.pts_bbox_head.get_detections(
            bev_embed,
            object_query_embeds=track_instances.query,
            ref_points=track_instances.ref_pts,
            img_metas=img_metas,
        )

        output_classes = det_output["all_cls_scores"]
        output_coords = det_output["all_bbox_preds"]
        output_past_trajs = det_output["all_past_traj_preds"]
        last_ref_pts = det_output["last_ref_points"]
        query_feats = det_output["query_feats"]

        out = {
            "pred_logits": output_classes[-1],
            "pred_boxes": output_coords[-1],
            "pred_past_trajs": output_past_trajs[-1],
            "ref_pts": last_ref_pts,
            "bev_embed": bev_embed,
            "bev_pos": bev_pos,
        }
        with torch.no_grad():
            track_scores = output_classes[-1, 0, :].sigmoid().max(dim=-1).values

        # Step-1 Update track instances with current prediction
        # [nb_dec, bs, num_query, xxx]
        nb_dec = output_classes.size(0)

        # the track id will be assigned by the matcher.
        track_instances_list = [
            self._copy_tracks_for_loss(track_instances) for i in range(nb_dec - 1)
        ]
        track_instances.output_embedding = query_feats[-1][0]  # [300, feat_dim]
        velo = output_coords[-1, 0, :, -2:]  # [num_query, 3]
        if l2g_r2 is not None:
            ref_pts = self.velo_update(
                last_ref_pts[0],
                velo,
                l2g_r1,
                l2g_t1,
                l2g_r2,
                l2g_t2,
                time_delta=time_delta,
            )
        else:
            ref_pts = last_ref_pts[0]

        dim = track_instances.query.shape[-1]
        track_instances.ref_pts = self.reference_points(
            track_instances.query[..., : dim // 2]
        )
        track_instances.ref_pts[..., :2] = ref_pts[..., :2]

        track_instances_list.append(track_instances)

        for i in range(nb_dec):
            track_instances = track_instances_list[i]

            track_instances.scores = track_scores
            track_instances.pred_logits = output_classes[i, 0]  # [300, num_cls]
            track_instances.pred_boxes = output_coords[i, 0]  # [300, box_dim]
            track_instances.pred_past_trajs = output_past_trajs[
                i, 0
            ]  # [300,past_steps, 2]

            out["track_instances"] = track_instances
            track_instances, matched_indices = self.criterion.match_for_single_frame(
                out, i, if_step=(i == (nb_dec - 1))
            )
            all_query_embeddings.append(query_feats[i][0])
            all_matched_indices.append(matched_indices)
            all_instances_pred_logits.append(output_classes[i, 0])
            all_instances_pred_boxes.append(output_coords[i, 0])  # Not used

        # print(track_instances.obj_idxes)
        # print(track_instances.obj_idxes.size())

        active_index = (
            (track_instances.obj_idxes >= 0)
            & (track_instances.iou >= self.gt_iou_threshold)
            & (track_instances.matched_gt_idxes >= 0)
        )
        out.update(
            self.select_active_track_query(track_instances, active_index, img_metas)
        )
        out.update(self.select_sdc_track_query(track_instances[900], img_metas))

        # memory bank
        if self.memory_bank is not None:
            track_instances = self.memory_bank(track_instances)
        # Step-2 Update track instances using matcher

        tmp = {}
        tmp["init_track_instances"] = self._generate_empty_tracks()
        tmp["track_instances"] = track_instances
        out_track_instances = self.query_interact(tmp)
        out["track_instances"] = out_track_instances
        return out

    def select_active_track_query(
        self, track_instances, active_index, img_metas, with_mask=True
    ):
        result_dict = self._track_instances2results(
            track_instances[active_index], img_metas, with_mask=with_mask
        )
        result_dict["track_query_embeddings"] = track_instances.output_embedding[
            active_index
        ][result_dict["bbox_index"]][result_dict["mask"]]
        result_dict["track_query_matched_idxes"] = track_instances.matched_gt_idxes[
            active_index
        ][result_dict["bbox_index"]][result_dict["mask"]]
        return result_dict

    def select_sdc_track_query(self, sdc_instance, img_metas):
        out = dict()
        result_dict = self._track_instances2results(
            sdc_instance, img_metas, with_mask=False
        )
        out["sdc_boxes_3d"] = result_dict["boxes_3d"]
        out["sdc_scores_3d"] = result_dict["scores_3d"]
        out["sdc_track_scores"] = result_dict["track_scores"]
        out["sdc_track_bbox_results"] = result_dict["track_bbox_results"]
        out["sdc_embedding"] = sdc_instance.output_embedding[0]
        return out
    
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
        sdc_planning=None,
        sdc_planning_mask=None,
        command=None,
        sdc_planning_past=None,
        sdc_planning_mask_past=None,
        gt_pre_command_sdc=None,
        sdc_status=None,
        no_at_fault_collisions=None,
        drivable_area_compliance=None,
        ego_progress=None,
        time_to_collision_within_bound=None,
        comfort=None,
        score=None,
        fail_mask=None,
        # fut gt for planning
        gt_future_boxes=None,
        gt_past_traj=None,
        gt_past_traj_mask=None,
        gt_sdc_bbox=None,
        gt_sdc_label=None,
        **kwargs,  # [1, 9]
    ):  

        if self.process_perception:
            track_losses, outs_track = self.forward_track_train(
                img, gt_bboxes_3d,
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
        else:
            outs_track = self.forward_track(
                img,
                gt_bboxes_3d,
                gt_labels_3d,
                gt_inds,
                l2g_t,
                l2g_r_mat,
                img_metas,
                timestamp,
            )

        bev_embed = outs_track['bev_embed']

        plan_results = self.planning_head.forward(
            bev_embed,
            command,
            sdc_planning_past, # 1 x 4 x 4
            sdc_status,
            sdc_planning_mask_past,  # 1 x 4 x 4
            gt_pre_command_sdc, #1*4
        )
        pdm_dict = {
            "no_at_fault_collisions":no_at_fault_collisions,
            "drivable_area_compliance":drivable_area_compliance,
            "ego_progress":ego_progress,
            "time_to_collision_within_bound":time_to_collision_within_bound,
            "comfort":comfort,
            "score":score,
        }

        if fail_mask is not None:
            pdm_dict['fail_mask'] = fail_mask

        losses = self.planning_head.loss(
            plan_results,
            gt_pdm_score=pdm_dict,
            sdc_planning=sdc_planning,
            sdc_planning_mask=sdc_planning_mask
        )

        if self.process_perception:
            losses.update(track_losses)

        for k, v in losses.items():
            losses[k] = torch.nan_to_num(v)

        return losses


    @auto_fp16(apply_to=("img", "points"))
    def forward_test(
        self,
        img=None,
        img_metas=None,
        gt_bboxes_3d=None,
        gt_labels_3d=None,
        gt_inds=None,
        l2g_t=None,
        l2g_r_mat=None,
        timestamp=None,
        sdc_planning=None,
        sdc_planning_mask=None,
        command=None,
        sdc_planning_past=None,
        sdc_planning_mask_past=None,
        gt_pre_command_sdc=None,
        sdc_status=None,
        no_at_fault_collisions=None,
        drivable_area_compliance=None,
        ego_progress=None,
        time_to_collision_within_bound=None,
        comfort=None,
        score=None,
        # fut gt for planning
        gt_future_boxes=None,
        **kwargs,  # [1, 9]
    ):
        outs_track = self.forward_track_test(
            img,
            gt_bboxes_3d,
            gt_labels_3d,
            gt_inds,
            l2g_t,
            l2g_r_mat,
            img_metas,
            timestamp,
        )

        bev_embed = outs_track['bev_embed']

        plan_results = self.planning_head.forward(
            bev_embed,
            command[0],
            sdc_planning_past[0],
            sdc_status[0],
            sdc_planning_mask_past[0],
            gt_pre_command_sdc[0],
        )

        chosen_indices = plan_results['selected_indices']
        b = img.shape[0]

        return_list = []
        for batch_idx in range(b):
            chosen_idx = chosen_indices[batch_idx]
            pdm_dict = {
                "token":img_metas[batch_idx][3]['sample_idx'],  # TODO: hard code 3rd frame for current frame
                "no_at_fault_collisions":no_at_fault_collisions[0][batch_idx, chosen_idx].item(),
                "drivable_area_compliance":drivable_area_compliance[0][batch_idx, chosen_idx].item(),
                "ego_progress":ego_progress[0][batch_idx, chosen_idx].item(),
                "time_to_collision_within_bound":time_to_collision_within_bound[0][batch_idx, chosen_idx].item(),
                "comfort":comfort[0][batch_idx, chosen_idx].item(),
                "score":score[0][batch_idx, chosen_idx].item(),
                'chosen_ind':chosen_idx.item(),
                'trajectory':plan_results['trajectory'][batch_idx].cpu().numpy(),
            }

            # Calculate ADE / FDE
            gt_traj = sdc_planning[0][batch_idx, 0, :, :2].cpu().numpy()  #[b, 1, 8, 3] -> [8, 2]
            # from 10Hz to 2Hz
            pred_traj = pdm_dict['trajectory'][4::5, :2]  # [40, 3] -> [8, 2]
            ade = np.mean(np.linalg.norm(pred_traj - gt_traj, axis=-1))
            fde = np.linalg.norm(pred_traj[-1] - gt_traj[-1])
            pdm_dict['ade_4s'] = ade
            pdm_dict['fde_4s'] = fde

            return_list.append(pdm_dict)
        return return_list

    @auto_fp16(apply_to=("img", "points"))
    def forward_track_test(
        self,
        img,
        gt_bboxes_3d,
        gt_labels_3d,
        gt_inds,
        l2g_t,
        l2g_r_mat,
        img_metas,
        timestamp,
    ):
        """Forward funciton
        Args:
        Returns:
        """
        track_instances = self._generate_empty_tracks()
        # img = img[:, -2:]
        num_frame = img.size(1)

        out = dict()

        for i in range(num_frame):
            prev_img = img[:, :i, ...] if i != 0 else img[:, :1, ...]
            prev_img_metas = copy.deepcopy(img_metas)
            # TODO: Generate prev_bev in an RNN way.

            img_single = torch.stack([img_[i] for img_ in img], dim=0)
            # img_metas_single = [copy.deepcopy(img_meta[2+i]) for img_meta in img_metas]
            img_metas_single = [copy.deepcopy(img_meta[i]) for img_meta in img_metas]
            if i == num_frame - 1:
                l2g_r2 = None
                l2g_t2 = None
                time_delta = None
            else:
                l2g_r2 = l2g_r_mat[0][i + 1]
                l2g_t2 = l2g_t[0][i + 1]
                time_delta = timestamp[0][i + 1] - timestamp[0][i]
            all_query_embeddings = []
            all_matched_idxes = []
            all_instances_pred_logits = []
            all_instances_pred_boxes = []
            frame_res = self._forward_single_frame(
                img_single,
                img_metas_single,
                track_instances,
                prev_img,
                prev_img_metas,
                l2g_r_mat[0][i],
                l2g_t[0][i],
                l2g_r2,
                l2g_t2,
                time_delta,
                all_query_embeddings,
                all_matched_idxes,
                all_instances_pred_logits,
                all_instances_pred_boxes,
            )
            

        get_keys = [
            "bev_embed",
            "bev_pos",
        ]
        out.update({k: frame_res[k] for k in get_keys})

        return out


    @auto_fp16(apply_to=("img", "points"))
    def forward_track(
        self,
        img,
        gt_bboxes_3d,
        gt_labels_3d,
        gt_inds,
        l2g_t,
        l2g_r_mat,
        img_metas,
        timestamp,
    ):
        """Forward funciton
        Args:
        Returns:
        """
        track_instances = self._generate_empty_tracks()
        # img = img[:, -2:]
        num_frame = img.size(1)

        # perform detection & tracking
        out = dict()
        for i in range(num_frame):
            prev_img = img[:, :i, ...] if i != 0 else img[:, :1, ...]
            prev_img_metas = copy.deepcopy(img_metas)
            # TODO: Generate prev_bev in an RNN way.

            img_single = torch.stack([img_[i] for img_ in img], dim=0)
            #List[b] List[frame] tensor:[dim]
            # img_metas_single = [copy.deepcopy(img_meta[2+i]) for img_meta in img_metas]
            img_metas_single = [copy.deepcopy(img_meta[i]) for img_meta in img_metas]
            if i == num_frame - 1:
                l2g_r2 = None
                l2g_t2 = None
                time_delta = None
            else:
                l2g_r2 = l2g_r_mat[0][i + 1]
                l2g_t2 = l2g_t[0][i + 1]
                time_delta = timestamp[0][i + 1] - timestamp[0][i]
            all_query_embeddings = []
            all_matched_idxes = []
            all_instances_pred_logits = []
            all_instances_pred_boxes = []
            frame_res = self._forward_single_frame(
                img_single,
                img_metas_single,
                track_instances,
                prev_img,
                prev_img_metas,
                l2g_r_mat[0][i],
                l2g_t[0][i],
                l2g_r2,
                l2g_t2,
                time_delta,
                all_query_embeddings,
                all_matched_idxes,
                all_instances_pred_logits,
                all_instances_pred_boxes,
            )
            

        get_keys = [
            "bev_embed",
            "bev_pos",
        ]
        out.update({k: frame_res[k] for k in get_keys})

        return out

    @property
    def with_planning_head(self):
        return hasattr(self, "planning_head") and self.planning_head is not None

    @auto_fp16(apply_to=("img", "points"))
    def forward_track_train(
        self,
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
    ):
        """Forward funciton
        Args:
        Returns:
        """
        track_instances = self._generate_empty_tracks()
        num_frame = img.size(1)

        # print(track_instances.obj_idxes)
        # print(track_instances.obj_idxes.size())

        # init gt instances!
        gt_instances_list = []
        for i in range(num_frame):
            gt_instances = Instances((1, 1))
            boxes = gt_bboxes_3d[0][i].tensor.to(img.device)
            # normalize gt bboxes here!
            boxes = normalize_bbox(boxes, self.pc_range)

            # TODO: debug the orientation and shape here
            sd_boxes = gt_sdc_bbox[0][i].tensor.to(img.device)
            sd_boxes = normalize_bbox(sd_boxes, self.pc_range)
            gt_instances.boxes = boxes
            gt_instances.labels = gt_labels_3d[0][i]
            gt_instances.obj_ids = gt_inds[0][i]
            gt_instances.past_traj = gt_past_traj[0][i].float()
            gt_instances.past_traj_mask = gt_past_traj_mask[0][i].float()
            if boxes.shape[0] == 0:
                gt_instances.sdc_boxes = torch.zeros((0, 7), device=img.device)
                gt_instances.sdc_labels = torch.zeros((0,), device=img.device)
            else:
                gt_instances.sdc_boxes = torch.cat(
                    [sd_boxes for _ in range(boxes.shape[0])], dim=0
                )  # boxes.shape[0] sometimes 0
                gt_instances.sdc_labels = torch.cat(
                    [gt_sdc_label[0][i] for _ in range(gt_labels_3d[0][i].shape[0])], dim=0
                )
            gt_instances_list.append(gt_instances)

        self.criterion.initialize_for_single_clip(gt_instances_list)

        # perform detection & tracking
        out = dict()
        for i in range(num_frame):
            prev_img = img[:, :i, ...] if i != 0 else img[:, :1, ...]
            prev_img_metas = copy.deepcopy(img_metas)
            # TODO: Generate prev_bev in an RNN way.

            img_single = torch.stack([img_[i] for img_ in img], dim=0)
            # img_metas_single = [copy.deepcopy(img_metas[0][i])]
            img_metas_single = [copy.deepcopy(img_meta[i]) for img_meta in img_metas]
            if i == num_frame - 1:
                l2g_r2 = None
                l2g_t2 = None
                time_delta = None
            else:
                l2g_r2 = l2g_r_mat[0][i + 1]
                l2g_t2 = l2g_t[0][i + 1]
                time_delta = timestamp[0][i + 1] - timestamp[0][i]
            all_query_embeddings = []
            all_matched_idxes = []
            all_instances_pred_logits = []
            all_instances_pred_boxes = []

            frame_res = self._forward_single_frame(
                img_single,
                img_metas_single,
                track_instances,
                prev_img,
                prev_img_metas,
                l2g_r_mat[0][i],
                l2g_t[0][i],
                l2g_r2,
                l2g_t2,
                time_delta,
                all_query_embeddings,
                all_matched_idxes,
                all_instances_pred_logits,
                all_instances_pred_boxes,
            )
            # all_query_embeddings: len=dec nums, N*256
            # all_matched_idxes: len=dec nums, N*2
            track_instances = frame_res["track_instances"]

        get_keys = [
            "bev_embed",
            "bev_pos",
            "track_query_embeddings",
            "track_query_matched_idxes",
            "track_bbox_results",
            "sdc_boxes_3d",
            "sdc_scores_3d",
            "sdc_track_scores",
            "sdc_track_bbox_results",
            "sdc_embedding",
        ]
        out.update({k: frame_res[k] for k in get_keys})

        losses = self.criterion.losses_dict
        return losses, out

    def upsample_bev_if_tiny(self, outs_track):
        if outs_track["bev_embed"].size(0) == 100 * 100:
            # For tiny model
            # bev_emb
            bev_embed = outs_track["bev_embed"]  # [10000, 1, 256]
            dim, _, _ = bev_embed.size()
            w = h = int(math.sqrt(dim))
            assert h == w == 100

            bev_embed = rearrange(
                bev_embed, "(h w) b c -> b c h w", h=h, w=w
            )  # [1, 256, 100, 100]
            bev_embed = nn.Upsample(scale_factor=2)(bev_embed)  # [1, 256, 200, 200]
            bev_embed = rearrange(bev_embed, "b c h w -> (h w) b c")
            outs_track["bev_embed"] = bev_embed

            # prev_bev
            prev_bev = outs_track.get("prev_bev", None)
            if prev_bev is not None:
                if self.training:
                    #  [1, 10000, 256]
                    prev_bev = rearrange(prev_bev, "b (h w) c -> b c h w", h=h, w=w)
                    prev_bev = nn.Upsample(scale_factor=2)(
                        prev_bev
                    )  # [1, 256, 200, 200]
                    prev_bev = rearrange(prev_bev, "b c h w -> b (h w) c")
                    outs_track["prev_bev"] = prev_bev
                else:
                    #  [10000, 1, 256]
                    prev_bev = rearrange(prev_bev, "(h w) b c -> b c h w", h=h, w=w)
                    prev_bev = nn.Upsample(scale_factor=2)(
                        prev_bev
                    )  # [1, 256, 200, 200]
                    prev_bev = rearrange(prev_bev, "b c h w -> (h w) b c")
                    outs_track["prev_bev"] = prev_bev

            # bev_pos
            bev_pos = outs_track["bev_pos"]  # [1, 256, 100, 100]
            bev_pos = nn.Upsample(scale_factor=2)(bev_pos)  # [1, 256, 200, 200]
            outs_track["bev_pos"] = bev_pos
        return outs_track

    def _inference_single_frame(
        self,
        img,
        img_metas,
        track_instances,
        prev_bev=None,
        l2g_r1=None,
        l2g_t1=None,
        l2g_r2=None,
        l2g_t2=None,
        time_delta=None,
    ):
        """
        img: B, num_cam, C, H, W = img.shape
        """

        if not self.skip_tracking:
            """ velo update """
            active_inst = track_instances[track_instances.obj_idxes >= 0]
            other_inst = track_instances[track_instances.obj_idxes < 0]

            if l2g_r2 is not None and len(active_inst) > 0 and l2g_r1 is not None:
                ref_pts = active_inst.ref_pts
                velo = active_inst.pred_boxes[:, -2:]
                ref_pts = self.velo_update(
                    ref_pts, velo, l2g_r1, l2g_t1, l2g_r2, l2g_t2, time_delta=time_delta
                )
                ref_pts = ref_pts.squeeze(0)
                dim = active_inst.query.shape[-1]
                active_inst.ref_pts = self.reference_points(
                    active_inst.query[..., : dim // 2]
                )
                active_inst.ref_pts[..., :2] = ref_pts[..., :2]

            track_instances = Instances.cat([other_inst, active_inst])

        # NOTE: You can replace BEVFormer with other BEV encoder and provide bev_embed here
        bev_embed, bev_pos = self.get_bevs(img, img_metas, prev_bev=prev_bev)
        
        if not self.skip_tracking:
            det_output = self.pts_bbox_head.get_detections(
                bev_embed,
                object_query_embeds=track_instances.query,
                ref_points=track_instances.ref_pts,
                img_metas=img_metas,
            )
            output_classes = det_output["all_cls_scores"]
            output_coords = det_output["all_bbox_preds"]
            last_ref_pts = det_output["last_ref_points"]
            query_feats = det_output["query_feats"]

            out = {
                "pred_logits": output_classes,
                "pred_boxes": output_coords,
                "ref_pts": last_ref_pts,
                "bev_embed": bev_embed,
                "query_embeddings": query_feats,
                "all_past_traj_preds": det_output["all_past_traj_preds"],
                "bev_pos": bev_pos,
            }

            """ update track instances with predict results """
            track_scores = output_classes[-1, 0, :].sigmoid().max(dim=-1).values
            # each track will be assigned an unique global id by the track base.
            track_instances.scores = track_scores
            # track_instances.track_scores = track_scores  # [300]
            track_instances.pred_logits = output_classes[-1, 0]  # [300, num_cls]
            track_instances.pred_boxes = output_coords[-1, 0]  # [300, box_dim]
            track_instances.output_embedding = query_feats[-1][0]  # [300, feat_dim]
            track_instances.ref_pts = last_ref_pts[0]

            # hard_code: assume the 901 query is sdc query
            track_instances.obj_idxes[900] = -2
            """ update track base """
            self.track_base.update(track_instances, None)

            active_index = (track_instances.obj_idxes >= 0) & (
                track_instances.scores >= self.track_base.filter_score_thresh
            )  # filter out sleep objects
            out.update(
                self.select_active_track_query(track_instances, active_index, img_metas)
            )
            out.update(
                self.select_sdc_track_query(
                    track_instances[track_instances.obj_idxes == -2], img_metas
                )
            )

            """ update with memory_bank """
            if self.memory_bank is not None:
                track_instances = self.memory_bank(track_instances)

            """  Update track instances using matcher """
            tmp = {}
            tmp["init_track_instances"] = self._generate_empty_tracks()
            tmp["track_instances"] = track_instances
            out_track_instances = self.query_interact(tmp)
            out["track_instances_fordet"] = track_instances
            out["track_instances"] = out_track_instances
            out["track_obj_idxes"] = track_instances.obj_idxes
        else:
            out = {
                "bev_embed": bev_embed,
                "bev_pos": bev_pos,
            }

        return out

    def simple_test_track(
        self,
        img=None,
        l2g_t=None,
        l2g_r_mat=None,
        img_metas=None,
        timestamp=None,
    ):
        """only support bs=1 and sequential input"""

        bs = img.size(0)
        # img_metas = img_metas[0]

        """ init track instances for first frame """
        if (
            self.test_track_instances is None
            or img_metas[0]["scene_token"] != self.scene_token
        ):
            self.timestamp = timestamp
            try:
                self.scene_token = img_metas[0]["scene_token"]
            except TypeError:
                self.scene_token = ''
            self.prev_bev = None
            track_instances = self._generate_empty_tracks()
            time_delta, l2g_r1, l2g_t1, l2g_r2, l2g_t2 = None, None, None, None, None

        else:
            track_instances = self.test_track_instances
            time_delta = timestamp - self.timestamp
            l2g_r1 = self.l2g_r_mat
            l2g_t1 = self.l2g_t
            l2g_r2 = l2g_r_mat
            l2g_t2 = l2g_t

        """ get time_delta and l2g r/t infos """
        """ update frame info for next frame"""
        self.timestamp = timestamp
        self.l2g_t = l2g_t
        self.l2g_r_mat = l2g_r_mat

        """ predict and update """
        prev_bev = self.prev_bev
        frame_res = self._inference_single_frame(
            img,
            img_metas,
            track_instances,
            prev_bev,
            l2g_r1,
            l2g_t1,
            l2g_r2,
            l2g_t2,
            time_delta,
        )

        self.prev_bev = frame_res["bev_embed"]
        results = [dict()]

        if not self.skip_tracking:
            track_instances = frame_res["track_instances"]
            track_instances_fordet = frame_res["track_instances_fordet"]
            self.test_track_instances = track_instances
            get_keys = [
                "bev_embed",    # HW x 1 x 256
                "bev_pos",      # 1 x 256 x H x W
                "track_query_embeddings",   # N x 256
                "track_bbox_results",   # List of 5, LiDARInstance3DBoxes, N x 9
                "boxes_3d",     # N x 9
                "scores_3d",    # N
                "labels_3d",    # N
                "track_scores", # N
                "track_ids",    # N
            ]
            if self.with_motion_head:
                get_keys += [
                    "sdc_boxes_3d",
                    "sdc_scores_3d",
                    "sdc_track_scores",
                    "sdc_track_bbox_results",
                    "sdc_embedding",
                ]
        else:
            get_keys = ["bev_embed", "bev_pos"]

        results[0].update({k: frame_res[k] for k in get_keys})

        if not self.skip_tracking:
            results = self._det_instances2results(
                track_instances_fordet, results, img_metas
            )

        return results

    def _track_instances2results(self, track_instances, img_metas, with_mask=True):
        bbox_dict = dict(
            cls_scores=track_instances.pred_logits,
            bbox_preds=track_instances.pred_boxes,
            track_scores=track_instances.scores,
            obj_idxes=track_instances.obj_idxes,
        )
        # bboxes_dict = self.bbox_coder.decode(bbox_dict, with_mask=with_mask)[0]
        bboxes_dict = self.bbox_coder.decode(
            bbox_dict, with_mask=with_mask, img_metas=img_metas
        )[0]
        bboxes = bboxes_dict["bboxes"]

        bboxes = img_metas[0]["box_type_3d"](bboxes, 9)
        labels = bboxes_dict["labels"]
        scores = bboxes_dict["scores"]
        bbox_index = bboxes_dict["bbox_index"]

        track_scores = bboxes_dict["track_scores"]
        obj_idxes = bboxes_dict["obj_idxes"]
        result_dict = dict(
            boxes_3d=bboxes.to("cpu"),
            scores_3d=scores.cpu(),
            labels_3d=labels.cpu(),
            track_scores=track_scores.cpu(),
            bbox_index=bbox_index.cpu(),
            track_ids=obj_idxes.cpu(),
            mask=bboxes_dict["mask"].cpu(),
            track_bbox_results=[
                [
                    bboxes.to("cpu"),
                    scores.cpu(),
                    labels.cpu(),
                    bbox_index.cpu(),
                    bboxes_dict["mask"].cpu(),
                ]
            ],
        )
        return result_dict

    def _det_instances2results(self, instances, results, img_metas):
        """
        Outs:
        active_instances. keys:
        - 'pred_logits':
        - 'pred_boxes': normalized bboxes
        - 'scores'
        - 'obj_idxes'
        out_dict. keys:
            - boxes_3d (torch.Tensor): 3D boxes.
            - scores (torch.Tensor): Prediction scores.
            - labels_3d (torch.Tensor): Box labels.
            - attrs_3d (torch.Tensor, optional): Box attributes.
            - track_ids
            - tracking_score
        """
        # filter out sleep querys
        if instances.pred_logits.numel() == 0:
            return [None]
        
        # normalized raw results
        bbox_dict = dict(
            cls_scores=instances.pred_logits,
            bbox_preds=instances.pred_boxes,    # N x 10
            track_scores=instances.scores,
            obj_idxes=instances.obj_idxes,
        )

        # unnormalized results
        bboxes_dict = self.bbox_coder.decode(bbox_dict, img_metas=img_metas)[0]
        bboxes = bboxes_dict["bboxes"]      
        bboxes = img_metas[0]["box_type_3d"](bboxes, 9) # N x 9, xyzlwhyaw
        labels = bboxes_dict["labels"]
        scores = bboxes_dict["scores"]

        track_scores = bboxes_dict["track_scores"]
        obj_idxes = bboxes_dict["obj_idxes"]
        result_dict = results[0]

        # print(result_dict.keys())
        result_dict_det = dict(
            boxes_3d_det=bboxes.to("cpu"),
            scores_3d_det=scores.cpu(),
            labels_3d_det=labels.cpu(),
        )
        if result_dict is not None:
            result_dict.update(result_dict_det)
        else:
            result_dict = None

        return [result_dict]
