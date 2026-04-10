from copy import deepcopy
from os import times

import torch
import torch.nn as nn
from mmcv.runner import auto_fp16, force_fp32

from mmdet3d.core import bbox3d2result, merge_aug_bboxes_3d
from mmdet3d.core.bbox.coders import build_bbox_coder
from mmdet3d.models.detectors.mvx_two_stage import MVXTwoStageDetector
from mmdet.models import DETECTORS, build_loss
from mmdet3d_plugin.core.bbox.util import denormalize_bbox, normalize_bbox
from .grid_mask import GridMask
from .memory_bank import build_memory_bank
from .qim import build_qim
from .radar_encoder import build_radar_encoder
from .structures import Instances


def inverse_sigmoid(x, eps=1e-5):
    """Inverse function of sigmoid.

    Args:
        x (Tensor): The tensor to do the
            inverse.
        eps (float): EPS avoid numerical
            overflow. Defaults 1e-5.
    Returns:
        Tensor: The x has passed the inverse
            function of sigmoid, has same
            shape with input.
    """
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1 / x2)


# this class is from MOTR
class RuntimeTrackerBase(object):
    # code from https://github.com/megvii-model/MOTR/blob/main/models/motr.py#L303
    def __init__(self, score_thresh=0.7, filter_score_thresh=0.6, miss_tolerance=5):
        self.score_thresh = score_thresh
        self.filter_score_thresh = filter_score_thresh
        self.miss_tolerance = miss_tolerance
        self.max_obj_id = 0

    def clear(self):
        self.max_obj_id = 0

    def update(self, track_instances: Instances):
        track_instances.disappear_time[track_instances.scores >= self.score_thresh] = 0
        for i in range(len(track_instances)):
            if (
                track_instances.obj_idxes[i] == -1
                and track_instances.scores[i] >= self.score_thresh
            ):
                # new track
                # print("track {} has score {}, assign obj_id {}".format(i, track_instances.scores[i], self.max_obj_id))
                track_instances.obj_idxes[i] = self.max_obj_id
                self.max_obj_id += 1
            elif (
                track_instances.obj_idxes[i] >= 0
                and track_instances.scores[i] < self.filter_score_thresh
            ):
                # sleep time ++
                track_instances.disappear_time[i] += 1
                if track_instances.disappear_time[i] >= self.miss_tolerance:
                    # mark deaded tracklets: Set the obj_id to -1.
                    # TODO: remove it by following functions
                    # Then this track will be removed by TrackEmbeddingLayer.
                    track_instances.obj_idxes[i] = -1

    def update_fix_label(self, track_instances: Instances, old_class_scores):
        track_instances.disappear_time[track_instances.scores >= self.score_thresh] = 0
        for i in range(len(track_instances)):
            if (
                track_instances.obj_idxes[i] == -1
                and track_instances.scores[i] >= self.score_thresh
            ):
                # new track
                # print("track {} has score {}, assign obj_id {}".format(i, track_instances.scores[i], self.max_obj_id))
                track_instances.obj_idxes[i] = self.max_obj_id
                self.max_obj_id += 1
            elif (
                track_instances.obj_idxes[i] >= 0
                and track_instances.scores[i] < self.filter_score_thresh
            ):
                # sleep time ++
                track_instances.disappear_time[i] += 1
                # keep class unchanged!
                track_instances.pred_logits[i] = old_class_scores[i]
                if track_instances.disappear_time[i] >= self.miss_tolerance:
                    # mark deaded tracklets: Set the obj_id to -1.
                    # TODO: remove it by following functions
                    # Then this track will be removed by TrackEmbeddingLayer.
                    track_instances.obj_idxes[i] = -1
            elif (
                track_instances.obj_idxes[i] >= 0
                and track_instances.scores[i] >= self.filter_score_thresh
            ):
                # keep class unchanged!
                track_instances.pred_logits[i] = old_class_scores[i]


@DETECTORS.register_module()
class MUTRCamTracker(MVXTwoStageDetector):
    """Tracker which support image w, w/o radar."""

    def __init__(
        self,
        embed_dims=256,
        num_query=300,
        num_classes=7,
        bbox_coder=dict(
            type="DETRTrack3DCoder",
            post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            pc_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
            max_num=300,
            num_classes=7,
        ),
        qim_args=dict(
            qim_type="QIMBase",
            merger_dropout=0,
            update_query_pos=False,
            fp_ratio=0.3,
            random_drop=0.1,
        ),
        mem_cfg=dict(
            memory_bank_type="MemoryBank",
            memory_bank_score_thresh=0.0,
            memory_bank_len=4,
        ),
        radar_encoder=None,
        fix_feats=False,
        score_thresh=0.2,
        filter_score_thresh=0.1,
        use_grid_mask=False,
        pts_voxel_layer=None,
        pts_voxel_encoder=None,
        pts_middle_encoder=None,
        pts_fusion_layer=None,
        img_backbone=None,
        pts_backbone=None,
        img_neck=None,
        pts_neck=None,
        loss_cfg=None,
        pts_bbox_head=None,
        img_roi_head=None,
        img_rpn_head=None,
        train_cfg=None,
        test_cfg=None,
        pretrained=None,
    ):
        super(MUTRCamTracker, self).__init__(
            pts_voxel_layer,
            pts_voxel_encoder,
            pts_middle_encoder,
            pts_fusion_layer,
            img_backbone,
            pts_backbone,
            img_neck,
            pts_neck,
            pts_bbox_head,
            img_roi_head,
            img_rpn_head,
            train_cfg,
            test_cfg,
            pretrained,
        )
        self.grid_mask = GridMask(
            True, True, rotate=1, offset=False, ratio=0.5, mode=1, prob=0.7
        )
        self.use_grid_mask = use_grid_mask
        self.num_classes = num_classes
        self.bbox_coder = build_bbox_coder(bbox_coder)
        self.pc_range = self.bbox_coder.pc_range

        self.embed_dims = embed_dims   # 256
        self.num_query = num_query
        self.fix_feats = fix_feats
        if self.fix_feats:
            self.img_backbone.eval()
            self.img_neck.eval()
        self.reference_points = nn.Linear(self.embed_dims, 3)
        self.bbox_size_fc = nn.Linear(self.embed_dims, 3)
        self.query_embedding = nn.Embedding(self.num_query, self.embed_dims * 2)
        self.mem_bank_len = mem_cfg["memory_bank_len"]
        self.memory_bank = None
        self.track_base = RuntimeTrackerBase(
            score_thresh=score_thresh,
            filter_score_thresh=filter_score_thresh,
            miss_tolerance=5,
        )  # hyper-param for removing inactive queries

        self.query_interact = build_qim(
            qim_args,
            dim_in=embed_dims,
            hidden_dim=embed_dims,
            dim_out=embed_dims,
        )

        self.memory_bank = build_memory_bank(
            args=mem_cfg,
            dim_in=embed_dims,
            hidden_dim=embed_dims,
            dim_out=embed_dims,
        )
        self.mem_bank_len = (
            0 if self.memory_bank is None else self.memory_bank.max_his_length
        )
        self.criterion = build_loss(loss_cfg)
        self.test_track_instances = None
        self.l2g_r_mat = None
        self.l2g_t = None

        self.radar_encoder = build_radar_encoder(radar_encoder)

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
        """
        # print(l2g_r1.type(), l2g_t1.type(), ref_pts.type())
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
        ref_pts = reference_points @ l2g_r1 + l2g_t1 - l2g_t2
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

    def extract_pts_feat(self, pts, img_feats, img_metas):
        """Extract features of points."""
        if not self.with_pts_bbox:
            return None
        voxels, num_points, coors = self.voxelize(pts)

        voxel_features = self.pts_voxel_encoder(voxels, num_points, coors)
        batch_size = coors[-1, 0] + 1
        x = self.pts_middle_encoder(voxel_features, coors, batch_size)
        x = self.pts_backbone(x)
        if self.with_pts_neck:
            x = self.pts_neck(x)
        return x

    def extract_img_feat(self, img, img_metas):
        """Extract features of images."""
        B = img.size(0)
        if self.with_img_backbone and img is not None:
            input_shape = img.shape[-2:]
            # update real input shape of each single img
            for img_meta in img_metas:
                img_meta.update(input_shape=input_shape)

            if img.dim() == 5 and img.size(0) == 1:
                img.squeeze_()
            elif img.dim() == 5 and img.size(0) > 1:
                B, N, C, H, W = img.size()
                img = img.view(B * N, C, H, W)
            if self.use_grid_mask:
                img = self.grid_mask(img)

            img_feats = self.img_backbone(img)
        else:
            return None
        if self.with_img_neck:
            img_feats = self.img_neck(img_feats)
        img_feats_reshaped = []
        for img_feat in img_feats:
            BN, C, H, W = img_feat.size()
            img_feats_reshaped.append(img_feat.view(B, int(BN / B), C, H, W))
        return img_feats_reshaped

    @auto_fp16(apply_to=("img"), out_fp32=True)
    def extract_feat(self, points, img, radar=None, img_metas=None):
        """Extract features from images and lidar points and radars."""

        # lidar feature distabled. (param points not used )

        radar = None  # don't use radar feature
        if radar is not None:
            radar_feats = self.radar_encoder(radar)
        else:
            radar_feats = None
        if self.fix_feats:
            with torch.no_grad():
                img_feats = self.extract_img_feat(img, img_metas)
        else:
            img_feats = self.extract_img_feat(img, img_metas)
        return (img_feats, radar_feats, None)

    def _targets_to_instances(
        self,
        gt_bboxes_3d=None,
        gt_labels_3d=None,
        instance_inds=None,
        img_shape=(
            1,
            1,
        ),
    ):
        gt_instances = Instances(tuple(img_shape))
        gt_instances.boxes = gt_bboxes_3d
        gt_instances.labels = gt_labels_3d
        gt_instances.obj_ids = instance_inds
        return gt_instances

    def _generate_empty_tracks(self):
        # create class for empty tracks in the first frame

        track_instances = Instances((1, 1))
        num_queries, dim = self.query_embedding.weight.shape  # (N, 256 * 2)
        device = self.query_embedding.weight.device
        query = self.query_embedding.weight    # N x 512

        # convert the query embedding to a point in 3D
        track_instances.ref_pts = self.reference_points(query[..., : dim // 2])  # N x 3

        # init boxes: xy, wl, z, h, sin, cos, vx, vy, vz
        box_sizes = self.bbox_size_fc(query[..., : dim // 2])   # N x 3
        pred_boxes_init = torch.zeros(
            (len(track_instances), 10), dtype=torch.float, device=device
        )     # N x 10
        pred_boxes_init[..., 2:4] = box_sizes[..., 0:2]
        pred_boxes_init[..., 5:6] = box_sizes[..., 2:3]

        # track instance class, add other properties
        track_instances.query = query
        track_instances.output_embedding = torch.zeros(
            (num_queries, dim >> 1), device=device
        )   # N x 256
        track_instances.obj_idxes = torch.full(
            (len(track_instances),), -1, dtype=torch.long, device=device
        )   # N x 1
        track_instances.matched_gt_idxes = torch.full(
            (len(track_instances),), -1, dtype=torch.long, device=device
        )   # N x 1
        track_instances.disappear_time = torch.zeros(
            (len(track_instances),), dtype=torch.long, device=device
        )   # N x 1
        track_instances.scores = torch.zeros(
            (len(track_instances),), dtype=torch.float, device=device
        )
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

    def _copy_tracks_for_loss(self, tgt_instances):

        device = self.query_embedding.weight.device
        track_instances = Instances((1, 1))

        track_instances.obj_idxes = deepcopy(tgt_instances.obj_idxes)
        track_instances.matched_gt_idxes = deepcopy(tgt_instances.matched_gt_idxes)
        track_instances.disappear_time = deepcopy(tgt_instances.disappear_time)

        track_instances.scores = torch.zeros(
            (len(track_instances),), dtype=torch.float, device=device
        )
        track_instances.track_scores = torch.zeros(
            (len(track_instances),), dtype=torch.float, device=device
        )
        track_instances.pred_boxes = torch.zeros(
            (len(track_instances), 10), dtype=torch.float, device=device
        )
        track_instances.pred_logits = torch.zeros(
            (len(track_instances), self.num_classes), dtype=torch.float, device=device
        )

        track_instances.save_period = deepcopy(tgt_instances.save_period)
        return track_instances.to(self.query_embedding.weight.device)

    @force_fp32(apply_to=("img", "points"))
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

    # @auto_fp16(apply_to=('img', 'radar'))
    def _forward_single(
        self,
        points,
        img,
        radar,
        img_metas,
        track_instances,
        l2g_r1=None,
        l2g_t1=None,
        l2g_r2=None,
        l2g_t2=None,
        time_delta=None,
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

        # retrieve feature in the frame
        B, num_cam, _, H, W = img.shape
        img_feats, radar_feats, pts_feats = self.extract_feat(
            points, img=img, radar=radar, img_metas=img_metas
        )
        # img_feats/pts_feats: 4 (stages) [1 x T x C x H x W]

        ref_box_sizes = torch.cat(
            [track_instances.pred_boxes[:, 2:4], track_instances.pred_boxes[:, 5:6]],
            dim=1,
        )

        # query for each reference points and get outputs
        # output_classes: L x B x N x 7
        # output_coords: L x B x N x 10
        # query_feats: B x N x embed_dim
        output_classes, output_coords, query_feats, last_ref_pts = self.pts_bbox_head(
            img_feats,
            radar_feats,
            track_instances.query,
            track_instances.ref_pts,
            ref_box_sizes,
            img_metas,
        )

        # initialize outputs from the last layer
        out = {
            "pred_logits": output_classes[-1],
            "pred_boxes": output_coords[-1],
            "ref_pts": last_ref_pts,
        }
        with torch.no_grad():
            track_scores = output_classes[-1, 0, :].sigmoid().max(dim=-1).values

        # Step-1 Update track instances with current prediction
        # the track id will be assigned by the matcher.
        # copy the track instances and initialize the loss for all decoder layers
        nb_dec = output_classes.size(0)
        track_instances_list = [
            self._copy_tracks_for_loss(track_instances) for i in range(nb_dec - 1)
        ]
        track_instances.output_embedding = query_feats[0]  # [300, feat_dim]
        
        # update the reference points based on the velocity, 
        # take it from the last layer of the outputs
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
        track_instances.ref_pts = ref_pts

        # add the last frame of track instance data
        track_instances_list.append(track_instances)

        # update results for all decoder layers
        for i in range(nb_dec):
            track_instances = track_instances_list[i]
            # track_scores = output_classes[i, 0, :].sigmoid().max(dim=-1).values
            track_instances.scores = track_scores
            track_instances.pred_logits = output_classes[i, 0]  # [300, num_cls]
            track_instances.pred_boxes = output_coords[i, 0]  # [300, box_dim]
            out["track_instances"] = track_instances

            # compute loss for this layer of results
            track_instances = self.criterion.match_for_single_frame(
                out, i, if_step=(i == (nb_dec - 1))
            )
            # print('\n\nlayer %d' % i)
            # print(track_instances.obj_idxes)  6893
            # print(track_instances.matched_gt_idxes)   3
            # print(track_instances.scores)
            # print(track_instances.pred_boxes)

        if self.memory_bank is not None:
            track_instances = self.memory_bank(track_instances)

        # Step-2 query interaction 
        tmp = {}
        tmp["init_track_instances"] = self._generate_empty_tracks()
        tmp["track_instances"] = track_instances
        out_track_instances = self.query_interact(tmp)
        out["track_instances"] = out_track_instances

        return out

    def forward_train(
        self,
        points=None,
        img=None,
        radar=None,
        img_metas=None,
        gt_bboxes_3d=None,
        gt_labels_3d=None,
        instance_inds=None,
        l2g_r_mat=None,
        l2g_t=None,
        gt_bboxes_ignore=None,
        timestamp=None,
    ):
        """Forward training function.
        This function will call _forward_single in a for loop

        Args:
            points (list(list[torch.Tensor]), optional): B-T-sample
                Points of each sample.
                Defaults to None.
            img (Torch.Tensor) of shape [B, T, num_cam, 3, H, W]
            radar (Torch.Tensor) of shape [B, T, num_points, radar_dim]
            img_metas (list[dict], optional): Meta information of each sample.
                Defaults to None.
            lidar2img = img_metas[bs]['lidar2img'] of shape [3, 6, 4, 4]. list
                of list of list of 4x4 array
            gt_bboxes_3d (list[list[:obj:`BaseInstance3DBoxes`]], optional):
                Ground truth 3D boxes. Defaults to None.
            gt_labels_3d (list[list[torch.Tensor]], optional): Ground truth labels
                of 3D boxes. Defaults to None.
            l2g_r_mat (list[Tensor]). element shape [T, 3, 3]
            l2g_t (list[Tensor]). element shape [T, 3]
                normally you should call points @ R_Mat.T + T
                here, just call points @ R_mat + T
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                2D boxes in images to be ignored. Defaults to None.
        Returns:
            dict: Losses of different branches.
        """

        # [T+1, 3, 3]
        l2g_r_mat = l2g_r_mat[0]
        # change to [T+1, 1, 3]
        l2g_t = l2g_t[0].unsqueeze(dim=1)

        # timestamp = timestamp
        # bs = img.size(0)   

        # img: 1 x T x N_cam x 3 x H x W
        num_frame = img.size(1) - 1
        track_instances = self._generate_empty_tracks()

        # init gt instances!
        gt_instances_list = []
        for i in range(num_frame):
            gt_instances = Instances((1, 1))
            boxes = gt_bboxes_3d[0][i].tensor.to(img.device)   # M x 9
            # nuScenes format: xyz, wlh, rot(radian), vx, vy in world coordinate

            # normalize gt bboxes
            boxes = normalize_bbox(boxes, self.pc_range)    # M x 10
            gt_instances.boxes = boxes
            gt_instances.labels = gt_labels_3d[0][i]        # (M, )
            gt_instances.obj_ids = instance_inds[0][i]      # (M, )
            gt_instances_list.append(gt_instances)

        # TODO init criterion
        self.criterion.initialize_for_single_clip(gt_instances_list)

        # for bs 1
        lidar2img = img_metas[0]["lidar2img"]  # [T, num_cam]
        for i in range(num_frame):
            points_single = [p_[i] for p_ in points]
            img_single = torch.stack([img_[i] for img_ in img], dim=0)
            radar_single = torch.stack([radar_[i] for radar_ in radar], dim=0)

            img_metas_single = deepcopy(img_metas)
            img_metas_single[0]["lidar2img"] = lidar2img[i]

            if i == num_frame - 1:
                l2g_r2 = None
                l2g_t2 = None
                time_delta = None
            else:
                l2g_r2 = l2g_r_mat[i + 1]
                l2g_t2 = l2g_t[i + 1]
                time_delta = timestamp[i + 1] - timestamp[i]

            frame_res = self._forward_single(
                points_single,
                img_single,
                radar_single,
                img_metas_single,
                track_instances,
                l2g_r_mat[i],
                l2g_t[i],
                l2g_r2,
                l2g_t2,
                time_delta,
            )
            track_instances = frame_res["track_instances"]

        outputs = self.criterion.losses_dict
        return outputs

    def _inference_single(
        self,
        points,
        img,
        radar,
        img_metas,
        track_instances,
        l2g_r1=None,
        l2g_t1=None,
        l2g_r2=None,
        l2g_t2=None,
        time_delta=None,
    ):
        """
        This function will be called at forward_test

        Warnning: Only Support BS=1
        img: shape [B, num_cam, 3, H, W]
        """

        # velo update:
        active_inst = track_instances[track_instances.obj_idxes >= 0]
        other_inst = track_instances[track_instances.obj_idxes < 0]

        if l2g_r2 is not None and len(active_inst) > 0 and l2g_r1 is not None:
            ref_pts = active_inst.ref_pts
            velo = active_inst.pred_boxes[:, -2:]
            ref_pts = self.velo_update(
                ref_pts, velo, l2g_r1, l2g_t1, l2g_r2, l2g_t2, time_delta=time_delta
            )
            active_inst.ref_pts = ref_pts
        track_instances = Instances.cat([other_inst, active_inst])

        B, num_cam, _, H, W = img.shape
        img_feats, radar_feats, pts_feats = self.extract_feat(
            points, img=img, radar=radar, img_metas=img_metas
        )
        img_feats = [a.clone() for a in img_feats]

        # output_classes: [num_dec, B, num_query, num_classes]
        # query_feats: [B, num_query, embed_dim]
        ref_box_sizes = torch.cat(
            [track_instances.pred_boxes[:, 2:4], track_instances.pred_boxes[:, 5:6]],
            dim=1,
        )

        output_classes, output_coords, query_feats, last_ref_pts = self.pts_bbox_head(
            img_feats,
            radar_feats,
            track_instances.query,
            track_instances.ref_pts,
            ref_box_sizes,
            img_metas,
        )

        out = {
            "pred_logits": output_classes[-1],
            "pred_boxes": output_coords[-1],
            "ref_pts": last_ref_pts,
        }

        # TODO: Why no max?
        track_scores = output_classes[-1, 0, :].sigmoid().max(dim=-1).values

        # Step-1 Update track instances with current prediction
        # [nb_dec, bs, num_query, xxx]

        # each track will be assigned an unique global id by the track base.
        track_instances.scores = track_scores
        # track_instances.track_scores = track_scores  # [300]
        track_instances.pred_logits = output_classes[-1, 0]  # [300, num_cls]
        track_instances.pred_boxes = output_coords[-1, 0]  # [300, box_dim]
        track_instances.output_embedding = query_feats[0]  # [300, feat_dim]

        track_instances.ref_pts = last_ref_pts[0]

        self.track_base.update(track_instances)

        if self.memory_bank is not None:
            track_instances = self.memory_bank(track_instances)

        # Step-2 Update track instances using matcher

        tmp = {}
        tmp["init_track_instances"] = self._generate_empty_tracks()
        tmp["track_instances"] = track_instances
        out_track_instances = self.query_interact(tmp)
        out["track_instances"] = out_track_instances
        return out

    def forward_test(
        self,
        points=None,
        img=None,
        radar=None,
        img_metas=None,
        timestamp=1e6,
        l2g_r_mat=None,
        l2g_t=None,
        **kwargs,
    ):
        """Forward test function.
        only support bs=1, single-gpu, num_frame=1 test
        Args:
            points (list(list[torch.Tensor]), optional): B-T-sample
                Points of each sample.
                Defaults to None.
            img (Torch.Tensor) of shape [B, T, num_cam, 3, H, W]
            img_metas (list[dict], optional): Meta information of each sample.
                Defaults to None.
            lidar2img = img_metas[bs]['lidar2img'] of shape [3, 6, 4, 4]. list
                of list of list of 4x4 array
            gt_bboxes_3d (list[list[:obj:`BaseInstance3DBoxes`]], optional):
                Ground truth 3D boxes. Defaults to None.
            gt_labels_3d (list[list[torch.Tensor]], optional): Ground truth labels
                of 3D boxes. Defaults to None.

            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                2D boxes in images to be ignored. Defaults to None.
        Returns:
            dict: Losses of different branches.
        """
        # [3, 3]
        l2g_r_mat = l2g_r_mat[0][0]
        # change to [1, 3]
        l2g_t = l2g_t[0].unsqueeze(dim=1)[0]

        bs = img.size(0)
        num_frame = img.size(1)

        timestamp = timestamp[0]
        if self.test_track_instances is None:
            track_instances = self._generate_empty_tracks()
            self.test_track_instances = track_instances
            self.timestamp = timestamp[0]
        # TODO: use scene tokens?
        if timestamp[0] - self.timestamp > 10:
            track_instances = self._generate_empty_tracks()
            time_delta = None
            l2g_r1 = None
            l2g_t1 = None
            l2g_r2 = None
            l2g_t2 = None
        else:
            track_instances = self.test_track_instances
            time_delta = timestamp[0] - self.timestamp
            l2g_r1 = self.l2g_r_mat
            l2g_t1 = self.l2g_t
            l2g_r2 = l2g_r_mat
            l2g_t2 = l2g_t
        self.timestamp = timestamp[-1]
        self.l2g_r_mat = l2g_r_mat
        self.l2g_t = l2g_t

        # for bs 1;
        lidar2img = img_metas[0]["lidar2img"]  # [T, num_cam]
        for i in range(num_frame):
            points_single = [p_[i] for p_ in points]
            img_single = torch.stack([img_[i] for img_ in img], dim=0)
            radar_single = torch.stack([radar_[i] for radar_ in radar], dim=0)

            img_metas_single = deepcopy(img_metas)
            img_metas_single[0]["lidar2img"] = lidar2img[i]

            frame_res = self._inference_single(
                points_single,
                img_single,
                radar_single,
                img_metas_single,
                track_instances,
                l2g_r1,
                l2g_t1,
                l2g_r2,
                l2g_t2,
                time_delta,
            )
            track_instances = frame_res["track_instances"]

        active_instances = self.query_interact._select_active_tracks(
            dict(track_instances=track_instances)
        )
        self.test_track_instances = track_instances

        results = self._active_instances2results(active_instances, img_metas)
        return results

    def _active_instances2results(self, active_instances, img_metas):
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
        active_idxes = active_instances.scores >= self.track_base.filter_score_thresh
        active_instances = active_instances[active_idxes]
        if active_instances.pred_logits.numel() == 0:
            return [None]
        bbox_dict = dict(
            cls_scores=active_instances.pred_logits,
            bbox_preds=active_instances.pred_boxes,
            track_scores=active_instances.scores,
            obj_idxes=active_instances.obj_idxes,
        )
        bboxes_dict = self.bbox_coder.decode(bbox_dict)[0]

        bboxes = bboxes_dict["bboxes"]
        bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 5] * 0.5
        bboxes = img_metas[0]["box_type_3d"][0](bboxes, 9)
        labels = bboxes_dict["labels"]
        scores = bboxes_dict["scores"]

        track_scores = bboxes_dict["track_scores"]
        obj_idxes = bboxes_dict["obj_idxes"]
        result_dict = dict(
            boxes_3d=bboxes.to("cpu"),
            scores_3d=scores.cpu(),
            labels_3d=labels.cpu(),
            track_scores=track_scores.cpu(),
            track_ids=obj_idxes.cpu(),
        )

        return [result_dict]
