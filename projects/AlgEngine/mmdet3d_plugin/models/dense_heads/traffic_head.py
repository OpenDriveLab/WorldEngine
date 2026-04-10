import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import Linear, bias_init_with_prob
from mmcv.runner import force_fp32
from torchvision.ops.focal_loss import sigmoid_focal_loss

from mmdet3d.core.bbox.coders import build_bbox_coder
from mmdet.core import multi_apply, reduce_mean
from mmdet.models import HEADS
from mmdet.models.dense_heads import DETRHead
from mmdet.models.utils.transformer import inverse_sigmoid
from mmdet3d_plugin.core.bbox.util import normalize_bbox


@HEADS.register_module()
class TrafficHead(DETRHead):
    """Head of Detr3D.
    Args:
        with_box_refine (bool): Whether to refine the reference points
            in the decoder. Defaults to False.
        as_two_stage (bool) : Whether to generate the proposal from
            the outputs of encoder.
        transformer (obj:`ConfigDict`): ConfigDict is used for building
            the Encoder and Decoder.
    """

    def __init__(
        self,
        *args,
        with_box_refine=False,
        as_two_stage=False,
        transformer=None,
        bbox_coder=None,
        num_cls_fcs=2,
        code_weights=None,
        cam_ids=[0, 1, 2, 3],
        traffic_keys=None,
        **kwargs,
    ):
        self.with_box_refine = with_box_refine
        self.as_two_stage = as_two_stage
        if self.as_two_stage:
            transformer["as_two_stage"] = self.as_two_stage
        if "code_size" in kwargs:
            self.code_size = kwargs["code_size"]
        else:
            self.code_size = 10
        if code_weights is not None:
            self.code_weights = code_weights
        else:
            self.code_weights = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2]

        self.bbox_coder = build_bbox_coder(bbox_coder)
        self.pc_range = self.bbox_coder.pc_range
        self.num_cls_fcs = num_cls_fcs - 1
        self.cam_ids = cam_ids
        self.traffic_keys = traffic_keys
        super(TrafficHead, self).__init__(*args, transformer=transformer, **kwargs)
        self.transformer = None

        self.count = 0

    def init_weights(self):
        pass

    def _init_layers(self):
        """Initialize classification branch and regression branch of head."""

        # traffic_light_branch = []
        # for _ in range(self.num_reg_fcs):
        #     traffic_light_branch.append(Linear(self.embed_dims, self.embed_dims))
        #     traffic_light_branch.append(nn.LayerNorm(self.embed_dims))
        #     traffic_light_branch.append(nn.ReLU(inplace=True))
        # traffic_light_branch.append(Linear(self.embed_dims, 1))
        # self.traffic_light_branch = nn.Sequential(*traffic_light_branch)

        # stop_sign_branch = []
        # for _ in range(self.num_reg_fcs):
        #     stop_sign_branch.append(Linear(self.embed_dims, self.embed_dims))
        #     stop_sign_branch.append(nn.LayerNorm(self.embed_dims))
        #     stop_sign_branch.append(nn.ReLU(inplace=True))
        # stop_sign_branch.append(Linear(self.embed_dims, 1))
        # self.stop_sign_branch = nn.Sequential(*stop_sign_branch)

        # working version
        # traffic_light_branch = []
        # # for _ in range(self.num_reg_fcs):
        # traffic_light_branch.append(Linear(8960, 256))
        # traffic_light_branch.append(nn.LayerNorm(256))
        # traffic_light_branch.append(nn.ReLU(inplace=True))
        # traffic_light_branch.append(Linear(256, 32))
        # traffic_light_branch.append(nn.LayerNorm(32))
        # traffic_light_branch.append(nn.ReLU(inplace=True))
        # traffic_light_branch.append(Linear(32, 1))
        # self.traffic_light_branch = nn.Sequential(*traffic_light_branch)

        # stop_sign_branch = []
        # # for _ in range(self.num_reg_fcs):
        # stop_sign_branch.append(Linear(8960, 256))
        # stop_sign_branch.append(nn.LayerNorm(256))
        # stop_sign_branch.append(nn.ReLU(inplace=True))
        # stop_sign_branch.append(Linear(256, 32))
        # stop_sign_branch.append(nn.LayerNorm(32))
        # stop_sign_branch.append(nn.ReLU(inplace=True))
        # stop_sign_branch.append(Linear(32, 1))
        # self.stop_sign_branch = nn.Sequential(*stop_sign_branch)

        ##### 3 branches -- shared backbone
        # shared_branch = []
        # shared_branch.append(Linear(8960, 1024))
        # shared_branch.append(nn.LayerNorm(1024))
        # shared_branch.append(nn.ReLU(inplace=True))
        # shared_branch.append(Linear(1024, 256))
        # shared_branch.append(nn.LayerNorm(256))
        # shared_branch.append(nn.ReLU(inplace=True))
        # self.shared_branch = nn.Sequential(*shared_branch)

        # light_inrange_branch = []
        # light_inrange_branch.append(Linear(256, 64))
        # light_inrange_branch.append(nn.LayerNorm(64))
        # light_inrange_branch.append(nn.ReLU(inplace=True))
        # light_inrange_branch.append(Linear(64, 16))
        # light_inrange_branch.append(nn.LayerNorm(16))
        # light_inrange_branch.append(nn.ReLU(inplace=True))
        # light_inrange_branch.append(Linear(16, 1))
        # self.light_inrange_branch = nn.Sequential(*light_inrange_branch)

        # light_hazard_branch = []
        # light_hazard_branch.append(Linear(256, 64))
        # light_hazard_branch.append(nn.LayerNorm(64))
        # light_hazard_branch.append(nn.ReLU(inplace=True))
        # light_hazard_branch.append(Linear(64, 16))
        # light_hazard_branch.append(nn.LayerNorm(16))
        # light_hazard_branch.append(nn.ReLU(inplace=True))
        # light_hazard_branch.append(Linear(16, 1))
        # self.light_hazard_branch = nn.Sequential(*light_hazard_branch)

        # sign_inrange_branch = []
        # sign_inrange_branch.append(Linear(256, 64))
        # sign_inrange_branch.append(nn.LayerNorm(64))
        # sign_inrange_branch.append(nn.ReLU(inplace=True))
        # sign_inrange_branch.append(Linear(64, 16))
        # sign_inrange_branch.append(nn.LayerNorm(16))
        # sign_inrange_branch.append(nn.ReLU(inplace=True))
        # sign_inrange_branch.append(Linear(16, 1))
        # self.sign_inrange_branch = nn.Sequential(*sign_inrange_branch)

        ##### 3 branches -- non-shared backbone
        light_inrange_branch = []
        light_inrange_branch.append(Linear(8960, 1024))
        light_inrange_branch.append(nn.LayerNorm(1024))
        light_inrange_branch.append(nn.ReLU(inplace=True))
        light_inrange_branch.append(Linear(1024, 128))
        light_inrange_branch.append(nn.LayerNorm(128))
        light_inrange_branch.append(nn.ReLU(inplace=True))
        light_inrange_branch.append(Linear(128, 16))
        light_inrange_branch.append(nn.LayerNorm(16))
        light_inrange_branch.append(nn.ReLU(inplace=True))
        light_inrange_branch.append(Linear(16, 1))
        self.light_inrange_branch = nn.Sequential(*light_inrange_branch)

        light_hazard_branch = []
        light_hazard_branch.append(Linear(8960, 1024))
        light_hazard_branch.append(nn.LayerNorm(1024))
        light_hazard_branch.append(nn.ReLU(inplace=True))
        light_hazard_branch.append(Linear(1024, 128))
        light_hazard_branch.append(nn.LayerNorm(128))
        light_hazard_branch.append(nn.ReLU(inplace=True))
        light_hazard_branch.append(Linear(128, 16))
        light_hazard_branch.append(nn.LayerNorm(16))
        light_hazard_branch.append(nn.ReLU(inplace=True))
        light_hazard_branch.append(Linear(16, 1))
        self.light_hazard_branch = nn.Sequential(*light_hazard_branch)

        sign_inrange_branch = []
        sign_inrange_branch.append(Linear(8960, 1024))
        sign_inrange_branch.append(nn.LayerNorm(1024))
        sign_inrange_branch.append(nn.ReLU(inplace=True))
        sign_inrange_branch.append(Linear(1024, 128))
        sign_inrange_branch.append(nn.LayerNorm(128))
        sign_inrange_branch.append(nn.ReLU(inplace=True))
        sign_inrange_branch.append(Linear(128, 16))
        sign_inrange_branch.append(nn.LayerNorm(16))
        sign_inrange_branch.append(nn.ReLU(inplace=True))
        sign_inrange_branch.append(Linear(16, 1))
        self.sign_inrange_branch = nn.Sequential(*sign_inrange_branch)

    def forward(self, mlvl_feats, mlvl_point_feats, img_metas):
        """Forward function.
        Args:
            mlvl_feats (tuple[Tensor]): Features from the upstream
                network, each is a 5D-tensor with shape
                (B, N, C, H, W).
        Returns:
            all_cls_scores (Tensor): Outputs from the classification head, \
                shape [nb_dec, bs, num_query, cls_out_channels]. Note \
                cls_out_channels should includes background.
            all_bbox_preds (Tensor): Sigmoid outputs from the regression \
                head with normalized coordinate format (cx, cy, w, l, cz, h, theta, vx, vy). \
                Shape [nb_dec, bs, num_query, 9].
        """

        global_feats = mlvl_feats[-1][
            :, self.cam_ids, :, :, :
        ]  # B x cams x 256 x 5 x 7

        bz, cams, dim, height, width = global_feats.size()
        global_feats = global_feats.view(
            bz, cams, dim * height * width
        )  # B x cams x 8960

        # compute share features for all heads
        shared_feats = torch.max(global_feats, dim=1)[0]  # B x 256

        # compute results for each head
        outs = {}
        for key in self.traffic_keys:

            if key == "light_inrange":
                result = self.light_inrange_branch(shared_feats)
            elif key == "light_hazard":
                result = self.light_hazard_branch(shared_feats)
            elif key == "sign_inrange":
                result = self.sign_inrange_branch(shared_feats)
            else:
                raise KeyError

            outs[key] = result

        return outs

    @force_fp32(apply_to=("preds_dicts"))
    def loss(
        self,
        gt_bboxes_list,
        gt_labels_list,
        additional_labels,
        preds_dicts,
        gt_bboxes_ignore=None,
    ):
        """ "Loss function.
        Args:

            gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
                with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels_list (list[Tensor]): Ground truth class indices for each
                image with shape (num_gts, ).
            preds_dicts:
                all_cls_scores (Tensor): Classification score of all
                    decoder layers, has shape
                    [nb_dec, bs, num_query, cls_out_channels].
                all_bbox_preds (Tensor): Sigmoid regression
                    outputs of all decode layers. Each is a 4D-tensor with
                    normalized coordinate format (cx, cy, w, h) and shape
                    [nb_dec, bs, num_query, 4].
                enc_cls_scores (Tensor): Classification scores of
                    points on encode feature map , has shape
                    (N, h*w, num_classes). Only be passed when as_two_stage is
                    True, otherwise is None.
                enc_bbox_preds (Tensor): Regression results of each points
                    on the encode feature map, has shape (N, h*w, 4). Only be
                    passed when as_two_stage is True, otherwise is None.
            gt_bboxes_ignore (list[Tensor], optional): Bounding boxes
                which can be ignored for each image. Default None.
        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        assert gt_bboxes_ignore is None, (
            f"{self.__class__.__name__} only supports "
            f"for gt_bboxes_ignore setting to None."
        )

        # self.count += 1

        loss_dict = dict()

        for key in self.traffic_keys:
            loss_dict["loss_" + key] = F.binary_cross_entropy_with_logits(
                preds_dicts[key], additional_labels[key], reduction="none"
            )

        return loss_dict

    @force_fp32(apply_to=("preds_dicts"))
    def get_bboxes(self, preds_dicts, img_metas, rescale=False):
        """Generate bboxes from bbox head predictions.
        Args:
            preds_dicts (tuple[list[dict]]): Prediction results.
            img_metas (list[dict]): Point cloud and image's meta info.
        Returns:
            list[dict]: Decoded bbox, scores and labels after nms.
        """
        preds_dicts = self.bbox_coder.decode(preds_dicts)
        num_samples = len(preds_dicts)
        ret_list = []
        for i in range(num_samples):
            preds = preds_dicts[i]
            bboxes = preds["bboxes"]
            bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 5] * 0.5
            bboxes = img_metas[i]["box_type_3d"](bboxes, 9)
            scores = preds["scores"]
            labels = preds["labels"]
            ret_list.append([bboxes, scores, labels])
        return ret_list
