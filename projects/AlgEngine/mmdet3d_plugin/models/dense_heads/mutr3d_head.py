import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import Linear, bias_init_with_prob, build_activation_layer, constant_init
from mmcv.cnn.bricks.transformer import build_positional_encoding
from mmcv.runner import auto_fp16, force_fp32
from torch.nn.init import normal_

from mmdet.models import HEADS
from mmdet.models.utils import build_transformer
from mmdet.models.utils.transformer import inverse_sigmoid


@HEADS.register_module()
class DeformableMUTRTrackingHead(nn.Module):
    """Head of DeformDETR3DCamTrack.

    Args:
        transformer (obj:`ConfigDict`): ConfigDict is used for building
            the Encoder and Decoder.
    """

    def __init__(
        self,
        num_classes,
        in_channels,
        num_reg_fcs=2,
        num_cams=6,
        num_feature_levels=4,
        transformer=None,
        pc_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
        positional_encoding=dict(
            type="SinePositionalEncoding", num_feats=128, normalize=True, offset=-0.5
        ),
        num_cls_fcs=2,
        test_cfg=dict(max_per_img=100),
        init_cfg=None,
        **kwargs
    ):
        """
        we decode bbox as (cx, cy, w.log(), l.log(), cz, h.log(), rot.sin(), rot.cos(), vx, vy)
            output space for wlh is in log space
            output space for xyz in inverse sigmoid space
            rotation: output unnormalized sine and cosine
        code weights: weights the bbox L1 loss
        """
        super(DeformableMUTRTrackingHead, self).__init__()

        if "code_size" in kwargs:
            self.code_size = kwargs["code_size"]
        else:
            self.code_size = [6, 2, 2]

        self.pc_range = pc_range

        self.test_cfg = test_cfg
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.cls_out_channels = num_classes
        self.num_cls_fcs = num_cls_fcs - 1
        self.num_reg_fcs = num_reg_fcs

        self.act_cfg = transformer.get("act_cfg", dict(type="ReLU", inplace=True))
        self.activate = build_activation_layer(self.act_cfg)
        self.positional_encoding = build_positional_encoding(positional_encoding)
        self.transformer = build_transformer(transformer)
        self.embed_dims = self.transformer.embed_dims
        self.num_feature_levels = num_feature_levels
        self.num_cams = num_cams
        self._init_layers()

    def _init_layers(self):
        """Initialize classification branch and regression branch of head."""
        cls_branch = []
        for _ in range(self.num_reg_fcs):
            cls_branch.append(Linear(self.embed_dims, self.embed_dims))
            cls_branch.append(nn.LayerNorm(self.embed_dims))
            cls_branch.append(nn.ReLU(inplace=True))
        cls_branch.append(Linear(self.embed_dims, self.cls_out_channels))
        fc_cls = nn.Sequential(*cls_branch)

        # typically, xyz(inverse sigmoid space), wlh(gt_w.log())
        reg_branch = []
        for _ in range(self.num_reg_fcs):
            reg_branch.append(Linear(self.embed_dims, self.embed_dims))
            # reg_branch.append(nn.LayerNorm(self.embed_dims))
            reg_branch.append(nn.ReLU())
        reg_branch.append(Linear(self.embed_dims, self.code_size[0]))
        reg_branch = nn.Sequential(*reg_branch)

        direction_branch = []
        for _ in range(self.num_reg_fcs):
            direction_branch.append(Linear(self.embed_dims, self.embed_dims))
            # reg_branch.append(nn.LayerNorm(self.embed_dims))
            direction_branch.append(nn.ReLU())
        direction_branch.append(Linear(self.embed_dims, self.code_size[1]))
        direction_branch = nn.Sequential(*direction_branch)

        # branch for velocity prediction
        velo_branch = []
        for _ in range(self.num_reg_fcs):
            velo_branch.append(Linear(self.embed_dims, self.embed_dims))
            # reg_branch.append(nn.LayerNorm(self.embed_dims))
            velo_branch.append(nn.ReLU())
        velo_branch.append(Linear(self.embed_dims, self.code_size[2]))
        velo_branch = nn.Sequential(*velo_branch)

        def _get_clones(module, N):
            return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

        num_pred = self.transformer.decoder.num_layers

        self.cls_branches = _get_clones(fc_cls, num_pred)
        self.reg_branches = _get_clones(reg_branch, num_pred)
        self.direction_branches = _get_clones(direction_branch, num_pred)
        self.velo_branches = _get_clones(velo_branch, num_pred)

        self.level_embeds = nn.Parameter(
            torch.Tensor(self.num_feature_levels, self.embed_dims)
        )
        self.cam_embeds = nn.Parameter(torch.Tensor(self.num_cams, self.embed_dims))

    def init_weights(self):
        """Initialize weights of the DeformDETR head."""
        self.transformer.init_weights()
        normal_(self.level_embeds)
        normal_(self.cam_embeds)

        bias_init = bias_init_with_prob(0.01)
        for m in self.cls_branches:
            nn.init.constant_(m[-1].bias, bias_init)

    # @auto_fp16(apply_to=('img', 'radar'))
    def forward(
        self, mlvl_feats, radar_feats, query_embeds, ref_points, ref_size, img_metas
    ):
        """Forward function.
        Args:
            mlvl_feats (tuple[Tensor]): List of Features from the upstream
                network, each is a 5D-tensor with shape
                (B, N, C, H, W).
            query_embeds (Tensor):  pos_embed and feature for querys of shape
                (num_query, embed_dim*2)
            ref_points (Tensor):  3d reference points associated with each query
                shape (num_query, 3)
                value is in inevrse sigmoid space
            ref_size (Tensor): the size(bbox size) associated with each query
                shape (num_query, 3)
                value in log space. 
        Returns:
            all_cls_scores (Tensor): Outputs from the classification head, \
                shape [nb_dec, bs, num_query, cls_out_channels]. Note \
                cls_out_channels should includes background.
            all_bbox_preds (Tensor): Sigmoid outputs from the regression \
                head with normalized coordinate format (cx, cy, w, l, cz, h, sine, cosine, vx, vy). \
                Shape [nb_dec, bs, num_query, 10].
            last_query_feats (Tensor): shape [bs, num_query, feat_dim]
            last_ref_points (Tensor): shape [bs, num_query, 3]
        """
        # TODO: add postional encoding here to multilevel feats
        batch_size = mlvl_feats[0].size(0)
        input_img_h, input_img_w = img_metas[0]["input_shape"]
        img_masks = mlvl_feats[0].new_ones((batch_size, input_img_h, input_img_w))
        for img_id in range(batch_size):
            img_h, img_w, _ = img_metas[img_id]["img_shape"][0][0]
            img_masks[img_id, :img_h, :img_w] = 0

        # add positional embedding into multi-level image features
        for i, feat in enumerate(mlvl_feats):
            B, N, C, H, W = feat.size()
            mlvl_masks = (
                F.interpolate(img_masks[None], size=feat.shape[-2:])
                .to(torch.bool)
                .squeeze(0)
            )
            # [B, embed_dim, H, W]
            pos_enc = self.positional_encoding(mlvl_masks)
            pos_enc = pos_enc.unsqueeze(dim=1).repeat(1, N, 1, 1, 1)
            # [B, N, embed_dim, H, W]
            lvl_enc = self.level_embeds[i].view(1, 1, -1, 1, 1)
            cam_enc = self.cam_embeds.view(1, N, C, 1, 1)

            pos_enc = pos_enc + lvl_enc + cam_enc
            mlvl_feats[i] = feat + pos_enc

        # Maybe Encoder Here

        # Decoder
        # hs: features: (num_dec_layers, num_query, bs, embed_dims)
        # init_reference: (1, bs, num_query, 3)
        # inter_references: (num_dec_layers-1, bs, num_query, 3)
        hs, inter_references, inter_box_sizes = self.transformer(
            mlvl_feats,
            query_embeds,
            ref_points,
            ref_size,
            reg_branches=self.reg_branches,
            img_metas=img_metas,
            radar_feats=radar_feats,
        ) 
        # change to: (num_dec, bs, num_query, embed_dim)
        hs = hs.permute(0, 2, 1, 3)    # L x B x N x C
        
        # loop through each level of the features
        outputs_classes = []
        outputs_coords = []
        for lvl in range(hs.shape[0]):
            if lvl == 0:
                reference = ref_points.sigmoid()
                ref_size_base = ref_size
            else:
                reference = inter_references[lvl - 1]
                ref_size_base = inter_box_sizes[lvl - 1]
            reference = inverse_sigmoid(reference)
            outputs_class = self.cls_branches[lvl](hs[lvl])
            xywlzh = self.reg_branches[lvl](hs[lvl])
            direction_pred = self.direction_branches[lvl](hs[lvl])
            velo_pred = self.velo_branches[lvl](hs[lvl])

            # TODO: check the shape of reference
            assert reference.shape[-1] == 3
            xywlzh[..., 0:2] += reference[..., 0:2]
            xywlzh[..., 0:2] = xywlzh[..., 0:2].sigmoid()
            xywlzh[..., 4:5] += reference[..., 2:3]
            xywlzh[..., 4:5] = xywlzh[..., 4:5].sigmoid()
            last_ref_points = torch.cat(
                [xywlzh[..., 0:2], xywlzh[..., 4:5]],
                dim=-1,
            )
            xywlzh[..., 0:1] = (
                xywlzh[..., 0:1] * (self.pc_range[3] - self.pc_range[0])
                + self.pc_range[0]
            )
            xywlzh[..., 1:2] = (
                xywlzh[..., 1:2] * (self.pc_range[4] - self.pc_range[1])
                + self.pc_range[1]
            )
            xywlzh[..., 4:5] = (
                xywlzh[..., 4:5] * (self.pc_range[5] - self.pc_range[2])
                + self.pc_range[2]
            )

            # TODO: Chose: Add in log. Add in exp.
            # add in exp
            # xywlzh[..., 2:4] = (xywlzh[..., 2:4].exp() + ref_size_base[..., 0:2].exp()).log()
            # xywlzh[..., 5:6] = (xywlzh[..., 5:6].exp() + ref_size_base[..., 2:3].exp()).log()
            # add in log
            xywlzh[..., 2:4] = xywlzh[..., 2:4] + ref_size_base[..., 0:2]
            xywlzh[..., 5:6] = xywlzh[..., 5:6] + ref_size_base[..., 2:3]

            bbox_pred = torch.cat([xywlzh, direction_pred, velo_pred], dim=2)
            outputs_classes.append(outputs_class)
            outputs_coords.append(bbox_pred)

        outputs_classes = torch.stack(outputs_classes)      # L x B x N x 7
        outputs_coords = torch.stack(outputs_coords)        # L x B x N x 10
        # change to inverse sigmoid space
        last_ref_points = inverse_sigmoid(last_ref_points)
        last_query_feats = hs[-1]

        return outputs_classes, outputs_coords, last_query_feats, last_ref_points
