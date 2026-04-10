# ---------------------------------------------------------------------------------#
# UniAD: Planning-oriented Autonomous Driving (https://arxiv.org/abs/2212.10156)  #
# Source code: https://github.com/OpenDriveLab/UniAD                              #
# Copyright (c) OpenDriveLab. All rights reserved.                                #
# ---------------------------------------------------------------------------------#

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.models.builder import HEADS, build_loss
from mmcv.runner import BaseModule
from einops import rearrange
from mmdet.core import reduce_mean
from mmcv.cnn.bricks.transformer import build_transformer_layer_sequence
import copy
from .occ_head_plugin import (
    MLP,
    BevFeatureSlicer,
    SimpleConv2d,
    CVT_Decoder,
    Bottleneck,
    UpsamplingAdd,
    predict_instance_segmentation_and_trajectories,
)


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


@HEADS.register_module()
class OccHead(BaseModule):
    def __init__(
        self,
        # General
        receptive_field=3,
        n_future=4,
        spatial_extent=(50, 50),
        ignore_index=255,
        # BEV
        grid_conf=None,
        bev_size=(200, 200),
        bev_emb_dim=256,
        bev_proj_dim=64,
        bev_proj_nlayers=1,
        # Query
        query_dim=256,
        query_mlp_layers=3,
        detach_query_pos=True,
        temporal_mlp_layer=2,
        # Transformer
        transformer_decoder=None,
        attn_mask_thresh=0.5,
        # Loss
        sample_ignore_mode="all_valid",
        aux_loss_weight=1.0,
        loss_mask=None,
        loss_dice=None,
        # Cfgs
        init_cfg=None,
        # Eval
        pan_eval=False,
        test_seg_thresh: float = 0.5,
        test_with_track_score=False,
        # use input query features
        use_track_query=True,
        use_traj_query=True,
    ):
        assert init_cfg is None, (
            "To prevent abnormal initialization "
            "behavior, init_cfg is not allowed to be set"
        )
        super().__init__(init_cfg)
        self.receptive_field = (
            receptive_field  # NOTE: Used by prepare_future_labels in E2EPredTransformer
        )
        self.n_future = n_future
        self.spatial_extent = spatial_extent
        self.ignore_index = ignore_index

        bevformer_bev_conf = {
            "xbound": [-51.2, 51.2, 0.512],
            "ybound": [-51.2, 51.2, 0.512],
            "zbound": [-10.0, 10.0, 20.0],
        }
        self.bev_sampler = BevFeatureSlicer(bevformer_bev_conf, grid_conf)

        self.bev_size = bev_size
        self.bev_proj_dim = bev_proj_dim

        if bev_proj_nlayers == 0:
            self.bev_light_proj = nn.Sequential()
        else:
            self.bev_light_proj = SimpleConv2d(
                in_channels=bev_emb_dim,
                conv_channels=bev_emb_dim,
                out_channels=bev_proj_dim,
                num_conv=bev_proj_nlayers,
            )

        # Downscale bev_feat -> /4
        self.base_downscale = nn.Sequential(
            Bottleneck(in_channels=bev_proj_dim, downsample=True),
            Bottleneck(in_channels=bev_proj_dim, downsample=True),
        )

        # Future blocks with transformer
        self.n_future_blocks = self.n_future + 1

        # - transformer
        self.attn_mask_thresh = attn_mask_thresh

        self.num_trans_layers = transformer_decoder.num_layers
        assert self.num_trans_layers % self.n_future_blocks == 0

        self.num_heads = transformer_decoder.transformerlayers.attn_cfgs.num_heads
        self.transformer_decoder = build_transformer_layer_sequence(transformer_decoder)

        # - temporal-mlps
        # query_out_dim = bev_proj_dim
        self.query_dim = query_dim
        temporal_mlp = MLP(
            query_dim, query_dim, bev_proj_dim, num_layers=temporal_mlp_layer
        )
        self.temporal_mlps = _get_clones(temporal_mlp, self.n_future_blocks)

        # - downscale-convs
        downscale_conv = Bottleneck(in_channels=bev_proj_dim, downsample=True)
        self.downscale_convs = _get_clones(downscale_conv, self.n_future_blocks)

        # - upsampleAdds
        upsample_add = UpsamplingAdd(
            in_channels=bev_proj_dim, out_channels=bev_proj_dim
        )
        self.upsample_adds = _get_clones(upsample_add, self.n_future_blocks)

        # Decoder
        self.dense_decoder = CVT_Decoder(
            dim=bev_proj_dim,
            blocks=[bev_proj_dim, bev_proj_dim],
        )

        # Query
        self.logging = False
        self.use_track_query = use_track_query
        self.use_traj_query = use_traj_query
        if self.use_traj_query:
            self.mode_fuser = nn.Sequential(
                nn.Linear(query_dim, bev_proj_dim),
                nn.LayerNorm(bev_proj_dim),
                nn.ReLU(inplace=True),
            )
        fuser_dim = 0
        fuser_dim = fuser_dim + 1 if use_traj_query else fuser_dim        
        fuser_dim = fuser_dim + 2 if use_track_query else fuser_dim                
        mid_dim = max(1, fuser_dim - 1)
        # none prior query is used
        if fuser_dim == 0:
            num_query = 100 # set the max query to 100
            self.query_embed_occ = nn.Embedding(num_query, query_dim)
            fuser_dim = 1   # set the dim to only 256
        self.multi_query_fuser = nn.Sequential(
            nn.Linear(query_dim * fuser_dim, query_dim * mid_dim),
            nn.LayerNorm(query_dim * mid_dim),
            nn.ReLU(inplace=True),
            nn.Linear(query_dim * mid_dim, bev_proj_dim),
        )

        self.detach_query_pos = detach_query_pos

        self.query_to_occ_feat = MLP(
            query_dim, query_dim, bev_proj_dim, num_layers=query_mlp_layers
        )
        self.temporal_mlp_for_mask = copy.deepcopy(self.query_to_occ_feat)

        # Loss
        # For matching
        self.sample_ignore_mode = sample_ignore_mode
        assert self.sample_ignore_mode in ["all_valid", "past_valid", "none"]

        self.aux_loss_weight = aux_loss_weight

        self.loss_dice = build_loss(loss_dice)
        self.loss_mask = build_loss(loss_mask)

        self.pan_eval = pan_eval
        self.test_seg_thresh = test_seg_thresh

        self.test_with_track_score = test_with_track_score
        self.init_weights()

    def init_weights(self):
        for p in self.transformer_decoder.parameters():
            if p.dim() > 1:
                nn.init.xavier_normal_(p)

    def get_attn_mask(self, state, ins_query):
        # state: B, C, H, W
        # ins_query: B, A, C
        
        # find correlation of the two high-d vector, the more similar, the more close to 1
        # check which pixel is occupied by which agent by their feature correlation
        # print(ins_query.size())
        ins_embed = self.temporal_mlp_for_mask(ins_query)   # B x A x C
        # print(ins_embed.size())
        # print(state.size())
        mask_pred = torch.einsum("bqc,bchw->bqhw", ins_embed, state)    # B x A x H x W
        # print(mask_pred.size())
        attn_mask = mask_pred.sigmoid() < self.attn_mask_thresh
        attn_mask = (
            rearrange(attn_mask, "b q h w -> b (h w) q")
            .unsqueeze(1)
            .repeat(1, self.num_heads, 1, 1)
            .flatten(0, 1)
        )
        attn_mask = attn_mask.detach()      # 8 x HW x 5

        # if a mask is all True(all background), then set it all False.
        attn_mask[torch.where(attn_mask.sum(-1) == attn_mask.shape[-1])] = False

        upsampled_mask_pred = F.interpolate(
            mask_pred, self.bev_size, mode="bilinear", align_corners=False
        )  # Supervised by gt, B x A x H x W

        return attn_mask, upsampled_mask_pred, ins_embed

    def forward(self, x, ins_query):

        base_state = rearrange(x, "(h w) b d -> b d h w", h=self.bev_size[0])   # 1 x 256 x H x W
        base_state = self.bev_sampler(base_state)       # 1 x 256 x H x W
        base_state = self.bev_light_proj(base_state)    # 1 x 256 x H x W
        base_state = self.base_downscale(base_state)    # 1 x 256 x 50 x 50
        base_ins_query = ins_query
        # print(ins_query.size())

        last_state = base_state
        last_ins_query = base_ins_query
        future_states = []
        mask_preds = []
        # temporal_query = []
        temporal_embed_for_mask_attn = []
        n_trans_layer_each_block = self.num_trans_layers // self.n_future_blocks
        assert n_trans_layer_each_block >= 1

        for i in range(self.n_future_blocks):
            # Downscale
            cur_state = self.downscale_convs[i](last_state)  # /4 -> /8

            # Attention
            # temporal_aware ins_query
            cur_ins_query = self.temporal_mlps[i](last_ins_query)  # [b, q, d]
            # temporal_query.append(cur_ins_query)
            # print(cur_ins_query.size())

            # Generate attn mask
            # cur_state: 1 x 256 x 25 x 25
            # cur_ins_query: B x A x 256
            attn_mask, mask_pred, cur_ins_emb_for_mask_attn = self.get_attn_mask(
                cur_state, cur_ins_query
            )
            # outputs
            # attn_mask: 8 x 625 x 4
            # mask_pred: 1 x 4 x H x W
            # cur_ins_emb_for_mask_attn: B x A x 256
            # print(cur_ins_emb_for_mask_attn.size())

            attn_masks = [None, attn_mask]

            mask_preds.append(mask_pred)  # /1
            temporal_embed_for_mask_attn.append(cur_ins_emb_for_mask_attn)

            cur_state = rearrange(cur_state, "b c h w -> (h w) b c")
            cur_ins_query = rearrange(cur_ins_query, "b q c -> q b c")

            # dense-sparse interaction between dense feature and sparse agents
            for j in range(n_trans_layer_each_block):
                trans_layer_ind = i * n_trans_layer_each_block + j
                trans_layer = self.transformer_decoder.layers[trans_layer_ind]
                cur_state = trans_layer(
                    query=cur_state,  # [h'*w', b, c]
                    key=cur_ins_query,  # [nq, b, c]
                    value=cur_ins_query,  # [nq, b, c]
                    query_pos=None,
                    key_pos=None,
                    attn_masks=attn_masks,
                    query_key_padding_mask=None,
                    key_padding_mask=None,
                )  # out size: [h'*w', b, c]

            cur_state = rearrange(
                cur_state, "(h w) b c -> b c h w", h=self.bev_size[0] // 8
            )

            # Upscale to /4
            cur_state = self.upsample_adds[i](cur_state, last_state)

            # Out
            future_states.append(cur_state)  # [b, d, h/4, w/4]
            last_state = cur_state

        future_states = torch.stack(future_states, dim=1)  # [b, t, d, h/4, w/4]
        # temporal_query = torch.stack(temporal_query, dim=1)  # [b, t, q, d]
        mask_preds = torch.stack(mask_preds, dim=2)  # B x A x T x H x W

        # updated query features after occupancy prediction
        ins_query = torch.stack(temporal_embed_for_mask_attn, dim=1)    # B x T x A x 256

        # Decode future states to larger resolution
        future_states = self.dense_decoder(future_states)   # B x T x 256 x H x W
        ins_occ_query = self.query_to_occ_feat(ins_query)   # B x T x A x 256

        # print(ins_occ_query.size())
        # print(future_states.size())

        # Generate final outputs
        ins_occ_logits = torch.einsum("btqc,btchw->bqthw", ins_occ_query, future_states)    
        # B x A x T x H x W

        return mask_preds, ins_occ_logits, ins_query

    def merge_queries(self, outs_dict, detach_query_pos=True):
        # merge input query from upstream

        query_pool = []

        # using traj queries
        # take the last layer of traj qeury and take the most of the temporal dimension
        if self.use_traj_query:
            traj_query = outs_dict.get("traj_query", None)  # [3, 1, A, n_modes, dim]        
            traj_query = traj_query[-1]   # B x A x 6 x C
            traj_query = self.mode_fuser(traj_query).max(2)[0]    # B x A x 256
            query_pool.append(traj_query)
            if self.logging:
                print("\nusing traj query")            
        else:
            if self.logging:
                print("NOT using traj query")

        # using track queries
        if self.use_track_query:
            track_query = outs_dict["track_query"]  # [b, nq, d]
            track_query_pos = outs_dict["track_query_pos"]  # [b, nq, d]
            if detach_query_pos:
                track_query_pos = track_query_pos.detach()
            query_pool.append(track_query)
            query_pool.append(track_query_pos)
            if self.logging:
                print("\nusing track query")            
        else:
            if self.logging:
                print("NOT using track query")

        # fuse track query, track position, and traj query together
        if len(query_pool) > 0:
            ins_query = torch.cat(query_pool, dim=-1)       # B x A x 256
        else:
            ins_query = self.query_embed_occ.weight         # A x 256
            ins_query = ins_query.expand(1, -1, self.query_dim)     # B x A x 256
        # print(ins_query.size())

        ins_query = self.multi_query_fuser(ins_query)   # B x A x 256

        # original code where we always just to merge three queries
        # ins_query = self.multi_query_fuser(
        #     torch.cat([traj_query, track_query, track_query_pos], dim=-1)
        # )   # B x A x 256

        return ins_query

    # With matched queries [a small part of all queries] and matched_gt results
    def forward_train(
        self,
        bev_feat,
        outs_dict,
        gt_inds_list=None,
        gt_segmentation=None,
        gt_instance=None,
        gt_img_is_valid=None,
        img_metas=None,
    ):
        # Generate warpped gt and related inputs
        gt_segmentation, gt_instance, gt_img_is_valid = self.get_occ_labels(
            gt_segmentation, gt_instance, gt_img_is_valid
        )

        all_matched_gt_ids = outs_dict["all_matched_idxes"]  # list of tensor, length bs

        ins_query = self.merge_queries(outs_dict, self.detach_query_pos)

        # Forward the occ-flow model
        mask_preds_batch, ins_seg_preds_batch, ins_query_updated = self(bev_feat, ins_query=ins_query)
        # both B x A x T x H x W
        # ins_query_updated: B x T x A x 256

        # Get pred and gt
        ins_seg_targets_batch = (
            gt_instance  # [1, 5, 200, 200] [b, t, h, w] # ins targets of a batch
        )

        # img_valid flag, for filtering out invalid samples in sequence when calculating loss
        img_is_valid = gt_img_is_valid  # [1, 7]
        assert (
            img_is_valid.size(1) == self.receptive_field + self.n_future
        ), f"Img_is_valid can only be 7 as for loss calculation and evaluation!!! Don't change it"
        frame_valid_mask = img_is_valid.bool()
        past_valid_mask = frame_valid_mask[:, : self.receptive_field]
        future_frame_mask = frame_valid_mask[
            :, (self.receptive_field - 1) :
        ]  # [1, 5]  including current frame

        # only supervise when all 3 past frames are valid
        past_valid = past_valid_mask.all(dim=1)
        future_frame_mask[~past_valid] = False

        # Calculate loss in the batch
        loss_dict = dict()
        loss_dice = ins_seg_preds_batch.new_zeros(1)[0].float()
        loss_mask = ins_seg_preds_batch.new_zeros(1)[0].float()
        loss_aux_dice = ins_seg_preds_batch.new_zeros(1)[0].float()
        loss_aux_mask = ins_seg_preds_batch.new_zeros(1)[0].float()

        bs = ins_query.size(0)
        assert bs == 1
        pred_occ_dict = {}
        for ind in range(bs):
            # Each gt_bboxes contains 3 frames, we only use the last one
            cur_gt_inds = gt_inds_list[ind][-1]

            cur_matched_gt = all_matched_gt_ids[ind]  # [n_gt]

            # Re-order gt according to matched_gt_inds
            cur_gt_inds = cur_gt_inds[cur_matched_gt]

            # Deal matched_gt: -1, its actually background(unmatched)
            cur_gt_inds[cur_matched_gt == -1] = -1  # Bugfixed
            cur_gt_inds[cur_matched_gt == -2] = -2

            frame_mask = future_frame_mask[ind]  # [t]

            # Prediction
            ins_seg_preds = ins_seg_preds_batch[ind]  # [q(n_gt for matched), t, h, w]
            ins_seg_targets = ins_seg_targets_batch[ind]  # [t, h, w]
            mask_preds = mask_preds_batch[ind]

            # Assigned-gt, extract GT for each instance given the IDs
            ins_seg_targets_ordered = []
            for ins_id in cur_gt_inds:
                # -1 for unmatched query
                # If ins_seg_targets is all 255, ignore (directly append occ-and-flow gt to list)
                # 255 for special object --> change to -20 (same as in occ_label.py)
                # -2 for no_query situation
                if (ins_seg_targets == self.ignore_index).all().item() is True:
                    ins_tgt = ins_seg_targets.long()
                elif ins_id.item() in [-1, -2]:  # false positive query (unmatched)
                    ins_tgt = (
                        torch.ones_like(ins_seg_targets).long() * self.ignore_index
                    )
                else:
                    SPECIAL_INDEX = -20
                    if ins_id.item() == self.ignore_index:
                        ins_id = torch.ones_like(ins_id) * SPECIAL_INDEX
                    ins_tgt = (ins_seg_targets == ins_id).long()  # [t, h, w], 0 or 1

                ins_seg_targets_ordered.append(ins_tgt)

            ins_seg_targets_ordered = torch.stack(
                ins_seg_targets_ordered, dim=0
            )  # [n_gt, t, h, w]

            # Sanity check
            t, h, w = ins_seg_preds.shape[-3:]
            assert t == 1 + self.n_future, f"{ins_seg_preds.size()}"
            assert (
                ins_seg_preds.size() == ins_seg_targets_ordered.size()
            ), f"{ins_seg_preds.size()}, {ins_seg_targets_ordered.size()}"

            num_total_pos = ins_seg_preds.size(0)  # Check this line

            # loss for a sample in batch
            num_total_pos = ins_seg_preds.new_tensor([num_total_pos])
            num_total_pos = torch.clamp(reduce_mean(num_total_pos), min=1).item()

            cur_dice_loss = self.loss_dice(
                ins_seg_preds,
                ins_seg_targets_ordered,
                avg_factor=num_total_pos,
                frame_mask=frame_mask,
            )

            cur_mask_loss = self.loss_mask(
                ins_seg_preds, ins_seg_targets_ordered, frame_mask=frame_mask
            )

            cur_aux_dice_loss = self.loss_dice(
                mask_preds,
                ins_seg_targets_ordered,
                avg_factor=num_total_pos,
                frame_mask=frame_mask,
            )
            cur_aux_mask_loss = self.loss_mask(
                mask_preds, ins_seg_targets_ordered, frame_mask=frame_mask
            )

            loss_dice += cur_dice_loss
            loss_mask += cur_mask_loss
            loss_aux_dice += cur_aux_dice_loss * self.aux_loss_weight
            loss_aux_mask += cur_aux_mask_loss * self.aux_loss_weight

            ######## convert instance-based to segmentation-based, maximum can be 2.0 rather than 1.0
            ins_seg_preds_prob = ins_seg_preds.sigmoid()    # A x T x H x W
            ins_seg_preds_semantic = torch.sum(ins_seg_preds_prob, dim=0)   # T x H x W
            mask_preds_prob = mask_preds.sigmoid()  # A x T x H x W 
            mask_preds_semantic = torch.sum(mask_preds_prob, dim=0)     # T x H x W
            # ins_seg_gt_semantic = torch.sum(ins_seg_targets_ordered, dim=0)    # T x H x W, only for matched GT instances
            ins_seg_gt_semantic = gt_segmentation[0, :, 0]     # T x H x W, more comprehensive

            pred_occ_dict['ins_seg_preds_semantic'] = ins_seg_preds_semantic
            pred_occ_dict['mask_preds_semantic'] = mask_preds_semantic
            # pred_occ_dict['ins_seg_gt_semantic'] = ins_seg_targets_semantic
            pred_occ_dict['ins_query_occ'] = ins_query_updated      # B x T x A x 256
            pred_occ_dict['ins_seg_gt_semantic'] = ins_seg_gt_semantic

        loss_dict["loss_dice"] = loss_dice / bs
        loss_dict["loss_mask"] = loss_mask / bs
        loss_dict["loss_aux_dice"] = loss_aux_dice / bs
        loss_dict["loss_aux_mask"] = loss_aux_mask / bs

        return loss_dict, pred_occ_dict

    def forward_test(
        self,
        bev_feat,
        outs_dict,
        no_query=False,
        gt_segmentation=None,
        gt_instance=None,
        gt_img_is_valid=None,
        img_metas=None,        
    ):
        gt_segmentation, gt_instance, gt_img_is_valid = self.get_occ_labels(
            gt_segmentation, gt_instance, gt_img_is_valid
        )

        out_dict = dict()
        out_dict["seg_gt"] = gt_segmentation[
            :, : 1 + self.n_future
        ]  # [1, 5, 1, 200, 200]
        out_dict["ins_seg_gt"] = self.get_ins_seg_gt(
            gt_instance[:, : 1 + self.n_future]
        )  # [1, 5, 200, 200]
        out_dict['ins_seg_gt_semantic'] = gt_segmentation[0, :, 0]    # T x H x W, binary 0 or 1

        if no_query:
            # output all zero results
            out_dict["seg_out"] = torch.zeros_like(
                out_dict["seg_gt"]
            ).long()  # [1, 5, 1, 200, 200]
            out_dict["ins_seg_out"] = torch.zeros_like(
                out_dict["ins_seg_gt"]
            ).long()  # [1, 5, 200, 200]

            # dummy outputs for the downstream module
            out_dict["ins_seg_preds_semantic"] = torch.zeros_like(
                out_dict["ins_seg_gt"]
            )[0].float()  # T x H x W
            out_dict["mask_preds_semantic"] = torch.zeros_like(
                out_dict["ins_seg_gt"]
            )[0].float()  # T x H x W    
            out_dict["ins_query_occ"] = None         

            return out_dict

        # forward
        ins_query = self.merge_queries(outs_dict, self.detach_query_pos)
        mask_preds_batch, pred_ins_logits, ins_query_updated = self(bev_feat, ins_query=ins_query)
        out_dict["pred_ins_logits"] = pred_ins_logits
        # mask_preds_batch: B x A x T x H x W
        # pred_ins_logits: B x A x T x H x W
        # ins_query_updated: B x T x A x C

        pred_ins_logits = pred_ins_logits[:, :, : 1 + self.n_future]  # [b, q, t, h, w]
        pred_ins_sigmoid = pred_ins_logits.sigmoid()  # [b, q, t, h, w]

        # generate outputs for downstream
        out_dict['ins_seg_preds_semantic'] = torch.sum(pred_ins_sigmoid[0], dim=0)   # T x H x W, iedal 0-1 sometimes 0-2
        mask_preds_prob = mask_preds_batch[0].sigmoid()  # A x T x H x W 
        out_dict['mask_preds_semantic'] = torch.sum(mask_preds_prob, dim=0)     # T x H x W, iedal 0-1 sometimes 0-2
        out_dict['ins_query_occ'] = ins_query_updated   # B x T x A x C

        if self.test_with_track_score:
            track_scores = outs_dict["track_scores"].to(pred_ins_sigmoid)  # [b, q]
            track_scores = track_scores[:, :, None, None, None]
            pred_ins_sigmoid = pred_ins_sigmoid * track_scores  # [b, q, t, h, w]

        out_dict["pred_ins_sigmoid"] = pred_ins_sigmoid
        pred_seg_scores = pred_ins_sigmoid.max(1)[0]
        seg_out = (
            (pred_seg_scores > self.test_seg_thresh).long().unsqueeze(2)
        )  # [b, t, 1, h, w]
        out_dict["seg_out"] = seg_out
        if self.pan_eval:
            # ins_pred
            pred_consistent_instance_seg = (
                predict_instance_segmentation_and_trajectories(
                    seg_out, pred_ins_sigmoid
                )
            )  # bg is 0, fg starts with 1, consecutive

            out_dict["ins_seg_out"] = pred_consistent_instance_seg  # [1, 5, 200, 200]

        return out_dict

    def get_ins_seg_gt(self, gt_instance):
        ins_gt_old = (
            gt_instance  # Not consecutive, 0 for bg, otherwise ins_ind(start from 1)
        )
        ins_gt_new = torch.zeros_like(ins_gt_old).to(ins_gt_old)  # Make it consecutive
        ins_inds_unique = torch.unique(ins_gt_old)
        new_id = 1
        for uni_id in ins_inds_unique:
            if uni_id.item() in [0, self.ignore_index]:  # ignore background_id
                continue
            ins_gt_new[ins_gt_old == uni_id] = new_id
            new_id += 1
        return ins_gt_new  # Consecutive

    def get_occ_labels(self, gt_segmentation, gt_instance, gt_img_is_valid):
        if not self.training:
            try:
                gt_segmentation = gt_segmentation[0]
                gt_instance = gt_instance[0]
                gt_img_is_valid = gt_img_is_valid[0]
            except TypeError:
                print('Invalid input, Occ head')
                gt_segmentation = torch.zeros(1, 7, 200, 200)
                gt_instance = torch.zeros(1, 7, 200, 200)
                gt_img_is_valid = torch.zeros(1, 9)
            
        gt_segmentation = gt_segmentation[:, : self.n_future + 1].long().unsqueeze(2)
        gt_instance = gt_instance[:, : self.n_future + 1].long()
        gt_img_is_valid = gt_img_is_valid[:, : self.receptive_field + self.n_future]
        return gt_segmentation, gt_instance, gt_img_is_valid
