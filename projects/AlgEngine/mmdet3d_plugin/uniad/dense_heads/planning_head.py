# ---------------------------------------------------------------------------------#
# UniAD: Planning-oriented Autonomous Driving (https://arxiv.org/abs/2212.10156)  #
# Source code: https://github.com/OpenDriveLab/UniAD                              #
# Copyright (c) OpenDriveLab. All rights reserved.                                #
# ---------------------------------------------------------------------------------#

import torch
import torch.nn as nn
from typing import Dict
from mmdet.models.builder import HEADS, build_loss
from mmcv.cnn.bricks.transformer import FFN, build_positional_encoding
from einops import rearrange
from mmdet3d_plugin.models.utils.functional import (
    bivariate_gaussian_activation,
)
from .planning_head_plugin import CollisionNonlinearOptimizer, SaveOutput, patch_attention
import numpy as np
import copy
import json


@HEADS.register_module()
class PlanningHeadSingleMode(nn.Module):
    def __init__(
        self,
        bev_h=200,
        bev_w=200,
        embed_dims=256,
        planning_steps=6,
        loss_planning=None,
        loss_collision=None,
        planning_eval=False,
        use_col_optim=False,
        col_optim_args=dict(
            occ_filter_range=5.0,
            sigma=1.0,
            alpha_collision=5.0,
        ),
        with_adapter=False,
        bev_map_interaction=0,
        bev_map_gt=False,
        use_lane_query=False,
        use_map_query=False,
        bev_map_option=1,
        bev_occ_gt=False,
        bev_occ_interaction=0,  # # of channels used for bev occuapncy interaction
        use_occ_query=False,
        use_bev=True,  # attend to the BEV feature
        use_sdc_track_query=True,  #
        use_sdc_traj_query=True,
        use_sdc_navi_embed=True,
        use_sdc_plan_query_extra=False,
        add_bev_pos=False,
        use_ego_his=False,
        use_can_bus=False,
        num_commands=9,
        num_layers=3,
    ):
        """
        Single Mode Planning Head for Autonomous Driving.

        Args:
            embed_dims (int): Embedding dimensions. Default: 256.
            planning_steps (int): Number of steps for motion planning. Default: 6.
            loss_planning (dict): Configuration for planning loss. Default: None.
            loss_collision (dict): Configuration for collision loss. Default: None.
            planning_eval (bool): Whether to use planning for evaluation. Default: False.
            use_col_optim (bool): Whether to use collision optimization. Default: False.
            col_optim_args (dict): Collision optimization arguments. Default: dict(occ_filter_range=5.0, sigma=1.0, alpha_collision=5.0).
        """
        super(PlanningHeadSingleMode, self).__init__()

        # Nuscenes
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.embed_dims = embed_dims
        self.logging = False
        self.planning_steps = planning_steps
        self.planning_eval = planning_eval

        # parameters for interaction with mapping results explicitly
        self.bev_map_interaction = bev_map_interaction
        self.bev_map_gt = bev_map_gt
        self.use_lane_query = use_lane_query
        self.bev_map_option = float(bev_map_option)
        self.use_map_query = use_map_query
        self.add_bev_pos = add_bev_pos

        # parameters for interaction with occupancy results explicitly
        self.bev_occ_interaction = bev_occ_interaction
        self.bev_occ_gt = bev_occ_gt
        self.use_occ_query = use_occ_query

        # navigation
        self.navi_embed = nn.Embedding(num_commands, embed_dims)

        # ego additional information
        self.use_ego_his = use_ego_his
        self.use_can_bus = use_can_bus
        ego_additional_dim = 0
        ego_additional_dim = (
            ego_additional_dim + 18 if self.use_can_bus else ego_additional_dim
        )
        ego_additional_dim = (
            ego_additional_dim + 24 if self.use_ego_his else ego_additional_dim
        )
        if ego_additional_dim > 0:
            self.mlp_ego = nn.Sequential(
                nn.Linear(ego_additional_dim, embed_dims),
                nn.LayerNorm(embed_dims),
                nn.ReLU(inplace=True),
            )

        # add dimension for regression head
        added_dim = (
            bev_map_interaction if self.bev_map_option == 2 else 0
        )  # mapping for planning
        added_dim = (
            added_dim + self.bev_occ_interaction
            if self.bev_occ_interaction > 0
            else added_dim
        )  # occupancy for planning

        dim_multiple = 1
        dim_multiple = (
            dim_multiple + 1
            if self.use_lane_query or self.use_map_query
            else dim_multiple
        )
        dim_multiple = dim_multiple + 1 if self.use_occ_query else dim_multiple
        dim_multiple = dim_multiple + 1 if ego_additional_dim > 0 else dim_multiple
        self.reg_branch = nn.Sequential(
            nn.Linear(embed_dims * dim_multiple + added_dim, embed_dims),
            nn.ReLU(),
            nn.Linear(embed_dims, planning_steps * 2),
        )

        ############# parallel training

        # attention with BEV feature
        self.use_bev = use_bev
        if use_bev:
            attn_module_layer = nn.TransformerDecoderLayer(
                embed_dims,
                8,
                dim_feedforward=embed_dims * 2,
                dropout=0.1,
                batch_first=False,
            )
            self.attn_module = nn.TransformerDecoder(attn_module_layer, num_layers)
            
        # add a learnable query for the ego vehicle in planning
        self.use_sdc_plan_query_extra = use_sdc_plan_query_extra
        if self.use_sdc_plan_query_extra:
            self.sdc_query_embed_plan = nn.Embedding(1, embed_dims)

        # sdc query
        self.pos_embed = nn.Embedding(1, embed_dims)
        self.use_sdc_track_query = use_sdc_track_query
        self.use_sdc_traj_query = use_sdc_traj_query
        self.use_sdc_navi_embed = use_sdc_navi_embed
        fuser_dim = 0
        fuser_dim = fuser_dim + 1 if use_sdc_track_query else fuser_dim
        fuser_dim = fuser_dim + 1 if use_sdc_traj_query else fuser_dim
        fuser_dim = fuser_dim + 1 if use_sdc_navi_embed else fuser_dim
        fuser_dim = fuser_dim + 1 if use_sdc_plan_query_extra else fuser_dim
        if fuser_dim > 0:
            self.mlp_fuser = nn.Sequential(
                nn.Linear(embed_dims * fuser_dim, embed_dims),
                nn.LayerNorm(embed_dims),
                nn.ReLU(inplace=True),
            )

        # loss function
        self.loss_planning = build_loss(loss_planning)
        self.loss_collision = []
        for cfg in loss_collision:
            self.loss_collision.append(build_loss(cfg))
        self.loss_collision = nn.ModuleList(self.loss_collision)

        # test time optimization
        self.use_col_optim = use_col_optim
        self.occ_filter_range = col_optim_args["occ_filter_range"]
        self.sigma = col_optim_args["sigma"]
        self.alpha_collision = col_optim_args["alpha_collision"]

        # TODO: reimplement it with down-scaled feature_map
        self.with_adapter = with_adapter
        if with_adapter:
            bev_adapter_block = nn.Sequential(
                nn.Conv2d(embed_dims, embed_dims // 2, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(embed_dims // 2, embed_dims, kernel_size=1),
            )
            N_Blocks = 3
            bev_adapter = [copy.deepcopy(bev_adapter_block) for _ in range(N_Blocks)]
            self.bev_adapter = nn.Sequential(*bev_adapter)

        # interact with the mapping BEV output
        if self.bev_map_interaction is not None and self.bev_map_interaction > 0:

            if self.add_bev_pos:
                positional_encoding = dict(
                    type="SinePositionalEncoding",
                    num_feats=2,  # final embedding dim / 2
                    normalize=True,
                )
                self.mapbev_positional_encoding = build_positional_encoding(
                    positional_encoding
                )

            # compress the BEV+Map features
            if self.bev_map_option == 1:
                self.bev_map_fusion = nn.Sequential(
                    nn.Linear(embed_dims + self.bev_map_interaction, embed_dims),
                )
            # compress the latent query features
            elif self.bev_map_option == 2:
                self.mlp_plan_query_map = nn.Sequential(
                    nn.Linear(embed_dims, self.bev_map_interaction),
                )
                self.pos_embed_map = nn.Embedding(1, self.bev_map_interaction)

                n_dims = self.bev_map_interaction
                nhead = self.bev_map_interaction
                attn_module_layer_map = nn.TransformerDecoderLayer(
                    n_dims,
                    nhead,
                    dim_feedforward=n_dims * 2,
                    dropout=0.1,
                    batch_first=False,
                )
                self.attn_module_map_bev = nn.TransformerDecoder(
                    attn_module_layer_map, 3
                )

                # # accommadate for the old name
                # self.attn_module_map = nn.TransformerDecoder(
                #     attn_module_layer_map, 3
                # )

        # interact with the mapping latent query feature
        if self.use_lane_query or self.use_map_query:
            attn_module_layer_map = nn.TransformerDecoderLayer(
                embed_dims,
                8,
                dim_feedforward=embed_dims * 2,
                dropout=0.1,
                batch_first=False,
            )
            self.attn_module_map_query = nn.TransformerDecoder(attn_module_layer_map, 3)

            # old name
            # self.lane_interaction_layer = nn.TransformerDecoder(attn_module_layer_map, 3)

        # interact with the occupancy BEV output
        if self.bev_occ_interaction is not None and self.bev_occ_interaction > 0:

            if self.add_bev_pos:
                positional_encoding = dict(
                    type="SinePositionalEncoding",
                    num_feats=4,  # final embedding dim / 2
                    normalize=True,
                )
                self.occbev_positional_encoding = build_positional_encoding(
                    positional_encoding
                )

            self.mlp_plan_query_occ = nn.Sequential(
                nn.Linear(embed_dims, self.bev_occ_interaction),
            )
            self.pos_embed_occ = nn.Embedding(1, self.bev_occ_interaction)

            n_dims = self.bev_occ_interaction
            nhead = self.bev_occ_interaction
            attn_module_layer_occ = nn.TransformerDecoderLayer(
                n_dims,
                nhead,
                dim_feedforward=n_dims * 2,
                dropout=0.1,
                batch_first=False,
            )
            self.attn_module_occ_bev = nn.TransformerDecoder(attn_module_layer_occ, 3)

            # # accomondate for the old name
            # self.attn_module_occ = nn.TransformerDecoder(attn_module_layer_occ, 3)

        # interact with the occupancy latent query feature
        if self.use_occ_query:
            occ_temporal_steps = 5
            self.mlp_occ_query_temporal = nn.Sequential(
                nn.Linear(occ_temporal_steps, 1),
            )
            attn_module_layer_occ = nn.TransformerDecoderLayer(
                embed_dims,
                8,
                dim_feedforward=embed_dims * 2,
                dropout=0.1,
                batch_first=False,
            )
            self.attn_module_occ_query = nn.TransformerDecoder(attn_module_layer_occ, 3)

    def get_additional_input_from_upstream(
        self,
        outs_seg=None,  # mapping results
        outs_occ=None,  # occupancy prediction results
    ):

        ############### mapping
        # extract mapping BEV outputs
        if (
            outs_seg is not None
            and self.bev_map_interaction is not None
            and self.bev_map_interaction > 0
        ):

            # use GT segmentation
            if self.bev_map_gt:
                map_bev = outs_seg["mask_semantic_flat_gt"]  # HW x 1 x 4
                if self.logging:
                    print("\nget mapping BEV GT in planning")

            # extract mapping results
            else:
                map_bev = outs_seg["mask_semantic_flat"]  # HW x 1 x 4
                if self.logging:
                    print("\nget mapping BEV outputs in planning")

        # Not using mapping BEV
        else:
            map_bev = None
            if self.logging:
                print("NOT get mapping BEV in planning")

        # extract lane latent query feature
        if outs_seg is not None and self.use_lane_query:
            (
                _,
                _,
                _,
                lane_query,  # 1 x 300 x 256
                _,
                lane_query_pos,  # 1 x 300 x 256
                _,
            ) = outs_seg["args_tuple"]
            if self.logging:
                print("\nget latent lane query in planning")
        else:
            lane_query = None
            lane_query_pos = None
            if self.logging:
                print("NOT get latent lane query in planning")

        # extract lane latent query feature
        if outs_seg is not None and self.use_map_query:
            map_query = outs_seg["mask_query_panoptic"]  # 1 x (M+1) x 256
            if self.logging:
                print("\nget latent map query in planning")
        else:
            map_query = None
            if self.logging:
                print("NOT get latent map query in planning")

        ############### occuapncy prediction
        # extract occuapncy BEV outputs
        if (
            outs_occ is not None
            and self.bev_occ_interaction is not None
            and self.bev_occ_interaction > 0
        ):

            # use GT occupancy
            if self.bev_occ_gt:
                occ_bev = outs_occ["ins_seg_gt_semantic"]  # HW x 1 x 4
                if self.logging:
                    print("\nget occupancy BEV GT in planning")

            # extract occupancy results
            else:
                occ_bev = outs_occ["ins_seg_preds_semantic"]  # HW x 1 x 4
                if self.logging:
                    print("\nget occupancy BEV outputs in planning")

        # Not using occupancy BEV
        else:
            occ_bev = None
            if self.logging:
                print("NOT get occupancy BEV in planning")

        # extract occupancy latent query feature
        if outs_occ is not None and self.use_occ_query:
            occ_query = outs_occ["ins_query_occ"]  # B x T x A x 256
        else:
            occ_query = None

        # construct the result dictionary
        return {
            "map_bev": map_bev,
            "map_query": map_query,
            "lane_query": lane_query,
            "lane_query_pos": lane_query_pos,
            "occ_bev": occ_bev,
            "occ_query": occ_query,
        }

    def forward_train(
        self,
        bev_embed,
        outs_motion={},
        sdc_planning=None,
        sdc_planning_mask=None,
        command=None,
        gt_future_boxes=None,
        img_metas=None,
        outs_seg=None,  # mapping results
        outs_occ=None,  # occupancy prediction results
        sdc_planning_past=None,  # 1 x 4 x 4
        sdc_planning_mask_past=None,  # 1 x 4 x 2
    ):
        """
        Perform forward planning training with the given inputs.
        Args:
            bev_embed (torch.Tensor): The input bird's eye view feature map.
            outs_motion (dict): A dictionary containing the motion outputs.
            outs_occflow (dict): A dictionary containing the occupancy flow outputs.
            sdc_planning (torch.Tensor, optional): The self-driving car's planned trajectory.
            sdc_planning_mask (torch.Tensor, optional): The mask for the self-driving car's planning.
            command (torch.Tensor, optional): The driving command issued to the self-driving car.
            gt_future_boxes (torch.Tensor, optional): The ground truth future bounding boxes.
            img_metas (list[dict], optional): A list of metadata information about the input images.

        Returns:
            ret_dict (dict): A dictionary containing the losses and planning outputs.
        """

        # no need if we only attend to the BEV features
        sdc_traj_query = outs_motion["sdc_traj_query"] if 'sdc_traj_query' in outs_motion else None
        sdc_track_query = outs_motion["sdc_track_query"] if 'sdc_track_query' in outs_motion else None
        bev_pos = outs_motion["bev_pos"]
        occ_mask = None

        input_upstream: Dict = self.get_additional_input_from_upstream(
            outs_seg=outs_seg,
            outs_occ=outs_occ,
        )

        outs_planning = self(
            bev_embed=bev_embed,
            occ_mask=occ_mask,
            bev_pos=bev_pos,
            sdc_traj_query=sdc_traj_query,
            sdc_track_query=sdc_track_query,
            command=command,
            img_metas=img_metas,
            input_upstream=input_upstream,
            sdc_planning_past=sdc_planning_past,
            sdc_planning_mask_past=sdc_planning_mask_past,
        )
        loss_inputs = [sdc_planning, sdc_planning_mask, outs_planning, gt_future_boxes]
        losses = self.loss(*loss_inputs)
        ret_dict = dict(losses=losses, outs_motion=outs_planning)
        return ret_dict

    def forward_test(
        self,
        bev_embed,
        outs_motion={},
        command=None,
        img_metas=None,
        outs_seg=None,
        outs_occ=None,  # occupancy prediction results
        sdc_planning_past=None,
        sdc_planning_mask_past=None,
    ):
        
        # no need if we only attend to the BEV features
        sdc_traj_query = outs_motion["sdc_traj_query"] if 'sdc_traj_query' in outs_motion else None
        sdc_track_query = outs_motion["sdc_track_query"] if 'sdc_track_query' in outs_motion else None
        bev_pos = outs_motion["bev_pos"]

        input_upstream: Dict = self.get_additional_input_from_upstream(
            outs_seg=outs_seg,
            outs_occ=outs_occ,
        )

        if self.use_col_optim:
            occ_mask = outs_occ["seg_out"]
        else:
            occ_mask = None
        outs_planning = self(
            bev_embed=bev_embed,
            occ_mask=occ_mask,
            bev_pos=bev_pos,
            sdc_traj_query=sdc_traj_query,
            sdc_track_query=sdc_track_query,
            command=command,
            img_metas=img_metas,
            input_upstream=input_upstream,
            sdc_planning_past=sdc_planning_past,
            sdc_planning_mask_past=sdc_planning_mask_past,
        )
        return outs_planning

    def forward(
        self,
        bev_embed,
        occ_mask,
        bev_pos,
        sdc_traj_query,
        sdc_track_query,
        command,
        img_metas,
        input_upstream,
        sdc_planning_past=None,  # 1 x 4 x 4
        sdc_planning_mask_past=None,  # 1 x 4 x 2
    ):
        """
        Forward pass for PlanningHeadSingleMode.

        Args:
            bev_embed (torch.Tensor): Bird's eye view feature embedding. HW x 1 x C
            occ_mask (torch.Tensor): Instance mask for occupancy. 1 x T x 1 x H x W
            bev_pos (torch.Tensor): BEV position. 1 x C x H x W
            sdc_traj_query (torch.Tensor): SDC trajectory query. 3 x 1 x F x C
            sdc_track_query (torch.Tensor): SDC track query.    1 x C
            command (int): Driving command.

        Returns:
            dict: A dictionary containing SDC trajectory and all SDC trajectories.
        """

        # reset the frame id for this sequence
        self.scene_token: str = img_metas[0]["scene_token"]
        self.sample_token: str = img_metas[0]["sample_idx"]
        self.frame_idx: int = img_metas[0]["frame_idx"]
        dtype = bev_embed.dtype
        n_future = 6  # number of future frames to predict for ego vehicle trajectory
        can_bus: torch.Tensor = torch.from_numpy(img_metas[0]["can_bus"]).to(
            device=bev_embed.device, dtype=dtype
        )  # (18, )

        ############## obtain sdc query feature

        sdc_query_pool = []

        # learnable sdc query from motion head output, with interaction in prediction
        if self.use_sdc_traj_query:
            sdc_traj_query = sdc_traj_query[
                -1
            ]  # 1 x F x C, only take the last layer of the traj query
            sdc_query_pool.append(sdc_traj_query)
            if self.logging:
                print("\nusing sdc traj query")
        else:
            if self.logging:
                print("NOT using sdc traj query")

        # learnable sdc query from track_head output, with interaction in tracking
        if self.use_sdc_track_query:
            sdc_track_query = sdc_track_query.detach()  # 1 x C
            sdc_track_query = sdc_track_query[:, None].expand(
                -1, n_future, -1
            )  # 1 x F x C
            sdc_query_pool.append(sdc_track_query)
            if self.logging:
                print("\nusing sdc track query")
        else:
            if self.logging:
                print("NOT using sdc track query")

        # choose which weight to use for navigation
        if self.use_sdc_navi_embed:
            if command == None:
                print('Invalid input, planning head')
                command = [0]

            navi_embed = self.navi_embed.weight[command]  # 1 x C
            navi_embed = navi_embed[None].expand(-1, n_future, -1)  # 1 x F x C
            sdc_query_pool.append(navi_embed)
            if self.logging:
                print("\nusing sdc navi embedding")
        else:
            if self.logging:
                print("NOT using sdc navi embedding")

        # use the learnable embedding in planning
        if self.use_sdc_plan_query_extra:
            sdc_query_embed = self.sdc_query_embed_plan.weight  # 1 x C
            sdc_query_embed = sdc_query_embed[None].expand(
                -1, n_future, -1
            )  # 1 x F x C
            sdc_query_pool.append(sdc_query_embed)
            if self.logging:
                print("\nusing fresh sdc query in planning")
        else:
            if self.logging:
                print("NOT using fresh sdc query in planning")

        # fusing sdc query from commands, track and prediction
        if len(sdc_query_pool) > 0:
            plan_query = torch.cat(sdc_query_pool, dim=-1)
            plan_query = self.mlp_fuser(plan_query).max(1, keepdim=True)[
                0
            ]  # expand, then fuse  # [1, 6, 768] -> [1, 1, 256]
            plan_query = rearrange(plan_query, "b p c -> p b c")

        # use planning query
        else:
            NotImplementedError

        # add positional embedding to the plan query
        # new models all use this
        pos_embed = self.pos_embed.weight
        plan_query = plan_query + pos_embed[None]  # [1, 1, 256]

        ############## interact with BEV output representation from upstream modules
        if self.bev_map_interaction is not None and self.bev_map_interaction > 0:

            # create positional embedding for map bev
            if self.add_bev_pos:
                bev_mask = torch.zeros(
                    (1, self.bev_h, self.bev_w), device=plan_query.device
                ).to(
                    dtype
                )  # 1 x H x W
                mapbev_pos = self.mapbev_positional_encoding(bev_mask).to(
                    dtype
                )  # 1 x 4 x H x W
                mapbev_pos = rearrange(mapbev_pos, "b c h w -> (h w) b c")  # HW x 1 x 4
                if self.logging:
                    print("\nusing bev positional embeding in mapping for planning")

            #### option 1, compress the BEV feature with BEV outputs together, deprecated
            # only take the dimension requested
            # 1 -> lane divider
            # 2 -> lane divider + pedestrian crossing
            # 3 -> lane divider + pedestrian crossing + road boundary
            # 4 -> lane divider + pedestrian crossing + road boundary + drivable area
            if self.bev_map_option == 1:
                mask_semantic_used = input_upstream["map_bev"][
                    :, :, : self.bev_map_interaction
                ]  # HW x 1 x 4
                bev_embed = torch.cat(
                    [bev_embed, mask_semantic_used], dim=2
                )  # HW x 1 x (D+?)
                bev_embed = self.bev_map_fusion(bev_embed)  # HW x 1 x (D)
                if self.logging:
                    print(
                        "\nusing bev mapping channel %d for planning, option 1"
                        % self.bev_map_interaction
                    )
            #### option 2, compress the plan query
            # TODO: add the key pos to the BEV map
            # interact with mapping BEV outputs
            elif self.bev_map_option == 2:

                # get BEV
                mask_semantic_used = input_upstream["map_bev"][
                    :, :, : self.bev_map_interaction
                ]  # HW x 1 x 4
                if self.add_bev_pos:
                    map_feat = mask_semantic_used + mapbev_pos  # HW x 1 x 4
                else:
                    map_feat = mask_semantic_used

                # get plan query
                pos_embed_map = self.pos_embed_map.weight  # 1 x 1 x 4
                plan_query_map_bev = self.mlp_plan_query_map(plan_query)  # 1 x 1 x 4
                plan_query_map_bev = (
                    plan_query_map_bev + pos_embed_map[None]
                )  # [1, 1, 4]

                # attention
                plan_query_map_bev = self.attn_module_map_bev(
                    plan_query_map_bev, map_feat
                )  # [1, 1, 4]

                # logging
                if self.logging:
                    print(
                        "\nusing bev mapping channel %d for planning, option 2"
                        % self.bev_map_interaction
                    )
            else:
                if self.logging:
                    print("NOT using mapping for planning")

        # interact with occupancy BEV outputs
        if self.bev_occ_interaction is not None and self.bev_occ_interaction > 0:

            # get BEV
            occ_bev = input_upstream["occ_bev"].view(
                5, 1, -1
            )  # T x H x W -> T x 1 x HW
            occ_bev = torch.transpose(occ_bev, 0, 2).to(dtype)  # HW x 1 x T
            # occ_bev = occ_bev.float()  # convert to float32

            # create positional embedding for occ bev
            if self.add_bev_pos:
                bev_mask = torch.zeros(
                    (1, self.bev_h, self.bev_w), device=plan_query.device
                ).to(
                    dtype
                )  # 1 x H x W
                occbev_pos = self.occbev_positional_encoding(bev_mask).to(
                    dtype
                )  # 1 x 8 x H x W
                occbev_pos = rearrange(occbev_pos, "b c h w -> (h w) b c")  # HW x 1 x 8
                if self.logging:
                    print("\nusing bev positional embeding in occupancy for planning")

                # only take the first 5 dimension needed for merging
                occ_feat = occ_bev + occbev_pos[:, :, :5]
            else:
                occ_feat = occ_bev

            # get plan query
            pos_embed_occ = self.pos_embed_occ.weight  # 1 x 1 x 5
            plan_query_occ_bev = self.mlp_plan_query_occ(plan_query)  # 1 x 1 x 5
            plan_query_occ_bev = plan_query_occ_bev + pos_embed_occ[None]  # [1, 1, 5]

            # attention
            plan_query_occ_bev = self.attn_module_occ_bev(
                plan_query_occ_bev, occ_feat
            )  # [1, 1, 5]

            # logging
            if self.logging:
                print(
                    "\nusing bev occupancy channel %d for planning"
                    % self.bev_occ_interaction
                )

        # # add positional embedding to the plan query
        # # TODO: shift to before the option 2, old model uses this
        # pos_embed = self.pos_embed.weight
        # plan_query = plan_query + pos_embed[None]  # [1, 1, 256]

        ############## interact with latent query feature from upstream modules

        # interaction with the lane latent query feature
        if self.use_lane_query:
            lane_query_merged = (
                input_upstream["lane_query"] + input_upstream["lane_query_pos"]
            )  # B x L x C
            lane_query_merged = lane_query_merged.view(
                -1, 1, self.embed_dims
            )  # L x B x C
            plan_query_map_latent = self.attn_module_map_query(
                plan_query,  # 1 x B x C
                lane_query_merged,  # L x B x C
            )  # 1 x B x C

        # interaction with the mapping latent query feature
        if self.use_map_query:  # B x M x C
            lane_query_merged = input_upstream["map_query"].view(
                -1, 1, self.embed_dims
            )  # M x B x C
            plan_query_map_latent = self.attn_module_map_query(
                plan_query,  # 1 x B x C
                lane_query_merged,  # M x B x C
            )  # 1 x B x C

        # interaction with the occupancy latent query feature
        if self.use_occ_query:
            if input_upstream["occ_query"] is not None:
                # occ_query: B x T x A x C
                occ_query = torch.transpose(
                    input_upstream["occ_query"], 1, 3
                )  # B x C x A x T
                occ_query_compressed = self.mlp_occ_query_temporal(
                    occ_query
                )  # B x C x A x 1
                occ_query_compressed = torch.transpose(
                    occ_query_compressed, 1, 2
                )  # B x A x C x 1
                occ_query_compressed = occ_query_compressed.view(
                    -1, 1, self.embed_dims
                )  # A x B x C
                plan_query_occ_latent = self.attn_module_occ_query(
                    plan_query,  # 1 x B x C
                    occ_query_compressed,  # A x B x C
                )  # 1 x B x C
            else:
                plan_query_occ_latent = torch.zeros_like(plan_query)

        ############## interact with BEV features

        # reshape the bev positional embedding
        if self.use_bev:
            bev_pos = rearrange(bev_pos, "b c h w -> (h w) b c")
            bev_feat = bev_embed + bev_pos  # HW x 1 x C

            # Plugin adapter
            if self.with_adapter:
                bev_feat = rearrange(
                    bev_feat, "(h w) b c -> b c h w", h=self.bev_h, w=self.bev_w
                )
                bev_feat = bev_feat + self.bev_adapter(bev_feat)  # residual connection
                bev_feat = rearrange(bev_feat, "b c h w -> (h w) b c")

            plan_query = self.attn_module(plan_query, bev_feat)  # [1, B, 256]

            if self.logging:
                print("\nusing bev feature in planning")
        else:
            if self.logging:
                print("NOT attending to bev feature in planning")

        ############## fusing plan queries from multiple sources
        if (
            self.bev_map_interaction is not None
            and self.bev_map_interaction > 0
            and self.bev_map_option == 2
        ):
            plan_query = torch.cat(
                [plan_query, plan_query_map_bev], dim=-1
            )  # 1 x 1 x 260
            if self.logging:
                print("\nusing BEV mapping to interact with planning")

        # fusing plan queries with occupancy prediction interaction
        if self.bev_occ_interaction is not None and self.bev_occ_interaction > 0:
            plan_query = torch.cat(
                [plan_query, plan_query_occ_bev], dim=-1
            )  # 1 x 1 x 260
            if self.logging:
                print("\nusing BEV occupancy to interact with planning")

        # fusing latent feature of the map queries
        if self.use_lane_query or self.use_map_query:
            plan_query = torch.cat(
                [plan_query, plan_query_map_latent], dim=-1
            )  # 1 x 1 x 516
            if self.logging:
                print("\nusing latent lane/map query to interact with planning")

        # fusing latent feature of the map queries
        if self.use_occ_query:
            plan_query = torch.cat(
                [plan_query, plan_query_occ_latent], dim=-1
            )  # 1 x 1 x 516
            if self.logging:
                print("\nusing latent occupancy query to interact with planning")

        ############## encode additional ego info
        if self.use_ego_his or self.use_can_bus:

            ego_additional_info = []

            # flatten the ego past trajectory and mask into one dimension
            # and concatenate the data
            if self.use_ego_his:

                # converting list into the Tensor
                if isinstance(sdc_planning_past, list):
                    sdc_planning_past = sdc_planning_past[0]
                    sdc_planning_mask_past = sdc_planning_mask_past[0]

                # concatenate the trajectory and mask
                ego_his = torch.cat(
                    [
                        sdc_planning_past.view(1, 1, -1),
                        sdc_planning_mask_past.view(1, 1, -1),
                    ],
                    dim=-1,
                ).to(dtype=dtype)  # 1 x 1 x 24
                ego_additional_info.append(ego_his)
            
            # add the can bus information
            if self.use_can_bus:
                ego_additional_info.append(can_bus.view(1, 1, -1))  # 1 x 1 x 18
            
            ego_additional_info = torch.cat(ego_additional_info, dim=-1)  # 1 x 1 x 42
            plan_ego = self.mlp_ego(ego_additional_info)  # 1 x 1 x 256
            plan_query = torch.cat([plan_query, plan_ego], dim=-1)  # 1 x 1 x 516

        ############## decoder to get the trajectory
        sdc_traj_all = self.reg_branch(plan_query).view(
            (-1, self.planning_steps, 2)
        )  # 1 x F x 2
        
        # old UniAD code with bug
        # sdc_traj_all[..., :2] = torch.cumsum(sdc_traj_all[..., :2], dim=2)  # 1 x F x 2
        
        # BUG fix
        sdc_traj_all[..., :2] = torch.cumsum(sdc_traj_all[..., :2], dim=1)  # 1 x F x 2
        sdc_traj_all[0] = bivariate_gaussian_activation(
            sdc_traj_all[0]
        )  # 1 x F x 2, wlh and in xy coordinate

        ############ cost-based optmization during test time
        if self.use_col_optim and not self.training:
            # post process, only used when testing
            assert occ_mask is not None  # 1 x 5 x 1 x H x W
            sdc_traj_all = self.collision_optimization(sdc_traj_all, occ_mask)

        return dict(
            sdc_traj=sdc_traj_all,
            sdc_traj_all=sdc_traj_all,
        )

    def collision_optimization(self, sdc_traj_all, occ_mask):
        """
        Optimize SDC trajectory with occupancy instance mask.

        Args:
            sdc_traj_all (torch.Tensor): SDC trajectory tensor, xy coordinate wlh
            occ_mask (torch.Tensor): Occupancy flow instance mask.
        Returns:
            torch.Tensor: Optimized SDC trajectory tensor.
        """
        pos_xy_t = []
        valid_occupancy_num = 0

        occ_mask_ori = copy.copy(occ_mask)

        # same as the GT, in the yx coordinate
        if occ_mask.shape[2] == 1:
            occ_mask = occ_mask.squeeze(2)
        occ_horizon = occ_mask.shape[1]
        assert occ_horizon == 5

        for t in range(self.planning_steps):
            cur_t = min(t + 1, occ_horizon - 1)

            # find the yx that is occupied
            pos_xy = torch.nonzero(occ_mask[0][cur_t], as_tuple=False)  # 118 x 2

            # switch from yx to xy coordinate to align with sdc_traj_all
            pos_xy = pos_xy[:, [1, 0]]  # 118 x 2

            # change the mask coordinate to xy grid
            pos_xy[:, 0] = (pos_xy[:, 0] - self.bev_h // 2) * 0.5 + 0.25
            pos_xy[:, 1] = (pos_xy[:, 1] - self.bev_w // 2) * 0.5 + 0.25

            # filter the occupancy in range, in the xy coordinate
            keep_index = (
                torch.sum(
                    (sdc_traj_all[0, t, :2][None, :] - pos_xy[:, :2]) ** 2, axis=-1
                )
                < self.occ_filter_range ** 2
            )

            pos_xy_t.append(pos_xy[keep_index].cpu().detach().numpy())
            valid_occupancy_num += torch.sum(keep_index > 0)

        # print(valid_occupancy_num)
        if valid_occupancy_num == 0:
            return sdc_traj_all

        col_optimizer = CollisionNonlinearOptimizer(
            self.planning_steps, 0.5, self.sigma, self.alpha_collision, pos_xy_t
        )
        col_optimizer.set_reference_trajectory(sdc_traj_all[0].cpu().detach().numpy())
        sol = col_optimizer.solve()
        sdc_traj_optim = np.stack(
            [sol.value(col_optimizer.position_x), sol.value(col_optimizer.position_y)],
            axis=-1,
        )  # F x 2, xy coordinate

        return torch.tensor(
            sdc_traj_optim[None], device=sdc_traj_all.device, dtype=sdc_traj_all.dtype
        )

    def loss(self, sdc_planning, sdc_planning_mask, outs_planning, future_gt_bbox=None):
        # sdc_planning: 1 x 1 x F x 3, both in xy coordinates, wlh format
        # future_gt_bbox: List of 1 x (F+1), N x 9, yx coordinates, lwh format

        sdc_traj_all = outs_planning[
            "sdc_traj_all"
        ]  # 1 x F x 2, xy coordinates, wlh format

        loss_dict = dict()
        for i in range(len(self.loss_collision)):
            loss_collision = self.loss_collision[i](
                sdc_traj_all,
                sdc_planning[0, :, : self.planning_steps, :3],
                torch.any(sdc_planning_mask[0, :, : self.planning_steps], dim=-1),
                future_gt_bbox[0][1 : self.planning_steps + 1],
                scene_token=self.scene_token,
                frame_idx=self.frame_idx,
            )
            loss_dict[f"loss_collision_{i}"] = loss_collision

        loss_ade = self.loss_planning(
            sdc_traj_all,
            sdc_planning[0, :, : self.planning_steps, :2],
            torch.any(sdc_planning_mask[0, :, : self.planning_steps], dim=-1),
        )
        loss_dict.update(dict(loss_ade=loss_ade))
        return loss_dict
