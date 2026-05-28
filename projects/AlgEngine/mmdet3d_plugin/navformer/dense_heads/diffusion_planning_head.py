"""DiffusionDrive planning head, ported into the WorldEngine NavFormer plugin.

This module is fully self-contained: it inlines every helper, attention layer
and loss originally distributed across
``DiffusionDrive-main/navsim/agents/diffusiondrive/{transfuser_model_v2.py,
modules/blocks.py, modules/conditional_unet1d.py, modules/multimodal_loss.py}``
so that the upstream ``DiffusionDrive-main`` directory can be removed without
breaking this code path.

The structural design (DiT block with three cross-attentions: BEV / agents /
ego, plus DiT-style time modulation) is preserved unchanged from the paper.
The difference relative to the original DiffusionDrive ``V2TransfuserModel``
is that the "query preparation" step (a small TransformerDecoder which
extracts an ego-query and 30 agent-queries from the BEV+status keyval) is
executed *inside* this head, so that the NavFormer detector keeps the same
calling convention as ``TrajScoringHead``.
"""

import copy
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusers.schedulers import DDIMScheduler
from mmcv.runner import auto_fp16, force_fp32
from mmdet.models.builder import HEADS

from .traj_scoring_head import nerf_positional_encoding


# ---------------------------------------------------------------------------
# Helpers ported from DiffusionDrive `modules/blocks.py` and
# `modules/conditional_unet1d.py`.
# ---------------------------------------------------------------------------


def linear_relu_ln(embed_dims, in_loops, out_loops, input_dims=None):
    """Build [Linear, ReLU, ..., LayerNorm] stack (DiffusionDrive blocks.py:8)."""
    if input_dims is None:
        input_dims = embed_dims
    layers = []
    for _ in range(out_loops):
        for _ in range(in_loops):
            layers.append(nn.Linear(input_dims, embed_dims))
            layers.append(nn.ReLU(inplace=True))
            input_dims = embed_dims
        layers.append(nn.LayerNorm(embed_dims))
    return layers


def gen_sineembed_for_position(pos_tensor, hidden_dim=256):
    """Sinusoidal positional embedding for 2-D points (DAB-DETR style)."""
    half_hidden_dim = hidden_dim // 2
    scale = 2 * math.pi
    dim_t = torch.arange(half_hidden_dim, dtype=torch.float32, device=pos_tensor.device)
    dim_t = 10000 ** (2 * (dim_t // 2) / half_hidden_dim)
    x_embed = pos_tensor[..., 0] * scale
    y_embed = pos_tensor[..., 1] * scale
    pos_x = x_embed[..., None] / dim_t
    pos_y = y_embed[..., None] / dim_t
    pos_x = torch.stack((pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()), dim=-1).flatten(-2)
    pos_y = torch.stack((pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()), dim=-1).flatten(-2)
    pos = torch.cat((pos_y, pos_x), dim=-1)
    return pos


def bias_init_with_prob(prior_prob):
    return float(-np.log((1 - prior_prob) / prior_prob))


class SinusoidalPosEmb(nn.Module):
    """Sinusoidal embedding for diffusion timesteps."""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class GridSampleCrossBEVAttention(nn.Module):
    """BEV cross-attention via ``F.grid_sample`` over trajectory waypoints.

    Modified from DiffusionDrive ``modules/blocks.py:42`` to accept explicit
    ``bev_range_x`` / ``bev_range_y`` instead of pulling values from a global
    config object.
    """

    def __init__(
        self,
        embed_dims,
        num_heads,
        num_points=8,
        bev_range_x=51.2,
        bev_range_y=51.2,
        in_bev_dims=256,
    ):
        super().__init__()
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.num_points = num_points
        self.bev_range_x = bev_range_x
        self.bev_range_y = bev_range_y

        self.attention_weights = nn.Linear(embed_dims, num_points)
        self.output_proj = nn.Linear(embed_dims, embed_dims)
        self.dropout = nn.Dropout(0.1)
        self.value_proj = nn.Sequential(
            nn.Conv2d(in_bev_dims, embed_dims, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(inplace=True),
        )
        self.init_weight()

    def init_weight(self):
        nn.init.constant_(self.attention_weights.weight, 0)
        nn.init.constant_(self.attention_weights.bias, 0)
        nn.init.xavier_uniform_(self.output_proj.weight)
        nn.init.constant_(self.output_proj.bias, 0)

    def forward(self, queries, traj_points, bev_feature, spatial_shape):
        bs, num_queries, num_points, _ = traj_points.shape

        # Normalize trajectory points to [-1, 1] for grid_sample.
        normalized_trajectory = traj_points.clone()
        normalized_trajectory[..., 0] = normalized_trajectory[..., 0] / self.bev_range_y
        normalized_trajectory[..., 1] = normalized_trajectory[..., 1] / self.bev_range_x
        normalized_trajectory = normalized_trajectory[..., [1, 0]]  # swap x and y

        attention_weights = self.attention_weights(queries)
        attention_weights = attention_weights.view(bs, num_queries, num_points).softmax(-1)

        value = self.value_proj(bev_feature)
        grid = normalized_trajectory.view(bs, num_queries, num_points, 2)
        sampled_features = F.grid_sample(
            value, grid, mode="bilinear", padding_mode="zeros", align_corners=False
        )  # bs, C, num_queries, num_points

        attention_weights = attention_weights.unsqueeze(1)
        out = (attention_weights * sampled_features).sum(dim=-1)
        out = out.permute(0, 2, 1).contiguous()
        out = self.output_proj(out)
        return self.dropout(out) + queries


# ---------------------------------------------------------------------------
# Loss helpers ported from DiffusionDrive `modules/multimodal_loss.py`.
# ---------------------------------------------------------------------------


def _reduce_loss(loss, reduction):
    reduction_enum = F._Reduction.get_enum(reduction)
    if reduction_enum == 0:
        return loss
    if reduction_enum == 1:
        return loss.mean()
    return loss.sum()


def _weight_reduce_loss(loss, weight=None, reduction="mean", avg_factor=None):
    if weight is not None:
        loss = loss * weight
    if avg_factor is None:
        loss = _reduce_loss(loss, reduction)
    else:
        if reduction == "mean":
            eps = torch.finfo(torch.float32).eps
            loss = loss.sum() / (avg_factor + eps)
        elif reduction != "none":
            raise ValueError('avg_factor can not be used with reduction="sum"')
    return loss


def py_sigmoid_focal_loss(
    pred, target, weight=None, gamma=2.0, alpha=0.25, reduction="mean", avg_factor=None
):
    pred_sigmoid = pred.sigmoid()
    target = target.type_as(pred)
    pt = (1 - pred_sigmoid) * target + pred_sigmoid * (1 - target)
    focal_weight = (alpha * target + (1 - alpha) * (1 - target)) * pt.pow(gamma)
    loss = F.binary_cross_entropy_with_logits(pred, target, reduction="none") * focal_weight
    if weight is not None:
        if weight.shape != loss.shape:
            if weight.size(0) == loss.size(0):
                weight = weight.view(-1, 1)
            else:
                assert weight.numel() == loss.numel()
                weight = weight.view(loss.size(0), -1)
        assert weight.ndim == loss.ndim
    return _weight_reduce_loss(loss, weight, reduction, avg_factor)


class LossComputer(nn.Module):
    """Anchor-matched winner-take-all focal-cls + L1-reg loss.

    Returns ``(cls_loss, reg_loss)`` (split out from the original implementation
    which returned the sum) so the planning head can log them separately.
    """

    def __init__(self, cls_loss_weight, reg_loss_weight):
        super().__init__()
        self.cls_loss_weight = cls_loss_weight
        self.reg_loss_weight = reg_loss_weight

    def forward(self, poses_reg, poses_cls, target_traj, plan_anchor, target_mask=None):
        """
        poses_reg: (B, M, T, 3)
        poses_cls: (B, M)
        target_traj: (B, T, 3)
        plan_anchor: (B, M, T, 2)
        target_mask: optional (B, T), 1 for valid future waypoints
        """
        bs, num_mode, ts, d = poses_reg.shape

        if target_mask is None:
            target_mask = target_traj.new_ones((bs, ts))
        else:
            target_mask = target_mask.to(dtype=target_traj.dtype, device=target_traj.device)

        dist = torch.linalg.norm(target_traj.unsqueeze(1)[..., :2] - plan_anchor, dim=-1)
        valid_count = target_mask.sum(dim=-1).clamp_min(1.0)
        dist = (dist * target_mask[:, None]).sum(dim=-1) / valid_count[:, None]
        cls_target = torch.argmin(dist, dim=-1)

        gather_idx = cls_target[..., None, None, None].repeat(1, 1, ts, d)
        best_reg = torch.gather(poses_reg, 1, gather_idx).squeeze(1)

        target_classes_onehot = torch.zeros(
            (bs, num_mode), dtype=poses_cls.dtype, layout=poses_cls.layout, device=poses_cls.device
        )
        target_classes_onehot.scatter_(1, cls_target.unsqueeze(1), 1)

        cls_loss = self.cls_loss_weight * py_sigmoid_focal_loss(
            poses_cls,
            target_classes_onehot,
            weight=None,
            gamma=2.0,
            alpha=0.25,
            reduction="mean",
            avg_factor=None,
        )
        reg_weight = target_mask[..., None]
        reg_denorm = reg_weight.sum().clamp_min(1.0) * d
        reg_loss = self.reg_loss_weight * (
            torch.abs(best_reg - target_traj) * reg_weight
        ).sum() / reg_denorm
        return cls_loss, reg_loss


# ---------------------------------------------------------------------------
# DiT block components ported from `transfuser_model_v2.py`.
# ---------------------------------------------------------------------------


class DiffMotionPlanningRefinementModule(nn.Module):
    """Per-anchor cls + reg MLP heads (transfuser_model_v2.py:182)."""

    def __init__(self, embed_dims=256, ego_fut_ts=8, ego_fut_mode=20):
        super().__init__()
        self.embed_dims = embed_dims
        self.ego_fut_ts = ego_fut_ts
        self.ego_fut_mode = ego_fut_mode

        self.plan_cls_branch = nn.Sequential(
            *linear_relu_ln(embed_dims, 1, 2),
            nn.Linear(embed_dims, 1),
        )
        self.plan_reg_branch = nn.Sequential(
            nn.Linear(embed_dims, embed_dims),
            nn.ReLU(),
            nn.Linear(embed_dims, embed_dims),
            nn.ReLU(),
            nn.Linear(embed_dims, ego_fut_ts * 3),
        )
        self.if_zeroinit_reg = False
        self.init_weight()

    def init_weight(self):
        if self.if_zeroinit_reg:
            nn.init.constant_(self.plan_reg_branch[-1].weight, 0)
            nn.init.constant_(self.plan_reg_branch[-1].bias, 0)
        bias_init = bias_init_with_prob(0.01)
        nn.init.constant_(self.plan_cls_branch[-1].bias, bias_init)

    def forward(self, traj_feature):
        bs, ego_fut_mode, _ = traj_feature.shape
        traj_feature = traj_feature.view(bs, ego_fut_mode, -1)
        plan_cls = self.plan_cls_branch(traj_feature).squeeze(-1)
        traj_delta = self.plan_reg_branch(traj_feature)
        plan_reg = traj_delta.reshape(bs, ego_fut_mode, self.ego_fut_ts, 3)
        return plan_reg, plan_cls


class ModulationLayer(nn.Module):
    """DiT-style adaLN/FiLM modulation (transfuser_model_v2.py:229)."""

    def __init__(self, embed_dims, condition_dims):
        super().__init__()
        self.if_zeroinit_scale = False
        self.embed_dims = embed_dims
        self.scale_shift_mlp = nn.Sequential(
            nn.Mish(),
            nn.Linear(condition_dims, embed_dims * 2),
        )
        self.init_weight()

    def init_weight(self):
        if self.if_zeroinit_scale:
            nn.init.constant_(self.scale_shift_mlp[-1].weight, 0)
            nn.init.constant_(self.scale_shift_mlp[-1].bias, 0)

    def forward(self, traj_feature, time_embed, global_cond=None, global_img=None):
        if global_cond is not None:
            global_feature = torch.cat([global_cond, time_embed], dim=-1)
        else:
            global_feature = time_embed
        if global_img is not None:
            global_img = global_img.flatten(2, 3).permute(0, 2, 1).contiguous()
            global_feature = torch.cat([global_img, global_feature], dim=-1)
        scale_shift = self.scale_shift_mlp(global_feature)
        scale, shift = scale_shift.chunk(2, dim=-1)
        return traj_feature * (1 + scale) + shift


class CustomTransformerDecoderLayer(nn.Module):
    """One DiT block: BEV-cross + agent-cross + ego-cross + FFN + time-modulation.

    Structurally identical to ``transfuser_model_v2.py:270`` — three cross
    attentions (BEV / agents / ego) are kept; only the constructor signature
    is unrolled to avoid passing a ``TransfuserConfig`` dataclass.
    """

    def __init__(
        self,
        num_poses,
        d_model,
        d_ffn,
        num_heads,
        dropout,
        bev_range_x,
        bev_range_y,
        num_anchors,
    ):
        super().__init__()
        self.dropout = nn.Dropout(0.1)
        self.dropout1 = nn.Dropout(0.1)

        self.cross_bev_attention = GridSampleCrossBEVAttention(
            d_model,
            num_heads,
            num_points=num_poses,
            bev_range_x=bev_range_x,
            bev_range_y=bev_range_y,
            in_bev_dims=d_model,
        )
        self.cross_agent_attention = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True
        )
        self.cross_ego_attention = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True
        )
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ffn),
            nn.ReLU(),
            nn.Linear(d_ffn, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.time_modulation = ModulationLayer(d_model, d_model)
        self.task_decoder = DiffMotionPlanningRefinementModule(
            embed_dims=d_model,
            ego_fut_ts=num_poses,
            ego_fut_mode=num_anchors,
        )

    def forward(
        self,
        traj_feature,
        noisy_traj_points,
        bev_feature,
        bev_spatial_shape,
        agents_query,
        ego_query,
        time_embed,
        status_encoding,
        global_img=None,
    ):
        traj_feature = self.cross_bev_attention(
            traj_feature, noisy_traj_points, bev_feature, bev_spatial_shape
        )
        traj_feature = traj_feature + self.dropout(
            self.cross_agent_attention(traj_feature, agents_query, agents_query)[0]
        )
        traj_feature = self.norm1(traj_feature)

        traj_feature = traj_feature + self.dropout1(
            self.cross_ego_attention(traj_feature, ego_query, ego_query)[0]
        )
        traj_feature = self.norm2(traj_feature)

        traj_feature = self.norm3(self.ffn(traj_feature))
        traj_feature = self.time_modulation(
            traj_feature, time_embed, global_cond=None, global_img=global_img
        )

        poses_reg, poses_cls = self.task_decoder(traj_feature)
        poses_reg[..., :2] = poses_reg[..., :2] + noisy_traj_points
        # heading is the third channel; tanh * pi gives a bounded angle prediction
        poses_reg[..., 2] = poses_reg[..., 2].tanh() * np.pi
        return poses_reg, poses_cls


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class CustomTransformerDecoder(nn.Module):
    """Stacks DiT blocks with deep supervision (transfuser_model_v2.py:350)."""

    def __init__(self, decoder_layer, num_layers):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers

    def forward(
        self,
        traj_feature,
        noisy_traj_points,
        bev_feature,
        bev_spatial_shape,
        agents_query,
        ego_query,
        time_embed,
        status_encoding,
        global_img=None,
    ):
        poses_reg_list = []
        poses_cls_list = []
        traj_points = noisy_traj_points
        for layer in self.layers:
            poses_reg, poses_cls = layer(
                traj_feature,
                traj_points,
                bev_feature,
                bev_spatial_shape,
                agents_query,
                ego_query,
                time_embed,
                status_encoding,
                global_img,
            )
            poses_reg_list.append(poses_reg)
            poses_cls_list.append(poses_cls)
            traj_points = poses_reg[..., :2].clone().detach()
        return poses_reg_list, poses_cls_list


# ---------------------------------------------------------------------------
# Top-level head registered with mmdet's HEADS registry.
# ---------------------------------------------------------------------------


@HEADS.register_module()
class DiffusionPlanningHead(nn.Module):
    """DiffusionDrive planning head adapted for the NavFormer detector.

    Calling convention matches :class:`TrajScoringHead`:

        forward(bev_embed, command, sdc_planning_past, sdc_status,
                sdc_planning_mask_past, gt_pre_command_sdc) -> dict
        loss(result, gt_pdm_score=None, sdc_planning=..., sdc_planning_mask=...)
            -> dict[str, Tensor]

    The dict returned by ``forward`` contains the generated trajectory
    (``trajectory`` of shape ``(B, 40, 3)``), plus extra intermediate tensors
    consumed by ``loss`` during training.
    """

    def __init__(
        self,
        num_poses=8,
        d_model=256,
        d_ffn=1024,
        num_heads=8,
        dropout=0.0,
        num_bounding_boxes=30,
        num_query_decoder_layers=3,
        query_keyval_size=8,
        num_anchors=20,
        num_diff_decoder_layers=2,
        plan_anchor_path=None,
        vocab_path=None,
        score_mode="recompute",
        bev_h=200,
        bev_w=200,
        bev_range_x=51.2,
        bev_range_y=51.2,
        odo_x_min=-1.2,
        odo_x_range=56.9,
        odo_y_min=-20.0,
        odo_y_range=46.0,
        num_train_timesteps=1000,
        train_timestep_max=50,
        inference_steps=2,
        trunc_timesteps=8,
        cls_loss_weight=10.0,
        reg_loss_weight=8.0,
        use_nerf=True,
        **kwargs,
    ):
        super().__init__()

        if plan_anchor_path is None:
            raise ValueError("`plan_anchor_path` must be provided.")

        self._num_poses = num_poses
        self.score_mode = score_mode
        self.d_model = d_model
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.num_bounding_boxes = num_bounding_boxes
        self.num_anchors = num_anchors
        self.query_keyval_size = query_keyval_size
        self.train_timestep_max = train_timestep_max
        self.inference_steps = inference_steps
        self.trunc_timesteps = trunc_timesteps
        self.use_nerf = use_nerf

        self.odo_x_min = odo_x_min
        self.odo_x_range = odo_x_range
        self.odo_y_min = odo_y_min
        self.odo_y_range = odo_y_range

        # (a) status token encoder (NeRF-style, mirroring TrajScoringHead)
        if self.use_nerf:
            self.status_embed = nn.Sequential(
                nn.Linear(4 + 24 + 2, d_model),
                nn.ReLU(),
            )
        else:
            self.status_embed = nn.Sequential(
                nn.Linear(4 + 2 + 2, d_model),
                nn.ReLU(),
            )

        # (b) learnable queries: 1 ego + N agents (DiffusionDrive V2 line 37)
        self._query_embedding = nn.Embedding(num_bounding_boxes + 1, d_model)

        # (c) keyval positional embedding for (keyval_size^2 BEV tokens + 1 status token).
        # We downsample the 200x200 NavFormer BEV to keyval_size x keyval_size before
        # feeding the query-extraction TransformerDecoder — this matches the spirit of
        # DiffusionDrive's 8x8 keyval grid (transfuser_model_v2.py:36) and keeps the
        # query decoder's memory footprint reasonable. The full-resolution BEV is still
        # used for the DiT block's grid_sample cross-attention.
        self._keyval_embedding = nn.Embedding(query_keyval_size * query_keyval_size + 1, d_model)

        # (d) TF decoder used to extract ego/agent queries from BEV+status
        query_decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_ffn,
            dropout=dropout,
            batch_first=True,
        )
        self._query_tf_decoder = nn.TransformerDecoder(
            query_decoder_layer, num_query_decoder_layers
        )

        # (e) DDIM scheduler (predicts the clean sample, scaled-linear betas)
        self.diffusion_scheduler = DDIMScheduler(
            num_train_timesteps=num_train_timesteps,
            beta_schedule="scaled_linear",
            prediction_type="sample",
        )

        # (f) plan anchor (M, T, 2) -> registered as a frozen parameter
        plan_anchor = np.load(plan_anchor_path)
        self.plan_anchor = nn.Parameter(
            torch.tensor(plan_anchor, dtype=torch.float32),
            requires_grad=False,
        )
        self.plan_anchor_encoder = nn.Sequential(
            *linear_relu_ln(d_model, 1, 1, 512),
            nn.Linear(d_model, d_model),
        )

        # (g) timestep embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(d_model),
            nn.Linear(d_model, d_model * 4),
            nn.Mish(),
            nn.Linear(d_model * 4, d_model),
        )

        # (h) stacked DiT decoder
        diff_decoder_layer = CustomTransformerDecoderLayer(
            num_poses=num_poses,
            d_model=d_model,
            d_ffn=d_ffn,
            num_heads=num_heads,
            dropout=dropout,
            bev_range_x=bev_range_x,
            bev_range_y=bev_range_y,
            num_anchors=num_anchors,
        )
        self.diff_decoder = CustomTransformerDecoder(
            diff_decoder_layer, num_diff_decoder_layers
        )

        # (i) loss
        self.loss_computer = LossComputer(cls_loss_weight, reg_loss_weight)

    # ------------------------------------------------------------------
    # Trajectory range normalization (x, y only — heading is regressed afresh
    # by the cls/reg head so it never enters the diffusion buffer).
    # ------------------------------------------------------------------

    def norm_odo_xy(self, traj_xy):
        x = 2 * (traj_xy[..., 0:1] - self.odo_x_min) / self.odo_x_range - 1
        y = 2 * (traj_xy[..., 1:2] - self.odo_y_min) / self.odo_y_range - 1
        return torch.cat([x, y], dim=-1)

    def denorm_odo_xy(self, traj_xy):
        x = (traj_xy[..., 0:1] + 1) / 2 * self.odo_x_range + self.odo_x_min
        y = (traj_xy[..., 1:2] + 1) / 2 * self.odo_y_range + self.odo_y_min
        return torch.cat([x, y], dim=-1)

    # ------------------------------------------------------------------
    # Status token + query preparation.
    # ------------------------------------------------------------------

    def _build_status_token(
        self,
        command,
        sdc_planning_past,
        sdc_status,
        sdc_planning_mask_past,
        gt_pre_command_sdc,
    ):
        """Mirrors TrajScoringHead.forward up to status_encoding (lines 213-260).

        Output: ``(B, 1, d_model)``.
        """
        gt_pre_command_sdc = gt_pre_command_sdc[:, 0, :, 0]
        sdc_planning_past = sdc_planning_past[:, 0]

        full_cmd = torch.cat([gt_pre_command_sdc, command[:, None]], dim=1).long()
        cmd_one_hot = F.one_hot(full_cmd, num_classes=4).float()

        full_ego_status = torch.cat([sdc_planning_past, sdc_status[:, None]], dim=1)
        if self.use_nerf:
            enc_ego_status = torch.cat(
                [
                    cmd_one_hot,
                    nerf_positional_encoding(full_ego_status[..., :2]),
                    torch.cos(full_ego_status[..., -1])[..., None],
                    torch.sin(full_ego_status[..., -1])[..., None],
                ],
                dim=-1,
            )
        else:
            enc_ego_status = torch.cat(
                [
                    cmd_one_hot,
                    full_ego_status[..., :2],
                    torch.cos(full_ego_status[..., -1])[..., None],
                    torch.sin(full_ego_status[..., -1])[..., None],
                ],
                dim=-1,
            )

        enc_ego_status = enc_ego_status.float()
        status_encoding = self.status_embed(enc_ego_status)  # (B, 5, d_model)

        mask_past = sdc_planning_mask_past[:, 0, :, 0].float()
        b = mask_past.shape[0]
        mask_past = torch.cat([mask_past, torch.zeros((b, 1), device=status_encoding.device)], dim=1)
        mask_past = mask_past[:, :, None]

        status_token = torch.max(status_encoding * mask_past, dim=1)[0]  # (B, d_model)
        return status_token.unsqueeze(1)  # (B, 1, d_model)

    def _prepare_queries(self, bev_feature, status_token):
        """Run the small TF decoder to extract ego_query + agents_query.

        Mirrors V2TransfuserModel.forward lines 110-128. We adaptively pool the
        full-resolution BEV down to ``query_keyval_size x query_keyval_size``
        before feeding it to the query decoder, otherwise 200x200=40000 tokens
        would blow up memory in the FFN.
        """
        B = bev_feature.shape[0]
        ds = F.adaptive_avg_pool2d(bev_feature, self.query_keyval_size)  # (B, C, S, S)
        bev_kv = ds.flatten(-2, -1).permute(0, 2, 1).contiguous()        # (B, S*S, C)

        keyval = torch.cat([bev_kv, status_token], dim=1)
        keyval = keyval + self._keyval_embedding.weight[None, ...]

        query = self._query_embedding.weight[None, ...].repeat(B, 1, 1)
        query_out = self._query_tf_decoder(query, keyval)
        ego_query, agents_query = query_out.split([1, self.num_bounding_boxes], dim=1)
        return ego_query, agents_query

    # ------------------------------------------------------------------
    # Forward: dispatch to train/test path.
    # ------------------------------------------------------------------

    @auto_fp16(apply_to=("bev_embed",))
    def forward(
        self,
        bev_embed,
        command=None,
        sdc_planning_past=None,
        sdc_status=None,
        sdc_planning_mask_past=None,
        gt_pre_command_sdc=None,
    ):
        # bev_embed comes as (H*W, B, C); reshape to (B, C, H, W).
        if bev_embed.dim() == 3 and bev_embed.shape[0] == self.bev_h * self.bev_w:
            HW, B, C = bev_embed.shape
            bev_feature = (
                bev_embed.permute(1, 2, 0).contiguous().view(B, C, self.bev_h, self.bev_w)
            )
        else:
            raise ValueError(
                f"Unexpected bev_embed shape {tuple(bev_embed.shape)}; expected (H*W, B, C)"
            )

        status_token = self._build_status_token(
            command,
            sdc_planning_past,
            sdc_status,
            sdc_planning_mask_past,
            gt_pre_command_sdc,
        )
        ego_query, agents_query = self._prepare_queries(bev_feature, status_token)

        if self.training:
            return self._forward_train(
                bev_feature, ego_query, agents_query, status_token
            )
        return self._forward_test(
            bev_feature, ego_query, agents_query, status_token
        )

    # ------------------------------------------------------------------
    # Train / test diffusion loops.
    # ------------------------------------------------------------------

    def _expand_to_40(self, traj_8):
        """Repeat (B, 8, 3) along time 5x to (B, 40, 3) so [4::5] recovers it."""
        B = traj_8.shape[0]
        return traj_8.unsqueeze(2).expand(B, self._num_poses, 5, 3).reshape(B, self._num_poses * 5, 3)

    def _forward_train(self, bev_feature, ego_query, agents_query, status_token):
        bs = ego_query.shape[0]
        device = ego_query.device

        plan_anchor = self.plan_anchor.unsqueeze(0).repeat(bs, 1, 1, 1)  # (B, M, T, 2)
        odo_info_fut = self.norm_odo_xy(plan_anchor)

        timesteps = torch.randint(0, self.train_timestep_max, (bs,), device=device)
        noise = torch.randn(odo_info_fut.shape, device=device)
        noisy_traj_points = self.diffusion_scheduler.add_noise(
            original_samples=odo_info_fut,
            noise=noise,
            timesteps=timesteps,
        ).float()
        noisy_traj_points = torch.clamp(noisy_traj_points, min=-1, max=1)
        noisy_traj_points = self.denorm_odo_xy(noisy_traj_points)

        ego_fut_mode = noisy_traj_points.shape[1]
        traj_pos_embed = gen_sineembed_for_position(noisy_traj_points, hidden_dim=64)
        traj_pos_embed = traj_pos_embed.flatten(-2)
        traj_feature = self.plan_anchor_encoder(traj_pos_embed)
        traj_feature = traj_feature.view(bs, ego_fut_mode, -1)

        time_embed = self.time_mlp(timesteps).view(bs, 1, -1)

        bev_spatial_shape = (self.bev_h, self.bev_w)
        poses_reg_list, poses_cls_list = self.diff_decoder(
            traj_feature,
            noisy_traj_points,
            bev_feature,
            bev_spatial_shape,
            agents_query,
            ego_query,
            time_embed,
            status_token,
            None,
        )

        # Best trajectory (last layer) for downstream consumers.
        last_reg = poses_reg_list[-1]
        last_cls = poses_cls_list[-1]
        mode_idx = last_cls.argmax(dim=-1)
        gather_idx = mode_idx[..., None, None, None].repeat(1, 1, self._num_poses, 3)
        best_reg = torch.gather(last_reg, 1, gather_idx).squeeze(1)  # (B, 8, 3)

        return {
            "trajectory": self._expand_to_40(best_reg),
            "trajectory_8": best_reg,
            "poses_reg_list": poses_reg_list,
            "poses_cls_list": poses_cls_list,
            "plan_anchor_expanded": plan_anchor,  # (B, M, T, 2) for loss
        }

    def _forward_test(self, bev_feature, ego_query, agents_query, status_token):
        bs = ego_query.shape[0]
        device = ego_query.device

        self.diffusion_scheduler.set_timesteps(1000, device)
        step_ratio = 20 / self.inference_steps
        roll_timesteps = (np.arange(0, self.inference_steps) * step_ratio).round()[::-1].copy()
        roll_timesteps = torch.from_numpy(roll_timesteps.astype(np.int64)).to(device)

        plan_anchor = self.plan_anchor.unsqueeze(0).repeat(bs, 1, 1, 1)  # (B, M, T, 2)
        img = self.norm_odo_xy(plan_anchor)

        noise = torch.randn(img.shape, device=device)
        trunc_t = torch.ones((bs,), device=device, dtype=torch.long) * self.trunc_timesteps
        img = self.diffusion_scheduler.add_noise(
            original_samples=img, noise=noise, timesteps=trunc_t
        )

        ego_fut_mode = img.shape[1]
        bev_spatial_shape = (self.bev_h, self.bev_w)

        poses_reg = None
        poses_cls = None
        for k in roll_timesteps:
            x_clamped = torch.clamp(img, min=-1, max=1)
            noisy_traj_points = self.denorm_odo_xy(x_clamped)

            traj_pos_embed = gen_sineembed_for_position(noisy_traj_points, hidden_dim=64)
            traj_pos_embed = traj_pos_embed.flatten(-2)
            traj_feature = self.plan_anchor_encoder(traj_pos_embed)
            traj_feature = traj_feature.view(bs, ego_fut_mode, -1)

            timesteps = k
            if not torch.is_tensor(timesteps):
                timesteps = torch.tensor([timesteps], dtype=torch.long, device=device)
            elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
                timesteps = timesteps[None].to(device)
            timesteps = timesteps.expand(bs)

            time_embed = self.time_mlp(timesteps).view(bs, 1, -1)

            poses_reg_list, poses_cls_list = self.diff_decoder(
                traj_feature,
                noisy_traj_points,
                bev_feature,
                bev_spatial_shape,
                agents_query,
                ego_query,
                time_embed,
                status_token,
                None,
            )
            poses_reg = poses_reg_list[-1]
            poses_cls = poses_cls_list[-1]

            # The DDIM scheduler stores (x, y) — we predict x_start in normalized
            # space and feed it back as the model output for the next step.
            x_start = self.norm_odo_xy(poses_reg[..., :2])
            img = self.diffusion_scheduler.step(
                model_output=x_start,
                timestep=k,
                sample=img,
            ).prev_sample

        mode_idx = poses_cls.argmax(dim=-1)
        gather_idx = mode_idx[..., None, None, None].repeat(1, 1, self._num_poses, 3)
        best_reg = torch.gather(poses_reg, 1, gather_idx).squeeze(1)  # (B, 8, 3)

        return {
            "trajectory": self._expand_to_40(best_reg),
            "trajectory_8": best_reg,
        }

    # ------------------------------------------------------------------
    # Loss.
    # ------------------------------------------------------------------

    @force_fp32(apply_to=("result", "gt_pdm_score", "sdc_planning"))
    def loss(
        self,
        result=None,
        gt_pdm_score=None,
        sdc_planning=None,
        sdc_planning_mask=None,
        il_target=None,
        il_target_mask=None,
    ):
        # sdc_planning: (B, 1, T_full, 3); we use the planning-step horizon directly
        # (T_full == num_poses == 8 in the current pipeline). DiffusionDrive's
        # supervision is the 8-step trajectory at native rate.
        target_traj = sdc_planning[:, 0]  # (B, T_full, 3)
        target_mask = None
        if sdc_planning_mask is not None:
            target_mask = sdc_planning_mask[:, 0, :, 0].float()
        if target_traj.shape[1] != self._num_poses:
            # Fall back to the same 4::5 stride that TrajScoringHead uses, so the
            # head still works if the dataset emits a 40-step ground truth.
            target_traj = target_traj[:, 4::5]
            if target_mask is not None:
                target_mask = target_mask[:, 4::5]

        plan_anchor = result["plan_anchor_expanded"]
        cls_total = sdc_planning.new_zeros(())
        reg_total = sdc_planning.new_zeros(())
        for poses_reg, poses_cls in zip(result["poses_reg_list"], result["poses_cls_list"]):
            cls_loss, reg_loss = self.loss_computer(
                poses_reg, poses_cls, target_traj, plan_anchor, target_mask
            )
            cls_total = cls_total + cls_loss
            reg_total = reg_total + reg_loss

        return {
            "loss.diff_cls": cls_total,
            "loss.diff_reg": reg_total,
        }
