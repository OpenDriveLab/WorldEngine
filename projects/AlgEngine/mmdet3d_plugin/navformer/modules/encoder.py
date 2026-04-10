import copy
import warnings
from functools import lru_cache
from mmcv.cnn.bricks.registry import (
    ATTENTION,
    TRANSFORMER_LAYER,
    TRANSFORMER_LAYER_SEQUENCE,
)
from mmcv.cnn.bricks.transformer import TransformerLayerSequence
from mmcv.runner import force_fp32, auto_fp16
import numpy as np
import torch
import cv2 as cv
import mmcv
from mmcv.utils import TORCH_VERSION, digit_version
from mmcv.utils import ext_loader


@TRANSFORMER_LAYER_SEQUENCE.register_module()
class NAVFormerEncoder(TransformerLayerSequence):

    """
    NAVFormerEncoder is a transformer encoder for NavFormer.
    The difference from BEVFormerEncoder is that it uses distortion sampling.
    """

    def __init__(
        self,
        *args,
        pc_range=None,
        num_points_in_pillar=4,
        return_intermediate=False,
        use_distortion_sampling=False,
        **kwargs,
    ):

        super(NAVFormerEncoder, self).__init__(*args, **kwargs)
        self.return_intermediate = return_intermediate

        self.num_points_in_pillar = num_points_in_pillar
        self.pc_range = pc_range
        self.fp16_enabled = False
        self.use_distortion_sampling = use_distortion_sampling

    @staticmethod
    @lru_cache(maxsize=4)
    def get_reference_points(
        H,
        W,
        Z=8,
        num_points_in_pillar=4,
        dim="3d",
        bs=1,
        device="cuda",
        dtype=torch.float,
    ):
        """Get the reference points used in SCA and TSA.
        Args:
            H, W: spatial shape of bev.
            Z: hight of pillar.
            D: sample D points uniformly from each pillar.
            device (obj:`device`): The device where
                reference_points should be.
        Returns:
            Tensor: reference points used in decoder, has \
                shape (bs, num_keys, num_levels, 2).
        """

        # reference points in 3D space, used in spatial cross-attention (SCA)
        if dim == "3d":
            zs = (
                torch.linspace(
                    0.5, Z - 0.5, num_points_in_pillar, dtype=dtype, device=device
                )
                .view(-1, 1, 1)
                .expand(num_points_in_pillar, H, W)
                / Z
            )
            xs = (
                torch.linspace(0.5, W - 0.5, W, dtype=dtype, device=device)
                .view(1, 1, W)
                .expand(num_points_in_pillar, H, W)
                / W
            )
            ys = (
                torch.linspace(0.5, H - 0.5, H, dtype=dtype, device=device)
                .view(1, H, 1)
                .expand(num_points_in_pillar, H, W)
                / H
            )
            ref_3d = torch.stack((xs, ys, zs), -1)
            ref_3d = ref_3d.permute(0, 3, 1, 2).flatten(2).permute(0, 2, 1)
            ref_3d = ref_3d[None].repeat(bs, 1, 1, 1)
            return ref_3d

        # reference points on 2D bev plane, used in temporal self-attention (TSA).
        elif dim == "2d":
            ref_y, ref_x = torch.meshgrid(
                torch.linspace(0.5, H - 0.5, H, dtype=dtype, device=device),
                torch.linspace(0.5, W - 0.5, W, dtype=dtype, device=device),
            )
            ref_y = ref_y.reshape(-1)[None] / H
            ref_x = ref_x.reshape(-1)[None] / W
            ref_2d = torch.stack((ref_x, ref_y), -1)
            ref_2d = ref_2d.repeat(bs, 1, 1).unsqueeze(2)
            return ref_2d

    # This function must use fp32!!!
    @force_fp32(apply_to=("reference_points", "img_metas"))
    def point_sampling(self, reference_points, pc_range, img_metas):

        lidar2img = []
        for img_meta in img_metas:
            lidar2img.append(img_meta["lidar2img"])
        lidar2img = np.asarray(lidar2img)
        lidar2img = reference_points.new_tensor(lidar2img)  # (B, N, 4, 4)
        reference_points = reference_points.clone()

        reference_points[..., 0:1] = (
            reference_points[..., 0:1] * (pc_range[3] - pc_range[0]) + pc_range[0]
        )
        reference_points[..., 1:2] = (
            reference_points[..., 1:2] * (pc_range[4] - pc_range[1]) + pc_range[1]
        )
        reference_points[..., 2:3] = (
            reference_points[..., 2:3] * (pc_range[5] - pc_range[2]) + pc_range[2]
        )

        reference_points = torch.cat(
            (reference_points, torch.ones_like(reference_points[..., :1])), -1
        )

        reference_points = reference_points.permute(1, 0, 2, 3)
        D, B, num_query = reference_points.size()[:3]
        num_cam = lidar2img.size(1)

        reference_points = (
            reference_points.view(D, B, 1, num_query, 4)
            .repeat(1, 1, num_cam, 1, 1)
            .unsqueeze(-1)
        )

        lidar2img = lidar2img.view(1, B, num_cam, 1, 4, 4).repeat(
            D, 1, 1, num_query, 1, 1
        )

        reference_points_cam = torch.matmul(
            lidar2img.to(torch.float32), reference_points.to(torch.float32)
        ).squeeze(-1)
        eps = 1e-5

        bev_mask = reference_points_cam[..., 2:3] > eps
        reference_points_cam = reference_points_cam[..., 0:2] / torch.maximum(
            reference_points_cam[..., 2:3],
            torch.ones_like(reference_points_cam[..., 2:3]) * eps,
        )

        reference_points_cam[..., 0] /= img_metas[0]["img_shape"][0][1]
        reference_points_cam[..., 1] /= img_metas[0]["img_shape"][0][0]

        bev_mask = (
            bev_mask
            & (reference_points_cam[..., 1:2] > 0.0)
            & (reference_points_cam[..., 1:2] < 1.0)
            & (reference_points_cam[..., 0:1] < 1.0)
            & (reference_points_cam[..., 0:1] > 0.0)
        )
        if digit_version(TORCH_VERSION) >= digit_version("1.8"):
            bev_mask = torch.nan_to_num(bev_mask)
        else:
            bev_mask = bev_mask.new_tensor(np.nan_to_num(bev_mask.cpu().numpy()))

        reference_points_cam = reference_points_cam.permute(2, 1, 3, 0, 4)
        bev_mask = bev_mask.permute(2, 1, 3, 0, 4).squeeze(-1)

        return reference_points_cam, bev_mask


    # This function must use fp32!!!
    @force_fp32(apply_to=("reference_points", "img_metas"))
    def point_sampling_distortion(self, reference_points, pc_range, img_metas):

        lidar2cam = []
        cam_intrinsic = []
        cam_optim_intrinsic = []
        cam_distortion = []
        for img_meta in img_metas:
            lidar2cam.append(img_meta["lidar2cam"])
            cam_intrinsic.append(img_meta["cam_intrinsic"])
            cam_optim_intrinsic.append(img_meta["cam_optim_intrinsic"])
            cam_distortion.append(img_meta["cam_distortion"])
        lidar2cam = np.asarray(lidar2cam)
        cam_intrinsic = np.asarray(cam_intrinsic)
        cam_optim_intrinsic = np.asarray(cam_optim_intrinsic)
        cam_distortion = np.asarray(cam_distortion)
        lidar2cam = reference_points.new_tensor(lidar2cam)  # (B, N_cam, 4, 4)
        lidar2cam_R = lidar2cam[:, :, :3, :3]
        lidar2cam_T = lidar2cam[:, :, :3, 3]
        cam_intrinsic = reference_points.new_tensor(cam_intrinsic)  # (B, N_cam, 3, 3)
        cam_optim_intrinsic = reference_points.new_tensor(cam_optim_intrinsic)  # (B, N_cam, 3, 3)
        cam_distortion = reference_points.new_tensor(cam_distortion)  # (B, N_cam, 5)

        reference_points = reference_points.clone()    # (B, D, num_query, 3)
        pc_range_t = reference_points.new_tensor(pc_range)
        ref_3d = reference_points * (pc_range_t[3:6] - pc_range_t[0:3]) + pc_range_t[0:3]

        # --- Transform lidar points into every camera frame ---
        # ref_3d:      (B, D, Q, 3)
        # lidar2cam_R: (B, C, 3, 3)
        # result:      (B, C, D, Q, 3)
        cam_points = (
            torch.einsum("bdqj,bcij->bcdqi", ref_3d, lidar2cam_R)
            + lidar2cam_T[:, :, None, None, :]
        )
        bev_mask = cam_points[..., 2] > 1e-5  # (B, C, D, Q)

        # --- Normalised camera coordinates ---
        z = cam_points[..., 2:3].clamp(min=1e-8)  # (B, C, D, Q, 1)
        x = cam_points[..., 0:1] / z
        y = cam_points[..., 1:2] / z

        # Helper: extract a scalar from (B, C, 3, 3) matrices and reshape to
        # (B, C, 1, 1, 1) so it broadcasts with (B, C, D, Q, 1).
        def _cam(mat, i, j):
            return mat[:, :, i, j][:, :, None, None, None]

        # --- Ideal (undistorted) projection for visibility masking ---
        u_ideal = _cam(cam_optim_intrinsic, 0, 0) * x + _cam(cam_optim_intrinsic, 0, 2)
        v_ideal = _cam(cam_optim_intrinsic, 1, 1) * y + _cam(cam_optim_intrinsic, 1, 2)
        points_2d_ideal = torch.cat([u_ideal, v_ideal], dim=-1)  # (B, C, D, Q, 2)

        points_2d_ideal = points_2d_ideal / points_2d_ideal.new_tensor([
            img_metas[0]["img_shape"][0][1], img_metas[0]["img_shape"][0][0]])

        bev_mask = (
            bev_mask
            & (points_2d_ideal[..., 0] < 1.)
            & (points_2d_ideal[..., 0] > 0.)
            & (points_2d_ideal[..., 1] < 1.)
            & (points_2d_ideal[..., 1] > 0.)
        )

        # --- Distorted projection (OpenCV distortion model) ---
        # distortion: (B, C, 5) -> each coeff: (B, C, 1, 1, 1)
        k1 = cam_distortion[..., 0][:, :, None, None, None]
        k2 = cam_distortion[..., 1][:, :, None, None, None]
        p1 = cam_distortion[..., 2][:, :, None, None, None]
        p2 = cam_distortion[..., 3][:, :, None, None, None]
        k3 = cam_distortion[..., 4][:, :, None, None, None]

        r2 = x * x + y * y
        r4 = r2 * r2
        r6 = r4 * r2

        # Radial distortion factor
        radial = 1.0 + k1 * r2 + k2 * r4 + k3 * r6

        # Tangential distortion
        xy = x * y
        x_dist = x * radial + 2.0 * p1 * xy + p2 * (r2 + 2.0 * x * x)
        y_dist = y * radial + p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * xy

        # Apply original intrinsic
        u = _cam(cam_intrinsic, 0, 0) * x_dist + _cam(cam_intrinsic, 0, 2)
        v = _cam(cam_intrinsic, 1, 1) * y_dist + _cam(cam_intrinsic, 1, 2)
        points_2d = torch.cat([u, v], dim=-1)  # (B, C, D, Q, 2)

        points_2d = points_2d / points_2d.new_tensor([
            img_metas[0]["img_shape"][0][1], img_metas[0]["img_shape"][0][0]])

        bev_mask = (
            bev_mask
            & (points_2d[..., 0] < 1.)
            & (points_2d[..., 0] > 0.)
            & (points_2d[..., 1] < 1.)
            & (points_2d[..., 1] > 0.)
        )
        points_2d[~bev_mask] = 0

        # --- Rearrange to output layout ---
        # (B, C, D, Q, 2) -> (C, B, Q, D, 2)
        points_2d = points_2d.permute(1, 0, 3, 2, 4)
        # (B, C, D, Q)   -> (C, B, Q, D)
        bev_mask = bev_mask.permute(1, 0, 3, 2)

        return points_2d, bev_mask


    @auto_fp16()
    def forward(
        self,
        bev_query,
        key,
        value,
        *args,
        bev_h=None,
        bev_w=None,
        bev_pos=None,
        spatial_shapes=None,
        level_start_index=None,
        valid_ratios=None,
        prev_bev=None,
        shift=0.0,
        img_metas=None,
        **kwargs,
    ):
        """Forward function for `TransformerDecoder`.
        Args:
            bev_query (Tensor): Input BEV query with shape
                `(num_query, bs, embed_dims)`.
            key & value (Tensor): Input multi-cameta features with shape
                (num_cam, num_value, bs, embed_dims)
            reference_points (Tensor): The reference
                points of offset. has shape
                (bs, num_query, 4) when as_two_stage,
                otherwise has shape ((bs, num_query, 2).
            valid_ratios (Tensor): The radios of valid
                points on the feature map, has shape
                (bs, num_levels, 2)
        Returns:
            Tensor: Results with shape [1, num_query, bs, embed_dims] when
                return_intermediate is `False`, otherwise it has shape
                [num_layers, num_query, bs, embed_dims].
        """

        output = bev_query
        intermediate = []

        bs = bev_query.size(1)
        device = bev_query.device
        dtype = bev_query.dtype

        ref_3d = self.get_reference_points(
            bev_h,
            bev_w,
            self.pc_range[5] - self.pc_range[2],
            self.num_points_in_pillar,
            dim="3d",
            bs=bs,
            device=device,
            dtype=dtype,
        )
        ref_2d = self.get_reference_points(
            bev_h,
            bev_w,
            dim="2d",
            bs=bs,
            device=device,
            dtype=dtype,
        )

        if self.use_distortion_sampling:
            reference_points_cam, bev_mask = self.point_sampling_distortion(
                ref_3d, self.pc_range, img_metas
            )
        else:
            reference_points_cam, bev_mask = self.point_sampling(
                ref_3d, self.pc_range, img_metas
            )

        # NOTE: we fix the shift_ref_2d to be the same as the ref_2d
        shift_ref_2d = ref_2d.clone()
        shift_ref_2d += shift[:, None, None, :]

        # (num_query, bs, embed_dims) -> (bs, num_query, embed_dims)
        bev_query = bev_query.permute(1, 0, 2)
        bev_pos = bev_pos.permute(1, 0, 2)
        bs, len_bev, num_bev_level, _ = ref_2d.shape
        if prev_bev is not None:
            prev_bev = prev_bev.permute(1, 0, 2)
            prev_bev = torch.stack([prev_bev, bev_query], 1).reshape(
                bs * 2, len_bev, -1
            )
            hybird_ref_2d = torch.stack([shift_ref_2d, ref_2d], 1).reshape(
                bs * 2, len_bev, num_bev_level, 2
            )
        else:
            hybird_ref_2d = torch.stack([ref_2d, ref_2d], 1).reshape(
                bs * 2, len_bev, num_bev_level, 2
            )

        for lid, layer in enumerate(self.layers):
            output = layer(
                bev_query,
                key,
                value,
                *args,
                bev_pos=bev_pos,
                ref_2d=hybird_ref_2d,
                ref_3d=ref_3d,
                bev_h=bev_h,
                bev_w=bev_w,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                reference_points_cam=reference_points_cam,
                bev_mask=bev_mask,
                prev_bev=prev_bev,
                **kwargs,
            )

            bev_query = output
            if self.return_intermediate:
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return output
