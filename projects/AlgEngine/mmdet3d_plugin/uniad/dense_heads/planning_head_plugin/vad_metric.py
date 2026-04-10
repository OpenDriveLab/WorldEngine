# ---------------------------------------------------------------------------------#
# UniAD: Planning-oriented Autonomous Driving (https://arxiv.org/abs/2212.10156)  #
# Source code: https://github.com/OpenDriveLab/UniAD                              #
# Copyright (c) OpenDriveLab. All rights reserved.                                #
# ---------------------------------------------------------------------------------#

import torch
import copy
import torch.nn as nn
import numpy as np
from skimage.draw import polygon
from scipy.ndimage import zoom
from pytorch_lightning.metrics.metric import Metric
from .uniad_utils import calculate_birds_eye_view_parameters, gen_dx_bx


class PlanningMetric_UniAD(Metric):
    def __init__(
        self,
        n_future=6,
        compute_on_step: bool = False,
    ):
        super().__init__(compute_on_step=compute_on_step)

        # low resolution: 0.5m
        dx, bx, _ = gen_dx_bx(
            [-50.0, 50.0, 0.5], [-50.0, 50.0, 0.5], [-10.0, 10.0, 20.0]
        )
        dx, bx = dx[:2], bx[:2]
        self.dx = nn.Parameter(dx, requires_grad=False)
        self.bx = nn.Parameter(bx, requires_grad=False)
        _, _, self.bev_dimension = calculate_birds_eye_view_parameters(
            [-50.0, 50.0, 0.5], [-50.0, 50.0, 0.5], [-10.0, 10.0, 20.0]
        )
        self.bev_dimension = self.bev_dimension.numpy()

        # high resolution: 0.1m
        dx_hr, bx_hr, _ = gen_dx_bx(
            [-50.0, 50.0, 0.1], [-50.0, 50.0, 0.1], [-10.0, 10.0, 20.0]
        )
        dx_hr, bx_hr = dx_hr[:2], bx_hr[:2]
        self.dx_hr = nn.Parameter(dx_hr, requires_grad=False)
        self.bx_hr = nn.Parameter(bx_hr, requires_grad=False)
        _, _, self.bev_dimension_hr = calculate_birds_eye_view_parameters(
            [-50.0, 50.0, 0.1], [-50.0, 50.0, 0.1], [-10.0, 10.0, 20.0]
        )
        self.bev_dimension_hr = self.bev_dimension_hr.numpy()

        self.W = 1.85
        self.H = 4.084

        # add dimension of 2 for collecting the averaged numbers
        self.n_future = n_future
        self.add_state(
            "obj_col", default=torch.zeros(self.n_future + 2), dist_reduce_fx="sum"
        )
        self.add_state(
            "obj_box_col", default=torch.zeros(self.n_future + 2), dist_reduce_fx="sum"
        )
        self.add_state(
            "gt_box_col", default=torch.zeros(self.n_future + 2), dist_reduce_fx="sum"
        )
        self.add_state(
            "L2", default=torch.zeros(self.n_future + 2), dist_reduce_fx="sum"
        )
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def set_resolution(self, test_cfg):

        if test_cfg["high_resolution_grid"]:
            self.dx_used = self.dx_hr
            self.bx_used = self.bx_hr
            self.bev_dimension_used = self.bev_dimension_hr
        else:
            self.dx_used = self.dx
            self.bx_used = self.bx
            self.bev_dimension_used = self.bev_dimension

    def evaluate_single_coll(
        self,
        traj_ori,
        segmentation_ori,
        yaw,
        use_GT=False,
        scene_token=None,
        frame_idx=None,
        test_cfg=None,
        test_setting=None,
    ):
        """
        gt_segmentation
        traj: torch.Tensor (T, 2)
        yaw: 6
        segmentation: torch.Tensor (T, 200, 200)
        """

        traj = torch.clone(traj_ori)
        segmentation = torch.clone(segmentation_ori).cpu().numpy()
        n_future, _ = traj.shape
        trajs = traj.view(n_future, 1, 2)
        trajs[:, :, [0, 1]] = trajs[:, :, [1, 0]]  # can also change original tensor
        trajs = trajs / self.dx_used  # T x 1 x 2
        trajs = trajs.cpu().numpy()
        collision = np.full(n_future, False)  # 6
        # print("\nuse_GT", use_GT)

        # initial box coordinate based on width/height
        pts = np.array(
            [
                [-self.H / 2.0, self.W / 2.0],
                [self.H / 2.0, self.W / 2.0],
                [self.H / 2.0, -self.W / 2.0],
                [-self.H / 2.0, -self.W / 2.0],
            ]
        )  # 4 x 2
        pts[:, [0, 1]] = pts[:, [1, 0]]

        # pts_rot = np.zeros((traj_ori.shape[0], pts.shape[0], 2))  # T x 4 x 2
        for timestamp in range(n_future):
            invalid_index = np.where(segmentation[timestamp] == 255)[0].tolist()
            # print(invalid_index)
            segmentation[timestamp][invalid_index] = 0
            # print(np.where(segmentation[timestamp] == 255)[0].tolist())

            # rotate the bbox coordinate
            yaw_tmp = yaw[timestamp] + np.pi
            rot_mat = np.array(
                [
                    [np.cos(yaw_tmp), -np.sin(yaw_tmp)],
                    [np.sin(yaw_tmp), np.cos(yaw_tmp)],
                ]
            )

            pts_rot = np.dot(rot_mat, np.transpose(pts)).transpose()  # 4 x 2
            pts_rot[:, [0, 1]] = pts_rot[:, [1, 0]]

            # boundary offset
            if test_cfg["high_resolution_grid"]:
                pts_rot[:, 0] += 0.1  # 4 x 2, BEV box coordinate
            else:
                pts_rot[:, 0] += 0.5  # 4 x 2, BEV box coordinate
            pts_rot = (pts_rot - self.bx_used.cpu().numpy()) / (
                self.dx_used.cpu().numpy()
            )  # 4 x 2
            pts_rot[:, [0, 1]] = pts_rot[
                :, [1, 0]
            ]  # 4 x 2, describe the box shape offset

            # get pixel within the bbox
            rr, cc = polygon(pts_rot[:, 1], pts_rot[:, 0])  # row/columns
            rc = np.concatenate(
                [rr[:, None], cc[:, None]], axis=-1
            )  # 32 x 2, coordinate of the boxes

            # add trajecotry to the initialized box coordinate
            traj_tmp = trajs[timestamp, :, :] + rc  # 32 x 2

            # clip the box coordinate to be within the range
            row = traj_tmp[:, 0].astype(np.int32)
            row = np.clip(row, 0, self.bev_dimension_used[0] - 1)  # 32
            col = traj_tmp[:, 1].astype(np.int32)
            col = np.clip(col, 0, self.bev_dimension_used[1] - 1)  # 32

            # find the valid coordinate index
            I = np.logical_and(
                np.logical_and(row >= 0, row < self.bev_dimension_used[0]),
                np.logical_and(col >= 0, col < self.bev_dimension_used[1]),
            )

            # print('\n')
            # print(np.max(segmentation[timestamp]))
            # print(np.min(segmentation[timestamp]))

            # caculate the collision
            # check if the valid pixel within box is colliding with GT occupancy
            collision_tmp = np.any(segmentation[timestamp, row[I], col[I]])
            collision[timestamp] = collision_tmp

        result = torch.from_numpy(collision).to(device=traj.device)
        return result

    def evaluate_coll(
        self,
        trajs,
        gt_trajs,
        segmentation,
        yaw,
        scene_token=None,
        frame_idx=None,
        test_cfg=None,
        test_setting=None,
    ):
        """
        trajs: torch.Tensor (B, n_future, 2)
        gt_trajs: torch.Tensor (B, n_future, 2)
        segmentation: torch.Tensor (B, n_future, 200, 200)
        yaw: 1 x 6
        """
        B, n_future, _ = trajs.shape
        trajs = trajs * torch.tensor([-1, 1], device=trajs.device)
        gt_trajs = gt_trajs * torch.tensor([-1, 1], device=gt_trajs.device)

        obj_coll_sum = torch.zeros(n_future, device=segmentation.device)
        obj_box_coll_sum = torch.zeros(n_future, device=segmentation.device)
        gt_box_coll_sum = torch.zeros(n_future, device=segmentation.device)

        # # skip two sequence where the ego vehicle is not moving and the driver of the ego vehicle
        # # goes off the car and causing random collision with pedestrians
        # # do not skip and count these artifical collision in VAD evaluation setting
        if (
            scene_token
            not in [
                "4bb9626f13184da7a67ba1b241c19558",
                "21a7ba093614493b83838b9656b3558d",
            ]
            or test_setting is "VAD"
            or test_cfg["filter_pedestrian"]
        ):
            for i in range(B):
                ti = torch.arange(n_future, device=segmentation.device)

                gt_box_coll = self.evaluate_single_coll(
                    gt_trajs[i],
                    segmentation[i],
                    yaw[i],
                    use_GT=True,
                    scene_token=scene_token,
                    frame_idx=frame_idx,
                    test_cfg=test_cfg,
                    test_setting=test_setting,
                )
                # print(i)
                # print(gt_box_coll)
                gt_box_coll_sum += gt_box_coll.long()
                # print(gt_box_coll_sum)

                # check if the coordinate is valid within the boundary
                xx, yy = trajs[i, :, 0], trajs[i, :, 1]
                yi = ((yy - self.bx_used[0]) / self.dx_used[0]).long()  # 6
                xi = ((xx - self.bx_used[1]) / self.dx_used[1]).long()
                m1 = torch.logical_and(
                    torch.logical_and(yi >= 0, yi < self.bev_dimension_used[0]),
                    torch.logical_and(xi >= 0, xi < self.bev_dimension_used[1]),
                )  # 6

                # filter the timestamp that has collision in GT trajectory data, i.e., false positive
                # print('\nnew frame')
                # print('before ', m1)
                m1 = torch.logical_and(m1, torch.logical_not(gt_box_coll))
                # print('after', m1)

                # BUG fix: XW, remove the invalid index with 255, bad number in GT segmentation
                obj_col_tmp = segmentation[i, ti[m1], yi[m1], xi[m1]].long()  # 6
                obj_col_tmp = torch.where(obj_col_tmp == 255, 0, obj_col_tmp)
                obj_coll_sum[ti[m1]] += obj_col_tmp

                m2 = torch.logical_not(gt_box_coll)
                box_coll = self.evaluate_single_coll(
                    trajs[i],
                    segmentation[i],
                    yaw[i],
                    use_GT=False,
                    scene_token=scene_token,
                    frame_idx=frame_idx,
                    test_cfg=test_cfg,
                    test_setting=test_setting,
                )
                obj_box_coll_sum[ti[m2]] += (box_coll[ti[m2]]).long()

        return obj_coll_sum, obj_box_coll_sum, gt_box_coll_sum

    def compute_L2(self, trajs, gt_trajs, gt_trajs_mask, test_cfg):
        """
        trajs: torch.Tensor (B, n_future, 3)
        gt_trajs: torch.Tensor (B, n_future, 3)
        unit is the actual meter
        """

        error = torch.sqrt(
            (((trajs[:, :, :2] - gt_trajs[:, :, :2]) ** 2) * gt_trajs_mask).sum(dim=-1)
        )

        return error

    def update(
        self,
        trajs_ori,
        gt_trajs_ori,
        gt_trajs_mask,
        segmentation,
        segmentations_finegrained=None,
        vehicles_finegrained=None,
        pedestrains_finegrained=None,
        test_cfg=None,
        test_setting=None,
        scene_token=None,
        frame_idx=None,
    ):
        """
        trajs: torch.Tensor (B, T+1, 2)
        gt_trajs: torch.Tensor (B, T+1, 3)
        gt_trajs_mask: B x T+1 x 2
        segmentation: torch.Tensor (B, T+1, 200, 200)
        """

        # copy the original data
        # otherwise the planned box is xy inversed, causing collision
        # because the underlying function update() will be called twice
        # see _forward_full_state_update() in metric.py under torchmetrics
        # trajs = trajs_ori
        # gt_trajs = gt_trajs_ori[:, :, :2]  # 1 x 6 x 2, only trajectory part
        trajs = torch.clone(trajs_ori)
        gt_trajs = torch.clone(
            gt_trajs_ori[:, :, :2]
        )  # 1 x 6 x 2, only trajectory part

        # the original eval metric does not take into account of the ego yaw to calculate collision
        if test_cfg["use_ego_orientation"]:
            assert gt_trajs_ori.size() == (1, 6, 3), "error, yaw not contained"
            yaw = (
                torch.clone(gt_trajs_ori[:, :, -1]).cpu().numpy()
            )  # 1 x 6, the rotation part
        else:
            yaw = np.zeros((1, 6))
        assert trajs.shape == gt_trajs.shape

        # if any of the future frame is invalid
        # equal to "fut_valid_flag" in VAD
        any_frame_is_invalid = True if torch.sum(1 - gt_trajs_mask) > 0 else False

        # skip this frame if any of the future frame is invalid, i.e., near the end of sequence
        if test_cfg["masking_if_one_fut_frame_bad"] and any_frame_is_invalid:
            return

        if test_cfg["filter_pedestrian"]:
            if test_cfg["high_resolution_grid"]:
                segmentation_used = vehicles_finegrained
            # original UniAD eval, low-resolution vehicle only
            else:
                segmentation_used = segmentation
        else:
            # best eval, high-resolution vehicle+pedestrians+barrier
            if test_cfg["high_resolution_grid"]:
                segmentation_used = segmentations_finegrained
            # original VAD eval, low-resolution vehicle+pedestrains
            else:
                # pedestrains = pedestrains_finegrained[0].cpu().numpy()  # 6 x 1000 x 1000
                # pedestrains = zoom(pedestrains, (1, 0.2, 0.2))    # 6 x 200 x 200
                # pedestrains = torch.from_numpy(pedestrains)[None, ...].to(segmentation.device)  # 1 x 6 x 200 x 200
                # segmentation_used = torch.maximum(segmentation, pedestrains)    # 200 x 200

                # in this code the pedestrian is not high-resolution, also
                # the max value is 1, so we can directly use logical_or
                segmentation_used = torch.logical_or(
                    segmentation, pedestrains_finegrained
                )

        # Change coordinate to match the segmentation, left/right flip
        trajs[..., 0] = -trajs[..., 0]
        gt_trajs[..., 0] = -gt_trajs[..., 0]

        # calculate L2
        L2_error_tmp = self.compute_L2(
            trajs, gt_trajs, gt_trajs_mask, test_cfg=test_cfg
        )  # 1 x 6
        L2_error_tmp = L2_error_tmp.sum(
            dim=0
        )  # 6, measured in actual meters, error measure at every individual timestamp

        if test_cfg["double_time_averaging"]:
            L2_error_tmp = time_averaging(L2_error_tmp)

        L2_error = torch.zeros(8).to(L2_error_tmp)
        L2_error[:6] = L2_error_tmp  # the error at every timestamp
        # add the average error for 1s/2s/3s (UniAD used in paper)
        L2_error[6] = (L2_error_tmp[1] + L2_error_tmp[3] + L2_error_tmp[5]) / 3
        # add the average error for all timestamps
        L2_error[7] = torch.mean(L2_error_tmp)

        # calculate collision
        obj_coll_sum = torch.zeros(8).to(L2_error_tmp)
        obj_box_coll_sum = torch.zeros(8).to(L2_error_tmp)
        gt_box_coll_sum = torch.zeros(8).to(L2_error_tmp)
        (
            obj_coll_sum_tmp,
            obj_box_coll_sum_tmp,
            gt_box_coll_sum_tmp,
        ) = self.evaluate_coll(
            trajs[:, :, :2],
            gt_trajs[:, :, :2],
            segmentation_used,
            yaw,
            scene_token=scene_token,
            frame_idx=frame_idx,
            test_cfg=test_cfg,
            test_setting=test_setting,
        )  # 6

        if test_cfg["double_time_averaging"]:
            obj_coll_sum_tmp = time_averaging(obj_coll_sum_tmp)
        if test_cfg["double_time_averaging"]:
            obj_box_coll_sum_tmp = time_averaging(obj_box_coll_sum_tmp)
        if test_cfg["double_time_averaging"]:
            gt_box_coll_sum_tmp = time_averaging(gt_box_coll_sum_tmp)

        obj_coll_sum[:6] = obj_coll_sum_tmp
        obj_coll_sum[6] = (
            obj_coll_sum_tmp[1] + obj_coll_sum_tmp[3] + obj_coll_sum_tmp[5]
        ) / 3
        obj_coll_sum[7] = torch.mean(obj_coll_sum_tmp)
        obj_box_coll_sum[:6] = obj_box_coll_sum_tmp
        obj_box_coll_sum[6] = (
            obj_box_coll_sum_tmp[1] + obj_box_coll_sum_tmp[3] + obj_box_coll_sum_tmp[5]
        ) / 3
        obj_box_coll_sum[7] = torch.mean(obj_box_coll_sum_tmp)
        gt_box_coll_sum[:6] = gt_box_coll_sum_tmp
        gt_box_coll_sum[6] = (
            gt_box_coll_sum_tmp[1] + gt_box_coll_sum_tmp[3] + gt_box_coll_sum_tmp[5]
        ) / 3
        gt_box_coll_sum[7] = torch.mean(gt_box_coll_sum_tmp)

        # aggregate over the data samples
        self.obj_col += obj_coll_sum
        self.obj_box_col += obj_box_coll_sum
        self.gt_box_col += gt_box_coll_sum
        self.L2 += L2_error
        self.total += len(trajs)  # add one per eval for ego

    def compute(self):
        return {
            "obj_col": self.obj_col / self.total,
            "obj_box_col": self.obj_box_col / self.total,
            "gt_box_col": self.gt_box_col / self.total,
            "L2": self.L2 / self.total,
        }


def time_averaging(error):

    error = copy.deepcopy(error)

    # aggregated average error, up to a timestamp
    error_upto1s = torch.mean(error[:2])
    error_upto2s = torch.mean(error[:4])
    error_upto3s = torch.mean(error[:6])

    # assign averaged error
    error[1] = error_upto1s
    error[3] = error_upto2s
    error[5] = error_upto3s

    # duplicated value in order to compute average for convenience
    error[0] = error_upto1s
    error[2] = error_upto2s
    error[4] = error_upto3s

    return error
