import torch
import numpy as np
import cv2
import copy
from typing import List

from mmdet3d_plugin.uniad.dense_heads.occ_head_plugin import (
    calculate_birds_eye_view_parameters,
)

from mmdet.datasets.builder import PIPELINES
import os


@PIPELINES.register_module()
class GenerateOccFlowLabels(object):
    def __init__(
        self,
        grid_conf,
        ignore_index=255,
        only_vehicle=True,
        filter_invisible=True,
        deal_instance_255=False,
        dataset_name='nusc',
    ):
        
        # relatively small resolution for training, 0.5m per cell, 200x200
        self.grid_conf = grid_conf
        (
            self.bev_resolution,
            self.bev_start_position,
            self.bev_dimension,
        ) = calculate_birds_eye_view_parameters(
            grid_conf["xbound"],
            grid_conf["ybound"],
            grid_conf["zbound"],
        )

        # more finegrained resolution for testing, 0.1m per cell, 1000x1000
        # using 0.2m per cell is perfect for GT, but prediction could go wrong a bit more
        self.grid_hr = 0.1
        (
            self.bev_resolution_finegrained,
            self.bev_start_position_finegrained,
            self.bev_dimension_finegrained,
        ) = calculate_birds_eye_view_parameters(
            [-50, 50, self.grid_hr],
            [-50, 50, self.grid_hr],
            [-10, 10, 20.0],
        )

        # convert numpy
        self.bev_resolution = self.bev_resolution.numpy()
        self.bev_start_position = self.bev_start_position.numpy()
        self.bev_dimension = self.bev_dimension.numpy()
        self.bev_resolution_finegrained = self.bev_resolution_finegrained.numpy()
        self.bev_start_position_finegrained = self.bev_start_position_finegrained.numpy()
        self.bev_dimension_finegrained = self.bev_dimension_finegrained.numpy()        
        self.spatial_extent = (grid_conf["xbound"][1], grid_conf["ybound"][1])
        self.ignore_index = ignore_index
        self.only_vehicle = only_vehicle
        self.filter_invisible = filter_invisible
        self.deal_instance_255 = deal_instance_255
        assert self.deal_instance_255 is False

        if dataset_name == 'nusc':
            class_names = [
                "car",
                "truck",
                "construction_vehicle",
                "bus",
                "trailer",
                "barrier",
                "motorcycle",
                "bicycle",
                "pedestrian",
                "traffic_cone",
            ]
            vehicle_classes = [
                "car",
                "bus",
                "construction_vehicle",
                "bicycle",
                "motorcycle",
                "truck",
                "trailer",
            ]
        else:
            class_names = ['vehicle', 'bicycle', 'pedestrian',
               'traffic_cone', 'barrier', 'czone_sign', 'generic_object']
            vehicle_classes = ['vehicle', 'bicycle']

        plan_classes = vehicle_classes + ["pedestrian"]
        pedestrian_classes = ['pedestrian']

        self.vehicle_cls_ids = np.array(
            [class_names.index(cls_name) for cls_name in vehicle_classes]
        )
        self.plan_cls_ids = np.array(
            [class_names.index(cls_name) for cls_name in plan_classes]
        )
        self.ped_cls_ids = np.array(
            [class_names.index(cls_name) for cls_name in pedestrian_classes]
        )

        if only_vehicle:
            self.filter_cls_ids = self.vehicle_cls_ids
        else:
            self.filter_cls_ids = self.plan_cls_ids

    def reframe_boxes(self, boxes, t_init, t_curr):
        l2e_r_mat_curr = t_curr["l2e_r"]
        l2e_t_curr = t_curr["l2e_t"]
        e2g_r_mat_curr = t_curr["e2g_r"]
        e2g_t_curr = t_curr["e2g_t"]

        l2e_r_mat_init = t_init["l2e_r"]
        l2e_t_init = t_init["l2e_t"]
        e2g_r_mat_init = t_init["e2g_r"]
        e2g_t_init = t_init["e2g_t"]

        # print(boxes.tensor[10])

        # to bbox under curr ego frame  # TODO: Uncomment
        boxes.rotate(l2e_r_mat_curr.T)
        boxes.translate(l2e_t_curr)

        # print(boxes.tensor[10])

        # to bbox under world frame
        boxes.rotate(e2g_r_mat_curr.T)
        boxes.translate(e2g_t_curr)

        # print(boxes.tensor[10])

        # to bbox under initial ego frame, first inverse translate, then inverse rotate
        boxes.translate(-e2g_t_init)
        m1 = np.linalg.inv(e2g_r_mat_init)
        boxes.rotate(m1.T)

        # print(boxes.tensor[10])

        # to bbox under curr ego frame, first inverse translate, then inverse rotate
        boxes.translate(-l2e_t_init)
        m2 = np.linalg.inv(l2e_r_mat_init)
        boxes.rotate(m2.T)

        return boxes


    def __call__(self, results):
        """
        # Given lidar frame bboxes for curr frame and each future frame,
        # generate segmentation, instance, centerness, offset, and fwd flow map
        """
        # Avoid ignoring obj with index = self.ignore_index
        SPECIAL_INDEX = -20

        all_gt_bboxes_3d = results["future_gt_bboxes_3d"]
        all_gt_labels_3d = results["future_gt_labels_3d"]
        all_gt_inds = results["future_gt_inds"]
        all_vis_tokens = results["future_gt_vis_tokens"]
        num_frame = len(all_gt_bboxes_3d)
        # print('num_frame\n', num_frame)

        # motion related transforms, of seq lengths
        l2e_r_mats = results["occ_l2e_r_mats"]
        l2e_t_vecs = results["occ_l2e_t_vecs"]
        e2g_r_mats = results["occ_e2g_r_mats"]
        e2g_t_vecs = results["occ_e2g_t_vecs"]

        # reference frame transform
        t_ref = dict(
            l2e_r=l2e_r_mats[0],
            l2e_t=l2e_t_vecs[0],
            e2g_r=e2g_r_mats[0],
            e2g_t=e2g_t_vecs[0],
        )

        segmentations: List[np.ndarray] = []  # L, 200 x 200
        vehicles_finegrained: List[np.ndarray] = []  # L, 500 x 500
        pedestrains_finegrained: List[np.ndarray] = []  # L, 500 x 500
        segmentations_finegrained: List[np.ndarray] = []  # L, 500 x 500
        instances = []
        gt_future_boxes = []
        gt_future_labels = []

        # debug_index = 0

        # print('\n\nnew seq')

        # num_frame is 5
        for i in range(num_frame):
            # bbox, label, index of curr frame
            gt_bboxes_3d, gt_labels_3d = all_gt_bboxes_3d[i], all_gt_labels_3d[i]
            # gt_bboxes_3d: N x 9

            # print(gt_bboxes_3d)

            ins_inds = all_gt_inds[i]
            vis_tokens = all_vis_tokens[i]

            # print(i)
            # print(gt_bboxes_3d.tensor.size())

            # yx coordinate (wh)
            if gt_bboxes_3d is None:
                # for invalid samples, no loss calculated
                segmentation = (
                    np.ones((self.bev_dimension[1], self.bev_dimension[0]))
                    * self.ignore_index
                )             
                instance = (
                    np.ones((self.bev_dimension[1], self.bev_dimension[0]))
                    * self.ignore_index
                )

                vehicle_finegrained = (
                    np.ones((self.bev_dimension_finegrained[1], self.bev_dimension_finegrained[0]))
                    * self.ignore_index
                )   
                pedestrain_finegrained = (
                    np.ones((self.bev_dimension_finegrained[1], self.bev_dimension_finegrained[0]))
                    * self.ignore_index
                )   
                segmentation_finegrained = (
                    np.ones((self.bev_dimension_finegrained[1], self.bev_dimension_finegrained[0]))
                    * self.ignore_index
                )   
            else:
                # reframe bboxes to reference frame
                t_curr = dict(
                    l2e_r=l2e_r_mats[i],
                    l2e_t=l2e_t_vecs[i],
                    e2g_r=e2g_r_mats[i],
                    e2g_t=e2g_t_vecs[i],
                )

                # print(gt_bboxes_3d[:10])
                # yx coordinate, lwh format
                ref_bboxes_3d = self.reframe_boxes(gt_bboxes_3d, t_ref, t_curr) # N x 9
                ref_bboxes_3d_all = copy.deepcopy(ref_bboxes_3d)
                ins_inds_all = copy.deepcopy(ins_inds)
                gt_labels_3d_all = copy.deepcopy(gt_labels_3d)

                gt_future_boxes.append(ref_bboxes_3d)
                gt_future_labels.append(gt_labels_3d)

                # for valid samples
                segmentation = np.zeros((self.bev_dimension[1], self.bev_dimension[0]))
                instance = np.zeros((self.bev_dimension[1], self.bev_dimension[0]))

                segmentation_finegrained = np.zeros((self.bev_dimension_finegrained[1], self.bev_dimension_finegrained[0]))
                pedestrain_finegrained = np.zeros((self.bev_dimension_finegrained[1], self.bev_dimension_finegrained[0]))
                vehicle_finegrained = np.zeros((self.bev_dimension_finegrained[1], self.bev_dimension_finegrained[0]))

                if self.only_vehicle:
                    vehicle_mask = np.isin(gt_labels_3d, self.filter_cls_ids)
                    ref_bboxes_3d = ref_bboxes_3d[vehicle_mask]
                    gt_labels_3d = gt_labels_3d[vehicle_mask]
                    ins_inds = ins_inds[vehicle_mask]
                    if vis_tokens is not None:
                        vis_tokens = vis_tokens[vehicle_mask]

                # filter for pedestrian
                pedestrain_mask = np.isin(gt_labels_3d_all, self.ped_cls_ids)
                ref_boxes_3d_ped = ref_bboxes_3d_all[pedestrain_mask]
                ins_inds_ped = ins_inds_all[pedestrain_mask]

                # filter for non-visible objects, not used now
                if self.filter_invisible:
                    assert vis_tokens is not None
                    visible_mask = (
                        vis_tokens != 1
                    )  # obj are filtered out with visibility(1) between 0 and 40%
                    ref_bboxes_3d = ref_bboxes_3d[visible_mask]
                    gt_labels_3d = gt_labels_3d[visible_mask]
                    ins_inds = ins_inds[visible_mask]
                
                # print(ref_bboxes_3d.tensor[10])
                # print(ref_bboxes_3d.corners[10])

                # valid sample and has objects
                if len(ref_bboxes_3d.tensor) > 0:

                    bbox_corners_ori = ref_bboxes_3d.corners[:, [0, 3, 7, 4], :2].numpy()
                    # print(bbox_corners[10:13])

                    # rescale the box to the BEV grid, in yx coordinate
                    # change range [-50, 50] -> [0, 200]
                    # (x - -49.75 + 0.5/2) / 0.5
                    bbox_corners = np.round(
                        (
                            bbox_corners_ori
                            - self.bev_start_position[:2]
                            + self.bev_resolution[:2] / 2.0
                        )
                        / self.bev_resolution[:2]
                    ).astype(np.int32)
                    bbox_corners_finegrained = np.round(
                        (
                            bbox_corners_ori
                            - self.bev_start_position_finegrained[:2]
                            + self.bev_resolution_finegrained[:2] / 2.0
                        )
                        / self.bev_resolution_finegrained[:2]
                    ).astype(np.int32)

                    for index, gt_ind in enumerate(ins_inds):
                        if gt_ind == self.ignore_index:
                            gt_ind = SPECIAL_INDEX  # 255 -> -20
                        poly_region = bbox_corners[index]
                        poly_region_finegrained = bbox_corners_finegrained[index]

                        cv2.fillPoly(segmentation, [poly_region], 1.0)
                        cv2.fillPoly(instance, [poly_region], int(gt_ind))
                        cv2.fillPoly(vehicle_finegrained, [poly_region_finegrained], 1.0)

                # render pedestrian in high resolution
                if len(ref_boxes_3d_ped.tensor) > 0:
                    bbox_corners_ped = ref_boxes_3d_ped.corners[:, [0, 3, 7, 4], :2].numpy()
                    bbox_corners_ped = np.round(
                        (
                            bbox_corners_ped
                            - self.bev_start_position_finegrained[:2]
                            + self.bev_resolution_finegrained[:2] / 2.0
                        )
                        / self.bev_resolution_finegrained[:2]
                    ).astype(np.int32)
                    for index, gt_ind in enumerate(ins_inds_ped):
                        if gt_ind == self.ignore_index:
                            gt_ind = SPECIAL_INDEX  # 255 -> -20
                        poly_region = bbox_corners_ped[index]
                        cv2.fillPoly(pedestrain_finegrained, [poly_region], 1.0)

                # render all things in high resolution
                if len(ref_bboxes_3d_all.tensor) > 0:
                    bbox_corners_all = ref_bboxes_3d_all.corners[:, [0, 3, 7, 4], :2].numpy()
                    bbox_corners_all = np.round(
                        (
                            bbox_corners_all
                            - self.bev_start_position_finegrained[:2]
                            + self.bev_resolution_finegrained[:2] / 2.0
                        )
                        / self.bev_resolution_finegrained[:2]
                    ).astype(np.int32)
                    for index, gt_ind in enumerate(ins_inds_all):
                        if gt_ind == self.ignore_index:
                            gt_ind = SPECIAL_INDEX  # 255 -> -20
                        poly_region = bbox_corners_all[index]
                        cv2.fillPoly(segmentation_finegrained, [poly_region], 1.0)

            segmentations.append(segmentation)  
            instances.append(instance)
            segmentations_finegrained.append(segmentation_finegrained)  
            pedestrains_finegrained.append(pedestrain_finegrained)
            vehicles_finegrained.append(vehicle_finegrained)

        # segmentation = 1 where objects are located
        segmentations = torch.from_numpy(np.stack(segmentations, axis=0)).long()    # 7 x H x W
        # print(segmentations.size())
        instances = torch.from_numpy(np.stack(instances, axis=0)).long()    # 7 x H x W
        vehicles_finegrained = torch.from_numpy(np.stack(vehicles_finegrained, axis=0)).long()    # 7 x H x W
        pedestrains_finegrained = torch.from_numpy(np.stack(pedestrains_finegrained, axis=0)).long()    # 7 x H x W
        segmentations_finegrained = torch.from_numpy(np.stack(segmentations_finegrained, axis=0)).long()    # 7 x H x W

        # generate heatmap & offset from segmentation & instance
        (
            instance_centerness,    # 7 x 1 x H x W
            instance_offset,        # 7 x 2 x H x W
            instance_flow,          # 7 x 2 x H x W
            instance_backward_flow, # 7 x 2 x H x W
        ) = self.center_offset_flow(
            instances,
            all_gt_inds,
            ignore_index=255,
        )

        invalid_mask = segmentations[:, 0, 0] == self.ignore_index
        instance_centerness[invalid_mask] = self.ignore_index

        results["gt_occ_has_invalid_frame"] = results.pop("occ_has_invalid_frame")
        results["gt_occ_img_is_valid"] = results.pop("occ_img_is_valid")
        results.update(
            {
                "gt_segmentation": segmentations,
                "gt_instance": instances,
                "gt_centerness": instance_centerness,
                "gt_offset": instance_offset,
                "gt_flow": instance_flow,
                "gt_backward_flow": instance_backward_flow,
                "gt_future_boxes": gt_future_boxes,
                "gt_future_labels": gt_future_labels,
                "segmentations_finegrained": segmentations_finegrained,
                "vehicles_finegrained": vehicles_finegrained,
                "pedestrains_finegrained": pedestrains_finegrained,
            }
        )

        return results

    def center_offset_flow(
        self, instance_img, all_gt_inds, ignore_index=255, sigma=3.0
    ):
        seq_len, h, w = instance_img.shape
        # heatmap
        center_label = torch.zeros(seq_len, 1, h, w)
        # offset from parts to centers
        offset_label = ignore_index * torch.ones(seq_len, 2, h, w)
        # future flow
        future_displacement_label = ignore_index * torch.ones(seq_len, 2, h, w)

        # backward flow
        backward_flow = ignore_index * torch.ones(seq_len, 2, h, w)

        # x is vertical displacement, y is horizontal displacement
        x, y = torch.meshgrid(
            torch.arange(h, dtype=torch.float), torch.arange(w, dtype=torch.float)
        )

        gt_inds_all = []
        for ins_inds_per_frame in all_gt_inds:
            if ins_inds_per_frame is None:
                continue
            for ins_ind in ins_inds_per_frame:
                gt_inds_all.append(ins_ind)
        gt_inds_unique = np.unique(np.array(gt_inds_all))

        # iterate over all instances across this sequence
        for instance_id in gt_inds_unique:
            instance_id = int(instance_id)
            prev_xc = None
            prev_yc = None
            prev_mask = None
            for t in range(seq_len):
                instance_mask = instance_img[t] == instance_id
                if instance_mask.sum() == 0:
                    # this instance is not in this frame
                    prev_xc = None
                    prev_yc = None
                    prev_mask = None
                    continue

                # the Bird-Eye-View center of the instance
                xc = x[instance_mask].mean()
                yc = y[instance_mask].mean()

                off_x = xc - x
                off_y = yc - y
                g = torch.exp(-(off_x ** 2 + off_y ** 2) / sigma ** 2)
                center_label[t, 0] = torch.maximum(center_label[t, 0], g)
                offset_label[t, 0, instance_mask] = off_x[instance_mask]
                offset_label[t, 1, instance_mask] = off_y[instance_mask]

                if prev_xc is not None and instance_mask.sum() > 0:
                    delta_x = xc - prev_xc
                    delta_y = yc - prev_yc
                    future_displacement_label[t - 1, 0, prev_mask] = delta_x
                    future_displacement_label[t - 1, 1, prev_mask] = delta_y
                    backward_flow[t - 1, 0, instance_mask] = -1 * delta_x
                    backward_flow[t - 1, 1, instance_mask] = -1 * delta_y

                prev_xc = xc
                prev_yc = yc
                prev_mask = instance_mask

        return center_label, offset_label, future_displacement_label, backward_flow

    def visualize_instances(self, instances, vis_root=""):
        if vis_root is not None and vis_root != "":
            os.makedirs(vis_root, exist_ok=True)

        for i, ins in enumerate(instances):
            ins_c = ins.astype(np.uint8)
            ins_c = cv2.applyColorMap(ins_c, cv2.COLORMAP_JET)  # H x W x 3

            # FLIP to match image
            ins_c = np.flip(ins_c, axis=0)  # H x W x 3            

            save_path = os.path.join(vis_root, "{}.png".format(i))
            cv2.imwrite(save_path, ins_c)

        vid_path = os.path.join(vis_root, "vid_ins.avi")
        height, width = instances[0].shape
        size = (height, width)
        v_out = cv2.VideoWriter(vid_path, cv2.VideoWriter_fourcc(*"DIVX"), 4, size)
        for i in range(len(instances)):
            ins_c = instances[i].astype(np.uint8)
            ins_c = cv2.applyColorMap(ins_c, cv2.COLORMAP_JET)
            v_out.write(ins_c)
        v_out.release()
        return
