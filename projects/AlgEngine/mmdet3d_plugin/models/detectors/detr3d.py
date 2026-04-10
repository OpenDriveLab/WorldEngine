import torch
from mmcv.runner import auto_fp16, force_fp32

from mmdet3d.core import bbox3d2result
from mmdet3d.models.detectors.mvx_two_stage import MVXTwoStageDetector
from mmdet.models import DETECTORS


@DETECTORS.register_module()
class Detr3D(MVXTwoStageDetector):
    """Detr3D."""

    def __init__(
        self,
        pts_voxel_layer=None,
        pts_voxel_encoder=None,
        pts_middle_encoder=None,
        pts_fusion_layer=None,
        img_backbone=None,
        pts_backbone=None,
        img_neck=None,
        pts_neck=None,
        pts_bbox_head=None,
        img_roi_head=None,
        img_rpn_head=None,
        train_cfg=None,
        test_cfg=None,
        pretrained=None,
    ):
        super(Detr3D, self).__init__(
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

    def extract_img_feat(self, img, img_metas):
        """Extract features of images."""
        B = img.size(0)
        if img is not None:
            input_shape = img.shape[-2:]
            # update real input shape of each single img
            for img_meta in img_metas:
                img_meta.update(input_shape=input_shape)

            if img.dim() == 5 and img.size(0) == 1:
                img.squeeze_()
            elif img.dim() == 5 and img.size(0) > 1:
                B, N, C, H, W = img.size()
                img = img.view(B * N, C, H, W)
            img_feats = self.img_backbone(img)
            if isinstance(img_feats, dict):
                img_feats = list(img_feats.values())
        else:
            return None
        if self.with_img_neck:
            img_feats = self.img_neck(img_feats)
        img_feats_reshaped = []
        for img_feat in img_feats:
            BN, C, H, W = img_feat.size()
            img_feats_reshaped.append(img_feat.view(B, int(BN / B), C, H, W))
        return img_feats_reshaped

    @auto_fp16(apply_to=("img", "points"), out_fp32=True)
    def extract_feat(self, img, points, img_metas):
        """Extract features from images and points."""
        img_feats = self.extract_img_feat(img, img_metas)
        if hasattr(self, "pts_voxel_encoder") and self.pts_voxel_encoder is not None:
            voxels, num_points, coors = self.voxelize(points)

            voxel_features = self.pts_voxel_encoder(voxels, num_points, coors)
            batch_size = coors[-1, 0] + 1
            point_features = self.pts_middle_encoder(voxel_features, coors, batch_size)
            point_features = self.pts_backbone(point_features)
            if self.with_pts_neck:
                point_features = self.pts_neck(point_features)
        else:
            point_features = None
        return img_feats, point_features

    def forward_pts_train(
        self,
        img_feats,
        pts_feats,
        gt_bboxes_3d,
        gt_labels_3d,
        img_metas,
        gt_bboxes_ignore=None,
    ):
        """Forward function for point cloud branch.
        Args:
            img_feats (list[torch.Tensor]): Features of point cloud branch
            pts_feats (list[torch.Tensor]): Features of point cloud branch
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`]): Ground truth
                boxes for each sample.
            gt_labels_3d (list[torch.Tensor]): Ground truth labels for
                boxes of each sampole
            img_metas (list[dict]): Meta information of samples.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                boxes to be ignored. Defaults to None.
        Returns:
            dict: Losses of each branch.
        """
        outs = self.pts_bbox_head(img_feats, pts_feats, img_metas)
        additional_labels = {}

        # loss for the traffic classification
        if self.pts_bbox_head.traffic_keys is not None:
            for key in self.pts_bbox_head.traffic_keys:
                label_tmp = (
                    torch.stack(
                        [img_meta[key].data for img_meta in img_metas],
                        axis=0,
                    )
                    .to(img_feats[-1].device)
                    .float()
                )
                additional_labels[key] = label_tmp

        loss_inputs = [
            gt_bboxes_3d,
            gt_labels_3d,
            additional_labels,
            outs,
        ]
        losses = self.pts_bbox_head.loss(*loss_inputs)
        return losses

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

    def forward_train(
        self,
        points=None,
        img_metas=None,
        gt_bboxes_3d=None,
        gt_labels_3d=None,
        gt_labels=None,
        gt_bboxes=None,
        img=None,
        proposals=None,
        gt_bboxes_ignore=None,
        img_depth=None,
        img_mask=None,
    ):
        """Forward training function.
        Args:
            points (list[torch.Tensor], optional): Points of each sample.
                Defaults to None.
            img_metas (list[dict], optional): Meta information of each sample.
                Defaults to None.
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`], optional):
                Ground truth 3D boxes. Defaults to None.
            gt_labels_3d (list[torch.Tensor], optional): Ground truth labels
                of 3D boxes. Defaults to None.
            gt_labels (list[torch.Tensor], optional): Ground truth labels
                of 2D boxes in images. Defaults to None.
            gt_bboxes (list[torch.Tensor], optional): Ground truth 2D boxes in
                images. Defaults to None.
            img (torch.Tensor optional): Images of each sample with shape
                (N, C, H, W). Defaults to None.
            proposals ([list[torch.Tensor], optional): Predicted proposals
                used for training Fast RCNN. Defaults to None.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                2D boxes in images to be ignored. Defaults to None.
        Returns:
            dict: Losses of different branches.
        """
        img_feats, point_feats = self.extract_feat(
            img=img, points=points, img_metas=img_metas
        )
        losses = dict()
        losses_pts = self.forward_pts_train(
            img_feats,
            point_feats,
            gt_bboxes_3d,
            gt_labels_3d,
            img_metas,
            gt_bboxes_ignore,
        )
        losses.update(losses_pts)
        return losses

    def forward_test(self, img_metas, img=None, points=None, **kwargs):
        for var, name in [(img_metas, "img_metas")]:
            if not isinstance(var, list):
                raise TypeError("{} must be a list, but got {}".format(name, type(var)))
        img = [img] if img is None else img
        points = [points] if points is None else points
        return self.simple_test(img_metas[0], img[0], points[0], **kwargs)

    def simple_test_pts(
        self, img_feats, point_feats, img_metas, points=None, rescale=False
    ):
        """Test function of point cloud branch."""
        outs = self.pts_bbox_head(img_feats, point_feats, img_metas)
        bbox_list = self.pts_bbox_head.get_bboxes(outs, img_metas, rescale=rescale)
        bbox_results = [
            bbox3d2result(bboxes, scores, labels)
            for bboxes, scores, labels in bbox_list
        ]

        # extract traffic detection results
        traffic_results = {}
        if self.pts_bbox_head.traffic_keys is not None:
            for key in self.pts_bbox_head.traffic_keys:
                traffic_results[key] = outs[key].cpu()

        return bbox_results, traffic_results

    def simple_test(self, img_metas, img=None, points=None, rescale=False):
        """Test function without augmentaiton."""
        img_feats, point_feats = self.extract_feat(
            img=img, points=points, img_metas=img_metas
        )

        bbox_list = [dict() for i in range(len(img_metas))]
        (
            bbox_pts,
            traffic_results,
        ) = self.simple_test_pts(img_feats, point_feats, img_metas, rescale=rescale)

        # (XW) to handle the case of no traffic detection results to output
        # i.e., only outputing for the bounding box detection
        try:
            light_inrange_list = traffic_results["light_inrange"]
            light_hazard_list = traffic_results["light_hazard"]
            sign_inrange_list = traffic_results["sign_inrange"]
        except KeyError:
            # print('no traffic classification module')
            light_inrange_list = [-1]
            light_hazard_list = [-1]
            sign_inrange_list = [-1]

        # (XW) to handle the case of no bounding box results to output
        # i.e., only outputing for the traffic part
        if len(bbox_pts) == 0:
            bbox_pts = [-1]

        for result_dict, pts_bbox, light_inrange, light_hazard, sign_inrange in zip(
            bbox_list,
            bbox_pts,
            light_inrange_list,
            light_hazard_list,
            sign_inrange_list,
        ):
            result_dict["pts_bbox"] = pts_bbox
            
            # only valid when there are traffic classification
            try:
                result_dict["light_inrange"] = light_inrange
                result_dict["light_hazard"] = light_hazard
                result_dict["sign_inrange"] = sign_inrange
            except KeyError:
                print('no traffic classification module')

        return bbox_list

    def aug_test_pts(self, feats, img_metas, rescale=False):
        feats_list = []
        for j in range(len(feats[0])):
            feats_list_level = []
            for i in range(len(feats)):
                feats_list_level.append(feats[i][j])
            feats_list.append(torch.stack(feats_list_level, -1).mean(-1))
        outs = self.pts_bbox_head(feats_list, img_metas)
        bbox_list = self.pts_bbox_head.get_bboxes(outs, img_metas, rescale=rescale)
        bbox_results = [
            bbox3d2result(bboxes, scores, labels)
            for bboxes, scores, labels in bbox_list
        ]
        return bbox_results

    def aug_test(self, img_metas, imgs=None, rescale=False):
        """Test function with augmentaiton."""
        img_feats = self.extract_feats(img_metas, imgs)
        img_metas = img_metas[0]
        bbox_list = [dict() for i in range(len(img_metas))]
        bbox_pts = self.aug_test_pts(img_feats, img_metas, rescale)
        for result_dict, pts_bbox in zip(bbox_list, bbox_pts):
            result_dict["pts_bbox"] = pts_bbox
        return bbox_list
