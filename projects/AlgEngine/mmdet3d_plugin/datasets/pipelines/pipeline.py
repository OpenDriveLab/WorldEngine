import mmcv
import numpy as np
import torch
from mmcv.parallel import DataContainer as DC
from mmdet3d.core.bbox import BaseInstance3DBoxes
from mmdet3d.core.points import BasePoints
from mmdet3d.datasets.pipelines import DefaultFormatBundle
from mmdet.datasets.builder import PIPELINES
from mmdet.datasets.pipelines import to_tensor
from nuscenes.utils.data_classes import RadarPointCloud


@PIPELINES.register_module()
class FormatBundle3DTrack(DefaultFormatBundle):
    """Default formatting bundle.

    It simplifies the pipeline of formatting common fields for voxels,
    "gt_bboxes", "gt_labels", "gt_masks" and
    "gt_semantic_seg".
    These fields are formatted as follows.

    - img: (1)transpose, (2)to tensor, (3)to DataContainer (stack=True)
    - proposals: (1)to tensor, (2)to DataContainer
    - gt_bboxes: (1)to tensor, (2)to DataContainer
    - gt_bboxes_ignore: (1)to tensor, (2)to DataContainer
    - gt_labels: (1)to tensor, (2)to DataContainer
    """

    def __init__(self, with_gt=True, with_label=True):
        super(FormatBundle3DTrack, self).__init__()
        self.with_gt = with_gt
        self.with_label = with_label

    def __call__(self, results):
        """Call function to transform and format common fields in results.

        Args:
            results (dict): Result dict contains the data to convert.

        Returns:
            dict: The result dict contains the data that is formatted with
                default bundle.
        """
        # Format 3D data
        if "points" in results:
            points_cat = []
            for point in results["points"]:
                assert isinstance(point, BasePoints)
                points_cat.append(point.tensor)
            # results['points'] = DC(torch.stack(points_cat, dim=0))
            results["points"] = DC(points_cat)

        if "img" in results:
            imgs_list = results["img"]
            imgs_cat_list = []
            for imgs_frame in imgs_list:
                imgs = [img.transpose(2, 0, 1) for img in imgs_frame]
                imgs = np.ascontiguousarray(np.stack(imgs, axis=0))
                imgs_cat_list.append(to_tensor(imgs))

            results["img"] = DC(torch.stack(imgs_cat_list, dim=0), stack=True)

        # default in DefaultFormatBundle
        for key in [
            "proposals",
            "gt_bboxes",
            "gt_bboxes_ignore",
            "gt_labels",
            "gt_labels_3d",
            "attr_labels",
            "pts_instance_mask",
            "pts_semantic_mask",
            "centers2d",
            "depths",
        ]:
            if key not in results:
                continue
            if isinstance(results[key], list):
                results[key] = DC([to_tensor(res) for res in results[key]])
            else:
                results[key] = DC(to_tensor(results[key]))

        if "gt_bboxes_3d" in results:
            results["gt_bboxes_3d"] = DC(results["gt_bboxes_3d"], cpu_only=True)

        # add instance IDs
        if "instance_inds" in results:
            instance_inds = [torch.tensor(_t) for _t in results["instance_inds"]]
            results["instance_inds"] = DC(instance_inds)
        
        # add other entries
        keys = ["l2g_r_mat", "l2g_t", "radar"]
        for key in keys:
            if key in results:
                results[key] = DC(torch.tensor(results[key], dtype=torch.float))

        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f"(class_names={self.class_names}, "
        repr_str += f"with_gt={self.with_gt}, with_label={self.with_label})"
        return repr_str


@PIPELINES.register_module()
class InstanceRangeFilter(object):
    """Filter objects by the range.

    Args:
        point_cloud_range (list[float]): Point cloud range.
    """

    def __init__(self, point_cloud_range):
        self.pcd_range = np.array(point_cloud_range, dtype=np.float32)
        self.bev_range = self.pcd_range[[0, 1, 3, 4]]

    def __call__(self, input_dict):
        """Call function to filter objects by the range.

        Args:
            input_dict (dict): Result dict from loading pipeline.

        Returns:
            dict: Results after filtering, 'gt_bboxes_3d', 'gt_labels_3d' \
                keys are updated in the result dict.
        """
        gt_bboxes_3d = input_dict["gt_bboxes_3d"]
        gt_labels_3d = input_dict["gt_labels_3d"]
        instance_inds = input_dict["ann_info"]["instance_inds"]
        mask = gt_bboxes_3d.in_range_bev(self.bev_range)
        gt_bboxes_3d = gt_bboxes_3d[mask]
        # mask is a torch tensor but gt_labels_3d is still numpy array
        # using mask to index gt_labels_3d will cause bug when
        # len(gt_labels_3d) == 1, where mask=1 will be interpreted
        # as gt_labels_3d[1] and cause out of index error
        gt_labels_3d = gt_labels_3d[mask.numpy().astype(np.bool)]
        instance_inds = instance_inds[mask.numpy().astype(np.bool)]

        # limit rad to [-pi, pi]
        gt_bboxes_3d.limit_yaw(offset=0.5, period=2 * np.pi)
        input_dict["gt_bboxes_3d"] = gt_bboxes_3d
        input_dict["gt_labels_3d"] = gt_labels_3d
        input_dict["ann_info"]["instance_inds"] = instance_inds

        return input_dict

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f"(point_cloud_range={self.pcd_range.tolist()})"
        return repr_str


@PIPELINES.register_module()
class ScaleMultiViewImage3D(object):

    def __init__(self, scale=1.0):
        self.scale = scale

    def __call__(self, results):
        """Call function to pad images, masks, semantic segmentation maps.
        Args:
            results (dict): Result dict from loading pipeline.
            'img': list of imgs
            'lidar2img' (list of 4x4 array)
            'intrinsic' (list of 4x4 array)
            'extrinsic' (list of 4x4 array)
        Returns:
            dict: Updated result dict.
        """
        img_shape = results["img_shape"]
        y_size = int((img_shape[0] * self.scale) // 32) * 32
        x_size = int((img_shape[1] * self.scale) // 32) * 32
        y_scale = y_size * 1.0 / img_shape[0]
        x_scale = x_size * 1.0 / img_shape[1]
        scale_factor = np.eye(4)
        scale_factor[0, 0] *= x_scale
        scale_factor[1, 1] *= y_scale
        for key in ["img"]:
            result_img = [
                mmcv.imresize(img, (x_size, y_size), return_scale=False)
                for img in results[key]
            ]
            results[key] = result_img
            results["lidar2img"] = [scale_factor @ l2i for l2i in results["lidar2img"]]
            results["cam_intrinsic"] = [scale_factor[:3, :3] @ cam_intrinsic for cam_intrinsic in results["cam_intrinsic"]]
            if "cam_optim_intrinsic" in results:
                results["cam_optim_intrinsic"] = [scale_factor[:3, :3] @ cam_optim_intrinsic for cam_optim_intrinsic in results["cam_optim_intrinsic"]]

        results["scale_factor"] = self.scale
        results["img_shape"] = [img.shape for img in result_img]
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f"(size={self.size}, "
        return repr_str


@PIPELINES.register_module()
class LoadRadarPointsMultiSweeps(object):
    """Load radar points from multiple sweeps.

    This is usually used for nuScenes dataset to utilize previous sweeps.

    Args:
        sweeps_num (int): Number of sweeps. Defaults to 10.
        load_dim (int): Dimension number of the loaded points. Defaults to 5.
        use_dim (list[int]): Which dimension to use. Defaults to [0, 1, 2, 4].
        file_client_args (dict): Config dict of file clients, refer to
            https://github.com/open-mmlab/mmcv/blob/master/mmcv/fileio/file_client.py
            for more details. Defaults to dict(backend='disk').
        pad_empty_sweeps (bool): Whether to repeat keyframe when
            sweeps is empty. Defaults to False.
        remove_close (bool): Whether to remove close points.
            Defaults to False.
        test_mode (bool): If test_model=True used for testing, it will not
            randomly sample sweeps but select the nearest N frames.
            Defaults to False.
    """

    def __init__(
        self,
        load_dim=18,
        use_dim=[0, 1, 2, 3, 4],
        sweeps_num=3,
        file_client_args=dict(backend="disk"),
        max_num=300,
        pc_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
        test_mode=False,
    ):
        self.load_dim = load_dim
        self.use_dim = use_dim
        self.sweeps_num = sweeps_num
        self.file_client_args = file_client_args.copy()
        self.file_client = None
        self.max_num = max_num
        self.test_mode = test_mode
        self.pc_range = pc_range

    def _load_points(self, pts_filename):
        """Private function to load point clouds data.

        Args:
            pts_filename (str): Filename of point clouds data.

        Returns:
            np.ndarray: An array containing point clouds data.
            [N, 18]
        """
        radar_obj = RadarPointCloud.from_file(pts_filename)

        # [18, N]
        points = radar_obj.points

        return points.transpose().astype(np.float32)

    def _pad_or_drop(self, points):
        """
        points: [N, 18]
        """

        num_points = points.shape[0]

        if num_points == self.max_num:
            masks = np.ones((num_points, 1), dtype=points.dtype)

            return points, masks

        if num_points > self.max_num:
            points = np.random.permutation(points)[: self.max_num, :]
            masks = np.ones((self.max_num, 1), dtype=points.dtype)

            return points, masks

        if num_points < self.max_num:
            zeros = np.zeros(
                (self.max_num - num_points, points.shape[1]), dtype=points.dtype
            )
            masks = np.ones((num_points, 1), dtype=points.dtype)

            points = np.concatenate((points, zeros), axis=0)
            masks = np.concatenate((masks, zeros.copy()[:, [0]]), axis=0)

            return points, masks

    def __call__(self, results):
        """Call function to load multi-sweep point clouds from files.

        Args:
            results (dict): Result dict containing multi-sweep point cloud \
                filenames.

        Returns:
            dict: The result dict containing the multi-sweep points data. \
                Added key and value are described below.

                - points (np.ndarray | :obj:`BasePoints`): Multi-sweep point \
                    cloud arrays.
        """
        radars_dict = results["radar"]

        points_sweep_list = []
        for key, sweeps in radars_dict.items():
            if len(sweeps) < self.sweeps_num:
                idxes = list(range(len(sweeps)))
            else:
                idxes = list(range(self.sweeps_num))

            ts = sweeps[0]["timestamp"] * 1e-6
            for idx in idxes:
                sweep = sweeps[idx]

                points_sweep = self._load_points(sweep["data_path"])
                points_sweep = np.copy(points_sweep).reshape(-1, self.load_dim)

                timestamp = sweep["timestamp"] * 1e-6
                time_diff = ts - timestamp
                time_diff = np.ones((points_sweep.shape[0], 1)) * time_diff

                # velocity compensated by the ego motion in sensor frame
                velo_comp = points_sweep[:, 8:10]
                velo_comp = np.concatenate(
                    (velo_comp, np.zeros((velo_comp.shape[0], 1))), 1
                )
                velo_comp = velo_comp @ sweep["sensor2lidar_rotation"].T
                velo_comp = velo_comp[:, :2]

                # velocity in sensor frame
                velo = points_sweep[:, 6:8]
                velo = np.concatenate((velo, np.zeros((velo.shape[0], 1))), 1)
                velo = velo @ sweep["sensor2lidar_rotation"].T
                velo = velo[:, :2]

                points_sweep[:, :3] = (
                    points_sweep[:, :3] @ sweep["sensor2lidar_rotation"].T
                )
                points_sweep[:, :3] += sweep["sensor2lidar_translation"]

                points_sweep_ = np.concatenate(
                    [
                        points_sweep[:, :6],
                        velo,
                        velo_comp,
                        points_sweep[:, 10:],
                        time_diff,
                    ],
                    axis=1,
                )
                points_sweep_list.append(points_sweep_)

        points = np.concatenate(points_sweep_list, axis=0)

        points = points[:, self.use_dim]

        points[:, 0:1] = (points[:, 0:1] - self.pc_range[0]) / (
            self.pc_range[3] - self.pc_range[0]
        )
        points[:, 1:2] = (points[:, 1:2] - self.pc_range[1]) / (
            self.pc_range[4] - self.pc_range[1]
        )
        points[:, 2:3] = (points[:, 2:3] - self.pc_range[2]) / (
            self.pc_range[5] - self.pc_range[2]
        )

        if self.max_num > 0:
            points, mask = self._pad_or_drop(points)
        else:
            mask = np.ones((points.shape[0], points.shape[1]), dtype=points.dtype)

        points = np.concatenate((points, mask), axis=-1)

        results["radar"] = points.astype(np.float32)
        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        return f"{self.__class__.__name__}(sweeps_num={self.sweeps_num})"
