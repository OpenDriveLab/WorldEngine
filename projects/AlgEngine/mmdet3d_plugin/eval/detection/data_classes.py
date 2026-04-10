from typing import Dict, List, Tuple

from nuscenes.eval.detection.data_classes import DetectionConfig

# from nuscenes.eval.detection.constants import ATTRIBUTE_NAMES, TP_METRICS
# from mmdet3d_plugin.eval.detection.constants import DETECTION_NAMES_CARLA


class CustomizedDetectionConfig(DetectionConfig):
    """Inherit nuScenes DetectionConfig from nuscenes.eval.detection.data_classes.py
    but change the class names
    """

    def __init__(
        self,
        class_range: Dict[str, int],
        dist_fcn: str,
        dist_ths: List[float],
        dist_th_tp: float,
        min_recall: float,
        min_precision: float,
        max_boxes_per_sample: int,
        mean_ap_weight: int,
    ):

        assert dist_th_tp in dist_ths, "dist_th_tp must be in set of dist_ths."

        self.class_range = class_range
        self.dist_fcn = dist_fcn
        self.dist_ths = dist_ths
        self.dist_th_tp = dist_th_tp
        self.min_recall = min_recall
        self.min_precision = min_precision
        self.max_boxes_per_sample = max_boxes_per_sample
        self.mean_ap_weight = mean_ap_weight

        self.class_names = list(self.class_range.keys())
