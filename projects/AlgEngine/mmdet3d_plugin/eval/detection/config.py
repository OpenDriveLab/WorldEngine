from mmdet3d_plugin.eval.detection.data_classes import CustomizedDetectionConfig

nuplan_detection_configs = {
    "class_range": {
        "vehicle": 50,
        "pedestrian": 40,
        "bicycle": 40,
        'generic_object': 40,
        "traffic_cone": 30,
        "barrier": 30,
        'czone_sign': 30
    },
    "dist_fcn": "center_distance",
    "dist_ths": [0.5, 1.0, 2.0, 4.0],
    "dist_th_tp": 2.0,
    "min_recall": 0.1,
    "min_precision": 0.1,
    "max_boxes_per_sample": 500,
    "mean_ap_weight": 5,
    # "dataset_name": 'nuplan',
}

def config_factory_nuPlan() -> CustomizedDetectionConfig:
    """
    Creates a DetectionConfig instance that can be used to initialize a NuScenesEval instance.
    Note that this only works if the config file is located in the nuscenes/eval/detection/configs folder.
    :param configuration_name: Name of desired configuration in eval_detection_configs.
    :return: DetectionConfig instance.
    """

    cfg = CustomizedDetectionConfig.deserialize(nuplan_detection_configs)

    return cfg
