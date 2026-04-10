# nuScenes dev-kit.
# Code written by Freddy Boulton, 2020.
""" Script for computing metrics for a submission to the nuscenes prediction challenge. """
# import argparse
# import json
from collections import defaultdict
from typing import List, Dict, Any

import numpy as np

# from nuscenes import NuScenes
# from nuscenes.eval.prediction.config import PredictionConfig, load_prediction_config
from .data_classes import Prediction
from nuscenes.prediction import PredictHelper

# import json
# import os
from typing import List, Dict, Any

from .lane_metric import Metric, deserialize_metric
# from nuscenes.prediction import PredictHelper



class PredictionConfig:

    def __init__(self,
                 metrics: List[Metric],
                 seconds: int = 6,
                 frequency: int = 2):
        """
        Data class that specifies the prediction evaluation settings.
        Initialized with:
        metrics: List of nuscenes.eval.prediction.metric.Metric objects.
        seconds: Number of seconds to predict for each agent.
        frequency: Rate at which prediction is made, in Hz.
        """
        self.metrics = metrics
        self.seconds = seconds
        self.frequency = frequency  # Hz

    def serialize(self) -> Dict[str, Any]:
        """ Serialize instance into json-friendly format. """

        return {'metrics': [metric.serialize() for metric in self.metrics],
                'seconds': self.seconds}

    @classmethod
    def deserialize(cls, content: Dict[str, Any], helper: PredictHelper):
        """ Initialize from serialized dictionary. """
        return cls([deserialize_metric(metric, helper) for metric in content['metrics']],
                   seconds=content['seconds'])


def compute_metrics(predictions: List[Dict[str, Any]], config: PredictionConfig) -> Dict[str, Any]:
    """
    Computes metrics from a set of predictions.
    :param predictions: List of prediction JSON objects.
    :param helper: Instance of PredictHelper that wraps the nuScenes val set.
    :param config: Config file.
    :return: Metrics. Nested dictionary where keys are metric names and value is a dictionary
        mapping the Aggregator name to the results.
    """

    n_preds = len(predictions)
    containers = {metric.name: np.zeros((n_preds, metric.shape)) for metric in config.metrics}  # 81 x 1

    # go through result for each frame
    for i, prediction_dict in enumerate(predictions):

        # create my own Prediction dict for the ego vehicle
        sample_token = prediction_dict['sample_token']
        scene_token = prediction_dict['scene_token']
        frame_idx = prediction_dict['frame_idx']

        # skip the evaluation of the first frame, which is typically noisy for all models
        # requiring the past timestamp of information as input, e.g., BEV feature
        if int(frame_idx) not in [0]:

            try:
                # in global coordinate, xyz
                prediction = prediction_dict['planning_traj_global']    # 1 x 6 x 2
                ground_truth = prediction_dict['planning_traj_gt_global'].cpu().numpy()    # 1 x 6 x 3
                
                # in lidar coordinate, xy yaw
                ground_truth_yaw = prediction_dict['planning_traj_gt'][0][0].cpu().numpy()    # 1 x 6 x 3

                ground_truth_mask = prediction_dict['planning_traj_gt_mask'].cpu().numpy()    # 1 x 6 x 2
            
            # data in numpy already
            except AttributeError:
                prediction = prediction_dict['planning_traj_global']    # 1 x 6 x 2
                ground_truth = prediction_dict['planning_traj_gt_global']    # 1 x 6 x 3
                ground_truth_yaw = prediction_dict['planning_traj_gt'][0][0]    # 1 x 6 x 3
                ground_truth_mask = prediction_dict['planning_traj_gt_mask']    # 1 x 6 x 2

            pred_dict = {
                'sample': sample_token,
                'prediction': prediction.tolist(),  # 1 x 6 x 2
                'scene_token': scene_token,     # str
                'frame_idx': frame_idx,         # int  
            }
            prediction = Prediction.deserialize(pred_dict)

            for metric in config.metrics:
                containers[metric.name][i] = metric(ground_truth, prediction, ground_truth_mask, ground_truth_yaw)

    aggregations: Dict[str, Dict[str, List[float]]] = defaultdict(dict)
    for metric in config.metrics:
        for agg in metric.aggregators:
            aggregations[metric.name][agg.name] = agg(containers[metric.name])
    return aggregations
