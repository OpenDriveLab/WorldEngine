# nuScenes dev-kit.
# Code written by Freddy Boulton, Eric Wolff 2020.
""" Implementation of metrics used in the nuScenes prediction challenge. """
import abc
from typing import List, Dict, Any, Tuple

import numpy as np
import math
from scipy import interpolate

# from .compute_metrics import Prediction
from .data_classes import Prediction
from nuscenes.map_expansion.map_api import NuScenesMap
from nuscenes.prediction import PredictHelper
from nuscenes.prediction.input_representation.static_layers import load_all_maps
from nuscenes.map_expansion import arcline_path_utils


def returns_2d_array(function):
    """Makes sure that the metric returns an array of shape [batch_size, num_modes]."""

    def _returns_array(*args, **kwargs):
        result = function(*args, **kwargs)

        if isinstance(result, (int, float)):
            result = np.array([[result]])

        elif result.ndim == 1:
            result = np.expand_dims(result, 0)

        return result

    return _returns_array


@returns_2d_array
def mean_distances(
    stacked_trajs: np.ndarray, stacked_ground_truth: np.ndarray
) -> np.ndarray:
    """
    Efficiently compute mean L2 norm between trajectories and ground truths (pairwise over states).
    :param stacked_trajs: Array of [batch_size, num_modes, horizon_length, state_dim].
    :param stacked_ground_truth: Array of [batch_size, num_modes, horizon_length, state_dim].
    :return: Array of mean L2 norms as [batch_size, num_modes].
    """
    return np.mean(
        np.linalg.norm(stacked_trajs - stacked_ground_truth, axis=-1), axis=-1
    )


@returns_2d_array
def max_distances(
    stacked_trajs: np.ndarray, stacked_ground_truth: np.ndarray
) -> np.ndarray:
    """
    Efficiently compute max L2 norm between trajectories and ground truths (pairwise over states).
    :pram stacked_trajs: Array of shape [num_modes, horizon_length, state_dim].
    :pram stacked_ground_truth: Array of [num_modes, horizon_length, state_dim].
    :return: Array of max L2 norms as [num_modes].
    """
    return np.max(
        np.linalg.norm(stacked_trajs - stacked_ground_truth, axis=-1), axis=-1
    )


@returns_2d_array
def final_distances(
    stacked_trajs: np.ndarray, stacked_ground_truth: np.ndarray
) -> np.ndarray:
    """
    Efficiently compute the L2 norm between the last points in the trajectory.
    :param stacked_trajs: Array of shape [num_modes, horizon_length, state_dim].
    :param stacked_ground_truth: Array of shape [num_modes, horizon_length, state_dim].
    :return: mean L2 norms between final points. Array of shape [num_modes].
    """
    # We use take to index the elements in the last dimension so that we can also
    # apply this function for a batch
    diff_of_last = (
        np.take(stacked_trajs, [-1], -2).squeeze()
        - np.take(stacked_ground_truth, [-1], -2).squeeze()
    )
    return np.linalg.norm(diff_of_last, axis=-1)


@returns_2d_array
def miss_max_distances(
    stacked_trajs: np.ndarray, stacked_ground_truth: np.ndarray, tolerance: float
) -> np.array:
    """
    Efficiently compute 'miss' metric between trajectories and ground truths.
    :param stacked_trajs: Array of shape [num_modes, horizon_length, state_dim].
    :param stacked_ground_truth: Array of shape [num_modes, horizon_length, state_dim].
    :param tolerance: max distance (m) for a 'miss' to be True.
    :return: True iff there was a 'miss.' Size [num_modes].
    """
    return max_distances(stacked_trajs, stacked_ground_truth) >= tolerance


@returns_2d_array
def rank_metric_over_top_k_modes(
    metric_results: np.ndarray, mode_probabilities: np.ndarray, ranking_func: str
) -> np.ndarray:
    """
    Compute a metric over all trajectories ranked by probability of each trajectory.
    :param metric_results: 2-dimensional array of shape [batch_size, num_modes].
    :param mode_probabilities: 2-dimensional array of shape [batch_size, num_modes].
    :param ranking_func: Either 'min' or 'max'. How you want to metrics ranked over the top
            k modes.
    :return: Array of shape [num_modes].
    """

    if ranking_func == "min":
        func = np.minimum.accumulate
    elif ranking_func == "max":
        func = np.maximum.accumulate
    else:
        raise ValueError(
            f"Parameter ranking_func must be one of min or max. Received {ranking_func}"
        )

    p_sorted = np.flip(mode_probabilities.argsort(axis=-1), axis=-1)
    indices = np.indices(metric_results.shape)

    sorted_metrics = metric_results[indices[0], p_sorted]

    return func(sorted_metrics, axis=-1)


def miss_rate_top_k(
    stacked_trajs: np.ndarray,
    stacked_ground_truth: np.ndarray,
    mode_probabilities: np.ndarray,
    tolerance: float,
) -> np.ndarray:
    """Compute the miss rate over the top k modes."""

    miss_rate = miss_max_distances(stacked_trajs, stacked_ground_truth, tolerance)
    return rank_metric_over_top_k_modes(miss_rate, mode_probabilities, "min")


def min_ade_k(
    stacked_trajs: np.ndarray,
    stacked_ground_truth: np.ndarray,
    mode_probabilities: np.ndarray,
) -> np.ndarray:
    """Compute the min ade over the top k modes."""

    ade = mean_distances(stacked_trajs, stacked_ground_truth)
    return rank_metric_over_top_k_modes(ade, mode_probabilities, "min")


def min_fde_k(
    stacked_trajs: np.ndarray,
    stacked_ground_truth: np.ndarray,
    mode_probabilities: np.ndarray,
) -> np.ndarray:
    """Compute the min fde over the top k modes."""

    fde = final_distances(stacked_trajs, stacked_ground_truth)
    return rank_metric_over_top_k_modes(fde, mode_probabilities, "min")


def stack_ground_truth(ground_truth: np.ndarray, num_modes: int) -> np.ndarray:
    """
    Make k identical copies of the ground truth to make computing the metrics across modes
    easier.
    :param ground_truth: Array of shape [horizon_length, state_dim].
    :param num_modes: number of modes in prediction.
    :return: Array of shape [num_modes, horizon_length, state_dim].
    """
    return np.repeat(np.expand_dims(ground_truth, 0), num_modes, axis=0)


class SerializableFunction(abc.ABC):
    """Function that can be serialized/deserialized to/from json."""

    @abc.abstractmethod
    def serialize(self) -> Dict[str, Any]:
        pass

    @property
    @abc.abstractmethod
    def name(
        self,
    ) -> str:
        pass


class Aggregator(SerializableFunction):
    """Function that can aggregate many metrics across predictions."""

    @abc.abstractmethod
    def __call__(self, array: np.ndarray, **kwargs) -> List[float]:
        pass


class RowMean(Aggregator):
    def __call__(self, array: np.ndarray, **kwargs) -> np.ndarray:
        return array.mean(axis=0).tolist()

    def serialize(self) -> Dict[str, Any]:
        return {"name": self.name}

    @property
    def name(
        self,
    ) -> str:
        return "RowMean"


class Metric(SerializableFunction):
    @abc.abstractmethod
    def __call__(self, ground_truth: np.ndarray, prediction: Prediction) -> np.ndarray:
        pass

    @property
    @abc.abstractmethod
    def aggregators(
        self,
    ) -> List[Aggregator]:
        pass

    @property
    @abc.abstractmethod
    def shape(
        self,
    ) -> str:
        pass


def desired_number_of_modes(results: np.ndarray, k_to_report: List[int]) -> np.ndarray:
    """Ensures we return len(k_to_report) values even when results has less modes than what we want."""
    return results[:, [min(k, results.shape[1]) - 1 for k in k_to_report]]


class MinADEK(Metric):
    def __init__(self, k_to_report: List[int], aggregators: List[Aggregator]):
        """
        Computes the minimum average displacement error over the top k predictions.
        :param k_to_report:  Will report the top k result for the k in this list.
        :param aggregators: How to aggregate the results across the dataset.
        """
        super().__init__()
        self.k_to_report = k_to_report
        self._aggregators = aggregators

    def __call__(self, ground_truth: np.ndarray, prediction: Prediction) -> np.ndarray:
        ground_truth = stack_ground_truth(ground_truth, prediction.number_of_modes)
        results = min_ade_k(
            prediction.prediction, ground_truth, prediction.probabilities
        )
        return desired_number_of_modes(results, self.k_to_report)

    def serialize(self) -> Dict[str, Any]:
        return {
            "k_to_report": self.k_to_report,
            "name": self.name,
            "aggregators": [agg.serialize() for agg in self.aggregators],
        }

    @property
    def aggregators(
        self,
    ) -> List[Aggregator]:
        return self._aggregators

    @property
    def name(self):
        return "MinADEK"

    @property
    def shape(self):
        return len(self.k_to_report)


class MinFDEK(Metric):
    def __init__(self, k_to_report, aggregators: List[Aggregator]):
        """
        Computes the minimum final displacement error over the top k predictions.
        :param k_to_report:  Will report the top k result for the k in this list.
        :param aggregators: How to aggregate the results across the dataset.
        """
        super().__init__()
        self.k_to_report = k_to_report
        self._aggregators = aggregators

    def __call__(self, ground_truth: np.ndarray, prediction: Prediction) -> np.ndarray:
        ground_truth = stack_ground_truth(ground_truth, prediction.number_of_modes)
        results = min_fde_k(
            prediction.prediction, ground_truth, prediction.probabilities
        )
        return desired_number_of_modes(results, self.k_to_report)

    def serialize(self) -> Dict[str, Any]:
        return {
            "k_to_report": self.k_to_report,
            "name": self.name,
            "aggregators": [agg.serialize() for agg in self.aggregators],
        }

    @property
    def aggregators(
        self,
    ) -> List[Aggregator]:
        return self._aggregators

    @property
    def name(self):
        return "MinFDEK"

    @property
    def shape(self):
        return len(self.k_to_report)


class MissRateTopK(Metric):
    def __init__(
        self,
        k_to_report: List[int],
        aggregators: List[Aggregator],
        tolerance: float = 2.0,
    ):
        """
        If any point in the prediction is more than tolerance meters from the ground truth, it is a miss.
        This metric computes the fraction of predictions that are misses over the top k most likely predictions.
        :param k_to_report: Will report the top k result for the k in this list.
        :param aggregators: How to aggregate the results across the dataset.
        :param tolerance: Threshold to consider if a prediction is a hit or not.
        """
        self.k_to_report = k_to_report
        self._aggregators = aggregators
        self.tolerance = tolerance

    def __call__(self, ground_truth: np.ndarray, prediction: Prediction) -> np.ndarray:
        ground_truth = stack_ground_truth(ground_truth, prediction.number_of_modes)
        results = miss_rate_top_k(
            prediction.prediction,
            ground_truth,
            prediction.probabilities,
            self.tolerance,
        )
        return desired_number_of_modes(results, self.k_to_report)

    def serialize(self) -> Dict[str, Any]:
        return {
            "k_to_report": self.k_to_report,
            "name": "MissRateTopK",
            "aggregators": [agg.serialize() for agg in self.aggregators],
            "tolerance": self.tolerance,
        }

    @property
    def aggregators(
        self,
    ) -> List[Aggregator]:
        return self._aggregators

    @property
    def name(self):
        return f"MissRateTopK_{self.tolerance}"

    @property
    def shape(self):
        return len(self.k_to_report)


class OffRoadRate(Metric):
    def __init__(self, helper: PredictHelper, aggregators: List[Aggregator]):
        """
        The OffRoadRate is defined as the fraction of trajectories that are not entirely contained
        in the drivable area of the map.
        :param helper: Instance of PredictHelper. Used to determine the map version for each prediction.
        :param aggregators: How to aggregate the results across the dataset.
        """
        self._aggregators = aggregators
        self.helper = helper
        self.drivable_area_polygons = self.load_drivable_area_masks(helper)
        self.pixels_per_meter = 10
        self.number_of_points = 200

    @staticmethod
    def load_drivable_area_masks(helper: PredictHelper) -> Dict[str, np.ndarray]:
        """
        Loads the polygon representation of the drivable area for each map.
        :param helper: Instance of PredictHelper.
        :return: Mapping from map_name to drivable area polygon.
        """

        maps: Dict[str, NuScenesMap] = load_all_maps(helper)

        masks = {}
        for map_name, map_api in maps.items():
            masks[map_name] = map_api.get_map_mask(
                patch_box=None,
                patch_angle=0,
                layer_names=["drivable_area"],
                canvas_size=None,
            )[0]

        return masks

    @staticmethod
    def interpolate_path(
        mode: np.ndarray, number_of_points: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Interpolate trajectory with a cubic spline if there are enough points."""

        # interpolate.splprep needs unique points.
        # We use a loop as opposed to np.unique because
        # the order of the points must be the same
        seen = set()
        ordered_array = []
        for row in mode:
            row_tuple = tuple(row)
            if row_tuple not in seen:
                seen.add(row_tuple)
                ordered_array.append(row_tuple)

        new_array = np.array(ordered_array)

        unique_points = np.atleast_2d(new_array)

        if unique_points.shape[0] <= 3:
            return unique_points[:, 0], unique_points[:, 1]
        else:
            knots, _ = interpolate.splprep(
                [unique_points[:, 0], unique_points[:, 1]], k=3, s=0.1
            )
            x_interpolated, y_interpolated = interpolate.splev(
                np.linspace(0, 1, number_of_points), knots
            )
            return x_interpolated, y_interpolated

    def __call__(
        self,
        ground_truth: np.ndarray,
        prediction: Prediction,
        ground_truth_mask: np.ndarray,
        ground_truth_yaw: np.ndarray,
    ) -> np.ndarray:
        """
        Computes the fraction of modes in prediction that are not entirely contained in the drivable area.
        :param ground_truth:                    1 x 6 x 3, global coordinate, xyz
        :param prediction: Model prediction.    1 x 6 x 3, global coordinate, xyz
        :param ground_truth_mask:               1 x 6 x 2
        :param ground_truth_yaw:                1 x 6 x 3, lidar coordinate, xy yaw
        :return: Array of shape (1, ) containing the fraction of modes that are not entirely contained in the
            drivable area.
        """
        map_name = self.helper.get_map_name_from_sample_token(prediction.sample)
        drivable_area = self.drivable_area_polygons[map_name]
        max_row, max_col = drivable_area.shape

        # filter future points that are invalid, i.e., 0, near the end of sequences
        ground_truth_mask = ground_truth_mask[0, :, 0]  # 6
        valid_index = np.where(ground_truth_mask == 1)[0].tolist()
        ground_truth = ground_truth[:, valid_index, :]
        prediction.prediction = prediction.prediction[:, valid_index, :]

        scene_token = prediction.scene_token
        frame_idx = prediction.frame_idx

        # check if the entire predicted horizon has any offroad, not calculated per timestamp
        # only do the check when there are at least two points
        n_violations = 0
        if prediction.prediction.shape[1] > 1:
            mode: np.ndarray  # 6 x 3
            for mode in prediction.prediction:
                # Fit a cubic spline to the trajectory and interpolate with 200 points
                try:
                    x_interpolated, y_interpolated = self.interpolate_path(
                        mode, self.number_of_points
                    )  # (200, )

                # in the case of the objects not moving, interpolation will trigger bug
                except ValueError:
                    x_interpolated = mode[:, 0]  # (6, )
                    y_interpolated = mode[:, 1]  # (6, )
                    print(prediction.scene_token)
                    print(prediction.frame_idx)
                    print(prediction.sample)
                    print(mode)


                # x coordinate -> col, y coordinate -> row
                index_row = (y_interpolated * self.pixels_per_meter).astype("int")
                index_col = (x_interpolated * self.pixels_per_meter).astype("int")

                row_out_of_bounds = np.any(index_row >= max_row) or np.any(
                    index_row < 0
                )
                col_out_of_bounds = np.any(index_col >= max_col) or np.any(
                    index_col < 0
                )
                out_of_bounds = row_out_of_bounds or col_out_of_bounds

                if out_of_bounds or not np.all(drivable_area[index_row, index_col]):
                    n_violations += 1

        return np.array([n_violations / prediction.prediction.shape[0]])

    def serialize(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "aggregators": [agg.serialize() for agg in self.aggregators],
        }

    @property
    def aggregators(
        self,
    ) -> List[Aggregator]:
        return self._aggregators

    @property
    def name(self):
        return "OffRoadRate"

    @property
    def shape(self):
        return 1


def deserialize_aggregator(config: Dict[str, Any]) -> Aggregator:
    """Helper for deserializing Aggregators."""
    if config["name"] == "RowMean":
        return RowMean()
    else:
        raise ValueError(f"Cannot deserialize Aggregator {config['name']}.")


def deserialize_metric(config: Dict[str, Any], helper: PredictHelper) -> Metric:
    """Helper for deserializing Metrics."""
    if config["name"] == "MinADEK":
        return MinADEK(
            config["k_to_report"],
            [deserialize_aggregator(agg) for agg in config["aggregators"]],
        )
    elif config["name"] == "MinFDEK":
        return MinFDEK(
            config["k_to_report"],
            [deserialize_aggregator(agg) for agg in config["aggregators"]],
        )
    elif config["name"] == "MissRateTopK":
        return MissRateTopK(
            config["k_to_report"],
            [deserialize_aggregator(agg) for agg in config["aggregators"]],
            tolerance=config["tolerance"],
        )
    elif config["name"] == "OffRoadRate":
        return OffRoadRate(
            helper, [deserialize_aggregator(agg) for agg in config["aggregators"]]
        )
    elif config["name"] == "LaneChangeRate":
        return LaneChangeRate(
            helper, [deserialize_aggregator(agg) for agg in config["aggregators"]]
        )
    elif config["name"] == "LaneChangeRate_New":
        return LaneChangeRate_New(
            helper, [deserialize_aggregator(agg) for agg in config["aggregators"]]
        )
    elif config["name"] == "OffLane":
        return OffLane(
            helper, [deserialize_aggregator(agg) for agg in config["aggregators"]]
        )
    else:
        raise ValueError(f"Cannot deserialize function {config['name']}.")


def flatten_metrics(
    results: Dict[str, Any], metrics: List[Metric]
) -> Dict[str, List[float]]:
    """
    Collapses results into a 2D table represented by a dictionary mapping the metric name to
    the metric values.
    :param results: Mapping from metric function name to result of aggregators.
    :param metrics: List of metrics in the results.
    :return: Dictionary mapping metric name to the metric value.
    """

    metric_names = {metric.name: metric for metric in metrics}

    flattened_metrics = {}

    for metric_name, values in results.items():
        metric_class = metric_names[metric_name]

        if hasattr(metric_class, "k_to_report"):
            for value, k in zip(values["RowMean"], metric_class.k_to_report):
                flattened_metrics[f"{metric_name}_{k}"] = value
        else:
            flattened_metrics[metric_name] = values["RowMean"]

    return flattened_metrics


class LaneChangeRate_New(Metric):
    def __init__(self, helper: PredictHelper, aggregators: List[Aggregator]):
        """
        The LaneChangeRate is defined as the fraction of trajectories that are not associated
        with the same lane as ground truth.
        :param helper: Instance of PredictHelper. Used to determine the map version for each prediction.
        :param aggregators: How to aggregate the results across the dataset.
        """
        self._aggregators = aggregators
        self.helper = helper
        self.debug = False
        self.drivable_area_polygons, self.maps = self.load_drivable_area_masks(
            helper, self.debug
        )
        self.pixels_per_meter = 10
        self.number_of_points = 200
        # self.maps = load_all_maps(helper)
        self._radius = 5
        self.lane_discretization_resolution = 0.5

    @staticmethod
    def load_drivable_area_masks(
        helper: PredictHelper, debug: bool
    ) -> Dict[str, np.ndarray]:
        """
        Loads the polygon representation of the drivable area for each map.
        :param helper: Instance of PredictHelper.
        :return: Mapping from map_name to drivable area polygon.
        """

        maps: Dict[str, NuScenesMap] = load_all_maps(helper)

        if debug:
            masks = {}
            for map_name, map_api in maps.items():
                # a = map_api.get_arcline_path()

                masks[map_name] = map_api.get_map_mask(
                    patch_box=None,
                    patch_angle=0,
                    layer_names=["drivable_area"],
                    canvas_size=None,
                )[0]
        else:
            return None, maps

        return masks, maps

    def _get_closest_lanes(self, map_api, x, y, radius):
        lanes = map_api.get_records_in_radius(x, y, radius, ["lane", "lane_connector"])
        lanes = lanes["lane"] + lanes["lane_connector"]

        discrete_points = map_api.discretize_lanes(lanes, 0.5)

        # find the two most closest lane IDs, index 0 is the closest
        min_distances = [np.inf, np.inf]
        closest_lane_ids = ["", ""]
        for lane_id, points in discrete_points.items():
            distance = np.linalg.norm(np.array(points)[:, :2] - [x, y], axis=1).min()
            if distance <= min_distances[1]:
                # closest
                if distance <= min_distances[0]:
                    min_distances[1], closest_lane_ids[1] = (
                        min_distances[0],
                        closest_lane_ids[0],
                    )
                    min_distances[0], closest_lane_ids[0] = distance, lane_id
                # 2nd closest
                else:
                    min_distances[1], closest_lane_ids[1] = distance, lane_id

        # find the projected points onto the two closest lanes and their distance
        if closest_lane_ids[0] != "":
            closest_pose_on_lane0, _ = arcline_path_utils.project_pose_to_lane(
                (x, y), map_api.get_arcline_path(closest_lane_ids[0])
            )
        else:
            closest_pose_on_lane0 = [0, 0]
        if closest_lane_ids[1] != "":
            closest_pose_on_lane1, _ = arcline_path_utils.project_pose_to_lane(
                (x, y), map_api.get_arcline_path(closest_lane_ids[1])
            )
        else:
            closest_pose_on_lane1 = [0, 0]
        # print(closest_pose_on_lane0)
        # print(closest_pose_on_lane1)

        # if the distance of two projected points is very small, meaning that the two lanes
        # might have a strong overlap, i.e., the boundary lane
        dist = math.sqrt(
            (closest_pose_on_lane0[0] - closest_pose_on_lane1[0]) ** 2
            + (closest_pose_on_lane0[1] - closest_pose_on_lane1[1]) ** 2
        )
        # print(dist)

        return closest_lane_ids, dist

    def __call__(
        self,
        ground_truth: np.ndarray,
        prediction: Prediction,
        ground_truth_mask: np.ndarray,
        ground_truth_yaw: np.ndarray,
    ) -> np.ndarray:
        """
        Computes the fraction of the points that are not associated the same lane as grond truth.
        :param ground_truth:                    1 x 6 x 3, global coordinate, xyz
        :param prediction: Model prediction.    1 x 6 x 3, global coordinate, xyz
        :param ground_truth_mask:               1 x 6 x 2
        :param ground_truth_yaw:                1 x 6 x 3, lidar coordinate, xy yaw
        :return: Array of shape (1, ) containing the fraction of the points that are not associated the same lane
            as grond truth.
        """
        map_name = self.helper.get_map_name_from_sample_token(prediction.sample)
        map_api = self.maps[map_name]
        scene_token = prediction.scene_token
        frame_idx = prediction.frame_idx

        # filter future points that are invalid, i.e., 0, near the end of sequences
        ground_truth_mask = ground_truth_mask[0, :, 0]  # 6
        valid_index = np.where(ground_truth_mask == 1)[0].tolist()
        ground_truth = ground_truth[:, valid_index, :]
        prediction.prediction = prediction.prediction[:, valid_index, :]

        def _compute_lane_id(point):
            return map_api.get_closest_lane(x=point[0], y=point[1], radius=self._radius)

        # compute the ground truth lane association.
        # consider both the lane in front and in the back
        ground_truth = ground_truth[0, :, :2]  # 6 x 2
        num_points = ground_truth.shape[0]
        gt_ids = []
        for idx in range(num_points):
            gt_id = _compute_lane_id(ground_truth[idx])
            gt_income = map_api.get_incoming_lane_ids(gt_id)
            gt_outcome = map_api.get_outgoing_lane_ids(gt_id)
            gt_id = [gt_id] + gt_income + gt_outcome
            # gt_id = [gt_id]
            gt_ids.append(gt_id)

        n_violations = 0
        mismatch_index = []
        pred_ids_all = []
        dists_all = []
        chosen_lane = []
        num_mode = prediction.prediction.shape[0]
        for mode in prediction.prediction:
            for idx in range(num_points):
                pred_ids, dists = self._get_closest_lanes(
                    map_api, x=mode[idx][0], y=mode[idx][1], radius=self._radius
                )
                pred_ids_all.append(pred_ids)
                dists_all.append(dists)

                matched = False

                # these two cloest lanes matched to predicted trajectories are essentially
                # the same lane as the projected points are nearly at the same location
                if dists < self.lane_discretization_resolution * 1.5:
                    for pred_id in pred_ids:
                        if pred_id in gt_ids[idx]:
                            matched = True
                            chosen_lane.append(pred_id)

                # in the case of two closest lanes do not have the same projected points
                # we consider them different lanes, then we choose the closest lane
                else:
                    if pred_ids[0] in gt_ids[idx]:
                        matched = True
                        chosen_lane.append(pred_ids[0])

                if not matched:
                    n_violations += 1
                    mismatch_index.append(idx)

        # to handle the edge case where it has 0 points, e.g., near the end of sequence
        if num_points > 0:
            error_rate = np.array([n_violations / (num_points * num_mode)])
        else:
            error_rate = 0

        return error_rate

    def serialize(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "aggregators": [agg.serialize() for agg in self.aggregators],
        }

    @property
    def aggregators(
        self,
    ) -> List[Aggregator]:
        return self._aggregators

    @property
    def name(self):
        return "LaneChangeRate_New"

    @property
    def shape(self):
        return 1


class LaneChangeRate(Metric):
    def __init__(self, helper: PredictHelper, aggregators: List[Aggregator]):
        """
        The LaneChangeRate is defined as the fraction of trajectories that are not associated
        with the same lane as ground truth.
        :param helper: Instance of PredictHelper. Used to determine the map version for each prediction.
        :param aggregators: How to aggregate the results across the dataset.
        """
        self._aggregators = aggregators
        self.helper = helper
        self.debug = False
        self.drivable_area_polygons, self.maps = self.load_drivable_area_masks(
            helper, self.debug
        )
        self.pixels_per_meter = 10
        self.number_of_points = 200
        self._radius = 5

    @staticmethod
    def load_drivable_area_masks(
        helper: PredictHelper, debug: bool
    ) -> Dict[str, np.ndarray]:
        """
        Loads the polygon representation of the drivable area for each map.
        :param helper: Instance of PredictHelper.
        :return: Mapping from map_name to drivable area polygon.
        """

        maps: Dict[str, NuScenesMap] = load_all_maps(helper)

        if debug:
            masks = {}
            for map_name, map_api in maps.items():
                # a = map_api.get_closest_lane()

                masks[map_name] = map_api.get_map_mask(
                    patch_box=None,
                    patch_angle=0,
                    layer_names=["drivable_area"],
                    canvas_size=None,
                )[0]
        else:
            return None, maps

        return masks, maps

    def __call__(
        self,
        ground_truth: np.ndarray,
        prediction: Prediction,
        ground_truth_mask: np.ndarray,
    ) -> np.ndarray:
        """
        Computes the fraction of the points that are not associated the same lane as grond truth.
        :param ground_truth:                    1 x 6 x 3, global coordinate, xyz
        :param prediction: Model prediction.    1 x 6 x 3, global coordinate, xyz
        :param ground_truth_mask:               1 x 6 x 2
        :param ground_truth_yaw:                1 x 6 x 3, lidar coordinate, xy yaw
        :return: Array of shape (1, ) containing the fraction of the points that are not associated the same lane
            as grond truth.
        """
        map_name = self.helper.get_map_name_from_sample_token(prediction.sample)
        map_api = self.maps[map_name]
        scene_token = prediction.scene_token
        frame_idx = prediction.frame_idx

        # filter future points that are invalid, i.e., 0, near the end of sequences
        ground_truth_mask = ground_truth_mask[0, :, 0]  # 6
        valid_index = np.where(ground_truth_mask == 1)[0].tolist()
        ground_truth = ground_truth[:, valid_index, :]
        prediction.prediction = prediction.prediction[:, valid_index, :]

        def _compute_lane_id(point):
            return map_api.get_closest_lane(x=point[0], y=point[1], radius=self._radius)

        # compute the ground truth lane association.
        ground_truth = ground_truth[0, :, :2]  # 6 x 2
        num_points = ground_truth.shape[0]
        gt_ids = []
        for idx in range(num_points):
            gt_id = _compute_lane_id(ground_truth[idx])
            gt_ids.append(gt_id)

        n_violations = 0
        pred_ids = []
        mismatch_index = []
        num_mode = prediction.prediction.shape[0]
        for mode in prediction.prediction:
            for idx in range(num_points):
                pred_id = _compute_lane_id(mode[idx])
                pred_ids.append(pred_id)
                if gt_ids[idx] != pred_id:
                    n_violations += 1
                    mismatch_index.append(idx)

        # to handle the edge case where it has 0 points, e.g., near the end of sequence
        if num_points > 0:
            error_rate = np.array([n_violations / (num_points * num_mode)])
        else:
            error_rate = 0

        return error_rate

    def serialize(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "aggregators": [agg.serialize() for agg in self.aggregators],
        }

    @property
    def aggregators(
        self,
    ) -> List[Aggregator]:
        return self._aggregators

    @property
    def name(self):
        return "LaneChangeRate"

    @property
    def shape(self):
        return 1


# import numpy as np
import heapq  # For efficiently managing a min-heap

# from nuscenes.map_expansion import arcline_path_utils

class OffLane(Metric):
    def __init__(
        self,
        helper: PredictHelper,
        aggregators: List[Aggregator],
        radius=5.0,
        num_nearest_lanes=3,
        gt_dist_th=1.0,
        pred_dist_th=0.5,
        max_radius=np.pi / 2,
    ):
        """
        The OffLane is defined as the fraction of trajectories that are not associated
        with the same lane as ground truth.
        :param helper: Instance of PredictHelper. Used to determine the map version for each prediction.
        :param aggregators: How to aggregate the results across the dataset.
        """

        self._aggregators = aggregators
        self.helper = helper
        self.pixels_per_meter = 10
        self.number_of_points = 200
        self.debug = False
        self.drivable_area_polygons, self.maps = self.load_drivable_area_masks(
            helper, self.debug
        )
        # self.maps = load_all_maps(helper)
        self._radius = radius
        self._num_nearest_lanes = num_nearest_lanes
        self._gt_dist_th = gt_dist_th
        self._pred_dist_th = pred_dist_th
        self._max_radius = max_radius

    @staticmethod
    def load_drivable_area_masks(
        helper: PredictHelper, debug: bool
    ) -> Dict[str, np.ndarray]:
        """
        Loads the polygon representation of the drivable area for each map.
        :param helper: Instance of PredictHelper.
        :return: Mapping from map_name to drivable area polygon.
        """

        maps: Dict[str, NuScenesMap] = load_all_maps(helper)

        if debug:
            masks = {}
            for map_name, map_api in maps.items():
                # a = map_api.get_arcline_path()

                masks[map_name] = map_api.get_map_mask(
                    patch_box=None,
                    patch_angle=0,
                    layer_names=["drivable_area"],
                    canvas_size=None,
                )[0]
        else:
            return None, maps

        return masks, maps
    
    def _get_closest_lanes(self, map_api, x, y, dist_th):

        # discretize the lane from the map api
        lanes = map_api.get_records_in_radius(
            x, y, self._radius, ["lane", "lane_connector"]
        )
        lanes = lanes["lane"] + lanes["lane_connector"]
        discrete_points = map_api.discretize_lanes(lanes, 0.5)

        # Initialize a min-heap to store (distance, lane_id) tuples
        closest_lanes_heap = []
        for lane_id, points in discrete_points.items():
            distance = np.linalg.norm(np.array(points)[:, :2] - [x, y], axis=1).min()
            
            # Maintain a heap of size N
            if len(closest_lanes_heap) < self._num_nearest_lanes:
                heapq.heappush(closest_lanes_heap, (-distance, lane_id))
            else:
                heapq.heappushpop(closest_lanes_heap, (-distance, lane_id))

        # Extract lane IDs and distances from the heap
        closest_lanes = heapq.nsmallest(self._num_nearest_lanes, closest_lanes_heap)
        if len(closest_lanes) == 0:
            return [
                "",
            ], 0
        closest_lane_ids = [lane_id for _, lane_id in closest_lanes][::-1]
        closest_distances = [-distance for distance, _ in closest_lanes][::-1]

        # Calculate the distances for lane_{0} with lane_{1, ..., N}.
        # find the projected points onto the two closest lanes and their distance        
        pose_on_lane_0, _ = arcline_path_utils.project_pose_to_lane(
            (x, y), map_api.get_arcline_path(closest_lane_ids[0])
        )

        num_close_lane = 1
        for i in range(len(closest_lane_ids) - 1):

            # if the distance of two projected points is very small, meaning that the two lanes
            # might have a strong overlap, i.e., the boundary lane
            pose_on_lane_i, _ = arcline_path_utils.project_pose_to_lane(
                (x, y), map_api.get_arcline_path(closest_lane_ids[i + 1])
            )
            dist = abs(pose_on_lane_0[0] - pose_on_lane_i[0]) + abs(
                pose_on_lane_0[1] - pose_on_lane_i[1]
            )
            if dist <= dist_th:
                num_close_lane += 1
                
        return closest_lane_ids, num_close_lane

    def normalize_radius(self, radius):
        # Normalizing the angle to be within -pi and pi
        return np.arctan2(np.sin(radius), np.cos(radius))

    def __call__(
        self,
        ground_truth: np.ndarray,
        prediction: Prediction,
        ground_truth_mask: np.ndarray,
        ground_truth_yaw: np.ndarray,
    ) -> np.ndarray:
        """
        Computes the fraction of the points that are not associated the same lane as grond truth.
        :param ground_truth:                    1 x 6 x 3, global coordinate, xyz
        :param prediction: Model prediction.    1 x 6 x 3, global coordinate, xyz
        :param ground_truth_mask:               1 x 6 x 2
        :param ground_truth_yaw:                1 x 6 x 3, lidar coordinate, xy yaw
        :return: Array of shape (1, ) containing the fraction of the points that are not associated the same lane
            as grond truth.
        """
        map_name = self.helper.get_map_name_from_sample_token(prediction.sample)
        map_api = self.maps[map_name]
        scene_token = prediction.scene_token
        frame_idx = prediction.frame_idx

        # filter future points that are invalid, i.e., 0, near the end of sequences
        ground_truth_mask = ground_truth_mask[0, :, 0]  # 6
        valid_index = np.where(ground_truth_mask == 1)[0].tolist()
        ground_truth = ground_truth[:, valid_index, :]          # 1 x 6 x 3
        ground_truth_lidar_yaw = ground_truth_yaw[:, valid_index, :]          # 1 x 6 x 3
        prediction.prediction = prediction.prediction[:, valid_index, :]        # 1 x 6 x 3

        # compute the ground truth lane association.
        ground_truth = ground_truth[0, :, :2]  # 6 x 2
        ground_truth_lidar_yaw = ground_truth_lidar_yaw[0]  # 6 x 3

        print('\n\n')
        print(ground_truth)

        num_points = ground_truth.shape[0]
        gt_ids = []
        for idx in range(num_points):
            gt_close_ids, num_close_lanes = self._get_closest_lanes(
                map_api,
                x=ground_truth[idx][0],
                y=ground_truth[idx][1],
                dist_th=self._gt_dist_th,
            )

            # check angles
            # gt_radius = self.normalize_radius(ground_truth[idx][2])
            # print(gt_radius)
            gt_radius = self.normalize_radius(ground_truth_lidar_yaw[idx][2])
            print('gt_radius')
            print(gt_radius)

            print(ground_truth[idx][:2])

            unalign_angle_list = []
            gt_id = gt_close_ids[0]
            if num_close_lanes > 1:
                for i in range(num_close_lanes):
                    lane_point_pose, _ = arcline_path_utils.project_pose_to_lane(
                        ground_truth[idx][:2], map_api.get_arcline_path(gt_close_ids[i])
                    )

                    print('lane_point_pose')
                    print(lane_point_pose)

                    lane_point_radius = self.normalize_radius(lane_point_pose[2])
                    print('lane_point_radius')
                    print(lane_point_radius)

                    if abs(gt_radius - lane_point_radius) > self._max_radius:
                        unalign_angle_list.append(gt_close_ids[i])
                for i in range(num_close_lanes):
                    if gt_close_ids[i] not in unalign_angle_list:
                        gt_id = gt_close_ids[i]
                        break
            gt_income = map_api.get_incoming_lane_ids(gt_id)
            gt_outcome = map_api.get_outgoing_lane_ids(gt_id)
            gt_id = [gt_id] + gt_income + gt_outcome
            gt_ids.append(gt_id)

        n_violations = 0
        num_mode = prediction.prediction.shape[0]
        for mode in prediction.prediction:
            for idx in range(num_points):
                pred_ids, num_close_lanes = self._get_closest_lanes(
                    map_api, x=mode[idx][0], y=mode[idx][1], dist_th=self._pred_dist_th
                )
                matched = False
                for i in range(num_close_lanes):
                    if pred_ids[i] in gt_ids[idx]:
                        matched = True
                if not matched:
                    n_violations += 1

        # to handle the edge case where it has 0 points, e.g., near the end of sequence
        if num_points > 0:
            error_rate = np.array([n_violations / (num_points * num_mode)])
        else:
            error_rate = 0

        return error_rate

    def serialize(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "aggregators": [agg.serialize() for agg in self.aggregators],
        }

    @property
    def aggregators(
        self,
    ) -> List[Aggregator]:
        return self._aggregators

    @property
    def name(self):
        return "OffLane"

    @property
    def shape(self):
        return 1


import colorsys


def random_colors(
    num_colors: int = 20, bright: bool = True
) -> List[Tuple[float, float, float]]:
    """Generate random colors

    Note: to get visually distinct colors, generate them in HSV space then
    convert to RGB.

    Args:
        num_colors: split the space into how many different colors
        bringht: make the color more bright

    Returns
        colors: colors are in [0, 1] range
    """

    brightness = 1.0 if bright else 0.7
    hsv: List[Tuple[float, float, float]] = [
        (i / float(num_colors), 1, brightness) for i in range(num_colors)
    ]

    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))

    return colors
