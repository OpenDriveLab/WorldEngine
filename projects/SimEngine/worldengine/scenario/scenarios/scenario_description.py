"""
Describe attributes used for scenarios.

This is related to scripts for converting specific dataset format to
    worldengine format.

An example after nuPlan conversion is as the following:

    scenario = {

        # ===== Meta data about the scenario =====
        "id": "2021.05.12.22.00.38_veh-35_01008_01518",
        "name": "2021.05.12.22.00.38_veh-35_01008_01518",
        "dataset": "nuplan",
        "map": "us-nv-las-vegas-strip",
        "token": "d908a069f1135360",

        # number of <frames> of all trajectories and states.
        "log_length": 1020,
        "sample_rate": 10,  # subsample frame rate.
        "base_timestamp": <int>,   # UNIX timestamp of the first frame.

        "sdc_id": "ego"
        # ===== Information about the sensors calibration =====
        "cameras": {
            f"{camera_name}": {
                "channel": f"{camera_name}",
                "sensor2ego_rotation": np.array([...]),     # quaternion
                "sensor2ego_translation": np.array([...]),  # [x, y, z]
                "intrinsic": np.array([...]),               # 3x3 matrix
                "distortion": np.array([...]),              # [k1, k2, p1, p2, k3]
                "height": TODO,
                "width": TODO
            }
        }
        "lidar": {   # only LIDAR_TOP is supported now.
            "channel": f"{lidar_name}",
            "sensor2ego_rotation": np.array([...]),     # quaternion
            "sensor2ego_translation": np.array([...]),  # [x, y, z]
        }

        # ===== Trajectories of active participants, e.g. vehicles, pedestrians =====
        # a dict mapping object ID to it's state dict.
        "object_track": {
            f"{object_id}": {

                # The type string in metadrive.type.MetaDriveType
                "type": "VEHICLE" / "PEDESTRIAN" / etc,

                # The meta data dict. Store useful information about the object. type in metadata could be those from
                # different dataset
                "metadata": {
                    "type": "VEHICLE",
                    "track_length": 200,  # length of sequence.
                    "object_id": f"{nuplan_track_id}",
                    "nuplan_id": f"{nuplan_track_id}",
                    "nuplan_type": f"{nuplan_type}",
                }

                # The state dict. All values must have T elements.
                "state": {
                    "position": np.ones([200, 3], dtype=np.float32),
                    ...
                },

            },

            f"{object_id_2}": ...
        },

        # ===== States sequence of dynamics objects, e.g. traffic light =====
        # a dict mapping object ID to it's state dict.
        "dynamic_map_states": {
            f"{traffic_light_1}": {

                # The type string in metadrive.type.MetaDriveType
                "type": "TRAFFIC_LIGHT",

                # The state dict. All values must have T elements.
                "state": {
                    "object_state": np.ones([200, ], dtype=int),
                },

                # The meta data dict. Store useful information about the object
                "metadata": {
                    "type": "TRAFFIC_LIGHT",
                    "track_length": 200,
                    "object_id": f"{lane_id_of_the_object}",
                    "lane_id": f"{lane_id_of_the_object}",
                }
        }

        # ===== Map features =====
        # A dict mapping from map feature ID to a line segment
        "map_features": {
            f"{lane_id}": {
                "type": "LANE_SURFACE_STREET",

                # centerline.
                "polyline": np.array in [21, 2],  # A set of 2D points describing a line segment

                # convexhull of the centerline.
                "polygon": np.array in [N, 2] # A set of 2D points representing convexhull

                "entry_lanes": a list of entry lanes [f"{entry_lane_id}"].
                "exit_lanes": a list of exit lanes [f"{exit_lane_id}"].

                "left_neighbor": left traffic lane within the same road block,
                "right_neighbor": right traffic lane within the same road block.,
            },
            f"{lane_id_2}": ...
            ...
        }
    }
"""
import math
import os
from collections import defaultdict
from typing import Optional
import numpy as np

from worldengine.utils.type import WorldEngineObjectType


class ScenarioDescription(dict):
    """
    MetaDrive Scenario Description. It stores keys of the data dict.
    """

    # The outer dictionary structure.
    ID = "id"
    LENGTH = "log_length"
    METADATA = "metadata"
    BASE_TIMESTAMP = "base_timestamp"
    CAMERAS = "cameras"
    LIDAR = "lidar"

    OBJECT_TRACKS = "object_track"
    DYNAMIC_MAP_STATES = "dynamic_map_states"
    MAP_FEATURES = "map_features"
    FIRST_LEVEL_KEYS = {OBJECT_TRACKS, ID, DYNAMIC_MAP_STATES, MAP_FEATURES, LENGTH}

    # map feature lane keys
    POLYLINE = "polyline"
    POLYGON = "polygon"
    LEFT_NEIGHBORS = "left_neighbor"
    RIGHT_NEIGHBORS = "right_neighbor"
    ENTRY = "entry_lanes"
    EXIT = "exit_lanes"

    # object
    TYPE = "type"
    STATE = "state"
    OBJECT_ID = "object_id"
    STATE_DICT_KEYS = {TYPE, STATE, METADATA}
    ORIGINAL_ID_TO_OBJ_ID = "original_id_to_obj_id"
    OBJ_ID_TO_ORIGINAL_ID = "obj_id_to_original_id"
    #  for object position/heading
    POSITION = "position"
    HEADING = "heading"
    VALID = "valid"

    # dynamic map features.
    TRAFFIC_LIGHT_POSITION = "traffic_light_position"
    TRAFFIC_LIGHT_STATES = "traffic_light_state"
    TRAFFIC_LIGHT_LANE = "traffic_light_state"

    METADRIVE_PROCESSED = "metadrive_processed"
    TIMESTEP = "ts"
    COORDINATE = "coordinate"
    SDC_ID = "sdc_id"  # Not necessary, but can be stored in metadata.
    METADATA_KEYS = {METADRIVE_PROCESSED, COORDINATE, TIMESTEP}
    OLD_ORIGIN_IN_CURRENT_COORDINATE = "old_origin_in_current_coordinate"

    # CARLA scenario_configurations
    TRIGGER_POINTS = "trigger_points"
    ROUTE_VAR_NAME = "route_var_name"

    ALLOW_TYPES = (int, float, str, np.ndarray, dict, list, tuple, type(None), set)

    class SUMMARY:
        SUMMARY = "summary"
        OBJECT_SUMMARY = "object_summary"
        NUMBER_SUMMARY = "number_summary"

        # for each object summary
        TYPE = "type"
        OBJECT_ID = "object_id"
        TRACK_LENGTH = "track_length"
        MOVING_DIST = "moving_distance"
        VALID_LENGTH = "valid_length"
        CONTINUOUS_VALID_LENGTH = "continuous_valid_length"

        # for number summary:
        OBJECT_TYPES = "object_types"
        NUM_OBJECTS = "num_objects"
        NUM_MOVING_OBJECTS = "num_moving_objects"
        NUM_OBJECTS_EACH_TYPE = "num_objects_each_type"
        NUM_MOVING_OBJECTS_EACH_TYPE = "num_moving_objects_each_type"

        NUM_TRAFFIC_LIGHTS = "num_traffic_lights"
        NUM_TRAFFIC_LIGHT_TYPES = "num_traffic_light_types"
        NUM_TRAFFIC_LIGHTS_EACH_STEP = "num_traffic_light_each_step"

        NUM_MAP_FEATURES = "num_map_features"
        MAP_HEIGHT_DIFF = "map_height_diff"

    class DATASET:
        SUMMARY_FILE = "dataset_summary.pkl"  # dataset summary file name
        MAPPING_FILE = "dataset_mapping.pkl"  # store the relative path of summary file and each scenario

    @classmethod
    def sanity_check(cls, scenario_dict, check_self_type=False, valid_check=False):
        """Check if the input scenario dict is self-consistent and has filled required fields.

        The required high-level fields include tracks, dynamic_map_states, metadata, map_features.
        For each object, the tracks[obj_id] should at least contain type, state, metadata.
        For each object, the tracks[obj_id]['state'] should at least contain position, heading.
        For each lane in map_features, map_feature[map_feat_id] should at least contain polyline.
        For metadata, it should at least contain metadrive_processed, coordinate and timestep.
        We have more checks to ensure the consistency of the data.

        Args:
            scenario_dict: the input dict.
            check_self_type: if True, assert the input dict is a native Python dict.
            valid_check: if True, we will assert the values for a given timestep are zeros if valid=False at that
                timestep.
        """
        if check_self_type:
            assert isinstance(scenario_dict, dict)
            assert not isinstance(scenario_dict, ScenarioDescription)

        # Whether input has all required keys
        assert cls.FIRST_LEVEL_KEYS.issubset(set(scenario_dict.keys())), \
            "You lack these keys in first level: {}".format(cls.FIRST_LEVEL_KEYS.difference(set(scenario_dict.keys())))

        # Check types, only native python objects
        # This is to avoid issue in pickle deserialization
        _recursive_check_type(scenario_dict, cls.ALLOW_TYPES)

        scenario_length = scenario_dict[cls.LENGTH]

        # Check tracks data
        assert isinstance(scenario_dict[cls.OBJECT_TRACKS], dict)
        for obj_id, obj_state in scenario_dict[cls.OBJECT_TRACKS].items():
            cls._check_object_state_dict(
                obj_state, scenario_length=scenario_length, object_id=obj_id, valid_check=valid_check
            )
            # position heading check
            assert ScenarioDescription.HEADING in obj_state[ScenarioDescription.STATE
                                                            ], "heading is required for an object"
            assert ScenarioDescription.POSITION in obj_state[ScenarioDescription.STATE
                                                             ], "position is required for an object"

        # Check dynamic_map_state
        assert isinstance(scenario_dict[cls.DYNAMIC_MAP_STATES], dict)
        for obj_id, obj_state in scenario_dict[cls.DYNAMIC_MAP_STATES].items():
            cls._check_object_state_dict(obj_state, scenario_length=scenario_length, object_id=obj_id)

        # Check map features
        assert isinstance(scenario_dict[cls.MAP_FEATURES], dict)
        cls._check_map_features(scenario_dict[cls.MAP_FEATURES])

    @classmethod
    def _check_map_features(cls, map_feature):
        """Check if all lanes in the map contain the polyline (center line) feature and if they are in correct types."""
        for id, feature in map_feature.items():
            if WorldEngineObjectType.is_lane(feature[ScenarioDescription.TYPE]):
                assert ScenarioDescription.POLYLINE in feature, "No lane center line in map feature"
                assert isinstance(
                    feature[ScenarioDescription.POLYLINE], (np.ndarray, list, tuple)
                ), "lane center line is in invalid type"
            if ScenarioDescription.POLYGON in feature and ScenarioDescription.POLYLINE in feature:
                line_centroid = np.mean(feature["polyline"], axis=0)[:2]
                polygon_centroid = np.mean(feature["polygon"], axis=0)[:2]
                diff = line_centroid - polygon_centroid
                assert math.sqrt(diff[0] ** 2 + diff[1] ** 2) < 100, \
                    "The distance between centroids of polyline and polygon is greater than 100m. " \
                    "The map converter should be wrong!"

    @classmethod
    def _check_object_state_dict(cls, obj_state, scenario_length, object_id, valid_check=True):
        """Check the state dict of an object (the dynamic objects such as road users, vehicles or traffic lights).

        Args:
            obj_state: the state dict of the object.
            scenario_length: the length (# of timesteps) of the scenario.
            object_id: the ID of the object.
            valid_check: if True, we will examine the data at each timestep and see if it's non-zero when valid=False
                at that timestep.
        """
        # Check keys
        assert set(obj_state).issuperset(cls.STATE_DICT_KEYS)

        # Check type
        assert WorldEngineObjectType.has_type(
            obj_state[cls.TYPE]), "MetaDrive doesn't have this type: {}".format(obj_state[cls.TYPE])

        # Check set type
        assert obj_state[cls.TYPE] != WorldEngineObjectType.UNSET, \
            "Types should be set for objects and traffic lights"

        # Check state arrays temporal consistency
        assert isinstance(obj_state[cls.OBJECT_TRACKS], dict)
        for state_key, state_array in obj_state[cls.STATE].items():
            assert isinstance(state_array, (np.ndarray, list, tuple))
            assert len(state_array) == scenario_length

            if not isinstance(state_array, np.ndarray):
                continue

            assert state_array.ndim in [1, 2], "Haven't implemented test array with dim {} yet".format(state_array.ndim)
            if state_array.ndim == 2:
                assert state_array.shape[
                    1] != 0, "Please convert all state with dim 1 to a 1D array instead of 2D array."

            if state_key == cls.VALID and valid_check:
                assert np.sum(state_array) >= 1, "No frame valid for this object. Consider removing it"

            # check valid
            if cls.VALID in obj_state[cls.STATE] and valid_check:
                _array = state_array[..., :2] if state_key == "position" else state_array
                assert abs(np.sum(_array[np.where(obj_state[cls.STATE][cls.VALID], False, True)])) < 1e-2, \
                    "Valid array mismatches with {} array, some frames in {} have non-zero values, " \
                    "so it might be valid".format(state_key, state_key)

        # Check metadata
        assert isinstance(obj_state[cls.METADATA], dict)
        for metadata_key in (cls.TYPE, cls.OBJECT_ID):
            assert metadata_key in obj_state[cls.METADATA]

        # Check metadata alignment
        if cls.OBJECT_ID in obj_state[cls.METADATA]:
            assert obj_state[cls.METADATA][cls.OBJECT_ID] == object_id

    def to_dict(self):
        """Convert the object to a native python dict.

        Returns:
            A python dict
        """
        return dict(self)

    def get_sdc_track(self):
        """Return the object info dict for the SDC.

        Returns:
            The info dict for the SDC.
        """
        assert self.SDC_ID in self
        sdc_id = str(self[self.SDC_ID])
        return self[self.OBJECT_TRACKS][sdc_id]

    @staticmethod
    def get_object_summary(object_dict, object_id: str):
        """Summarize the information of one dynamic object.

        Args:
            object_dict: the info dict of a particular object, aka scenario['tracks'][obj_id] (not the ['state'] dict!)
            object_id: the ID of the object

        Returns:
            A dict summarizing the information of this object.
        """
        object_type = object_dict[ScenarioDescription.TYPE]
        state_dict = object_dict[ScenarioDescription.STATE]
        track = state_dict[ScenarioDescription.POSITION]
        valid_track = track[np.where(
            state_dict[ScenarioDescription.VALID].astype(int))][..., :2]
        distance = float(
            sum(np.linalg.norm(valid_track[i] - valid_track[i + 1]) for i in range(valid_track.shape[0] - 1))
        )  # total moving distance.
        valid_length = int(sum(state_dict[ScenarioDescription.VALID]))

        continuous_valid_length = 0
        for v in state_dict[ScenarioDescription.VALID]:
            if v:
                continuous_valid_length += 1
            if continuous_valid_length > 0 and not v:
                break

        return {
            ScenarioDescription.SUMMARY.TYPE: object_type,
            ScenarioDescription.SUMMARY.OBJECT_ID: str(object_id),
            ScenarioDescription.SUMMARY.TRACK_LENGTH: int(len(track)),
            ScenarioDescription.SUMMARY.MOVING_DIST: float(distance),
            ScenarioDescription.SUMMARY.VALID_LENGTH: int(valid_length),
            ScenarioDescription.SUMMARY.CONTINUOUS_VALID_LENGTH: int(continuous_valid_length)
        }

    @staticmethod
    def _calculate_num_moving_objects(scenario):
        """Calculate the number of moving objects, whose moving distance > 1m in this scenario."""
        # moving object
        number_summary_dict = {
            ScenarioDescription.SUMMARY.NUM_MOVING_OBJECTS: 0,
            ScenarioDescription.SUMMARY.NUM_MOVING_OBJECTS_EACH_TYPE: defaultdict(int)
        }
        for v in scenario[ScenarioDescription.SUMMARY.SUMMARY][
            ScenarioDescription.SUMMARY.OBJECT_SUMMARY].values():

            if v[ScenarioDescription.SUMMARY.MOVING_DIST] > 1:
                number_summary_dict[ScenarioDescription.SUMMARY.NUM_MOVING_OBJECTS] += 1
                number_summary_dict[ScenarioDescription.SUMMARY.NUM_MOVING_OBJECTS_EACH_TYPE][
                    v[ScenarioDescription.SUMMARY.TYPE]] += 1
        return number_summary_dict

    @staticmethod
    def update_summaries(scenario):
        """Update the object summary and number summary of one scenario in-place.

        Args:
            scenario: The input scenario

        Returns:
            The same scenario with the scenario['metadata']['object/number_summary'] be overwritten.
        """
        SD = ScenarioDescription

        # add agents summary
        summary_dict = {}
        for track_id, track in scenario[SD.OBJECT_TRACKS].items():
            summary_dict[track_id] = SD.get_object_summary(
                object_dict=track, object_id=track_id)

        # update object and number summary.
        scenario[SD.SUMMARY.SUMMARY] = dict()
        scenario[SD.SUMMARY.SUMMARY][SD.SUMMARY.OBJECT_SUMMARY] = summary_dict

        # count some objects occurrence
        scenario[SD.SUMMARY.SUMMARY][SD.SUMMARY.NUMBER_SUMMARY] = SD.get_number_summary(scenario)
        return scenario

    @staticmethod
    def get_number_summary(scenario):
        """Return the stats of all objects in a scenario.

        Examples:
            {'num_objects': 211,
             'object_types': {'CYCLIST', 'PEDESTRIAN', 'VEHICLE'},
             'num_objects_each_type': {'VEHICLE': 184, 'PEDESTRIAN': 25, 'CYCLIST': 2},
             'num_moving_objects': 69,
             'num_moving_objects_each_type': defaultdict(int, {'VEHICLE': 52, 'PEDESTRIAN': 15, 'CYCLIST': 2}),
             'num_traffic_lights': 8,
             'num_traffic_light_types': {'LANE_STATE_STOP', 'LANE_STATE_UNKNOWN'},
             'num_traffic_light_each_step': {'LANE_STATE_UNKNOWN': 164, 'LANE_STATE_STOP': 564},
             'num_map_features': 358,
             'map_height_diff': 2.4652252197265625}

        Args:
            scenario: The input scenario.

        Returns:
            A dict describing the number of different kinds of data.
        """
        SD = ScenarioDescription

        number_summary_dict = {}

        # object
        number_summary_dict[SD.SUMMARY.NUM_OBJECTS] = len(
            scenario[SD.OBJECT_TRACKS])
        number_summary_dict[SD.SUMMARY.OBJECT_TYPES] = \
            set(v[SD.TYPE] for v in scenario[SD.OBJECT_TRACKS].values())
        object_types_counter = defaultdict(int)
        for v in scenario[SD.OBJECT_TRACKS].values():
            object_types_counter[v[SD.TYPE]] += 1
        number_summary_dict[SD.SUMMARY.NUM_OBJECTS_EACH_TYPE] = dict(object_types_counter)

        # If object summary does not exist, fill them here
        object_summaries = {}
        for track_id, track in scenario[SD.OBJECT_TRACKS].items():
            object_summaries[track_id] = scenario.get_object_summary(object_dict=track, object_id=track_id)
        scenario[SD.SUMMARY.SUMMARY][SD.SUMMARY.OBJECT_SUMMARY] = object_summaries

        # moving object
        number_summary_dict.update(SD._calculate_num_moving_objects(scenario))

        # Number of different dynamic object states
        dynamic_object_states_types = set()
        dynamic_object_states_counter = defaultdict(int)
        for v in scenario[SD.DYNAMIC_MAP_STATES].values():
            for step_state in v[SD.STATE][SD.TRAFFIC_LIGHT_STATES]:
                if step_state is None:
                    continue
                dynamic_object_states_types.add(step_state)
                dynamic_object_states_counter[step_state] += 1
        number_summary_dict[SD.SUMMARY.NUM_TRAFFIC_LIGHTS] = \
            len(scenario[SD.DYNAMIC_MAP_STATES])
        number_summary_dict[SD.SUMMARY.NUM_TRAFFIC_LIGHT_TYPES] = \
            dynamic_object_states_types
        number_summary_dict[SD.SUMMARY.NUM_TRAFFIC_LIGHTS_EACH_STEP] = \
            dict(dynamic_object_states_counter)

        # map
        number_summary_dict[SD.SUMMARY.NUM_MAP_FEATURES] = \
            len(scenario[SD.MAP_FEATURES])
        number_summary_dict[SD.SUMMARY.MAP_HEIGHT_DIFF] = \
            SD.map_height_diff(scenario[SD.MAP_FEATURES])
        return number_summary_dict

    @staticmethod
    def sdc_moving_dist(scenario):
        """Get the moving distance of SDC in this scenario. This is useful to filter the scenario.

        Args:
            scenario: The scenario description.

        Returns:
            (float) The moving distance of SDC.
        """
        SD = ScenarioDescription
        scenario = SD(scenario)

        sdc_id = scenario[SD.SDC_ID]
        sdc_info = scenario[SD.SUMMARY.SUMMARY][SD.SUMMARY.OBJECT_SUMMARY][sdc_id]

        if SD.SUMMARY.MOVING_DIST not in sdc_info:
            sdc_info = SD.get_object_summary(object_dict=scenario.get_sdc_track(), object_id=sdc_id)

        moving_dist = sdc_info[SD.SUMMARY.MOVING_DIST]
        return moving_dist

    @staticmethod
    def get_num_objects(scenario, object_type: Optional[str] = None):
        """Return the number of objects (vehicles, pedestrians, cyclists, ...).

        Args:
            scenario: The input scenario.
            object_type: The string of the object type. If None, return the number of all objects.

        Returns:
            (int) The number of objects.
        """
        SD = ScenarioDescription
        summary = scenario[SD.SUMMARY.SUMMARY]
        if SD.SUMMARY.NUMBER_SUMMARY not in summary:
            scenario[SD.SUMMARY.SUMMARY][SD.SUMMARY.NUMBER_SUMMARY] = \
                SD.get_number_summary(scenario)
        num_summary = scenario[SD.SUMMARY.SUMMARY][SD.SUMMARY.NUMBER_SUMMARY]
        if object_type is None:
            return num_summary[SD.SUMMARY.NUM_OBJECTS]
        else:
            return num_summary[SD.SUMMARY.NUM_OBJECTS_EACH_TYPE].get(object_type, 0)

    @staticmethod
    def num_object(scenario, object_type: Optional[str] = None):
        """Return the number of objects (vehicles, pedestrians, cyclists, ...).

        Args:
            scenario: The input scenario.
            object_type: The string of the object type. If None, return the number of all objects.

        Returns:
            (int) The number of objects.
        """
        return ScenarioDescription.get_num_objects(scenario, object_type)

    @staticmethod
    def get_num_moving_objects(scenario, object_type=None):
        """Return the number of moving objects (vehicles, pedestrians, cyclists, ...).

        Args:
            scenario: The input scenario.
            object_type: The string of the object type. If None, return the number of all objects.

        Returns:
            (int) The number of moving objects.
        """
        SD = ScenarioDescription
        summary = scenario[SD.SUMMARY.SUMMARY]
        if SD.SUMMARY.NUM_MOVING_OBJECTS not in summary[SD.SUMMARY.NUMBER_SUMMARY]:
            num_summary = SD._calculate_num_moving_objects(scenario)
        else:
            num_summary = summary[SD.SUMMARY.NUMBER_SUMMARY]

        if object_type is None:
            return num_summary[SD.SUMMARY.NUM_MOVING_OBJECTS]
        else:
            return num_summary[SD.SUMMARY.NUM_MOVING_OBJECTS_EACH_TYPE].get(object_type, 0)

    @staticmethod
    def num_moving_object(scenario, object_type=None):
        """Return the number of moving objects (vehicles, pedestrians, cyclists, ...).

        Args:
            scenario: The input scenario.
            object_type: The string of the object type. If None, return the number of all objects.

        Returns:
            (int) The number of moving objects.
        """
        return ScenarioDescription.get_num_moving_objects(scenario, object_type=object_type)

    @staticmethod
    def map_height_diff(map_features, target=10):
        """Compute the maximum height difference in a map.

        Args:
            map_features: The map feature dict of a scenario.
            target: The target height difference, default to 10. If we find height difference > 10, we will return 10
                immediately. This can be used to accelerate computing if we are filtering a batch of scenarios.

        Returns:
            (float) The height difference in the map feature, or the target height difference if the diff > target.
        """
        max = -math.inf
        min = math.inf
        for feature in map_features.values():
            if not WorldEngineObjectType.is_road_line(feature[ScenarioDescription.TYPE]):
                continue
            polyline = feature[ScenarioDescription.POLYLINE]
            if len(polyline[0]) == 3:
                z = np.asarray(polyline)[..., -1]
                z_max = np.max(z)
                if z_max > max:
                    max = z_max
                z_min = np.min(z)
                if z_min < min:
                    min = z_min
            if max - min > target:
                break
        return float(max - min)

    @staticmethod
    def centralize_to_ego_car_initial_position(scenario):
        """
        All positions of polylines/polygons/objects are offset to ego car's first frame position.
        Returns: a modified scenario file
        """
        SD = ScenarioDescription
        sdc_id = scenario[SD.SDC_ID]
        initial_pos = np.array(scenario[SD.OBJECT_TRACKS][sdc_id][SD.STATE][SD.POSITION][0], copy=True)[:2]
        if abs(np.sum(initial_pos)) < 100:
            # has been centralized to the ego center.
            return scenario
        return ScenarioDescription.offset_scenario_with_new_origin(scenario, initial_pos)

    @staticmethod
    def offset_scenario_with_new_origin(scenario, new_origin):
        """
        Set a new origin for the whole scenario. The new origin's position in old coordinate system is recorded, so you
        can add it back and restore the raw data
        Args:
            scenario: The scenario description
            new_origin: The new origin's coordinate in old coordinate system

        Returns: modified data

        """
        SD = ScenarioDescription
        new_origin = np.copy(np.asarray(new_origin))
        for track in scenario[SD.OBJECT_TRACKS].values():
            track[SD.STATE][SD.POSITION] = np.asarray(track[SD.STATE][SD.POSITION])
            track[SD.STATE][SD.POSITION][..., :2] -= new_origin

        for map_feature in scenario[SD.MAP_FEATURES].values():
            if SD.POLYLINE in map_feature:
                map_feature[SD.POLYLINE] = np.asarray(map_feature[SD.POLYLINE])
                map_feature[SD.POLYLINE][..., :2] -= new_origin
            if SD.POLYGON in map_feature:
                map_feature[SD.POLYGON] = np.asarray(map_feature[SD.POLYGON])
                map_feature[SD.POLYGON][..., :2] -= new_origin

        for light in scenario[SD.DYNAMIC_MAP_STATES].values():
            if SD.TRAFFIC_LIGHT_POSITION in light:
                light[SD.TRAFFIC_LIGHT_POSITION] = np.asarray(light[SD.TRAFFIC_LIGHT_POSITION])
                light[SD.TRAFFIC_LIGHT_POSITION][..., :2] -= new_origin

        scenario[SD.METADATA][SD.OLD_ORIGIN_IN_CURRENT_COORDINATE] = -new_origin
        return scenario


def _recursive_check_type(obj, allow_types, depth=0):
    if isinstance(obj, dict):
        for k, v in obj.items():
            assert isinstance(k, str), "Must use string to be dict keys"
            _recursive_check_type(v, allow_types, depth=depth + 1)

    if isinstance(obj, list):
        for v in obj:
            _recursive_check_type(v, allow_types, depth=depth + 1)

    assert isinstance(obj, allow_types), "Object type {} not allowed! ({})".format(type(obj), allow_types)

    if depth > 1000:
        raise ValueError()

# Test
def main():
    scene = dict()
    scene["dummy"] = 12345
    test_SD = ScenarioDescription(scene)
    import pdb; pdb.set_trace()

if __name__ == "__main__":
    main()