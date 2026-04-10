import logging
import math
import numpy as np

from worldengine.scenario.scenarios.scenario_description import ScenarioDescription as SD
from worldengine.components.maps.lanes.center_lane import CenterLane


# return the nearest point"s index of the line
def nearest_point(point, line):
    dist = np.square(line - point)
    dist = np.sqrt(dist[:, 0] + dist[:, 1])
    return np.argmin(dist)


def norm(x, y):
    return math.sqrt(x**2 + y**2)


def mph_to_kmh(speed_in_mph: float):
    speed_in_kmh = speed_in_mph * 1.609344
    return speed_in_kmh


class ScenarioLane(CenterLane):
    DEFAULT_MAX_SPEED_LIMIT = 100

    def __init__(self, lane_id: int, map_data: dict, need_lane_localization=True):
        """
        Extract the lane information of one lane, and do coordinate shift if required
        """
        center_line_points = np.asarray(map_data[lane_id][SD.POLYLINE])
        if SD.POLYGON in map_data[lane_id] and len(map_data[lane_id][SD.POLYGON]) > 3:
            polygon = np.asarray(map_data[lane_id][SD.POLYGON])
        else:
            polygon = None
        if "speed_limit_kmh" in map_data[lane_id] or "speed_limit_mph" in map_data[lane_id]:
            speed_limit_kmh = map_data[lane_id].get("speed_limit_kmh", None)
            if speed_limit_kmh is None:
                speed_limit_kmh = mph_to_kmh(map_data[lane_id]["speed_limit_mph"])
        else:
            speed_limit_kmh = self.DEFAULT_MAX_SPEED_LIMIT

        super(ScenarioLane, self).__init__(
            center_line_points=center_line_points,
            width=None,
            polygon=polygon,
            speed_limit=speed_limit_kmh,
            need_lane_localization=need_lane_localization,
            lane_type=map_data[lane_id][SD.TYPE]
        )

        self.index = lane_id
        self.lane_type = map_data[lane_id]["type"]
        self.entry_lanes = map_data[lane_id].get(SD.ENTRY, None)
        self.exit_lanes = map_data[lane_id].get(SD.EXIT, None)
        self.left_lanes = map_data[lane_id].get(SD.LEFT_NEIGHBORS, None)
        self.right_lanes = map_data[lane_id].get(SD.RIGHT_NEIGHBORS, None)
        self.roadblock_id = map_data[lane_id]["roadblock_id"]

    def __del__(self):
        logging.debug("ScenarioLane is released")

    def destroy(self):
        self.index = None
        self.entry_lanes = None
        self.exit_lanes = None
        self.left_lanes = None
        self.right_lanes = None
        super(ScenarioLane, self).destroy()
