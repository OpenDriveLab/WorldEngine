"""
Basic class for road network management.
This class stores the topological roadmap.
"""

from typing import List, Tuple, Union
from worldengine.components.maps.lanes.base_lane import BaseLane

LaneIndex = Union[str, Tuple[str, str, int]]


class BaseRoadNetwork:
    def __init__(self):

        # a graph to store the topology between lanes.
        self.graph = None

        self.bounding_box = None
        self._lines_np = []

    def clear(self):
        self.graph.clear()

    # get areas of the roadnetwork.
    def get_bounding_box(self):
        """
        Get bounding box of this road network
        """
        if self.bounding_box is None:
            self.bounding_box = self._get_bounding_box()
        return self.bounding_box

    def get_center_point(self):
        """
        Get the center point of this roadnetwork
        """
        x_min, x_max, y_min, y_max = self.get_bounding_box()
        return (x_max + x_min) / 2, (y_max + y_min) / 2

    def _get_bounding_box(self):
        raise NotImplementedError

    # add or remove lanes in this roadnetwork.
    def add_lane(self, *args, **kwargs) -> None:
        """
        Add one lane to the roadnetwork for querying
        """
        raise NotImplementedError

    def get_lane(self, index: LaneIndex) -> BaseLane:
        """
        Get the lane corresponding to a given index in the road network.
        """
        raise NotImplementedError

    def get_closest_lane_index(self, position, return_all=False):
        raise NotImplementedError

    def shortest_path(self, start: str, goal: str) -> List[str]:
        """
        Breadth-first search of shortest checkpoints from start to goal.

        :param start: starting node
        :param goal: goal node
        :return: shortest checkpoints from start to goal.
        """
        raise NotImplementedError

    def __isub__(self, other):
        raise NotImplementedError

    def add(self, other, no_intersect=True):
        """
        Add another network to this one, no intersect means that the same lane should noly exist in self or other
        return: self
        """
        raise NotImplementedError

    def __sub__(self, other):
        ret = self.__class__()
        ret.graph = self.graph
        ret -= other
        return ret

    def destroy(self):
        """
        Destroy all lanes in this network
        Returns: None

        """
        self.bounding_box = None

    def has_connection(self, lane_index_1, lane_index_2):
        """
        Return True if lane 1 is the previous lane of lane 2
        """
        if lane_index_1[1] in self.graph:
            if lane_index_2[1] in self.graph[lane_index_1[1]]:
                return True
        return False

    def get_map_features(self, interval=2):
        raise NotImplementedError

    def get_all_lanes(self):
        """
        This function will return all lanes in the road network
        :return: list of lanes
        """
        raise NotImplementedError
