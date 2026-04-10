import numpy as np
from scipy.interpolate import interp1d

from worldengine.utils.type import WorldEngineObjectType
from worldengine.components.maps.base_map import BaseMap
from worldengine.components.maps.road_networks.edge_road_network import EdgeRoadNetwork
from worldengine.components.maps.blocks.scenario_block import ScenarioBlock
from worldengine.scenario.scenarios.scenario_description import ScenarioDescription as SD


def get_polyline_length(points_array):
    diff = np.diff(points_array, axis=0)
    squared_diff = diff**2
    squared_diff_sum = np.sum(squared_diff, axis=1)
    distances = np.sqrt(squared_diff_sum)
    return np.sum(distances)


def resample_polyline(points, target_distance):
    # Calculate the cumulative distance along the original polyline
    distances = np.cumsum(np.sqrt(np.sum(np.diff(points, axis=0)**2, axis=1)))
    distances = np.insert(distances, 0, 0., axis=0)

    # Create a linearly spaced array of distances for the resampled polyline
    resampled_distances = np.arange(0, distances[-1], target_distance)

    # Interpolate the points along the resampled distances
    resampled_points = interp1d(distances, points, axis=0)(resampled_distances)

    return resampled_points


class ScenarioMap(BaseMap):
    def __init__(self, map_index, map_data, random_seed=None):
        """
        map_data is a dictionary with line elements in this road.
        Each element in the dictionary includes:
        <line_id>: <line_type> & <line_polygon>
        """

        self.map_index = map_index
        self.map_data = map_data
        super(ScenarioMap, self).__init__(dict(id=self.map_index), random_seed=random_seed)

    def _generate(self):
        """ Function to generate maps. """

        block = ScenarioBlock(
            block_index=0,
            global_network=self.road_network,
            random_seed=0,
            map_index=self.map_index,
            map_data=self.map_data,
        )
        self.crosswalks = block.crosswalks
        self.sidewalks = block.sidewalks
        self.blocks.append(block)

    @property
    def road_network_type(self):
        return EdgeRoadNetwork

    def destroy(self):
        self.map_index = None
        super(ScenarioMap, self).destroy()

    def get_boundary_line_vector(self, interval):
        """
        Get the polylines of the map, represented by a set of points
        """
        ret = {}
        for lane_id, data in self.blocks[-1].map_data.items():
            type = data.get(SD.TYPE, None)
            map_feat_id = str(lane_id)
            if WorldEngineObjectType.is_road_line(type):
                if len(data[SD.POLYLINE]) <= 1:
                    continue
                line = np.asarray(data[SD.POLYLINE])[..., :2]
                length = get_polyline_length(line)
                resampled = resample_polyline(line, interval) if length > interval * 2 else line
                if WorldEngineObjectType.is_broken_line(type):
                    ret[map_feat_id] = {
                        SD.TYPE: WorldEngineObjectType.LINE_BROKEN_SINGLE_YELLOW
                        if WorldEngineObjectType.is_yellow_line(type)
                        else WorldEngineObjectType.LINE_BROKEN_SINGLE_WHITE,
                        SD.POLYLINE: resampled
                    }
                else:
                    ret[map_feat_id] = {
                        SD.POLYLINE: resampled,
                        SD.TYPE: WorldEngineObjectType.LINE_SOLID_SINGLE_YELLOW
                        if WorldEngineObjectType.is_yellow_line(type)
                        else WorldEngineObjectType.LINE_SOLID_SINGLE_WHITE
                    }
            elif WorldEngineObjectType.is_road_boundary_line(type):
                line = np.asarray(data[SD.POLYLINE])[..., :2]
                length = get_polyline_length(line)
                resampled = resample_polyline(line, interval) if length > interval * 2 else line
                ret[map_feat_id] = {SD.POLYLINE: resampled,
                                    SD.TYPE: WorldEngineObjectType.BOUNDARY_LINE}
            elif WorldEngineObjectType.is_lane(type):
                continue
        return ret
