"""
Base class of map related objects.
"""


import logging
import math
from abc import ABC

import cv2
import numpy as np

from worldengine.base_class.base_runnable import BaseRunnable
from worldengine.components.maps.map_constants import MapTerrainSemanticColor, DrivableAreaProperty
from worldengine.utils.type import WorldEngineObjectType
from worldengine.components.maps.geometry_utils import find_longest_edge
from worldengine.scenario.scenarios.scenario_description import ScenarioDescription as SD


logger = logging.getLogger(__name__)


class BaseMap(BaseRunnable, ABC):
    """
    Base class for Map generation!
    """

    def __init__(self, map_config: dict = None, random_seed=None):
        """
        Map can be stored and recover to save time when we access the map encountered before
        """

        # ignore the random_seed, as the map object is determined
        #  for scenario map object.
        assert random_seed is None

        # randomly initialize the name and random_seed,
        #  and update the parameter_space by map_config.
        super(BaseMap, self).__init__(config=map_config)

        # map features
        # concrete road element management. lane_line, boundary_lane, etc.
        self.road_network = self.road_network_type()
        self.crosswalks = {}
        self.sidewalks = {}

        # A flatten representation of blocks.
        self.blocks = []

        # Generate map and insert blocks
        self._generate()
        assert self.blocks, "The generate methods does not fill blocks!"

    def _generate(self):
        """Key function! Please overwrite it! This func aims at fill the self.road_network adn self.blocks"""
        raise NotImplementedError("Please use child class to specific concrete map"
                                  "generation method!")

    def get_meta_data(self):
        """
        Save the generated map to map file
        """
        return dict(map_type=self.class_name, map_features=self.get_map_features())

    @property
    def num_blocks(self):
        return len(self.blocks)

    def destroy(self):
        for block in self.blocks:
            block.destroy()
        self.blocks = []

        if self.road_network is not None:
            self.road_network.destroy()
        self.road_network = None

        super(BaseMap, self).destroy()

    def __del__(self):
        # self.destroy()
        logger.debug("{} 2is being deleted.".format(type(self)))

    # road network related properties.
    @property
    def road_network_type(self):
        raise NotImplementedError

    def get_center_point(self):
        x_min, x_max, y_min, y_max = self.road_network.get_bounding_box()
        return (x_max + x_min) / 2, (y_max + y_min) / 2

    def get_map_features(self, interval=2):
        """
        Get the map features represented by a set of point lists or polygons
        Args:
            interval: Sampling rate

        Returns: None

        """
        map_features = self.road_network.get_map_features(interval)
        boundary_line_vector = self.get_boundary_line_vector(interval)
        map_features.update(boundary_line_vector)
        map_features.update(self.sidewalks)
        map_features.update(self.crosswalks)
        return map_features

    def get_boundary_line_vector(self, interval):
        return {}

    def get_semantic_map(
        self,
        center_point,
        size=512,
        pixels_per_meter=8,
        color_setting=MapTerrainSemanticColor,
        line_sample_interval=2,
        polyline_thickness=1,
        layer=("lane_line", "lane")
    ):
        """
        Get semantics of the map.
        :param center_point: 2D point, the center to select the rectangular region
        :param size: [m] length and width
        :param pixels_per_meter: the returned map will be in (size*pixels_per_meter * size*pixels_per_meter) size
        :param color_setting: color palette for different attribute. When generating terrain, make sure using
        :param line_sample_interval: [m] It determines the resolution of sampled points.
        :param polyline_thickness: [m] The width of the road lines
        :param layer: layer to get
        MapTerrainAttribute
        :return: semantic map
        """
        center_p = center_point

        # if self._semantic_map is None:
        all_lanes = self.get_map_features(interval=line_sample_interval)
        polygons = []
        polylines = []

        points_to_skip = math.floor(DrivableAreaProperty.STRIPE_LENGTH * 2 / line_sample_interval)
        for obj in all_lanes.values():
            if WorldEngineObjectType.is_lane(obj[SD.TYPE]) and "lane" in layer:
                # polygons.append((obj[SD.POLYGON], MapTerrainSemanticColor.get_color(obj[SD.TYPE])))

                # also draw lane on image.
                polylines.append((obj[SD.POLYLINE], MapTerrainSemanticColor.get_color(obj[SD.TYPE])))

            elif "lane_line" in layer and (WorldEngineObjectType.is_road_line(obj[SD.TYPE])
                                           or WorldEngineObjectType.is_road_boundary_line(obj[SD.TYPE])):
                if WorldEngineObjectType.is_broken_line(obj[SD.TYPE]):
                    for index in range(0, len(obj[SD.POLYLINE]) - 1, points_to_skip * 2):
                        if index + points_to_skip < len(obj[SD.POLYLINE]):
                            polylines.append(
                                (
                                    [obj[SD.POLYLINE][index],
                                     obj[SD.POLYLINE][index + points_to_skip]],
                                    MapTerrainSemanticColor.get_color(obj[SD.TYPE])
                                )
                            )
                else:
                    polylines.append((obj[SD.POLYLINE],
                                      MapTerrainSemanticColor.get_color(obj[SD.TYPE])))

        # draw road lines on the map to obtain semantic_map.
        size = int(size * pixels_per_meter)
        mask = np.zeros([size, size, 4], dtype=np.float32)
        mask[..., :] = color_setting.get_color(WorldEngineObjectType.GROUND)
        for polygon, color in polygons:
            points = [
                [
                    int((x - center_p[0]) * pixels_per_meter + size / 2),
                    int(- (y - center_p[1]) * pixels_per_meter) + size / 2
                ] for x, y in polygon
            ]
            cv2.fillPoly(mask, np.array([points]).astype(np.int32), color=color)
        for line, color in polylines:
            points = [
                [
                    int((p[0] - center_p[0]) * pixels_per_meter + size / 2),
                    int(- (p[1] - center_p[1]) * pixels_per_meter + size / 2)
                ] for p in line
            ]
            thickness = polyline_thickness * 2 if color == MapTerrainSemanticColor.YELLOW else polyline_thickness
            thickness = min(thickness, 2)  # clip
            cv2.polylines(mask, np.array([points]).astype(np.int32), False, color, thickness)

        if "crosswalk" in layer:
            for id, sidewalk in self.crosswalks.items():
                polygon = sidewalk[SD.POLYGON]
                points = [
                    [
                        int((x - center_p[0]) * pixels_per_meter + size / 2),
                        int(- (y - center_p[1]) * pixels_per_meter + size / 2)
                    ] for x, y in polygon
                ]
                p_1, p_2 = find_longest_edge(polygon)[0]
                dir = (
                    p_2[0] - p_1[0],
                    - (p_2[1] - p_1[1]),
                )
                # 0-2pi
                angle = np.arctan2(*dir) / np.pi * 180 + 180
                # normalize to 0.4-0.714
                angle = angle / 1000 + MapTerrainSemanticColor.get_color(WorldEngineObjectType.CROSSWALK)
                cv2.fillPoly(mask, np.array([points]).astype(np.int32), color=angle)
        return mask

    def get_trajectory_map(
        self,
        start_points,
        end_points,
        traj_lanes,

        center_point,
        size=512,
        pixels_per_meter=8,
        color_setting=MapTerrainSemanticColor,
        line_sample_interval=2,
        polyline_thickness=1,
        layer=("lane_line", "lane"),

        trajectory_color=MapTerrainSemanticColor.NAVI_COLOR,
        trajectory_thickness=1,
        semantic_map=None,
    ):
        """
        A function to visualize the navigation lane along with the map.
        """
        if semantic_map is not None:
            mask = semantic_map
        else:
            mask = self.get_semantic_map(
                center_point,
                size=size,
                pixels_per_meter=pixels_per_meter,
                color_setting=color_setting,
                line_sample_interval=line_sample_interval,
                polyline_thickness=polyline_thickness,
                layer=layer)
        size = int(size * pixels_per_meter)

        # draw lane polygons and lane polylines.
        polylines = []
        for lane in traj_lanes:
            polylines.append((
                lane.get_polyline(line_sample_interval),
                trajectory_color
            ))

        for line, color in polylines:
            points = [
                [
                    int((p[0] - center_point[0]) * pixels_per_meter + size / 2),
                    int(- (p[1] - center_point[1]) * pixels_per_meter + size / 2)
                ] for p in line
            ]
            cv2.polylines(mask, np.array([points]).astype(np.int32), False, color, trajectory_thickness)

        # draw starting points and destination points.
        for point in start_points:
            cv2.circle(
                mask,
                (
                    int((point[0] - center_point[0]) * pixels_per_meter + size / 2),
                    int(- (point[1] - center_point[1]) * pixels_per_meter + size / 2)
                ),
                radius=3,
                color=MapTerrainSemanticColor.START_POINT_COLOR,
                thickness=3)

        for point in end_points:
            cv2.circle(
                mask,
                (
                    int((point[0] - center_point[0]) * pixels_per_meter + size / 2),
                    int(- (point[1] - center_point[1]) * pixels_per_meter + size / 2)
                ),
                radius=3,
                color=MapTerrainSemanticColor.END_POINT_COLOR,
                thickness=3)
        return mask

    def get_trajectory_map_with_box(
        self,
        box_polygons,

        start_points,
        end_points,
        traj_lanes,
        center_point,
        size=512,
        pixels_per_meter=8,
        color_setting=MapTerrainSemanticColor,
        line_sample_interval=2,
        polyline_thickness=1,
        layer=("lane_line", "lane"),

        trajectory_color=MapTerrainSemanticColor.NAVI_COLOR,
        trajectory_thickness=1,
        semantic_map=None,
    ):
        """
        Draw the oriented box on the map.
        """

        mask = self.get_trajectory_map(
            start_points=start_points,
            end_points=end_points,
            traj_lanes=traj_lanes,
            center_point=center_point,
            size=size,
            pixels_per_meter=pixels_per_meter,
            color_setting=color_setting,
            line_sample_interval=line_sample_interval,
            polyline_thickness=polyline_thickness,
            layer=layer,
            trajectory_color=trajectory_color,
            trajectory_thickness=trajectory_thickness,
            semantic_map=semantic_map,
        )
        size = int(size * pixels_per_meter)

        # draw oriented boxes.
        box_color = [0, 1, 0]
        for box_polygon in box_polygons:
            # draw lines on the mask image.
            # each box_polygon has 4 lines, indicating the
            # 4 boundaries of vehicle.
            points = [
                [
                    int((p[0] - center_point[0]) * pixels_per_meter + size / 2),
                    int(- (p[1] - center_point[1]) * pixels_per_meter + size / 2)
                ] for p in box_polygon
            ]
            assert len(points) == 4
            for i in range(4):
                j = (i + 1) % 4
                cv2.line(mask, (points[i][0], points[i][1]), (points[j][0], points[j][1]), box_color, polyline_thickness)

        return mask
