"""
Construct the lane object from polyline.
"""
import math

import numpy as np

from worldengine.utils.type import WorldEngineObjectType

from worldengine.components.maps.lanes.base_center_lane import BaseCenterLine
from worldengine.components.maps.map_constants import PGLineType


def wrap_to_pi(x: float) -> float:
    """Wrap the input radian to (-pi, pi]. Note that -pi is exclusive and +pi is inclusive.

    Args:
        x (float): radian.

    Returns:
        The radian in range (-pi, pi].
    """
    angles = x
    angles %= 2 * np.pi
    angles -= 2 * np.pi * (angles > np.pi)
    return angles


class CenterLane(BaseCenterLine):
    DEFAULT_LANE_WIDTH = 6.5
    DEFAULT_POLYGON_SAMPLE_RATE = 1

    def __init__(self,
                 center_line_points,
                 width: float,
                 polygon=None,
                 forbidden: bool = False,
                 speed_limit: float = 1000,
                 priority: int = 0,
                 need_lane_localization=True,
                 auto_generate_polygon=True,
                 lane_type=WorldEngineObjectType.LANE_SURFACE_STREET,
                 segment_threshold=1,
                 segment_eps=1e-6):

        super(CenterLane, self).__init__(center_line_points=center_line_points,
                                         lane_type=lane_type,
                                         segment_threshold=segment_threshold,
                                         segment_eps=segment_eps)

        self._polygon = polygon

        # if polygon is not given, generate polygons by default width setting.
        self.width = width if width else self.DEFAULT_LANE_WIDTH
        if self._polygon is None and auto_generate_polygon:
            self._polygon = self.auto_generate_polygon()

        self.need_lane_localization = need_lane_localization
        self.set_speed_limit(speed_limit)
        self.forbidden = forbidden
        self.priority = priority

        # waymo lane line will be processed separately
        self.line_types = (PGLineType.NONE, PGLineType.NONE)
        self.is_straight = True if abs(self.heading_theta_at(0.1) -
                                       self.heading_theta_at(self.length - 0.1)) < np.deg2rad(10) else False
        self.start = self.position(0, 0)
        assert np.linalg.norm(self.start - center_line_points[0]) < 0.1, "Start point error!"
        self.end = self.position(self.length, 0)
        assert np.linalg.norm(self.end - center_line_points[-1]) < 1, "End point error!"

    def auto_generate_polygon(self):
        start_heading = self.heading_theta_at(0)
        start_dir = [math.cos(start_heading), math.sin(start_heading)]

        end_heading = self.heading_theta_at(self.length)
        end_dir = [math.cos(end_heading), math.sin(end_heading)]
        polygon = []
        longs = np.arange(0,
                          self.length + self.DEFAULT_POLYGON_SAMPLE_RATE,
                          self.DEFAULT_POLYGON_SAMPLE_RATE)
        for k in range(2):  # positive / negative direction so for a circular polygon.
            if k == 1:
                longs = longs[::-1]

            for t, longitude, in enumerate(longs):
                lateral = self.width_at(longitude) / 2
                lateral *= -1 if k == 0 else 1
                point = self.position(longitude, lateral)
                if (t == 0 and k == 0) or (t == len(longs) - 1 and k == 1):
                    # control the adding sequence
                    if k == 1:
                        # last point
                        polygon.append([point[0], point[1]])

                    # extend
                    polygon.append(
                        [
                            point[0] - start_dir[0] * self.DEFAULT_POLYGON_SAMPLE_RATE,
                            point[1] - start_dir[1] * self.DEFAULT_POLYGON_SAMPLE_RATE
                        ]
                    )

                    if k == 0:
                        # first point
                        polygon.append([point[0], point[1]])
                elif (t == 0 and k == 1) or (t == len(longs) - 1 and k == 0):

                    if k == 0:
                        # second point
                        polygon.append([point[0], point[1]])

                    polygon.append(
                        [
                            point[0] + end_dir[0] * self.DEFAULT_POLYGON_SAMPLE_RATE,
                            point[1] + end_dir[1] * self.DEFAULT_POLYGON_SAMPLE_RATE
                        ]
                    )

                    if k == 1:
                        # third point
                        polygon.append([point[0], point[1]])
                else:
                    polygon.append([point[0], point[1]])
        return np.asarray(polygon)

    def width_at(self, longitudinal: float) -> float:
        return self.width

    def is_in_same_direction(self, another_lane):
        """
        Return True if two lane is in same direction
        """
        my_start_heading = self.heading_theta_at(0.1)
        another_start_heading = another_lane.heading_theta_at(0.1)

        my_end_heading = self.heading_theta_at(self.length - 0.1)
        another_end_heading = another_lane.heading_theta_at(self.length - 0.1)

        return True if abs(wrap_to_pi(my_end_heading) - wrap_to_pi(another_end_heading)) < 0.2 and abs(
            wrap_to_pi(my_start_heading) - wrap_to_pi(another_start_heading)
        ) < 0.2 else False

    def destroy(self):
        self.width = None
        self.forbidden = None
        self.priority = None

        self.line_types = None
        self.is_straight = None
        self.start = None
        self.end = None
        self._polygon = None
        BaseCenterLine.destroy(self)

    @property
    def polygon(self):
        return self._polygon


