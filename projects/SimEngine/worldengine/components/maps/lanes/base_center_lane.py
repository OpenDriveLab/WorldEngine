import math

import numpy as np

from worldengine.utils.type import WorldEngineObjectType
from worldengine.components.maps.lanes.base_lane import BaseLane


def norm(x, y):
    return math.sqrt(x**2 + y**2)


def get_vertical_vector(vector: np.array):
    length = norm(vector[0], vector[1])
    return (-vector[1] / length, vector[0] / length), (vector[1] / length, -vector[0] / length)


class BaseCenterLine(BaseLane):
    def __init__(self,
                 center_line_points,
                 lane_type=WorldEngineObjectType.LANE_SURFACE_STREET,

                 # segment with length over <segment_threshold> are considered valid.
                 segment_threshold=1,
                 segment_eps=1e-6,
                 ):

        self.segment_threshold = segment_threshold
        self.segment_eps = segment_eps

        super(BaseCenterLine, self).__init__(lane_type=lane_type)
        # points among the polyline.
        self.center_line_points = np.array(center_line_points)[..., :2]
        self.lane_segments, self.lane_start_points, self.lane_end_points = self.init_lanes()
        self.length = sum([seg["length"] for seg in self.lane_segments])

    def init_lanes(self):
        segments = []
        start_points = []
        end_points = []
        p_start_idx = 0
        while p_start_idx < len(self.center_line_points) - 1:
            for p_end_idx in range(p_start_idx + 1, len(self.center_line_points)):
                _p = (self.center_line_points[p_start_idx] -
                      self.center_line_points[p_end_idx])
                if norm(_p[0], _p[1]) > self.segment_threshold:  # valid segments longer than 1m.
                    break

            p_start = self.center_line_points[p_start_idx]
            p_end = self.center_line_points[p_end_idx]

            # This is to ensure the last point must be selected.
            while p_end_idx < len(self.center_line_points) - 1:
                # still has the next end point.
                p_end_next = self.center_line_points[p_end_idx + 1]
                _p_next = p_end_next - p_end
                _p_dist = norm(_p_next[0], _p_next[1])
                if _p_dist < self.segment_eps:
                    p_end_idx += 1
                    p_end = self.center_line_points[p_end_idx]
                else:
                    break

            _p = p_start - p_end
            if norm(_p[0], _p[1]) < self.segment_eps:
                p_start_idx = p_end_idx  # not find a valid lane segment, continue.
                continue

            seg_property = {
                "length": self.points_distance(p_start, p_end),
                "direction": np.asarray(self.points_direction(p_start, p_end)),
                "lateral_direction": np.asarray(self.points_lateral_direction(p_start, p_end)),
                "heading": self.points_heading(p_start, p_end),
                "start_point": p_start,
                "end_point": p_end,
                "start_idx": p_start_idx,
                "end_idx": p_end_idx,
            }
            segments.append(seg_property)
            start_points.append(seg_property["start_point"])
            end_points.append(seg_property["end_point"])
            p_start_idx = p_end_idx  # next
        if len(segments) == 0:
            # static, length=zero
            seg_property = {
                "length": 0.1,
                "direction": np.asarray((1, 0)),
                "lateral_direction": np.asarray((0, 1)),
                "heading": 0,
                "start_point": self.center_line_points[0],
                "end_point": np.asarray([self.center_line_points[0][0] +
                                         0.1, self.center_line_points[0][1]]),
                "start_idx": 0,
                "end_idx": 0,
            }
            segments.append(seg_property)
            start_points.append(seg_property["start_point"])
            end_points.append(seg_property["end_point"])
        return segments, np.asarray(start_points), np.asarray(end_points)

    def points_distance(self, start_p, end_p):
        return norm((end_p - start_p)[0], (end_p - start_p)[1])

    def points_direction(self, start_p, end_p):
        return (end_p - start_p) / norm((end_p - start_p)[0], (end_p - start_p)[1])

    def points_lateral_direction(self, start_p, end_p):
        return np.asarray(get_vertical_vector(end_p - start_p)[1])

    def points_heading(self, start_p, end_p):
        return math.atan2(end_p[1] - start_p[1], end_p[0] - start_p[0])

    def get_point(self, longitudinal, lateral=None):
        """
        Get point on this line by interpolating
        """
        accumulate_len = 0
        for seg in self.lane_segments:
            accumulate_len += seg["length"]
            if accumulate_len + 0.1 >= longitudinal:
                break
        if lateral is not None:
            return (seg["start_point"] + (longitudinal - accumulate_len + seg["length"]) *
                    seg["direction"]) + lateral * seg["lateral_direction"]
        else:
            return seg["start_point"] + (longitudinal - accumulate_len + seg["length"]) * seg["direction"]

    def position(self, longitudinal: float, lateral: float):
        return self.get_point(longitudinal, lateral)

    def min_lineseg_dist(self, pos, start, end):
        """Cartesian distance from point to line segment
        Edited to support arguments as series, from:
        https://stackoverflow.com/a/54442561/11208892

        Args:
            - pos: np.array of single point, shape (2,) or 2D array, shape (x, 2)
            - start: np.array of shape (x, 2)
            - end: np.array of shape (x, 2)
        """
        # normalized tangent vectors
        pos = np.asarray(pos)
        distance_start_end = end - start
        d = np.divide(distance_start_end,
                      (np.hypot(distance_start_end[:, 0], distance_start_end[:, 1]).reshape(-1, 1)))

        # signed parallel distance components
        # rowwise dot products of 2D vectors
        s = np.multiply(start - pos, d).sum(axis=1)
        t = np.multiply(pos - end, d).sum(axis=1)

        # clamped parallel distance
        h = np.maximum.reduce([s, t, np.zeros(len(s))])

        # perpendicular distance component
        # rowwise cross products of 2D vectors
        d_pa = pos - start
        c = d_pa[:, 0] * d[:, 1] - d_pa[:, 1] * d[:, 0]
        min_dists = np.hypot(h, c)
        return min_dists

    def local_coordinates(self, position: np.ndarray):
        """
        Convert a physx_world position to local lane coordinates.

        :param position: a physx_world position [m]
        :return: the (longitudinal, lateral) lane coordinates [m]
        """
        min_dists = self.min_lineseg_dist(
            position, self.lane_start_points, self.lane_end_points)
        target_segment_idx = np.argmin(min_dists)

        long = 0
        for idx, seg in enumerate(self.lane_segments):
            if idx != target_segment_idx:
                long += seg["length"]
            else:
                delta_x = position[0] - seg["start_point"][0]
                delta_y = position[1] - seg["start_point"][1]
                long += delta_x * seg["direction"][0] + delta_y * seg["direction"][1]
                lateral = delta_x * seg["lateral_direction"][0] + delta_y * seg["lateral_direction"][1]
                return long, lateral

    def segment(self, longitudinal: float):
        """
        Return the segment piece on this lane of current position
        """
        accumulate_len = 0
        for index, seg in enumerate(self.lane_segments):
            accumulate_len += seg["length"]
            if accumulate_len + 0.1 >= longitudinal:
                return self.lane_segments[index]
        return self.lane_segments[index]

    def lateral_direction(self, longitude):
        lane_segment = self.segment(longitude)
        lateral = lane_segment["lateral_direction"]
        return lateral

    def heading_theta_at(self, longitudinal: float) -> float:
        """
        Get the lane heading at a given longitudinal lane coordinate.

        :param longitudinal: longitudinal lane coordinate [m]
        :return: the lane heading [rad]
        """
        seg = self.segment(longitudinal)
        return seg["heading"]

    def width_at(self, longitudinal: float) -> float:
        """
        Get the lane width at a given longitudinal lane coordinate.

        :param longitudinal: longitudinal lane coordinate [m]
        :return: the lane width [m]
        """
        raise NotImplementedError()

    def destroy(self):
        del self.lane_segments
        self.lane_segments = []
        self.length = None
