"""
Shapely geometry utils for dealing with map elements.
"""

import math
import shapely
import heapq
from typing import Union
from shapely.geometry import Polygon
from itertools import combinations


def calculate_slope(p1, p2):
    """
    Calculate the slope of a line segment.
    Args:
        p1: point 1
        p2: point 2

    Returns:

    """
    # Handle the case of a vertical line segment
    if p1[0] == p2[0]:
        return float('inf')
    else:
        return (p2[1] - p1[1]) / (p2[0] - p1[0])


def length(edge):
    """
    Return the length of an edge
    Args:
        edge: edge, two points

    Returns:

    """
    p_1 = edge[0]
    p_2 = edge[1]
    return math.sqrt(
        (p_1[0] - p_2[0]) ** 2 + (p_1[1] - p_2[1]) ** 2
    )


def size(edge):
    """
    The size of the edge
    Args:
        edge: two points defining an edge

    Returns: length^2 of a vector

    """
    p_1 = edge[0]
    p_2 = edge[1]
    x = p_1[0] - p_2[0]
    y = p_1[1] - p_2[1]
    return x**2 + y**2


def find_longest_parallel_edges(polygon: Union[shapely.geometry.Polygon, list]):
    """
    Given a boundary,
        Find and return the longest parallel edges of a polygon.
        If it can not find, return the longest two edges instead.
    Args:
        polygon: shapely.Polygon or list of 2D points representing a polygon

    Returns:

    """

    edges = []
    longest_parallel_edges = None
    # get the position of surrounding linear ring.
    coords = list(polygon.exterior.coords) if isinstance(polygon, shapely.geometry.Polygon) else polygon

    # Extract the edges from the polygon
    for i in range(len(coords) - 1):
        edge = (coords[i], coords[i + 1])
        edges.append(edge)
    edges.append((coords[-1], coords[0]))

    # Compare each edge with every other edge
    for edge1, edge2 in combinations(edges, 2):
        # for any combination between 2 edges.

        # compute the slope, to see whether parallel.
        slope1 = calculate_slope(*edge1)
        slope2 = calculate_slope(*edge2)

        # Check if the slopes are equal (or both are vertical)
        if abs(slope1 - slope2) < 0.5:
            max_len = max(length(edge1), length(edge2))
            if longest_parallel_edges is None or max_len > longest_parallel_edges[-1]:
                longest_parallel_edges = ((edge1, edge2), max_len)

    if longest_parallel_edges:
        return longest_parallel_edges[0]
    else:
        # return sorted(edges, key=lambda edge: size(edge))[-2:]
        return heapq.nlargest(2, edges, key=lambda edge: size(edge))


def find_longest_edge(polygon: Union[shapely.geometry.Polygon, list]):
    """
    Return the longest edge of a polygon
    Args:
        polygon: shapely.Polygon or list of 2D points representing a polygon

    Returns: the longest edge

    """
    coords = list(polygon.exterior.coords) if isinstance(polygon, shapely.geometry.Polygon) else polygon
    edges = []
    # Extract the edges from the polygon
    for i in range(len(coords) - 1):
        edge = (coords[i], coords[i + 1])
        edges.append(edge)
    edges.append((coords[-1], coords[0]))
    return heapq.nlargest(1, edges, key=lambda edge: size(edge))


if __name__ == '__main__':
    polygon = Polygon(
        [
            [356.83858017, -234.46019451], [355.12995531, -239.44667613], [358.76606674, -240.73795931],
            [360.27632766, -235.80099687], [356.83858017, -234.46019451]
        ]
    )
    parallel_edges = find_longest_parallel_edges(polygon)
    assert parallel_edges == [
        (
            ((355.12995531, -239.44667613), (358.76606674, -240.73795931)),
            ((360.27632766, -235.80099687), (356.83858017, -234.46019451))
        )
    ]
    print(parallel_edges)
