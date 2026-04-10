"""
Category mapping from nuplan track objects to world engine.
"""


import logging

from worldengine.utils.type import WorldEngineObjectType

from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType
from nuplan.common.maps.maps_datatypes import TrafficLightStatusType
NuPlanEgoType = TrackedObjectType.EGO


def get_traffic_obj_type(nuplan_type):
    if nuplan_type == TrackedObjectType.VEHICLE:
        return WorldEngineObjectType.VEHICLE
    elif nuplan_type == TrackedObjectType.TRAFFIC_CONE:
        return WorldEngineObjectType.TRAFFIC_CONE
    elif nuplan_type == TrackedObjectType.PEDESTRIAN:
        return WorldEngineObjectType.PEDESTRIAN
    elif nuplan_type == TrackedObjectType.BICYCLE:
        return WorldEngineObjectType.CYCLIST
    elif nuplan_type == TrackedObjectType.BARRIER:
        return WorldEngineObjectType.TRAFFIC_BARRIER
    elif nuplan_type == TrackedObjectType.GENERIC_OBJECT:
        return WorldEngineObjectType.TRAFFIC_OBJECT
    elif nuplan_type == TrackedObjectType.EGO:
        raise ValueError("Ego should not be in detected results")
    else:
        return None


def set_light_status(status):
    if status == TrafficLightStatusType.GREEN:
        return WorldEngineObjectType.LIGHT_GREEN
    elif status == TrafficLightStatusType.RED:
        return WorldEngineObjectType.LIGHT_RED
    elif status == TrafficLightStatusType.YELLOW:
        return WorldEngineObjectType.LIGHT_YELLOW
    elif status == TrafficLightStatusType.UNKNOWN:
        return WorldEngineObjectType.LIGHT_UNKNOWN
