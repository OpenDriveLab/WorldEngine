from nuplan.common.actor_state.state_representation import StateSE2, TimePoint, StateVector2D
from nuplan.common.actor_state.ego_state import EgoState
from worldengine.components.agents.vehicle_model.pacifica_vehicle import get_pacifica_parameters
from worldengine.utils import math_utils
from worldengine.common.dataclasses import Trajectory
from nuplan.planning.simulation.observation.observation_type import DetectionsTracks
from nuplan.common.actor_state.agent import Agent
from nuplan.common.actor_state.oriented_box import OrientedBox
from nuplan.common.actor_state.tracked_objects import TrackedObjects
from nuplan.common.actor_state.static_object import StaticObject
from nuplan.common.actor_state.scene_object import SceneObjectMetadata
from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType
from nuplan.common.maps.maps_datatypes import TrafficLightStatusData, TrafficLightStatusType

from typing import List, Dict, Union
import numpy as np

TrackedObject = Union[Agent, StaticObject]

tracked_object_type_mapping = {
    "VEHICLE": TrackedObjectType.VEHICLE,
    "PEDESTRIAN": TrackedObjectType.PEDESTRIAN,
    "CYCLIST": TrackedObjectType.BICYCLE,
    "TRAFFIC_CONE": TrackedObjectType.TRAFFIC_CONE,
    "BARRIER": TrackedObjectType.BARRIER,
    "TRAFFIC_BARRIER": TrackedObjectType.BARRIER,
    "CZONE_SIGN": TrackedObjectType.CZONE_SIGN,
    "GENERIC_OBJECT": TrackedObjectType.GENERIC_OBJECT,
    "TRAFFIC_OBJECT": TrackedObjectType.GENERIC_OBJECT,
    "EGO": TrackedObjectType.EGO
}

# Utils
def denormalize_from_ego_center(vector, ego_center):
    "Denormalize position-related vectors from the ego_center of the first frame."
    vector = np.array(vector)
    vector += np.asarray(ego_center)
    return vector

def normalize_to_ego_center(vector, ego_center=(0, 0)):
    "Normalize position-related vectors to the ego_center of the first frame."
    vector = np.array(vector)
    vector -= np.asarray(ego_center)
    return vector


class WE2NuPlanConverter:
    def __init__(self, scene, agent, engine):
        self.scene = scene
        self.agent = agent
        self.engine = engine
        self.base_timestamp = scene.get("base_timestamp", 0.0)  # Default 0 if not present
        self.initial_ego_center = - self.scene["metadata"]["old_origin_in_current_coordinate"] # take the opposite number

    def convert_to_current_ego_state(self, current_step: int) -> EgoState:
        """
        Convert WorldEngine scene data to nuPlan's initial_ego_state format.

        Args:
            current_step: Current simulation step to extract the corresponding state.

        Returns:
            EgoState: nuPlan-compatible initial_ego_state object.
        """
        # Extract ego vehicle's data
        ego_id = self.scene["sdc_id"]
        ego_state = self.scene["object_track"][ego_id]["state"]
        

        # Center position and heading
        position = self.agent.rear_vehicle.current_position
        rear_position = denormalize_from_ego_center(position, self.initial_ego_center)  # Denormalize from ego_center
        heading = self.agent.current_heading   # First frame heading
        
        # Center velocity and acceleration
        # velocity = self.agent.current_velocity
        # velocity = denormalize_from_ego_center(velocity, [0.0, 0.0])  # Denormalize from ego_center
        
        # Transfer to rear_axle
        rear_velocity = self.agent.current_velocity
        cos, sin = np.cos(heading), np.sin(heading)
        rear_velocity = np.array([
            cos * rear_velocity[0] + sin * rear_velocity[1],
            -sin * rear_velocity[0] + cos * rear_velocity[1]
        ])
        # rear_position = math_utils.translate_longitudinally(
        #     position, heading, -self.agent.rear_vehicle.rear_axle_to_center_dist).reshape(2)

        # Timestamp
        timestamp = self.base_timestamp + current_step * self.scene["sample_rate"] * 0.05 * 1e6 
        # scene["sample_rate"] * 0.05 represents sim_time_interval

        # Construct StateSE2
        state_se2 = StateSE2(x=rear_position[0], y=rear_position[1], heading=heading)

        # Construct EgoState
        current_ego_state = EgoState.build_from_rear_axle(
            rear_axle_pose=state_se2,
            rear_axle_velocity_2d=StateVector2D(x=rear_velocity[0], y=rear_velocity[1]),
            rear_axle_acceleration_2d=StateVector2D(x=0.0, y=0.0),  # TODO: set to 0.0 for now
            tire_steering_angle=0.0,  # Assuming 0 for simplicity
            vehicle_parameters=get_pacifica_parameters(),
            time_point=TimePoint(timestamp)
        )

        return current_ego_state

    def convert_to_detections_tracks_from_scene(self, current_step: int) -> DetectionsTracks:
        """
        Convert WorldEngine object_track data to nuPlan DetectionsTracks format.

        Args:
            current_step: Current simulation step to extract the corresponding state.

        Returns:
            DetectionsTracks: A DetectionsTracks object for the current step.
        """
        tracked_objects = []  # List of TrackedObject
        # Timestamp
        timestamp = self.base_timestamp + current_step * self.scene["sample_rate"] * 0.05 * 1e6 
        for object_id, object_data in self.scene["object_track"].items():
            # Extract and map object type
            object_type = object_data["type"]
            tracked_object_type = tracked_object_type_mapping.get(object_type)
            if tracked_object_type is None:
                raise ValueError(f"Unknown object type: {object_type}")

            # Extract state for the current step
            state = object_data["state"]
            position = state["position"][current_step]  # [x, y, z]
            if np.all(position == np.array([0.0, 0.0, 0.0])):
                continue
            heading = state.get("heading", [0] * len(state["position"]))[current_step]  # Default 0 if not present
            velocity = state.get("velocity", np.zeros_like(state["position"]))[current_step]
            length = state.get("length", 0.0)
            width = state.get("width", 0.0)
            height = state.get("height", 0.0)

            position = denormalize_from_ego_center(position[:2], self.initial_ego_center)

            metadata = object_data["metadata"]

            # Create OrientedBox
            box = OrientedBox(
                center = StateSE2(x=position[0], y=position[1], heading=heading),
                length = float(length[0]),
                width = float(width[0]),
                height = float(height[0])
            )

            # Create TrackedObject based on type
            if object_type in ["VEHICLE", "PEDESTRIAN", "CYCLIST", "EGO"]:
                tracked_object = Agent(
                    tracked_object_type=tracked_object_type,
                    oriented_box=box,
                    velocity=StateVector2D(x=velocity[0], y=velocity[1]),
                    metadata=SceneObjectMetadata(
                        timestamp_us = timestamp,
                        token = self.scene["token"],
                        track_id = metadata.get("nuplan_id", object_id),
                        track_token = object_id
                    )
                )
            else:
                tracked_object = StaticObject(
                    tracked_object_type = tracked_object_type,
                    oriented_box=box,
                    metadata=SceneObjectMetadata(
                        timestamp_us = timestamp,
                        token = self.scene["token"],
                        track_id = metadata.get("nuplan_id", object_id),
                        track_token = object_id
                    )
                )

            # Append to the tracked objects list
            tracked_objects.append(tracked_object)

        # Wrap in TrackedObjects
        tracked_objects_container = TrackedObjects(tracked_objects=tracked_objects)

        # Wrap in DetectionsTracks
        detections_tracks = DetectionsTracks(tracked_objects=tracked_objects_container)

        return detections_tracks
    
    def convert_to_detections_tracks_from_agent_input(self, current_step) -> DetectionsTracks:
        """
        Convert real-time agent states from engine to nuPlan's DetectionsTracks format.
        This function differs from convert_to_detections_tracks_from_scene as it reads current states
        directly from running agents in the engine instead of pre-recorded scene data.

        Args:
            current_step (int): Current simulation step to calculate the timestamp

        Returns:
            DetectionsTracks: A container of all tracked objects (agents) in nuPlan format, including:
                - Dynamic objects (vehicles, pedestrians, cyclists)
                - Static objects (traffic cones, barriers)
                Each object contains:
                - Position, heading, and velocity
                - Object type and dimensions
                - Metadata (timestamp, track_id, etc.)
        """
        tracked_objects = []  # List of TrackedObject
        # Timestamp
        timestamp = self.base_timestamp + current_step * self.scene["sample_rate"] * 0.05 * 1e6
        agent_map = {agent.id: agent for agent in self.engine.agent_manager.all_agents.values()}

        for object_id, object_data in self.scene["object_track"].items():
            agent = agent_map.get(object_id)
            if agent is None:
                continue  
            
            object_type = object_data["type"]
            tracked_object_type = tracked_object_type_mapping.get(object_type)
            if tracked_object_type is None:
                raise ValueError(f"Unknown object type: {object_type}")
            
            position = agent.current_position
            heading = agent.current_heading
            velocity = agent.current_velocity
            length = agent._length
            width = agent._width
            height = agent._height

            position = denormalize_from_ego_center(position, self.initial_ego_center)

            # Create OrientedBox
            box = OrientedBox(
            center = StateSE2(x=position[0], y=position[1], heading=heading),
            length = float(length),
            width = float(width),
            height = float(height)
            )

            # Create TrackedObject based on type
            if object_type in ["VEHICLE", "PEDESTRIAN", "CYCLIST", "EGO"]:
                tracked_object = Agent(
                    tracked_object_type=tracked_object_type,
                    oriented_box=box,
                    velocity=StateVector2D(x=velocity[0], y=velocity[1]),
                    metadata=SceneObjectMetadata(
                        timestamp_us = timestamp,
                        token = self.scene["token"],
                        track_id = object_id,
                        track_token = object_id
                    )
                )
            else:
                tracked_object = StaticObject(
                    tracked_object_type = tracked_object_type,
                    oriented_box=box,
                    metadata=SceneObjectMetadata(
                        timestamp_us = timestamp,
                        token = self.scene["token"],
                        track_id = object_id,
                        track_token = object_id
                    )
                )

            # Append to the tracked objects list
            tracked_objects.append(tracked_object)

        # Wrap in TrackedObjects
        tracked_objects_container = TrackedObjects(tracked_objects=tracked_objects)

        # Wrap in DetectionsTracks
        detections_tracks = DetectionsTracks(tracked_objects=tracked_objects_container)

        return detections_tracks
    
    def convert_to_traffic_lights(self, current_step: int) -> DetectionsTracks:
        """
        Convert WorldEngine traffic_light data to nuPlan DetectionsTracks format.

        Args:
            current_step: Current simulation step to extract the corresponding state.

        Returns:
            DetectionsTracks: A DetectionsTracks object for the current step.
        """
        traffic_lights = []
        # Timestamp
        timestamp = self.base_timestamp + current_step * self.scene["sample_rate"] * 0.05 * 1e6
        
        for _, map_object_data in self.scene["dynamic_map_states"].items():
            if map_object_data["type"] == "TRAFFIC_LIGHT":
                state = map_object_data["state"]
                traffic_light_state = state["traffic_light_state"]

                tracked_map_object_type_mapping = {
                "TRAFFIC_LIGHT_UNKNOWN": TrafficLightStatusType.UNKNOWN,
                "TRAFFIC_LIGHT_GREEN": TrafficLightStatusType.GREEN,
                "TRAFFIC_LIGHT_RED": TrafficLightStatusType.RED,
                }

                status = tracked_map_object_type_mapping.get(traffic_light_state[current_step])
                if status is None:
                    raise ValueError(f"Unknown object type: {traffic_light_state[current_step]}")
                traffic_light = TrafficLightStatusData(
                    status = status,
                    lane_connector_id = map_object_data["traffic_light_lane"],
                    timestamp = timestamp,
                )
                traffic_lights.append(traffic_light)
        
        return traffic_lights
    
    def convert_to_trajectory(self, pdm_path):

        """
        Convert InterpolatedTrajectory to Trajectory.
        """
        pdm_trajectory = pdm_path._trajectory
        waypoints = []
        velocities = []
        headings = []
        for _, ego_state in enumerate(pdm_trajectory):
            waypoint = ego_state.waypoint
            x_world, y_world = waypoint.x, waypoint.y
            position = normalize_to_ego_center([x_world, y_world], self.initial_ego_center)
            waypoints.append(position)
            velocities.append([waypoint.velocity.x, waypoint.velocity.y])
            headings.append(waypoint.heading)
        
        waypoints = np.array(waypoints, dtype=np.float32)
        velocities = np.array(velocities, dtype=np.float32)
        headings = np.array(headings, dtype=np.float32)

        return Trajectory(
                waypoints=waypoints,
                velocities=velocities,
                headings=headings,
                angular_velocities=None
        )

    
    