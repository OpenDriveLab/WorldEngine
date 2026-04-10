import math
from typing import Dict, List, cast, Generator

import numpy as np
from pyquaternion import Quaternion
from tqdm import tqdm
import os

from shapely.geometry.linestring import LineString
from shapely.geometry.multilinestring import MultiLineString
from shapely.ops import unary_union
import geopandas as gpd

from nuplan.common.actor_state.state_representation import StateSE2, StateVector2D, TimePoint
from nuplan.common.maps.nuplan_map.map_factory import get_maps_api
from nuplan.database.nuplan_db_orm.nuplandb_wrapper import NuPlanDBWrapper
from nuplan.common.actor_state.agent import Agent
from nuplan.common.actor_state.state_representation import Point2D
from nuplan.common.actor_state.static_object import StaticObject
from nuplan.database.nuplan_db import nuplan_scenario_queries
from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters

from nuplan.common.maps.maps_datatypes import TrafficLightStatusData
from nuplan.common.maps.maps_datatypes import SemanticMapLayer, StopLineType

from worldengine.utils.type import WorldEngineObjectType
from worldengine.utils.dataset_utils.nuplan.nuplan_type_mapping import (
    get_traffic_obj_type, NuPlanEgoType, set_light_status)

EGO = 'ego'

NUPLAN_REAL_LIDAR2EGO_ROTATION = [-0.0016505558783280307, -0.00023289146777086609, 0.003725490480134295, 0.9999916710390838]
NUPLAN_REAL_LIDAR2EGO_TRANSLATION = [1.5185133218765259, 0.0, 1.6308990716934204]

def normalize_to_ego_center(vector, ego_center=(0, 0)):
    "Normalize position-related vectors to the ego_center of the first frame."
    vector = np.array(vector)
    vector -= np.asarray(ego_center)
    return vector


def parse_object_state(obj_state, ego_center):
    ret = {}
    ret["position"] = normalize_to_ego_center(
        [obj_state.center.x, obj_state.center.y], ego_center)
    ret["heading"] = obj_state.center.heading
    ret["velocity"] = normalize_to_ego_center(
        [obj_state.velocity.x, obj_state.velocity.y])
    ret["valid"] = 1
    ret["length"] = obj_state.box.length
    ret["width"] = obj_state.box.width
    ret["height"] = obj_state.box.height
    return ret


def parse_ego_vehicle_state(state, ego_center):
    ret = {}
    ret["position"] = normalize_to_ego_center([state.waypoint.x, state.waypoint.y], ego_center)
    ret["heading"] = state.waypoint.heading
    ret["velocity"] = normalize_to_ego_center([state.agent.velocity.x, state.agent.velocity.y])
    ret["angular_velocity"] = state.dynamic_car_state.angular_velocity
    ret["valid"] = 1
    ret["length"] = state.agent.box.length
    ret["width"] = state.agent.box.width
    ret["height"] = state.agent.box.height
    return ret


def compute_angular_velocity(initial_heading, final_heading, dt):
    """
    Calculate the angular velocity between two headings given in radians.

    Parameters:
    initial_heading (float): The initial heading in radians.
    final_heading (float): The final heading in radians.
    dt (float): The time interval between the two headings in seconds.

    Returns:
    float: The angular velocity in radians per second.
    """

    # Calculate the difference in headings
    delta_heading = final_heading - initial_heading

    # Adjust the delta_heading to be in the range (-π, π]
    delta_heading = (delta_heading + math.pi) % (2 * math.pi) - math.pi

    # Compute the angular velocity
    angular_vel = delta_heading / dt

    return angular_vel


def parse_ego_vehicle_state_trajectory(log_file, lidar_pcs, ego_center, sample_interval):
    egos = [lidar_pc.ego_pose for lidar_pc in lidar_pcs]

    # Then convert ego_status to carfootprint.
    ego_boxes = []
    for ego in egos:
        q = Quaternion(ego.qw, ego.qx, ego.qy, ego.qz)
        ego_boxes.append(
            EgoState.build_from_rear_axle(
                StateSE2(ego.x, ego.y, q.yaw_pitch_roll[0]),
                tire_steering_angle=0.0,
                vehicle_parameters=get_pacifica_parameters(),
                time_point=TimePoint(ego.timestamp),
                rear_axle_velocity_2d=StateVector2D(ego.vx, y=ego.vy),
                rear_axle_acceleration_2d=StateVector2D(
                    x=ego.acceleration_x, y=ego.acceleration_y),
            )
        )

    data = [
        parse_ego_vehicle_state(ego_box, ego_center)
        for ego_box in ego_boxes
    ]
    dt = sample_interval * 0.05
    for i in range(len(data) - 1):
        data[i]["angular_velocity"] = compute_angular_velocity(
            initial_heading=data[i]["heading"],
            final_heading=data[i + 1]["heading"],
            dt=dt
        )
    return data


def extract_traffic(log_file, lidar_pcs, initial_ego_center, sample_interval):
    """
    Modified from:
    https://github.com/metadriverse/scenarionet/blob/main/scenarionet/converter/nuplan/utils.py#L388

    A function to store the trajectory of each object appeared in the scene.
    All coordinates are normalizaed to the ego_center at the first frame.
    """
    log_length = len(lidar_pcs)

    detection_ret = []
    all_objs = set()  # for storing the track ids.
    all_objs.add(EGO)  # key of sdc.

    for lidar_pc in lidar_pcs:
        lidar_boxes = lidar_pc.lidar_boxes
        new_frame_data = {}
        for obj in lidar_boxes:
            new_frame_data[obj.track_token] = obj
            all_objs.add(obj.track_token)
        detection_ret.append(new_frame_data)

    tracks = {
        k: dict(
            type=WorldEngineObjectType.UNSET,
            state=dict(
                position=np.zeros(shape=(log_length, 3)),
                heading=np.zeros(shape=(log_length, )),
                velocity=np.zeros(shape=(log_length, 2)),
                valid=np.zeros(shape=(log_length, )),
                length=np.zeros(shape=(log_length, 1)),
                width=np.zeros(shape=(log_length, 1)),
                height=np.zeros(shape=(log_length, 1))
            ),
            metadata=dict(track_length=log_length, nuplan_type=None, type=None, object_id=k, nuplan_id=k)
        )
        for k in list(all_objs)
    }

    tracks_to_remove = set()

    for frame_idx, frame in enumerate(detection_ret):
        for nuplan_id, obj, in frame.items():
            tracked_objs = obj.tracked_object(None)
            assert isinstance(tracked_objs, Agent) or isinstance(tracked_objs, StaticObject)
            obj_type = get_traffic_obj_type(tracked_objs.tracked_object_type)
            if obj_type is None:
                tracks_to_remove.add(nuplan_id)
                continue
            tracks[nuplan_id]['type'] = obj_type
            if tracks[nuplan_id]['metadata']["nuplan_type"] is None:
                tracks[nuplan_id]['metadata']["nuplan_type"] = int(tracked_objs.tracked_object_type)
                tracks[nuplan_id]['metadata']["type"] = obj_type

            # normalize the tracked object position to the ego_center
            #  of the first frame ego.
            state = parse_object_state(tracked_objs, initial_ego_center)
            tracks[nuplan_id]["state"]["position"][frame_idx] = [state["position"][0], state["position"][1], 0.0]
            tracks[nuplan_id]["state"]["heading"][frame_idx] = state["heading"]
            tracks[nuplan_id]["state"]["velocity"][frame_idx] = state["velocity"]
            tracks[nuplan_id]["state"]["valid"][frame_idx] = 1
            tracks[nuplan_id]["state"]["length"][frame_idx] = state["length"]
            tracks[nuplan_id]["state"]["width"][frame_idx] = state["width"]
            tracks[nuplan_id]["state"]["height"][frame_idx] = state["height"]

    for track in list(tracks_to_remove):
        tracks.pop(track)

    # ego
    sdc_traj = parse_ego_vehicle_state_trajectory(
        log_file, lidar_pcs, initial_ego_center, sample_interval)
    ego_track = tracks[EGO]

    for frame_idx, obj_state in enumerate(sdc_traj):
        obj_type = WorldEngineObjectType.VEHICLE
        ego_track['type'] = obj_type
        if ego_track['metadata']["nuplan_type"] is None:
            ego_track['metadata']["nuplan_type"] = int(NuPlanEgoType)
            ego_track['metadata']["type"] = obj_type
        state = obj_state
        ego_track["state"]["position"][frame_idx] = [state["position"][0], state["position"][1], 0.0]
        ego_track["state"]["valid"][frame_idx] = 1
        ego_track["state"]["heading"][frame_idx] = state["heading"]
        local2global_rotation_2d = np.array([
            [np.cos(state["heading"]), -np.sin(state["heading"])],
            [np.sin(state["heading"]), np.cos(state["heading"])]
        ])
        ego_track["state"]["velocity"][frame_idx] = state["velocity"] @ local2global_rotation_2d.T

        ego_track["state"]["length"][frame_idx] = state["length"]
        ego_track["state"]["width"][frame_idx] = state["width"]
        ego_track["state"]["height"][frame_idx] = state["height"]

    # check
    assert EGO in tracks
    for track_id in tracks:
        assert tracks[track_id]['type'] != WorldEngineObjectType.UNSET
    return tracks


def set_light_position(map_api, lane_id, center, target_position=8):
    lane = map_api.get_map_object(str(lane_id), SemanticMapLayer.LANE_CONNECTOR)
    assert lane is not None, "Can not find lane: {}".format(lane_id)
    path = lane.baseline_path.discrete_path
    acc_length = 0
    point = [path[0].x, path[0].y]
    # Find the distance of the path's different target points;
    # as long as the accumulated distance is greater than target_position,
    # mark it as the traffic light position.
    for k, point in enumerate(path[1:], start=1):
        previous_p = path[k - 1]
        acc_length += np.linalg.norm([point.x - previous_p.x, point.y - previous_p.y])
        if acc_length > target_position:
            break
    return [point.x - center[0], point.y - center[1]]


def extract_traffic_light(
        log_file, lidar_pcs, map_api, initial_ego_center):
    log_length = len(lidar_pcs)

    frames = [
        {str(t.lane_connector_id): t.status
         for t in
            cast(
                Generator[TrafficLightStatusData, None, None],
                nuplan_scenario_queries.get_traffic_light_status_for_lidarpc_token_from_db(
                    log_file, lidar_pcs[i].token),
            )
         }
        for i in range(log_length)
    ]

    all_lights = set()
    for frame in frames:
        all_lights.update(frame.keys())

    lights = {
        k: {
            "type": WorldEngineObjectType.TRAFFIC_LIGHT,
            "state": {
                'traffic_light_state': [WorldEngineObjectType.LIGHT_UNKNOWN] * log_length
            },
            'traffic_light_position': None,
            'traffic_light_lane': str(k),
            "metadata": dict(track_length=log_length,
                             type=None,
                             object_id=str(k),
                             lane_id=str(k),
                             dataset="nuplan")
        }
        for k in list(all_lights)
    }

    for k, frame in enumerate(frames):
        for lane_id, status in frame.items():
            lane_id = str(lane_id)
            lights[lane_id]["state"]['traffic_light_state'][k] = set_light_status(status)
            if lights[lane_id]['traffic_light_position'] is None:
                assert isinstance(lane_id, str), "Lane ID should be str"
                lights[lane_id]['traffic_light_position'] = set_light_position(
                    map_api, lane_id, initial_ego_center)
                lights[lane_id]['metadata']['type'] = WorldEngineObjectType.TRAFFIC_LIGHT
    return lights


def extract_centerline(map_obj, nuplan_center):
    path = map_obj.baseline_path.discrete_path
    points = np.array([normalize_to_ego_center(
        [pose.x, pose.y], nuplan_center) for pose in path])
    return points


def get_points_from_boundary(boundary, center):
    path = boundary.discrete_path
    points = [(pose.x, pose.y) for pose in path]
    points = normalize_to_ego_center(points, center)
    return points


def extract_map_features(map_api, initial_ego_center, radius=500):
    ret = {}
    np.seterr(all='ignore')
    # Center is Important !
    layer_names = [
        SemanticMapLayer.LANE_CONNECTOR,  # connectors between different lanes.
        SemanticMapLayer.LANE,  # basic element of traffic lanes.
        SemanticMapLayer.CROSSWALK,  # crosswalk for pedestrian on intersections.
        SemanticMapLayer.INTERSECTION,  # intersections.
        SemanticMapLayer.STOP_LINE,  # stop line on roads.
        SemanticMapLayer.WALKWAYS,  # pedestrian walking ways
        SemanticMapLayer.CARPARK_AREA,
        SemanticMapLayer.ROADBLOCK,  # lanes following the same direction.
        SemanticMapLayer.ROADBLOCK_CONNECTOR,  # connectors of blocks

        # unsupported yet
        # SemanticMapLayer.STOP_SIGN,
        # SemanticMapLayer.DRIVABLE_AREA,
    ]
    center_for_query = Point2D(*initial_ego_center)
    nearest_vector_map = map_api.get_proximal_map_objects(center_for_query, radius, layer_names)
    # Filter out stop polygons in turn stop
    if SemanticMapLayer.STOP_LINE in nearest_vector_map:
        stop_polygons = nearest_vector_map[SemanticMapLayer.STOP_LINE]
        nearest_vector_map[SemanticMapLayer.STOP_LINE] = [
            stop_polygon for stop_polygon in stop_polygons if stop_polygon.stop_line_type != StopLineType.TURN_STOP
        ]
    block_polygons = []

    # Deal with lanes in the roadblocks.
    # Each lane in nuplan is a centerline with polygon.
    # You can also obtain its left / right border with LINE_BROKEN_SINGLE_WHITE.
    for layer in [SemanticMapLayer.ROADBLOCK, SemanticMapLayer.ROADBLOCK_CONNECTOR]:
        for block in nearest_vector_map[layer]:
            block_id = block.id
            # Sorting the inside edges from left-to-right.
            edges = sorted(block.interior_edges, key=lambda lane: lane.index) \
                if layer == SemanticMapLayer.ROADBLOCK else block.interior_edges
            for index, lane_meta_data in enumerate(edges):
                if not hasattr(lane_meta_data, "baseline_path"):
                    continue
                # points:
                #  a list of xy coordinates of the lane.
                #  x: a list of numbers / y: a list of numbers.
                if isinstance(lane_meta_data.polygon.boundary, MultiLineString):
                    # Multiple lane strings.
                    # lane boundary polygons.
                    boundary = gpd.GeoSeries(lane_meta_data.polygon.boundary).explode(index_parts=True)
                    sizes = []
                    for idx, polygon in enumerate(boundary[0]):
                        sizes.append(len(polygon.xy[1]))
                    points = boundary[0][np.argmax(sizes)].xy
                elif isinstance(lane_meta_data.polygon.boundary, LineString):
                    # one lane string.
                    points = lane_meta_data.polygon.boundary.xy
                polygon = [[points[0][i], points[1][i]] for i in range(len(points[0]))]
                polygon = normalize_to_ego_center(polygon, ego_center=initial_ego_center)

                # According to the map attributes, lanes are numbered left to right with smaller indices being on the
                # left and larger indices being on the right.
                # @ See NuPlanLane.adjacent_edges()
                ret[lane_meta_data.id] = {
                    'type': WorldEngineObjectType.LANE_SURFACE_STREET \
                        if layer == SemanticMapLayer.ROADBLOCK \
                        else WorldEngineObjectType.LANE_SURFACE_UNSTRUCTURE,
                    'polyline': extract_centerline(lane_meta_data, initial_ego_center),
                    'entry_lanes': [edge.id for edge in lane_meta_data.incoming_edges],
                    'exit_lanes': [edge.id for edge in lane_meta_data.outgoing_edges],
                    'left_neighbor': [edge.id for edge in block.interior_edges[:index]] \
                        if layer == SemanticMapLayer.ROADBLOCK else [],
                    'right_neighbor': [edge.id for edge in block.interior_edges[index + 1:]] \
                        if layer == SemanticMapLayer.ROADBLOCK else [],
                    'polygon': polygon,
                    'roadblock_id': block_id 
                }
                if layer == SemanticMapLayer.ROADBLOCK_CONNECTOR:
                    continue
                left = lane_meta_data.left_boundary
                if left.id not in ret:
                    # boundary lane of centerline.
                    line_type = WorldEngineObjectType.LINE_BROKEN_SINGLE_WHITE
                    if line_type != WorldEngineObjectType.LINE_UNKNOWN:
                        ret[left.id] = {
                            'type': line_type,
                            'polyline': get_points_from_boundary(left, initial_ego_center)}

            if layer == SemanticMapLayer.ROADBLOCK:
                block_polygons.append(block.polygon)

    # walkway
    for area in nearest_vector_map[SemanticMapLayer.WALKWAYS]:
        if isinstance(area.polygon.exterior, MultiLineString):
            boundary = gpd.GeoSeries(area.polygon.exterior).explode(index_parts=True)
            sizes = []
            for idx, polygon in enumerate(boundary[0]):
                sizes.append(len(polygon.xy[1]))
            points = boundary[0][np.argmax(sizes)].xy
        elif isinstance(area.polygon.exterior, LineString):
            points = area.polygon.exterior.xy
        polygon = [[points[0][i], points[1][i]] for i in range(len(points[0]))]
        polygon = normalize_to_ego_center(polygon, ego_center=initial_ego_center)
        ret[area.id] = {
            'type': WorldEngineObjectType.BOUNDARY_SIDEWALK,
            'polygon': polygon,
        }

    # corsswalk
    for area in nearest_vector_map[SemanticMapLayer.CROSSWALK]:
        if isinstance(area.polygon.exterior, MultiLineString):
            boundary = gpd.GeoSeries(area.polygon.exterior).explode(index_parts=True)
            sizes = []
            for idx, polygon in enumerate(boundary[0]):
                sizes.append(len(polygon.xy[1]))
            points = boundary[0][np.argmax(sizes)].xy
        elif isinstance(area.polygon.exterior, LineString):
            points = area.polygon.exterior.xy
        polygon = [[points[0][i], points[1][i]] for i in range(len(points[0]))]
        polygon = normalize_to_ego_center(polygon, ego_center=initial_ego_center)
        ret[area.id] = {
            'type': WorldEngineObjectType.CROSSWALK,
            'polygon': polygon,
        }

    # intersections (centerline)
    interpolygons = [block.polygon for block in nearest_vector_map[SemanticMapLayer.INTERSECTION]]
    # A group of linestrings.
    boundaries = gpd.GeoSeries(unary_union(interpolygons + block_polygons)).boundary.explode(index_parts=True)
    # boundaries.plot()
    # plt.show()
    for idx, boundary in enumerate(boundaries[0]):
        # for each boundary.
        block_points = np.array(list(i for i in zip(boundary.coords.xy[0], boundary.coords.xy[1])))
        block_points = normalize_to_ego_center(block_points, initial_ego_center)
        id = "boundary_{}".format(idx)
        ret[id] = {
            'type': WorldEngineObjectType.LINE_SOLID_SINGLE_WHITE,
            'polyline': block_points}
    np.seterr(all='warn')
    return ret


# The main method to convert nuplan data into WorldEngine required pickle file.
def create_nuplan_info(
        nuplan_db_wrapper: NuPlanDBWrapper, db_names: List[str], args, return_dict,
):

    nuplan_db_path = args.nuplan_db_path
    nuplan_map_version = args.nuplan_map_version
    nuplan_map_root = args.nuplan_map_root
    nuplan_sensor_root = args.nuplan_sensor_path

    # get all db files & assign db files for current thread.
    log_sensors = os.listdir(nuplan_sensor_root)

    # For each sequence...
    for log_db_name in tqdm(db_names):
        log_db = nuplan_db_wrapper.get_log_db(log_db_name)
        log_name = log_db.log_name
        log_token = log_db.log.token
        map_location = log_db.log.map_version
        vehicle_name = log_db.log.vehicle_name

        map_api = get_maps_api(nuplan_map_root, "nuplan-maps-v1.0", map_location)  # NOTE: lru cached

        log_file = os.path.join(nuplan_db_path, log_db_name + ".db")
        if log_db_name not in log_sensors:
            continue

        # list (sequence) of point clouds (each frame).
        lidar_pcs = log_db.lidar_pc
        lidar_pcs = lidar_pcs[::args.sample_interval]
        log_length = len(lidar_pcs)

        # Store some meta information here
        info_dict = dict(
            id=log_name,
            name=log_name,
            dataset='nuplan',
            map=map_location,
            token=log_token,
            log_length=log_length,
            sample_rate=args.sample_interval,
            base_timestamp=lidar_pcs[0].timestamp,
            metadata=dict()
        )

        ego_pose = lidar_pcs[0].ego_pose
        q = Quaternion(ego_pose.qw, ego_pose.qx, ego_pose.qy, ego_pose.qz)
        initial_ego_state = EgoState.build_from_rear_axle(
            StateSE2(ego_pose.x, ego_pose.y, q.yaw_pitch_roll[0]),
            tire_steering_angle=0.0,
            vehicle_parameters=get_pacifica_parameters(),
            time_point=TimePoint(ego_pose.timestamp),
            rear_axle_velocity_2d=StateVector2D(ego_pose.vx, y=ego_pose.vy),
            rear_axle_acceleration_2d=StateVector2D(
                x=ego_pose.acceleration_x, y=ego_pose.acceleration_y),
        )
        initial_center = [initial_ego_state.waypoint.x,
                          initial_ego_state.waypoint.y]
        info_dict['metadata']['old_origin_in_current_coordinate'] = -np.asarray(initial_center)

        # do ego sensor info extraction.
        log_cam_infos = {camera.token : camera for camera in log_db.log.cameras}
        cams = {}
        for cam_info in log_cam_infos.values():
            cam_name = cam_info.channel
            cams[cam_name] = {}
            cams[cam_name]['channel'] = cam_info.channel
            cams[cam_name]['sensor2ego_rotation'] = cam_info.quaternion.elements
            cams[cam_name]['sensor2ego_translation'] = cam_info.translation_np
            cams[cam_name]['intrinsic'] = cam_info.intrinsic_np
            cams[cam_name]['distortion'] = cam_info.distortion_np
            cams[cam_name]['height'] = 1080
            cams[cam_name]['width'] = 1920
        info_dict['cameras'] = cams
        info_dict['lidar'] = {
            'channel': 'LIDAR_TOP',
            'sensor2ego_rotation': np.array(NUPLAN_REAL_LIDAR2EGO_ROTATION),
            'sensor2ego_translation': np.array(NUPLAN_REAL_LIDAR2EGO_TRANSLATION),
        }

        # do object track statistics.
        info_dict.update(dict(
            object_track=extract_traffic(
                log_file=log_file,
                lidar_pcs=lidar_pcs,
                initial_ego_center=initial_center,
                sample_interval=args.sample_interval),
            sdc_id=EGO,
        ))

        # do traffic light statistics.
        #  dynamic_map_states: elements related to traffic light,
        #  some changeable components in the scene.
        info_dict.update(dict(
            dynamic_map_states=extract_traffic_light(
                log_file=log_file,
                lidar_pcs=lidar_pcs,
                map_api=map_api,
                initial_ego_center=initial_center,
            )
        ))

        # do map element extraction.
        # extract polygons of each map element.
        info_dict.update(
            map_features=extract_map_features(
                map_api=map_api,
                initial_ego_center=initial_center)
        )

        del map_api
        return_dict[log_name] = info_dict
