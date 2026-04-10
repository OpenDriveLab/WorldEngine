import argparse
import glob
import os
import pickle
from typing import Dict, List
from tqdm import tqdm
from pathlib import Path
from multiprocessing import Pool

import numpy as np
import scipy
from pyquaternion import Quaternion

from utils.miscell import find_unique_common_from_lists, points_in_single_box_obb
from utils.nuplan_pointcloud import PointCloud


FPS = 2
FRAME_INTERVAL = 1
FPS_KEYFRAME = FPS / FRAME_INTERVAL
WORLDENGINE_ROOT = os.getenv('WORLDENGINE_ROOT', os.path.abspath('.'))

parser = argparse.ArgumentParser()
parser.add_argument(
    "--data_root",
    type=str,
    default=f"{WORLDENGINE_ROOT}/data/openscene-v1.1",
    help="root directory of raw openscene data",
)
parser.add_argument(
    "--split",
    type=str,
    default="trainval",
    help="trainval/test",
)
parser.add_argument(
    "--num_workers",
    type=int,
    default=16,
    help="multi-processing",
)
parser.add_argument(
    "--filter_boxes",
    action="store_true",
    help="filter boxes with lidar points",
)
parser.add_argument(
    "--output_folder",
    type=str,
    default="meta_datas_navformer",
    help="output folder name",
)
args = parser.parse_args()


def parse_ego_sensor_calib(anno: Dict, data_split: str) -> Dict:
    """Parse calibration between world, ego, lidar and camera"""

    # from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters
    x, y, z = 0., 0., 0.
    l, w, h = 5.176, 2.297, 1.777
    sdc_vel_ego = anno["can_bus"][10:13]
    sdc_vel_global = anno["ego2global"][:3, :3] @ sdc_vel_ego
    sdc_vel_lidar = anno["lidar2ego"][:3, :3].T @ sdc_vel_ego

    gt_sdc_bbox_lidar = np.array([
            x, y, z,
            l, w, h,
            0.,
            sdc_vel_lidar[0], sdc_vel_lidar[1],
        ], dtype=np.float64)

    # load camera calib
    cams = anno["cams"]
    for cam_type, cam_dict in cams.items():
        sensor2lidar = np.identity(4)
        sensor2lidar[:3, :3] = cam_dict["sensor2lidar_rotation"]
        sensor2lidar[:3, 3] = cam_dict["sensor2lidar_translation"]
        sensor2ego: np.ndarray = anno["lidar2ego"] @ sensor2lidar

        # construct nuScenes-like camera record
        cams[cam_type].update(
            {
                "type": cam_type,
                "sensor2ego_translation": sensor2ego[:3, 3],
                "sensor2ego_rotation": Quaternion(matrix=sensor2ego[:3, :3]).elements,
            }
        )

    pts_filename = anno["lidar_path"]  # log_name + token
    data_root = os.path.join(args.data_root, f"sensor_blobs/{data_split}")
    if args.filter_boxes:
        if data_root not in pts_filename:
            pts_filename = os.path.join(data_root, pts_filename)
        points = PointCloud.parse_from_file(pts_filename).to_pcd_bin2()  # 6 x N
        points = points[:3].T  # N x 3
    else:
        points = np.zeros((0, 3))

    command = np.argmax(anno["driving_command"])  # int
    # update dictionary
    anno.update(
        {
            "lidar_pts": points,
            "sweeps": [],
            "cams": cams,
            "sdc_vel_global": sdc_vel_global,
            # sdc's information in lidar coordinate
            "gt_sdc_bbox_lidar": gt_sdc_bbox_lidar,  # 9
            "command": command,  # int, convert 1-indexed -> 0-indexed
        }
    )

    return anno


def parse_bbox(
    anno: Dict,
    global_track_id: int,
    mapping_tracktoken2globalid: Dict[str, int],
    data_split: str,
):
    """Parse each one of the bounding boxes and compute properties"""
    num_obj: int = np.array(anno["anns"]["gt_boxes"]).shape[0]
    anno.update(
        {
            "gt_velocity": np.array(anno["anns"]["gt_velocity_3d"])[:, :2],  # lidar coordinate
            "gt_boxes": anno["anns"]["gt_boxes"],  # lidar coordinate (x,y,z,l,w,h,yaw)
            "gt_bboxes_global": np.zeros((num_obj, 9), dtype=np.float64),
            "gt_names": anno["anns"]["gt_names"],
            "gt_inds": np.zeros((num_obj), dtype=int),
            "num_lidar_pts": np.zeros((num_obj,), dtype=int),
            "valid_flag": np.zeros((num_obj,), dtype=bool),
        }
    )

    ego_yaw = scipy.spatial.transform.Rotation.from_matrix(
        anno["lidar2global"][:3, :3]
    ).as_euler("zyx", degrees=False)[0]
    lidar2global = anno["lidar2global"]

    box_7dof_lidar = anno["anns"]["gt_boxes"]
    box_velo_3d = anno["anns"]["gt_velocity_3d"]

    bbox_center_global = box_7dof_lidar[:, :3] @ lidar2global[:3, :3].T  + lidar2global[:3, 3]
    bbox_yaw_world = box_7dof_lidar[:, -1] + ego_yaw
    bbox_velocity_world = box_velo_3d @ lidar2global[:3, :3].T
    anno["gt_bboxes_global"][:, :3] = bbox_center_global
    anno["gt_bboxes_global"][:, 3:6] = box_7dof_lidar[:, 3:6]
    anno["gt_bboxes_global"][:, 6] = bbox_yaw_world
    anno["gt_bboxes_global"][:, 7:] = bbox_velocity_world[:, :2]

    # go through all bbox to fill in database info
    for bbox_index in range(num_obj):
        track_token = anno["anns"]["track_tokens"][bbox_index]
        # assign global ID
        if track_token not in mapping_tracktoken2globalid:
            mapping_tracktoken2globalid[track_token] = global_track_id
            anno["gt_inds"][bbox_index] = global_track_id

            # update track_id for the next object
            global_track_id += 1
        else:
            anno["gt_inds"][bbox_index] = mapping_tracktoken2globalid[track_token]

        if args.filter_boxes:
            points_in_box = points_in_single_box_obb(
                points_xyz=anno["lidar_pts"],
                box_xyzlwhyaw=anno["anns"]["gt_boxes"][bbox_index],
            )
            anno["num_lidar_pts"][bbox_index] = points_in_box
            anno["valid_flag"][bbox_index] = points_in_box > 0
        else:
            anno["valid_flag"][bbox_index] = True

    # delete lidar points in the dicionary to reduce file size prior to saving
    del anno["lidar_pts"]

    return anno, global_track_id, mapping_tracktoken2globalid


def parse_frame(
    anno: Dict,
    global_track_id: int,
    mapping_tracktoken2globalid: Dict[str, int],
    data_split: str
) -> Dict:
    """Convert carla data into the collection of path needed by mmdetection3D dataloader"""

    anno: Dict = parse_ego_sensor_calib(anno, data_split=data_split)
    anno, global_track_id, mapping_tracktoken2globalid = parse_bbox(
        anno,
        global_track_id=global_track_id,
        mapping_tracktoken2globalid=mapping_tracktoken2globalid,
        data_split=data_split
    )

    return anno, global_track_id, mapping_tracktoken2globalid


def parse_clip(
    seq: str,
    annos: List[Dict],
    data_split: str,
    pre_sec: float = 1.5,
    fut_sec: float = 4
) -> List[Dict]:
    """Add past/future trajectory into data at each frame to allow prediction"""

    for index in range(len(annos)):
        anno: Dict = annos[index]  # data for the frame

        # check if the data is from the same sequence
        log_name: str = anno["log_name"]  # the sequence name
        assert log_name == seq, "error, not the same sequence"

        # retrieve info for other objects
        # gt_velocity: np.ndarray = anno["gt_velocity"]  # N x 2
        gt_bboxes_lidar: np.ndarray = np.array(anno["anns"]["gt_boxes"])  # N x 7
        num_obj: int = gt_bboxes_lidar.shape[0]

        # convert the str ID to global ID
        obj_ids: List[int] = anno["gt_inds"].tolist()

        # retrieve info for ego and calib
        l, w, h = anno["gt_sdc_bbox_lidar"][3:6]
        lidar2global: np.ndarray = anno["lidar2global"]  # 4 x 4
        global2lidar: np.ndarray = np.eye(4)
        global2lidar[:3, :3] = lidar2global[:3, :3].T
        global2lidar[:3, 3] = -lidar2global[:3, :3].T @ lidar2global[:3, 3]

        lidar2global_yaw = scipy.spatial.transform.Rotation.from_matrix(
            lidar2global[:3, :3]
        ).as_euler("zyx", degrees=False)[0]

        ########### get the target data we need for GT
        pre_frames: int = int(pre_sec * FPS_KEYFRAME)
        fut_frames: int = int(fut_sec * FPS_KEYFRAME)

        # the past & future trajectories for objects existing in the current frame
        # so we know the exact number of objects
        gt_fut_bbox_lidar: np.ndarray = np.zeros((num_obj, fut_frames, 9), dtype=np.float64)
        gt_pre_bbox_lidar: np.ndarray = np.zeros((num_obj, pre_frames, 9), dtype=np.float64)

        # initially set all mask as invalid until we identify valid frames
        gt_fut_bbox_mask: np.ndarray = np.zeros((num_obj, fut_frames, 1), dtype=bool)
        gt_pre_bbox_mask: np.ndarray = np.zeros((num_obj, pre_frames, 1), dtype=bool)

        # sdc's temporal info
        gt_pre_bbox_sdc_lidar: np.ndarray = np.zeros((1, pre_frames, 9), dtype=np.float64)
        gt_fut_bbox_sdc_lidar: np.ndarray = np.zeros((1, fut_frames, 9), dtype=np.float64)
        gt_pre_bbox_sdc_global: np.ndarray = np.zeros((1, pre_frames, 9), dtype=np.float64)
        gt_fut_bbox_sdc_global: np.ndarray = np.zeros((1, fut_frames, 9), dtype=np.float64)
        gt_pre_bbox_sdc_mask: np.ndarray = np.zeros((1, pre_frames, 1), dtype=bool)
        gt_fut_bbox_sdc_mask: np.ndarray = np.zeros((1, fut_frames, 1), dtype=bool)
        gt_pre_command_sdc: np.ndarray = np.zeros((1, pre_frames, 1), dtype=int)
        gt_fut_command_sdc: np.ndarray = np.zeros((1, fut_frames, 1), dtype=int)

        # get past trajectory, backward order, i.e., frame 4, 3, 2, 1
        # start with 1 to not include the current frame
        for pre_frame_index in range(1, pre_frames + 1):
            pre_frame_index_in_seq: int = index - pre_frame_index

            if pre_frame_index_in_seq < 0:
                continue

            # retrieve the data in the global coordinate
            anno_pre_tmp: Dict = annos[pre_frame_index_in_seq]
            gt_pre_bbox_global_tmp: np.ndarray = anno_pre_tmp[
                "gt_bboxes_global"
            ]  # N x 9
            num_obj_pre: int = gt_pre_bbox_global_tmp.shape[0]
            gt_pre_bbox_lidar_tmp = np.zeros((num_obj_pre, 9))

            ########## convert the global coordinate to lidar coordinate in current frame
            # i.e., ego motion compensation

            # location
            gt_pre_bbox_lidar_tmp[:, :3] = gt_pre_bbox_global_tmp[:, :3] @ global2lidar[:3, :3].T + global2lidar[:3, 3]
            # size
            gt_pre_bbox_lidar_tmp[:, 3:6] = gt_pre_bbox_global_tmp[:, 3:6]
            # rotation (yaw)
            gt_pre_bbox_lidar_tmp[:, 6] = gt_pre_bbox_global_tmp[:, 6] - lidar2global_yaw
            # velocity
            gt_pre_bbox_global_velo3d = np.concatenate([gt_pre_bbox_global_tmp[:, 7:9], np.zeros((num_obj_pre, 1))], axis=1)
            gt_pre_bbox_lidar_tmp[:, 7:] = (gt_pre_bbox_global_velo3d @ global2lidar[:3, :3].T)[:, :2]  # N x 2

            # now check the IDs that exist in the current frame
            # in order to produce the mask for future frames
            obj_ids_pre: List[int] = anno_pre_tmp["gt_inds"].tolist()
            (
                obj_ids_common,
                index_cur,
                index_past,
            ) = find_unique_common_from_lists(obj_ids, obj_ids_pre)

            # possible that we get 0 object in future frame matching with GT object currint
            if len(index_cur) > 0:
                gt_pre_bbox_mask[np.array(index_cur), pre_frame_index - 1] = 1

                # also assign the actual gt boxes into the array
                gt_pre_bbox_lidar[
                    np.array(index_cur), pre_frame_index - 1, :
                ] = gt_pre_bbox_lidar_tmp[np.array(index_past), :]

            ############### get past of the sdc box in lidar coordinate
            sdc_loc_global = anno_pre_tmp["ego2global"][:3, 3]
            sdc_loc_lidar = global2lidar[:3, :3] @ sdc_loc_global + global2lidar[:3, 3]
            # velocity
            sdc_vel_global = anno_pre_tmp["sdc_vel_global"]  # 3
            sdc_vel_lidar = global2lidar[:3, :3] @ sdc_vel_global   # 3
            # rotation
            sdc_trans_lidar = global2lidar @ anno_pre_tmp["ego2global"]  # 4 x 4
            yaw_lidar = scipy.spatial.transform.Rotation.from_matrix(
                sdc_trans_lidar[:3, :3]
            ).as_euler("zyx", degrees=False)[0]
            yaw_global = scipy.spatial.transform.Rotation.from_matrix(
                anno_pre_tmp["ego2global"][:3, :3]
            ).as_euler("zyx", degrees=False)[0]
            gt_pre_bbox_sdc_mask[0, pre_frame_index - 1] = 1

            # put things into bbox, 9 dof
            gt_sdc_bbox_lidar = np.array([
                    sdc_loc_lidar[0], sdc_loc_lidar[1], sdc_loc_lidar[2],
                    l, w, h,
                    yaw_lidar,
                    sdc_vel_lidar[0], sdc_vel_lidar[1],
                ], dtype=np.float64)
            gt_pre_bbox_sdc_lidar[0, pre_frame_index - 1] = gt_sdc_bbox_lidar

            gt_sdc_bbox_global = np.array([
                    sdc_loc_global[0], sdc_loc_global[1], sdc_loc_global[2],
                    l, w, h,
                    yaw_global,
                    sdc_vel_global[0], sdc_vel_global[1],
                ], dtype=np.float64)
            gt_pre_bbox_sdc_global[0, pre_frame_index - 1] = gt_sdc_bbox_global

            # get command
            gt_pre_command_sdc[0, pre_frame_index - 1]: int = anno_pre_tmp["command"]

        # get future trajectory
        # start with 1 to not include the current frame
        for fut_frame_index in range(1, fut_frames + 1):
            fut_frame_index_in_seq: int = index + fut_frame_index
            if fut_frame_index_in_seq > len(annos) - 1:
                break

            anno_fut_tmp: Dict = annos[fut_frame_index_in_seq]

            gt_fut_bbox_global_tmp: np.ndarray = anno_fut_tmp[
                "gt_bboxes_global"
            ]  # N x 9
            num_obj_fut: int = gt_fut_bbox_global_tmp.shape[0]
            gt_fut_bbox_lidar_tmp = np.zeros((num_obj_fut, 9))

            ########## convert the global coordinate to lidar coordinate in current frame
            # i.e., ego motion compensation

            # location
            gt_fut_bbox_lidar_tmp[:, :3] = gt_fut_bbox_global_tmp[:, :3] @ global2lidar[:3, :3].T + global2lidar[:3, 3]
            # size
            gt_fut_bbox_lidar_tmp[:, 3:6] = gt_fut_bbox_global_tmp[:, 3:6]
            # rotation
            gt_fut_bbox_lidar_tmp[:, 6] = gt_fut_bbox_global_tmp[:, 6] - lidar2global_yaw
            # velocity
            gt_fut_bbox_global_velo3d = np.concatenate([gt_fut_bbox_global_tmp[:, 7:9], np.zeros((num_obj_fut, 1))], axis=1)
            gt_fut_bbox_lidar_tmp[:, 7:] = (gt_fut_bbox_global_velo3d @ global2lidar[:3, :3].T)[:, :2]  # N x 2

            # now check the IDs that exist in the current frame
            # in order to produce the mask for future frames
            obj_ids_fut: List[int] = anno_fut_tmp["gt_inds"].tolist()
            (
                obj_ids_common,
                index_cur,
                index_fut,
            ) = find_unique_common_from_lists(obj_ids, obj_ids_fut)

            # possible that we get 0 object in future frame matching with GT object currint
            if len(index_cur) > 0:
                gt_fut_bbox_mask[np.array(index_cur), fut_frame_index - 1] = 1
                # also assign the actual gt boxes into the array
                gt_fut_bbox_lidar[
                    np.array(index_cur), fut_frame_index - 1, :
                ] = gt_fut_bbox_lidar_tmp[np.array(index_fut), :]

            ############### get future of the sdc box in lidar coordinate
            sdc_loc_global = anno_fut_tmp["ego2global"][:3, 3]
            sdc_loc_lidar = global2lidar[:3, :3] @ sdc_loc_global + global2lidar[:3, 3]
            # velocity
            sdc_vel_global = anno_fut_tmp["sdc_vel_global"]  # 3
            sdc_vel_lidar = global2lidar[:3, :3] @ sdc_vel_global   # 3
            # rotation
            sdc_trans_lidar = global2lidar @ anno_fut_tmp["ego2global"]  # 4 x 4
            yaw_lidar = scipy.spatial.transform.Rotation.from_matrix(
                sdc_trans_lidar[:3, :3]
            ).as_euler("zyx", degrees=False)[0]
            yaw_global = scipy.spatial.transform.Rotation.from_matrix(
                anno_fut_tmp["ego2global"][:3, :3]
            ).as_euler("zyx", degrees=False)[0]
            gt_fut_bbox_sdc_mask[0, fut_frame_index - 1] = 1

            # put things into bbox, 9 dof
            gt_sdc_bbox_lidar = np.array([
                    sdc_loc_lidar[0], sdc_loc_lidar[1], sdc_loc_lidar[2],
                    l, w, h,
                    yaw_lidar,
                    sdc_vel_lidar[0], sdc_vel_lidar[1],
                ], dtype=np.float64)
            gt_fut_bbox_sdc_lidar[0, fut_frame_index - 1] = gt_sdc_bbox_lidar

            gt_sdc_bbox_global = np.array([
                    sdc_loc_global[0], sdc_loc_global[1], sdc_loc_global[2],
                    l, w, h,
                    yaw_global,
                    sdc_vel_global[0], sdc_vel_global[1],
                ], dtype=np.float64)
            gt_fut_bbox_sdc_global[0, fut_frame_index - 1] = gt_sdc_bbox_global

            # get command
            gt_fut_command_sdc[0, fut_frame_index - 1]: int = anno_fut_tmp["command"]

        # flip array to allow time in forward order for the past trajectory
        gt_pre_bbox_lidar = np.flip(gt_pre_bbox_lidar, axis=1)
        gt_pre_bbox_mask = np.flip(gt_pre_bbox_mask, axis=1)
        gt_pre_bbox_sdc_lidar = np.flip(gt_pre_bbox_sdc_lidar, axis=1)
        gt_pre_bbox_sdc_global = np.flip(gt_pre_bbox_sdc_global, axis=1)
        gt_pre_bbox_sdc_mask = np.flip(gt_pre_bbox_sdc_mask, axis=1)
        gt_pre_command_sdc = np.flip(gt_pre_command_sdc, axis=1)

        # assign the value back into the dictionary
        # in-place assignment to the sorted anno
        anno["gt_pre_bbox_lidar"] = gt_pre_bbox_lidar.astype(np.float32)  # N x 3 x 9
        anno["gt_fut_bbox_lidar"] = gt_fut_bbox_lidar.astype(np.float32)  # N x 8 x 9
        anno["gt_pre_bbox_mask"] = gt_pre_bbox_mask.astype(bool)  # N x 3 x 1
        anno["gt_fut_bbox_mask"] = gt_fut_bbox_mask.astype(bool)  # N x 8 x 1
        anno["gt_pre_bbox_sdc_lidar"] = gt_pre_bbox_sdc_lidar.astype(np.float32)  # 1 x 3 x 9
        anno["gt_fut_bbox_sdc_lidar"] = gt_fut_bbox_sdc_lidar.astype(np.float32)  # 1 x 8 x 9
        anno["gt_pre_bbox_sdc_global"] = gt_pre_bbox_sdc_global.astype(np.float64)  # 1 x 3 x 9
        anno["gt_fut_bbox_sdc_global"] = gt_fut_bbox_sdc_global.astype(np.float64)  # 1 x 8 x 9
        anno["gt_pre_bbox_sdc_mask"] = gt_pre_bbox_sdc_mask.astype(bool)  # 1 x 3 x 1
        anno["gt_fut_bbox_sdc_mask"] = gt_fut_bbox_sdc_mask.astype(bool)  # 1 x 8 x 1
        anno["gt_pre_command_sdc"] = gt_pre_command_sdc.astype(int)  # 1 x 3 x 1
        anno["gt_fut_command_sdc"] = gt_fut_command_sdc.astype(int)  # 1 x 8 x 1


    return annos


def parse_log(log_metadata_file: str) -> None:
    # one log has multiple data clips in nuplan, one clip has ~40 frames
    # But, the clips are continuous in nuplan, so we treat the log as a whole clip.

    # skip if finished earlier
    output_path = log_metadata_file.replace("meta_datas", args.output_folder)
    if os.path.exists(output_path):
        print(f"log_name done: {Path(output_path).resolve().name}")
        return

    # create output dir
    save_dir = Path(output_path).resolve().parent
    save_dir.mkdir(parents=True, exist_ok=True)

    # global variable
    global_track_id = 1
    mapping_tracktoken2globalid = dict()

    # load merged detection data
    with open(log_metadata_file, "rb") as f:
        raw_data_infos = pickle.load(f)
    clip_data_infos = []
    save_data_infos = []

    log_name_pre = None
    timestamp_pre = None
    # loop through every frame, to find the clips within each log
    for frame_info in raw_data_infos:
        # check if the driving logs are the same
        log_name = frame_info["log_name"]
        if log_name_pre is None:
            log_name_pre = log_name
        assert log_name_pre == log_name, "error, must be the same log"

        # check if the frame is continuous
        timestamp = frame_info["timestamp"]
        if timestamp_pre is None:
            timestamp_pre = timestamp
        timestamp_diff = timestamp - timestamp_pre
        if timestamp_diff * 1e-6 > (1 / FPS) * 1.5:
            print(f"missing frame in log {log_name}! timestamp_diff: {timestamp_diff * 1e-6:.3f}s")
            # process the clip with continuous frames
            clip_data_infos = parse_clip(log_name_pre, clip_data_infos, args.split)
            for data_dict in clip_data_infos:
                if data_dict is not None:
                    save_data_infos.append(data_dict)
            clip_data_infos = []
        timestamp_pre = timestamp

        frame_info, global_track_id, mapping_tracktoken2globalid = parse_frame(
            frame_info,
            global_track_id=global_track_id,
            mapping_tracktoken2globalid=mapping_tracktoken2globalid,
            data_split=args.split
        )
        clip_data_infos.append(frame_info)

    # process the one last data clip
    clip_data_infos: List[Dict] = parse_clip(log_name_pre, clip_data_infos, args.split)
    for data_dict in clip_data_infos:
        if data_dict is not None:
            save_data_infos.append(data_dict)
    print(
        f"log_name: {log_name_pre}, len: {len(raw_data_infos)}, after cleaning: {len(save_data_infos)}"
    )

    # save
    data_to_save = {
        "infos": save_data_infos,
        "mapping_tracktoken2globalid": mapping_tracktoken2globalid,
    }
    with open(output_path, "wb") as f:
        pickle.dump(data_to_save, f, protocol=pickle.HIGHEST_PROTOCOL)


def run_singleprocess(log_list: List[str]):
    for log_metadata_file in log_list:
        parse_log(log_metadata_file)


def run_multiprocess(log_list: List[str], num_workers: int = 1):
    log_count = len(log_list)
    chunksize = max(1, log_count // (num_workers * 4))
    with Pool(num_workers, maxtasksperchild=8) as p:
        for _ in tqdm(p.imap_unordered(parse_log, log_list, chunksize=chunksize), total=log_count):
            pass


if __name__ == "__main__":

    # OpenScene/nuPlan/NAVSIM:
    # mini_train: 43261 (43417 pre-cleaning) -> 6h
    # mini_val: 8450 -> 1.17h
    # val: 115564 (115733 pre-cleaning) -> 16h
    # train: 605263 (607286 pre-cleaning) -> 84h

    metadata_path = os.path.join(args.data_root, "meta_datas", args.split)

    # go through all driving logs
    log_list: List[str] = glob.glob(f"{metadata_path}/*.pkl")
    # run_singleprocess(log_list)   # single process for debugging
    run_multiprocess(log_list, args.num_workers)
