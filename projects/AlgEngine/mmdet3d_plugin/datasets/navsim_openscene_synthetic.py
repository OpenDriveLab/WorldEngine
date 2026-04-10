import os
import pickle
from typing import List, Union
import numpy as np
import pandas as pd
from glob import glob
import yaml
import cv2

import mmcv
from mmdet.datasets import DATASETS

from mmdet3d_plugin.datasets.navsim_openscene_finetuning import NavSimOpenSceneE2EFineTune
from mmdet3d_plugin.utils import get_logger
logger = get_logger(__name__)

@DATASETS.register_module()
class NavSimOpenSceneE2EFineTuneSynthetic(NavSimOpenSceneE2EFineTune):
    r"""
    OpenScene E2E Finetuning Synthetic Dataset
    This dataset is used for finetuning the model on the failed + synthetic data.
    It will randomly select normal data from the same log and add to the finetuning dataset.
    Please only use this dataset for training, not for val/test.
    """

    def __init__(self, *args, 
        folder_name: Union[str, List[str]] = None,
        use_lane_keeping = False,
        customized_filter = None,
        include_real_failures = True,
        **kwargs):

        # Synthetic:
        self.synthetic_path = 'data/alg_engine/openscene-synthetic/meta_datas'
        self.synthetic_img_root = 'data/alg_engine/openscene-synthetic/sensor_blobs'
        self.syn_pdm_path = 'data/alg_engine/openscene-synthetic/pdms_pkl'
        self.folder_names = folder_name
        if isinstance(self.folder_names, str):
            self.folder_names = [self.folder_names]

        
        self.use_lane_keeping = use_lane_keeping
        self.customized_filter = customized_filter
        self.include_real_failures = include_real_failures

        super(NavSimOpenSceneE2EFineTuneSynthetic, self).__init__(*args, **kwargs)

    def load_syntheitc_infos(self):
        data_infos = []
        valid_idx = []

        plan_dfs = {}
        for folder_name in self.folder_names:
            # Load plan_idx file
            plan_df = pd.read_csv(self.syn_pdm_path + f"/{folder_name}/plan_idx.csv")
            plan_dfs[folder_name] = plan_df

            folder_data_infos = []
            logger.info(f'loading synthetic data from {folder_name}')
            # Sometimes we pre-combined the pickles
            combined_pkl_path = f'{self.synthetic_path}/{folder_name}/combined.pkl'
            if os.path.exists(combined_pkl_path):
                logger.info('loading from combined pickle file')
                with open(combined_pkl_path, 'rb') as f:
                    folder_data_infos = pickle.load(f)
            else:
                logger.info('loading from splitted pickle files')
                pkls_path = glob(f'{self.synthetic_path}/{folder_name}/*.pkl')
                for p in pkls_path:
                    with open(p, 'rb') as f:
                        data = pickle.load(f)
                        for d in data:
                            d['syn_name'] = folder_name
                        folder_data_infos.extend(data)
            data_infos.extend(folder_data_infos)

        for i, info in enumerate(data_infos):
            # skip the first N frames, they are history frames
            if info['frame_idx'] < self.history_frame_num:
                continue

            # restore the original log_name
            # 2021.05.12.19.36.12_veh-35_00215_00405-a77d536b271d516e -> 2021.05.12.19.36.12_veh-35_00215_00405, a77d536b271d516e
            # we can still use scene_token / scene_name for temporal usage
            synthetic_log_name = info['log_name']
            parts = info['log_name'].split('-')
            log_name = '-'.join(parts[:2])
            original_token = parts[2]
            if original_token not in self.fail_scene_filter:
                continue

            synthetic_token = info['log_name'].split('-', 2)[-1]

            # we set frames after violation to invalid in WorldEngine
            if info.get('invalid', False):
                continue

            if 'pdm' not in info:
                pdm_reward_path = self.syn_pdm_path + f"/{info['syn_name']}/{synthetic_log_name}_step_{info['frame_idx']}_scores.pkl"
                if not os.path.exists(pdm_reward_path):
                    logger.warning(f'{pdm_reward_path} not found')
                    continue
                with open(pdm_reward_path, 'rb') as f:
                    # pdm reward for 8192 trajectories
                    pdm = pickle.load(f)
            else:
                pdm = info['pdm']
            score = pdm['score']

            # customized filter
            if self.customized_filter == "v1":
                # select the data with feasible planning (max score >= 0.9)
                # and the failure case (plan score <= 2/3 or ep < 0.2)
                plan_df = plan_dfs[info['syn_name']]
                log_idx = plan_df[plan_df['prefix'] == synthetic_token]
                plan_idx = int(log_idx[log_idx['step'] - 1 == info['frame_idx']]['plan_idx'])
                plan_score = score[plan_idx]
                max_score = np.max(score)
                low_ep = np.logical_and(pdm['drivable_area_compliance'][plan_idx] == 1, pdm['no_at_fault_collisions'][plan_idx] == 1)
                low_ep = np.logical_and(low_ep, pdm['ego_progress'][plan_idx] < 0.2)
                select = (max_score >= 0.9) and (plan_score <= 2/3 or low_ep)
                if not select:
                    continue
            elif self.customized_filter == "v2":
                # only keep the first frame of each scene
                if info['frame_idx'] != self.history_frame_num:
                    continue
            elif self.customized_filter == "v3":
                # keep the 0, 1, 2, 3 frames for each scene
                if info['frame_idx'] < self.history_frame_num + 4:
                    continue
                # still select the data with feasible planning (max score >= 0.9)
                # and the failure case (plan score <= 2/3 or ep < 0.2)
                plan_df = plan_dfs[info['syn_name']]
                log_idx = plan_df[plan_df['prefix'] == synthetic_token]
                plan_idx = int(log_idx[log_idx['step'] - 1 == info['frame_idx']]['plan_idx'])
                plan_score = score[plan_idx]
                max_score = np.max(score)
                low_ep = np.logical_and(pdm['drivable_area_compliance'][plan_idx] == 1, pdm['no_at_fault_collisions'][plan_idx] == 1)
                low_ep = np.logical_and(low_ep, pdm['ego_progress'][plan_idx] < 0.2)
                select = (max_score >= 0.9) and (plan_score <= 2/3 or low_ep)
                if not select:
                    continue

            if self.use_lane_keeping and 'lane_keeping' in pdm:
                lk_violation_mask = pdm['lane_keeping'] < 1
                pdm['drivable_area_compliance'][lk_violation_mask] = 0.5 * pdm['drivable_area_compliance'][lk_violation_mask]

            for k in pdm.keys():
                pdm[k] = np.array(pdm[k], dtype=np.float32)

            info['pdm'] = pdm
            info['log_name'] = log_name
            valid_idx.append(i)

        logger.info(f'Loaded {len(data_infos)} synthetic data, filtering to {len(valid_idx)}')
        return data_infos, valid_idx


    def load_annotations(self, ann_file):

        self.val = False
        # load original data infos
        nav_filter = self.nav_filter_path
        with open(nav_filter, 'r') as file:
            nav_filter = yaml.safe_load(file)

        scene_filter = nav_filter['tokens']
        self.scene_filter = set(scene_filter)

        # load annotations:
        logger.info('loading dataset!')
        if isinstance(ann_file, str) and not ann_file.endswith('.pkl'):
            data_infos = self.merge_into_splits(ann_file)
        else:
            data_infos = mmcv.load(ann_file, file_format="pkl")
            if isinstance(data_infos, dict):
                data_infos = data_infos['infos']
            else:
                assert isinstance(data_infos, list)
        logger.info(f'dataset loaded, length: {len(data_infos)}')

        # use log_token in OpenScene as scene_token.
        for info in data_infos:
            info['scene_name'] = info['log_name']
            info['scene_token'] = info['log_token']
        raw_data_infos = data_infos.copy()

        # load finetuning yaml and fail_scene_filter
        assert self.finetune_yaml is not None, "finetuning must contains finetune_yaml file"
        logger.info(f'loading finetuning yaml from: {self.finetune_yaml}')
        if isinstance(self.finetune_yaml, str):
            self.finetune_yaml = [self.finetune_yaml]

        fail_scene_filter = []
        for yaml_path in self.finetune_yaml:
            with open(yaml_path, 'r') as file:
                f_nav_filter = yaml.safe_load(file)
            fail_scene_filter.extend(f_nav_filter['tokens'])
        self.fail_scene_filter = set(fail_scene_filter)

        # load synthetic data
        synthetic_data_infos, synthetic_index_map = self.load_syntheitc_infos()
        len_syn_infos = len(synthetic_data_infos)

        self.fail_index_map = synthetic_index_map.copy()
        self.normal_index_map = []
        self.normal_token_id_map = dict()
        self.all_valid_token_id_map = dict()
        normal_finetuning_mask = [0] * len(data_infos)
        for i, info in enumerate(raw_data_infos):
            if np.sum(info['gt_fut_bbox_sdc_mask']) != 8:
                continue
            self.all_valid_token_id_map[info['token']] = i + len_syn_infos
            # also add failure real data to finetuning
            if info['token'] in self.fail_scene_filter and self.include_real_failures:
                self.fail_index_map.append(i + len_syn_infos)
                normal_finetuning_mask[i] = 1
            elif info['token'] in self.scene_filter:
                self.normal_index_map.append(i + len_syn_infos)
                self.normal_token_id_map[info['token']] = i + len_syn_infos

        data_infos = synthetic_data_infos + raw_data_infos

        self.all_token_id_map_key = set(self.all_valid_token_id_map.keys())
        self.normal_len = len(self.normal_index_map)

        if self.normal_ratio != 0:
            normal_index_map = self.retreive_normal_data(data_infos)
        else:
            normal_index_map = []

        self.index_map = self.fail_index_map + normal_index_map

        # -1 for synthetic data, 0 for normal data, 1 for failed data
        # len(finetuning_mask) == len(data_infos)
        self.finetuning_mask = [-1] * len_syn_infos + normal_finetuning_mask
        # len(label) == len(index_map), used in DistributedGlobalRatioSampler
        self.label = [0] * len(self.fail_index_map) + [1] * len(normal_index_map)

        logger.info(f'After filtering dataset: fail + synthetic:{len(self.fail_index_map)}')

        return data_infos

    def update_sensor(self, input_dict, index):
        info = self.data_infos[index]

        if self.modality["use_camera"]:
            image_paths = []
            lidar2img_rts = []
            lidar2cam_rts = []
            cam_intrinsics = []
            cam_distortions = []
            cam_optim_intrinsics = []

            # loop through all cameras
            for cam_type, cam_info in info["cams"].items():
                data_path = cam_info["data_path"]
                if 'syn_name' in info:
                    # use absolute path for synthetic data, keep the relative path for original data
                    data_path = os.path.join(self.synthetic_img_root, info['syn_name'], data_path)
                    data_path = os.path.abspath(data_path)
                image_paths.append(data_path)

                # obtain lidar to image transformation matrix
                lidar2cam_r = cam_info["sensor2lidar_rotation"].T
                lidar2cam_t = -lidar2cam_r @ cam_info["sensor2lidar_translation"]
                lidar2cam_rt = np.eye(4)
                lidar2cam_rt[:3, :3] = lidar2cam_r
                lidar2cam_rt[:3, 3] = lidar2cam_t

                intrinsic = cam_info["cam_intrinsic"]
                viewpad = np.eye(4)
                viewpad[: intrinsic.shape[0], : intrinsic.shape[1]] = intrinsic
                lidar2img_rt = viewpad @ lidar2cam_rt
                lidar2img_rts.append(lidar2img_rt)

                cam_intrinsics.append(intrinsic)
                lidar2cam_rts.append(lidar2cam_rt)

                distortion = np.array(cam_info['distortion'])
                optim_intrinsic, _ = cv2.getOptimalNewCameraMatrix(
                    intrinsic, distortion, (1920, 1080), 1
                )
                cam_distortions.append(distortion)
                cam_optim_intrinsics.append(optim_intrinsic)

            input_dict.update(
                dict(
                    img_filename=image_paths,
                    lidar2img=lidar2img_rts,
                    cam_intrinsic=cam_intrinsics,
                    lidar2cam=lidar2cam_rts,
                    cam_distortion=cam_distortions,
                    cam_optim_intrinsic=cam_optim_intrinsics,
                )
            )
        return input_dict

    def get_pdm_score_info(self, input_dict, index=None, info=None):
        if 'syn_name' in info:
            data = info['pdm']
            for k in data.keys():
                if k in ['IL_plan_idx', 'target_point']:
                    continue
                # emit the pdm's score
                if data[k].shape[0] > 8192:
                    data[k] = data[k][1:]
            input_dict.update(data)
            return input_dict
        else:
            return super(NavSimOpenSceneE2EFineTuneSynthetic, self).get_pdm_score_info(input_dict, index, info)
