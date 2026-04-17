import os
import pickle
from typing import List, Union
import numpy as np
import pandas as pd
from glob import glob
import yaml
import cv2

from collections import defaultdict

import mmcv
from mmdet.datasets import DATASETS

from mmdet3d_plugin.datasets.navsim_openscene_synthetic import NavSimOpenSceneE2EFineTuneSynthetic
from mmdet3d_plugin.utils import get_logger
logger = get_logger(__name__)
WORLDENGINE_ROOT = os.getenv('WORLDENGINE_ROOT', os.path.abspath('.'))

@DATASETS.register_module()
class NavSimOpenSceneE2EFineTuneSimScale(NavSimOpenSceneE2EFineTuneSynthetic):
    r"""
    OpenScene E2E Finetuning Synthetic Dataset
    This dataset is used for finetuning the model on the failed + synthetic data.
    It will randomly select normal data from the same log and add to the finetuning dataset.
    Please only use this dataset for training, not for val/test.
    """

    def __init__(self,  *args, 
        cotrain=False, 
        **kwargs):
        # SimScale Only
        self.cotrain=cotrain
        self.simscale_path = os.path.join(WORLDENGINE_ROOT,'data/alg_engine/openscene-synthetic/meta_datas_navformer')
        self.simscale_filter_path = os.path.join(WORLDENGINE_ROOT, 'projects/AlgEngine/configs/navsim_splits/simscale_split')
        
        super(NavSimOpenSceneE2EFineTuneSimScale, self).__init__(*args, **kwargs)


    def load_syntheitc_infos(self):
        data_infos = []
        valid_idx = []

        token2pdms = {}
        self.simscale_filter = set()
        for folder_name in self.folder_names:

            folder_data_infos = []
            logger.info(f'loading synthetic data from {folder_name}')
            # Sometimes we pre-combined the pickles
            combined_pkl_path = f'{self.simscale_path}/{folder_name}/combined.pkl'
            if os.path.exists(combined_pkl_path):
                logger.info('loading from combined pickle file')
                with open(combined_pkl_path, 'rb') as f:
                    folder_data_infos = pickle.load(f)
            else:
                logger.info('loading from splitted pickle files')
                pkls_path = glob(f'{self.simscale_path}/{folder_name}/*.pkl')
                for p in pkls_path:
                    with open(p, 'rb') as f:
                        data = pickle.load(f)
                        data = data['infos']
                        for d in data:
                            d['syn_name'] = folder_name
                        folder_data_infos.extend(data)
            data_infos.extend(folder_data_infos)

            # pdms
            with open(f"{self.syn_pdm_path}/{folder_name}.pkl", 'rb') as f:
                token2pdms.update(pickle.load(f))

            # filter
            with open(f"{self.simscale_filter_path}/{folder_name}.yaml", 'r') as file:
                self.simscale_filter.update(yaml.safe_load(file)['tokens'])


        for i, info in enumerate(data_infos):
            # skip the first N frames, they are history frames
            if info['frame_idx'] != self.history_frame_num:
                continue

            # restore the original log_name
            # 2021.05.12.19.36.12_veh-35_00215_00405-a77d536b271d516e -> 2021.05.12.19.36.12_veh-35_00215_00405, a77d536b271d516e 
            # we can still use scene_token / scene_name for temporal usage
            synthetic_log_name = info['log_name']
            parts = info['log_name'].split('-')
            log_name = '-'.join(parts[:2])

            synthetic_token = info["token"]    

            if info.get('invalid', False):
                continue

            pdm = token2pdms[synthetic_token]

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
        fail_scene_filter = []
        if self.finetune_yaml is not None:
        # "finetuning must contains finetune_yaml file"
            logger.info(f'loading finetuning yaml from: {self.finetune_yaml}')
            if isinstance(self.finetune_yaml, str):
                self.finetune_yaml = [self.finetune_yaml]
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

        if self.cotrain: # navtrain + simsacle
            normal_index_map = self.normal_index_map
        elif self.normal_ratio != 0: # simscale 1:1
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

    def retreive_normal_data(self, data_infos):
        # create a map for log_name to non-failed data index
        self.log_name_to_idx_map = defaultdict(list)

        filter_union = self.fail_scene_filter.copy()
        filter_union.update(self.simscale_filter)

        for i, info in enumerate(data_infos):
            if info['token'] not in filter_union and info['token'] in self.scene_filter:
                self.log_name_to_idx_map[info['log_name']].append(i)

        normal_index_map = []
        for idx in self.fail_index_map:
            normal_index_map += self.single_retrieval_function(data_infos, idx)
        del self.log_name_to_idx_map
        return normal_index_map