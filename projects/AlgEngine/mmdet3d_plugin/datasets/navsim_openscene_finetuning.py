import random
import yaml
from collections import defaultdict
from typing import List, Union

import mmcv
from mmdet.datasets import DATASETS
from mmdet3d_plugin.datasets.navsim_openscene_nuplan import NavSimOpenSceneE2E
from mmdet3d_plugin.utils import get_logger
logger = get_logger(__name__)


@DATASETS.register_module()
class NavSimOpenSceneE2EFineTune(NavSimOpenSceneE2E):
    r"""
    OpenScene E2E Finetuning Dataset
    This dataset is used for finetuning the model on the failed data.
    It will randomly select normal data from the same log and add to the finetuning dataset.
    Please only use this dataset for training, not for val/test.
    """

    def __init__(
        self,
        *args,
        normal_ratio: int = 1,
        finetune_yaml: Union[str, List[str]] = None,
        normal_only = False,
        test_mode: bool = False,
        **kwargs):

        assert test_mode is False, "finetuning must be in train mode"

        self.normal_ratio = normal_ratio
        self.finetune_yaml = finetune_yaml
        self.normal_only = normal_only    # For normal FT ablation.

        super(NavSimOpenSceneE2EFineTune, self).__init__(*args, test_mode=test_mode, **kwargs)

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

        # load finetuning yaml
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

        self.fail_index_map, self.normal_index_map = [], []
        self.normal_token_id_map = dict()
        self.all_valid_token_id_map = dict()
        finetuning_mask = [0] * len(data_infos)
        for i, info in enumerate(data_infos):
            self.all_valid_token_id_map[info['token']] = i
            if info['token'] in self.fail_scene_filter:
                self.fail_index_map.append(i)
                finetuning_mask[i] = 1
            elif info['token'] in self.scene_filter:
                self.normal_index_map.append(i)
                self.normal_token_id_map[info['token']] = i

        self.all_token_id_map_key = set(self.all_valid_token_id_map.keys())
        self.normal_len = len(self.normal_index_map)

        if self.normal_ratio != 0:
            normal_index_map = self.retreive_normal_data(data_infos)
        else:
            normal_index_map = []

        # 0 for normal data, 1 for failed data
        # len(finetuning_mask) == len(data_infos)
        self.finetuning_mask = finetuning_mask

        if not self.normal_only:
            self.index_map = self.fail_index_map + normal_index_map
            # len(label) == len(index_map), used in DistributedGlobalRatioSampler
            self.label = [0] * len(self.fail_index_map) + [1] * len(normal_index_map)
        else:
            self.index_map = normal_index_map
            self.label = [1] * len(normal_index_map)

        logger.info(f'filtering {len(data_infos)} frames to {len(self.index_map)}; fail:{len(self.fail_index_map)}')
        return data_infos

    def retreive_normal_data(self, data_infos):
        # create a map for log_name to non-failed data index
        self.log_name_to_idx_map = defaultdict(list)
        for i, info in enumerate(data_infos):
            if info['token'] not in self.fail_scene_filter and info['token'] in self.scene_filter:
                self.log_name_to_idx_map[info['log_name']].append(i)

        normal_index_map = []
        for idx in self.fail_index_map:
            normal_index_map += self.single_retrieval_function(data_infos, idx)
        del self.log_name_to_idx_map
        return normal_index_map

    def select_ids(self, id_list, n):
        if len(id_list) < n:
            return random.choices(id_list, k=n)  # Select n items with replacement
        else:
            return random.sample(id_list, n)  # Select n unique items

    def single_retrieval_function(self, data_infos, fail_idx):
        '''
        retrerival function from the normal dataset by queried failure data token
        current logic is to randomly select a normal data from the same scene log
        [TODO]: This can be updated by KNN search or others
        '''
        fail_info = data_infos[fail_idx]
        fail_log_name = fail_info['log_name']

        valid_idx_list = self.log_name_to_idx_map[fail_log_name].copy()

        if len(valid_idx_list) == 0:
            logger.info(f'no valid token in {fail_log_name}')
            retrevied_tokens_ids = [fail_idx]
        else:
            retrevied_tokens_ids = self.select_ids(valid_idx_list, int(self.normal_ratio))

        return retrevied_tokens_ids

    def get_data_info(self, index, prev_frame=False, **kwargs):
        data_info = super(NavSimOpenSceneE2EFineTune, self).get_data_info(index, prev_frame=prev_frame, **kwargs)
        if not prev_frame:
            data_info['fail_mask'] = self.finetuning_mask[index]
        else:
            data_info['fail_mask'] = 0
        return data_info
