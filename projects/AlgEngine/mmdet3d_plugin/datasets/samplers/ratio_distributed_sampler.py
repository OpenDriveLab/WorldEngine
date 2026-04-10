import math

import numpy as np
import torch
from mmcv.runner import get_dist_info
from torch.utils.data import Sampler
from .sampler import SAMPLER

from mmdet3d_plugin.utils import get_logger
logger = get_logger(__name__)

@SAMPLER.register_module()
class DistributedGlobalRatioSampler(Sampler):
    def __init__(self, dataset, samples_per_gpu=1, ratio_0_to_1=1, num_replicas=None, rank=None, seed=0):
        _rank, _num_replicas = get_dist_info()
        self.rank = rank if rank is not None else _rank
        self.num_replicas = num_replicas if num_replicas is not None else _num_replicas
        self.dataset = dataset
        self.samples_per_gpu = samples_per_gpu
        self.seed = seed
        self.epoch = 0
        self.ratio_0_to_1 = ratio_0_to_1

        assert hasattr(dataset, "label"), "Dataset must have `label` attribute with 0/non-0 values"
        self.labels = np.array(dataset.label)

        self.label0_idx = np.where(self.labels == 0)[0]
        self.label1_idx = np.where(self.labels != 0)[0]

        logger.info(f'label 0, 1 length: {len(self.label0_idx)}, {len(self.label1_idx)}')

        self.global_batch_size = samples_per_gpu * self.num_replicas
        self.num_0 = int(self.global_batch_size * ratio_0_to_1 / (1 + ratio_0_to_1))
        self.num_1 = self.global_batch_size - self.num_0

        # Skip iteration support
        self.skip_iter_at_epoch = False
        self.start_iter = 0

        logger.info(f'DistributedGlobalRatioSampler len:{len(self)}')

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.epoch + self.seed)

        label0_idx = self.label0_idx[torch.randperm(len(self.label0_idx), generator=g).numpy()]
        label1_idx = self.label1_idx[torch.randperm(len(self.label1_idx), generator=g).numpy()]

        max_batches = min(len(label0_idx) // self.num_0, len(label1_idx) // self.num_1)
        total_size = max_batches * self.global_batch_size

        indices = []
        for i in range(max_batches):
            batch_0 = label0_idx[i * self.num_0 : (i + 1) * self.num_0]
            batch_1 = label1_idx[i * self.num_1 : (i + 1) * self.num_1]
            batch = np.concatenate([batch_0, batch_1])
            # print(batch, self.labels[self.index_map[batch[0]]], self.labels[self.index_map[batch[1]]])
            batch = batch[torch.randperm(len(batch), generator=g).numpy()]  # shuffle within batch
            indices.append(batch)

        # Shuffle global batches
        indices = np.stack(indices)
        indices = indices[torch.randperm(len(indices), generator=g).numpy()]
        indices = indices.reshape(-1)

        # Get subset for this rank
        assert len(indices) % self.num_replicas == 0
        rank_indices = indices[self.rank::self.num_replicas]
        # print(rank_indices.shape)

        # Support skipping at epoch (for resume)
        if self.skip_iter_at_epoch:
            rank_indices = rank_indices[self.start_iter:]

        return iter(rank_indices.tolist())

    def __len__(self):
        # Number of samples per replica
        global_batches = min(len(self.label0_idx) // self.num_0, len(self.label1_idx) // self.num_1)
        total_samples = global_batches * self.global_batch_size
        return total_samples // self.num_replicas

    def set_epoch(self, epoch):
        self.epoch = epoch

    def skip_iter_at_epoch_x(self, inner_iter):
        if inner_iter > 0:
            self.skip_iter_at_epoch = True
            self.start_iter = inner_iter
        else:
            self.skip_iter_at_epoch = False
            self.start_iter = 0