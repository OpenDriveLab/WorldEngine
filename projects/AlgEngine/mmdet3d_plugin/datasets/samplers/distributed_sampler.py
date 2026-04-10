import math

import torch
from torch.utils.data import DistributedSampler as _DistributedSampler
from .sampler import SAMPLER


@SAMPLER.register_module()
class DistributedSampler(_DistributedSampler):
    def __init__(
        self, dataset=None, num_replicas=None, rank=None, shuffle=True, seed=0
    ):
        super().__init__(dataset, num_replicas=num_replicas, rank=rank, shuffle=shuffle)
        # for the compatibility from PyTorch 1.3+
        self.seed = seed if seed is not None else 0

    def __iter__(self):
        # deterministically shuffle based on epoch
        if self.shuffle:
            assert False
        else:
            indices = torch.arange(len(self.dataset)).tolist()

        total_size = math.ceil(len(indices) / self.num_replicas) * self.num_replicas
        indices = (indices * math.ceil(total_size / len(indices)))[:total_size]
        indices = indices[self.rank::self.num_replicas]

        return iter(indices)
