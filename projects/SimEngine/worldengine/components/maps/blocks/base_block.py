"""
Definition of block objects, used for managing multiple roads within a block.
"""

import logging
import math
from abc import ABC

import cv2
import numpy as np

from worldengine.base_class.base_runnable import BaseRunnable
from worldengine.utils.type import WorldEngineObjectType
from worldengine.components.maps.map_constants import DrivableAreaProperty


class BaseBlock(BaseRunnable, WorldEngineObjectType, DrivableAreaProperty, ABC):
    """
    Base class for RoadBlock.
    """

    object_ID = "RoadBlock_"

    def __init__(self, block_index, global_network, random_seed):
        super(BaseBlock, self).__init__(self.object_ID + str(block_index), random_seed)

        self.block_index = block_index

        # The roadmap object for managing lanes in this block.
        self.global_network = global_network
        self.block_network = self.block_network_type()

        # polygons representing crosswalk and sidewalk
        self.crosswalks = {}
        self.sidewalks = {}

    def _sample_topology(self) -> bool:
        """
        Sample a new topology to fill self.block_network
        """
        raise NotImplementedError

    def destroy(self):
        self.block_network = None
        self.crosswalks = None
        self.sidewalks = None
        super(BaseBlock, self).destroy()



