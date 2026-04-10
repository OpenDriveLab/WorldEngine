from abc import ABC

import torch

from worldengine.base_class.base_runnable import BaseRunnable


class RenderState(dict):
    CAMERAS = "cameras"
    LIDAR = "lidar"
    AGENT_STATE = "agent_state"   # {object_id: np.ndarray([x, y, heading])}
    TIMESTAMP = "timestamp"


class BaseRenderer(BaseRunnable, ABC):

    def __init__(
        self,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.device = device
        self.__from_scratch__()

    def __from_scratch__(self):
        pass

    def reset(self):
        self.__from_scratch__()

    @property
    def background_asset(self):
        return None

    def set_asset(self, asset):
        pass

    def render(self, render_state: RenderState):
        pass
