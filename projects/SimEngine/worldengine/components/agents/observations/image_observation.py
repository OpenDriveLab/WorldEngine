"""
An example implementation for observation.

Rendering interface for providing sensor observation at each timestamp.
"""

import numpy as np

from worldengine.components.parameter_space import Box, Dict
from worldengine.components.agents.observations.base_observation import BaseObservation
from worldengine.components.agents.observations.ego_state_observation import EgoStateObservation


class ImageStateObservation(BaseObservation):
    """
    Use ego state info, navigation info and front cam image/top down image as input
    The shape needs special handling
    """
    IMAGE = "image"
    STATE = "state"

    def __init__(self, agent):
        super(ImageStateObservation, self).__init__(agent)
        self.img_obs = ImageObservation(agent)
        self.state_obs = EgoStateObservation(agent)

    @property
    def observation_space(self):
        return Dict(
            {
                self.IMAGE: self.img_obs.observation_space,
                self.STATE: self.state_obs.observation_space
            }
        )

    def observe(self):
        return {self.IMAGE: self.img_obs.observe(),
                self.STATE: self.state_obs.observe()}

    def destroy(self):
        super(ImageStateObservation, self).destroy()
        self.img_obs.destroy()
        self.state_obs.destroy()


class ImageObservation(BaseObservation):
    # TODO: need to rewrite after rendering part is ready.
    """
    Method to utilize engine for rendering sensor observation.
    """

    def __init__(self, agent):
        super(ImageObservation, self).__init__(agent)
        self.state = np.zeros(self.observation_space.shape, dtype=np.uint8)

    @property
    def observation_space(self):
        shape = (256, 256, 3)
        return Box(0, 255, shape=shape, dtype=np.uint8)

    def observe(self):
        """
        Get the image Observation. By setting new_parent_node and the reset parameters, it can capture a new image from
        a different position and pose
        """

        # get image from engine.
        new_obs = self.engine.get_sensor()
        self.state = new_obs
        return self.state

    def get_image(self):
        return self.state.copy()

    def reset(self, env, vehicle=None):
        """
        Clear stack
        :param env: MetaDrive
        :param vehicle: BaseVehicle
        :return: None
        """
        self.state = np.zeros(self.observation_space.shape, dtype=np.uint8)

    def destroy(self):
        """
        Clear memory
        """
        super(ImageObservation, self).destroy()
        self.state = None
