"""
Base class for vehicle / pedestrian / etc. objects.
"""
import copy
import math
import numpy as np
import logging
from collections import deque

# controllers.
from worldengine.components.agents.controller import build_controller
from worldengine.components.agents.vehicle_model.vehicle_utils import RearVehicle

from worldengine.components.agents.base_agent import BaseAgent
from worldengine.common.dataclasses import Trajectory

from worldengine.utils import math_utils

logger = logging.getLogger(__name__)


class EgoAgent(BaseAgent):
    """
    BaseAgent is something interacting with game engine.
    """

    def __init__(self, object_id, object_track, name, random_seed=None, config=None):

        assert object_id == 'ego'
        super(EgoAgent, self).__init__(
            object_id=object_id,
            object_track=object_track,
            name=name,
            random_seed=random_seed,
            config=config,)

        self.rear_vehicle = RearVehicle(self)

        self._length = self.vehicle.length
        self._height = self.vehicle.height
        self._width = self.vehicle.width

    def reset(self,):
        super(EgoAgent, self).reset()

        # other info
        #  control information.
        self.throttle_brake = 0.0
        self.steering = 0
        self.last_current_action = deque([(0.0, 0.0), (0.0, 0.0)], maxlen=2)

        # vehicle information
        self.tire_steering = 0
        self.acceleration = [0, 0]
        self.angular_acceleration = 0

    def _preprocess_action(self, action):
        if action is None:
            return None, {"raw_action": None}
        return action, {'raw_action': (action[0], action[1])}

    def before_step(self, action=None):
        """
        Save info and make decision before action
        """
        if action is not None:
            assert len(action) == 2
        action, step_info = self._preprocess_action(action)

        self.last_position = self._cur_pos
        self.last_heading_dir = self._cur_heading_theta
        self.last_velocity = self._cur_velocity  # 2D vector
        if action is not None:
            self.set_action(action)
        return step_info

    def step(self):

        # EgoStateObservation is not compatable with TrajectoryNavigation
        if type(self.navigation).__name__ == 'EgoLaneNavigation':
            obs = self.observation.observe()

        # TODO: policy to provide trajectory.
        if self.policy.is_current_step_valid:
            # The planned trajectory for current timestamp.
            # NOTE THAT!!!!!!
            #   The planned trajectory should be in the rear-axis. (For controller.)
            self.trajectory = self.policy.act()  # waypoints in the centered coordinates.
            
            # controller to update the position.
            self.controller.step()
            # set acceleration
            self.set_acceleration(None)
            self._traj_step += 1

    def after_step(self):
        if self.navigation is not None:
            self.navigation.update_localization()

        step_info = {
            "position": self.current_position,
            "heading": self.current_heading,
            "velocity": float(self.current_speed),
            "length": self.length,
            "width": self.width,
            "height": self.height,
        } # TODO: valid judgement

        return step_info

    def set_action(self, action):
        if action is None:
            return
        self.steering = action[0]
        self.throttle_brake = action[1]

        self.last_current_action.append(action)  # the real step of physics world is implemented in taskMgr.step()

    @property
    def current_steering(self):
        return self.steering

    def set_acceleration(self, acc):
        if hasattr(self, 'last_velocity') and self.last_velocity is not None:
            velocity_diff = self._cur_velocity - self.last_velocity
            dt = 0.5   
            self.acceleration = velocity_diff / dt
        else:
            self.acceleration = acc

    @property
    def current_acceleration(self):
        return self.acceleration

    def set_angular_acceleration(self, angular_acc):
        self.angular_acceleration = angular_acc

    @property
    def current_angular_acceleration(self):
        return self.angular_acceleration

    def set_tire_steering(self, tire_steering):
        self.tire_steering = tire_steering

    @property
    def current_tire_steering(self):
        """
        return the steering of the car.
        """
        return self.tire_steering

    @property
    def last_and_current_action(self):
        """
        return the last and current action of the car.
        """
        return self.last_current_action
