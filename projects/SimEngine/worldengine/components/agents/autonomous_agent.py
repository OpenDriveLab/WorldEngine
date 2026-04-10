import numpy as np
from collections import deque

from worldengine.components.agents.vehicle_model.build_vehicle import build_vehicle
from worldengine.components.agents.vehicle_model.vehicle_utils import RearVehicle

from worldengine.components.agents.base_agent import BaseAgent
from worldengine.common.dataclasses import Trajectory

from worldengine.utils import math_utils
import logging
logger = logging.getLogger(__name__)

class AutonomousAgent(BaseAgent):
    """
    AutonomousAgent has controllers to interact in the scenarios.
    """

    def __init__(self, object_id, object_track, name, random_seed=None, config=None):

        super(AutonomousAgent, self).__init__(
            object_id=object_id,
            object_track=object_track,
            name=name,
            random_seed=random_seed,
            config=config,)

        self.rear_vehicle = RearVehicle(self)

    def reset(self,):
        super(AutonomousAgent, self).reset()

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
        # NOTE Autonomous agent's behaviors are implemented in scenarios!!!
        # obs = self.observation.observe()

        # TODO: policy to provide trajectory.
        trajectory = None
        centered_waypoints = trajectory.waypoints
        centered_headings = trajectory.headings
        rear_waypoints = self.rear_vehicle.get_rear_trajectory(centered_waypoints, centered_headings)
        self.trajectory = Trajectory(waypoints=rear_waypoints)

        self.controller.step()

    def after_step(self):
        self.navigation.update_localization()

        step_info = {
            "steering": float(self.steering),
            "throttle_brake": float(self.throttle_brake),
            "velocity": float(self.current_speed),
            'acceleration': float(math_utils.norm(
                self.current_acceleration[0], self.current_acceleration[1]
            )),
            "angular_velocity": float(self.current_angular_velocity),
            "angular_acceleration": float(self.current_angular_acceleration),
            "policy": self.policy.name,
            "controller": self.controller.name,
        }

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
