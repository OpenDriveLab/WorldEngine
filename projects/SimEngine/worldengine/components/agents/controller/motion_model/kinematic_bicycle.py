import numpy as np

from worldengine.components.agents.controller.motion_model.abstract_motion_model import AbstractMotionModel
from worldengine.components.agents.controller import controller_utils

from worldengine.utils import math_utils


class KinematicBicycleModel(AbstractMotionModel):
    """
    A class describing the kinematic motion model where the rear axle is the point of reference.
    """

    def __init__(self, agent,):
        super(KinematicBicycleModel, self).__init__(agent)

        self.config = self.agent.config

        self._frame_rate = self.config.get('frame_rate')

        # low pass filter time constant for acceleration in s
        self._accel_time_constant = self.config.get('accel_time_constant')

        # low pass filter time constant for steering angle in s
        self._steering_angle_time_constant = self.config.get('steering_angle_time_constant')

    def propagate_state(self, accel_cmd, steering_rate_cmd):
        """Inherited, see super class."""
        cur_accel_vector = self.agent.rear_vehicle.current_acceleration
        cur_accel_value = math_utils.norm(cur_accel_vector[0], cur_accel_vector[1])
        cur_steering_angle = self.agent.current_tire_steering

        updated_accel_value = (
            self._frame_rate / (self._frame_rate + self._accel_time_constant) *
            (accel_cmd - cur_accel_value) + cur_accel_value
        )

        updated_steering_angle = (
            self._frame_rate / (self._frame_rate + self._steering_angle_time_constant) *
            (steering_rate_cmd - cur_steering_angle) + cur_steering_angle
        )
        updated_steering_rate = (updated_steering_angle - cur_steering_angle) / self._frame_rate

        # Update the state.
        x_dot = self.agent.rear_vehicle.current_velocity[0]
        y_dot = self.agent.rear_vehicle.current_velocity[1]
        longitudinal_speed = math_utils.norm(x_dot, y_dot)
        yaw_dot = longitudinal_speed * np.tan(self.agent.current_tire_steering) / self.agent.vehicle.wheel_base

        ########## Then update the pos / heading / vel / acc / angular_vel / angular_acc #######
        new_rear_pos = controller_utils.forward_integrate(
            np.array(self.agent.rear_vehicle.current_position),
            np.array([x_dot, y_dot]),
            self._frame_rate)

        new_rear_heading = math_utils.principal_value(controller_utils.forward_integrate(
            self.agent.rear_vehicle.current_heading,
            yaw_dot, self._frame_rate))

        # No speed in the Lateral dimension in the Kinematic Bicycle model.
        new_rear_longitudinal_speed = controller_utils.forward_integrate(
            longitudinal_speed, updated_accel_value, self._frame_rate)
        new_rear_velocity = [new_rear_longitudinal_speed, 0]
        new_rear_acceleration = [updated_accel_value, 0]

        new_tire_steering = np.clip(
            controller_utils.forward_integrate(
                cur_steering_angle, updated_steering_rate, self._frame_rate
            ),
            -self.agent.MAX_STEERING,
            self.agent.MAX_STEERING
        )

        new_angular_velocity = (
            new_rear_longitudinal_speed * np.tan(new_tire_steering) / self.agent.vehicle.wheel_base
        )
        new_angular_acc = (
            (new_angular_velocity - self.agent.rear_vehicle.current_angular_velocity) / self._frame_rate
        )

        self.agent.rear_vehicle.update_center_agent(
            new_rear_pos=new_rear_pos,
            new_rear_heading=new_rear_heading,
            new_rear_velocity=new_rear_velocity,
            new_rear_acceleration=new_rear_acceleration,
            new_rear_angular_velocity=new_angular_velocity,
            new_rear_angular_acc=new_angular_acc,
            new_tire_steering=new_tire_steering,
            new_action=[steering_rate_cmd, accel_cmd],
        )
