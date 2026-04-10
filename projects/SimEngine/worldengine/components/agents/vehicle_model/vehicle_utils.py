import numpy as np

from worldengine.utils import math_utils
from worldengine.scenario.scenarios.scenario_description import ScenarioDescription as SD


def get_local_velocity_shifted(displacement, ref_velocity, ref_angular_vel):
    """
    Computes the velocity at a query point on the same planar rigid body as a reference point.
    """
    # From cross product of velocity transfer formula in 2D
    velocity_shift = -displacement[:, ::-1] * ref_angular_vel
    return ref_velocity + velocity_shift


def get_acceleration_shifted(displacement, ref_accel, ref_angular_vel, ref_angular_accel):
    """
    Computes the acceleration at a query point on the same planar rigid body as a reference point.

    Forked from: https://physics.stackexchange.com/questions/328494/what-is-the-relation-between-angular-and-linear-acceleration
    Linear acceleration includes tangential acceleration and centripetal acceleration.
    """
    tangential_acc = displacement * ref_angular_accel
    centri_acc = displacement * ref_angular_vel ** 2
    return ref_accel + tangential_acc + centri_acc


class RearVehicle:
    """
    Vehicle tools to transform center information to rear.
    """

    def __init__(self, agent):
        super().__init__()
        self.agent = agent

    def get_rear_trajectory(self, waypoints, headings):
        """
        Convert center positions in object_track <expert data> into the rear-axle.

        Args:
            waypoints - The centered 3D position in different timestamps.
            headings - The corresponding vehicle orientation in different timestamps.

        Return:
            rear_position - The 3D position in the rear-axle.
        """

        # to local coordinates.
        local_shift = np.array([-self.rear_axle_to_center_dist, 0]).reshape(1, 2)
        local_shift = np.tile(local_shift, [waypoints.shape[0], 1])  # N, 2
        shift = math_utils.rotate_multi_points(local_shift, 0, headings)

        waypoints = waypoints + shift
        return waypoints

    @property
    def rear_axle_to_center_dist(self) -> float:
        """
        Getter for the distance from the rear axle to the center of mass of Ego.
        :return: Distance from rear axle to COG
        """
        return float(self.agent.vehicle.rear_axle_to_center)

    @property
    def current_position(self):
        """
        Getter for the pose at the middle of the rear axle
        """
        cur_position = np.array(self.agent.current_position).reshape(1, 2)
        return math_utils.translate_longitudinally(
            cur_position, self.agent.current_heading, -self.rear_axle_to_center_dist).reshape(2)

    @property
    def current_velocity(self):
        """
        Getter for the velocity at the middle of the rear axle.

        The rear velocity equals to the center velocity - angular velocity
        """
        # to local coordinates.
        cur_vel = np.array(self.agent.current_velocity).reshape(1, 2)
        local_velocity = math_utils.rotate_points(cur_vel, 0, -self.agent.current_heading)

        displacement = np.array([-self.rear_axle_to_center_dist, 0.0]).reshape(1, 2)
        rear_axle_velocity_2d = get_local_velocity_shifted(
            displacement, local_velocity, self.agent.current_angular_velocity)

        # to global coordinates.
        global_vel = math_utils.rotate_points(
            rear_axle_velocity_2d, 0, self.agent.current_heading).reshape(2)
        return global_vel

    @property
    def current_acceleration(self):
        """
        Getter for the acceleration at the middle of the rear axle.

        The rear acc equals to the center acc - angular acc.
        """
        # to local coordinates.
        local_acc = np.array(self.agent.current_acceleration).reshape(1, 2)
        local_acc = math_utils.rotate_points(local_acc, 0, -self.agent.current_heading)

        displacement = np.array([-self.rear_axle_to_center_dist, 0.0]).reshape(1, 2)
        rear_axle_acceleration_2d = get_acceleration_shifted(
            displacement,
            local_acc,
            self.agent.current_angular_velocity,
            self.agent.current_angular_acceleration,
        )

        # to global coordinates.
        global_acc = math_utils.rotate_points(
            rear_axle_acceleration_2d, 0, self.agent.current_heading).reshape(2)
        return global_acc

    @property
    def current_heading(self):
        return self.agent.current_heading

    @property
    def current_angular_velocity(self):
        return self.agent.current_angular_velocity

    @property
    def current_angular_acceleration(self):
        return self.agent.current_angular_acceleration

    def update_center_agent(
        self, new_rear_pos, new_rear_heading, new_rear_velocity,
        new_rear_acceleration, new_rear_angular_velocity, new_rear_angular_acc,
        new_tire_steering, new_action):
        """
        An interface to set the centered agent through rear_vehicle.

        ############### !!!! NOTE THAT: ###############
        the new_rear_velocity and new_rear_acceleration are in the <local> coordinates.
        ############### Note Done !!!! ###############
        """

        displacement = np.array([self.rear_axle_to_center_dist, 0.0]).reshape(1, 2)

        # position
        new_center_pos = np.array(new_rear_pos).reshape(1, 2)
        new_center_pos = math_utils.translate_longitudinally(
            new_center_pos, new_rear_heading, self.rear_axle_to_center_dist)
        self.agent.set_position(new_center_pos.reshape(2))

        # heading
        self.agent.set_heading_theta(new_rear_heading)

        # velocity
        new_center_velocity = np.array(new_rear_velocity).reshape(1, 2)
        new_center_velocity = get_local_velocity_shifted(
            displacement, new_center_velocity, new_rear_angular_velocity)
        new_center_velocity = math_utils.rotate_points(
            new_center_velocity, 0, new_rear_heading).reshape(2)
        self.agent.set_velocity(new_center_velocity, in_local_frame=False)

        # acceleration.
        new_center_acceleration = np.array(new_rear_acceleration).reshape(1, 2)
        new_center_acceleration = get_acceleration_shifted(
            displacement,
            new_center_acceleration,
            new_rear_angular_velocity,
            new_rear_angular_acc,)
        new_center_acceleration = math_utils.rotate_points(
            new_center_acceleration, 0, new_rear_heading).reshape(2)
        self.agent.set_acceleration(new_center_acceleration)

        # angular velocity.
        self.agent.set_angular_velocity(new_rear_angular_velocity)

        # angular acceleration.
        self.agent.set_angular_acceleration(new_rear_angular_acc)

        # tire steering.
        self.agent.set_tire_steering(new_tire_steering)

        # action.
        self.agent.set_action(new_action)
