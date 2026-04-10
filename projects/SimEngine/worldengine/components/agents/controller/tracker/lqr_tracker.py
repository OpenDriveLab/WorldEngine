"""
Implementation of <LQR> tracking module,
    Used for translating the trajectory signal into steering and braking control.
"""


import logging
from enum import IntEnum
from typing import List, Tuple

import numpy as np
import numpy.typing as npt

from worldengine.components.agents.controller.tracker.abstract_tracker import AbstractTracker
from worldengine.components.agents.controller.tracker import tracker_utils
from worldengine.components.maps.lanes.center_lane import CenterLane

from worldengine.utils import math_utils


class LateralStateIndex(IntEnum):
    """
    Index for solving lateral state transformation equation.
    """

    LATERAL_ERROR = 0  # [m] The lateral error with respect to the planner centerline at the vehicle's rear axle center.
    HEADING_ERROR = 1  # [rad] The heading error "".
    STEERING_ANGLE = 2  # [rad] The wheel angle relative to the longitudinal axis of the vehicle.


class LQRTracker(AbstractTracker):
    """
    Forked from: https://github.com/motional/nuplan-devkit/blob/master/nuplan/planning/simulation/controller/tracker/lqr.py

    Implements an LQR tracker for a kinematic bicycle model.

    We decouple into two subsystems, longitudinal and lateral, with small angle approximations for linearization.
    We then solve two sequential LQR subproblems to find acceleration and steering rate inputs.

    Longitudinal Subsystem:
        States: [velocity]
        Inputs: [acceleration]
        Dynamics (continuous time):
            velocity_dot = acceleration

    Lateral Subsystem (After Linearization/Small Angle Approximation):
        States: [lateral_error, heading_error, steering_angle]
        Inputs: [steering_rate]
        Parameters: [velocity, curvature]
        Dynamics (continuous time):
            lateral_error_dot  = velocity * heading_error
            heading_error_dot  = velocity * (steering_angle / wheelbase_length - curvature)
            steering_angle_dot = steering_rate

    The continuous time dynamics are discretized using Euler integration and zero-order-hold on the input.
    In case of a stopping reference, we use a simplified stopping P controller instead of LQR.

    The final control inputs passed on to the motion model are:
        - acceleration
        - steering_rate
    """

    def __init__(self, agent,):
        """
        Constructor for LQR controller
        agent.config includes:
            - q_longitudinal: The weights for the Q matrix for the longitudinal subystem.
            - r_longitudinal: The weights for the R matrix for the longitudinal subystem.
            - q_lateral: The weights for the Q matrix for the lateral subystem.
            - r_lateral: The weights for the R matrix for the lateral subystem.
            - discretization_time: [s] The time interval used for discretizing the continuous time dynamics.
            - tracking_horizon: How many discrete time steps ahead to consider for the LQR objective.
            - stopping_proportional_gain: The proportional_gain term for the P controller when coming to a stop.
            - stopping_velocity: [m/s] The velocity below which we are deemed to be stopping and we don't use LQR.
        """
        super(LQRTracker, self).__init__(agent=agent)

        self.config = self.agent.config

        # hyperparameters for LQR controller.
        q_longitudinal = np.array(self.config['q_longitudinal'])
        r_longitudinal = np.array(self.config['r_longitudinal'])
        q_lateral = np.array(self.config['q_lateral'])
        r_lateral = np.array(self.config['r_lateral'])

        # Longitudinal LQR Parameters
        assert isinstance(q_longitudinal, np.ndarray) and len(q_longitudinal) == 1, \
            "q_longitudinal should be 1 float ndarray (velocity)."
        assert isinstance(r_longitudinal, np.ndarray) and len(r_longitudinal) == 1, \
            "r_longitudinal should be 1 float ndarray (acceleration)."
        self._q_longitudinal = np.diag(q_longitudinal)
        self._r_longitudinal = np.diag(r_longitudinal)

        # Lateral LQR Parameters
        assert isinstance(q_lateral, np.ndarray) and len(q_lateral) == 3, \
            "q_lateral should be 3 element-float ndarray (lateral_error, heading_error, steering_angle)."
        assert isinstance(r_lateral, np.ndarray) and len(r_lateral) == 1, \
            "r_lateral should be 1 element-float ndarray (steering_rate)."
        self._q_lateral = np.diag(q_lateral)
        self._r_lateral = np.diag(r_lateral)

        # assert all cost element in Q and R are positive definite.
        for attr in ["_q_lateral", "_q_longitudinal"]:
            assert np.all(np.diag(getattr(self, attr)) >= 0.0), f"self.{attr} must be positive semi-definite."

        for attr in ["_r_lateral", "_r_longitudinal"]:
            assert np.all(np.diag(getattr(self, attr)) > 0.0), f"self.{attr} must be positive definite."

        # Simulator related parameters for LQR
        # Note we want a horizon > 1 so that steering rate actually can impact lateral/heading error in discrete time.
        discretization_time = self.config['discretization_time']
        tracking_horizon = self.config['tracking_horizon']
        assert discretization_time > 0.0, "The discretization_time should be positive."
        assert (
            tracking_horizon > 1
        ), "We expect the horizon to be greater than 1 - else steering_rate has no impact with Euler integration."
        self._discretization_time = discretization_time
        self._tracking_horizon = tracking_horizon  # look ahead horizon for tracking action.
        self._wheel_base = self.agent.vehicle.wheel_base

        # Velocity/Curvature Estimation Parameters
        jerk_penalty = float(self.config['jerk_penalty']) # acceleration penalty.
        curvature_rate_penalty = float(self.config['curvature_rate_penalty'])  # steering rate penalty.
        assert jerk_penalty > 0.0, "The jerk penalty must be positive."
        assert curvature_rate_penalty > 0.0, "The curvature rate penalty must be positive."
        self._jerk_penalty = jerk_penalty
        self._curvature_rate_penalty = curvature_rate_penalty

        # Stopping Controller Parameters
        stopping_proportional_gain = self.config['stopping_proportional_gain']
        stopping_velocity = self.config['stopping_velocity']  # stopping threshold.
        assert stopping_proportional_gain > 0, "stopping_proportional_gain has to be greater than 0."
        assert stopping_velocity > 0, "stopping_velocity has to be greater than 0."
        self._stopping_proportional_gain = stopping_proportional_gain
        self._stopping_velocity = stopping_velocity

    def track_trajectory(self):
        """Inherited, see superclass."""
        trajectory = self.agent.trajectory

        # transform waypoints to rear-axle coordinates.
        centered_waypoints = trajectory.waypoints
        centered_headings = trajectory.headings
        rear_waypoints = self.agent.rear_vehicle.get_rear_trajectory(centered_waypoints, centered_headings)
        self.waypoints = rear_waypoints

        if len(self.waypoints) == 0:
            raise ValueError("The waypoints should not be empty.")
        self.route = CenterLane(self.waypoints, width=2, segment_threshold=1, segment_eps=1e-6)
        self._start_time = 0
        self._planning_frame_rate = self.agent.config['planning_frame_rate']
        self._end_time = (len(self.waypoints) - 1) * self._planning_frame_rate

        initial_velocity, initial_lateral_state_vector = self._compute_initial_velocity_and_lateral_state()

        # Compute the velocity and curvature profile,
        #  This is optimized by minimal mean-square optimization.
        reference_velocity, reference_curvature = self._compute_reference_velocity_and_curvature_profile()

        should_stop = reference_velocity <= self._stopping_velocity and initial_velocity <= self._stopping_velocity

        if should_stop:
            accel_cmd, steering_rate_cmd = self._stopping_controller(initial_velocity, reference_velocity)
        else:
            # Do acceleration and steering command optimization.
            # This is achieved by LQR.
            accel_cmd = self._longitudinal_lqr_controller(initial_velocity, reference_velocity)
            velocity_profile = tracker_utils._generate_profile_from_initial_condition_and_derivatives(
                initial_condition=initial_velocity,
                derivatives=np.ones(self._tracking_horizon) * accel_cmd,
                discretization_time=self._discretization_time,
            )[: self._tracking_horizon]
            steering_rate_cmd = self._lateral_lqr_controller(
                initial_lateral_state_vector,
                velocity_profile,
                reference_curvature,
            )
            
        # Return acceleration and steering command
        #  at the rear-axle.
        return accel_cmd, steering_rate_cmd

    def _compute_initial_velocity_and_lateral_state(self):
        """
        This method projects the initial tracking error into vehicle/Frenet frame.  It also extracts initial velocity.
        """
        initial_trajectory_waypoints = self.waypoints
        initial_trajectory_heading = self.route.heading_theta_at(0.)

        # Determine initial error state.
        # position error.
        xy_error = self.agent.rear_vehicle.current_position - initial_trajectory_waypoints[0]

        # heading error.
        lateral_error = math_utils.rotate_points(
            np.array(xy_error).reshape(1, 2), 0, -initial_trajectory_heading)[0, 1]
        heading_error = math_utils.angle_diff(
            self.agent.rear_vehicle.current_heading, initial_trajectory_heading, 2 * np.pi)

        # Return initial velocity and lateral state vector.
        cur_velocity = self.agent.rear_vehicle.current_velocity
        cur_speed = math_utils.norm(cur_velocity[0], cur_velocity[1])

        initial_lateral_state_vector = np.array(
            [
                lateral_error,
                heading_error,
                self.agent.current_tire_steering,
            ],
        )

        return cur_speed, initial_lateral_state_vector

    def _compute_reference_velocity_and_curvature_profile(self):
        """
        This method computes reference velocity and curvature profile based on the reference trajectory.
        We use a lookahead time equal to self._tracking_horizon * self._discretization_time.
        :param current_iteration: Used to get the current time.
        :param trajectory: The reference trajectory we are tracking.
        :return: The reference velocity [m/s] and curvature profile [rad] to track.
        """
        times_s, pos_s, heading_s = tracker_utils.get_interpolated_reference_trajectory_poses(
            planning_trajectory=self.route,
            planning_waypoints=self.waypoints,
            planning_interval=self._planning_frame_rate,
            discretization_time=self._discretization_time,
            start_time=self._start_time,
            end_time=self._end_time)

        (
            velocity_profile,  # optimized velocity at each timestamp.
            acceleration_profile,  # optimized acceleration at each timestamp.
            curvature_profile,  # optimized curvature at each timestamp.
            curvature_rate_profile,  # optimized curvature_rate at each timestamp.
        ) = tracker_utils.get_velocity_curvature_profiles_with_derivatives_from_poses(
            discretization_time=self._discretization_time,
            pos_s=pos_s,
            heading_s=heading_s,
            jerk_penalty=self._jerk_penalty,
            curvature_rate_penalty=self._curvature_rate_penalty,
        )

        reference_time = self._start_time + self._tracking_horizon * self._discretization_time
        reference_velocity = np.interp(reference_time, times_s[:-1], velocity_profile)

        profile_times = [
            self._start_time + x * self._discretization_time for x in range(self._tracking_horizon)
        ]
        reference_curvature_profile = np.interp(profile_times, times_s[:-1], curvature_profile)

        return float(reference_velocity), reference_curvature_profile

    def _stopping_controller(self, initial_velocity: float, reference_velocity: float):
        """
        Apply proportional controller when at near-stop conditions.

        Args:
            initial_velocity: [m/s] The current velocity of ego.
            reference_velocity: [m/s] The reference velocity to track.

        Return:
            Acceleration [m/s^2] and zero steering_rate [rad/s] command.
        """
        accel = -self._stopping_proportional_gain * (initial_velocity - reference_velocity)
        return accel, 0.0

    def _longitudinal_lqr_controller(self, initial_velocity: float, reference_velocity: float):
        """
        This longitudinal controller determines an acceleration input to minimize velocity error at a lookahead time.

        Args:
            initial_velocity: [m/s] The current velocity of ego.
            reference_velocity: [m/s] The reference_velocity to track at a lookahead time.

        Return:
            Acceleration [m/s^2] command based on LQR.
        """
        # We assume that we hold the acceleration constant for the entire tracking horizon.
        # Given this, we can show the following where N = self._tracking_horizon and dt = self._discretization_time:
        # velocity_N = velocity_0 + (N * dt) * acceleration (transformation function in LQR)
        # Thus: A = 1
        # B = N * dt
        # g = 0
        A = np.array([1.0], dtype=np.float32)
        B = np.array([self._tracking_horizon * self._discretization_time], dtype=np.float32)

        accel_cmd = self._solve_one_step_lqr(
            initial_state=np.array([initial_velocity], dtype=np.float32),
            reference_state=np.array([reference_velocity], dtype=np.float32),
            Q=self._q_longitudinal,
            R=self._r_longitudinal,
            A=A,
            B=B,
            g=np.zeros(1, dtype=np.float32),
            angle_diff_indices=[],
        )

        return float(accel_cmd)

    def _lateral_lqr_controller(
        self,
        initial_lateral_state_vector: np.ndarray,
        velocity_profile: np.ndarray,
        curvature_profile: np.ndarray) -> float:
        """
        This lateral controller determines a steering_rate input to minimize lateral errors at a lookahead time.
        It requires a velocity sequence as a parameter to ensure linear time-varying lateral dynamics.

        Args:
            initial_lateral_state_vector: The current lateral state of ego.
            velocity_profile: [m/s] The velocity over the entire self._tracking_horizon-step lookahead.
            curvature_profile: [rad] The curvature over the entire self._tracking_horizon-step lookahead..

        Return:
            Steering rate [rad/s] command based on LQR.
        """
        assert len(velocity_profile) == self._tracking_horizon, (
            f"The linearization velocity sequence should have length {self._tracking_horizon} "
            f"but is {len(velocity_profile)}."
        )
        assert len(curvature_profile) == self._tracking_horizon, (
            f"The linearization curvature sequence should have length {self._tracking_horizon} "
            f"but is {len(curvature_profile)}."
        )

        # Set up the lateral LQR problem using the constituent linear time-varying (affine) system dynamics.
        # Ultimately, we'll end up with the following problem structure where N = self._tracking_horizon:
        # lateral_error_N = A @ lateral_error_0 + B @ steering_rate + g
        n_lateral_states = len(LateralStateIndex)
        I = np.eye(n_lateral_states, dtype=np.float32)

        A = I
        B = np.zeros((n_lateral_states, 1), dtype=np.float32)
        g = np.zeros(n_lateral_states, dtype=np.float32)

        # Convenience aliases for brevity.
        idx_lateral_error = LateralStateIndex.LATERAL_ERROR
        idx_heading_error = LateralStateIndex.HEADING_ERROR
        idx_steering_angle = LateralStateIndex.STEERING_ANGLE

        input_matrix = np.zeros((n_lateral_states, 1), np.float32)
        input_matrix[idx_steering_angle] = self._discretization_time

        for index_step, (velocity, curvature) in enumerate(zip(velocity_profile, curvature_profile)):
            state_matrix_at_step = np.eye(n_lateral_states, dtype=np.float32)
            state_matrix_at_step[idx_lateral_error, idx_heading_error] = velocity * self._discretization_time
            state_matrix_at_step[idx_heading_error, idx_steering_angle] = (
                velocity * self._discretization_time / self._wheel_base
            )

            affine_term = np.zeros(n_lateral_states, dtype=np.float32)
            affine_term[idx_heading_error] = -velocity * curvature * self._discretization_time

            A = state_matrix_at_step @ A
            B = state_matrix_at_step @ B + input_matrix
            g = state_matrix_at_step @ g + affine_term

        steering_rate_cmd = self._solve_one_step_lqr(
            initial_state=initial_lateral_state_vector,
            reference_state=np.zeros(n_lateral_states, dtype=np.float64),
            Q=self._q_lateral,
            R=self._r_lateral,
            A=A,
            B=B,
            g=g,
            angle_diff_indices=[idx_heading_error, idx_steering_angle],
        )

        return float(steering_rate_cmd)

    @staticmethod
    def _solve_one_step_lqr(
        initial_state: np.ndarray,
        reference_state: np.ndarray,
        Q: np.ndarray,
        R: np.ndarray,
        A: np.ndarray,
        B: np.ndarray,
        g: np.ndarray,
        angle_diff_indices: List[int] = [],):
        """
        This function uses LQR to find an optimal input to minimize tracking error in one step of dynamics.
        The dynamics are next_state = A @ initial_state + B @ input + g and our target is the reference_state.
        :param initial_state: The current state.
        :param reference_state: The desired state in 1 step (according to A,B,g dynamics).
        :param Q: The state tracking 2-norm cost matrix.
        :param R: The input 2-norm cost matrix.
        :param A: The state dynamics matrix.
        :param B: The input dynamics matrix.
        :param g: The offset/affine dynamics term.
        :param angle_diff_indices: The set of state indices for which we need to apply angle differences, if defined.
        :return: LQR optimal input for the 1-step problem.
        """
        state_error_zero_input = A @ initial_state + g - reference_state

        for angle_diff_index in angle_diff_indices:
            state_error_zero_input[angle_diff_index] = math_utils.angle_diff(
                state_error_zero_input[angle_diff_index], 0.0, 2 * np.pi
            )

        lqr_input = -np.linalg.inv(B.T @ Q @ B + R) @ B.T @ Q @ state_error_zero_input
        return lqr_input
