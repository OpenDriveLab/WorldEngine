import numpy as np

from worldengine.components.agents.controller.abstract_controller import AbstractController
from worldengine.components.agents.controller.motion_model.build_motion_model import build_motion_model

class PIDController:
    """A PID controller to manage proportional, integral, and derivative terms for error correction."""
    def __init__(self, k_p: float, k_i: float, k_d: float):
        """
        Initializes the PID controller with specified gain values.

        Args:
            k_p: float, proportional gain coefficient.
            k_i: float, integral gain coefficient.
            k_d: float, derivative gain coefficient.
        """
        self.k_p = k_p
        self.k_i = k_i
        self.k_d = k_d

        self.p_error = 0
        self.i_error = 0
        self.d_error = 0

    def _update_error(self, current_error: float):
        """
        Updates the error terms for PID control based on the current error.

        Args:
            current_error: float, the current error value.
        """
        self.i_error += current_error
        self.d_error = current_error - self.p_error
        self.p_error = current_error

    def get_result(self, current_error: float, make_up_coefficient=1.0):
        """
        Calculate the control output based on the current error and the PID coefficients.

        Args:
            current_error: float, the current error value.
            make_up_coefficient: float, a coefficient to adjust the final output (default is 1.0).

        Return:
            float, the control output after applying the PID formula.
        """
        self._update_error(current_error)
        return (-self.k_p * self.p_error - \
                 self.k_i * self.i_error - \
                 self.k_d * self.d_error) * \
                make_up_coefficient

    def reset(self):
        self.p_error = 0
        self.i_error = 0
        self.d_error = 0

class SimplePIDController(AbstractController):
    """
    Full PID controller with steering and throttle control.
    """

    def __init__(self, agent):
        raise DeprecationWarning("We don't use PID Controller now")
    
        super(AbstractController, self).__init__(agent)
        
        self._motion_model = build_motion_model(self.agent.config)(self.agent)

        self.steering_controller = PIDController(k_p=0.1, k_i=0.1, k_d=0.1)
        self.throttle_controller = PIDController(k_p=0.1, k_i=0.1, k_d=0.1)

    def step(self):
        """Inherited, see superclass."""

        longitudinal_error = None
        accel_cmd = self.throttle_controller.get_result(longitudinal_error)

        lateral_error = None
        steering_rate_cmd = self.steering_controller.get_result(lateral_error)

        self._motion_model.propagate_state(accel_cmd, steering_rate_cmd)

        return accel_cmd, steering_rate_cmd