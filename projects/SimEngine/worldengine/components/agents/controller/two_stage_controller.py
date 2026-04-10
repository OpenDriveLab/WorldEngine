from worldengine.components.agents.controller.abstract_controller import AbstractController
from worldengine.components.agents.controller.motion_model.build_motion_model import build_motion_model
from worldengine.components.agents.controller.tracker.build_tracker import build_tracker
from worldengine.engine.engine_utils import get_engine


class TwoStageController(AbstractController):
    """
    Implements a two stage tracking controller. The two stages comprises of:
        1. an AbstractTracker - This is to simulate a low level controller layer that is present in real AVs.
        2. an AbstractMotionModel - Describes how the AV evolves according to a physical model.
    """

    def __init__(self, agent):
        """
        Constructor for TwoStageController
        :param scenario: Scenario
        :param tracker: The tracker used to compute control actions
        :param motion_model: The motion model to propagate the control actions
        """
        super(TwoStageController, self).__init__(agent)

        self._tracker = build_tracker(self.agent.config)(self.agent)
        self._motion_model = build_motion_model(self.agent.config)(self.agent)

    def step(self):
        """Inherited, see superclass."""
        if self.agent.trajectory is not None:
            for attr in ['waypoints', 'velocities', 'headings', 'angular_velocities']:
                if hasattr(self.agent.trajectory, attr):
                    value = getattr(self.agent.trajectory, attr)
                    if isinstance(value, list) and len(value) > 1:
                        setattr(self.agent.trajectory, attr, value[1:])

            accel_cmd, steering_rate_cmd = self._tracker.track_trajectory()
        else:
            accel_cmd, steering_rate_cmd = self.agent.lower_action
        self._motion_model.propagate_state(accel_cmd, steering_rate_cmd)
