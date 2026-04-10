import abc


class AbstractMotionModel(abc.ABC):
    """
    Interface for generic ego controllers.
    """

    def __init__(self, agent):
        super(AbstractMotionModel, self).__init__()

        self.agent = agent

    @abc.abstractmethod
    def propagate_state(self, accel_cmd, steering_rate_cmd):
        pass
