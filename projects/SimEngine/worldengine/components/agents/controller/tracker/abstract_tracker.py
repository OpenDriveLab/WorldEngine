import abc


class AbstractTracker(abc.ABC):
    def __init__(self, agent):
        super(AbstractTracker, self).__init__()

        self.agent = agent

    @abc.abstractmethod
    def track_trajectory(self):
        pass
