from worldengine.components.agents.policy.base_policy import BasePolicy
from worldengine.scenario.base_scenario import BaseScenario

class LogPolicy(BasePolicy):
    def __init__(self, scneario: BaseScenario, future_time_horizon: float):
        self.trajectory = None

    def act(self):
        return [0.0, 1.0]

    def reset(self):
        super(LogPolicy, self).reset()