from worldengine.common.dataclasses import Trajectory
from worldengine.engine.engine_utils import get_engine

class OutsidePlanner:
    def __init__(self, agent, traj_folder_path: str):
        self.agent = agent
        self._traj_info = self.agent.object_track
        self.traj_folder_path = traj_folder_path
    
    def reset(self, agent):
        self.agent = agent
        self._traj_info = self.agent.object_track

    def get_trajectory(self, step: int) -> Trajectory:
        return self.agent.client.get_trajectory(step)

    @property
    def engine(self):
        return get_engine()
