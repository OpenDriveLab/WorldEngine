from typing import Dict

class BaseClient:
    def __init__(self, agent):
        self.agent = agent

    def process_frame(self, frame_data: Dict, step: int):
        pass

    def _postprocess_frame_data(self, frame_data: Dict):
        pass
