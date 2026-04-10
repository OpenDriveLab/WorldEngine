import numpy as np
from hydra.utils import instantiate

from worldengine.manager.base_manager import BaseManager
from worldengine.scenario.scenarios.scenario_description import ScenarioDescription as SD
from worldengine.render.base_renderer import RenderState
import logging
logger = logging.getLogger(__name__)

class RenderManager(BaseManager):

    PRIORITY = 10000    # lowest priority

    def __init__(self):
        super(RenderManager, self).__init__()

        self.current_scene: SD = self.engine.managers['scenario_manager'].current_scene
        self.current_scene_id = self.current_scene[SD.ID]
        self.base_timestamp = self.current_scene[SD.BASE_TIMESTAMP]
        self.local2global_translation_xy = -np.array(self.current_scene[SD.METADATA][SD.OLD_ORIGIN_IN_CURRENT_COORDINATE])

        self.renderer = instantiate(self.global_config.renderer)
        self.rendering_results = {}

    def render(self):
        render_state = RenderState()

        # get dynamic agents state. (x, y, heading)
        agents_state = {}
        for obj_id, agent in self.engine.managers['agent_manager'].get_dynamic_agents.items():
            if not agent.policy.is_current_step_valid:
                continue

            if obj_id == 'ego':
                # pass rear axle position and heading
                bbox = np.array([agent.rear_vehicle.current_position[0], agent.rear_vehicle.current_position[1], 0.0, 0.0, 0.0, agent.current_heading])
            else:
                bbox = agent.bounding_box
            # transform to nuplan global coordinate system.
            bbox[:2] += self.local2global_translation_xy
            agents_state[obj_id] = bbox

        render_state[RenderState.AGENT_STATE] = agents_state

        # get timestamp
        global_step = self.engine.episode_step
        timestamp = self.base_timestamp + global_step * self.engine.sim_time_interval * 1e6
        render_state[RenderState.TIMESTAMP] = timestamp

        # get sensor parameters
        # TODO: pertube camera location by timeshift.
        render_state[RenderState.CAMERAS] = self.current_scene[SD.CAMERAS]
        render_state[RenderState.LIDAR] = self.current_scene[SD.LIDAR]
        
        return self.renderer.render(render_state)

    def before_reset(self):
        pass

    def reset(self):
        self.current_scene: SD = self.engine.managers['scenario_manager'].current_scene
        self.current_scene_id = self.current_scene[SD.ID]
        self.base_timestamp = self.current_scene[SD.BASE_TIMESTAMP]
        self.local2global_translation_xy = -np.array(self.current_scene[SD.METADATA][SD.OLD_ORIGIN_IN_CURRENT_COORDINATE])
        self.renderer.reset(self.current_scene_id)

    def after_reset(self):
        pass

    def before_step(self):
        pass

    def step(self):
        pass

    def get_observations(self):
        step = self.engine.episode_step
        logger.debug(f"Rendering step {step} for scenario {self.current_scene_id}")
        render_results = self.render()

        self.rendering_results[step] = render_results

        # trigger data saving after getting the render results
        self.engine.managers['data_manager'].save_current_frame_data()

        return render_results

    def after_step(self):
        pass

    def close(self):
        pass

    def destroy(self):
        pass
