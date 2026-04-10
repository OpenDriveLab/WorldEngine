import copy
import numpy as np

from nuplan.common.actor_state.state_representation import StateSE2

from worldengine.manager.base_manager import BaseManager

from worldengine.scenario.scenarios.scenario_description import ScenarioDescription as SD

import logging
logger = logging.getLogger(__name__)

class ScenarioManager(BaseManager):
    DEFAULT_DATA_BUFFER_SIZE = 100
    PRIORITY = -10

    def __init__(self, info_dict=None):
        super(ScenarioManager, self).__init__()
        scenes = info_dict
        num_scenes = len(scenes)
        logger.info(f"Loading {num_scenes} scenarios")

        self.scenes = dict()
        self.scenes_ids = dict()
        for idx, (scene_id, scene) in enumerate(scenes.items()):
            scene["route_var_name"] = None
            scene["trigger_points"] = []
            scene["scenarios"] = [
                {
                "name": "Freeride",
                "trigger_points": [StateSE2(1, 1, 0)]
                },
            ]

            converted_scene = SD.centralize_to_ego_car_initial_position(SD(scene))
            converted_scene = SD.update_summaries(converted_scene)
            self.scenes[scene_id] = converted_scene
            self.scenes_ids[idx] = scene_id

        self.num_scenarios = num_scenes
        # scenarios for this worker.
        self.available_scenario_indices = [
            i for i in range(self.num_scenarios)
        ]

        # stat
        self.coverage = [0 for _ in range(self.num_scenarios)]

    # some properties of the class.
    @property
    def current_scene_summary(self):
        return self.current_scene[SD.SUMMARY.SUMMARY]

    @property
    def current_scene_length(self):
        return self.current_scene[SD.LENGTH]

    @property
    def current_scene_index(self):
        idx = self.engine.global_random_seed
        assert idx in self.available_scenario_indices, \
            "scenario index exceeds range, scenario index: {}".format(idx)
        return idx

    @property
    def current_scene_id(self):
        index = self.current_scene_index
        return self.scenes_ids[index]

    @property
    def current_scene(self):
        index = self.current_scene_index
        return self.get_scene(index)

    def _get_scene(self, i):
        assert i in self.available_scenario_indices, \
            "scenario index exceeds range, scenario index: {}".format(i)
        scenario_id = self.scenes_ids[i]
        ret = self.scenes[scenario_id]
        assert isinstance(ret, SD)
        return ret

    def get_scene(self, i, should_copy=False):
        ret = self._get_scene(i)
        self.coverage[i] = 1
        if should_copy:
            return copy.deepcopy(ret)
        return ret

    @property
    def data_coverage(self):
        return sum(self.coverage) / len(self.coverage)

    @property
    def all_scenes_completed(self):
        return all(self.coverage)

    def before_step(self):
        pass

    def step(self):
        step_info = {}
        return step_info

    def clear_stored_scenarios(self):
        self.scenes = {}
        self.scenes_ids = {}

    def resume_from_completed_scenarios(self, completed_ids: set):
        original_count = len(self.available_scenario_indices)

        # Filter out completed scenario indices
        remaining_indices = []
        for idx in self.available_scenario_indices:
            scene_id = self.scenes_ids[idx]
            if scene_id not in completed_ids:
                remaining_indices.append(idx)
            else:
                # Mark completed scenarios as covered
                self.coverage[idx] = 1

        self.available_scenario_indices = remaining_indices
        self.engine.seed(self.available_scenario_indices[0])

        skipped_count = original_count - len(remaining_indices)
        logger.info(f"Resume: Skipping {skipped_count} completed scenarios, {len(remaining_indices)} remaining.")

        if len(remaining_indices) == 0:
            logger.info("All scenarios already completed. Nothing to do.")

    def filter_short_scenarios(self, min_length: int):
        """Filter out scenarios whose log_length is shorter than min_length."""
        original_count = len(self.available_scenario_indices)
        remaining_indices = []
        for idx in self.available_scenario_indices:
            scene_id = self.scenes_ids[idx]
            scene = self.scenes[scene_id]
            log_length = scene.get(SD.LENGTH, float('inf'))
            if log_length < min_length:
                logger.warning(
                    f"Skipping scenario {scene_id}: log_length ({log_length}) < required obs_len ({min_length})"
                )
                self.coverage[idx] = 1
            else:
                remaining_indices.append(idx)

        skipped_count = original_count - len(remaining_indices)
        if skipped_count > 0:
            logger.warning(f"Filtered out {skipped_count} short scenarios, {len(remaining_indices)} remaining.")
            self.available_scenario_indices = remaining_indices
            if len(remaining_indices) > 0:
                self.engine.seed(remaining_indices[0])

    def reset(self):
        reset_info = {}
        return reset_info

    def after_reset(self):
        reset_info = {}
        return reset_info

    def get_metadata(self):
        state = super().get_metadata()
        raw_scene_data = self.current_scene
        state["raw_scene_data"] = raw_scene_data
        return state

    def destroy(self):
        """
        Clear memory
        """
        super().destroy()
        self.clear_stored_scenarios()

    def next_scene(self):
        """
        Switch to the next available scene
        """
        current_idx = self.current_scene_index

        current_position = self.available_scenario_indices.index(current_idx)

        next_position = (current_position + 1) % len(self.available_scenario_indices)

        self.engine.seed(self.available_scenario_indices[next_position])
