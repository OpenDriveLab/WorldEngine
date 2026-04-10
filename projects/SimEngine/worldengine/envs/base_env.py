import logging
import warnings
import time
import os
from pathlib import Path
from typing import Any, Dict, List, Union, Tuple
import traceback

from omegaconf import DictConfig

from worldengine.utils import merge_dicts, concat_step_infos
from worldengine.engine.engine_utils import get_engine, close_engine, \
    engine_initialized, initialize_engine, initialize_global_config

from worldengine.manager.scenario_manager import ScenarioManager
from worldengine.manager.render_manager import RenderManager
from worldengine.manager.map_manager import ScenarioMapManager
from worldengine.manager.agent_manager import BaseAgentManager
from worldengine.manager.data_manager import DataManager
from worldengine.manager.metric_manager import MetricManager
from worldengine.manager.dense_reward_manager import DenseRewardManager
from worldengine.runner.utils import RunnerReport
from worldengine.components.agents.planner.outside_planner import OutsidePlanner
from worldengine.utils.multithreading.process_utils import resolve_worker_placeholders

class BaseEnv:

    # ===== Intialization =====
    def __init__(self, config: DictConfig, name: str, data: Dict):
        if config is None:
            config = {}
        self.logger = logging.getLogger("WorldEngine")
        self.config = config
        self.name = name
        self.info_dicts = data

        # Flag that keeps track whether simulation is still running
        self._is_simulation_running = False
        self.max_step = self.config.max_step

        self.outside_planner = None

    def lazy_init(self):
        """
        Only init once in runtime, variable here exists till the close_env is called
        :return: None
        """
        # It is the true init() func to create the main vehicle and its module, to avoid incompatible with ray
        if engine_initialized():
            return
        # Resolve worker-specific placeholders (e.g., __WORKER_ID__) in config
        # This must happen inside each worker to get the correct worker/GPU ID
        self.config = resolve_worker_placeholders(self.config)
        initialize_global_config(self.config)
        initialize_engine(self.config)
        # engine setup
        self.setup_engine()
        # other optional initialization
        self._after_lazy_init()

    def _after_lazy_init(self):
        pass

    def _get_completed_scenarios_dir(self) -> Path:
        """Get the path to the completed scenarios directory."""
        completed_dir = Path(self.config.completed_scenarios_dir)
        completed_dir.mkdir(parents=True, exist_ok=True)
        return completed_dir

    def _get_completed_scenarios_file(self) -> Path:
        """Get the path to the completed scenario file for this worker."""
        completed_dir = self._get_completed_scenarios_dir()
        return completed_dir / f"completed_{self.name}.txt"

    def _load_completed_scenarios(self) -> set:
        """Load the set of completed scenario IDs from all files under the completed scenarios dir."""
        completed_dir = self._get_completed_scenarios_dir()
        completed_ids = set()
        if completed_dir.exists() and completed_dir.is_dir():
            for file in completed_dir.glob("completed_*.txt"):
                if not file.is_file():
                    continue
                with open(file, "r") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            completed_ids.add(line)
        return completed_ids

    def _save_completed_scenario(self, scenario_id: str):
        """Append a completed scenario ID to the file."""
        completed_file = self._get_completed_scenarios_file()
        with open(completed_file, 'a') as f:
            f.write(f"{scenario_id}\n")

    def _filter_completed_scenarios(self):
        """Filter out already completed scenarios from the scenario manager."""
        if not self.config.enable_resume:
            return

        completed_ids = self._load_completed_scenarios()
        if not completed_ids:
            self.logger.info("Resume enabled but no completed scenarios found.")
            return

        scenario_manager = self.engine.managers['scenario_manager']
        scenario_manager.resume_from_completed_scenarios(completed_ids)
        return

    def _filter_short_scenarios(self):
        """Filter out scenarios with log_length shorter than required obs_len when dense_reward_manager is enabled."""
        if not self.config.with_dense_reward_manager:
            return
        obs_len = (self.engine.global_config['num_history']
                   + self.engine.global_config['num_future']
                   + self.engine.global_config['reward_sampling_poses'])
        scenario_manager = self.engine.managers['scenario_manager']
        scenario_manager.filter_short_scenarios(obs_len)

    @property
    def engine(self):
        return get_engine()

    # ===== Run-time =====
    def run(self) -> List[RunnerReport]:

        self.reset(seed=0)
        scenario_manager = self.engine.managers['scenario_manager']
        if len(scenario_manager.available_scenario_indices) == 0:
            self.logger.info("No scenarios to process. Returning empty reports.")
            return []

        reports = []
        start_time = time.perf_counter()
        current_scene_length = self.engine.global_config['num_history'] + self.engine.global_config['num_future'] - 1

        for step in range(self.max_step):
            try:
                self.logger.info(f"Step {self.engine.episode_step}/{current_scene_length} for scenario "
                                    f"({self.engine.managers['scenario_manager'].current_scene_index+1}/{self.engine.managers['scenario_manager'].num_scenarios}) "
                                    f"{self.engine.managers['scenario_manager'].current_scene_id}")
                actions = self._get_actions(self.engine.episode_step)
                _, _, termination, truncation = self.step(actions)

                if truncation or step == self.max_step - 1:
                    # append the last report
                    current_time = time.perf_counter()
                    completed_scenario_id = self.engine.managers['scenario_manager'].current_scene_id
                    report = RunnerReport(
                        succeeded=True,
                        error_message=None,
                        start_time=start_time,
                        end_time=current_time,
                        planner_report=None,
                        scenario_name=completed_scenario_id,
                        planner_name=None,
                        log_name=self.name,
                    )
                    reports.append(report)

                    # save completed scenario ID for resume functionality
                    self._save_completed_scenario(completed_scenario_id)

                    # all scenarios done
                    if termination:
                        break

                    # next scenario
                    _ = self._reset_scenario()
                    start_time = time.perf_counter()

            except Exception as e:
                error = traceback.format_exc()

                # Print to the terminal
                # self.logger.warning("----------- Simulation failed: with the following trace:")
                # self.logger.warning(f"Simulation failed with error:\n {e}")
                self.logger.exception("----------- Simulation failed")

                # Log the failed scene log/tokens
                # TODO: modify scene related materials
                failed_scenes = f"[{self.engine.managers['scenario_manager'].current_scene_id}, {self.name}]\n"
                self.logger.warning(f"\nFailed simulation [log,token]:\n {failed_scenes}")

                self.logger.warning("----------- Simulation failed!")
                
                # append the last report
                # Fail if desired
                if self.config.exit_on_failure:
                    raise RuntimeError('Simulation failed')
                
                current_time = time.perf_counter()
                report = RunnerReport(
                    succeeded=False,
                    error_message=error,
                    start_time=start_time,
                    end_time=current_time,
                    planner_report=None,
                    scenario_name=self.engine.managers['scenario_manager'].current_scene_id,
                    planner_name=None,
                    log_name=self.name,
                )
                reports.append(report)

                # Mark failed scenario as attempted
                scenario_manager.coverage[scenario_manager.current_scene_index] = 1
                # Check if all scenarios have been attempted
                if scenario_manager.all_scenes_completed:
                    break

                # next scenario
                _ = self._reset_scenario()
                start_time = time.perf_counter()

        return reports

        
    def step(self, actions):
        # prepare for stepping the simulation
        scene_manager_before_step_infos = self.engine.before_step(actions)
        # step all entities and the simulator
        self.engine.step(self.config.decision_repeat)
        # update states, if restore from episode data, position and heading will be force set in update_state() function
        scene_manager_after_step_infos = self.engine.after_step()

        engine_info = merge_dicts(scene_manager_after_step_infos, scene_manager_before_step_infos,
                                  allow_new_keys=True, without_copy=True)
        
        return self._get_step_return(actions, engine_info=engine_info)  # collect observation, reward, termination

    def done_function(self) -> Tuple[bool, Dict]:
        # warnings.warn("Done function is not implemented. Return Done = False")
        return False, {}

    def reset(self, seed: Union[None, int] = None):
        """
        Reset the env, scene can be restored and replayed by giving episode_data
        Reset the environment or load an episode from episode data to recover is
        :param seed: The seed to set the env. It is actually the scene index you intend to choose
        :return: None
        """

        # Start simulation
        self._is_simulation_running = True
        self.lazy_init()

        self.seed(seed)

        self._filter_completed_scenarios()
        self._filter_short_scenarios()

        if len(self.engine.managers['scenario_manager'].available_scenario_indices) == 0:
            self.logger.info("No scenarios remaining after filtering.")
            return None

        return self._reset_scenario(from_reset=True)

    def _reset_scenario(self, from_reset=False):
        """
        Reset the scenario to the initial state.
        """
        if not from_reset:
            # switch to the next scenario
            self.engine.managers['scenario_manager'].next_scene()

        reset_info = self.engine.reset()

        if self.config.use_planner_actions:
            ego_agent = self.engine.managers['agent_manager'].ego_agent
            if from_reset:
                self.outside_planner = OutsidePlanner(ego_agent, self.config.planner_data_path)
            else:
                self.outside_planner.reset(ego_agent)

        return self._get_reset_return(reset_info)

    def _get_actions(self, step: int):
        if self.config.use_planner_actions:
            actions = self.outside_planner.get_trajectory(step)
        else:
            actions = None
        return actions

    def _get_reset_return(self, reset_info):
        # TODO: figure out how to get the information of the before step
        scene_manager_before_step_infos = reset_info
        # Do rendering at the first frame
        obses = self.engine.get_sensor()
        obses_infos = {'render': obses}
        scene_manager_after_step_infos = self.engine.after_step()

        done_infos = {}

        engine_info = merge_dicts(
            scene_manager_after_step_infos, scene_manager_before_step_infos, allow_new_keys=True, without_copy=True
        )

        step_infos = concat_step_infos([engine_info, done_infos, obses_infos])
        return step_infos

    def _get_step_return(self, actions, engine_info):
        # get observations
        obses = self.engine.get_sensor()

        # get done info
        done, done_info = self.done_function()
        done_infos = done_info

        # merge all info
        step_infos = concat_step_infos([engine_info, done_infos])

        # set truncation flag
        truncateds = self.engine.episode_step >= self.engine.global_config['num_history'] + self.engine.global_config['num_future'] - 1

        # set termination flag
        terminates = done
        if self.engine.managers['scenario_manager'].all_scenes_completed and truncateds:
            terminates = True
            step_infos['all_scenes_completed'] = True

        return obses, step_infos, terminates, truncateds

    def close(self):
        if self.engine is not None:
            close_engine()

    def setup_engine(self):
        """
        Engine setting after launching
        """
        self.engine.register_manager("scenario_manager", ScenarioManager(self.info_dicts))
        if self.config.with_render_manager:
            self.engine.register_manager("render_manager", RenderManager())
        self.engine.register_manager('map_manager', ScenarioMapManager())
        self.engine.register_manager('agent_manager', BaseAgentManager())
        if self.config.with_data_manager:
            if not self.config.with_render_manager:
                raise ValueError("Data manager requires render manager to be enabled")
            self.engine.register_manager('data_manager', DataManager())
        if self.config.with_metric_manager:
            self.engine.register_manager('metric_manager', MetricManager())
        if self.config.with_dense_reward_manager:
            self.engine.register_manager('dense_reward_manager', DenseRewardManager())

    def seed(self, seed=None):
        if seed is not None:
            self.engine.seed(seed)

    @property
    def current_seed(self):
        return self.engine.global_random_seed
