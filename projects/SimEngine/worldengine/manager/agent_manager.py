"""
This manager allows one to use object like vehicles/traffic lights as agent with multi-agent support.
You would better make your own agent manager based on this class.
"""

import copy
import numpy as np

from worldengine.manager.base_manager import BaseManager

from worldengine.utils import math_utils
from worldengine.utils.type import WorldEngineObjectType
from worldengine.utils.agent_utils import BEV_visualizer
from worldengine.scenario.scenarios.scenario_description import ScenarioDescription as SD
from worldengine.scenario.scenarios.parse_scenario_state import parse_object_track

from worldengine.components.agents.base_agent import BaseAgent
from worldengine.components.agents.ego_agent import EgoAgent

import logging
logger = logging.getLogger(__name__)


class BaseAgentManager(BaseManager):
    PRIORITY = 10  
    STATIC_THRESHOLD = 3  # m, static if moving distance < 3

    # TODO: set debug flag to debugging the route / navigation
    #  observaton / policy for ego car.
    DEBUG_CAR_ID = 'ego'

    def __init__(self):
        """
        Each agent has the properties of:
            * observation.
            * actions.
        """
        super(BaseAgentManager, self).__init__()

        # Dynamic agents:
        #  use their policies and update position and each frame.
        self._dynamic_agents = {}  # {object.id: BaseAgent}

        # Static agents:
        #  objects without policy, like barriers and cones.
        self._static_agents = {}  # {object.id: BaseAgent}

        # Dead agents:
        #  once the agent is unavailable or out-of-visible areas,
        #  remove.
        self._dead_agents = {}  # {object.id: BaseAgent}

        self._BEV_vis = None
        self._trajectory_buffer = {}

    # set up agents in the scenarios.
    def before_reset(self):  # remove agents.
        ret = super(BaseAgentManager, self).before_reset()

        for k, v in self._dead_agents.items():
            v.destroy()
        self._dead_agents = {}

        for k, v in self._static_agents.items():
            v.destroy()
        self._static_agents = {}

        for k, v in self._dynamic_agents.items():
            v.destroy()
        self._dynamic_agents = {}

        return ret

    def _process_agent_track(self, obj_id, obj_track):
        """
        processing agent track data.
        Args:
            obj_id: agent id
            obj_track: agent track data
        Returns:
            state: dict, agent state
            valid_periods: list, valid periods of the agent
        """
        time_idx = np.arange(len(obj_track[SD.STATE][SD.POSITION]))
        state = parse_object_track(obj_track, time_idx, sim_time_interval=self.engine.sim_time_interval)
        
        # Extract valid time periods
        valid_array = state[SD.VALID]
        valid_periods = []
        start_idx = None
        
        for i in range(len(valid_array)):
            if valid_array[i] and start_idx is None:
                start_idx = i
            elif not valid_array[i] and start_idx is not None:
                valid_periods.append((start_idx, i-1))
                start_idx = None
        
        if start_idx is not None:
            valid_periods.append((start_idx, len(valid_array)-1))
            
        return state, valid_periods

    def spawn_agent(self, obj_id, obj_track):
        """
        Create and initialize an agent.
        """
        state, valid_periods = self._process_agent_track(obj_id, obj_track)

        # Determine agent type (static or dynamic)
        # if object are not vehicle, pedestrian, cyclist, set to static
        if obj_id in self._static_agents or not WorldEngineObjectType.is_participant(obj_track['type']):
            set_to_add = self._static_agents
            is_static = True
        elif obj_id in self._dynamic_agents:
            set_to_add = self._dynamic_agents
            is_static = False
        else:
            valid_points = state[SD.POSITION][np.where(state[SD.VALID])]
            moving = (np.max(np.std(valid_points, axis=0)[:2]) > self.STATIC_THRESHOLD) or (obj_id == 'ego')
            set_to_add = self._dynamic_agents if moving else self._static_agents
            is_static = not moving
        
        # for static agents, set agent_policy == 'trajectory_policy'
        if is_static:
            static_config = self.engine.global_config.copy()
            static_config['agent_policy'] = 'trajectory_policy'
            static_config['agent_controller'] = 'log_play_controller'
            agent_config = static_config
        else:
            agent_config = self.engine.global_config

        # Spawn agent
        if obj_id == 'ego':
            set_to_add[obj_id] = EgoAgent(
                obj_id, state, name=obj_id, config=agent_config)
        else:
            set_to_add[obj_id] = BaseAgent(
                obj_id, state, name=obj_id, config=agent_config, traj_step=self.engine.episode_step)

        # Initialize agent
        set_to_add[obj_id].reset()
        
        # Update valid_periods
        if obj_id not in self._agent_valid_periods:
            self._agent_valid_periods[obj_id] = valid_periods
            
        return state

    def reset(self):
        super(BaseAgentManager, self).reset()
        self._trajectory_buffer = {}
        self._agent_valid_periods = {}
        
        if self.engine.global_config.visualize_BEV:
            if self._BEV_vis is None:
                self._BEV_vis = BEV_visualizer()
            else:
                self._BEV_vis.clear()
        
        sim_length = self.engine.global_config.max_step + 1

        for obj_id, obj_track in self.current_agent_data.items():
            if obj_track.get('type') in ['PEDESTRIAN', 'CYCLIST']:
                continue

            _, valid_periods = self._process_agent_track(obj_id, obj_track)
            self._agent_valid_periods[obj_id] = valid_periods
            
            should_spawn = (
                obj_id == 'ego' or  
                any(start == 0 for start, _ in valid_periods)  
            )
            
            if should_spawn:
                self.spawn_agent(obj_id, obj_track)
                logger.debug(f"Spawned agent {obj_id} at reset")

            self._trajectory_buffer[obj_id] = {
                "type": obj_track.get('type'),
                "metadata": {
                    "track_length": sim_length,
                    "type": obj_track.get('type'),
                    "object_id": obj_id
                },
                "state": {
                    "position": np.zeros([sim_length, 2], dtype=np.float32),
                    "heading": np.zeros([sim_length, 1], dtype=np.float32),
                    "velocity": np.zeros([sim_length, 1], dtype=np.float32),
                    "valid": np.zeros([sim_length, 1], dtype=np.float32),
                    "length": np.zeros([sim_length, 1], dtype=np.float32),
                    "width": np.zeros([sim_length, 1], dtype=np.float32),
                    "height": np.zeros([sim_length, 1], dtype=np.float32),
                }
            }

    def _get_current_lane(self, obj_id):
        """
        Used to find current lane information.
        """
        agent = self._dynamic_agents[obj_id]
        map = self.engine.current_map
        possible_lanes = map.road_network.get_closest_lane_index(
            agent.current_position, return_all=True)
        possible_lanes = possible_lanes[:3] # extract the most possible 3 lanes
    
        min_heading_diff = float('inf')
        best_lane = None
        
        for lane_info in possible_lanes:
            lane = lane_info[2]
            long, _ = lane.local_coordinates(agent.current_position)
            lane_heading = lane.heading_theta_at(long)
            heading_diff = abs(agent.current_heading - lane_heading)
            
            if heading_diff < min_heading_diff:
                min_heading_diff = heading_diff
                best_lane = lane
        
        if best_lane is None:
            return None
        
        return best_lane
    
    def get_surrounding_agents(self, obj):
        """
        Get the agents in front of and behind the given agent.
        
        Args:
            obj: The agent.
        
        Returns:
            tuple: A tuple containing four lists:
                - List of IDs of agents in front of the given agent.
                - List of distances of agents in front of the given agent.
                - List of IDs of agents behind the given agent.
                - List of distances of agents behind the given agent.
        """

        if obj.id not in self.all_agents:
            raise ValueError(f"Agent with ID {obj.id} not found in dynamic agents.")
        
        agent = self.all_agents[obj.id]
        agent_position = agent.current_position
        front_agents, front_distances, back_agents, back_distances = [], [], [], []
        
        for other_id, other_agent in self.all_agents.items():
            if other_id == obj.id:
                continue
            if other_agent.height == 0.0 or other_agent.width == 0.0 or other_agent.length == 0.0:
                continue
            # _, _, ref_lane = agent_utils.get_current_lane(other_agent, self.engine.current_map)
            ref_lane = agent.navigation.current_lane
            agent_long, agent_lat = ref_lane.local_coordinates(agent_position)
            other_position = other_agent.current_position
            other_long, other_lat = ref_lane.local_coordinates(other_position)
            distance = np.linalg.norm(np.array(agent_position) - np.array(other_position))
            heading_diff = abs(math_utils.wrap_to_pi(other_agent.current_heading - agent.current_heading))
            if abs(agent_lat - other_lat) > 2.0 or heading_diff > np.pi / 2:
                continue
            if other_long > agent_long:
                front_agents.append(other_id)
                front_distances.append(distance)
            else:
                back_agents.append(other_id)
                back_distances.append(distance)
        
        return front_agents, front_distances, back_agents, back_distances

    ##### Step Function #####
    # TODO: to debug later.
    def before_step(self, *args, **kwargs):
        for v in self._dynamic_agents.values():
            v.before_step(None)

    def step(self):
        current_step = self.engine.episode_step
        for v in self.all_agents.values():
            # TODO: set action to None for now.
            v.step()

        if self.engine.global_config.agent_policy == 'trajectory_policy':
            for obj_id, valid_periods in self._agent_valid_periods.items():
                if obj_id == 'ego':  
                    continue

                should_be_active = any(start <= current_step <= end 
                                    for start, end in valid_periods)
                

                if not should_be_active and obj_id in self._dynamic_agents:
                    self._dynamic_agents[obj_id].destroy()
                    del self._dynamic_agents[obj_id]
                    logger.debug(f"Removed agent {obj_id} at step {current_step}")
                
                if not should_be_active and obj_id in self._static_agents:
                    self._static_agents[obj_id].destroy()
                    del self._static_agents[obj_id]
                    logger.debug(f"Removed agent {obj_id} at step {current_step}")
                
                elif should_be_active and obj_id not in self.all_agents:

                    if current_step > 0:
                        self.spawn_agent(obj_id, self.current_agent_data[obj_id])
                        logger.debug(f"Spawned agent {obj_id} at step {current_step}")
        else:
            for obj_id, valid_periods in self._agent_valid_periods.items():
                if obj_id == 'ego':
                    continue

                should_be_active = any(start <= current_step <= end 
                                    for start, end in valid_periods)
                
                if not should_be_active and obj_id in self._static_agents:
                    self._static_agents[obj_id].destroy()
                    del self._static_agents[obj_id]
                    logger.debug(f"Removed agent {obj_id} at step {current_step}")

                if should_be_active and obj_id not in self.all_agents:

                    if current_step > 0:
                        self.spawn_agent(obj_id, self.current_agent_data[obj_id])
                        logger.debug(f"Spawned agent {obj_id} at step {current_step}")
            
    def after_step(self, *args, **kwargs):
        for v in self._dynamic_agents.values():
            # TODO: set action to None for now.
            return_info = v.after_step()
            if return_info is not None:
                for key, value in return_info.items():
                    if key in self._trajectory_buffer[v.id]["state"]:
                        self._trajectory_buffer[v.id]["state"][key][self.engine.episode_step] = value
                        self._trajectory_buffer[v.id]["state"]["valid"][self.engine.episode_step] = 1

        if self.engine.global_config.visualize_BEV:
            self._BEV_vis.draw(self.engine.episode_step)
        
        if self.engine.episode_step == self.engine.managers['scenario_manager'].current_scene["log_length"] - 1:
            return self._trajectory_buffer
        else:
            return None
    
    def output_gif(self):
        if self.engine.global_config.visualize_BEV:
            self._BEV_vis.output_gif()
        
    @property
    def current_agent_data(self):
        return self.engine.current_scene[SD.OBJECT_TRACKS]

    @property
    def sdc_track_id(self):
        return str(self.engine.current_scene[SD.SDC_ID])

    @property
    def ego_agent(self):
        return self._dynamic_agents['ego']

    @property
    def alive_agents(self):
        dynamic_agents = self._dynamic_agents.copy()
        static_agents = self._static_agents.copy()
        return dynamic_agents.update(static_agents)

    @property
    def get_dynamic_agents(self):
        return self._dynamic_agents

    @property
    def all_agents(self):
        """
        Return a merged dictionary of all agents, including both dynamic and static agents.
        """
        all_agents = self._dynamic_agents.copy()
        all_agents.update(self._static_agents)
        return all_agents