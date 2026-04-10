import logging
import numpy as np

from worldengine.components.agents.policy.base_policy import BasePolicy
from worldengine.utils.math_utils import not_zero, wrap_to_pi, norm
from worldengine.common.dataclasses import Trajectory


class IDMPolicy(BasePolicy):
    """
    An agent policy that describes the agent's behaviour with respect to a lead agent. The policy only controls the
    longitudinal states (progress, velocity) of the agent. These longitudinal states are used to propagate the agent
    along a given path.

    Attributes:
        target_speed: float, the target speed for the agent in km/h.
        time_stamps: int, the number of time stamps for the simulation.
        dt: float, the time step for the simulation in seconds.
        routing_target_lane: Lane, the target lane for the agent to follow.
        available_routing_index_range: list, the range of available lane indices for routing.
        overtake_timer: int, a timer to control the frequency of lane changes for overtaking.
        enable_lane_change: bool, a flag to enable or disable lane changing.
        disable_idm_deceleration: bool, a flag to disable IDM deceleration.
        heading_pid: PIDController, a PID controller for heading control.
        lateral_pid: PIDController, a PID controller for lateral control.
    """

    def __init__(self, agent, config = None, random_seed = None):
        """
        Initialize the IDMPolicy.

        Args:
            agent: BaseAgent, the agent that this policy controls.
            config: Dict, the configuration for the policy.
            random_seed: int, the random seed for reproducibility.
        """
        super(IDMPolicy, self).__init__(agent=agent, config=config, random_seed=random_seed)
        self.target_speed = self.agent.config.NORMAL_SPEED # km/h
        self.time_stamps = self.agent.config.TIME_STAMPS
        self.dt = self.agent.config.dt
        self.routing_target_lane = None
        self.available_routing_index_range = None
        self.overtake_timer = self.np_random.randint(0, self.agent.config.LANE_CHANGE_FREQ)
        self.enable_lane_change = False
        self.disable_idm_deceleration = False
        

    def act(self):
        """
        Perform an action for the agent. This method is called at each time step of the simulation.
        
        Return:
        planned_trajectory: Trajectory, the planned trajectory for the agent.
        """
        last_target_lane = self.routing_target_lane
        # concat lane
        success = self.move_to_next_road()
        all_agents = self.engine.agents  
        # filter self.agent
        all_agents = [agent for agent in all_agents if agent != self.agent]

        if last_target_lane != self.routing_target_lane:
            logging.debug(f"step:{self.engine.episode_step}, routing target lane has changed")

        try:
            if success and self.enable_lane_change:
                # perform lane change due to routing
                acc_front_obj, acc_front_dist, steering_target_lane = self.lane_change_policy(all_agents)
            else:
                # can not find routing target lane
                acc_front_obj_id, acc_front_dist, _, _ = self.engine.agent_manager.get_surrounding_agents(self.agent)
                if acc_front_obj_id and acc_front_dist is not None:
                    min_dist_idx = np.argmin(acc_front_dist)
                    acc_front_obj = self.engine.agent_manager.all_agents[acc_front_obj_id[min_dist_idx]]
                    acc_front_dist = acc_front_dist[min_dist_idx]
                else:
                    acc_front_obj = None
                    acc_front_dist = 15
                steering_target_lane = self.routing_target_lane
        except:
            # error fallback
            acc_front_obj = None
            acc_front_dist = 15
            steering_target_lane = self.routing_target_lane
            logging.warning("IDM bug! fall back")

        target_acceleration = self.calculate_target_acc(acc_front_obj, acc_front_dist)
        planned_trajectory = self.generate_trajectory(target_acceleration, steering_target_lane)
        
        return planned_trajectory
    
    def filter_surrounding_agents(self, all_agents, radius=10):
        """
        Filter the agents that are within a certain radius around the ego vehicle.

        Args:
            all_agents: list, a list of all agents in the simulation environment.
            radius: float, the radius within which to filter the agents (d.

        Return:
            list, a list of agents that are within the specified radius around the ego vehicle.
        """
        surrounding_agents = []
        ego_position = self.agent.current_position

        for agent in all_agents:
            distance = np.linalg.norm(np.array(agent.current_position) - np.array(ego_position))
            if distance <= radius:
                surrounding_agents.append(agent)

        return surrounding_agents

    def steering_control(self, target_lane) -> float:
        """
        Calculate the steering control for the ego vehicle.

        Args: 
        target_lane: Lane, the target lane for the ego vehicle.

        Return: 
        steering: float, the calculated steering angle in radians.
        """
        # heading control following a lateral distance control
        ego_vehicle = self.agent
        long, lat = target_lane.local_coordinates(ego_vehicle.current_position)
        lane_heading = target_lane.heading_theta_at(long + 1)
        v_heading = float(ego_vehicle.current_heading)
        steering = self.heading_pid.get_result(wrap_to_pi(lane_heading - v_heading)) 
        steering += self.lateral_pid.get_result(lat) 
        return float(steering)

    def generate_trajectory(self, target_acc, target_lane, lane_change = False):
        """
        Generate a trajectory from the given acceleration and steering values.
        
        Args:
        acc: float, the acceleration value to use.
        steering: float, the steering value to use.

        Return:
        Trajectory: Trajectory, a Trajectory object that represents the generated trajectory
        """
        if lane_change is not True:
            # lane_change is False means vehicle's heading is the same as lane heading
            cur_long, cur_lat = target_lane.local_coordinates(self.agent.current_position)
            if cur_long > target_lane.length: # if the vehicle is out of the target_lane, update here
                target_lane = self.agent.navigation.current_lane
                cur_long, cur_lat = target_lane.local_coordinates(self.agent.current_position)
            
            initial_speed = self.agent.current_speed 

            timestamps = np.arange(0, self.time_stamps * self.dt, self.dt)
            valid_longtitude = cur_long + initial_speed * timestamps + 0.5 * target_acc * timestamps**2
            valid_latitude = np.linspace(cur_lat, cur_lat, self.time_stamps)
            
            waypoints_in_lane = np.vstack((valid_longtitude, valid_latitude)).T

            waypoints = np.array([
            target_lane.position(p[0], p[1]) for p in waypoints_in_lane
            ]).reshape(-1, 2)

            headings = np.array([
                target_lane.heading_theta_at(p) for p in valid_longtitude
            ])
            angular_velocities = []
            speed = []

        waypoints = np.array(waypoints, dtype=np.float32)
        speed = (waypoints[1:] - waypoints[:-1]) / self.dt
        speed = np.vstack([speed, speed[-1]])
        headings = np.array(headings, dtype=np.float32)
        angular_velocities = (headings[1:] - headings[:-1]) / self.dt
        angular_velocities = np.append(angular_velocities, angular_velocities[-1])
        
        return Trajectory(
            waypoints=waypoints,
            velocities=speed,
            headings=headings,
            angular_velocities=angular_velocities
        )

    @property
    def is_current_step_valid(self):
        return True

    def calculate_target_acc(self, front_obj, dist_to_front):
        """
        Calculate the target acceleration for the ego vehicle.

        Args:
        front_obj: BaseAgent, the object in front of the ego vehicle. This can be None if there is no object in front.
        dist_to_front: float, the distance to the object in front. This should be a positive value.

        Return:
        acceleration: float, the calculated acceleration in meters per second squared (m/s^2).
        """
        ego_vehicle = self.agent
        ego_target_speed = not_zero(self.target_speed, 0) # km/h

        speed_ratio = max(ego_vehicle.speed_km_h, 0) / ego_target_speed
        speed_ratio = min(speed_ratio, 1.0) 

        speed_ratio_power = np.power(speed_ratio, self.agent.config.DELTA)
        acceleration = self.agent.config.ACC_FACTOR * (1 - speed_ratio_power)
        if front_obj and (not self.disable_idm_deceleration):
            d = dist_to_front
            speed_diff = self.desired_gap(ego_vehicle, front_obj) / not_zero(d)
            acceleration += self.agent.config.DEACC_FACTOR * (speed_diff**2)
            # DEACC_FACTOR is a minus value 
        elif  ego_vehicle.speed_km_h >= self.target_speed:
            acceleration += self.agent.config.DEACC_FACTOR * (ego_vehicle.speed_km_h - self.target_speed) / ego_target_speed
        
        acceleration = np.clip(acceleration, self.agent.config.DEACC_FACTOR, self.agent.config.ACC_FACTOR)

        if ego_vehicle.speed_km_h < 0.008 and dist_to_front < 10.5: # 0.008 is set from experiment
            acceleration = 0.0

        return acceleration  # m/s^2
    
    def move_to_next_road(self):
        current_lanes = self.agent.navigation.current_ref_lanes
        
        # First check if the vehicle is out of routing_target_lane
        if self.routing_target_lane is not None:
            cur_long, _ = self.routing_target_lane.local_coordinates(self.agent.current_position)
            if cur_long > self.routing_target_lane.length * 0.9:
                # If out of routing_target_lane, update to current lane
                self.routing_target_lane = self.agent.navigation.current_lane
                return True if self.routing_target_lane in current_lanes else False
        
        if self.routing_target_lane is None:
            self.routing_target_lane = self.agent.navigation.current_lane
            return True if self.routing_target_lane in current_lanes else False
        routing_network = self.agent.navigation.map.road_network
        if self.routing_target_lane not in current_lanes:
            for lane in current_lanes:
                if self.routing_target_lane.is_previous_lane_of(lane) or \
                        routing_network.has_connection(self.routing_target_lane.index, lane.index):
                    # two lanes connect
                    self.routing_target_lane = lane
                    return True
                    # lane change for lane num change
            return False
        elif self.agent.navigation.current_lane in current_lanes and self.routing_target_lane is not self.agent.navigation.current_lane:
            # lateral routing lane change
            self.routing_target_lane = self.agent.navigation.current_lane
            self.overtake_timer = self.np_random.randint(0, int(self.agent.config.LANE_CHANGE_FREQ / 2))
            return True
        else:
            return True
        
    def desired_gap(self, ego_vehicle, front_obj, projected: bool = True) -> float:
        """
        Calculate the desired gap between the ego vehicle and the front object.

        Args:
        ego_vehicle: BaseAgent, the ego vehicle object, which contains information about its current state.
        front_obj: BaseAgent, the object in front of the ego vehicle, which contains information about its current state.
        projected: bool, a boolean indicating whether to project the velocity difference onto the heading direction of the ego vehicle.
                        If True, the velocity difference is projected onto the heading direction. If False, the raw velocity difference is used.
        Return:
        d_star: float, the desired gap (d_star) in meters.
        """
        d0 = self.agent.config.DISTANCE_WANTED
        tau = self.agent.config.TIME_WANTED
        ab = -self.agent.config.ACC_FACTOR * self.agent.config.DEACC_FACTOR
        dv = ego_vehicle.speed_km_h - front_obj.speed_km_h
        d_star = d0 + max(0, ego_vehicle.speed_km_h * tau + ego_vehicle.speed_km_h * dv / (2 * np.sqrt(ab)))
        return d_star

    def reset(self):
        super(IDMPolicy, self).reset()
        self.target_speed = self.agent.config.NORMAL_SPEED
        self.routing_target_lane = None
        self.available_routing_index_range = None
        self.overtake_timer = self.np_random.randint(0, self.agent.config.LANE_CHANGE_FREQ)
        

    def lane_change_policy(self, all_agents): 
        """
        Determine the lane change policy for the ego vehicle based on the surrounding agents and road conditions.

        Args:
            all_agents: list, a list of all agents in the simulation environment.

        Return:
            Tuple containing:
                - front_object: BaseAgent, the object in front of the ego vehicle after lane change decision.
                - front_distance: float, the distance to the front object after lane change decision.
                - target_lane: Lane, the target lane for the ego vehicle after lane change decision.
        """
        current_lanes = self.agent.navigation.current_ref_lanes
        
        surrounding_objects = FrontBackObjects.get_find_front_back_objs(
            all_agents, self.routing_target_lane, self.agent.current_position, self.agent.config.MAX_LONG_DIST, current_lanes
        )
        surrounding_objects = None
        self.available_routing_index_range = [i for i in range(len(current_lanes))]

        next_lanes = self.agent.navigation.next_ref_lanes
        

        lane_num_diff = len(current_lanes) - len(next_lanes) if next_lanes is not None else 0

        # We have to perform lane changing because the number of lanes in next road is less than current road
        if lane_num_diff > 0:
            # lane num decreasing happened in left road or right road
            if current_lanes[0].is_previous_lane_of(next_lanes[0]):
                index_range = [i for i in range(len(next_lanes))]
            else:
                index_range = [i for i in range(lane_num_diff, len(current_lanes))]
            self.available_routing_index_range = index_range
            if self.routing_target_lane.index[-1] not in index_range:
                # not on suitable lane do lane change !!!
                if self.routing_target_lane.index[-1] > index_range[-1]:
                    # change to left
                    if surrounding_objects.left_back_min_distance(
                    ) < self.agent.config.SAFE_LANE_CHANGE_DISTANCE or surrounding_objects.left_front_min_distance() < 5:
                        # creep to wait
                        self.target_speed = self.agent.config.CREEP_SPEED
                        return surrounding_objects.front_object(), surrounding_objects.front_min_distance(
                        ), self.routing_target_lane
                    else:
                        # it is time to change lane!
                        self.target_speed = self.agent.config.NORMAL_SPEED
                        return surrounding_objects.left_front_object(), surrounding_objects.left_front_min_distance(), \
                               current_lanes[self.routing_target_lane.index[-1] - 1]
                else:
                    # change to right
                    if surrounding_objects.right_back_min_distance(
                    ) < self.agent.config.SAFE_LANE_CHANGE_DISTANCE or surrounding_objects.right_front_min_distance() < 5:
                        # unsafe, creep and wait
                        self.target_speed = self.config.CREEP_SPEED
                        return surrounding_objects.front_object(), surrounding_objects.front_min_distance(
                        ), self.routing_target_lane,
                    else:
                        # change lane
                        self.target_speed = self.agent.config.NORMAL_SPEED
                        return surrounding_objects.right_front_object(), surrounding_objects.right_front_min_distance(), \
                               current_lanes[self.routing_target_lane.index[-1] + 1]

        # lane follow or active change lane/overtake for high driving speed
        if abs(self.agent.speed_km_h - self.agent.config.NORMAL_SPEED) > 3 and surrounding_objects.has_front_object(
        ) and abs(surrounding_objects.front_object().speed_km_h -
                  self.agent.config.NORMAL_SPEED) > 3 and self.overtake_timer > self.agent.config.LANE_CHANGE_FREQ:
            # may lane change
            right_front_speed = surrounding_objects.right_front_object().speed_km_h if surrounding_objects.has_right_front_object() else self.agent.config.MAX_SPEED \
                if surrounding_objects.right_lane_exist() and surrounding_objects.right_front_min_distance() > self.agent.config.SAFE_LANE_CHANGE_DISTANCE and surrounding_objects.right_back_min_distance() > self.agent.config.SAFE_LANE_CHANGE_DISTANCE else None
            front_speed = surrounding_objects.front_object().speed_km_h if surrounding_objects.has_front_object(
            ) else self.agent.config.MAX_SPEED
            left_front_speed = surrounding_objects.left_front_object().speed_km_h if surrounding_objects.has_left_front_object() else self.agent.config.MAX_SPEED \
                if surrounding_objects.left_lane_exist() and surrounding_objects.left_front_min_distance() > self.agent.config.SAFE_LANE_CHANGE_DISTANCE and surrounding_objects.left_back_min_distance() > self.agent.config.SAFE_LANE_CHANGE_DISTANCE else None
            if left_front_speed is not None and left_front_speed - front_speed > self.agent.config.LANE_CHANGE_SPEED_INCREASE:
                # left overtake has a high priority
                expect_lane_idx = current_lanes.index(self.routing_target_lane) - 1
                if expect_lane_idx in self.available_routing_index_range:
                    return surrounding_objects.left_front_object(), surrounding_objects.left_front_min_distance(), \
                           current_lanes[expect_lane_idx]
            if right_front_speed is not None and right_front_speed - front_speed > self.agent.config.LANE_CHANGE_SPEED_INCREASE:
                expect_lane_idx = current_lanes.index(self.routing_target_lane) + 1
                if expect_lane_idx in self.available_routing_index_range:
                    return surrounding_objects.right_front_object(), surrounding_objects.right_front_min_distance(), \
                           current_lanes[expect_lane_idx]

        # fall back to lane follow
        self.target_speed = self.agent.config.NORMAL_SPEED
        self.overtake_timer += 1
        return surrounding_objects.front_object(), surrounding_objects.front_min_distance(), self.routing_target_lane