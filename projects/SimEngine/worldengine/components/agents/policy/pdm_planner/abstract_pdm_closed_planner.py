from typing import List, Optional
import logging

import numpy as np
import numpy.typing as npt

from nuplan.common.actor_state.state_representation import StateSE2, TimePoint, StateVector2D
from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.geometry.convert import relative_to_absolute_poses
from nuplan.common.maps.abstract_map_objects import LaneGraphEdgeMapObject
from nuplan.planning.simulation.planner.abstract_planner import PlannerInput
from nuplan.planning.simulation.planner.ml_planner.transform_utils import (
    _get_fixed_timesteps,
    _se2_vel_acc_to_ego_state,
)
from nuplan.planning.simulation.trajectory.interpolated_trajectory import (
    InterpolatedTrajectory,
)
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from worldengine.common.dataclasses import NavsimTrajectory
from worldengine.components.agents.policy.pdm_planner.abstract_pdm_planner import AbstractPDMPlanner
from worldengine.components.agents.policy.pdm_planner.observation.pdm_observation import PDMObservation
from worldengine.components.agents.policy.pdm_planner.proposal.batch_idm_policy import BatchIDMPolicy
from worldengine.components.agents.policy.pdm_planner.proposal.pdm_generator import PDMGenerator
from worldengine.components.agents.policy.pdm_planner.proposal.pdm_proposal import PDMProposalManager
from worldengine.components.agents.policy.pdm_planner.scoring.pdm_scorer import PDMScorer
from worldengine.components.agents.policy.pdm_planner.simulation.pdm_simulator import PDMSimulator
from worldengine.components.agents.policy.pdm_planner.utils.pdm_emergency_brake import PDMEmergencyBrake
from worldengine.components.agents.policy.pdm_planner.utils.pdm_geometry_utils import parallel_discrete_path
from worldengine.components.agents.policy.pdm_planner.utils.pdm_array_representation import ego_states_to_state_array
from worldengine.components.agents.policy.pdm_planner.observation.pdm_occupancy_map import PDMDrivableMap
from worldengine.common.dataclasses import PDMResults
from worldengine.components.agents.policy.pdm_planner.utils.pdm_path import PDMPath
from worldengine.components.agents.policy.pdm_planner.utils.pdm_enums import (
    MultiMetricIndex,
    WeightedMetricIndex,
)


class AbstractPDMClosedPlanner(AbstractPDMPlanner):
    """
    Interface for planners incorporating PDM-Closed. Used for PDM-Closed and PDM-Hybrid.
    """

    def __init__(
        self,
        trajectory_sampling: TrajectorySampling,
        proposal_sampling: TrajectorySampling,
        idm_policies: BatchIDMPolicy,
        lateral_offsets: Optional[List[float]],
        map_radius: float,
    ):
        """
        Constructor for AbstractPDMClosedPlanner
        :param trajectory_sampling: Sampling parameters for final trajectory
        :param proposal_sampling: Sampling parameters for proposals
        :param idm_policies: BatchIDMPolicy class
        :param lateral_offsets: centerline offsets for proposals (optional)
        :param map_radius: radius around ego to consider
        """

        super(AbstractPDMClosedPlanner, self).__init__(map_radius)

        assert (
            trajectory_sampling.interval_length == proposal_sampling.interval_length
        ), "AbstractPDMClosedPlanner: Proposals and Trajectory must have equal interval length!"

        # config parameters
        self._trajectory_sampling: int = trajectory_sampling
        self._proposal_sampling: int = proposal_sampling
        self._idm_policies: BatchIDMPolicy = idm_policies
        self._lateral_offsets: Optional[List[float]] = lateral_offsets

        # observation/forecasting class
        self._observation = PDMObservation(
            trajectory_sampling, proposal_sampling, map_radius
        )

        # proposal/trajectory related classes
        self._generator = PDMGenerator(trajectory_sampling, proposal_sampling)
        self._simulator = PDMSimulator(proposal_sampling)
        self._scorer = PDMScorer(proposal_sampling)
        self._emergency_brake = PDMEmergencyBrake(trajectory_sampling)

        # lazy loaded
        self._proposal_manager: Optional[PDMProposalManager] = None
        self.logger = logging.getLogger("PDMPlanner")

    def _update_proposal_manager(self, ego_state: EgoState):
        """
        Updates or initializes PDMProposalManager class
        :param ego_state: state of ego-vehicle
        """

        current_lane = self._get_starting_lane(ego_state)

        # TODO: Find additional conditions to trigger re-planning
        create_new_proposals = self._iteration == 0

        if create_new_proposals:
            proposal_paths: List[PDMPath] = self._get_proposal_paths(current_lane)

            self._proposal_manager = PDMProposalManager(
                lateral_proposals=proposal_paths,
                longitudinal_policies=self._idm_policies,
            )

        # update proposals
        self._proposal_manager.update(current_lane.speed_limit_mps)

    def _get_proposal_paths(
        self, current_lane: LaneGraphEdgeMapObject
    ) -> List[PDMPath]:
        """
        Returns a list of path's to follow for the proposals. Inits a centerline.
        :param current_lane: current or starting lane of path-planning
        :return: lists of paths (0-index is centerline)
        """
        centerline_discrete_path = self._get_discrete_centerline(current_lane)
        self._centerline = PDMPath(centerline_discrete_path)

        # 1. save centerline path (necessary for progress metric)
        output_paths: List[PDMPath] = [self._centerline]

        # 2. add additional paths with lateral offset of centerline
        if self._lateral_offsets is not None:
            for lateral_offset in self._lateral_offsets:
                offset_discrete_path = parallel_discrete_path(
                    discrete_path=centerline_discrete_path, offset=lateral_offset
                )
                output_paths.append(PDMPath(offset_discrete_path))

        return output_paths

    def _get_closed_loop_trajectory(
        self,
        current_input: PlannerInput,
    ) -> InterpolatedTrajectory:
        """
        Creates the closed-loop trajectory for PDM-Closed planner.
        :param current_input: planner input
        :return: trajectory
        """

        ego_state, observation = current_input.history.current_state

        # 1. Environment forecast and observation update
        self._observation.update(
            ego_state,
            observation,
            current_input.traffic_light_data,
            self._route_lane_dict,
        )

        # 2. Centerline extraction and proposal update
        self._update_proposal_manager(ego_state)

        # 3. Generate/Unroll proposals
        proposals_array = self._generator.generate_proposals(
            ego_state, self._observation, self._proposal_manager
        )

        # 4. Simulate proposals
        simulated_proposals_array = self._simulator.simulate_proposals(
            proposals_array, ego_state
        )

        # 5. Score proposals
        proposal_scores = self._scorer.score_proposals(
            simulated_proposals_array,
            ego_state,
            self._observation,
            self._centerline,
            self._route_lane_dict,
            self._drivable_area_map,
            self._map_api,
        )

        # 6.a Apply brake if emergency is expected
        # trajectory = self._emergency_brake.brake_if_emergency(
        #     ego_state, proposal_scores, self._scorer
        # )

        # 6.b Otherwise, extend and output best proposal
        # if trajectory is not None:
        #     self.logger.info("emergency brake in PDM traj")
        # else:
        trajectory = simulated_proposals_array[np.argmax(proposal_scores)]

        return trajectory
    
    def _get_pdm_results_from_input(
        self,
        current_input: PlannerInput,
        input_trajectory: NavsimTrajectory,
        input_trajectory_velocities: npt.NDArray[np.float64],
    ) -> PDMResults:
        """
        Creates the closed-loop trajectory for PDM-Closed planner.
        :param current_input: planner input
        :return: trajectory
        """

        ego_state, observation = current_input.history.current_state
        self._drivable_area_map = PDMDrivableMap.from_simulation(
            self._map_api, ego_state, self._map_radius
        )

        # 1. Environment forecast and observation update
        self._observation.update(
            ego_state,
            observation,
            current_input.traffic_light_data,
            self._route_lane_dict,
        )

        # 2. Centerline extraction
        current_lane = self._get_starting_lane(ego_state)
        centerline_discrete_path = self._get_discrete_centerline(current_lane)
        self._centerline = PDMPath(centerline_discrete_path)


        # 3. Set Trajectory
        pred_trajectory = transform_trajectory(input_trajectory, ego_state)
        pred_states = get_trajectory_as_array(pred_trajectory, input_trajectory.trajectory_sampling, ego_state.time_point)
        trajectory_states = np.concatenate([pred_states[None, ...]], axis=0)
        trajectory_states[0,:,3] = input_trajectory_velocities[:,0]

        # 4. Simulate proposals
        simulated_proposals_array = self._simulator.simulate_proposals(
            trajectory_states, ego_state
        )
       

        # 5. Score proposals
        scores = self._scorer.score_proposals(
            simulated_proposals_array,
            ego_state,
            self._observation,
            self._centerline,
            self._route_lane_dict,
            self._drivable_area_map,
            self._map_api,
        )

        pred_idx = 0 # only exists pred_states

        no_at_fault_collisions = self._scorer._multi_metrics[MultiMetricIndex.NO_COLLISION, pred_idx]
        drivable_area_compliance = self._scorer._multi_metrics[MultiMetricIndex.DRIVABLE_AREA, pred_idx]
        driving_direction_compliance = self._scorer._multi_metrics[MultiMetricIndex.DRIVING_DIRECTION, pred_idx]

        ego_progress = self._scorer._weighted_metrics[WeightedMetricIndex.PROGRESS, pred_idx]
        time_to_collision_within_bound = self._scorer._weighted_metrics[WeightedMetricIndex.TTC, pred_idx]
        comfort = self._scorer._weighted_metrics[WeightedMetricIndex.COMFORTABLE, pred_idx]

        score = scores[pred_idx]

        return PDMResults(
            no_at_fault_collisions,
            drivable_area_compliance,
            ego_progress,
            time_to_collision_within_bound,
            comfort,
            driving_direction_compliance,
            score,
        )

def transform_trajectory(pred_trajectory: NavsimTrajectory, initial_ego_state: EgoState) -> InterpolatedTrajectory:
    """
    Transform trajectory in global frame and return as InterpolatedTrajectory
    :param pred_trajectory: trajectory dataclass in ego frame
    :param initial_ego_state: nuPlan's ego state object
    :return: nuPlan's InterpolatedTrajectory
    """

    future_sampling = pred_trajectory.trajectory_sampling
    timesteps = _get_fixed_timesteps(initial_ego_state, future_sampling.time_horizon - future_sampling.interval_length, future_sampling.interval_length)

    relative_poses = np.array(pred_trajectory.poses, dtype=np.float64)
    relative_states = [StateSE2.deserialize(pose) for pose in relative_poses]
    absolute_states = []
    for relative_state in relative_states:
        absolute_state = StateSE2(
            x=relative_state.x + initial_ego_state.rear_axle.x,
            y=relative_state.y + initial_ego_state.rear_axle.y,
            heading=relative_state.heading,
        )
        absolute_states.append(absolute_state)

    # NOTE: velocity and acceleration ignored by LQR + bicycle model
    agent_states = [
        _se2_vel_acc_to_ego_state(
            state,
            [0.0, 0.0],
            [0.0, 0.0],
            timestep,
            initial_ego_state.car_footprint.vehicle_parameters,
        )
        for state, timestep in zip(absolute_states[1:], timesteps)
    ]

    # NOTE: maybe make addition of initial_ego_state optional
    return InterpolatedTrajectory([initial_ego_state] + agent_states)

def get_trajectory_as_array(
    trajectory: InterpolatedTrajectory,
    future_sampling: TrajectorySampling,
    start_time: TimePoint,
) -> npt.NDArray[np.float64]:
    """
    Interpolated trajectory and return as numpy array
    :param trajectory: nuPlan's InterpolatedTrajectory object
    :param future_sampling: Sampling parameters for interpolation
    :param start_time: TimePoint object of start
    :return: Array of interpolated trajectory states.
    """

    times_s = np.arange(
        0.0,
        future_sampling.time_horizon,
        future_sampling.interval_length,
    )
    times_s += start_time.time_s
    times_us = [int(time_s * 1e6) for time_s in times_s]
    times_us = np.clip(times_us, trajectory.start_time.time_us, trajectory.end_time.time_us)
    time_points = [TimePoint(time_us) for time_us in times_us]

    trajectory_ego_states: List[EgoState] = trajectory.get_state_at_times(time_points)

    return ego_states_to_state_array(trajectory_ego_states)
