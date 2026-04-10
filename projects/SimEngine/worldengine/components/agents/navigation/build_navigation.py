from worldengine.components.agents.navigation.trajectory_navigation import TrajectoryNavigation
from worldengine.components.agents.navigation.ego_lane_navigation import EgoLaneNavigation
from worldengine.components.agents.navigation.idm_navigation import IDMNavigation


def build_navigation(object_id: str, config):
    """
    Return the navigation class for target object.
    """

    if object_id == 'ego':
        navigation = config.get('ego_navigation', 'ego_lane_navigation')
    else:
        navigation = config.get('agent_navigation', 'trajectory_navigation')

    if navigation == 'ego_lane_navigation':
        return EgoLaneNavigation
    elif navigation == 'trajectory_navigation':
        return TrajectoryNavigation
    elif navigation == 'idm_navigation':
        return IDMNavigation
    else:
        raise NotImplementedError(f'The assigned navigation {navigation} is not'
                                  f'implemented.')