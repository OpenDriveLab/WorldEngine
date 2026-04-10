from worldengine.components.agents.policy.trajectory_policy import TrajectoryPolicy
from worldengine.components.agents.policy.human_policy import HumanPolicy
from worldengine.components.agents.policy.dbg_ego_policy import DebugEgoPolicy
from worldengine.components.agents.policy.idm_policy import IDMPolicy
from worldengine.components.agents.policy.pdm_policy import PDMPolicy
from worldengine.components.agents.policy.env_input_policy import EnvInputPolicy

def build_policy(object_id: str, config):
    """
    Return the navigation class for target object.
    """

    if object_id == 'ego':
        policy = config.get('ego_policy')
    else:
        policy = config.get('agent_policy', 'trajectory_policy')

    if policy == 'trajectory_policy':
        return TrajectoryPolicy
    elif policy == 'dbg_ego_policy':
        return DebugEgoPolicy
    elif policy == 'human_policy':
        return HumanPolicy
    elif policy == 'idm_policy':
        return IDMPolicy
    elif policy == 'pdm_policy':
        return PDMPolicy
    elif policy == 'env_input_policy':
        return EnvInputPolicy
    else:
        raise NotImplementedError(f'The assigned policy {policy} is not'
                                  f'implemented.')
