from worldengine.components.agents.controller.two_stage_controller import TwoStageController
from worldengine.components.agents.controller.log_play_controller import LogPlayController


def build_controller(object_id, config):
    """
    Return the controller class for target object.
    """
    if object_id == 'ego':
        controller = config.get('ego_controller', 'two_stage_controller')
    else:
        controller = config.get('agent_controller', 'log_play_controller')

    if controller == 'two_stage_controller':
        return TwoStageController
    elif controller == 'log_play_controller':
        return LogPlayController
    else:
        raise NotImplementedError(f'The assigned controller {controller} is not'
                                  f'implemented.')
