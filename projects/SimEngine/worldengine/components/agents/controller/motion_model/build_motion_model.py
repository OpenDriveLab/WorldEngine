from worldengine.components.agents.controller.motion_model.kinematic_bicycle import KinematicBicycleModel


def build_motion_model(config):

    motion_model = config.get('motion_model', 'kinematic_bicycle')
    if motion_model == 'kinematic_bicycle':
        return KinematicBicycleModel
    else:
        raise NotImplementedError(f'The assigned MotionModel {motion_model} is not'
                                  f'implemented.')
