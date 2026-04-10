from worldengine.common.randomizable import Randomizable
from worldengine.engine.engine_utils import get_engine, engine_initialized, get_global_config


class BaseManager(Randomizable):
    """
    Managers should be created and registered after launching BaseEngine
    """
    PRIORITY = 10  # the engine will call managers according to the priority

    def __init__(self):
        assert engine_initialized(), "You should not create manager before the initialization of BaseEngine"
        Randomizable.__init__(self, get_engine().global_random_seed)

    # some properties.
    @property
    def class_name(self):
        return self.__class__.__name__

    @property
    def engine(self):
        return get_engine()

    def get_metadata(self):
        """
        This function will store the metadata of each manager before the episode start, usually, we put some raw real
        world data in it, so that we won't lose information
        """
        assert self.episode_step == 0, "This func can only be called after env.reset() without any env.step() called"
        return {}
    
    @property
    def episode_step(self):
        """
        Return how many steps are taken from env.reset() to current step
        Returns:

        """
        return self.engine.episode_step

    @property
    def global_config(self):
        return get_global_config()

    # Step methods.
    def before_step(self, *args, **kwargs) -> dict:
        """
        Usually used to set actions for all elements with their policies
        """
        return dict()

    def step(self, *args, **kwargs):
        pass

    def after_step(self, *args, **kwargs) -> dict:
        """
        Update state for this manager after system advancing dt
        """
        return dict()

    # reset methods.
    def before_reset(self):
        """
        Update episode level config to this manager and clean element or detach element
        """
        return dict()

    def reset(self):
        """
        Generate objects according to some pre-defined rules
        """
        pass

    def after_reset(self):
        """
        Usually used to record information after all managers called reset(),
        Since reset() of managers may influence each other
        """
        pass
