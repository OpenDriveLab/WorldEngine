from omegaconf import DictConfig, OmegaConf

from worldengine.envs.base_env import BaseEnv


def build_env(config, name, data):
    """
    Return the navigation class for target object.
    """
    # Resolve OmegaConf interpolations (like ${now:...}) before passing to env.
    # This is necessary because Hydra-specific resolvers are not available
    # when the config is serialized to Ray workers.
    if isinstance(config, DictConfig):
        config = OmegaConf.to_container(config, resolve=True)
        config = OmegaConf.create(config)

    if config["env"] == "base_env":
        return BaseEnv(config, name, data)
    else:
        raise NotImplementedError(f'The assigned env {config["env"]} is not'
                                  f'implemented.')
