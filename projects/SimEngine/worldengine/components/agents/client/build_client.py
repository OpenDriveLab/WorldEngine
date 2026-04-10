from worldengine.components.agents.client.navformer_client import NAVFormerClient
from worldengine.components.agents.client.base_client import BaseClient

def build_client(object_id: str, config):
    """
    Return the client class for target object.
    """
    if object_id == 'ego':
        client = config.get('ego_client')
    else:
        return BaseClient

    if client == 'navformer_client':
        return NAVFormerClient
    elif client == 'base_client':
        return BaseClient
    else:
        raise NotImplementedError
