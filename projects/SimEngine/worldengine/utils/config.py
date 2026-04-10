"""
Configuration merging.
"""
import copy

from worldengine.base_class.config import Config


def merge_config_with_unknown_keys(old_dict, new_dict):
    return merge_config(old_dict, new_dict, new_keys_allowed=True)


def merge_config(old_dict, new_dict, new_keys_allowed=False):
    if isinstance(old_dict, Config):
        old_dict = old_dict.get_dict()
    if isinstance(new_dict, Config):
        new_dict = new_dict.get_dict()
    merged = merge_dicts(old_dict, new_dict, allow_new_keys=new_keys_allowed)
    return Config(merged)


def merge_dicts(old_dict, new_dict, allow_new_keys=False, without_copy=False):
    """
    Args:
        old_dict (dict, Config): Dict 1.
        new_dict (dict, Config): Dict 2.
        raise_error (bool): Whether to raise error if new key is found.

    Returns:
         dict: A new dict that is d1 and d2 deep merged.
    """
    old_dict = old_dict or dict()
    new_dict = new_dict or dict()
    if without_copy:
        merged = old_dict
    else:
        merged = copy.deepcopy(old_dict)
    _deep_update(
        merged, new_dict, new_keys_allowed=allow_new_keys, allow_new_subkey_list=[], raise_error=not allow_new_keys
    )
    return merged


def _deep_update(
    original,
    new_dict,
    new_keys_allowed=False,
    allow_new_subkey_list=None,
    override_all_if_type_changes=None,
    raise_error=True
):
    allow_new_subkey_list = allow_new_subkey_list or []
    override_all_if_type_changes = override_all_if_type_changes or []

    for k, value in new_dict.items():
        if k not in original and not new_keys_allowed:
            if raise_error:
                raise Exception("Unknown config parameter `{}` ".format(k))
            else:
                continue

        # Both orginal value and new one are dicts.
        if isinstance(original.get(k), dict) and isinstance(value, dict):
            # Check old type vs old one. If different, override entire value.
            if k in override_all_if_type_changes and \
                    "type" in value and "type" in original[k] and \
                    value["type"] != original[k]["type"]:
                original[k] = value
            # Allowed key -> ok to add new subkeys.
            elif k in allow_new_subkey_list:
                _deep_update(original[k], value, True, raise_error=raise_error)
            # Non-allowed key.
            else:
                _deep_update(original[k], value, new_keys_allowed, raise_error=raise_error)
        # Original value not a dict OR new value not a dict:
        # Override entire value.
        else:
            original[k] = value
    return original