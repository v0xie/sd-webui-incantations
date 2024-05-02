from typing import Optional, Callable, Dict
from collections import OrderedDict
from warnings import warn
import logging
import torch


from modules import shared


logger = logging.getLogger(__name__)


def modules_add_field(modules, field, value=None):
    """ Add a field to a module if it isn't already added.
    Args:
        modules (list): Module or list of modules to add the field to
        field (str): Field name to add
        value (any): Value to assign to the field
    Returns:
        None

    """
    if not isinstance(modules, list):
        modules = [modules]
    for module in modules:
        if not hasattr(module, field):
            setattr(module, field, value)
        else:
            logger.warning(f"Field {field} already exists in module {module}")


def modules_remove_field(modules, field):
    """ Remove a field from a module if it exists.
    Args:
        modules (list): Module or list of modules to add the field to
        field (str): Field name to add
        value (any): Value to assign to the field
    Returns:
        None

    """
    if not isinstance(modules, list):
        modules = [modules]
    for module in modules:
        if hasattr(module, field):
                delattr(module, field)
        else:
            logger.warning(f"Field {field} does not exist in module {module}")


def get_modules(network_layer_name_filter: Optional[str] = None, module_name_filter: Optional[str] = None):
    """ Get all modules from the shared.sd_model that match the filters provided. If no filters are provided, all modules are returned.

    Args:
        network_layer_name_filter (Optional[str], optional): Filters the modules by network layer name. Defaults to None. Example: "attn1" will return all modules that have "attn1" in their network layer name.
        module_name_filter (Optional[str], optional): Filters the modules by module class name. Defaults to None. Example: "CrossAttention" will return all modules that have "CrossAttention" in their class name.

    Returns:
        list: List of modules that match the filters provided.
    """
    try:
        m = shared.sd_model
        nlm = m.network_layer_mapping
        sd_model_modules = nlm.values()

        # Apply filters if they are provided
        if network_layer_name_filter is not None:
            sd_model_modules = list(filter(lambda m: network_layer_name_filter in m.network_layer_name, sd_model_modules))
        if module_name_filter is not None:
            sd_model_modules = list(filter(lambda m: module_name_filter in m.__class__.__name__, sd_model_modules))
        return sd_model_modules
    except AttributeError:
        logger.exception("AttributeError in get_modules", stack_info=True)
        return []
    except Exception:
        logger.exception("Exception in get_modules", stack_info=True)
        return []


# workaround for torch remove hooks issue
# thank you to @ProGamerGov for this https://github.com/pytorch/pytorch/issues/70455
def remove_module_forward_hook(
    module: torch.nn.Module, hook_fn_name: Optional[str] = None
) -> None:
    """
    This function removes all forward hooks in the specified module, without requiring
    any hook handles. This lets us clean up & remove any hooks that weren't property
    deleted.

    Warning: Various PyTorch modules and systems make use of hooks, and thus extreme
    caution should be exercised when removing all hooks. Users are recommended to give
    their hook function a unique name that can be used to safely identify and remove
    the target forward hooks.

    Args:

        module (nn.Module): The module instance to remove forward hooks from.
        hook_fn_name (str, optional): Optionally only remove specific forward hooks
            based on their function's __name__ attribute.
            Default: None
    """

    if hook_fn_name is None:
        warn("Removing all active hooks can break some PyTorch modules & systems.")

    def _remove_hooks(m: torch.nn.Module, name: Optional[str] = None) -> None:
        if hasattr(module, "_forward_hooks"):
            if m._forward_hooks != OrderedDict():
                if name is not None:
                    dict_items = list(m._forward_hooks.items())
                    m._forward_hooks = OrderedDict(
                        [(i, fn) for i, fn in dict_items if fn.__name__ != name]
                    )
                else:
                    m._forward_hooks: Dict[int, Callable] = OrderedDict()

    def _remove_child_hooks(
        target_module: torch.nn.Module, hook_name: Optional[str] = None
    ) -> None:
        for name, child in target_module._modules.items():
            if child is not None:
                _remove_hooks(child, hook_name)
                _remove_child_hooks(child, hook_name)

    # Remove hooks from target submodules
    _remove_child_hooks(module, hook_fn_name)

    # Remove hooks from the target module
    _remove_hooks(module, hook_fn_name)


def module_add_forward_hook(module, hook_fn, hook_type="forward", with_kwargs=False):
    """ Adds a forward hook to a module.

    hook_fn should be a function that accepts the following arguments:
        forward hook, no kwargs: hook(module, args, output) -> None or modified output
        forward hook, with kwargs: hook(module, args, kwargs output) -> None or modified output

    Args:
        module (torch.nn.Module): Module to hook
        hook_fn (Callable): Function to call when the hook is triggered
        hook_type (str, optional): Type of hook to create. Defaults to "forward". Can be "forward" or "pre_forward".
        with_kwargs (bool, optional): Whether the hook function should accept keyword arguments. Defaults to False.

    Returns:
        torch.utils.hooks.RemovableHandle: Handle for the hook
    """
    if module is None:
        raise ValueError("module must be provided")
    if not callable(hook_fn):
        raise ValueError("hook_fn must be a callable function")

    if hook_type == "forward":
        handle = module.register_forward_hook(hook_fn, with_kwargs=with_kwargs)
    elif hook_type == "pre_forward":
        handle = module.register_forward_pre_hook(hook_fn, with_kwargs=with_kwargs)
    else:
        raise ValueError(f"Invalid hook type {hook_type}. Must be 'forward' or 'pre_forward'.")

    return handle