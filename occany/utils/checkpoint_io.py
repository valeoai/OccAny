from __future__ import annotations

import importlib
import pkgutil
import sys

import torch
import torch.distributed as dist


_LEGACY_MODULE_ALIASES_REGISTERED = False


def register_legacy_checkpoint_modules() -> None:
    """Register import aliases so old checkpoints can still be unpickled.

    Older OccAny checkpoints may contain references to
    ``occany.must3r_blocks.*`` while current code lives under
    ``occany.model.must3r_blocks.*``.
    """

    global _LEGACY_MODULE_ALIASES_REGISTERED
    if _LEGACY_MODULE_ALIASES_REGISTERED:
        return

    legacy_package = "occany.must3r_blocks"
    current_package = "occany.model.must3r_blocks"

    try:
        current_pkg_module = importlib.import_module(current_package)
    except ModuleNotFoundError:
        return

    sys.modules.setdefault(legacy_package, current_pkg_module)

    package_path = getattr(current_pkg_module, "__path__", None)
    if package_path is not None:
        for _, module_name, _ in pkgutil.iter_modules(package_path):
            current_module_name = f"{current_package}.{module_name}"
            legacy_module_name = f"{legacy_package}.{module_name}"
            try:
                current_module = importlib.import_module(current_module_name)
            except ModuleNotFoundError:
                continue
            sys.modules.setdefault(legacy_module_name, current_module)

    _LEGACY_MODULE_ALIASES_REGISTERED = True


def save_on_master(*args, **kwargs):

    if not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0:
        return torch.save(*args, **kwargs)

    return None
