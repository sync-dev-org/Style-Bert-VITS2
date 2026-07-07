from collections.abc import Mapping
from typing import Any

from torch import nn
from torch.nn.utils.parametrizations import weight_norm as parametrized_weight_norm
from torch.nn.utils.parametrize import remove_parametrizations


def weight_norm(module: nn.Module, name: str = "weight", dim: int = 0) -> nn.Module:
    return parametrized_weight_norm(module, name=name, dim=dim)


def remove_weight_norm(module: nn.Module, name: str = "weight") -> None:
    remove_parametrizations(module, name, leave_parametrized=True)


def migrate_legacy_weight_norm_state_dict(
    state_dict: Mapping[str, Any],
) -> dict[str, Any]:
    migrated: dict[str, Any] = {}
    for key, value in state_dict.items():
        if key.endswith(".weight_g"):
            new_key = key[: -len(".weight_g")] + ".parametrizations.weight.original0"
        elif key == "weight_g":
            new_key = "parametrizations.weight.original0"
        elif key.endswith(".weight_v"):
            new_key = key[: -len(".weight_v")] + ".parametrizations.weight.original1"
        elif key == "weight_v":
            new_key = "parametrizations.weight.original1"
        else:
            new_key = key
        migrated[new_key] = value
    return migrated
