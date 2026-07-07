from __future__ import annotations

import platform
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class TrainingDevice:
    device: torch.device
    backend: str
    ddp_device_ids: list[int] | None
    pin_memory: bool


def resolve_training_device(
    local_rank: int,
    system_name: str | None = None,
) -> TrainingDevice:
    if torch.cuda.is_available():
        if system_name is None:
            system_name = platform.system()
        backend = (
            "nccl"
            if system_name != "Windows" and torch.distributed.is_nccl_available()
            else "gloo"
        )
        return TrainingDevice(
            device=torch.device("cuda", local_rank),
            backend=backend,
            ddp_device_ids=[local_rank],
            pin_memory=True,
        )

    return TrainingDevice(
        device=torch.device("cpu"),
        backend="gloo",
        ddp_device_ids=None,
        pin_memory=False,
    )


def prepare_training_device(local_rank: int) -> TrainingDevice:
    training_device = resolve_training_device(local_rank)
    if training_device.device.type == "cuda":
        torch.cuda.set_device(training_device.device)
    return training_device


def move_to_device(tensor: torch.Tensor, device: torch.device) -> torch.Tensor:
    return tensor.to(device, non_blocking=device.type == "cuda")


def module_to_device(module: torch.nn.Module, device: torch.device) -> torch.nn.Module:
    return module.to(device)


def dataloader_worker_settings(
    device: torch.device,
    requested_num_workers: int,
) -> tuple[int, bool]:
    if device.type == "cpu":
        return 0, False

    num_workers = requested_num_workers
    return num_workers, num_workers > 0


def grad_scaler(device: torch.device, enabled: bool) -> torch.amp.GradScaler:
    return torch.amp.GradScaler(device.type, enabled=enabled)


def autocast(
    device: torch.device,
    enabled: bool,
    dtype: torch.dtype,
) -> torch.amp.autocast:
    return torch.amp.autocast(device.type, dtype=dtype, enabled=enabled)
