import torch

from style_bert_vits2.models.training import (
    autocast,
    dataloader_worker_settings,
    grad_scaler,
    module_to_device,
    move_to_device,
    resolve_training_device,
)


def test_resolve_training_device_uses_cpu_when_cuda_is_unavailable(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    device_config = resolve_training_device(local_rank=2)

    assert device_config.device == torch.device("cpu")
    assert device_config.backend == "gloo"
    assert device_config.ddp_device_ids is None
    assert device_config.pin_memory is False


def test_resolve_training_device_uses_cuda_rank_when_cuda_is_available(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_nccl_available", lambda: True)

    device_config = resolve_training_device(local_rank=1, system_name="Linux")

    assert device_config.device == torch.device("cuda", 1)
    assert device_config.backend == "nccl"
    assert device_config.ddp_device_ids == [1]
    assert device_config.pin_memory is True


def test_move_to_device_uses_selected_torch_device():
    tensor = torch.tensor([1.0])

    moved = move_to_device(tensor, torch.device("cpu"))

    assert moved.device == torch.device("cpu")
    assert torch.equal(moved, tensor)


def test_module_to_device_uses_selected_torch_device():
    module = torch.nn.Linear(1, 1)

    moved = module_to_device(module, torch.device("cpu"))

    assert next(moved.parameters()).device == torch.device("cpu")


def test_dataloader_worker_settings_disable_workers_on_cpu():
    num_workers, persistent_workers = dataloader_worker_settings(
        torch.device("cpu"),
        requested_num_workers=1,
    )

    assert num_workers == 0
    assert persistent_workers is False


def test_dataloader_worker_settings_keep_cuda_workers():
    num_workers, persistent_workers = dataloader_worker_settings(
        torch.device("cuda", 0),
        requested_num_workers=2,
    )

    assert num_workers == 2
    assert persistent_workers is True


def test_amp_helpers_use_selected_device_type(monkeypatch):
    scaler_calls = []
    autocast_calls = []

    class FakeGradScaler:
        def __init__(self, device_type, enabled):
            scaler_calls.append((device_type, enabled))

    class FakeAutocast:
        def __init__(self, device_type, dtype, enabled):
            autocast_calls.append((device_type, dtype, enabled))

    monkeypatch.setattr(torch.amp, "GradScaler", FakeGradScaler)
    monkeypatch.setattr(torch.amp, "autocast", FakeAutocast)

    scaler = grad_scaler(torch.device("cpu"), enabled=True)
    context = autocast(torch.device("cpu"), enabled=True, dtype=torch.bfloat16)

    assert isinstance(scaler, FakeGradScaler)
    assert isinstance(context, FakeAutocast)
    assert scaler_calls == [("cpu", True)]
    assert autocast_calls == [("cpu", torch.bfloat16, True)]
