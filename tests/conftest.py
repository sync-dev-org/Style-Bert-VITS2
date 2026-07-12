from __future__ import annotations

import os
import platform
import tempfile
from dataclasses import dataclass
from importlib.util import find_spec

import pytest


if "NUMBA_CACHE_DIR" not in os.environ:
    os.environ.setdefault(
        "NUMBA_CACHE_DIR", tempfile.mkdtemp(prefix="style-bert-vits2-numba-cache-")
    )


@dataclass(frozen=True)
class EnvironmentCapabilities:
    available_onnx_providers: tuple[str, ...]
    cuda_available: bool
    platform_system: str


def _package_available(package_name: str) -> bool:
    return find_spec(package_name) is not None


def _available_onnx_providers() -> tuple[str, ...]:
    try:
        import onnxruntime
    except Exception:
        return ()

    return tuple(onnxruntime.get_available_providers())


def _cuda_available() -> bool:
    if not _package_available("torch"):
        return False

    try:
        import torch
    except Exception:
        return False

    return bool(torch.cuda.is_available())


def detect_environment_capabilities() -> EnvironmentCapabilities:
    return EnvironmentCapabilities(
        available_onnx_providers=_available_onnx_providers(),
        cuda_available=_cuda_available(),
        platform_system=platform.system(),
    )


ENVIRONMENT_CAPABILITIES = detect_environment_capabilities()


@pytest.fixture(scope="session")
def environment_capabilities() -> EnvironmentCapabilities:
    return ENVIRONMENT_CAPABILITIES


def requires_python_package(package_name: str, purpose: str):
    return pytest.mark.skipif(
        not _package_available(package_name),
        reason=f"requires optional dependency {package_name}: {purpose}",
    )


def requires_cuda():
    return pytest.mark.skipif(
        not ENVIRONMENT_CAPABILITIES.cuda_available,
        reason=(
            "requires CUDA; torch.cuda.is_available() is False "
            f"on {ENVIRONMENT_CAPABILITIES.platform_system}"
        ),
    )


def requires_onnx_provider(provider_name: str):
    available = ENVIRONMENT_CAPABILITIES.available_onnx_providers
    available_text = ", ".join(available) if available else "none"
    return pytest.mark.skipif(
        provider_name not in available,
        reason=(
            f"requires ONNX ExecutionProvider {provider_name}; "
            f"available providers: {available_text}"
        ),
    )
