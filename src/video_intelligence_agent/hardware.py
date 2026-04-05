from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from typing import Any, Protocol, cast


class TensorFlowExperimentalConfigProtocol(Protocol):
    def set_memory_growth(self, device: object, enable: bool) -> None: ...


class TensorFlowConfigProtocol(Protocol):
    experimental: TensorFlowExperimentalConfigProtocol

    def list_physical_devices(self, device_type: str) -> list[object]: ...


class TensorFlowModuleProtocol(Protocol):
    config: TensorFlowConfigProtocol


@dataclass(slots=True)
class HardwareProfile:
    torch_cuda_available: bool = False
    tensorflow_gpu_available: bool = False
    tensorflow_gpu_count: int = 0

    @property
    def preferred_torch_device(self) -> str:
        return "cuda" if self.torch_cuda_available else "cpu"


def detect_hardware() -> HardwareProfile:
    torch_cuda_available = False
    tensorflow_gpu_available = False
    tensorflow_gpu_count = 0

    try:
        torch_module = import_module("torch")
    except ImportError:
        torch_module = None
    if torch_module is not None:
        cuda = getattr(torch_module, "cuda", None)
        is_available = getattr(cuda, "is_available", None)
        if callable(is_available):
            try:
                torch_cuda_available = bool(is_available())
            except Exception:
                torch_cuda_available = False

    try:
        tensorflow_module = import_module("tensorflow")
    except ImportError:
        tensorflow_module = None
    if tensorflow_module is not None:
        config = cast(
            TensorFlowConfigProtocol | None,
            getattr(cast(TensorFlowModuleProtocol, tensorflow_module), "config", None),
        )
        list_physical_devices = getattr(config, "list_physical_devices", None)
        if callable(list_physical_devices):
            try:
                gpus = cast(list[object], list_physical_devices("GPU"))
            except Exception:
                gpus = []
            tensorflow_gpu_count = len(gpus)
            tensorflow_gpu_available = tensorflow_gpu_count > 0

    return HardwareProfile(
        torch_cuda_available=torch_cuda_available,
        tensorflow_gpu_available=tensorflow_gpu_available,
        tensorflow_gpu_count=tensorflow_gpu_count,
    )


def configure_tensorflow_runtime(
    *,
    prefer_gpu: bool = True,
    enable_memory_growth: bool = True,
) -> HardwareProfile:
    profile = detect_hardware()
    if not prefer_gpu or not enable_memory_growth or not profile.tensorflow_gpu_available:
        return profile

    try:
        tensorflow_module = import_module("tensorflow")
    except ImportError:
        return profile

    config = cast(
        TensorFlowConfigProtocol | None,
        getattr(cast(TensorFlowModuleProtocol, tensorflow_module), "config", None),
    )
    experimental = getattr(config, "experimental", None)
    if experimental is None:
        return profile

    list_physical_devices = getattr(config, "list_physical_devices", None)
    set_memory_growth = getattr(experimental, "set_memory_growth", None)
    if not callable(list_physical_devices) or not callable(set_memory_growth):
        return profile

    try:
        for gpu in cast(list[object], list_physical_devices("GPU")):
            set_memory_growth(gpu, True)
    except Exception:
        return profile

    return profile
