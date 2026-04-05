from __future__ import annotations

from importlib import import_module
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, TypeAlias, cast

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    from video_intelligence_agent.models import BoundingBox

ImageArray: TypeAlias = npt.NDArray[np.uint8]
GenericImageArray: TypeAlias = npt.NDArray[Any]


class OpenCVModuleProtocol(Protocol):
    def imread(self, filename: str) -> ImageArray | None: ...

    def imwrite(self, filename: str, img: ImageArray) -> bool: ...


class PillowImageProtocol(Protocol):
    def convert(self, mode: str) -> "PillowImageProtocol": ...

    def save(self, fp: str | Path) -> None: ...


class PillowModuleProtocol(Protocol):
    def open(self, fp: str | Path) -> PillowImageProtocol: ...

    def fromarray(self, obj: ImageArray) -> PillowImageProtocol: ...


def _load_cv2() -> OpenCVModuleProtocol | None:
    try:
        module = import_module("cv2")
    except ImportError:  # pragma: no cover - depends on local environment
        return None
    return cast(OpenCVModuleProtocol, module)


def _load_pil_image() -> PillowModuleProtocol | None:
    try:
        module = import_module("PIL.Image")
    except ImportError:  # pragma: no cover - depends on local environment
        return None
    return cast(PillowModuleProtocol, module)


def ensure_uint8(image: GenericImageArray) -> ImageArray:
    if image.dtype == np.uint8:
        return image.astype(np.uint8, copy=False)

    clipped = np.clip(image, 0, 255)
    max_value = float(np.max(clipped)) if clipped.size else 0.0
    if max_value <= 1.0:
        clipped = clipped * 255
    return clipped.astype(np.uint8)


def crop_image(image: GenericImageArray, bbox: BoundingBox) -> GenericImageArray:
    x0 = max(bbox.x, 0)
    y0 = max(bbox.y, 0)
    x1 = max(x0 + bbox.w, x0)
    y1 = max(y0 + bbox.h, y0)
    return image[y0:y1, x0:x1].copy()


def load_image(path: str | Path) -> ImageArray:
    image_path = Path(path)
    cv2_module = _load_cv2()
    if cv2_module is not None:
        image = cv2_module.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(f"Unable to load image: {image_path}")
        return image

    pil_image_module = _load_pil_image()
    if pil_image_module is not None:
        rgb = np.asarray(pil_image_module.open(image_path).convert("RGB"), dtype=np.uint8)
        bgr = rgb[:, :, ::-1].copy()
        return bgr

    raise RuntimeError("opencv-python or Pillow is required to load images.")


def save_image(path: str | Path, image: GenericImageArray) -> None:
    image_path = Path(path)
    image_path.parent.mkdir(parents=True, exist_ok=True)
    payload = ensure_uint8(image)

    cv2_module = _load_cv2()
    if cv2_module is not None:
        if not cv2_module.imwrite(str(image_path), payload):
            raise RuntimeError(f"Unable to save image: {image_path}")
        return

    pil_image_module = _load_pil_image()
    if pil_image_module is not None:
        if payload.ndim == 3 and payload.shape[2] == 3:
            payload = payload[:, :, ::-1]
        pil_image_module.fromarray(payload).save(image_path)
        return

    raise RuntimeError("opencv-python or Pillow is required to save images.")

