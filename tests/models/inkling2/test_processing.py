# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU-only checks of the Inkling2 image processor, including the longest-side cap."""

import io
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

from vllm.config.multimodal import MultiModalConfig
from vllm.models.inkling2 import processing
from vllm.models.inkling2.processing import (
    MAX_IMAGE_SIDE_KWARG,
    PATCH_SIZE,
    Inkling2ImageProcessor,
)
from vllm.multimodal.processing.context import (
    InputProcessingContext,
    _resolve_mm_processor_kwargs,
)

CAP = 2048
# A capped image is at most this many cells per side (2048 / 40, rounded up).
CAP_CELLS = 52


def _png(width: int, height: int, color: tuple[int, int, int] = (220, 30, 30)) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (width, height), color).save(buffer, format="PNG")
    return buffer.getvalue()


def _grids(out: dict[str, torch.Tensor]) -> list[tuple[int, int, int]]:
    return [(int(t), int(h), int(w)) for t, h, w in out["vision_grid_thw"]]


def _processor_with_engine_setting(
    mm_processor_kwargs: dict[str, object] | None,
) -> processing.Inkling2MultiModalProcessor:
    """A processor whose engine was started with ``mm_processor_kwargs``."""
    config = MultiModalConfig(
        limit_per_prompt={"image": 1}, mm_processor_kwargs=mm_processor_kwargs
    )
    ctx = InputProcessingContext.__new__(InputProcessingContext)
    # A frozen dataclass: set the one field the merge reads directly.
    object.__setattr__(
        ctx,
        "model_config",
        SimpleNamespace(get_multimodal_config=lambda: config),
    )
    info = processing.Inkling2ProcessingInfo.__new__(processing.Inkling2ProcessingInfo)
    info.ctx = ctx
    processor = processing.Inkling2MultiModalProcessor.__new__(
        processing.Inkling2MultiModalProcessor
    )
    processor.info = info
    return processor


def test_image_within_the_cap_is_untouched() -> None:
    image = np.full((1000, 2048, 3), 90, dtype=np.uint8)
    plain = Inkling2ImageProcessor().preprocess([image])
    capped = Inkling2ImageProcessor().preprocess([image], max_side=CAP)

    assert _grids(capped) == [(1, 25, CAP_CELLS)]
    assert torch.equal(capped["vision_pixels"], plain["vision_pixels"])
    assert torch.equal(capped["vision_grid_thw"], plain["vision_grid_thw"])


@pytest.mark.parametrize(
    ("width", "height", "rows", "cols"),
    [
        (4200, 4200, 52, 52),
        (6000, 3000, 26, 52),
        (1000, 5000, 52, 11),
        (2048, 2048, 52, 52),
    ],
)
def test_longer_images_are_scaled_to_the_cap(
    width: int, height: int, rows: int, cols: int
) -> None:
    out = Inkling2ImageProcessor().preprocess([_png(width, height)], max_side=CAP)

    assert _grids(out) == [(1, rows, cols)]
    assert out["vision_pixels"].shape == (rows * cols, PATCH_SIZE, PATCH_SIZE, 3)
    assert out["vision_pixels"].dtype == torch.float16


def test_cap_bounds_every_image_to_the_encoder_budget() -> None:
    out = Inkling2ImageProcessor().preprocess([_png(9000, 7000)], max_side=CAP)
    _, rows, cols = _grids(out)[0]

    assert rows <= CAP_CELLS and cols <= CAP_CELLS
    assert rows * cols <= 2704


def test_scaled_image_keeps_its_aspect_ratio_and_content() -> None:
    image = np.zeros((3000, 6000, 3), dtype=np.uint8)
    image[:, :3000] = (220, 30, 30)
    image[:, 3000:] = (30, 30, 220)
    out = Inkling2ImageProcessor().preprocess([image], max_side=CAP)
    _, rows, cols = _grids(out)[0]

    assert (rows, cols) == (26, 52)
    assert abs(rows / cols - 0.5) < 0.03
    cells = out["vision_pixels"].reshape(rows, cols, PATCH_SIZE, PATCH_SIZE, 3)
    left = cells[0, 0].reshape(-1, 3).float().mean(0)
    right = cells[0, cols - 2].reshape(-1, 3).float().mean(0)
    assert left[0] > left[2]
    assert right[2] > right[0]


def test_without_a_cap_images_keep_their_size() -> None:
    # 75 x 75 = 5,625 tokens: the size the server refused before the cap.
    out = Inkling2ImageProcessor().preprocess([_png(3000, 3000)])

    assert _grids(out) == [(1, 75, 75)]


@pytest.mark.parametrize("max_side", [0, 39, 2048.0, True, "2048"])
def test_invalid_caps_are_rejected(max_side: object) -> None:
    with pytest.raises(ValueError, match="max_side"):
        Inkling2ImageProcessor().preprocess([_png(100, 100)], max_side=max_side)


def test_cap_is_a_declared_processor_kwarg() -> None:
    info = processing.Inkling2ProcessingInfo.__new__(processing.Inkling2ProcessingInfo)

    supported = info.get_supported_mm_processor_kwargs()

    assert _resolve_mm_processor_kwargs({MAX_IMAGE_SIDE_KWARG: CAP}, supported) == {
        "images_kwargs": {MAX_IMAGE_SIDE_KWARG: CAP}
    }


def test_cap_comes_from_the_engine_setting_unless_a_request_overrides_it() -> None:
    processor = _processor_with_engine_setting({MAX_IMAGE_SIDE_KWARG: CAP})

    assert processor._max_image_side({}) == CAP
    assert processor._max_image_side({MAX_IMAGE_SIDE_KWARG: 1024}) == 1024
    assert processor._max_image_side({"images_kwargs": {"max_side": 1024}}) == 1024


def test_without_an_engine_setting_images_keep_their_size() -> None:
    processor = _processor_with_engine_setting(None)

    assert processor._max_image_side({}) is None
    assert processor._max_image_side({MAX_IMAGE_SIDE_KWARG: 1024}) == 1024


def test_non_integer_engine_setting_is_rejected() -> None:
    processor = _processor_with_engine_setting({MAX_IMAGE_SIDE_KWARG: "2048"})

    with pytest.raises(ValueError, match="max_side"):
        processor._max_image_side({})
