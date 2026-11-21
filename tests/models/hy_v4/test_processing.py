# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from copy import deepcopy
from types import SimpleNamespace

import pytest

from vllm.models.hy_v4.nvidia.mm_preprocess import HYV4VLMultiModalProcessor


@pytest.mark.parametrize("on_device", [True, False])
def test_scoped_pixel_limits_keep_normalization_on_the_requested_device(on_device):
    # Transformers uses the nested modality scope whenever pixel limits are set.
    # The image and video paths both need to preserve raw bytes for a GPU norm.
    original = {
        "images_kwargs": {
            "max_pixels": 65536,
            "do_normalize": True,
            "do_rescale": True,
        },
        "videos_kwargs": {"max_frames": 4, "do_normalize": True, "do_rescale": True},
    }
    saved = deepcopy(original)
    context = SimpleNamespace(
        model_config=SimpleNamespace(
            get_multimodal_config=lambda: SimpleNamespace(
                mm_device_do_normalize=on_device
            )
        ),
        call_hf_processor=lambda processor, data, kwargs: kwargs,
    )
    proc = object.__new__(HYV4VLMultiModalProcessor)
    proc.info = SimpleNamespace(
        ctx=context,
        get_hf_processor=lambda **kwargs: None,
        _merge_and_resolve_mm_processor_kwargs=lambda options: dict(options),
    )
    received = proc._call_hf_processor({}, original)
    assert received["images_kwargs"]["max_pixels"] == 65536
    assert received["videos_kwargs"]["max_frames"] == 4
    for scope in received.values():
        assert scope["do_normalize"] is (not on_device)
        assert scope["do_rescale"] is (not on_device)
    assert original == saved
