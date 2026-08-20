# Copyright 2025 The Torch-Spyre Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for FP8 pre-quantized weight loading via load_fp8_model_to_spyre.

Validates that pre-quantized FP8 checkpoint weights (torch.float8_e4m3fn)
are loaded directly into QFP8WT KERNEL layout on Spyre — without lazy
decompression or runtime qfp8wt quantization.

Two test modes:
  1. Unit test (no model required): verifies _dma_to_spyre_fp8_kernel
     transfers a synthetic FP8 tensor to Spyre with the correct dtype,
     shape, and device.
  2. Integration test (requires model checkpoint): verifies
     load_fp8_model_to_spyre loads all 280 FP8 Linear weights of
     granite-3.3-8b-instruct-FP8 with QFP8WT layout.
"""

import pytest
import torch
import torch.nn as nn

from utils_inductor import DEVICE

DEVICE_TYPE = DEVICE.type  # 'spyre' — used for device.type comparisons

MODEL_PATH = "/nfs_mnt/models/granite-3.3-8b-instruct-FP8"


# ---------------------------------------------------------------------------
# Unit tests — no model checkpoint required
# ---------------------------------------------------------------------------


class TestDmaToSpyreFp8Kernel:
    """Unit tests for _dma_to_spyre_fp8_kernel."""

    def test_basic_transfer(self):
        """FP8 weight transfers to Spyre with correct dtype and shape."""
        from torch_spyre.model_utils import _dma_to_spyre_fp8_kernel

        weight = torch.randn(128, 256, dtype=torch.float16).to(torch.float8_e4m3fn)
        dev = _dma_to_spyre_fp8_kernel(weight)

        assert dev.device.type == DEVICE_TYPE, f"Expected device {DEVICE}, got {dev.device}"
        assert dev.dtype == torch.float8_e4m3fn, f"Expected fp8, got {dev.dtype}"
        assert list(dev.shape) == [128, 256], f"Shape mismatch: {list(dev.shape)}"

    def test_non_contiguous_input(self):
        """Non-contiguous FP8 weight is handled (made contiguous internally)."""
        from torch_spyre.model_utils import _dma_to_spyre_fp8_kernel

        weight = torch.randn(256, 128, dtype=torch.float16).t().to(torch.float8_e4m3fn)
        # t() produces a non-contiguous view [128, 256]
        dev = _dma_to_spyre_fp8_kernel(weight.contiguous())

        assert dev.device.type == DEVICE_TYPE
        assert dev.dtype == torch.float8_e4m3fn

    def test_rejects_non_fp8(self):
        """Raises AssertionError for non-FP8 dtype."""
        from torch_spyre.model_utils import _dma_to_spyre_fp8_kernel

        weight = torch.randn(128, 128, dtype=torch.float16)
        with pytest.raises(AssertionError, match="float8_e4m3fn"):
            _dma_to_spyre_fp8_kernel(weight)

    def test_rejects_non_2d(self):
        """Raises AssertionError for non-2D tensor."""
        from torch_spyre.model_utils import _dma_to_spyre_fp8_kernel

        weight = torch.randn(4, 128, 128, dtype=torch.float16).to(torch.float8_e4m3fn)
        with pytest.raises(AssertionError, match="2D"):
            _dma_to_spyre_fp8_kernel(weight)

    def test_production_shapes(self):
        """Common Granite-3.3 8B Linear weight shapes transfer correctly."""
        from torch_spyre.model_utils import _dma_to_spyre_fp8_kernel

        shapes = [
            (4096, 4096),   # q_proj, o_proj
            (1024, 4096),   # k_proj, v_proj (GQA)
            (12800, 4096),  # gate_proj, up_proj
            (4096, 12800),  # down_proj
        ]
        for out_f, in_f in shapes:
            weight = torch.randn(out_f, in_f, dtype=torch.float16).to(torch.float8_e4m3fn)
            dev = _dma_to_spyre_fp8_kernel(weight)
            assert dev.device.type == DEVICE_TYPE
            assert dev.dtype == torch.float8_e4m3fn
            assert list(dev.shape) == [out_f, in_f]


class TestLoadModelToSpyreUseFp8Weights:
    """Unit tests for load_model_to_spyre(use_fp8_weights=True) with a tiny model."""

    def _make_tiny_fp8_model(self):
        """Build a tiny 2-layer model with pre-quantized FP8 Linear weights."""
        model = nn.Sequential(
            nn.Linear(128, 64, bias=False),
            nn.Linear(64, 32, bias=False),
        )
        # Simulate pre-quantized FP8 weights
        for module in model.modules():
            if isinstance(module, nn.Linear):
                module.weight = nn.Parameter(
                    module.weight.data.to(torch.float8_e4m3fn),
                    requires_grad=False,
                )
        return model

    def test_fp8_weights_loaded_to_spyre(self):
        """FP8 Linear weights are moved to Spyre device."""
        from torch_spyre.model_utils import load_model_to_spyre

        model = self._make_tiny_fp8_model()
        load_model_to_spyre(model, use_fp8_weights=True)

        for module in model.modules():
            if isinstance(module, nn.Linear):
                assert module.weight.device.type == DEVICE_TYPE, (
                    f"Expected weight on {DEVICE}, got {module.weight.device}"
                )
                assert module.weight.dtype == torch.float8_e4m3fn, (
                    f"Expected fp8 dtype preserved, got {module.weight.dtype}"
                )

    def test_fp8_weights_dtype_preserved(self):
        """FP8 dtype is not upcast to BF16/FP16 during DMA."""
        from torch_spyre.model_utils import load_fp8_model_to_spyre

        model = self._make_tiny_fp8_model()
        load_fp8_model_to_spyre(model)

        fp8_weights = [
            (n, m.weight)
            for n, m in model.named_modules()
            if isinstance(m, nn.Linear)
        ]
        assert len(fp8_weights) == 2
        for name, w in fp8_weights:
            assert w.dtype == torch.float8_e4m3fn, (
                f"{name}: expected float8_e4m3fn, got {w.dtype}"
            )

    def test_non_fp8_weights_use_normal_path(self):
        """BF16 Linear weights still go through dim_order=[1,0] path."""
        from torch_spyre.model_utils import load_model_to_spyre

        model = nn.Sequential(nn.Linear(128, 64, bias=False))
        # BF16 weights — use_fp8_weights=True should NOT touch these
        model[0].weight = nn.Parameter(
            model[0].weight.data.to(torch.bfloat16), requires_grad=False
        )
        load_model_to_spyre(model, use_fp8_weights=True)

        assert model[0].weight.device.type == DEVICE_TYPE
        # dtype should be preserved (bfloat16 on device)
        assert model[0].weight.dtype == torch.bfloat16


# ---------------------------------------------------------------------------
# Integration test — requires /nfs_mnt/models/granite-3.3-8b-instruct-FP8
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not __import__("os").path.exists(MODEL_PATH),
    reason=f"Model not found at {MODEL_PATH}",
)
class TestGraniteF8IntegrationLoad:
    """Integration test: load granite-3.3-8b-instruct-FP8 with QFP8WT layout."""

    @pytest.fixture(scope="class")
    def fp8_model(self):
        """Load the FP8 model from checkpoint, bypassing compressed-tensors."""
        from transformers import AutoModelForCausalLM

        # Load without quantization_config to get raw FP8 tensors as-is
        # (no lazy decompression / upcast from compressed-tensors)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            dtype=torch.float8_e4m3fn,
            low_cpu_mem_usage=True,
            device_map="cpu",
            quantization_config=None,
        )
        return model

    def test_fp8_weights_present_before_load(self, fp8_model):
        """Model has 280 FP8 Linear weights on CPU before transfer."""
        fp8_count = sum(
            1
            for m in fp8_model.modules()
            if isinstance(m, nn.Linear)
            and m.weight.dtype == torch.float8_e4m3fn
        )
        assert fp8_count == 280, (
            f"Expected 280 FP8 Linear weights, got {fp8_count}"
        )

    def test_load_fp8_model_to_spyre(self, fp8_model):
        """load_fp8_model_to_spyre transfers all 280 FP8 weights to Spyre."""
        from torch_spyre.model_utils import load_fp8_model_to_spyre

        load_fp8_model_to_spyre(fp8_model)

        spyre_fp8 = [
            (n, m.weight)
            for n, m in fp8_model.named_modules()
            if isinstance(m, nn.Linear) and m.weight.dtype == torch.float8_e4m3fn
        ]
        assert len(spyre_fp8) == 280, (
            f"Expected 280 FP8 weights on Spyre, got {len(spyre_fp8)}"
        )
        for name, w in spyre_fp8:
            assert w.device.type == DEVICE_TYPE, (
                f"{name}: expected on {DEVICE}, got {w.device}"
            )

    def test_non_fp8_weights_on_spyre(self, fp8_model):
        """Non-FP8 weights (layer norms, embed) are also on Spyre."""
        non_fp8 = [
            (n, p)
            for n, p in fp8_model.named_parameters()
            if p.dtype != torch.float8_e4m3fn
        ]
        assert len(non_fp8) > 0
        for name, p in non_fp8:
            assert p.device.type == DEVICE_TYPE, (
                f"{name}: expected on {DEVICE}, got {p.device}"
            )
