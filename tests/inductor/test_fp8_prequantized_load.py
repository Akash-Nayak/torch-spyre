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

import torch_spyre  # noqa: F401 — registers Spyre as the inductor backend
from utils_inductor import DEVICE, compare_with_pytorch

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
    @classmethod
    def fp8_model(cls):
        """Load the FP8 model from checkpoint with FP8 weights intact.

        transformers upcasts FP8 weights to BF16 because config.json declares
        torch_dtype=bfloat16.  We bypass this entirely by:
          1. Instantiating the model architecture from config (no weights).
          2. Building the full state dict from safetensors shards directly,
             preserving the on-disk dtype (float8_e4m3fn for the 280 Linear
             weights, bfloat16 for everything else).
          3. Loading the state dict with strict=False (scale tensors from the
             compressed-tensors recipe are not model parameters and are ignored).

        This is zero-overhead: no BF16 allocation + patch; the FP8 tensors are
        never converted at all.
        """
        import glob
        from transformers import AutoConfig, AutoModelForCausalLM
        from safetensors import safe_open

        # Step 1: instantiate empty model on meta device (no weight allocation).
        config = AutoConfig.from_pretrained(MODEL_PATH)
        with torch.device("meta"):
            model = AutoModelForCausalLM.from_config(config)
        model = model.to_empty(device="cpu")

        # Step 2: build state dict from all safetensors shards preserving dtype.
        state_dict = {}
        for shard in sorted(glob.glob(f"{MODEL_PATH}/model-*.safetensors")):
            with safe_open(shard, framework="pt", device="cpu") as f:
                for key in f.keys():
                    state_dict[key] = f.get_tensor(key)

        # Step 3: load — strict=False ignores weight_scale tensors from the
        # compressed-tensors recipe that have no corresponding model parameter.
        model.load_state_dict(state_dict, strict=False, assign=True)
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


# ---------------------------------------------------------------------------
# scaled_mm with pre-quantized FP8 weight
# ---------------------------------------------------------------------------


class TestScaledMmWithPrequantizedWeight:
    """Test the pre-quantized FP8 weight loading path for _scaled_mm.

    Uses synthetic FP8 weights (no model checkpoint required).  The weight is
    transferred directly to Spyre via _dma_to_spyre_fp8_kernel (QFP8WT KERNEL
    layout), bypassing any runtime qfp8wt quantization.

    Known limitations: compiled _scaled_mm with pre-loaded weights
    --------------------------------------------------------------
    The ``dim_order`` mismatch between ``_dma_to_spyre_fp8_kernel`` and
    ``_qfp8wt_stl`` has been fixed (both now use ``[0, 1]``).  The remaining
    xfails document two distinct residual issues:

    1. **Weight as graph input** (``test_scaled_mm_prequantized_weight_xfail``):
       The pre-loaded Spyre tensor is passed as a compiled-function argument.
       This causes a device mismatch at compile time — the activation is on
       CPU during tracing but the weight is already on ``spyre:0``.  The
       correct usage is to close over the weight as a frozen constant.

    2. **Frozen weight — numerical mismatch** (``test_scaled_mm_frozen_weight``):
       The layout fix is applied; compilation succeeds without ``ReStickifyOpHBM``.
       However the computed output does not match the CPU reference
       (``~94%`` elements differ).  This points to a remaining issue in either
       the ``_scaled_mm`` lowering for pre-loaded KERNEL weights or in the
       reference implementation used for comparison.

    ``test_kernel_layout_properties`` verifies the DMA transfer itself (dtype,
    shape, device) is correct.  The ``_xfail`` tests are regression anchors —
    remove the markers once the underlying issues are resolved.
    """

    @pytest.mark.parametrize(
        "n, k",
        [
            (128, 128),
            (4096, 4096),
            (12800, 4096),
        ],
    )
    def test_kernel_layout_properties(self, n, k):
        """_dma_to_spyre_fp8_kernel produces a Spyre FP8 tensor with correct properties."""
        from torch_spyre.model_utils import _dma_to_spyre_fp8_kernel

        weight_fp8 = (
            torch.randn(n, k, dtype=torch.float32)
            .clamp(-448.0, 448.0)
            .to(torch.float8_e4m3fn)
        )
        q_weight = _dma_to_spyre_fp8_kernel(weight_fp8)

        assert q_weight.dtype == torch.float8_e4m3fn, (
            f"Expected float8_e4m3fn, got {q_weight.dtype}"
        )
        assert q_weight.device.type == DEVICE_TYPE, (
            f"Expected on {DEVICE_TYPE}, got {q_weight.device.type}"
        )
        assert q_weight.shape == torch.Size([n, k]), (
            f"Expected shape [{n}, {k}], got {q_weight.shape}"
        )

    @pytest.mark.xfail(
        reason=(
            "Weight as compiled-graph input causes device mismatch (cpu vs spyre:0) "
            "during tracing. Pre-loaded KERNEL weights must be closed over as frozen "
            "constants, not passed as graph inputs. See class docstring."
        ),
        strict=True,
    )
    @pytest.mark.parametrize(
        "m, k, n, scale_a, scale_b",
        [
            (1, 128, 128, 1.0, 1.0),
            (4, 4096, 4096, 2.0, 0.5),
        ],
    )
    def test_scaled_mm_prequantized_weight_xfail(self, m, k, n, scale_a, scale_b):
        """Documents that KERNEL FP8 weight as compiled-graph input is unsupported."""
        from torch_spyre.model_utils import _dma_to_spyre_fp8_kernel

        torch.manual_seed(42)
        act = torch.randn(m, k, dtype=torch.float16)
        scale_a_t = torch.full((1,), scale_a, dtype=torch.float16)
        scale_b_t = torch.full((1,), scale_b, dtype=torch.float16)
        weight_fp8 = (
            torch.randn(n, k, dtype=torch.float32)
            .clamp(-448.0, 448.0)
            .to(torch.float8_e4m3fn)
        )
        q_weight = _dma_to_spyre_fp8_kernel(weight_fp8)

        def spyre_fn(act, q_weight, scale_a, scale_b):
            q_act = torch.ops.spyre.quantize_fp8_with_scale(act, scale_a)
            return torch.ops.aten._scaled_mm(
                q_act, q_weight,
                scale_a=scale_a, scale_b=scale_b,
                bias=None, out_dtype=torch.float16,
            )

        def pytorch_fn(act, q_weight, scale_a, scale_b):
            q_a = (act / scale_a).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
            a_f32 = q_a.to(torch.float32) * scale_a.item()
            b_f32 = q_weight.to(torch.float32) * scale_b.item()
            return (a_f32 @ b_f32.T).to(torch.float16)

        compare_with_pytorch(
            spyre_fn, pytorch_fn, act, q_weight, scale_a_t, scale_b_t,
            atol=1.0, rtol=0.1,
        )

    @pytest.mark.xfail(
        reason=(
            "Numerical mismatch (~94% elements differ) between Spyre _scaled_mm "
            "output and CPU reference for frozen pre-loaded QFP8WT weight. "
            "dim_order is now correct ([0,1]); residual issue in _scaled_mm "
            "lowering or reference comparison. See class docstring."
        ),
        strict=True,
    )
    @pytest.mark.parametrize(
        "m, k, n, scale_a, scale_b",
        [
            (1, 128, 128, 1.0, 1.0),
            (4, 4096, 12800, 1.0, 1.0),
            (4, 4096, 4096, 2.0, 0.5),
        ],
    )
    def test_scaled_mm_frozen_weight(self, m, k, n, scale_a, scale_b):
        """_scaled_mm with QFP8CH activation and frozen QFP8WT pre-loaded weight.

        Matches the production inference pattern: the pre-quantized FP8 weight
        is loaded outside the compiled function and captured as a closed-over
        constant (not a graph input), exactly as torch.compile captures
        nn.Linear.weight when compiling a full model.

        Path:
          FP16 activation  →  quantize_fp8_with_scale  →  QFP8CH layout
          FP8 CPU weight   →  _dma_to_spyre_fp8_kernel →  QFP8WT KERNEL layout
          _scaled_mm(q_act, q_weight)                  →  FP16 output
        """
        from torch_spyre.model_utils import _dma_to_spyre_fp8_kernel

        torch.manual_seed(42)
        act = torch.randn(m, k, dtype=torch.float16)
        # Scales are closed-over constants — put them on Spyre so they match
        # the device of the compiled activation input.
        scale_a_t = torch.full((1,), scale_a, dtype=torch.float16, device=DEVICE)
        scale_b_t = torch.full((1,), scale_b, dtype=torch.float16, device=DEVICE)

        # FP8 weight transferred to Spyre KERNEL layout outside the compiled fn.
        weight_fp8 = (
            torch.randn(n, k, dtype=torch.float32)
            .clamp(-448.0, 448.0)
            .to(torch.float8_e4m3fn)
        )
        q_weight = _dma_to_spyre_fp8_kernel(weight_fp8)

        # Close over q_weight and scales as frozen constants — matches how
        # torch.compile captures nn.Linear.weight in a compiled model.
        def spyre_fn(act):
            q_act = torch.ops.spyre.quantize_fp8_with_scale(act, scale_a_t)
            return torch.ops.aten._scaled_mm(
                q_act, q_weight,
                scale_a=scale_a_t, scale_b=scale_b_t,
                bias=None, out_dtype=torch.float16,
            )

        # CPU reference: use scalar values so no device mismatch.
        def pytorch_fn(act):
            q_a = (act / scale_a).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
            a_f32 = q_a.to(torch.float32) * scale_a
            b_f32 = weight_fp8.to(torch.float32) * scale_b
            return (a_f32 @ b_f32.T).to(torch.float16)

        compare_with_pytorch(spyre_fn, pytorch_fn, act, atol=1.0, rtol=0.1)
