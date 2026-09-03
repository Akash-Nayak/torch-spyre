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

"""
Unit tests for FP8 quantization operations.

Tests cover:
- qfp8ch: Channel-wise FP8 format conversion
- fp8todl16: FP8→FP16 dtype conversion (tests .to(torch.float16) lowering)
- quantize/dequantize: Comprehensive roundtrip tests with various scales and input ranges
"""

import pytest
import torch

from torch_spyre._inductor.constants import FP8_E4M3FN_MAX, FP8_E4M3FN_MIN
from utils_inductor import (
    cached_randn,
    compare_with_pytorch,
)

# Maximum spacing between adjacent representable values in FP8 E4M3
FP8_E4M3_MAX_SPACING = 32.0


class TestFP8Operations:
    """Test suite for FP8 quantization operations not covered in test_inductor_ops.py."""

    def test_qfp8ch_basic_conversion(self):
        """Test basic FP16→FP8 format conversion with qfp8ch.

        Tests:
        - Basic conversion with shape [1, 2, 8]
        - Roundtrip: FP16 → FP8 → FP16 with scaling
        - Verifies qfp8ch operation is used internally

        Note: We use dequantize_fp8_with_scale for FP8→FP16 conversion
        because direct .to(torch.float16) cannot transfer to CPU.
        """
        x = cached_randn((1, 2, 8), scale=1.0, dtype=torch.float16)
        scale = torch.ones((1, 2, 1), dtype=torch.float16)

        def spyre_fn(x, scale):
            # Test qfp8ch format conversion directly (no pre-scaling)
            # Input x is already in valid FP8 range from cached_randn
            x_fp8 = torch.ops.spyre.qfp8ch(x)
            verify_fp8_dtype(x_fp8)
            # Dequantize with identity scale to verify format conversion
            return torch.ops.spyre.dequantize_fp8_with_scale(x_fp8, scale)

        def pytorch_fn(x, scale):
            # CPU reference: direct format conversion with identity scale
            x_fp8 = x.clamp(FP8_E4M3FN_MIN, FP8_E4M3FN_MAX).to(torch.float8_e4m3fn)
            return x_fp8.to(torch.float16) * scale

        compare_with_pytorch(
            spyre_fn,
            pytorch_fn,
            x,
            scale,
            atol=0.0,
            rtol=0.0,
        )

    def test_fp8todl16_basic_conversion(self):
        """Test FP8→FP16 dtype conversion with fp8todl16.

        Tests:
        - FP8→FP16 conversion using dequantize_fp8_with_scale with identity scale
        - Verifies fp8todl16 operation is triggered by the decomposition
        - Confirms output dtype is FP16
        - Tests the lowering path: x_fp8.to(torch.float16) * scale

        This test validates that the fp8todl16 deeptools operation is correctly
        invoked during dequantization. Note: Direct .to(torch.float16) without
        scaling cannot transfer to CPU, so we use identity scale (ones) to enable
        CPU transfer while still testing the fp8todl16 operation.
        """
        x = cached_randn((1, 2, 8), scale=1.0, dtype=torch.float16)
        scale = torch.ones((1, 2, 1), dtype=torch.float16)

        def spyre_fn(x, scale):
            # Convert FP16 → FP8 using qfp8ch
            x_fp8 = torch.ops.spyre.qfp8ch(x)
            verify_fp8_dtype(x_fp8)

            # Convert FP8 → FP16 using dequantize_fp8_with_scale with identity scale
            # This triggers fp8todl16 operation and allows CPU transfer
            x_fp8_fp16 = torch.ops.spyre.dequantize_fp8_with_scale(x_fp8, scale)
            verify_fp16_dtype(x_fp8_fp16)

            return x_fp8_fp16

        def pytorch_fn(x, scale):
            # CPU reference: FP16 → FP8 → FP16 conversion with identity scale
            x_fp8 = x.clamp(FP8_E4M3FN_MIN, FP8_E4M3FN_MAX).to(torch.float8_e4m3fn)
            return x_fp8.to(torch.float16) * scale

        compare_with_pytorch(
            spyre_fn,
            pytorch_fn,
            x,
            scale,
            atol=0.5,
            rtol=0.1,
        )

    # Tolerance categories:
    # - small_range: low FP8 spacing regions (atol=2)
    # - medium_range: may enter spacing=16 regions (atol=16)
    # - boundary_cases: may enter spacing=32 regions (atol=32 * scale)

    @pytest.mark.parametrize(
        "shape,scale_value,mean,std",
        [
            ((1, 2, 32), 0.01, 0.0, 1.0),
            ((1, 2, 32), 0.01, 0.0, 5.0),
            ((1, 2, 32), 0.1, 0.0, 1.0),
            ((1, 2, 32), 0.1, 0.0, 5.0),
            ((1, 2, 32), 0.5, 0.0, 1.0),
            ((1, 2, 32), 0.5, 0.0, 5.0),
            ((1, 2, 32), 1.0, 0.0, 1.0),
            ((1, 2, 32), 1.0, 0.0, 5.0),
            ((1, 2, 32), 2.0, 0.0, 1.0),
            ((1, 2, 32), 2.0, 0.0, 5.0),
        ],
    )
    def test_quantize_dequantize_fp8_small_range(
        self,
        shape,
        scale_value,
        mean,
        std,
    ):
        """Test quantize/dequantize for typical FP8 value ranges."""
        self._run_quantize_dequantize_fp8_test(
            shape,
            scale_value,
            mean,
            std,
            atol=2.0,
            rtol=0.0,
        )

    @pytest.mark.parametrize(
        "shape,scale_value,mean,std",
        [
            ((1, 2, 32), 0.01, 10.0, 50.0),
            ((1, 2, 32), 0.1, 10.0, 50.0),
            ((1, 2, 32), 0.5, 10.0, 50.0),
            ((1, 2, 32), 1.0, 10.0, 50.0),
            ((1, 2, 32), 2.0, 10.0, 50.0),
        ],
    )
    def test_quantize_dequantize_fp8_medium_range(
        self,
        shape,
        scale_value,
        mean,
        std,
    ):
        """Test quantize/dequantize for moderate input ranges.

        These cases may enter higher FP8 spacing regions but do not
        intentionally target FP8 representation boundaries.
        """
        self._run_quantize_dequantize_fp8_test(
            shape,
            scale_value,
            mean,
            std,
            atol=16.0,
            rtol=0.0,
        )

    @pytest.mark.parametrize(
        "shape,scale_value,mean,std",
        [
            ((1, 2, 32), 0.01, 100.0, 100.0),
            ((1, 2, 32), 0.01, 200.0, 200.0),
            ((1, 2, 32), 0.1, 100.0, 100.0),
            ((1, 2, 32), 0.1, 200.0, 200.0),
            ((1, 2, 32), 0.5, 100.0, 100.0),
            ((1, 2, 32), 0.5, 200.0, 200.0),
            ((1, 2, 32), 1.0, 100.0, 100.0),
            ((1, 2, 32), 1.0, 200.0, 200.0),
            ((1, 2, 32), 2.0, 100.0, 100.0),
            ((1, 2, 32), 2.0, 200.0, 200.0),
        ],
    )
    def test_quantize_dequantize_fp8_boundary_cases(
        self,
        shape,
        scale_value,
        mean,
        std,
    ):
        """Test FP8 E4M3 representation boundary cases."""
        self._run_quantize_dequantize_fp8_test(
            shape,
            scale_value,
            mean,
            std,
            atol=FP8_E4M3_MAX_SPACING * scale_value,
            rtol=0.0,
        )

    @pytest.mark.parametrize(
        "shape",
        [
            (1, 128, 512),
            (4, 128, 512),
            (1, 128, 1024),
            (1, 128, 2048),
            (1, 128, 4096),
        ],
    )
    def test_quantize_dequantize_fp8_production_shapes(self, shape):
        """Test FP8 quantize/dequantize with production-scale tensor shapes.

        Uses standard tolerance values (atol=0.5, rtol=0.1) with typical
        input distributions (mean=1.0, std=2.0, scale=1.0) that don't trigger
        edge cases in FP8 representation.
        """
        # Generate deterministic input with typical distribution
        x = cached_randn(shape, dtype=torch.float16, scale=1.0) * 2.0 + 1.0
        scale = torch.tensor([1.0], dtype=torch.float16)

        def spyre_fn(x, scale):
            x_fp8 = torch.ops.spyre.quantize_fp8_with_scale(x, scale)
            return torch.ops.spyre.dequantize_fp8_with_scale(x_fp8, scale)

        def pytorch_fn(x, scale):
            return (x / scale).clamp(FP8_E4M3FN_MIN, FP8_E4M3FN_MAX).to(
                torch.float8_e4m3fn
            ).to(torch.float16) * scale

        compare_with_pytorch(spyre_fn, pytorch_fn, x, scale, atol=0.5, rtol=0.1)

    def _run_quantize_dequantize_fp8_test(
        self,
        shape,
        scale_value,
        mean,
        std,
        atol,
        rtol=0.0,
    ):
        x = cached_randn(shape, dtype=torch.float16, scale=1.0) * std + mean
        scale = torch.tensor([scale_value], dtype=torch.float16)

        def spyre_fn(x, scale):
            x_fp8 = torch.ops.spyre.quantize_fp8_with_scale(x, scale)
            return torch.ops.spyre.dequantize_fp8_with_scale(x_fp8, scale)

        def pytorch_fn(x, scale):
            return (x / scale).clamp(FP8_E4M3FN_MIN, FP8_E4M3FN_MAX).to(
                torch.float8_e4m3fn
            ).to(torch.float16) * scale

        compare_with_pytorch(
            spyre_fn,
            pytorch_fn,
            x,
            scale,
            atol=atol,
            rtol=rtol,
        )


# Test utilities for FP8 operations
def verify_fp8_dtype(tensor):
    """Verify tensor has FP8 E4M3 dtype."""
    assert tensor.dtype == torch.float8_e4m3fn, (
        f"Expected dtype torch.float8_e4m3fn, got {tensor.dtype}"
    )


def verify_fp16_dtype(tensor):
    """Verify tensor has FP16 dtype."""
    assert tensor.dtype == torch.float16, (
        f"Expected dtype torch.float16, got {tensor.dtype}"
    )


class TestFP8IssuesVerification:
    """Regression tests for reported FP8 _scaled_mm compilation failures.

    Each test targets a specific shape that previously caused a SuperDSC abort.
    The full runtime-quantized graph is compiled:
        quantize_fp8_with_scale + quantize_weight_fp8_with_scale + _scaled_mm
    """

    # Granite 8B fused shapes that were aborting (issue #4179):
    #   - fused QKV:     N=6144  (4096 + 1024 + 1024), decode M=1
    #   - fused gate_up: N=25600 (2 × 12800),           decode M=1
    #   - prefill warmup M=16,  N=4096
    #   - bench prefill  M=64,  N=4096
    # Also include the M=1/2, N=4096 baselines that must keep working.
    @pytest.mark.parametrize(
        "m, k, n",
        [
            pytest.param(1, 4096, 4096, id="decode_m1_n4096_baseline"),
            pytest.param(2, 4096, 4096, id="decode_m2_n4096_baseline"),
            pytest.param(1, 4096, 6144, id="fused_qkv_n6144"),
            pytest.param(1, 4096, 25600, id="fused_gate_up_n25600"),
            pytest.param(4, 4096, 4096, id="wide_m4_n4096"),
            pytest.param(8, 4096, 4096, id="wide_m8_n4096"),
            pytest.param(16, 4096, 4096, id="prefill_warmup_m16_n4096"),
            pytest.param(64, 4096, 4096, id="bench_prefill_m64_n4096"),
            pytest.param(8, 4096, 1024, id="m8_n1024"),
            pytest.param(8, 4096, 12800, id="m8_n12800"),
        ],
    )
    def test_scaled_mm_granite_fp8_shapes(self, m, k, n):
        """Verify _scaled_mm compiles and produces correct output for Granite 8B FP8 shapes.

        Regression test for SuperDSC aborts on FP8 _scaled_mm at Granite
        fused N (6144 / 25600) and wide M=4/8/16/64.

        Two independent fixes are exercised here:
          Fix 1 (compute_ops.py): corrects the out/N-dim elemArr constants
            emitted by gen_coord_info_value for the QFP8WT KERNEL weight.
          Fix 2 (work_division.py): routes batchmatmulfp8 through the analytic
            cost model (_cost_model_divide_op), which was previously skipped for
            FP8 BMM ops. The cost model now produces a correct split for all N.
        """
        # Use controlled input range to avoid overflow in reference implementation
        # Scale inputs to stay within reasonable FP8 range
        x = cached_randn(
            (m, k), dtype=torch.float16, differentiation=("x", m, k, n), scale=0.5
        )
        w = cached_randn(
            (k, n), dtype=torch.float16, differentiation=("w", m, k, n), scale=0.5
        )
        # Use fixed small scales to avoid overflow: (xa @ wb) * (scale_a * scale_b)
        # With scale=0.1, the product is 0.01, keeping results in FP16 range.
        # float16 is intentional: quantize_fp8_with_scale and
        # quantize_weight_fp8_with_scale are documented to require float16 scales
        # (see customops.py — "scale: Quantization scale (FP16)").
        scale_a = torch.tensor([0.1], dtype=torch.float16)
        scale_b = torch.tensor([0.1], dtype=torch.float16)

        def spyre_fn(x, w, scale_a, scale_b):
            xa = torch.ops.spyre.quantize_fp8_with_scale(x, scale_a)
            wb = torch.ops.spyre.quantize_weight_fp8_with_scale(w, scale_b)
            return torch.ops.aten._scaled_mm(
                xa, wb, scale_a=scale_a, scale_b=scale_b, out_dtype=torch.float16
            )

        def pytorch_fn(x, w, scale_a, scale_b):
            xa = (
                (x / scale_a)
                .clamp(-448.0, 448.0)
                .to(torch.float8_e4m3fn)
                .to(torch.float16)
            )
            wb = (
                (w / scale_b)
                .clamp(-448.0, 448.0)
                .to(torch.float8_e4m3fn)
                .to(torch.float16)
            )
            return (xa @ wb) * (scale_a * scale_b)

        compare_with_pytorch(
            spyre_fn,
            pytorch_fn,
            x,
            w,
            scale_a,
            scale_b,
            atol=2.0,
            rtol=0.2,
        )
