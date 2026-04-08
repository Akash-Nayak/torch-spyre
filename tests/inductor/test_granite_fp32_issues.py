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

"""Reproduction tests for Granite 3.3-8b issues identified in granite_issue.pdf.

Each test demonstrates a specific failure mode when running the Granite HF
model on Spyre hardware.  Tests are marked with the issue number from the PDF.

Usage:
    # Run all reproduction tests
    pytest tests/inductor/test_granite_fp32_issues.py -v

    # Run only issue 5 (mean fp32)
    pytest tests/inductor/test_granite_fp32_issues.py -v -k "issue5"

    # Run on real Spyre hardware (requires torch_spyre installed)
    pytest tests/inductor/test_granite_fp32_issues.py -v --device spyre
"""

import pytest
import torch

from utils_inductor import compare_with_cpu


# ---------------------------------------------------------------------------
# Issue 5 — mean on IEEE_FP32 not in SPYRE_FP32_OPS
# ---------------------------------------------------------------------------
# RMSNorm in Granite uses: variance = hidden_states.pow(2).mean(-1, keepdim=True)
# hidden_states is float32.  Spyre's mean lowering emits a "mean" op but
# "mean" is missing from SPYRE_FP32_OPS in constants.py, so create_op_spec
# raises Unsupported("mean on DataFormats.IEEE_FP32").
#
# Fix: add "mean" to SPYRE_FP32_OPS and verify the backend compiler handles it.
# Tracked in: torch-spyre/issues (granite_issue.pdf slide 5)


@pytest.mark.parametrize(
    "shape",
    [
        (1, 41, 4096),  # prefill, seq_len=41
        (1, 1, 4096),  # decode, seq_len=1
    ],
)
def test_issue5_mean_fp32(shape):
    """Issue 5: mean on float32 raises Unsupported in compiled path."""

    def fn(x):
        return x.pow(2).mean(-1, keepdim=True)

    compare_with_cpu(fn, torch.randn(*shape, dtype=torch.float32))


# ---------------------------------------------------------------------------
# Issue 11 — rsqrt on IEEE_FP32 not in SPYRE_FP32_OPS
# ---------------------------------------------------------------------------
# RMSNorm continues: hidden_states = hidden_states * torch.rsqrt(variance + eps)
# rsqrt receives a float32 tensor.  Same root cause as issue 5: "rsqrt" is
# missing from SPYRE_FP32_OPS.
#
# Fix: add "rsqrt" to SPYRE_FP32_OPS.
# Tracked in: github.com/torch-spyre/torch-spyre/issues/1368


@pytest.mark.parametrize(
    "shape",
    [
        (1, 41, 1),  # prefill, keepdim variance shape
        (1, 1, 1),  # decode
    ],
)
def test_issue11_rsqrt_fp32(shape):
    """Issue 11: rsqrt on float32 raises Unsupported in compiled path."""

    def fn(x):
        return torch.rsqrt(x + 1e-5)

    compare_with_cpu(fn, torch.randn(*shape, dtype=torch.float32).abs())


# ---------------------------------------------------------------------------
# Issue 12 — cat on IEEE_FP32 not supported
# ---------------------------------------------------------------------------
# RoPE embedding computation uses:
#   emb = torch.cat((freqs, freqs), dim=-1)  # freqs is float32
# cat falls back to eager on Spyre (no compiled-path lowering for cat on fp32).
# Tracked in: github.com/torch-spyre/torch-spyre/issues/1369


@pytest.mark.parametrize(
    "shape,dim",
    [
        ((1, 41, 64), -1),  # prefill: concat two halves of RoPE freqs
        ((1, 1, 64), -1),  # decode
        ((1, 41, 128), -2),  # rotate_half: torch.cat((-x2, x1), dim=-2)
    ],
)
def test_issue12_cat_fp32(shape, dim):
    """Issue 12: cat on float32 not supported in compiled path."""

    def fn(a, b):
        return torch.cat((a, b), dim=dim)

    x = torch.randn(*shape, dtype=torch.float32)
    compare_with_cpu(fn, x, x)


# ---------------------------------------------------------------------------
# Issue 2 — aten::all not supported on Spyre
# ---------------------------------------------------------------------------
# Granite HF checks:
#   if (position_ids == -1).all()
# aten::all is not lowered in lowering.py and is not present as a custom op.
# The op falls through to the fallback path and raises Unsupported.


@pytest.mark.parametrize(
    "shape",
    [
        (1, 41),  # prefill batch=1 seq_len=41
        (1, 1),  # decode
        (2, 41),  # batch=2
    ],
)
def test_issue2_all_not_supported(shape):
    """Issue 2: aten::all (bool reduction) not supported on Spyre."""

    def fn(position_ids):
        # Replicate the HF Granite pattern
        return (position_ids == -1).all()

    ids = torch.zeros(*shape, dtype=torch.long)
    compare_with_cpu(fn, ids)


# ---------------------------------------------------------------------------
# Issue 8 — SDPA with GQA shapes (mismatched q/k heads)
# ---------------------------------------------------------------------------
# Granite 3.3-8b uses GQA: 32 query heads, 8 KV heads (group size = 4).
# PyTorch >= 2.3 natively broadcasts K/V in SDPA when num_heads is a multiple
# of num_kv_heads.  Spyre's SDPA lowering (bmm-based) assumes q_heads ==
# kv_heads and does not implement grouped-query broadcast, causing a shape
# mismatch error at codegen time.
#
# Shapes are: Q=[B, 32, S_q, 128], K=V=[B, 8, S_k, 128]


@pytest.mark.parametrize(
    "q_shape,kv_shape,scale",
    [
        # prefill: S_q = S_k = 41
        ((1, 32, 41, 128), (1, 8, 41, 128), 0.0883883476),
        # decode (KV cache): S_q=1, S_k=2048
        ((1, 32, 1, 128), (1, 8, 2048, 128), 0.0883883476),
    ],
)
def test_issue8_sdpa_gqa(q_shape, kv_shape, scale):
    """Issue 8: SDPA with GQA shapes fails — Spyre does not support broadcast."""

    def fn(q, k, v):
        return torch.nn.functional.scaled_dot_product_attention(
            q, k, v, dropout_p=0.0, scale=scale, is_causal=False
        )

    q = torch.randn(*q_shape, dtype=torch.float16)
    k = torch.randn(*kv_shape, dtype=torch.float16)
    v = torch.randn(*kv_shape, dtype=torch.float16)
    compare_with_cpu(fn, q, k, v)
