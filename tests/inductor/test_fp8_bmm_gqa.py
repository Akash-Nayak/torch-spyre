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
Tests for GQA broadcast support in spyre.scaled_bmm / spyre.bmm_fp8.

GQA (Grouped Query Attention) allows mat2 (K cache) to have fewer heads than
mat1 (Q), with a kv_repeat factor bridging the gap.  lower_bmm_fp8 emits
FloorDiv(h, kv_repeat) for the mat2 batch index, which maps to a zero
mb-stride KERNEL SDSC — hardware natively broadcasts the same K block across
all Q heads in each group.

Two test classes:

1. ``TestFp8BmmGQABroadcast`` — unit test for ``spyre.scaled_bmm`` GQA broadcast
   with two geometries (gqa2_small, gqa4_granite).

2. ``TestFp8KvCacheQFP8WTDecodeEndToEndGQA`` — full fill+decode pipeline with
   GQA broadcast: K cache [NUM_BLOCKS, n_kv, D, BS], Q [1, H, 1, D] with
   H = n_kv * gqa.
"""

import pytest
import torch
from utils_inductor import DEVICE, cached_randn

from torch_spyre._inductor.constants import FP8_E4M3FN_MAX, FP8_E4M3FN_MIN

# ──────────────────────────────────────────────────────────────────────────────
# Shared geometry constants
# ──────────────────────────────────────────────────────────────────────────────
NUM_BLOCKS = 4
BLOCK_SIZE = 128
HEAD_SIZE = 128
NUM_KV_HEADS = 2
NUM_HEADS = 4  # GQA: 2 query groups per KV head


# ──────────────────────────────────────────────────────────────────────────────
# CPU reference helpers
# ──────────────────────────────────────────────────────────────────────────────


def _cpu_quantize(x_fp16: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return (
        (x_fp16 / scale).clamp(FP8_E4M3FN_MIN, FP8_E4M3FN_MAX).to(torch.float8_e4m3fn)
    )


def _cpu_dequantize(x_fp8: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x_fp8.to(torch.float16) * scale


def _cpu_per_token_scale(x: torch.Tensor) -> torch.Tensor:
    return torch.amax(torch.abs(x), dim=-1, keepdim=True) / FP8_E4M3FN_MAX


# ──────────────────────────────────────────────────────────────────────────────
# 1. GQA broadcast in scaled_bmm — mat1 [B,H,...] × mat2 [B,n_kv,...] with H=n_kv*gqa
# ──────────────────────────────────────────────────────────────────────────────


class TestFp8BmmGQABroadcast:
    """GQA broadcast in ``spyre.scaled_bmm``: mat2 batch dim < mat1 batch dim.

    Tests the ``lower_bmm_fp8`` GQA extension where mat2 (K, QFP8WT) has
    ``n_kv`` heads while mat1 (Q, QFP8CH) has ``H = n_kv * gqa`` heads.  The
    lowering emits ``FloorDiv(h, gqa)`` for the mat2 head index, mapping each
    group of ``gqa`` Q heads to the same K head.  The hardware executes this
    as a zero mb-stride KERNEL broadcast.

    Two geometries are tested:

    ``gqa2_small``
        ``NUM_KV_HEADS=2, gqa=2, H=4`` — uses the module-level constants.

    ``gqa4_granite``
        ``n_kv=8, gqa=4, H=32, D=128, BS=128`` — Granite-3B attention geometry.
        Confirms the fix works at the scale of a real model.
    """

    @pytest.mark.parametrize(
        "geometry",
        [
            # (n_kv, gqa, head_size, block_size, label)
            (
                NUM_KV_HEADS,
                NUM_HEADS // NUM_KV_HEADS,
                HEAD_SIZE,
                BLOCK_SIZE,
                "gqa2_small",
            ),
            (8, 4, 128, 128, "gqa4_granite"),
        ],
    )
    def test_fp8_scaled_bmm_gqa_broadcast(self, geometry):
        """mat1 [1,H,1,D] QFP8CH × mat2 [1,n_kv,D,BS] QFP8WT → [1,H,1,BS] fp16.

        mat2 stays at the KV head count; ``lower_bmm_fp8`` uses
        ``FloorDiv(h, gqa)`` to map Q head index to KV head index.  The CPU
        reference expands K to ``[1,H,D,BS]`` and computes fp32 matmul for
        comparison.
        """
        n_kv, gqa, head_size, block_size, label = geometry
        num_heads = n_kv * gqa
        attn_scale = float(head_size**-0.5)

        k_fp16 = cached_randn(
            (block_size, n_kv, head_size),
            dtype=torch.float16,
            scale=0.5,
            differentiation=f"gqa_k_{label}",
        )
        q_fp16 = cached_randn(
            (1, num_heads, head_size),
            dtype=torch.float16,
            scale=0.5,
            differentiation=f"gqa_q_{label}",
        )

        # ── CPU scales ────────────────────────────────────────────────────────
        # Q: per-token scale [1, H, 1, 1]
        q_4d_cpu = q_fp16.reshape(1, num_heads, 1, head_size)
        sq_cpu = _cpu_per_token_scale(q_4d_cpu)  # [1, H, 1, 1]

        # K: per-tensor scalar (conservative max of per-token scales)
        k_perm_cpu = k_fp16.permute(1, 2, 0).contiguous()  # [n_kv, D, BS]
        sk_cpu = torch.tensor(
            [_cpu_per_token_scale(k_perm_cpu).amax().item()], dtype=torch.float16
        )

        # ── Device tensors ────────────────────────────────────────────────────
        # mat1: Q [1, H, 1, D]  QFP8CH
        # mat2: K [1, n_kv, D, BS]  QFP8WT  — n_kv != H, no expand
        k_mat2_cpu = k_perm_cpu.unsqueeze(0)  # [1, n_kv, D, BS]

        q_d = q_fp16.to(DEVICE)
        k_d = k_mat2_cpu.to(DEVICE)
        sq_d = sq_cpu.to(DEVICE)
        sk_d = sk_cpu.to(DEVICE)

        def score_fn(q, k, sq, sk):
            q_4d = q.reshape(1, num_heads, 1, head_size)
            q_fp8 = torch.ops.spyre.quantize_fp8_with_scale(q_4d, sq)
            k_fp8 = torch.ops.spyre.quantize_weight_fp8_with_scale(k, sk)
            # mat1 [1,H,1,D] × mat2 [1,n_kv,D,BS] — GQA broadcast in lower_bmm_fp8
            raw = torch.ops.spyre.scaled_bmm(q_fp8, k_fp8, out_dtype=torch.float16)
            return raw * sq * sk * attn_scale  # [1, H, 1, BS]

        result = torch.compile(score_fn, dynamic=False)(q_d, k_d, sq_d, sk_d)

        # ── CPU reference ─────────────────────────────────────────────────────
        # Expand K to [1, H, D, BS] for the reference matmul.
        q_dq = _cpu_dequantize(_cpu_quantize(q_4d_cpu, sq_cpu), sq_cpu).float()
        k_expanded_cpu = k_mat2_cpu.repeat_interleave(gqa, dim=1)  # [1, H, D, BS]
        k_dq = _cpu_dequantize(_cpu_quantize(k_expanded_cpu, sk_cpu), sk_cpu).float()
        expected = (torch.matmul(q_dq, k_dq) * attn_scale).to(torch.float16)

        torch.testing.assert_close(result.cpu(), expected, atol=1e-1, rtol=0.0)


# ──────────────────────────────────────────────────────────────────────────────
# 2. GQA end-to-end fill+decode pipeline
# ──────────────────────────────────────────────────────────────────────────────


class TestFp8KvCacheQFP8WTDecodeEndToEndGQA:
    """Full fill-and-decode pipeline with GQA broadcast (gqa > 1).

    Extends the non-GQA end-to-end test by using a smaller K cache
    ``[NUM_BLOCKS, n_kv, D, BS]`` (no KV-head expansion) while Q has
    ``H = n_kv * gqa`` heads.  ``spyre.scaled_bmm`` receives mat1 with ``H``
    heads and mat2 with ``n_kv`` heads and performs the GQA broadcast inside
    the SDSC via ``FloorDiv(h, gqa)`` in the KERNEL batch coordinate.

    Tested geometry: ``n_kv=8, gqa=4, H=32, D=128, BS=128`` (Granite-3B).
    """

    @pytest.mark.parametrize(
        "geometry",
        [
            # (n_kv, gqa, head_size, block_size, label)
            (8, 4, 128, 128, "gqa4_granite"),
        ],
    )
    def test_fp8_qfp8wt_cache_fill_and_decode_gqa(self, geometry):
        """Fill K as QFP8WT into [NUM_BLOCKS, n_kv, D, BS], decode with GQA broadcast.

        The K cache has ``n_kv`` heads; Q has ``H = n_kv * gqa`` heads.
        ``spyre.scaled_bmm`` broadcasts the same KV head across ``gqa`` Q heads
        via the FloorDiv coordinate in the KERNEL tensor's SDSC.
        """
        n_kv, gqa, head_size, block_size, label = geometry
        num_heads = n_kv * gqa
        attn_scale = float(head_size**-0.5)

        k_fp16 = cached_randn(
            (block_size, n_kv, head_size),
            dtype=torch.float16,
            scale=0.5,
            differentiation=f"e2e_gqa_{label}_k",
        )
        v_fp16 = cached_randn(
            (block_size, n_kv, head_size),
            dtype=torch.float16,
            scale=0.5,
            differentiation=f"e2e_gqa_{label}_v",
        )
        query_fp16 = cached_randn(
            (1, num_heads, head_size),
            dtype=torch.float16,
            scale=0.5,
            differentiation=f"e2e_gqa_{label}_q",
        )

        block_idx = torch.tensor([0], dtype=torch.int32)

        # K cache: [NUM_BLOCKS, n_kv, D, BS] — QFP8WT [K=D, N=BS].
        # n_kv < num_heads — GQA broadcast.
        k_cache = torch.zeros(
            (NUM_BLOCKS, n_kv, head_size, block_size),
            dtype=torch.float8_e4m3fn,
        )
        k_scale_cache = torch.zeros((NUM_BLOCKS, 1, 1, 1), dtype=torch.float16)
        # V cache: [NUM_BLOCKS, BS, n_kv, D] token-major QFP8CH.
        v_cache = torch.zeros(
            (NUM_BLOCKS, block_size, n_kv, head_size),
            dtype=torch.float8_e4m3fn,
        )
        # V scale cache: [NUM_BLOCKS, BS, n_kv, 64] — trailing dim padded to
        # 64 (one FP16 stick) so index_copy_ issues stick-aligned writes.
        v_scale_cache = torch.zeros(
            (NUM_BLOCKS, block_size, n_kv, 64), dtype=torch.float16
        )

        k_d = k_fp16.to(DEVICE)
        v_d = v_fp16.to(DEVICE)
        q_d = query_fp16.to(DEVICE)
        block_d = block_idx.to(DEVICE)
        k_cache_d = k_cache.to(DEVICE)
        k_scale_cache_d = k_scale_cache.to(DEVICE)
        v_cache_d = v_cache.to(DEVICE)
        v_scale_cache_d = v_scale_cache.to(DEVICE)

        def fill_and_decode_gqa(
            k, v, q, block, k_cache, k_sc_cache, v_cache, v_sc_cache
        ):
            # ── Fill K as QFP8WT ───────────────────────────────────────────────
            # k [BS, n_kv, D] → [n_kv, D, BS]; .contiguous() avoids spurious
            # K-inner heuristic in _qfp8wt_stl.
            k_perm = k.permute(1, 2, 0).contiguous()  # [n_kv, D, BS]
            sk = torch.amax(
                torch.ops.spyre.quantscalepertokenfp8(k_perm)
            )  # scalar
            k_fp8 = torch.ops.spyre.quantize_weight_fp8_with_scale(k_perm, sk)
            # Scatter: unsqueeze(0) → [1, n_kv, D, BS]; block dim = 0.
            k_cache.index_copy_(0, block, k_fp8.unsqueeze(0))
            k_sc_cache.index_copy_(0, block, sk.reshape(1, 1, 1, 1))

            # ── Fill V as QFP8CH ───────────────────────────────────────────────
            v_in = v.unsqueeze(0)  # [1, BS, n_kv, D]
            sv = torch.ops.spyre.quantscalepertokenfp8(v_in)  # [1, BS, n_kv, 1]
            v_fp8 = torch.ops.spyre.quantize_fp8_with_scale(v_in, sv)
            v_cache.index_copy_(0, block, v_fp8)
            # Pad sv [1, BS, n_kv, 1] → [1, BS, n_kv, 64] for stick-aligned scatter.
            sv_padded = torch.nn.functional.pad(sv, (0, 63))
            v_sc_cache.index_copy_(0, block, sv_padded)

            # ── Decode: gather K (QFP8WT EA preserved) ────────────────────────
            k_block = k_cache.index_select(0, block)  # [1, n_kv, D, BS]
            sk_block = k_sc_cache.index_select(0, block)  # [1, 1, 1, 1]
            k_4d = k_block  # [1, n_kv, D, BS] — GQA broadcast in bmm.

            # ── Decode: quantize Q ─────────────────────────────────────────────
            q_4d = q.reshape(1, num_heads, 1, head_size)  # [1, H, 1, D]
            sq = torch.ops.spyre.quantscalepertokenfp8(q_4d)  # [1, H, 1, 1]
            q_fp8 = torch.ops.spyre.quantize_fp8_with_scale(q_4d, sq)

            # ── Decode: FP8 Q·Kᵀ with GQA broadcast ──────────────────────────
            # mat1 Q  [1, H,    1, D]  H = n_kv * gqa
            # mat2 K  [1, n_kv, D, BS] n_kv < H — GQA broadcast
            # out     [1, H,    1, BS] fp16
            scores = torch.ops.spyre.scaled_bmm(
                q_fp8, k_4d, out_dtype=torch.float16
            )
            scores = scores * sq * sk_block * attn_scale
            probs = torch.softmax(scores, dim=-1)  # [1, H, 1, BS]

            # ── Decode: gather V (QFP8CH), dequantize, P·V ────────────────────
            v_block = v_cache.index_select(0, block)  # [1, BS, n_kv, D]
            sv_block = v_sc_cache.index_select(0, block)  # [1, BS, n_kv, 64]
            sv_block = sv_block[:, :, :, :1]  # slice back to real scale
            v_dq = torch.ops.spyre.dequantize_fp8_with_scale(v_block, sv_block)
            # Reshape probs [1, H, 1, BS] → [1, n_kv, gqa, BS] and v_dq to
            # [1, n_kv, BS, D], then matmul.  Avoids repeat_interleave which
            # triggers a lossy work-division scheduler path for n_kv=8, gqa=4.
            v_3d = v_dq.squeeze(0).permute(1, 0, 2).unsqueeze(0)  # [1, n_kv, BS, D]
            probs_nkv = probs.reshape(1, n_kv, gqa, -1)            # [1, n_kv, gqa, BS]
            out = torch.matmul(probs_nkv, v_3d)                    # [1, n_kv, gqa, D]
            return out.reshape(1, num_heads, head_size)             # [1, H, D]

        result = torch.compile(fill_and_decode_gqa, dynamic=False)(
            k_d, v_d, q_d, block_d,
            k_cache_d, k_scale_cache_d, v_cache_d, v_scale_cache_d,
        )

        # Guard: V scale cache block-0 must be non-zero after scatter — regression
        # check for sub-stick scatter / co-fused scatter+gather SDSC role swap bug.
        sv_elem0 = v_scale_cache_d.index_select(0, block_d).cpu()[0, :, :, 0]
        assert sv_elem0.abs().max().item() > 0.0, (
            f"V scale cache block-0 read back all-zeros after scatter! "
            f"sub-stick scatter hardware error — sv_elem0 max={sv_elem0.abs().max().item()}"
        )

        # ── CPU reference ──────────────────────────────────────────────────────
        # K: permute → per-tensor scale → quantize → dequantize → expand H.
        k_perm_cpu = k_fp16.permute(1, 2, 0).contiguous()  # [n_kv, D, BS]
        sk_cpu = _cpu_per_token_scale(k_perm_cpu).amax()
        k_dq = _cpu_dequantize(_cpu_quantize(k_perm_cpu, sk_cpu), sk_cpu).float()
        k_4d_cpu = k_dq.repeat_interleave(gqa, dim=0).unsqueeze(0)  # [1, H, D, BS]

        # Q: per-token scale → quantize → dequantize.
        q_4d_cpu = query_fp16.reshape(1, num_heads, 1, head_size)
        sq_cpu = _cpu_per_token_scale(q_4d_cpu)
        q_dq = _cpu_dequantize(_cpu_quantize(q_4d_cpu, sq_cpu), sq_cpu).float()

        sc_ref = torch.matmul(q_dq, k_4d_cpu) * attn_scale  # [1, H, 1, BS]
        pr_ref = torch.softmax(sc_ref.to(torch.float16), dim=-1)

        # V: per-token scale → quantize → dequantize → reshape for P·V.
        v_in_cpu = v_fp16.unsqueeze(0)
        sv_cpu = _cpu_per_token_scale(v_in_cpu)
        v_dq_cpu = _cpu_dequantize(
            _cpu_quantize(v_in_cpu, sv_cpu), sv_cpu
        ).float()
        v_3d_cpu = v_dq_cpu.squeeze(0).permute(1, 0, 2).unsqueeze(0)  # [1, n_kv, BS, D]
        pr_nkv_cpu = pr_ref.to(torch.float32).reshape(1, n_kv, gqa, -1)

        out_ref = torch.matmul(pr_nkv_cpu, v_3d_cpu)  # [1, n_kv, gqa, D]
        expected = out_ref.reshape(1, num_heads, head_size).to(torch.float16)

        torch.testing.assert_close(result.cpu(), expected, atol=2e-1, rtol=0.0)
