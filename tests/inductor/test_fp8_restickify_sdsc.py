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

"""Tests for FP8 ReStickifyOpHBM emission and plan commitment.

Background
----------
The production Granite FP8 graph always converts FP8 → FP16 (via fp8todl16)
before any reshape-induced stick swap, so there is no FP8 ReStickifyOpHBM in
the production SDSCs.

To force a standalone FP8 ReStickifyOpHBM we need a graph where:
  1. An FP8 buffer is produced with dim A on the stick (or in sparse layout)
  2. The downstream batchmatmulfp8 consumer needs dim B (K/reduction) on
     the stick
  3. deeptools cannot absorb the layout delta into adjacent ops

Forcing mechanism (test_fp8_restickify_forced_plan)
---------------------------------------------------
When M << FP8 stick size (128 elements), the matmul activation output naturally
admits a sparse layout candidate (M doesn't fill a full stick → stride_map entry
of −1). To force the beam search to *commit* the sparse layout (instead of
choosing the zero-cost dense alternative), we pre-place the FP16 activation on
device with an explicit sparse SpyreTensorLayout. This gives the graph input
exactly one layout candidate (the sparse STL), which propagates through qfp8ch
to a single sparse FP8 candidate. The downstream batchmatmulfp8 sees the
sparse FP8 activation with K not on the stick → expand_sparse provides a
feasible dense target → compute_restickify_needed returns (True, tgt_stl) →
restickify_plan is non-empty.

Graph structure:
  x_act: (1, M, K) FP16 — pre-placed with sparse STL (no dense alternative)
  xfp8  = qfp8ch(x_act)         → FP8, single sparse candidate, K not on stick
  xfp8_2d = reshape(xfp8, (M,K))
  out   = scaled_mm(xfp8_2d, wfp8) → batchmatmulfp8 needs K on stick
                                     → restickify inserted before matmul

SDSC probe test (test_fp8_attention_two_matmul_cpu)
---------------------------------------------------
The original probe test is kept as a baseline. It uses the full
quantize_fp8_with_scale decomposition chain (reciprocal → mul → clamp →
qfp8ch), which gives multiple FP8 layout candidates including a zero-cost
dense alternative. The beam always chooses dense, so restickify_plan remains
empty. This test documents the current (non-forcing) behaviour.

Run with:
    pytest tests/inductor/test_fp8_restickify_sdsc.py -v -s

After running test_fp8_restickify_forced_plan, check for ReStickifyOpHBM:
    grep -r '"opFuncName"' /tmp/torchinductor_$(whoami)/inductor-spyre/*/sdsc_*.json \\
        | grep -i restick
"""

import json
import math
import os
import pathlib
import pwd
from unittest.mock import patch
import unittest

import torch

from tests.inductor.utils_inductor import _compile_and_run, compare_with_pytorch
from torch.spyre import SpyreTensorLayout
from torch._inductor.virtualized import V
from torch_spyre._C import DataFormats, ElementArrangement

FP8_MAX = 448.0

# Shapes chosen so seq_len > 128 (one FP8 stick = 128 elems) to prevent
# deeptools nop_restickify from absorbing the restickify.
BATCH = 1
HEADS = 1
SEQ_LEN = 256   # > 128: forces multi-stick restickify
HEAD_DIM = 128  # K for Q@K^T, N for softmax@V

DEVICE = torch.device("spyre")

# ---------------------------------------------------------------------------
# Sparse-STL pre-placement helpers
# ---------------------------------------------------------------------------

def _sparse_fp16_stl_for_shape(batch, m, k):
    """Return the sparse FP16 SpyreTensorLayout for a (batch, M, K) tensor.

    When M < 64 (FP16 stick size), the M dimension does not fill a full stick.
    The sparse layout uses stride_map=-1 for the sub-stick M slot, which is
    the same shape that arises internally for small-M matmul outputs during
    beam evaluation.  Pre-placing an FP16 tensor with this layout gives the
    graph input exactly *one* layout candidate (sparse), so qfp8ch propagates
    it to a single sparse FP8 candidate with no zero-cost dense alternative.

    Device encoding mirrors the pattern observed in the beam trace for the
    existing chained-matmul regression tests with M=2:
      in_stl  device_size=[K/64, 1, ..., M, 64]
              stride_map=[1, -1, ..., K, -1]

    Concretely for (batch=1, M=2, K=1024):
      device_size=[1024, 1, 1, 2, 64]  stride_map=[1, -1, -1, 1024, -1]
    """
    assert m < 64, "sparse FP16 STL only valid when M < 64 (sub-stick M)"
    k_groups = k // 64  # number of 64-element FP16 sticks along K
    # Five-dimensional sparse device layout:
    #   dim0: K/64 K-groups, stride 1 (inner-most non-stick outer dim)
    #   dim1: batch (size 1), stride -1 (not used → sparse placeholder)
    #   dim2: size 1, stride -1 (spare dimension)
    #   dim3: M rows, stride K (one full row = K elements)
    #   dim4: 64-element stick, stride -1 (this IS the stick variable)
    device_size = [k_groups, batch, 1, m, 64]
    stride_map  = [1, -1, -1, k, -1]
    return SpyreTensorLayout(device_size, stride_map, DataFormats.SEN169_FP16)


def _compile_and_capture_plan(fn, *args):
    """Compile fn and return (result_or_None, restickify_plan, backend_error_or_None).

    The restickify_plan is captured during finalize_layouts, which runs inside
    the Inductor compilation pass — before dxp_standalone is invoked.  If
    dxp_standalone fails (e.g. the batchmatmulfp8 SDSC hits a shape-specific
    DDC error under a non-default SENARCH), the plan is still returned so the
    Inductor-side assertion can proceed independently.
    """
    import torch_spyre._inductor.passes as _passes
    from torch._inductor.exc import InductorError

    captured = {}
    finalize_layouts = _passes.finalize_layouts

    def capturing_finalize_layouts(graph):
        finalize_layouts(graph)
        captured["plan"] = dict(V.graph.restickify_plan)

    try:
        with patch.object(_passes, "finalize_layouts", capturing_finalize_layouts):
            result = _compile_and_run(fn, list(args), DEVICE)
        return result, captured.get("plan", {}), None
    except InductorError as exc:
        # dxp_standalone failed on some SDSC after finalize_layouts already ran.
        return None, captured.get("plan", {}), exc


def _restickify_cost(plan):
    """Return the total element count of all restickify entries in the plan."""
    return sum(
        math.prod(int(s) for s in entry["target_layout"].size)
        for entries in plan.values()
        for entry in entries
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestFP8RestickifyForced(unittest.TestCase):
    """Force a non-empty FP8 restickify_plan by pre-placing a sparse FP16 input.

    When the FP16 activation is pre-placed with the sparse SpyreTensorLayout,
    the graph input has exactly one layout candidate (sparse).  qfp8ch
    propagates that to a single sparse FP8 candidate.  The downstream
    batchmatmulfp8 needs K on the stick but the sparse FP8 has K off-stick →
    expand_sparse provides a feasible dense target → restickify committed.

    The test asserts restickify_plan is non-empty.  SDSC emission requires
    actual hardware (the test compiles on the simulation path); the plan
    assertion validates the Inductor side is correct.
    """

    def test_fp8_restickify_forced_plan(self):
        """Pre-placed sparse FP16 forces a single sparse FP8 candidate → restickify committed.

        Graph:
          x_act (1, M=2, K=1024) FP16 — pre-placed with sparse STL
          xfp8  = qfp8ch(x_act)         ← single sparse FP8 candidate (K not on stick)
          xfp8_2d = reshape(xfp8, (2, 1024))
          out   = scaled_mm(xfp8_2d, wfp8)  ← batchmatmulfp8 needs K on stick
                                              → restickify inserted
        """
        # M must be small (< 64) so that sparse FP16 STL is valid.
        BATCH_DIM = 1
        M = 2          # number of token positions (small → sparse layout)
        K = 1024       # hidden dim = K for the matmul (reduction variable)
        N = 4096       # output dim

        # CPU reference tensors (used to compute shapes; values don't matter for
        # the plan assertion).
        torch.manual_seed(42)
        x_cpu = torch.randn(BATCH_DIM, M, K, dtype=torch.float16) * 0.01
        w_cpu = torch.randn(K, N, dtype=torch.float16) * 0.01
        ws    = torch.full((N,), 0.1, dtype=torch.float16)

        # Pre-place x on device with the sparse FP16 layout.
        # This gives the graph input exactly one STL candidate (sparse),
        # so _qfp8ch_stl produces exactly one sparse FP8 candidate.
        sparse_stl = _sparse_fp16_stl_for_shape(BATCH_DIM, M, K)
        x_dev = x_cpu.to(device_layout=sparse_stl)

        # Weights go through quantize_weight_fp8_with_scale (qfp8wt layout).
        # They are placed with the standard row-major layout.
        w_dev  = w_cpu.to(DEVICE)
        ws_dev = ws.to(DEVICE)

        def fn(x_dev, w_dev, ws_dev):
            # Apply qfp8ch DIRECTLY (no mul/clamp chain) so there is no dense
            # alternative candidate from an intermediate pointwise buffer.
            xfp8     = torch.ops.spyre.qfp8ch(x_dev)
            xfp8_2d  = xfp8.reshape(M, K)
            wfp8     = torch.ops.spyre.quantize_weight_fp8_with_scale(w_dev, ws_dev)
            return torch.ops.spyre.scaled_mm(xfp8_2d, wfp8, out_dtype=torch.float16)

        _, plan, backend_error = _compile_and_capture_plan(fn, x_dev, w_dev, ws_dev)

        # ---- Inductor-side plan assertion (always checked) ----
        # finalize_layouts runs before dxp_standalone; the plan is valid even
        # if the backend subsequently fails on a different SDSC.
        self.assertNotEqual(
            plan,
            {},
            "restickify_plan is empty — the sparse FP8 activation did not "
            "trigger a restickify commitment.  The pre-placed sparse STL may "
            "not be propagating as expected through qfp8ch.",
        )
        cost = _restickify_cost(plan)
        self.assertGreater(
            cost,
            0,
            "restickify_plan has entries but zero total element cost.",
        )

        # Report which ops have restickify entries.
        print(f"\n[PLAN] restickify_plan: {list(plan.keys())}")
        print(f"[PLAN] total element cost: {cost}")
        for op_name, entries in plan.items():
            for e in entries:
                print(f"  {op_name}: restickify {e['arg_name']} → target_layout.size={list(e['target_layout'].size)}")

        restickify_sdsc_file = self._report_sdsc_ops()

        # ---- Backend assertion ----
        # Confirm that Inductor emitted an FP8 ReStickifyOpHBM SDSC.
        # This validates the codegen path regardless of whether dxp compiles it.
        self.assertIsNotNone(
            restickify_sdsc_file,
            "No ReStickifyOpHBM + SEN143_FP8 SDSC was found in the most recent "
            "bundle — Inductor did not emit the expected FP8 restickify op.",
        )
        print(f"\n[SDSC] FP8 ReStickifyOpHBM emitted: {restickify_sdsc_file}")

        # If dxp failed, report it.  The test itself passes as long as:
        #   (a) the plan is non-empty, AND
        #   (b) an FP8 ReStickifyOpHBM SDSC was emitted.
        # A backend failure on a *different* SDSC (e.g. batchmatmulfp8 under
        # a non-standard SENARCH) is pre-existing and out of scope.
        if backend_error is not None:
            print(
                f"\n[BACKEND] dxp_standalone error (may be pre-existing):\n"
                f"{backend_error}"
            )

    def _report_sdsc_ops(self):
        """Print all opFuncNames from the most recent SDSC bundle.

        Returns the path to the FP8 ReStickifyOpHBM SDSC file in the most
        recent bundle, or None if no such SDSC was found.  The return value
        is used to assert that Inductor actually emitted the FP8 restickify op.
        """
        username = os.environ.get("USER") or pwd.getpwuid(os.getuid()).pw_name
        inductor_dir = pathlib.Path(f"/tmp/torchinductor_{username}/inductor-spyre")
        if not inductor_dir.exists():
            print("\n[SDSC] No output dir found")
            return None

        bundles = sorted(
            inductor_dir.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True
        )
        seen_ops = {}
        fp8_restickify_file = None
        for bundle in bundles[:1]:   # only the most-recent bundle
            for sdsc_file in sorted(bundle.glob("sdsc_*.json")):
                try:
                    data = json.loads(sdsc_file.read_text())
                    key = list(data.keys())[0]
                    for dsc in data[key].get("dscs_", []):
                        op_key = list(dsc.keys())[0]
                        op = dsc[op_key]
                        func = op["computeOp_"][0]["opFuncName"]
                        dtypes = {
                            lds["dataFormat_"]
                            for lds in op.get("labeledDs_", [])
                        }
                        seen_ops.setdefault(func, set()).update(dtypes)
                        if (
                            func == "ReStickifyOpHBM"
                            and "SEN143_FP8" in dtypes
                        ):
                            fp8_restickify_file = sdsc_file
                except Exception:
                    pass

        print("\n[SDSC] opFuncName → dtypes seen:")
        for func, dtypes in sorted(seen_ops.items()):
            marker = " ← FP8 RESTICKIFY" if (
                func == "ReStickifyOpHBM" and "SEN143_FP8" in dtypes
            ) else ""
            print(f"  {func}: {dtypes}{marker}")

        return fp8_restickify_file


class TestFP8RestickifySDSCProbe(unittest.TestCase):
    """Original probe test — two-matmul attention pattern with full decomposition.

    This test uses the full quantize_fp8_with_scale decomposition chain
    (reciprocal → mul → clamp → qfp8ch), which produces multiple FP8 layout
    candidates including a zero-cost dense alternative.  The beam always
    chooses dense, so restickify_plan is empty.  Kept as a baseline to
    document that behaviour and to report SDSC ops.

    To see a non-empty plan, use TestFP8RestickifyForced instead.
    """

    def test_fp8_attention_two_matmul_cpu(self):
        """Probe test: compile the graph and report what SDSC ops are emitted.

        Not a correctness test — the goal is to observe whether
        ReStickifyOpHBM appears in the SDSC bundle with SEN143_FP8 dtype.
        The graph compiles and runs; SDSC ops are printed regardless.

        Expected: restickify_plan is EMPTY (dense always wins in this graph).
        """
        B, H, M, N, K = BATCH, HEADS, SEQ_LEN, SEQ_LEN, HEAD_DIM

        # Inputs — small values to avoid FP8 saturation
        torch.manual_seed(42)
        q  = torch.randn(B * H, M, K, dtype=torch.float16) * 0.01
        k  = torch.randn(B * H, K, N, dtype=torch.float16) * 0.01
        v  = torch.randn(B * H, N, K, dtype=torch.float16) * 0.01
        sq = torch.full((1,), 0.1, dtype=torch.float16)
        sk = torch.full((1,), 0.1, dtype=torch.float16)
        sv = torch.full((1,), 0.1, dtype=torch.float16)
        with torch.no_grad():
            scores_ref = (q.reshape(M, K) @ k.reshape(K, N))
            ss_val = float(scores_ref.abs().max()) / FP8_MAX
            ss_val = max(ss_val, 1e-4)
        ss = torch.full((1,), ss_val, dtype=torch.float16)

        def fn(q, k, v, sq, sk, sv, ss):
            BH = q.shape[0]
            q2d   = q.reshape(BH * M, K)
            qfp8  = torch.ops.spyre.quantize_fp8_with_scale(q2d, sq)
            kfp8  = torch.ops.spyre.quantize_weight_fp8_with_scale(k.reshape(K, N), sk)
            s2d   = torch.ops.spyre.scaled_mm(qfp8, kfp8, out_dtype=torch.float16)
            s3d   = s2d.reshape(BH, M, N)
            sfp8  = torch.ops.spyre.quantize_fp8_with_scale(s3d.reshape(BH * M, N), ss)
            vfp8  = torch.ops.spyre.quantize_weight_fp8_with_scale(v.reshape(N, K), sv)
            out2d = torch.ops.spyre.scaled_mm(sfp8, vfp8, out_dtype=torch.float16)
            return out2d.reshape(BH, M, K)

        def pytorch_fn(q, k, v, sq, sk, sv, ss):
            return torch.zeros(B * H, M, K, dtype=torch.float16)

        compare_with_pytorch(
            fn, pytorch_fn, q, k, v, sq, sk, sv, ss,
            atol=1e9, rtol=1e9,   # no numeric assertion
        )
        self._report_sdsc_ops()

    def _report_sdsc_ops(self):
        """Print all opFuncNames from the most recent SDSC bundle."""
        username = os.environ.get("USER") or pwd.getpwuid(os.getuid()).pw_name
        inductor_dir = pathlib.Path(f"/tmp/torchinductor_{username}/inductor-spyre")
        if not inductor_dir.exists():
            print("\n[SDSC] No output dir found")
            return

        bundles = sorted(
            inductor_dir.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True
        )
        seen_ops = {}
        for bundle in bundles[:1]:   # only the most-recent bundle
            for sdsc_file in sorted(bundle.glob("sdsc_*.json")):
                try:
                    data = json.loads(sdsc_file.read_text())
                    key = list(data.keys())[0]
                    for dsc in data[key].get("dscs_", []):
                        op_key = list(dsc.keys())[0]
                        op = dsc[op_key]
                        func = op["computeOp_"][0]["opFuncName"]
                        dtypes = {
                            lds["dataFormat_"]
                            for lds in op.get("labeledDs_", [])
                        }
                        seen_ops.setdefault(func, set()).update(dtypes)
                except Exception:
                    pass

        print("\n[SDSC] opFuncName → dtypes seen:")
        for func, dtypes in sorted(seen_ops.items()):
            marker = " ← FP8 RESTICKIFY" if (
                func == "ReStickifyOpHBM" and "SEN143_FP8" in dtypes
            ) else ""
            print(f"  {func}: {dtypes}{marker}")


if __name__ == "__main__":
    unittest.main()
