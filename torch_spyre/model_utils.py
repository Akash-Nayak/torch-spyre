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

"""Optimal weight layout utilities for loading models onto Spyre.

Transfers ``nn.Linear`` weights to Spyre with a device layout where the
``out_features`` dimension is stickified (the optimal layout for Spyre
matmul where both operands need their rows in the stick).

This is achieved using ``dim_order=[1, 0]`` in ``SpyreTensorLayout``,
which tells the DMA engine to stickify along host dim-0 (out_features)
instead of the default last dim (in_features). No CPU transpose or
intermediate copy is required.

Critically, the tensor's PyTorch shape stays ``(out, in)`` -- only the
*device* layout changes. This means:

  * ``nn.Linear.forward`` works unmodified
  * ``F.linear`` / ``aten.linear`` works unmodified. The Spyre
    decomposition still does ``weight.transpose(-1, -2)`` (a metadata-
    only op), and the Spyre layout propagation engine recognizes the
    stickification matches the matmul's needs -- no restickify cost.
  * Models loaded with this utility are drop-in compatible with all
    existing inference paths.

Resolves:
  * Issue #1339 (optimal weight layout for Spyre)

Usage::

    # Explicit:
    from torch_spyre.model_utils import load_model_to_spyre
    load_model_to_spyre(model)

    # Transparent for any code that uses .to("spyre"):
    from torch_spyre.model_utils import patch_module_to_for_spyre
    patch_module_to_for_spyre()
    model.to("spyre")
"""

from torch_spyre._inductor.logging_utils import get_inductor_logger


import torch
import torch.nn as nn

from torch_spyre._C import (
    DataFormats,
    SpyreTensorLayout,
    copy_tensor,
    get_device_dtype,
    spyre_empty_with_layout,
)
from torch_spyre.constants import DEVICE_NAME

logger = get_inductor_logger("model_utils")


def _ensure_spyre_runtime() -> None:
    """Ensure Spyre runtime is up before calling DMA helpers from _C."""
    spyre = getattr(torch, DEVICE_NAME)
    if spyre.is_initialized():
        return
    torch.empty(0, dtype=torch.float16, device=DEVICE_NAME)


def _validate_target_dtype(dtype: torch.dtype) -> None:
    """Raise early if ``dtype`` has no Spyre device representation."""
    if get_device_dtype(dtype) == DataFormats.INVALID:
        raise ValueError(
            f"dtype {dtype} has no Spyre device representation. "
            f"See torch_spyre._C.DataFormats for the list of supported "
            f"formats, or torch_spyre._inductor.dtype_ops.DtypeOpTable "
            f"for the conversion pairs."
        )


# --- DMA helpers -----------------------------------------------------


def _dma_to_spyre_default(
    cpu_tensor: torch.Tensor,
    target_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Transfer a CPU tensor to Spyre with the default layout.

    Used for non-Linear-weight tensors (biases, embeddings, layer norm
    parameters, buffers). Stickifies along the last dimension.
    """
    if not cpu_tensor.is_contiguous():
        cpu_tensor = cpu_tensor.contiguous()
    dev_dtype = target_dtype if target_dtype is not None else cpu_tensor.dtype
    layout = SpyreTensorLayout(list(cpu_tensor.shape), dev_dtype)
    dst = spyre_empty_with_layout(
        cpu_tensor.size(), cpu_tensor.stride(), dev_dtype, layout
    )
    copy_tensor(cpu_tensor, dst, non_blocking=False)
    return dst


def _dma_to_spyre_dim_order_swapped(
    weight: torch.Tensor,
    target_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Transfer a 2D Linear weight to Spyre with dim_order=[1, 0].

    The host tensor shape ``(out_features, in_features)`` is preserved
    on the device, but the data is stickified along ``out_features``
    (dim 0) rather than the default ``in_features`` (dim 1). This
    matches the layout Spyre needs for efficient matmul and avoids
    both a CPU transpose and a device-side restickify.

    Caller must ensure ``weight.ndim == 2``.
    """
    assert weight.ndim == 2, "dim_order=[1,0] path is for 2D weights only"

    if not weight.is_contiguous():
        weight = weight.contiguous()
    dev_dtype = target_dtype if target_dtype is not None else weight.dtype
    layout = SpyreTensorLayout(
        list(weight.shape),  # host_size: (out, in)
        list(weight.stride()),  # host_strides: row-major
        dev_dtype,
        [1, 0],  # dim_order: stick on dim-0 = out_features
    )
    dst = spyre_empty_with_layout(weight.size(), weight.stride(), dev_dtype, layout)
    copy_tensor(weight, dst, non_blocking=False)
    return dst


def _dma_to_spyre_fp8_kernel(
    weight: torch.Tensor,
) -> torch.Tensor:
    """Transfer pre-quantized FP8 weight to Spyre with KERNEL layout.

    Creates a KERNEL tensor with 2D stick layout [2, 64] and
    ElementArrangement.QFP8WT for use with _scaled_mm.

    This function is specifically for pre-quantized FP8 weights loaded
    from model checkpoints (e.g., granite-3.3-8b-instruct-fp8). It
    creates the optimal device layout for FP8 matrix multiplication
    without requiring runtime quantization.

    Args:
        weight: Pre-quantized FP8 weight tensor (torch.float8_e4m3fn)
                Shape: (out_features, in_features)
                Must be 2D and contiguous

    Returns:
        Spyre tensor with KERNEL layout optimized for FP8 matmul:
        - Tensor type: KERNEL (not OUTPUT)
        - Stick layout: 2D [2, 64] (not 1D [128])
        - Element arrangement: QFP8WT
        - Compatible with torch.ops.aten._scaled_mm

    Raises:
        AssertionError: If weight is not 2D or not FP8 dtype

    Example:
        >>> # Load pre-quantized weight from checkpoint
        >>> weight_fp8 = model.layer.weight  # torch.float8_e4m3fn
        >>> weight_device = _dma_to_spyre_fp8_kernel(weight_fp8)
        >>> # Now compatible with _scaled_mm
    """
    assert weight.ndim == 2, "FP8 KERNEL layout is for 2D weights only"
    assert weight.dtype == torch.float8_e4m3fn, (
        f"Weight must be torch.float8_e4m3fn, got {weight.dtype}"
    )

    if not weight.is_contiguous():
        weight = weight.contiguous()

    # Import ElementArrangement enum
    from torch_spyre._inductor.constants import ElementArrangement

    # Create KERNEL layout with 2D stick [2, 64] and QFP8WT arrangement
    # This matches the layout created by qfp8wt operation but without
    # runtime quantization overhead
    layout = SpyreTensorLayout(
        list(weight.shape),  # host_size: (out_features, in_features)
        list(weight.stride()),  # host_strides: row-major (out, 1)
        torch.float8_e4m3fn,  # FP8 E4M3 dtype
        [1, 0],  # dim_order: stick on dim-0 = out_features
        ElementArrangement.QFP8WT,  # 2D stick [2, 64] for KERNEL tensor
    )

    # Create empty tensor with KERNEL layout
    dst = spyre_empty_with_layout(
        weight.size(), weight.stride(), torch.float8_e4m3fn, layout
    )

    # DMA transfer from CPU to device
    copy_tensor(weight, dst, non_blocking=False)

    return dst


# --- Model loading ---------------------------------------------------


def load_model_to_spyre(
    model: nn.Module,
    dtype: torch.dtype | None = None,
    use_fp8_weights: bool = False,
) -> nn.Module:
    """Transfer model to Spyre with optimal weight layout.

    For each ``nn.Linear``, the weight is transferred using the optimal
    layout for Spyre matmul operations:

    - **FP8 pre-quantized weights** (if use_fp8_weights=True): KERNEL
      layout with 2D stick [2, 64] and ElementArrangement.QFP8WT
    - **FP16 weights**: dim_order=[1, 0] so out_features is stickified
    - **Other parameters**: default Spyre layout

    Tensor shapes are preserved, so the model works unmodified with
    existing inference paths.

    Args:
        model: Model to transfer to Spyre device
        dtype: Target dtype for conversion (optional). If None, preserves
               original dtypes. Ignored for FP8 weights when use_fp8_weights=True.
        use_fp8_weights: If True, treat torch.float8_e4m3fn weights as
                        pre-quantized and use KERNEL layout with 2D stick.
                        This eliminates runtime quantization overhead for
                        models like granite-3.3-8b-instruct-fp8.

    Returns:
        The model with all parameters transferred to Spyre device

    Example:
        >>> # FP16 model
        >>> model = AutoModelForCausalLM.from_pretrained("granite-3.3-8b")
        >>> model = load_model_to_spyre(model)

        >>> # Pre-quantized FP8 model
        >>> model = AutoModelForCausalLM.from_pretrained(
        ...     "granite-3.3-8b-instruct-fp8"
        ... )
        >>> model = load_model_to_spyre(model, use_fp8_weights=True)
    """
    if dtype is not None:
        _validate_target_dtype(dtype)
    # Ensure Spyre runtime is initialized before using _C functions
    _ensure_spyre_runtime()

    linear_count = 0
    fp8_kernel_count = 0
    other_param_count = 0
    buffer_count = 0

    for name, module in model.named_modules():
        is_linear = isinstance(module, nn.Linear)

        for param_name, param in list(module._parameters.items()):
            if param is None:
                continue
            if param.device.type == DEVICE_NAME:
                continue

            p = param.data

            # FP8 pre-quantized weight -> KERNEL layout with 2D stick
            if (
                is_linear
                and param_name == "weight"
                and p.ndim == 2
                and p.dtype == torch.float8_e4m3fn
                and use_fp8_weights
            ):
                logger.debug(
                    "  %s.%s: shape=%s dtype=%s -> Spyre FP8 KERNEL layout (2D stick)",
                    name,
                    param_name,
                    list(p.shape),
                    p.dtype,
                )
                dev = _dma_to_spyre_fp8_kernel(p)
                fp8_kernel_count += 1

            # 2D Linear weight -> optimal stickified layout via dim_order
            elif is_linear and param_name == "weight" and p.ndim == 2:
                logger.debug(
                    "  %s.%s: shape=%s -> Spyre dim_order=[1, 0]",
                    name,
                    param_name,
                    list(p.shape),
                )
                dev = _dma_to_spyre_dim_order_swapped(p, target_dtype=dtype)
                linear_count += 1

            # Everything else (bias, embeddings, norms, ...) -> default layout
            else:
                logger.debug(
                    "  %s.%s: shape=%s -> Spyre default layout",
                    name,
                    param_name,
                    list(p.shape),
                )
                dev = _dma_to_spyre_default(p, target_dtype=dtype)
                other_param_count += 1

            module._parameters[param_name] = nn.Parameter(
                dev, requires_grad=param.requires_grad
            )

        for buf_name, buf in list(module._buffers.items()):
            if buf is None or buf.device.type == DEVICE_NAME:
                continue
            module._buffers[buf_name] = _dma_to_spyre_default(buf, target_dtype=dtype)
            buffer_count += 1

    logger.info(
        "load_model_to_spyre: %d Linear weights (FP16), %d FP8 KERNEL weights, "
        "%d other params, %d buffers transferred",
        linear_count,
        fp8_kernel_count,
        other_param_count,
        buffer_count,
    )
    return model


def load_fp8_model_to_spyre(model: nn.Module) -> nn.Module:
    """Load pre-quantized FP8 model to Spyre with KERNEL layouts.

    Convenience wrapper for loading models with pre-quantized FP8 weights
    (e.g., granite-3.3-8b-instruct-fp8). Automatically detects FP8 weights
    and uses KERNEL layout with 2D stick [2, 64] for optimal _scaled_mm
    performance.

    This eliminates the need for runtime quantization using qfp8wt,
    significantly reducing model startup time.

    Args:
        model: Model with pre-quantized torch.float8_e4m3fn weights

    Returns:
        Model with FP8 weights in KERNEL layout on Spyre device

    Example:
        >>> from transformers import AutoModelForCausalLM
        >>> from torch_spyre.model_utils import load_fp8_model_to_spyre
        >>>
        >>> # Load pre-quantized FP8 model from HuggingFace
        >>> model = AutoModelForCausalLM.from_pretrained(
        ...     "ibm-granite/granite-3.3-8b-instruct-fp8"
        ... )
        >>>
        >>> # Transfer to Spyre with optimal FP8 KERNEL layout
        >>> model = load_fp8_model_to_spyre(model)
        >>>
        >>> # Now ready for inference with _scaled_mm
        >>> # No runtime quantization needed!

    See Also:
        load_model_to_spyre: General model loading function
        _dma_to_spyre_fp8_kernel: Low-level FP8 KERNEL transfer
    """
    return load_model_to_spyre(model, use_fp8_weights=True)


# --- nn.Module.to() monkeypatch --------------------------------------


def patch_module_to_for_spyre() -> None:
    """Monkeypatch ``nn.Module.to`` for automatic optimal Spyre loading.

    After patching, ``model.to("spyre")`` will use the optimal weight
    layout for every ``nn.Linear`` in the model. Non-Spyre destinations
    fall through to the original ``nn.Module.to``.
    # Robust idempotency: check the live attribute on the patched callable
    # rather than a module-level flag.
    """
    if getattr(nn.Module.to, "_spyre_patched", False):
        return
    orig_module_to = nn.Module.to

    def _spyre_module_to(self, *args, **kwargs):
        def _is_spyre(d):
            return d is not None and torch.device(d).type == DEVICE_NAME

        target_is_spyre = any(
            _is_spyre(a) for a in args if isinstance(a, (str, torch.device))
        ) or _is_spyre(kwargs.get("device"))

        if not target_is_spyre:
            return orig_module_to(self, *args, **kwargs)

        dtype = kwargs.get("dtype")
        if dtype is None:
            for arg in args:
                if isinstance(arg, torch.dtype):
                    dtype = arg
                    break
        return load_model_to_spyre(self, dtype=dtype)

    _spyre_module_to._spyre_patched = True  # type: ignore[attr-defined]
    nn.Module.to = _spyre_module_to  # type: ignore[method-assign]
    logger.info("Patched nn.Module.to() for automatic Spyre weight layout optimization")
