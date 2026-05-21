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

from typing import Optional, Sequence
import torch
import torch._dynamo
from torch._inductor.fx_passes.reinplace import inplaceable_ops, InplaceableOp
from torch_spyre.ops.eager import compile_once
from torch_spyre.ops.fallbacks import warn_fallback

from .errors import Unsupported


@torch.library.custom_op("spyre::softplus", mutates_args=(), device_types="spyre")
def softplus(
    input: torch.Tensor, beta: float = 1.0, threshold: float = 20.0
) -> torch.Tensor:
    pass


@softplus.register_fake
def _(input: torch.Tensor, beta: float = 1.0, threshold: float = 20.0):
    return input.new_empty(input.size())


@torch.library.custom_op("spyre::layer_norm", mutates_args=())
def layer_norm(
    x: torch.Tensor,
    normalized_shape: list[int],
    weight: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
    eps: float = 1e-5,
) -> torch.Tensor:
    if len(normalized_shape) != 1:
        raise Unsupported(
            f"spyre.layernorm: unsupported reduction shape {normalized_shape}"
        )
    return torch.native_layer_norm(x, normalized_shape, weight, bias, eps)[0].clone()


@layer_norm.register_fake
def _(
    x: torch.Tensor,
    normalized_shape: list[int],
    weight: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
    eps: float = 1e-5,
):
    return x.new_empty(x.size())


@torch.library.custom_op("spyre::exx2", mutates_args=(), device_types="spyre")
def exx2(x: torch.Tensor, exx2Scale: float, useZeroMean: bool) -> torch.Tensor:  # type: ignore[empty-body]
    pass


@exx2.register_fake
def _(x: torch.Tensor, exx2Scale: float, useZeroMean: bool):
    return x.new_empty(x.size()[:-1])


@torch.library.custom_op("spyre::layernormscale", mutates_args=(), device_types="spyre")
def layernormscale(x: torch.Tensor, eps: float) -> torch.Tensor:  # type: ignore[empty-body]
    pass


@layernormscale.register_fake
def _(x: torch.Tensor, eps: float) -> torch.Tensor:
    return x.new_empty(x.size())


@torch.library.custom_op("spyre::layernormnorm", mutates_args=(), device_types="spyre")
def layernormnorm(  # type: ignore[empty-body]
    x: torch.Tensor,
    mean: torch.Tensor,
    norm_mean: torch.Tensor,
    weight: Optional[torch.Tensor],
    bias: Optional[torch.Tensor],
) -> torch.Tensor:
    pass


@layernormnorm.register_fake
def _(
    x: torch.Tensor,
    mean: torch.Tensor,
    norm_mean: torch.Tensor,
    weight: Optional[torch.Tensor],
    bias: Optional[torch.Tensor],
) -> torch.Tensor:
    return x.new_empty(x.size())


@torch.library.custom_op("spyre::rms_norm", mutates_args=())
def rms_norm(
    x: torch.Tensor,
    normalized_shape: list[int],
    weight: Optional[torch.Tensor] = None,
    eps: float = 1e-5,
) -> torch.Tensor:
    if len(normalized_shape) != 1:
        raise Unsupported(
            f"spyre.layernorm: unsupported reduction shape {normalized_shape}"
        )
    return torch.compile(torch.ops.spyre.rms_norm)(x, normalized_shape, weight, eps)


@rms_norm.register_fake
def _(
    x: torch.Tensor,
    normalized_shape: list[int],
    weight: Optional[torch.Tensor] = None,
    eps: float = 1e-5,
) -> torch.Tensor:
    return x.new_empty(x.size())


@torch.library.custom_op("spyre::topkvalue", mutates_args=(), device_types="spyre")
def topkvalue(x: torch.Tensor, k: int, dim: int) -> torch.Tensor:
    if len(x.size()) != 2:
        raise Unsupported("topk only implemented for 2-D tensors")
    pass


@topkvalue.register_fake
def _(x: torch.Tensor, k: int, dim: int) -> torch.Tensor:
    if len(x.size()) != 2:
        raise Unsupported("topk only implemented for 2-D tensors")
    norm_dim = dim % len(x.size())
    out_size = list(x.size())
    out_size[norm_dim] = k
    return x.new_empty(out_size)


@torch.library.custom_op("spyre::topkindex", mutates_args=(), device_types="spyre")
def topkindex(x: torch.Tensor, k: int, dim: int) -> torch.Tensor:
    if len(x.size()) != 2:
        raise Unsupported("topk only implemented for 2-D tensors")
    pass


@topkindex.register_fake
def _(x: torch.Tensor, k: int, dim: int) -> torch.Tensor:
    if len(x.size()) != 2:
        raise Unsupported("topk only implemented for 2-D tensors")
    norm_dim = dim % len(x.size())
    out_size = list(x.size())
    out_size[norm_dim] = k
    return x.new_empty(out_size, dtype=torch.int64)


@torch.library.custom_op("spyre::gelu", mutates_args=(), device_types="spyre")
def gelu(
    input: torch.Tensor,
    approximate: str = "none",
) -> torch.Tensor:
    pass


@gelu.register_fake
def _(input: torch.Tensor, approximate: str = "none"):
    return input.new_empty(input.size())


@torch.library.custom_op("spyre::clamp", mutates_args=(), device_types="spyre")
def clamp(
    input: torch.Tensor,
    min: Optional[torch.types.Number] = None,
    max: Optional[torch.types.Number] = None,
) -> torch.Tensor:
    pass


@clamp.register_fake
def _(
    input: torch.Tensor,
    min: Optional[torch.types.Number] = None,
    max: Optional[torch.types.Number] = None,
):
    return input.new_empty(input.size())


@torch.library.custom_op("spyre::empty", mutates_args=(), device_types="spyre")
def spyre_empty(
    size: Sequence[int],
    device: torch.device,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    # Eager-mode simulation: allocate on CPU and move to the Spyre device.
    # This is not a compute fallback — on hardware the compiled kernel receives
    # a device allocation from SpyreAllocator with no host-side initialisation.
    tmp = torch.empty(size, dtype=dtype, device="cpu")
    return tmp.to(device)


@spyre_empty.register_fake
def _(
    size: Sequence[int],
    device: torch.device,
    dtype: Optional[torch.dtype] = None,
):
    return torch.empty(size, dtype=dtype, device="spyre")


@torch.library.custom_op("spyre::logical_not", mutates_args=(), device_types="spyre")
def logical_not(input: torch.Tensor) -> torch.Tensor:
    pass


@logical_not.register_fake
def _(input: torch.Tensor):
    return input.new_empty(input.size())


@torch.library.custom_op(
    "spyre::copy_from_d2d", mutates_args=("dst",), device_types="spyre"
)
@compile_once("spyre.copy_from_d2d")
def copy_from_d2d(
    src: torch.Tensor,
    dst: torch.Tensor,
    compiled,
) -> None:
    return compiled(src, dst)


@copy_from_d2d.register_fake
def _(
    src: torch.Tensor,
    dst: torch.Tensor,
) -> None:
    pass


# Copy input into output starting at offsets along dimensions dims and
# return the updated output.
@torch.library.custom_op(
    "spyre::overwrite", mutates_args=("output",), device_types="spyre"
)
@compile_once("spyre.overwrite")
def overwrite(
    input: torch.Tensor,
    output: torch.Tensor,
    dims: Sequence[int],
    offsets: Sequence[int],
    compiled,
) -> None:
    # specialize_int=True installs int-equality guards on the int-list
    # args so each unique (dims, offsets) triggers a fresh trace and a
    # fresh SDSC binary; without this dynamo's default specialize_int=
    # False reuses one baked binary across all values and scatters all
    # writes to the first call's offset (see test_overwrite.py).
    # Patch is call-scoped to leave process-wide dynamo behavior alone.
    # Note: this gives one compiled binary per unique (input shape, dims,
    # offsets) tuple. dynamo's cache_size_limit is bumped to 1024 in
    # torch_spyre/__init__.py — long-running workloads that scatter into
    # many distinct slots can blow past that. Symbolic offsets (one
    # binary, any value) are tracked in issues #220 / #1371-3.
    with torch._dynamo.config.patch(specialize_int=True):
        return compiled(input, output, dims, offsets)


@overwrite.register_fake
def _(
    input: torch.Tensor,
    output: torch.Tensor,
    dims: Sequence[int],
    offsets: Sequence[int],
) -> None:
    return None


@torch.library.register_kernel("spyre::overwrite", ["cpu"])
def overwrite_cpu(
    input: torch.Tensor,
    output: torch.Tensor,
    dims: Sequence[int],
    offsets: Sequence[int],
) -> None:
    sliced_t = output
    for i, dim in enumerate(dims):
        sliced_t = torch.narrow(sliced_t, dim, offsets[i], input.size(dim))
    sliced_t.copy_(input)


@torch.library.custom_op("spyre::overwrite_f", mutates_args=(), device_types="spyre")
def overwrite_f(
    input: torch.Tensor,
    output: torch.Tensor,
    dims: Sequence[int],
    offsets: Sequence[int],
) -> torch.Tensor:
    result = output.clone()
    torch.ops.spyre.overwrite(input, result, dims, offsets)
    return result


@overwrite_f.register_fake
def _(
    input: torch.Tensor,
    output: torch.Tensor,
    dims: Sequence[int],
    offsets: Sequence[int],
) -> torch.Tensor:
    return output.clone()


inplaceable_ops[torch.ops.spyre.overwrite_f.default] = InplaceableOp(
    torch.ops.spyre.overwrite.default, 1
)


@torch.library.custom_op("spyre::restickify", mutates_args=(), device_types="spyre")
def restickify(  # type: ignore[empty-body]
    x: torch.Tensor,
) -> torch.Tensor:
    pass


@torch.library.custom_op("spyre::max_dim_int64_fallback", mutates_args=())
def max_dim_int64_fallback(
    input: torch.Tensor, dim: int, keepdim: bool = False
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    CPU fallback for torch.max(input, dim) when input is int64.
    This custom op will be registered with a CPU fallback in fallbacks.py.
    Returns a tuple (values, indices) as expected by torch.max.
    """
    # This should never be called directly; the fallback in fallbacks.py handles it
    raise RuntimeError(
        "spyre::max_dim_int64_fallback should be handled by CPU fallback registration"
    )


@max_dim_int64_fallback.register_fake
def _(input: torch.Tensor, dim: int, keepdim: bool = False):
    """
    Fake implementation for shape inference.
    Returns the expected output shapes for torch.max(input, dim, keepdim).
    """
    # Compute output shape based on dim and keepdim
    if keepdim:
        output_shape = list(input.size())
        output_shape[dim] = 1
    else:
        output_shape = list(input.size())
        output_shape.pop(dim)

    # Return tuple of (values, indices) with the computed shape
    values = input.new_empty(output_shape)
    indices = torch.empty(output_shape, dtype=torch.int64, device=input.device)
    return (values, indices)


@torch.library.custom_op("spyre::min_dim_int64_fallback", mutates_args=())
def min_dim_int64_fallback(
    input: torch.Tensor, dim: int, keepdim: bool = False
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    CPU fallback for torch.min(input, dim) when input is int64.
    This custom op will be registered with a CPU fallback in fallbacks.py.
    Returns a tuple (values, indices) as expected by torch.min.
    """
    raise RuntimeError(
        "spyre::min_dim_int64_fallback should be handled by CPU fallback registration"
    )


@min_dim_int64_fallback.register_fake
def _(input: torch.Tensor, dim: int, keepdim: bool = False):
    """
    Fake implementation for shape inference.
    Returns the expected output shapes for torch.min(input, dim, keepdim).
    """
    if keepdim:
        output_shape = list(input.size())
        output_shape[dim] = 1
    else:
        output_shape = list(input.size())
        output_shape.pop(dim)

    values = input.new_empty(output_shape)
    indices = torch.empty(output_shape, dtype=torch.int64, device=input.device)
    return (values, indices)


## TODO (imaihal): This needs scalar tensor support from Spyre to CPU. issues #1172
#
# @torch.library.custom_op("spyre::max_default_int64_fallback", mutates_args=())
# def max_default_int64_fallback(input: torch.Tensor) -> torch.Tensor:
#    """
#    CPU fallback for torch.max(input) when input is int64.
#    This custom op will be registered with a CPU fallback in fallbacks.py.
#    Returns a 1D tensor with shape [1] containing the maximum value.
#    """
#    # This should never be called directly; the fallback in fallbacks.py handles it
#    raise RuntimeError(
#        "spyre::max_default_int64_fallback should be handled by CPU fallback registration"
#    )
#
#
# @max_default_int64_fallback.register_fake
# def _(input: torch.Tensor):
#    """
#    Fake implementation for shape inference.
#    Returns a scalar (0D) tensor matching the input dtype.
#    """
#    return input.new_empty([])


@torch.library.custom_op("spyre::batched_matmul", mutates_args=(), device_types="spyre")
def batched_matmul(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:  # type: ignore[empty-body]
    pass


@batched_matmul.register_fake
def _(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    output_shape = list(x.shape[:-1]) + [y.shape[-1]]
    return x.new_empty(output_shape)


@torch.library.custom_op("spyre::constant", mutates_args=(), device_types="spyre")
def spyre_constant(
    fill_value: torch.types.Number, dtype: torch.dtype, device: torch.device
) -> torch.types.Number:
    # This custom operator marks scalar constant in the FX graph.
    # Returning the scalar constant to avoid change in the operator schema which
    # consume the scalar constant as input.
    # This node will have a special handling at lowering to convert the scalar
    # constant to tensor.
    return fill_value


@spyre_constant.register_fake
def _constant(
    fill_value: torch.types.Number, dtype: torch.dtype, device: torch.device
) -> torch.types.Number:
    return fill_value


@torch.library.custom_op("spyre::to_dtype_cpu", mutates_args=(), device_types="spyre")
def to_dtype_cpu(input: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    warn_fallback(f"conversion from {input.dtype} to {dtype}")
    return input.cpu().to(dtype=dtype).to(input.device)


@to_dtype_cpu.register_fake
def _(input: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    return torch.empty_like(input, dtype=dtype)

# ============================================================================
# FP8 Quantization Operations
# ============================================================================
# These operations map to deeptools FP8 operations for efficient quantization
# on Spyre hardware.


@torch.library.custom_op("spyre::quantize_fp8", mutates_args=(), device_types="spyre")
def quantize_fp8(input: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    High-level FP8 quantization operation.
    
    Implements the 5-step quantization flow:
    1. quantscalepertokenfp8 - Compute scale = max(abs(x)) / 448 (REDUCTION)
    2. reciprocal - Compute inv_scale = 1 / scale (POINTWISE, sfp unit)
    3. mul - x_scaled = x * inv_scale (POINTWISE)
    4. clamp - x_clamped = clamp(x_scaled, -448, 448) (POINTWISE)
    5. qfp8 - x_fp8 = qfp8(x_clamped) (POINTWISE format conversion)
    
    Args:
        input: Input tensor (FP16/FP32/BF16) to quantize
    
    Returns:
        tuple: (fp8_tensor, scale) where:
            - fp8_tensor: FP8 E4M3 quantized tensor
            - scale: Quantization scale (needed for dequantization)
    
    Example:
        >>> x = torch.randn(4, 512, 4096, device="spyre")
        >>> x_fp8, scale = torch.ops.spyre.quantize_fp8(x)
        >>> # Later for dequantization:
        >>> # x_dequant = x_fp8.to(fp16) * scale
    
    Note:
        All operations map to deeptools hardware primitives for maximum efficiency.
    """
    pass


@quantize_fp8.register_fake
def _(input: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    # Return tuple: (FP8 tensor, scale tensor)
    fp8_tensor = torch.empty(input.size(), dtype=torch.float8_e4m3fn, device=input.device)
    # Scale has shape [..., 1] (reduced along last dimension)
    scale_shape = list(input.size())
    scale_shape[-1] = 1
    scale_tensor = torch.empty(scale_shape, dtype=torch.float32, device=input.device)
    return (fp8_tensor, scale_tensor)


@torch.library.custom_op("spyre::quantscalepertokenfp8", mutates_args=(), device_types="spyre")
def quantscalepertokenfp8(input: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    Compute FP8 quantization scale per token (REDUCTION operation).
    
    Computes: scale = max(abs(x)) / 448
    
    This is a REDUCTION operation that reduces along the specified dimension (token dimension).
    For every coordinate along the specified dimension, it finds the scale.
    Similar to exx2 operation - uses hardware reduction, not pointwise.
    
    Args:
        input: Input tensor (FP16/FP32/BF16) to compute scales for
        dim: Dimension along which to compute scales (default: -1, last dimension)
             This is the "token dimension" - the dimension that will be reduced.
    
    Returns:
        Scale tensor (FP32) with the specified dimension reduced to size 1
    
    Maps to: deeptools QuantScalePerTokenFP8 (REDUCTION operation)
    
    Example:
        >>> x = torch.randn(4, 512, 4096, device="spyre")
        >>> # Reduce along last dimension (default, typical for tokens)
        >>> scale = torch.ops.spyre.quantscalepertokenfp8(x)
        >>> # scale.shape = [4, 512, 1]
        >>>
        >>> # Or specify dimension explicitly
        >>> scale = torch.ops.spyre.quantscalepertokenfp8(x, dim=-1)
        >>> # scale.shape = [4, 512, 1]
    
    Note:
        The dim parameter specifies which dimension to reduce when computing scales.
        For each coordinate along the specified dimension, a scale value is computed.
    """
    pass


@quantscalepertokenfp8.register_fake
def _(input: torch.Tensor, dim: int = -1) -> torch.Tensor:
    # Normalize dimension to positive index
    if dim < 0:
        dim = input.ndim + dim
    
    # Scale tensor has the specified dimension reduced to 1
    scale_shape = list(input.size())
    scale_shape[dim] = 1
    return torch.empty(scale_shape, dtype=torch.float32, device=input.device)


@torch.library.custom_op("spyre::reciprocal", mutates_args=(), device_types="spyre")
def reciprocal(input: torch.Tensor) -> torch.Tensor:
    """
    Compute reciprocal: 1 / x (POINTWISE operation).
    
    Hardware-optimized reciprocal operation executed on sfp (special function processor) unit.
    Used to convert scale to inverse scale for efficient multiplication.
    
    Args:
        input: Input tensor to compute reciprocal of
    
    Returns:
        Reciprocal tensor (same shape and dtype as input)
    
    Maps to: deeptools reciprocal (POINTWISE operation, sfp unit)
    
    Example:
        >>> scale = torch.tensor([2.0, 4.0, 8.0], device="spyre")
        >>> inv_scale = torch.ops.spyre.reciprocal(scale)
        >>> # inv_scale = [0.5, 0.25, 0.125]
    
    Note:
        Multiplication by reciprocal is faster than division on hardware.
    """
    pass


@reciprocal.register_fake
def _(input: torch.Tensor) -> torch.Tensor:
    # Output has same shape and dtype as input
    return torch.empty(input.size(), dtype=input.dtype, device=input.device)


@torch.library.custom_op("spyre::qfp8", mutates_args=(), device_types="spyre")
def qfp8(input: torch.Tensor) -> torch.Tensor:
    """
    FP8 format conversion operation (POINTWISE).
    
    Converts input tensor from FP16/FP32/BF16 to FP8 E4M3 format.
    This operation ONLY performs format conversion - it does NOT apply scales.
    
    Input should already be scaled and clamped to [-448, 448] range.
    
    Args:
        input: Input tensor (FP16/FP32/BF16) to convert to FP8
               Should already be scaled and clamped
    
    Returns:
        FP8 E4M3 tensor (same shape as input)
    
    Maps to: deeptools Qfp8 operation (POINTWISE format conversion)
    
    Example:
        >>> # Correct usage (after scaling and clamping):
        >>> scale = torch.ops.spyre.quantscalepertokenfp8(x)
        >>> inv_scale = torch.ops.spyre.reciprocal(scale)
        >>> x_scaled = x * inv_scale
        >>> x_clamped = torch.clamp(x_scaled, -448.0, 448.0)
        >>> x_fp8 = torch.ops.spyre.qfp8(x_clamped)
    """
    pass


@qfp8.register_fake
def _(input: torch.Tensor) -> torch.Tensor:
    # Output is FP8 with same shape as input
    return torch.empty(input.size(), dtype=torch.float8_e4m3fn, device=input.device)


@torch.library.custom_op("spyre::qfp8ch", mutates_args=(), device_types="spyre")
def qfp8ch(input: torch.Tensor) -> torch.Tensor:
    """
    Channel-wise FP8 format conversion (pointwise, optimized for matmul).
    
    Converts input tensor to FP8 E4M3 format with channel-wise semantics.
    This operation ONLY performs format conversion - scaling must be done separately.
    
    Args:
        input: Input tensor (FP16/FP32/BF16) to convert to FP8
               Should already be scaled and clamped
    
    Returns:
        FP8 E4M3 tensor (same shape as input)
    
    Maps to: deeptools Qfp8ch operation
    """
    pass


@qfp8ch.register_fake
def _(input: torch.Tensor) -> torch.Tensor:
    # Output is FP8 with same shape as input
    return torch.empty(input.size(), dtype=torch.float8_e4m3fn, device=input.device)


@torch.library.custom_op("spyre::qfp8chil", mutates_args=(), device_types="spyre")
def qfp8chil(input: torch.Tensor) -> torch.Tensor:
    """
    Channel-wise interleaved FP8 format conversion (pointwise).

    Converts input tensor to FP8 E4M3 format with channel-wise interleaved semantics.
    This operation ONLY performs format conversion - scaling must be done separately.

    Args:
        input: Input tensor (FP16/FP32/BF16) to convert to FP8
               Should already be scaled and clamped

    Returns:
        FP8 E4M3 tensor (same shape as input)

    Maps to: deeptools Qfp8chil operation
    """
    pass


@qfp8chil.register_fake
def _(input: torch.Tensor) -> torch.Tensor:
    # Output is FP8 with same shape as input
    return torch.empty(input.size(), dtype=torch.float8_e4m3fn, device=input.device)




@torch.library.custom_op("spyre::qfp8mb", mutates_args=(), device_types="spyre")
def qfp8mb(input: torch.Tensor) -> torch.Tensor:
    """
    Mini-batch FP8 format conversion (pointwise).
    
    Converts input tensor to FP8 E4M3 format with mini-batch semantics.
    This operation ONLY performs format conversion - scaling must be done separately.
    
    Args:
        input: Input tensor (FP16/FP32/BF16) to convert to FP8
               Should already be scaled and clamped
    
    Returns:
        FP8 E4M3 tensor (same shape as input)
    
    Maps to: deeptools Qfp8mb operation
    """
    pass


@qfp8mb.register_fake
def _(input: torch.Tensor) -> torch.Tensor:
    # Output is FP8 with same shape as input
    return torch.empty(input.size(), dtype=torch.float8_e4m3fn, device=input.device)


@torch.library.custom_op("spyre::qfp8wt", mutates_args=(), device_types="spyre")
def qfp8wt(input: torch.Tensor) -> torch.Tensor:
    """
    Weight FP8 format conversion (pointwise).
    
    Converts weight tensor to FP8 E4M3 format, optimized for model weights.
    This operation ONLY performs format conversion - scaling must be done separately.
    
    Args:
        input: Weight tensor (FP16/FP32/BF16) to convert to FP8
               Should already be scaled and clamped
    
    Returns:
        FP8 E4M3 weight tensor (same shape as input)
    
    Maps to: deeptools Qfp8wt operation
    """
    pass


@qfp8wt.register_fake
def _(input: torch.Tensor) -> torch.Tensor:
    # Output is FP8 with same shape as input
    return torch.empty(input.size(), dtype=torch.float8_e4m3fn, device=input.device)
