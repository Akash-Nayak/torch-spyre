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

import torch

BATCH_MATMUL_OP = "batchmatmul"
IDENTITY_OP = "identity"
RESTICKIFY_OP = "ReStickifyOpHBM"

# Type casting operators from deeptools
DL16TOFP32_OP = "dl16tofp32"
FP32TODL16_OP = "fp32todl16"

DEVICE_NAME = "spyre"


SEGMENT_OFFSETS = [
    0x0,
    0x400000000,
    0x800000000,
    0xC00000000,
    0x1000000000,
    0x1400000000,
    0x1800000000,
]

INTERMEDIATES_SEGMENT = 0x0
SEGMENT_SIZE = 0x400000000

SPYRE_FP32_OPS = [
    "add",
    "sub",
    "mul",
    "where",
    "realdiv",
    "relufwd",
    "reciprocal",
    "layernormscale",
    "abs",
    "neg",
    "exp",
    "sigmoid",
    "exx2",
    "layernormnorm",
    "identity",
    "topkvalue",
    "topkindex",
    "floor",
    "to_dtype",
    "maximum",
    "minimum",
]

TOPK_OPS = {"topkvalue", "topkindex"}

LAYOUT_LABELS = ["OUTPUT", "KERNEL", "INPUT", "KERNEL_IDX"]
MATMUL_LAYOUT_LABELS = ["INPUT", "KERNEL", "OUTPUT", "KERNEL_IDX"]


# Populate more valid labels from deeptools here if needed
INPUT_DIM_LABELS = ["mb", "x", "y", "i", "j", "ki", "kj"]
OUTPUT_DIM_LABELS = ["out"]
MATMUL_DIM_LABELS = ["ki", "kj", "y", "x", "mb", "out", "in"]

# FP8 Support
# Maps PyTorch FP8 dtype to deeptools SEN143_FP8 format
SUPPORTED_DTYPES = {
    torch.float16,
    torch.bfloat16,
    torch.float32,
    torch.float8_e4m3fn,  # FP8 E4M3 format
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
}

# Map PyTorch dtypes to deeptools/sendnn dtypes
TORCH_TO_SENDNN_DTYPE = {
    torch.float16: "float16",
    torch.bfloat16: "bfloat16",
    torch.float32: "float32",
    torch.float8_e4m3fn: "SEN143_FP8",  # FP8 format for deeptools
    torch.int8: "int8",
    torch.int16: "int16",
    torch.int32: "int32",
    torch.int64: "int64",
}

# FP8 quantization operations (map to deeptools ops)
FP8_QUANTIZATION_OPS = {
    "qfp8",           # Basic FP8 quantization
    "qfp8ch",         # Channel-wise FP8 quantization (for matmul)
    "qfp8mb",         # Mini-batch FP8 quantization
    "qfp8wt",         # Weight FP8 quantization
    "quantscalepertokenfp8",  # Compute FP8 scales per token
}
