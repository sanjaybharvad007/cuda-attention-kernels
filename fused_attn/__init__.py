"""fused_attn: fused multi-head attention kernels for LLM inference (CUDA C++ + Triton)."""
from .ops import attention, attention_varlen, available_backends, pack_varlen
from .reference import attention_reference, attention_varlen_reference

try:
    from .fp8 import attention_fp8_triton, dequantize_kv_fp8, quantize_kv_fp8
    from .triton_splitk import attention_splitk_triton
except Exception:  # pragma: no cover - triton missing
    attention_fp8_triton = attention_splitk_triton = quantize_kv_fp8 = dequantize_kv_fp8 = None

__all__ = [
    "attention",
    "attention_varlen",
    "attention_reference",
    "attention_varlen_reference",
    "available_backends",
    "pack_varlen",
    "attention_splitk_triton",
    "attention_fp8_triton",
    "quantize_kv_fp8",
    "dequantize_kv_fp8",
]
__version__ = "0.3.0"
