"""fused_attn: fused multi-head attention kernels for LLM inference (CUDA C++ + Triton)."""
from .ops import attention, attention_varlen, available_backends, pack_varlen
from .reference import attention_reference, attention_varlen_reference

__all__ = [
    "attention",
    "attention_varlen",
    "attention_reference",
    "attention_varlen_reference",
    "available_backends",
    "pack_varlen",
]
__version__ = "0.2.0"
