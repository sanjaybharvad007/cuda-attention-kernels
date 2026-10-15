"""The op is a torch.library custom op: torch.compile must trace through it
without a graph break and the fake kernel must give correct shapes/dtypes."""
import pytest
import torch

import fused_attn


def test_fake_tensor_shape():
    from torch._subclasses.fake_tensor import FakeTensorMode

    with FakeTensorMode():
        q = torch.empty(2, 8, 128, 64, dtype=torch.float16)
        k = torch.empty(2, 2, 128, 64, dtype=torch.float16)
        out = fused_attn.attention(q, k, k, causal=True)
    assert out.shape == (2, 8, 128, 64) and out.dtype == torch.float16


def test_opcheck():
    q, k, v = (torch.randn(1, 2, 16, 32) for _ in range(3))
    torch.library.opcheck(fused_attn.attention, (q, k, v, True, None))


@pytest.mark.skipif(not hasattr(torch, "compile"), reason="no torch.compile")
def test_torch_compile_no_graph_break():
    torch._dynamo.reset()

    def f(q, k, v):
        return fused_attn.attention(q, k, v, causal=True) * 2.0

    q, k, v = (torch.randn(1, 2, 16, 32) for _ in range(3))
    try:
        compiled = torch.compile(f, fullgraph=True)
        out = compiled(q, k, v)
    except Exception as e:  # pragma: no cover - missing C compiler for inductor on some CI images
        if "compiler" in str(e).lower() or "inductor" in str(e).lower():
            pytest.skip(f"inductor unavailable: {e}")
        raise
    torch.testing.assert_close(out, f(q, k, v), atol=1e-5, rtol=1e-5)
