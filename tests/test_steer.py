import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tvlm import steer as S  # noqa: E402


def test_compute_direction_unit_and_sign():
    pos = np.array([[2.0, 0.0], [4.0, 0.0]])
    neg = np.array([[0.0, 0.0], [0.0, 0.0]])
    d = S.compute_direction(pos, neg)
    assert np.allclose(d, [1.0, 0.0])


def test_parse_decision_takes_last():
    assert S.parse_decision("hmm DECISION: NO ... wait\nDECISION: YES") == 1
    assert S.parse_decision("decision : no") == 0
    assert S.parse_decision("I am not sure") is None


class _Layer(torch.nn.Module):
    def forward(self, x):
        return (x,)


def test_residual_steering_generated_only():
    layers = torch.nn.ModuleList([_Layer(), _Layer()])
    v = torch.tensor([1.0, 0.0])
    steer = S.ResidualSteer(1, v, alpha=0.5, where="generated")
    prefill = torch.ones(1, 3, 2)
    step = torch.ones(1, 1, 2)
    with S.residual_steering(layers, [steer]):
        a = layers[1](prefill.clone())[0]
        b = layers[1](step.clone())[0]
    assert torch.allclose(a, prefill)                 # prefill'e dokunulmaz
    scale = step.norm(dim=-1).mean()
    assert torch.allclose(b[0, 0], step[0, 0] + 0.5 * scale * v)
    c = layers[1](step.clone())[0]                    # bağlamdan çıkınca kanca kalkar
    assert torch.allclose(c, step)


def test_residual_steering_image_positions():
    layers = torch.nn.ModuleList([_Layer()])
    v = torch.tensor([0.0, 1.0])
    pos = torch.tensor([1, 2])
    x = torch.zeros(1, 4, 2) + 1.0
    with S.residual_steering(layers, [S.ResidualSteer(0, v, 1.0, "image", pos)]):
        y = layers[0](x.clone())[0]
    assert torch.allclose(y[0, 0], x[0, 0]) and torch.allclose(y[0, 3], x[0, 3])
    assert (y[0, 1, 1] > x[0, 1, 1]) and (y[0, 2, 1] > x[0, 2, 1])


class _Attn(torch.nn.Module):
    def __init__(self, idx):
        super().__init__()
        self.layer_idx = idx


def _ref_attention(q, k, v):
    n = q.shape[1] // k.shape[1]
    k = k.repeat_interleave(n, dim=1)
    v = v.repeat_interleave(n, dim=1)
    w = torch.softmax(q @ k.transpose(2, 3) * q.shape[-1] ** -0.5, -1)
    return (w @ v).transpose(1, 2), w


def test_visual_boost_matches_eager_when_inactive_and_boosts_when_active():
    torch.manual_seed(0)
    q = torch.randn(1, 4, 1, 8)       # 4 sorgu başı, tek üretim adımı
    k = torch.randn(1, 2, 10, 8)      # 2 anahtar başı (GQA)
    v = torch.randn(1, 2, 10, 8)
    mod = _Attn(3)
    out, w = S.visual_boost_attention(mod, q, k, v, None)
    ref_out, ref_w = _ref_attention(q, k, v)
    assert torch.allclose(out, ref_out, atol=1e-5) and torch.allclose(w, ref_w, atol=1e-5)

    img = torch.tensor([2, 3, 4, 5])
    with S.visual_boost(2.0, [3], img):
        _, w2 = S.visual_boost_attention(mod, q, k, v, None)
        _, w_other = S.visual_boost_attention(_Attn(0), q, k, v, None)
    share = lambda ww: ww[..., img].sum(-1).mean().item()
    assert share(w2) > share(w) and torch.allclose(w_other, w, atol=1e-6)
    # log(2) önyargısı: görüntü olasılık kütlesinin odds'u tam 2 katına çıkar
    p, p2 = w[0, 0, 0, img].sum(), w2[0, 0, 0, img].sum()
    assert abs((p2 / (1 - p2)) / (p / (1 - p)) - 2.0) < 1e-4
