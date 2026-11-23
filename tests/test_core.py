import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tvlm.budget import bstar_distribution, instability_rate, is_unstable, stable_bstar
from tvlm.geometry import Box, context_crop, iou, random_negative_crop, size_bin, token_grid_mask
from tvlm.probe import early_exit

B = (0, 32, 64, 128, 256, 512)


def r(*vals):
    return dict(zip(B, vals))


def test_bstar_stable_after_flip():
    assert stable_bstar(r(False, True, False, True, True, True)) == 128


def test_bstar_never():
    assert stable_bstar(r(False, True, False, False, False, False)) is None


def test_bstar_zero():
    assert stable_bstar(r(True, True, True, True, True, True)) == 0


def test_instability():
    assert is_unstable(r(False, True, False, True, True, True))
    assert not is_unstable(r(False, False, True, True, True, True))
    rows = [r(True, True, True, True, True, True), r(False, True, False, True, True, True)]
    assert instability_rate(rows) == 0.5
    d = bstar_distribution(rows)
    assert d["0"] == 0.5 and d["128"] == 0.5 and abs(sum(d.values()) - 1) < 1e-9


def test_iou_and_bins():
    assert iou(Box(0, 0, 10, 10), Box(0, 0, 10, 10)) == 1.0
    assert iou(Box(0, 0, 10, 10), Box(20, 20, 5, 5)) == 0.0
    assert size_bin(Box(0, 0, 8, 8)) == "tiny"
    assert size_bin(Box(0, 0, 40, 40)) == "medium"


def test_context_crop_inside_image():
    c = context_crop(Box(5, 5, 10, 10), 640, 512, context=4, min_side=112)
    assert c.x >= 0 and c.y >= 0 and c.x + c.w <= 640 and c.y + c.h <= 512
    assert c.w == c.h == 112


def test_negative_crop_no_overlap():
    rng = random.Random(0)
    drone = Box(300, 200, 20, 20)
    for _ in range(20):
        n = random_negative_crop(112, 640, 512, drone, rng)
        assert n is not None and iou(n, drone) == 0.0


def test_token_grid_mask():
    # 112 px kırpma 448'e büyütülür (x4); 28 px/token -> 16x16 ızgara
    m = token_grid_mask(Box(42, 42, 28, 28), 112, 112, 448, 448, 28)
    assert m.shape == (16, 16)
    rows, cols = np.where(m)
    assert rows.min() == 6 and rows.max() == 9 and cols.min() == 6 and cols.max() == 9


def test_early_exit_policy():
    probs = np.array([[0.5, 0.99, 0.99], [0.5, 0.6, 0.7]])
    out = early_exit(probs, (0, 32, 64), 0.95, np.array([1, 0]), np.array([64, 64]))
    assert out["tokens"].tolist() == [32, 64]
    assert out["decisions"].tolist() == [1, 0]
