"""Doğrusal probe ve iç sinyalle erken durdurma.

Fikir: muhakemenin k. tokenındaki gizli durum, modelin vereceği kararı zaten
taşıyorsa, k'dan sonraki tokenlar boşa harcanıyordur. Probe bu kararı okur;
yeterince eminse muhakeme durdurulur.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

CHECKPOINTS: tuple[int, ...] = (0, 16, 32, 64, 128, 256, 512)


def collect_features(hidden_states, prompt_len: int, reasoning_len: int,
                     layers: Sequence[int], checkpoints: Sequence[int] = CHECKPOINTS) -> np.ndarray:
    """Tek bir ileri geçişin hidden_states çıktısından (checkpoint, katman, d) özellik çıkarır.

    k. checkpoint'te 'k muhakeme tokenı okunduktan sonraki' son konum alınır.
    Muhakeme k'dan kısa bittiyse son muhakeme konumu kullanılır.
    """
    feats = []
    for k in checkpoints:
        pos = prompt_len + min(k, reasoning_len) - 1
        feats.append(np.stack([hidden_states[l + 1][0, pos].float().cpu().numpy() for l in layers]))
    return np.stack(feats)


@dataclass
class ProbeResult:
    accuracy: np.ndarray  # (checkpoint, katman)
    auroc: np.ndarray


def train_probes(X_train: np.ndarray, y_train: np.ndarray,
                 X_eval: np.ndarray, y_eval: np.ndarray, C: float = 0.1):
    """X: (örnek, checkpoint, katman, d). Her (checkpoint, katman) için ayrı lojistik regresyon."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline

    _, n_ck, n_l, _ = X_train.shape
    acc = np.zeros((n_ck, n_l))
    auc = np.zeros((n_ck, n_l))
    models = {}
    for c in range(n_ck):
        for l in range(n_l):
            clf = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=2000))
            clf.fit(X_train[:, c, l], y_train)
            p = clf.predict_proba(X_eval[:, c, l])[:, 1]
            acc[c, l] = ((p > 0.5) == y_eval).mean()
            auc[c, l] = roc_auc_score(y_eval, p) if len(set(y_eval)) > 1 else np.nan
            models[(c, l)] = clf
    return ProbeResult(acc, auc), models


def early_exit(probs: np.ndarray, checkpoints: Sequence[int], threshold: float,
               full_decisions: np.ndarray, reasoning_lens: np.ndarray) -> dict[str, np.ndarray]:
    """Probe olasılıklarıyla erken durdurma politikası.

    probs: (örnek, checkpoint) — drone olasılığı
    full_decisions: (örnek,) — tam muhakeme sonunda modelin kararı (0/1)
    Kural: max(p, 1-p) >= threshold olan ilk checkpoint'te dur ve probe kararını ver;
    hiç ulaşılmazsa tam muhakemenin kararını kullan.
    Dönüş: örnek başına harcanan token ve verilen karar.
    """
    n = probs.shape[0]
    tokens = np.zeros(n)
    decisions = np.zeros(n, dtype=int)
    for i in range(n):
        tokens[i] = reasoning_lens[i]
        decisions[i] = full_decisions[i]
        for j, k in enumerate(checkpoints):
            if k > reasoning_lens[i]:
                break
            conf = max(probs[i, j], 1 - probs[i, j])
            if conf >= threshold:
                tokens[i] = k
                decisions[i] = int(probs[i, j] >= 0.5)
                break
    return {"tokens": tokens, "decisions": decisions}
