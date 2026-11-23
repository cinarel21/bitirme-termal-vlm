"""Muhakeme bütçesi tanımları: kararlı minimum yeterli bütçe (B*) ve kararsızlık.

Bu dosya modelden bağımsızdır; CPU'da test edilebilir.
"""
from __future__ import annotations

from typing import Mapping, Optional, Sequence

DEFAULT_BUDGETS: tuple[int, ...] = (0, 32, 64, 128, 256, 512)


def stable_bstar(correct_by_budget: Mapping[int, bool],
                 budgets: Sequence[int] = DEFAULT_BUDGETS) -> Optional[int]:
    """Kararın doğru olduğu ve daha büyük HİÇBİR bütçede bozulmadığı en küçük bütçe.

    Örnek: {0: F, 32: T, 64: F, 128: T, 256: T, 512: T} -> 128
    Hiçbir bütçe bu koşulu sağlamıyorsa None döner (B* > max bütçe).
    """
    ordered = sorted(b for b in budgets if b in correct_by_budget)
    if not ordered:
        raise ValueError("Hiç bütçe sonucu yok.")
    bstar: Optional[int] = None
    # Sondan başa yürü: doğru kaldığı sürece B* geriye kayar, ilk yanlışta dur.
    for b in reversed(ordered):
        if correct_by_budget[b]:
            bstar = b
        else:
            break
    return bstar


def is_unstable(correct_by_budget: Mapping[int, bool],
                budgets: Sequence[int] = DEFAULT_BUDGETS) -> bool:
    """Bir bütçede doğru olup daha büyük bir bütçede yanlışa dönen aday var mı?"""
    ordered = [correct_by_budget[b] for b in sorted(budgets) if b in correct_by_budget]
    seen_correct = False
    for ok in ordered:
        if ok:
            seen_correct = True
        elif seen_correct:
            return True
    return False


def bstar_distribution(rows: Sequence[Mapping[int, bool]],
                       budgets: Sequence[int] = DEFAULT_BUDGETS) -> dict[str, float]:
    """B* değerlerinin oranlarını döndürür. Anahtar 'never' -> B* > max bütçe."""
    if not rows:
        return {}
    counts: dict[str, int] = {str(b): 0 for b in sorted(budgets)}
    counts["never"] = 0
    for r in rows:
        b = stable_bstar(r, budgets)
        counts["never" if b is None else str(b)] += 1
    n = len(rows)
    return {k: v / n for k, v in counts.items()}


def instability_rate(rows: Sequence[Mapping[int, bool]],
                     budgets: Sequence[int] = DEFAULT_BUDGETS) -> float:
    if not rows:
        return 0.0
    return sum(is_unstable(r, budgets) for r in rows) / len(rows)
