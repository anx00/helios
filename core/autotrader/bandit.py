"""
Contextual bandit (LinUCB) for strategy selection.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Iterable, List, Optional, Tuple
from zoneinfo import ZoneInfo

import numpy as np

from .models import BanditState

UTC = ZoneInfo("UTC")


@dataclass
class LinUCBSelection:
    strategy_name: str
    scores: Dict[str, float]


class LinUCBBandit:
    """
    Minimal LinUCB implementation:
    - one linear model per strategy
    - optimistic score: theta^T x + alpha * sqrt(x^T A^-1 x)
    """

    def __init__(self, strategies: Iterable[str], feature_dim: int, alpha: float = 0.8):
        self.strategies = list(strategies)
        self.feature_dim = int(feature_dim)
        self.alpha = float(alpha)
        self._A: Dict[str, np.ndarray] = {}
        self._b: Dict[str, np.ndarray] = {}

        for name in self.strategies:
            self._A[name] = np.eye(self.feature_dim, dtype=np.float64)
            self._b[name] = np.zeros(self.feature_dim, dtype=np.float64)

    def reset(self) -> None:
        for name in self.strategies:
            self._A[name] = np.eye(self.feature_dim, dtype=np.float64)
            self._b[name] = np.zeros(self.feature_dim, dtype=np.float64)

    def select(
        self,
        feature_vector: List[float],
        candidates: Optional[Iterable[str]] = None,
    ) -> LinUCBSelection:
        x = np.asarray(feature_vector, dtype=np.float64)
        if x.shape[0] != self.feature_dim:
            raise ValueError(f"feature_dim mismatch: expected {self.feature_dim}, got {x.shape[0]}")

        cands = list(candidates) if candidates is not None else self.strategies
        if not cands:
            raise ValueError("No candidate strategies provided")

        scores: Dict[str, float] = {}
        best_name = cands[0]
        best_score = float("-inf")

        for name in cands:
            A = self._A[name]
            b = self._b[name]
            A_inv = np.linalg.inv(A)
            theta = A_inv @ b
            exploit = float(theta.T @ x)
            explore = float(self.alpha * np.sqrt(x.T @ A_inv @ x))
            score = exploit + explore
            scores[name] = score
            if score > best_score:
                best_name = name
                best_score = score

        return LinUCBSelection(strategy_name=best_name, scores=scores)

    def update(self, strategy_name: str, feature_vector: List[float], reward: float) -> None:
        if strategy_name not in self._A:
            return

        x = np.asarray(feature_vector, dtype=np.float64)
        if x.shape[0] != self.feature_dim:
            return

        self._A[strategy_name] += np.outer(x, x)
        self._b[strategy_name] += float(reward) * x

    def to_state(self) -> BanditState:
        state = {
            "A": {k: v.tolist() for k, v in self._A.items()},
            "b": {k: v.tolist() for k, v in self._b.items()},
        }
        return BanditState(
            version="linucb_v1",
            alpha=self.alpha,
            feature_dim=self.feature_dim,
            strategies=self.strategies,
            state=state,
            updated_at_utc=datetime.now(UTC),
        )

    @classmethod
    def from_state(cls, state: BanditState) -> "LinUCBBandit":
        bandit = cls(
            strategies=state.strategies,
            feature_dim=state.feature_dim,
            alpha=state.alpha,
        )
        A = state.state.get("A", {})
        b = state.state.get("b", {})

        for name in bandit.strategies:
            if name in A:
                bandit._A[name] = np.asarray(A[name], dtype=np.float64)
            if name in b:
                bandit._b[name] = np.asarray(b[name], dtype=np.float64)
        return bandit
