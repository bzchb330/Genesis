from __future__ import annotations
from typing import Protocol
import numpy as np

class RetentionController(Protocol):
    """Future tactile closed-loop retention interface using existing signals only."""
    def residual(self, q: np.ndarray, qdot: np.ndarray, tactile: dict[str, np.ndarray], phase: int) -> np.ndarray: ...

class ZeroRetentionController:
    """Engineering placeholder that applies no retention residual."""
    def residual(self, q: np.ndarray, qdot: np.ndarray, tactile: dict[str, np.ndarray], phase: int) -> np.ndarray:
        # TODO(PI): define the closed-loop tactile retention law.
        return np.zeros_like(np.asarray(q, dtype=float))
