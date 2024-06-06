"""Prediction Game (PG) implementation."""
from typing import Any

import numpy as np

from .base import FinancialGame
from .fundamental import Fundamental


class Expert(Fundamental):
    """Expert aka fundamental trader with arbitrary fundamental price.

    It implements only ``fixed-raw`` model.

    Attributes
    ----------
    N
        Number of agents.
    W
        Length of price history considered by traders.
    M
        Total memory length, which should be equal to ``W+1``.
    X
        Constant factor for inverse proportional rescaling
        of price fluctuations.
    P
        2D array with price histories seen by different traders.
    fundamental_prob
        Fundamental probability belief of experts.
    """
    def __init__(
        self,
        *args: Any,
        fundamental_prob: float = .5,
        **kwargs: Any
    ) -> None:
        super().__init__(*args, model="fixed-raw", **kwargs)
        if fundamental_prob < 0 or fundamental_prob > 1:
            errmsg = "'fundamental_prob' must be between 0 and 1"
            raise ValueError(errmsg)
        self.fundamental_prob = fundamental_prob

    def get_gamma(self) -> np.ndarray[tuple[int], np.floating]:
        r""":math:`\gamma(t)` factor."""
        i = self.game._init_history_size+self.game.n_step-1
        price = self.game._price[i]
        prob  = price / (price + 1)
        gamma = (prob - self.fundamental_prob) / self.X
        return gamma


class PredictionGame(FinancialGame):
    """Prediction game class."""

    @property
    def _probability(self) -> np.ndarray[tuple[float]]:
        """Prediction market probability with the arbitrary starting values."""
        eP = np.exp(self._price)
        return eP / (1 + eP)

    @property
    def probability(self) -> np.ndarray[tuple[float]]:
        """Prediction market probability without the arbitrary starting values."""
        return self._probability[self._init_history_size:]
