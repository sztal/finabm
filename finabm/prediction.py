"""Prediction Game (PG) implementation."""
from typing import Any

import numpy as np

from .base import FinancialGame
from .dollar import Dollar
from .fundamental import Fundamental


class Expert(Fundamental):
    """Expert aka fundamental trader with arbitrary fundamental price.

    It implements only ``fixed-raw`` model.

    Attributes
    ----------
    N
        Number of agents.
    W
        Length of prediction history considered by agents.
    M
        Total memory length, which should be equal to ``W+1``.
    X
        Constant factor for inverse proportional rescaling
        of predited probability fluctuations.
    P
        2D array with price histories seen by different agents.
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
        if price > 0:
            prob = 1 / (1 + np.exp(-price))
        else:
            price = np.exp(price)
            prob = price / (1 + price)
        gamma = (prob - self.fundamental_prob) / self.X
        return gamma


class Influencer(Dollar):
    r"""Influencer aka technical trader with non-zero manipulation bias probability.

    Attributes
    ----------
    beta
        Manipulation/bias probability.
        Agent decisions are fixed to ``-1`` with probability :math:`\beta`.
    """
    def __init__(
        self,
        *args: Any,
        beta: float = 0.0,
        **kwargs: Any
    ) -> None:
        super().__init__(*args, **kwargs)
        self.beta = beta

    def _step(self) -> tuple[
        np.ndarray[tuple[int], np.integer],
        np.ndarray[tuple[int], np.integer],
        np.ndarray[tuple[int], np.integer]
    ]:
        attendance, agent_decisions, strategy_decisions = super()._step()
        mask = np.random.random(agent_decisions.size) <= self.beta  # noqa
        agent_decisions[mask] = -1
        return attendance, agent_decisions, strategy_decisions

class PredictionGame(FinancialGame):
    """Prediction game class."""

    @property
    def _probability(self) -> np.ndarray[tuple[float]]:
        """Prediction market probability with the arbitrary starting values."""
        P = self._price.copy()
        mask = P > 0
        P[mask] = 1 / (1 + np.exp(-P[mask]))
        eP = np.exp(P[~mask])
        P[~mask] = eP / (1 + eP)
        return P

    @property
    def probability(self) -> np.ndarray[tuple[float]]:
        """Prediction market probability without the arbitrary starting values."""
        return self._probability[self._init_history_size:]
