"""Dollar Game (DG) implementation."""
from typing import Any

import numpy as np

from .base import FinancialGame
from .minority import Minority


class Dollar(Minority):
    r"""Dollar Game module class.

    Attributes
    ----------
    N
        Number of agents.
    M
        Memory size(s) (scalar or 1D array).
    S
        1D array of agent strategy bag sizes.
    V
        2D ragged (masked) array virtual strategy scores.
        Each row correspond to a strategy bag of a single agent.
    P
        Strategy array blocked by agents.
        Each row defines a single strategy, which is represented
        as 1D array of length ``2**M[i]`` which maps codes of every
        possible sequence of length ``M[i]`` to a decision, that is,
        either ``1`` or ``-1``.
    A
        Action array storing (at time :math:`t`)
        decisions of all strategies for times :math:`t-2` and :math:`t-1`
        in columns.
    D
        Decision array storing (at time :math:`t`)
        decisions of all agents for time :math:`t-2` and :math:`t-1`
        in columns.
    C
        History code(s) (scalar or 1D array).
    scores
        1D array with cumulative payoffs obtained by individual agents.
    """
    def __init__(
        self,
        *args: Any,
        **kwds: Any
    ) -> None:
        super().__init__(*args, **kwds)
        self.A = None
        self.D = None

    # Methods -----------------------------------------------------------------

    def initialize(self, game: FinancialGame) -> None:
        """Initialize as game module."""
        super().initialize(game)
        self.A = np.empty((self.nS, 2), dtype=np.int8)
        self.D = np.empty((self.N, 2), dtype=np.int8)
        self.scores = self.scores.astype(float)
        self.V = self.V.astype(float)

    def step(self) -> tuple[int, int, int]:
        """Make step of the simulation.

        Returns
        -------
        attendance
            Attendance score.
        agent_decisions
            Vector of decisions of agents.
        strategy_decisions
            Vector of decisions for all strategies.
        """
        self._update_V()
        self._update_scores()
        attendance, agent_decisions, strategy_decisions = self._step()
        self._update_action_array(self.A, strategy_decisions)
        self._update_action_array(self.D, agent_decisions)
        return attendance, agent_decisions, strategy_decisions

    # Internals ---------------------------------------------------------------

    def _get_payoffs(
        self,
        attendance: int,
        decisions: np.ndarray[tuple[int], np.integer]
    ) -> np.ndarray[tuple[int], np.floating]:
        """Get payoff vector from the global attendance score
        and agent decisions vector. This is used to update strategy
        scores at time :math:`t` before agents make decisions.

        Parameters
        ----------
        attendance
            Attendance at time :math:`t-1` is expected.
        decisions
            Strategy decisions at time :math:`t-2` are expected.
        """
        return decisions*attendance / self.game.liquidity

    def _update_action_array(
        self,
        X: np.ndarray[tuple[int, int], np.integer],
        decisions: np.ndarray[tuple[int], np.integer]
    ) -> None:
        X[:, 0] = X[:, 1]
        X[:, 1] = decisions

    def _update_V(self) -> None:
        """Update ``V`` array with virtual strategy scores."""
        if self.game.n_step >= 2:
            attendance = self.game.attendance[self.game.n_step-1]
            decisions = self.A[:, 0]
            scores = self._get_payoffs(attendance, decisions)
            self.V[self.i, self.j] += scores.astype(self.V.dtype)

    def _update_scores(self) -> None:
        """Update agent scores."""
        if self.game.n_step >= 2:
            attendance = self.game.attendance[self.game.n_step-1]
            decisions = self.D[:, 0]
            scores = self._get_payoffs(attendance, decisions)
            self.scores += scores


class DollarGame(FinancialGame):
    """Dollar Game.

    Attributes
    ----------
    N
        Number of agents.
    M
        Memory size(s) (scalar or 1D array).
    S
        1D array of agent strategy bag sizes.
    """
    def __init__(
        self,
        N: int,
        M: int | np.ndarray[tuple[int], np.integer],
        S: int | np.ndarray[tuple[int], np.integer],
        **kwds: Any
    ) -> None:
        super().__init__(Dollar(N, M, S), **kwds)

    def __getattr__(self, name: str) -> Any:
        try:
            return getattr(self.dg, name)
        except AttributeError as exc:
            errmsg = f"'{self.__class__.__name__}' has no attribute '{name}'"
            raise AttributeError(errmsg) from exc

    # Properties --------------------------------------------------------------

    @property
    def dg(self) -> Minority:
        """DG module object."""
        return self.modules[0]
