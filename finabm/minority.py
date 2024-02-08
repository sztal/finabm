"""Minority Game (MG) implementation."""
from __future__ import annotations
from typing import Any, Optional
from functools import cached_property
import numpy as np
from .base import Game, GameModule


class Minority(GameModule):
    r"""Minority Game module class.

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
    C
        History code(s) (scalar or 1D array).
    scores
        1D array with cumulative payoffs obtained by individual agents.
    """
    # pylint: disable=too-many-instance-attributes
    def __init__(
        self,
        N: int,
        M: int | np.ndarray[tuple[int], np.integer],
        S: int | np.ndarray[tuple[int], np.integer]
    ) -> None:
        """Initialization method.

        Parameters
        ----------
        seed
            Seed for generating (pseudo)random numbers.
        """
        super().__init__(N, M)
        if N % 2 == 0:
            raise ValueError("'N' has to be odd")
        if not np.isscalar(S):
            if S.ndim != 1:
                raise ValueError("'S' has to be 1-dimensional")
            if S.size != N:
                raise ValueError("'S' size is inconsistent with 'n'")
        self.S = S
        self.V = None
        self.P = None
        self.C = None
        self.scores = None

    def __repr__(self) -> str:
        cn = self.__class__.__name__
        M = self.M if np.isscalar(self.M) else self.M.mean()
        S = self.S if np.isscalar(self.S) else self.S.mean()
        return f"<{cn} with {self.N} agents " \
            f"and mean M={M:.2f} and S={S:.2f} at {hex(id(self))}>"

    # Properties --------------------------------------------------------------

    @cached_property
    def Ms(self) -> np.ndarray[tuple[int], np.integer]:
        """Memory vector for agent strategies."""
        if np.isscalar(self.M):
            if np.isscalar(self.S):
                S = self.S*self.N
            else:
                S = self.S.sum()
            out = np.repeat(self.M, S)
        else:
            out = np.repeat(self.M, self.S)
        return out.astype(np.uint16)

    @cached_property
    def nS(self) -> int:
        """Total number of strategies."""
        if np.isscalar(self.S):
            return self.S * self.N
        return self.S.sum()

    @cached_property
    def smax(self) -> int:
        """Maximum strategy bag size."""
        if np.isscalar(self.S):
            return self.S
        return self.S.max()

    @cached_property
    def sprod(self) -> int:
        """Product of unique strategy bag sizes."""
        return np.prod(np.unique(self.S))

    @cached_property
    def shift(self) -> np.ndarray[tuple[int], np.integer]:
        """Vector of shifts for traversing over blocks of strategy
        array corresponding to strategies of different agents.
        """
        if np.isscalar(self.S):
            S = np.full(self.N, self.S)
        else:
            S = self.S
        shift = np.roll(S, shift=1)
        shift[0] = 0
        return shift.cumsum().astype(np.uint32)

    @cached_property
    def sidx(self) -> np.ndarray[tuple[int], np.integer]:
        """Strategy index array."""
        return np.arange(self.nS, dtype=np.uint32)

    @cached_property
    def i(self) -> np.ndarray[tuple[int], np.integer]:
        """Array of agent indices multiples by their strategy bag sizes."""
        return np.repeat(np.arange(self.N), self.S).astype(np.uint32)

    @cached_property
    def j(self) -> np.ndarray[tuple[int], np.integer]:
        """Array of strategy indices blocked by agents."""
        return (np.arange(self.nS) - np.repeat(self.shift, self.S)) \
            .astype(np.uint32)

    @cached_property
    def cmask(self) -> np.ndarray[tuple[int, int], np.bool_]:
        """Boolean array for masking per agent memory sequences."""
        return np.arange(self.mmax) < self.mmax - self.Ms[:, None]

    @cached_property
    def cbases(self) -> np.ma.MaskedArray[tuple[int, int], np.integer]:
        """Masked array with bases for calculating sequence
        integer codes per agent.
        """
        mseq = np.ma.array(self.cmask, mask=self.cmask)
        X = np.cumsum(np.ma.ones_like(mseq, dtype=np.uint32), axis=1) \
            .astype(np.uint32)
        X = X.max(axis=1)[:, None] - X
        return 2**X

    # Methods -----------------------------------------------------------------

    def initialize(self, game: Game) -> None:
        """Initialize as game module."""
        super().initialize(game)
        self._init_P()
        self._init_V()
        self._init_C()
        self.scores = np.zeros(self.N, dtype=int)

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
        attendance, agent_decisions, strategy_decisions = self._step()
        self.scores += self._get_payoffs(attendance, agent_decisions)
        scores = self._get_payoffs(attendance, strategy_decisions)
        self.V[self.i, self.j] += scores.astype(self.V.dtype)
        return attendance, agent_decisions, strategy_decisions

    # Internals ---------------------------------------------------------------

    def _step(self) -> tuple[int, np.ndarray, np.ndarray]:
        strategy_decisions = self._get_decisions()
        strategies = self._get_strategies()
        agent_decisions = self._get_agent_decisions(strategy_decisions, strategies)
        attendance = agent_decisions.sum()
        majority = np.sign(attendance)
        self._update_C(majority)
        return attendance, agent_decisions, strategy_decisions

    def _get_decisions(self) -> np.ndarray[tuple[int], np.integer]:
        """Get vector of decisions for all strategies."""
        return self.P[self.sidx, self.C].data

    def _get_agent_decisions(
        self,
        decisions: np.ndarray[tuple[int], np.integer],
        strategies: np.ndarray[tuple[int, np.integer]]
    ) -> np.ndarray[tuple[int], np.integer]:
        """Get agent decisions according to selected strategies."""
        return decisions[strategies+self.shift]

    def _get_strategies(
        self,
        idx: Optional[np.ndarray[tuple[int], np.integer]] = None
    ) -> np.ndarray[tuple[int, int], np.integer]:
        """Get strategies array.

        Parameters
        ----------
        idx
            Vector of per player strategy indices.
            The best performing stragies are selected if ``None``.
        """
        if idx is None:
            mask = self.V == self.V.max(axis=1)[:, None]
            if isinstance(mask, np.ma.MaskedArray):
                mask = mask.data
            mask = mask.cumsum(axis=1)
            B = mask / mask[:, -1, None]
            X = self.game.rng.random(self.N)
            idx = np.argmax(B > X[:, None], axis=1)
        return idx

    def _get_payoffs(
        self,
        attendance: int,
        decisions: np.ndarray[tuple[int], np.integer]
    ) -> np.ndarray[tuple[int], np.integer]:
        """Get payoff vector from the global attendance score
        and agent decisions vector.
        """
        return (-decisions*np.sign(attendance)).astype(int)

    def _update_C(self, majority: int) -> np.ndarray[tuple[int], np.integer]:
        """Update history codes."""
        majority = 1 if majority == 1 else 0
        M = self.M if np.isscalar(self.M) else self.Ms
        self.C = (self.C % 2**(M-1)) * 2 + majority

    def _init_P(self) -> None:
        """Initialize strategy array ``P``."""
        nS = self.sidx.size
        X = self.game.rng.integers(0, 2, (nS, 2**self.mmax), dtype=np.int8)
        X = 2*X-1
        I = np.arange(X.size, dtype=np.uint32).reshape(nS, -1)
        I -= I[:, None, 0]
        mask = I >= 2**self.Ms[:, None]
        self.P = np.ma.array(X, mask=mask)

    def _init_V(self) -> np.ndarray[tuple[int, int], np.bool_]:
        """Initialize array of virtual strategy scores."""
        M = np.arange(self.N*self.smax).reshape(self.N, -1)
        M -= M[:, None, 0]
        M = M >= (self.S if np.isscalar(self.S) else self.S[:, None])
        V = np.zeros_like(M, dtype=int)
        if not np.isscalar(self.S):
            V = np.ma.array(V, mask=M)
        self.V = V

    def _init_C(self) -> np.ndarray[tuple[int], np.integer]:
        """Initialize integer codes for memorized binary sequences."""
        start = self.game.n_step
        end = start+self.mmax
        hist = np.where(self.game.history[start:end] == 1, 1, 0)[None, :]
        seq = np.repeat(hist, self.sidx.size, axis=0)
        mseq = np.ma.array(seq, mask=self.cmask)
        C = (self.cbases*mseq).sum(axis=1).data
        if np.isscalar(self.M):
            C = C[0]
        self.C = C


class MinorityGame(Game):
    """Minority Game.

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
        super().__init__(Minority(N, M, S), **kwds)

    def __getattr__(self, name: str) -> Any:
        try:
            return getattr(self.mg, name)
        except AttributeError as e:
            cn = self.__class__.__name__
            raise AttributeError(
                f"'{cn}' has no attribute '{name}'"
            ) from e

    # Properties --------------------------------------------------------------

    @property
    def mg(self) -> Minority:
        """MG module object."""
        return self.modules[0]
