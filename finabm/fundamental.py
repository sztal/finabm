"""Fundamental Game (FG) implementation."""
import warnings
from typing import Any, Literal

import numpy as np
from scipy.stats import norm

from .base import FinancialGame, GameModule


class Fundamental(GameModule):
    r"""Fundamental Game module class.

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
    """
    # pylint: disable=too-many-instance-attributes
    _models = (
        "fixed-raw", "fixed",
        "raw", "log",
        "recent-relative", "relative",
    )

    def __init__(
        self,
        N: int,
        W: int | np.ndarray[tuple[int], np.integer],
        X: float = 1,
        *,
        deterministic: bool = False,
        model: Literal[_models] = _models[0]  # type: ignore
    ) -> None:
        if model not in self._models:
            errmsg = f"'model' has to be one of {self._models}"
            raise ValueError(errmsg)
        if np.any(W < 2):
            errmsg = "'W' must be greater than 1"
            raise ValueError(errmsg)
        super().__init__(N, M=W+1)
        if not np.isscalar(X) and X <= 0:
            errmsg = "'X' has to be a positive scalar"
            raise ValueError(errmsg)
        self.X = X
        self.P = None
        self._gamma = []
        self._prob = []
        self.deterministic = deterministic
        self.model = model
        self.norm = norm()

    # Properties --------------------------------------------------------------

    @property
    def W(self) -> int:
        """Length of price history considered by traders."""
        return self.M - 1

    @property
    def current_price(self) -> float:
        """Get current price value."""
        # pylint: disable=protected-access
        i = self.game._init_history_size+self.game.n_step-1
        if self.model == "raw":
            return np.exp(self.game._price[i])
        if self.model == "log":
            return self.game._price[i]
        return self.game._attendance[i] / self.game.liquidity

    # Methods -----------------------------------------------------------------

    def initialize(self, game: FinancialGame) -> None:
        """Initialize as game module."""
        super().initialize(game)
        self._init_P()

    def step(self) -> tuple[int, float, float]:
        """Make a step of module simulation.

        Returns
        -------
        attendance
            Module attendance result.
        gamma
            Gamma factor.
        prob
            Action probability.
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            gamma = self.get_gamma()
            prob = self.get_prob(gamma)
        attendance = self.get_A(gamma, prob)
        if self.model != "recent-relative":
            self._update_P()
        return attendance, gamma, prob

    def get_gamma(self) -> np.ndarray[tuple[int], np.floating]:
        r""":math:`\gamma(t)` factor."""
        # pylint: disable=protected-access,unsubscriptable-object
        i = self.game._init_history_size+self.game.n_step-1
        if self.model == "raw":
            price = np.exp(self.game._price[i])
        elif self.model == "log":
            price = self.game._price[i]
        else:
            if self.model == "recent-relative":
                change = self.game._attendance[i] / self.game.liquidity
                out = change / self.X
                if np.isscalar(out):
                    out = np.full(self.N, out)
                return out
            price = self.game._price[i]
            if self.model == "fixed":
                return (price - self.game.initial_price) / self.X
            if self.model == "fixed-raw":
                p0 = np.exp(self.game.initial_price)
                price = np.exp(price)
                # return (price - p0) / (p0*self.X)
                return (price - p0) / (self.X)
        diff = price - self.P.mean(axis=1)
        denom = self.P.std(axis=1, ddof=1)*self.X
        gamma = np.full(diff.shape, np.inf)
        mask = denom != 0
        gamma[mask] = diff[mask] / denom[mask]
        mask2 = diff == 0
        gamma[~mask & mask2] = 0
        gamma[~mask & ~mask2] = 1
        return gamma

    def get_prob(
        self,
        gamma: np.ndarray[tuple[float], np.floating]
    ) -> np.ndarray[tuple[float], np.floating]:
        """Get current :math:`p(t)` value."""
        # return 2*self.norm.cdf(np.abs(gamma))-1
        return 1 - np.exp(-gamma**2)

    def get_A(
        self,
        gamma: np.ndarray[tuple[int], np.floating],
        prob: np.ndarray[tuple[int], np.floating]
    ) -> np.ndarray[tuple[int], np.integer]:
        """Get :math:`A(t+1)` value."""
        sign = -np.sign(gamma).astype(int)
        # sign = np.where(gamma < 0, 1, -1)
        if self.deterministic:
            attendance = round((sign*prob).sum())
        else:
            action = np.where(self.game.rng.random(self.N) <= prob, 1, 0)
            action *= sign
            attendance = action.sum()
        return attendance

    # Internals ---------------------------------------------------------------

    def _init_P(self) -> None:
        """Initialize ``P`` array."""
        # pylint: disable=protected-access
        if self.model == "raw":
            P = np.exp(self.game._price[:-1])
        elif self.model == "log":
            P = self.game._price[:-1]
        else:
            P = self.game._attendance[:-1] / self.game.liquidity

        mmax = self.mmax
        P = np.repeat(P[None, :], self.N, axis=0)
        if not np.isscalar(self.M):
            mask = np.repeat(np.arange(P.shape[1])[None, :], self.N, axis=0)
            mask = mask < (mmax - self.M)[:, None]
            P = np.ma.array(P, mask=mask)
        self.P = P

    def _update_P(self) -> None:
        """Update ``P`` array."""
        # pylint: disable=protected-access
        if isinstance(self.P, np.ma.MaskedArray):
            P = np.ma.array(np.roll(self.P.data, -1, axis=1), mask=self.P.mask)
        else:
            P = np.roll(self.P, -1, axis=1)
        i = self.game._init_history_size + self.game.n_step - 1
        if self.model == "raw":
            new = np.exp(self.game._price[i])
        elif self.model == "log":
            new = self.game._price[i]
        else:
            new = self.game._attendance[i] / self.game.liquidity
        P[:, -1] = new
        self.P = P


class FundamentalGame(FinancialGame):
    """Fundamental Game.

    Attributes
    ----------
    N
        Number of agents.
    W
        Length of price history considered by traders.
    X
        Constant factor for inverse proportional rescaling
        of price fluctuations.
    """
    def __init__(
        self,
        N: int,
        W: int | np.ndarray[tuple[int], np.integer],
        X: float = 1,
        *,
        model: Literal[Fundamental._models] = Fundamental._models[0],  # type: ignore
        deterministic: bool = False,
        **kwds: Any
    ) -> None:
        module = Fundamental(N, W, X, model=model, deterministic=deterministic)
        super().__init__(module, **kwds)

    def __getattr__(self, name: str) -> Any:
        try:
            return getattr(self.fg, name)
        except AttributeError as exc:
            errmsg = f"'{self.__class__.__name__}' has no attribute '{name}'"
            raise AttributeError(errmsg) from exc

    # Properties --------------------------------------------------------------

    @property
    def fg(self) -> Fundamental:
        """FG module object."""
        return self.modules[0]
