"""Base classes for ABMs."""
# pylint: disable=abstract-method
from __future__ import annotations

from functools import cached_property
from typing import Any

import numpy as np
from tqdm.auto import tqdm


class Game:
    """Game class.

    This is a general game class that may consist of multiple
    modules such as $Game (DG) and Fundamental Game (FG).
    Its main purpose is to keep track of attendance values
    stored as a signed integer attendance vector of which values
    indicate the sum over agents' binary decision marked ``+/-1``.
    Among other things it is used to derive ``+/-1`` history vector
    indicating majority decisions at each time step.
    Moreover, initially it may be populated with some
    arbitrary values in order to supply some starting history
    necessary for agents' to make decisions.

    Attributes
    ----------
    *modules
        Game modules.
    attendance
        1D array of attendance values or `size of group A - size of group B`.
        The property prefixed with ``_`` contains also the arbitrary
        initial values.
    """
    def __init__(
        self,
        *modules,
        attendance: np.ndarray[tuple[int], np.integer] | None = None,
        seed: int | None = None,
        init_modules: bool = True
    ) -> None:
        """Initialization method.

        Parameters
        ----------
        seed
            Seed for generating (pseudo)random numbers.
        init_modules
            Should modules be initialized.
        """
        self.n_step = 0
        self.rng = np.random.default_rng(seed)
        if attendance is not None \
        and (attendance.ndim != 1 or not np.issubdtype(attendance.dtype, np.integer)):
            errmsg = "'attendance' has to be 1-dimensional integer array"
            raise ValueError(errmsg)
        self.modules = modules
        self._attendance = None
        self._init_attendance(attendance)
        self._init_history_size = self._attendance.size
        if init_modules:
            self.init_modules()

    # Properties --------------------------------------------------------------

    @property
    def N(self) -> int:
        """Total number of agents in modules."""
        return sum(m.N for m in self.modules)

    @property
    def mmax(self) -> int:
        """Maximum agent memory size."""
        return max(m.mmax for m in self.modules)

    @property
    def attendance(self) -> np.ndarray[tuple[int], np.integer]:
        """Attendance vector without the arbitrary initial values."""
        return self._attendance[self._init_history_size:]

    @property
    def history(self) -> np.ndarray[tuple[int], np.integer]:
        """1D array storing the history of the last `self.mmax`
        majority decisions.
        """
        # pylint: disable=invalid-unary-operand-type
        return np.sign(self._attendance[-self._init_history_size:])

    @property
    def full_history(self) -> np.ndarray[tuple[int], np.integer]:
        """1D array with the full history of the majority decisions."""
        return np.sign(self._attendance)

    # Methods -----------------------------------------------------------------

    def init_modules(self) -> None:
        """Initialize game modules."""
        for module in self.modules:
            module.initialize(self)
        self.modules = tuple(self.modules)

    def prepare(self, n: int) -> None:
        """Prepare before running for ``n`` steps."""
        att = np.empty(n, dtype=int)
        if self._attendance is None:
            self._attendance = att
        else:
            self._attendance = np.concatenate((self._attendance, att))

    def postprocess(self) -> None:
        """Postprocess after running."""
        self._attendance = self._attendance[:self._init_history_size+self.n_step]

    def step(self) -> Any:
        """Make a step of the game."""
        attendance = 0
        for module in self.modules:
            att, *other = module.step()
            module._attendance.append(att) # pylint: disable=protected-access
            attendance += att
        self._update_attendance(attendance)
        self.n_step += 1
        return (attendance, *other)

    def run(
        self,
        n: int,
        *,
        progress: bool = False,
        stop_absorbing: int | None = None,
        **kwds: Any
    ) -> None:
        """Run simulation for ``n`` steps.

        Parameters
        ----------
        n
            Number of simulation steps to run.
        progress
            Should progress bars be shown.
        stop_absorbing
            Stop simulation after ``stop_absorbing`` number
            of steps with zero :math:`A(t)`.
            Ignored when falsy or negative.
        **kwds
            Passed to :py:func:`tqdm.tqdm` when ``progress=True``.
        """
        self.prepare(n)
        kwds = { "leave": False, **kwds }
        n_zero = 0
        for _ in tqdm(range(n), disable=not progress, **kwds):
            attendance, *_ = self.step()
            if attendance == 0:
                n_zero += 1
            if stop_absorbing and n_zero >= stop_absorbing:
                break
        self.postprocess()

    # Internals ---------------------------------------------------------------

    def _init_attendance(
        self,
        attendance: np.ndarray[tuple[int], np.integer] | None,
        *,
        mmax: int | None = None
    ) -> None:
        """Initialize artificial attendance vector."""
        mmax = mmax or self.mmax
        if attendance is None:
            sdev = self.N / 3
            attendance = np.round(self.rng.normal(0, sdev, mmax))
        attendance = np.clip(attendance.astype(int), -self.N, self.N)
        self._attendance = attendance

    def _update_attendance(self, value: int) -> None:
        """Update attendance vector."""
        self._attendance[self._init_history_size+self.n_step] = value


class GameModule:
    """Base class for game modules.

    Attributes
    ----------
    N
        Number of agents.
    M
        Memory size(s) (scalar or 1D array).
    game
        Parent game object.
    """
    def __init__(
        self,
        N: int,
        M: int | np.ndarray[tuple[int], np.integer],
    ) -> None:
        # pylint: disable=isinstance-second-argument-not-valid-type
        if not np.isscalar(N) or not isinstance(N, int | np.integer) or N < 0:
            errmsg = "'N' has to be a positive integer"
            raise ValueError(errmsg)
        if not np.isscalar(M):
            if M.ndim != 1:
                errmsg = "'M' has to be 1-dimensional"
                raise ValueError(errmsg)
            if M.size != N:
                errmsg = "'M' size is inconsistent with 'N'"
                raise ValueError(errmsg)
        self.N = int(N)
        self.M = M
        self.game = None
        self._attendance = []

    # Properties --------------------------------------------------------------

    @cached_property
    def mmax(self) -> int:
        """Maximum memory size."""
        if np.isscalar(self.M):
            return self.M
        return self.M.max()

    @cached_property
    def idx(self) -> np.ndarray[tuple[int], np.integer]:
        """Agent index array."""
        return np.arange(self.N)

    @property
    def attendance(self) -> np.ndarray[tuple[int], np.integer]:
        """Module-specific attendance array."""
        return np.array(self._attendance)

    # Methods -----------------------------------------------------------------

    def initialize(self, game: Game) -> None:
        """Initialize as game module."""
        self.game = game

    def step(self) -> tuple[Any, ...]:
        """Make a step of module simulation."""
        raise NotImplementedError


class FinancialGame(Game):
    """Financial game class.

    Attributes
    ----------
    *modules
        Game modules.
    attendance
        1D array of attendance values or `size of group A - size of group B`.
        The property prefixed with ``_`` contains also the arbitrary
        initial values.
    price
        1D float array of log-prices. The attribute prefixed with ``_``
        contains also arbitrary initial prices.
    initial_price
        Initial log-price used for determining absolute prices
        from relative prices.
    liquidity
        Constant of inverse proportionality relating attendance
        to log-relative prices. Defaults to ``self.N``.
    """
    def __init__(
        self,
        *modules,
        liquidity: float | None = None,
        initial_price: float = 0,
        **kwds: Any
    ) -> None:
        super().__init__(*modules, **kwds, init_modules=False)
        self.liquidity = float(self.N if liquidity is None else liquidity)
        if not np.isscalar(initial_price):
            errmsg = "'initial_price' has to be a scalar value"
            raise ValueError(errmsg)
        self.initial_price = float(initial_price)
        self._price = None
        self._init_price()
        self.init_modules()
        # Rescale prices so the dynamics start from
        # the value passed as `initial_price`
        # delta = self.initial_price - self._price[-1]
        # self.initial_price += delta
        # self._price += delta

    # Properties --------------------------------------------------------------

    @property
    def price(self) -> np.ndarray[tuple[int], np.floating]:
        """1D price array without the arbitrary starting values."""
        return self._price[self._init_history_size:]

    @property
    def relative_price(self) -> np.ndarray[tuple[int], np.floating]:
        """1D array of log-relative prices."""
        return self.attendance / self.liquidity

    # Methods -----------------------------------------------------------------

    def prepare(self, n: int) -> None:
        """Prepare before running for ``n`` steps."""
        super().prepare(n)
        price = np.empty(n, dtype=float)
        self._price = np.concatenate((self._price, price))

    def postprocess(self) -> None:
        """Postprocess after running."""
        super().postprocess()
        self._price = self._price[:self._init_history_size+self.n_step]

    def step(self) -> Any:
        """Make a step of the game."""
        attendance, *other = super().step()
        self._update_price()
        return (attendance, *other)

    def get_price(
        self,
        attendance: int | np.ndarray[tuple[int], np.integer],
        *,
        initial_price: float | None = None
    ) -> np.ndarray[tuple[float], np.floating]:
        """Get prices from attendance values.

        Parameters
        ----------
        attendance
            Attendance value(s).
        initial_price
            Initial log-price to use for determining absolute prices
            from relative prices. Defaults to ``self.initial_price``.
        """
        if initial_price is None:
            initial_price = self.initial_price
        out = initial_price + np.cumsum(attendance) / self.liquidity
        return out

    # Internals ---------------------------------------------------------------

    def _init_price(self) -> None:
        """Initialize price vector from attendance values."""
        self._price = self.get_price(self._attendance)

    def _update_price(self) -> None:
        """Update price after updating attendance."""
        i = self._init_history_size+self.n_step-1
        new_price = self._price[i-1] + self._attendance[i] / self.liquidity
        self._price[i] = new_price
