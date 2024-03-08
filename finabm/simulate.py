"""Simulation utilities."""
from argparse import ArgumentParser, Namespace
from collections.abc import Iterable, Mapping, Sequence
from functools import partial
from itertools import product
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
from pqdm.processes import pqdm
from tqdm import tqdm


class Simulation:
    """Simulation runner.

    Attributes
    ----------
    name
        Simulation program name.
    description
        Description of the simulation program.
    params
        Simple namespace object with fixed simulation parameters.
        Values for each parameter should be passed as sequences.
        Can be passed as a mapping. Number of repretitions per
        configuration is passed separately.
    R
        Number of repetitions of each configuration of parameters.
    progress
        Should progress bars be shown.
    parser
        Argument parser object.
        Set up with :meth:`init_parser()`.
    cli
        Container with CLI arguments passed to the simulation.
    default_
    """
    def __init__(
        self,
        name: str,
        description: str,
        params: Mapping[str, Sequence],
        *,
        R: int = 1,
        progress: bool = True
    ) -> None:
        self.name = name
        self.description = description
        self.params = SimpleNamespace(**params)
        self.progress = progress
        self.parser = None
        self._default_R = R
        self.init_parser()
        self.cli = self.parser.parse_args()

    def __call__(self) -> pd.DataFrame:
        results = []
        for params in self.iter_param_blocks():
            records = params.itertuples(name=None)
            compute = partial(self.compute, cli=self.cli)
            out = pqdm(records, compute, n_jobs=self.cli.jobs, leave=False)
            out = pd.DataFrame(out, index=pd.MultiIndex.from_frame(params))
            out = self.postprocess(out)
            results.append(out)
        return pd.concat(results, axis=0, ignore_index=False)

    @property
    def param_names(self) -> tuple[str, ...]:
        """Get parameter names."""
        return tuple(dict(self.params.__dict__))

    @property
    def R(self) -> int:
        """Independent simulation runs per parameter configuration."""
        return self.cli.R

    def init_parser(self) -> None:
        """Initialize argument parser instance."""
        self.parser = ArgumentParser(
            prog=self.name,
            description=self.description
        )
        self.parser.add_argument(
            "-R", type=int, default=self._default_R,
            help="number of independent simulation runs " \
            "per parameter configuration"
        )
        self.parser.add_argument(
            "--jobs", type=int, default=1,
            help="number of parallel processes to use " \
            "(defaults to 1 meaning no parallel processing)"
        )

    def get_param_iterators(self) -> tuple[Iterable, ...]:
        """Get parameter iterators."""
        return tuple(self.params.__dict__.values())

    def iter_params(self) -> Iterable[tuple[Any, ...]]:
        """Iterate over unique parameter configurations."""
        first, *other = self.get_param_iterators()
        other = list(product(*other))
        disable = not self.progress
        for value in tqdm(list(first), disable=disable):
            if other:
                for ovals in tqdm(other, leave=False, disable=disable):
                    yield (value, *ovals)
            else:
                yield value

    def iter_param_blocks(self) -> Iterable[pd.DataFrame]:
        """Iterate over blocks corresponding to repetitions
        of the same parameters.
        """
        count = 0
        for params in self.iter_params():
            start, end = count, count+self.cli.R
            block = pd.DataFrame(
                [params]*self.cli.R,
                columns=self.param_names,
                index=np.arange(start, end)
            )
            count = end
            yield block

    @staticmethod
    def compute(
        params: tuple[int, pd.Series],
        *,
        cli: Namespace | None = None
    ) -> Any:
        """Run simulation for a specific configuration of parameters
        including repeating for ``self.cli.R`` times.

        Parameters
        ----------
        params
            2-tuple with repetition index and parameter series.
        cli
            Container object with command-line arguments.

        Returns
        -------
        out
            Order parameter(s) of interest.
        """
        raise NotImplementedError

    def postprocess(self, out: pd.DataFrame) -> pd.DataFrame:
        """Post-process output data frame."""
        return out
