"""Data for generating statistics for the PG."""
# pylint: disable=missing-function-docstring,import-error
# pylint: disable=protected-access
import gzip
import pickle
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from finabm import Simulation
from finabm.prediction import Expert, Influencer, PredictionGame

HERE = Path(__file__).parent


class PGSimulation(Simulation):
    """Simulation for generating data for detailed statistics
    for testing PG.
    """
    @staticmethod
    def compute(params, *, cli=None):
        idx, X, beta = params
        modules = []
        if (Nd := cli.N - cli.E) > 0:
            modules.append(Influencer(Nd, cli.M, cli.S, beta=beta))
        if cli.E > 0:
            modules.append(Expert(cli.E, cli.M, X, fundamental_prob=1))
        pg = PredictionGame(
            *modules,
            seed=idx,
            initial_price=0,
            liquidity=cli.L*cli.N
        )

        pg.run(cli.T)
        prob = pg.probability.astype(np.float32)
        out = { "prob": prob }
        for module in pg.modules:
            key = "A_influencer" if isinstance(module, Influencer) else "A_expert"
            out[key] = module.attendance
        return out

    def init_parser(self) -> None:
        # pylint: disable=useless-parent-delegation
        super().init_parser()
        self.parser.add_argument(
            "-N", type=int, default=1001,
            help="number of all agents"
        )
        self.parser.add_argument(
            "-E", type=int, default=500,
            help="number of experts"
        )
        self.parser.add_argument(
            "-M", type=int, default=10,
            help="agents' memory size"
        )
        self.parser.add_argument(
            "-S", type=int, default=5,
            help="strategy bag size for $-Game traders"
        )
        self.parser.add_argument(
            "-L", type=int, default=1,
            help="proportionality constant for the liquidity term"
        )
        self.parser.add_argument(
            "-T", type=int, default=1000,
            help="number of time steps to run simulation for"
        )
        self.parser.add_argument(
            "--quantiles", action="store_true",
            default=False, help="store only quantiles"
        )
        self.parser.add_argument(
            "--outdir", type=str, default=str(HERE/"data"),
            help="output directory, default to './data'"
        )


def main():
    # pylint: disable=too-many-locals
    simulation = PGSimulation(
        name="PG data generator",
        description="PG simulation generating data for calculating detailed statistics",
        params={
            "X": [.1, 1, 10, 100],
            "beta": [.3, .5, .7]
        },
        R=100
    )
    cli = simulation.cli
    N = cli.N
    E = cli.E
    M = cli.M
    S = cli.S
    L = cli.L
    T = cli.T
    R = cli.R

    if E > N:
        errmsg = "'E' cannot be greater than 'N'"
        raise ValueError(errmsg)
    # Run simulation
    if cli.jobs <= 1:
        warnings.filterwarnings("error")
    out = simulation()
    gdf = out.groupby(level=out.index.names)
    out = pd.concat([ gdf[col].apply(np.stack) for col in out ], axis=1)
    if cli.quantiles:
        qvs  = [.01, .05, .1, .25, .5, .75, .9, .95, .99]
        tvec = np.arange(1, T+1)
        out  = out["prob"] \
            .apply(lambda x: pd.DataFrame(
                np.quantile(x, q=qvs, axis=0).T,
                index=pd.Series(tvec, name="t"),
                columns=pd.Series(qvs, name="quantile")
            )) \
            .pipe(lambda s: pd.concat(s.tolist(), keys=s.index))

    dname  = "quant" if cli.quantiles else "data"
    fname  = f"{dname}-N{N}-E{E}-M{M}-S{S}-L{L}-T{T}-R{R}.pkl.gz"
    outdir = Path(cli.outdir)
    with gzip.open(outdir/fname, "wb") as fh:
        pickle.dump((cli, out), fh)


if __name__ == "__main__":
    sys.exit(main())
