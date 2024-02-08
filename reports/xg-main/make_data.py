"""Data for generating statistics for the XG."""
# pylint: disable=missing-function-docstring,import-error
# pylint: disable=protected-access
import gzip
import pickle
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from finabm import Dollar, FinancialGame, Fundamental, Simulation

HERE = Path(__file__).parent


class XGSimulation(Simulation):
    """Simulation for generating data for detailed statistics
    for testing XG.
    """
    @staticmethod
    def compute(params, *, cli=None):
        idx, X = params
        init_price = 100 if cli.model == "fixed-raw" else 1
        modules = []
        if (Nd := cli.N - cli.F) > 0:
            modules.append(Dollar(Nd, cli.M, cli.S))
        if cli.F > 0:
            modules.append(Fundamental(cli.F, cli.M, X, model=cli.model))
        xg = FinancialGame(
            *modules,
            seed=idx,
            initial_price=np.log(init_price),
            liquidity=cli.L*cli.N
        )

        xg.run(cli.T)
        price = xg.price.astype(np.float32)
        out = { "price": price, "Pf": xg.initial_price }
        for module in xg.modules:
            key = "A_dollar" if isinstance(module, Dollar) else "A_funda"
            out[key] = module.attendance
        return out

    def init_parser(self) -> None:
        # pylint: disable=useless-parent-delegation
        super().init_parser()
        self.parser.add_argument(
            "-N", type=int, default=1001,
            help="number of all traders"
        )
        self.parser.add_argument(
            "-F", type=int, default=500,
            help="number of traders using fundamental analysis "
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
            "--model", type=str, default="fixed-raw",
            help="model of fundamental analysis to use"
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
    simulation = XGSimulation(
        name="XG data generator",
        description="XG simulation generating data for calculating detailed statistics",
        params={
            "X": [.1, 1, 10, 100]
        },
        R=1000
    )
    cli = simulation.cli
    N = cli.N
    F = cli.F
    M = cli.M
    S = cli.S
    L = cli.L
    T = cli.T
    R = cli.R
    if F > N:
        errmsg = "'F' cannot be greater than 'N'"
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
        out  = out["price"] \
            .apply(lambda x: pd.DataFrame(
                np.quantile(x, q=qvs, axis=0).T,
                index=pd.Series(tvec, name="t"),
                columns=pd.Series(qvs, name="quantile")
            )) \
            .pipe(lambda s: pd.concat(s.tolist(), keys=s.index))

    dname  = "quant" if cli.quantiles else "data"
    fname  = f"{dname}-{cli.model}-N{N}-F{F}-M{M}-S{S}-L{L}-T{T}-R{R}.pkl.gz"
    outdir = Path(cli.outdir)
    with gzip.open(outdir/fname, "wb") as fh:
        pickle.dump((cli, out), fh)


if __name__ == "__main__":
    sys.exit(main())
