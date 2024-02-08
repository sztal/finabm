"""Data for reproducing figure 2.1. from the thesis."""
# pylint: disable=missing-function-docstring
import sys
import pickle
from pathlib import Path
from argparse import ArgumentParser
from multiprocessing import cpu_count
import numpy as np
from tqdm import tqdm
from pqdm.processes import pqdm
from finabm import MinorityGame


def main():
    # pylint: disable=too-many-locals
    here = Path(__file__).absolute().parent
    parser = make_parser()
    cli = parser.parse_args()

    T = 300 # multiplier for determining the number of iterations
    R = 50  # number of replications of each parameter combination
    M = 8   # memory size
    S = [2, 3, 4, 5, 6]
    # Values of N producing range of alpha values roughly between
    N = (np.logspace(*np.log10([25, 5120]), 30)/2).astype(int)*2 + 1
    alpha = 2**M/N

    A2 = np.empty((len(S), len(alpha), R))
    counter = 0
    for i, s in tqdm([*enumerate(S)]):
        for j, n in tqdm([*enumerate(N)], leave=False):
            args = np.column_stack([
                [ n for _ in range(R) ],
                [ M for _ in range(R) ],
                [ s for _ in range(R) ],
                [ T for _ in range(R) ],
                list(range(counter, counter+R))
            ])
            results = pqdm(args, compute, n_jobs=cli.jobs, leave=False)
            A2[i, j, :] = results

    with open(here/"fig-21.pkl", "wb") as fh:
        params = (N, M, S, R, T)
        pickle.dump((params, A2), fh)


def compute(args):
    n, m, s, t, seed = args
    mg = MinorityGame(n, m, s, seed=seed)
    mg.run(2**m*t)
    return (mg.attendance**2).mean()


def make_parser() -> ArgumentParser:
    parser = ArgumentParser(
        prog="fig-21",
        description="Generate data for reproducing Fig. 2.1."
    )
    use_cpu = int(np.ceil(cpu_count() / 4))
    parser.add_argument(
        "--jobs", type=int, default=use_cpu,
        help="number of parallel processes to use " \
        "(defaults to 1/4 of the available CPUs, rounded upwards)"
    )
    return parser


if __name__ == "__main__":
    sys.exit(main())
