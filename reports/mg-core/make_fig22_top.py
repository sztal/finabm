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

    T = 300         # multiplier for determining the number of iterations
    R = 200         # number of replications of each parameter combination
    M = [5, 6, 8]   # memory size
    S = [2]         # strategy bag sizes
    # Values of N producing range of alpha values roughly between
    alpha = np.logspace(-.8, .8, 60)

    A2 = np.empty((len(alpha), len(M), len(S), R))
    counter = 0
    for i, a in tqdm([*enumerate(alpha)]):
        for j, m in tqdm([*enumerate(M)], leave=False):
            for k, s in tqdm([*enumerate(S)], leave=False):
                n = int(2**m/a / 2)*2 + 1
                args = np.column_stack([
                    [ n for _ in range(R) ],
                    [ m for _ in range(R) ],
                    [ s for _ in range(R) ],
                    [ T for _ in range(R) ],
                    list(range(counter, counter+R))
                ])
                results = pqdm(args, compute, n_jobs=cli.jobs, leave=False)
                A2[i, j, k, :] = results
                counter += R

    with open(here/"fig-22-top.pkl", "wb") as fh:
        params = (alpha, M, S, R, T)
        pickle.dump((params, A2), fh)


def compute(args):
    n, m, s, t, seed = args
    mg = MinorityGame(n, m, s, seed=seed)
    mg.run(2**m*t)
    return (mg.attendance**2).mean()


def make_parser() -> ArgumentParser:
    parser = ArgumentParser(
        prog="fig-22-top",
        description="Generate data for reproducing Fig. 2.2. (top)"
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
