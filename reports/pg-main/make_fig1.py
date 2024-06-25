# %%
import gzip
import pickle
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

mpl.rcParams["axes.formatter.useoffset"] = False

HERE = Path(__file__).parent.absolute()
DATA = HERE/"data"
FILE = DATA/"data-N1001-E450-M10-S5-L500-T10000-R1000.pkl.gz"
FIGS  = HERE/"figs"
if not FIGS.exists():
    FIGS.mkdir()


with gzip.open(FILE, "rb") as fh:
    cli, df = pickle.load(fh)  # noqa
df["N"] = cli.N
df["M"] = cli.M
df["S"] = cli.S
df["rho"] = round(cli.E / cli.N, 2)
df = df.set_index(["N", "M", "S", "rho"], append=True) \
    .reorder_levels(["N", "M", "S", "rho", "beta", "X"]) \
    .sort_index()

# %%
RED = "#e84747"
Pf   = 1
Beta = df.index.get_level_values("beta").unique()
T    = np.arange(cli.T)
idx  = slice(100, 10000)
y0   = 0
y1   = 1

for beta in Beta:

    data = df.xs(beta, level="beta")
    Rho  = data.index.get_level_values("rho")
    X    = data.index.get_level_values("X")

    P = np.stack(data.prob, axis=0)
    div, mod = divmod(len(P), 2)
    fig, axes = plt.subplots(nrows=div+mod, ncols=div, figsize=(20, 10))

    for ax, x, rho, p in zip(axes.flat, X, Rho, P, strict=False):
        t = T[idx]
        for pt in p[10:20, idx]:
            ax.plot(t, pt, color="black", alpha=.7)
        ax.set_title(rf"$X = {x}, \beta = {beta:.2f}$", fontsize=20)
        ax.axhline(Pf, ls="-", color=RED, lw=3)
        ax.tick_params(axis="both", which="major", labelsize=16)
        ax.set_ylim(0, 1)
        if rho > .5:
            delta = x*np.sqrt(-np.log(2-1/rho))
            ax.axhline(min(1, Pf + delta), ls="--", color=RED, lw=3)
            ax.axhline(max(0, Pf - delta), ls="--", color=RED, lw=3)

    lines = [
        mpl.lines.Line2D([0], [0], color="red", ls="-", lw=3),
        mpl.lines.Line2D([0], [0], color="red", ls="--", lw=3)
    ]
    labels = ["Fundamental price", r"$\Delta$-bounds"]
    fig.legend(
        lines, labels, ncols=2, fontsize=30, frameon=False,
        loc="upper center", bbox_to_anchor=(.5, 1.1)
    )

    fig.supxlabel("Time step [t]", fontsize=30, fontweight="bold")
    fig.supylabel("Probability [p(t)]", x=.01, fontsize=30, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIGS/f"fig-1-beta{beta*100:.0f}.png", dpi=300, bbox_inches="tight")

# %%
