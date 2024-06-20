# %%
import gzip
import pickle
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

mpl.rcParams["axes.formatter.useoffset"] = False

BLUE  = "#0072FF"
HERE  = Path().absolute()
FIGS  = HERE/"figs"
DIR   = HERE/"data"
PATHS = {
    f"df{E}": DIR/f"quant-N1001-E{E}-M10-S5-L500-T10000-R100.pkl.gz"
    for E in (450, 550)
}
if not FIGS.exists():
    FIGS.mkdir()

# %%
DATA = []
for _, path in tqdm(PATHS.items()):
    with gzip.open(path, "rb") as fh:
        CLI, df = pickle.load(fh)  # noqa
    df["N"] = CLI.N
    df["M"] = CLI.M
    df["S"] = CLI.S
    df["rho"] = round(CLI.E / CLI.N, 2)
    df = df.set_index(["N", "M", "S", "rho"], append=True) \
        .reorder_levels(["N", "M", "S", "rho", "beta", "X", "t"])
    DATA.append(df)
DATA = pd.concat(DATA)


# %%
QVS  = [.01, .5, .99]
DROP = ("N", "M", "S")
T    = np.array(DATA.index.get_level_values("t").unique())


# %%
RED = "#e84747"
# Quantile selection
Qvs = QVS
# Time-based filering
TMIN  = 10
TMAX  = np.inf

qdf = DATA[
    (t := DATA.index.get_level_values("t") >= TMIN) & (t <= TMAX)
].droplevel(DROP).sort_index(ascending=True)

Rho  = qdf.index.get_level_values("rho").unique()
Beta = qdf.index.get_level_values("beta").unique()
X    = qdf.index.get_level_values("X").unique()

# Plotting
for beta in Beta:
    fig, axes = plt.subplots(nrows=X.size, ncols=Rho.size, figsize=(22, 18))

    for x, axrow in zip(X, axes, strict=False):
        for rho, ax in zip(Rho, axrow, strict=False):
            df = qdf[Qvs].loc[rho, beta, x]
            qv = df.columns
            t = df.index.get_level_values("t")
            P = df.to_numpy().T

            for q, p in zip(qv, P, strict=False):
                ls = "-" if q == .5 else ":"
                ax.plot(t, p, label=q, color="gray", ls=ls, lw=5)

                if rho > .5:
                    delta = x*np.sqrt(-np.log(2-1/rho))
                    ax.axhline(min(1, 1 + delta), ls="--", color=RED, lw=3)
                    ax.axhline(max(0, 1 - delta), ls="--", color=RED, lw=3)

            ax.set_title(
                rf"$\rho \approx {rho:.2f}, X = {x:.2f}, \beta = {beta:.2f}$",
                fontsize=20, fontweight="bold"
            )
            ax.ticklabel_format(useOffset=False, style="plain", axis="both")
            ax.tick_params(axis="both", which="major", labelsize=16)
            ax.set_ylim(0, 1)

    fig.supxlabel("Time step [t]", y=0, fontsize=30, fontweight="bold")
    fig.supylabel("Probability [p(t)]", x=.01, fontsize=30, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIGS/f"fig-2-beta{beta*100:.0f}.png", bbox_inches="tight", dpi=300)

# %%
