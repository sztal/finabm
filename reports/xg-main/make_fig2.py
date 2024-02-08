# %%
import gzip
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

mpl.rcParams["axes.formatter.useoffset"] = False

BLUE  = "#0072FF"
HERE  = Path(".").absolute()
FIGS  = HERE/"figs"
DIR   = HERE/"data"
PATHS = {
    f"df{F}": DIR/f"quant-fixed-raw-N1001-F{F}-M10-S5-L500-T10000-R10000.pkl.gz"
    for F in (550, 750, 950)
}
if not FIGS.exists():
    FIGS.mkdir()

# %%
DATA = []
for name, path in tqdm(PATHS.items()):
    with gzip.open(path, "rb") as fh:
        CLI, df = pickle.load(fh)
    df["N"] = CLI.N
    df["M"] = CLI.M
    df["S"] = CLI.S
    df["rho"] = round(CLI.F / CLI.N, 2)
    df = df.set_index(["N", "M", "S", "rho"], append=True) \
        .reorder_levels(["N", "M", "S", "rho", "X", "t"])
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
].droplevel(DROP)
Rho = qdf.index.get_level_values("rho").unique()
X   = qdf.index.get_level_values("X").unique()

# Plotting
fig, axes = plt.subplots(nrows=X.size, ncols=Rho.size, figsize=(22, 18))

for x, axrow in zip(X, axes):
    for rho, ax in zip(Rho, axrow):
        df = qdf[Qvs].loc[rho, x]
        qv = df.columns
        t = df.index.get_level_values("t")
        P = np.exp(df.to_numpy().T)

        for q, p in zip(qv, P):
            ls = "-" if q == .5 else ":"
            ax.plot(t, p, label=q, color="gray", ls=ls, lw=5)

            if rho > .5:
                delta = x*np.sqrt(-np.log(2-1/rho))
                ax.axhline(100 + delta, ls="--", color=RED, lw=3)
                ax.axhline(100 - delta, ls="--", color=RED, lw=3)

        ax.set_title(
            rf"$\rho \approx {rho:.2f}, X = {x:.2f}$",
            fontsize=20, fontweight="bold"
        )
        ax.ticklabel_format(useOffset=False, style="plain", axis="both")
        ax.tick_params(axis="both", which="major", labelsize=16)

fig.supxlabel("Time step [t]", y=0, fontsize=30, fontweight="bold")
fig.supylabel("Price [p(t)]", x=.01, fontsize=30, fontweight="bold")
fig.tight_layout()
fig.savefig(FIGS/"fig-2.png", bbox_inches="tight", dpi=300)

# %%
