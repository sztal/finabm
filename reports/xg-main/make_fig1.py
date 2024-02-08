# %%
import gzip
import pickle
from pathlib import Path
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams["axes.formatter.useoffset"] = False

HERE = Path(__file__).parent.absolute()
DATA = HERE/"data"
FILE = DATA/"fig1-data-fixed-raw-N1001-F750-M10-S5-L500-T5000-R100.pkl.gz"
FIGS  = HERE/"figs"
if not FIGS.exists():
    FIGS.mkdir()


with gzip.open(FILE, "rb") as fh:
    cli, df = pickle.load(fh)
df["N"] = cli.N
df["M"] = cli.M
df["S"] = cli.S
df["rho"] = round(cli.F / cli.N, 2)
df = df.set_index(["N", "M", "S", "rho"], append=True) \
    .reorder_levels(["N", "M", "S", "rho", "X"])

# %%
RED = "#e84747"
Pf  = 100
Rho = df.index.get_level_values("rho")
X   = df.index.get_level_values("X")
T   = np.arange(cli.T)
P   = np.exp(np.stack(df.price, axis=0))
idx = slice(100, 1000)
y0  = P.min()*.9
y1  = P.max()*1.01

div, mod = divmod(len(P), 2)
fig, axes = plt.subplots(nrows=div+mod, ncols=div, figsize=(20, 10))

for ax, x, rho, p in zip(axes.flat, X, Rho, P):
    t = T[idx]
    for pt in p[10:20, idx]:
        ax.plot(t, pt, color="black", alpha=.7)
    ax.set_title(rf"X = {x}", fontsize=20, fontweight="bold")
    ax.axhline(Pf, ls="-", color=RED, lw=3)
    ax.tick_params(axis="both", which="major", labelsize=16)
    if rho > .5:
        delta = x*np.sqrt(-np.log(2-1/rho))
        ax.axhline(Pf + delta, ls="--", color=RED, lw=3)
        ax.axhline(Pf - delta, ls="--", color=RED, lw=3)

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
fig.supylabel("Price [p(t)]", x=.01, fontsize=30, fontweight="bold")
fig.tight_layout()
fig.savefig(FIGS/"fig-1.png", dpi=300, bbox_inches="tight")

# %%
