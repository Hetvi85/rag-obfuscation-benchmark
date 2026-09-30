r"""Generate the paper's figures from results/*.csv.

Produces (into results/figs/):
  fig_pret.pdf      P_ret(rho) per family x retriever         (Fig. 2)
  fig_epsilon.pdf   epsilon(rho) per family                   (Fig. 3)
  fig_asr.pdf       ASR(rho) with rho* marked                 (Fig. 4)
  fig_tradeoff.pdf  epsilon vs Recall@k, coloured by rho      (Fig. 5, primary)
Replace the red PLACEHOLDER boxes in main.tex with \includegraphics of these.
"""
import csv, collections, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import make_tables as mt

DS = sys.argv[1] if len(sys.argv) > 1 else "nq"
R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", DS)
FIG = os.path.join(R, "figs"); os.makedirs(FIG, exist_ok=True)
RHOS = mt.RHOS
FAMILIES = mt.FAMILIES
S = mt.load(R)
KEY = lambda x: x  # rho keys are floats


def _rec():
    return {(rk, f, str(r)): S["ret"][(rk, f, r)].mean()
            for rk in S["retrievers"] for f in FAMILIES for r in RHOS}


def _eps():
    return {(f, str(r)): S["eps"][(f, r)].mean() for f in FAMILIES for r in RHOS}


def _asr():
    return {(rk, f, str(r)): float(np.prod(mt.factors(S, rk, f, r)))
            for rk in S["retrievers"] for f in FAMILIES for r in RHOS}


def fig_pret():
    rec = _rec()
    retrievers = S["retrievers"]
    fig, axes = plt.subplots(1, len(retrievers), figsize=(4 * len(retrievers), 3.2),
                             sharey=True)
    axes = np.atleast_1d(axes)
    for ax, rk in zip(axes, retrievers):
        for fam in FAMILIES:
            y = [rec.get((rk, fam, str(x)), np.nan) for x in RHOS]
            ax.plot(RHOS, y, marker="o", label=fam)
        ax.set_title(rk); ax.set_xlabel(r"$\rho$"); ax.grid(alpha=.3)
    axes[0].set_ylabel("Recall@10  ($P_{ret}$)")
    axes[-1].legend(fontsize=7, loc="lower left")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig_pret.pdf")); plt.close(fig)


def fig_epsilon():
    eps = _eps()
    fig, ax = plt.subplots(figsize=(4.2, 3.2))
    for fam in FAMILIES:
        y = [eps.get((fam, str(x)), np.nan) for x in RHOS]
        ax.plot(RHOS, y, marker="s", label=fam)
    ax.set_xlabel(r"$\rho$"); ax.set_ylabel(r"guardrail evasion $\varepsilon$")
    ax.grid(alpha=.3); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig_epsilon.pdf")); plt.close(fig)


def fig_asr():
    asr = _asr()
    retrievers = S["retrievers"]
    fig, axes = plt.subplots(1, len(retrievers), figsize=(4 * len(retrievers), 3.2),
                             sharey=True)
    axes = np.atleast_1d(axes)
    for ax, rk in zip(axes, retrievers):
        for fam in FAMILIES:
            y = [asr.get((rk, fam, str(x)), np.nan) for x in RHOS]
            ax.plot(RHOS, y, marker="o", label=fam)
            if not all(np.isnan(y)):
                i = int(np.nanargmax(y)); ax.scatter([RHOS[i]], [y[i]], s=90,
                        facecolors="none", edgecolors="k", zorder=5)
        ax.set_title(rk); ax.set_xlabel(r"$\rho$"); ax.grid(alpha=.3)
    axes[0].set_ylabel(r"$ASR^{*}=P_{ret}\,\varepsilon$")
    axes[-1].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig_asr.pdf")); plt.close(fig)


def fig_tradeoff():
    rec = _rec()
    eps = _eps()
    fig, ax = plt.subplots(figsize=(4.6, 3.6))
    markers = dict(zip(FAMILIES, "os^Dv*"))
    for fam in FAMILIES:
        xs = [rec.get(("bm25", fam, str(x)), np.nan) for x in RHOS]
        ys = [eps.get((fam, str(x)), np.nan) for x in RHOS]
        sc = ax.scatter(xs, ys, c=RHOS, cmap="viridis", marker=markers[fam],
                        s=60, label=fam, edgecolors="k", linewidths=.4)
        ax.plot(xs, ys, color="grey", alpha=.3, zorder=0)
    ax.set_xlabel("Recall@10  ($P_{ret}$, BM25)")
    ax.set_ylabel(r"guardrail evasion $\varepsilon$")
    ax.set_title(r"Retrieval-evasion trade-off (colour = $\rho$)")
    cb = fig.colorbar(sc, ax=ax); cb.set_label(r"$\rho$")
    ax.legend(fontsize=6, loc="center left"); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig_tradeoff.pdf")); plt.close(fig)


if __name__ == "__main__":
    for fn in (fig_pret, fig_epsilon, fig_asr, fig_tradeoff):
        try:
            fn(); print("ok:", fn.__name__)
        except FileNotFoundError as e:
            print("skip (missing csv):", fn.__name__, e)
