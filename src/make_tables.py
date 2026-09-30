r"""Turn results/<dataset>/*.csv into every number the report needs.

    python src/make_tables.py nq

Prints (and writes to results/<dataset>/tables.tex) ready-to-paste LaTeX rows
for Tables P_ret, epsilon, rho*, and the defence, plus the in-text statistics
(n, N, Spearman r_s and p for H1/H2, the typical rho*, the utility cost).
Statistics follow Sec. IV of the paper: seed-averaged means, 95% percentile
bootstrap CIs over queries (B = 2000), Spearman trend tests, and the
CI-separated interior-optimum test for H3.
"""
import collections, csv, json, os, sys
import numpy as np

RHOS = [0.0, 0.25, 0.5, 0.75, 1.0]
FAMILIES = ["unicode_tags", "zero_width", "homoglyph", "typoglycemia",
            "base64", "hybrid"]
SINGLE = FAMILIES[:5]
NAMES = {"unicode_tags": "Unicode Tags", "zero_width": "Zero-width  ",
         "homoglyph": "Homoglyph   ", "typoglycemia": "Typoglycemia",
         "base64": "Base64      ", "hybrid": "Hybrid      "}
RNAME = {"bm25": "BM25", "contriever": "Contriever", "e5": "E5"}
K, THRESH, B = 10, 0.5, 2000
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")


def _read(path, keycols, valfn):
    """-> {key: array(n_seeds, n_q)} from a per-query CSV."""
    if not os.path.exists(path):
        return None
    tmp = collections.defaultdict(dict)
    for r in csv.DictReader(open(path, encoding="utf-8")):
        key = tuple(r[c] if c != "rho" else float(r[c]) for c in keycols)
        tmp[key][(int(r["seed"]), int(r["qidx"]))] = valfn(r)
    out = {}
    for key, d in tmp.items():
        seeds = sorted({s for s, _ in d}); qs = sorted({q for _, q in d})
        out[key] = np.array([[d[(s, q)] for q in qs] for s in seeds], float)
    return out


def load(d):
    hit = lambda r: float(int(r["rank"]) <= K)
    ev = lambda r: float(float(r["p_injection"]) < THRESH)
    S = {
        "ret": _read(os.path.join(d, "retrieval.csv"), ["retriever", "family", "rho"], hit),
        "eps": _read(os.path.join(d, "evasion.csv"), ["family", "rho"], ev),
        "gen": _read(os.path.join(d, "execution.csv"), ["family", "rho"],
                     lambda r: float(r["hit"])),
        "ret_def": None if "--no-defence" in sys.argv else _read(os.path.join(d, "retrieval_def.csv"),
                         ["retriever", "family", "rho"], hit),
        "eps_def": None if "--no-defence" in sys.argv else _read(os.path.join(d, "evasion_def.csv"), ["family", "rho"], ev),
    }
    if "--no-defence" in sys.argv:
        S["ret_def"] = S["eps_def"] = None
    if S["ret"] is None or S["eps"] is None:
        sys.exit(f"Need retrieval.csv and evasion.csv in {d} (run those stages first).")
    S["retrievers"] = [r for r in ("bm25", "contriever", "e5")
                       if any(k[0] == r for k in S["ret"])]
    S["n"] = next(iter(S["ret"].values())).shape[1]
    up = os.path.join(d, "utility.csv")
    S["util"] = list(csv.DictReader(open(up))) if os.path.exists(up) else None
    return S


def _perq(a):  # seed-average -> per-query vector
    return a.mean(axis=0)


def factors(S, rk, fam, rho, defended=False):
    ret = S["ret_def" if defended else "ret"]; eps = S["eps_def" if defended else "eps"]
    pr = float(ret[(rk, fam, rho)].mean())
    ev = float(eps[(fam, rho)].mean()) if eps else float("nan")
    pg = float(S["gen"][(fam, rho)].mean()) if S["gen"] else 1.0
    return pr, ev, pg


def asr_boot(S, rk, fam, rho, defended=False, seed=0):
    """Point ASR (product of factor means) and 95% bootstrap CI over queries."""
    ret = S["ret_def" if defended else "ret"]; eps = S["eps_def" if defended else "eps"]
    r, e = _perq(ret[(rk, fam, rho)]), _perq(eps[(fam, rho)])
    g = _perq(S["gen"][(fam, rho)]) if S["gen"] else None
    rng = np.random.default_rng(seed)  # common random numbers across rho
    n = len(r); boots = []
    for _ in range(B):
        i = rng.integers(0, n, n)
        pg = g[rng.integers(0, len(g), len(g))].mean() if g is not None else 1.0
        boots.append(r[i].mean() * e[i].mean() * pg)
    point = r.mean() * e.mean() * (g.mean() if g is not None else 1.0)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return float(point), float(lo), float(hi)


def boot_mean(v, seed=0):
    rng = np.random.default_rng(seed)
    b = [v[rng.integers(0, len(v), len(v))].mean() for _ in range(B)]
    lo, hi = np.percentile(b, [2.5, 97.5])
    return float(v.mean()), float(lo), float(hi)


def spearman(x, y):
    try:
        import warnings
        from scipy.stats import spearmanr
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r, p = spearmanr(x, y)
        return float(r), float(p)
    except ImportError:
        import metrics
        return metrics.spearman_trend(x, y)[0], float("nan")


def rho_star(S, rk, fam):
    pts = [asr_boot(S, rk, fam, rho) for rho in RHOS]
    i = int(np.argmax([p[0] for p in pts]))
    interior = (0 < i < len(RHOS) - 1 and pts[i][1] > pts[0][2]
                and pts[i][1] > pts[-1][2])
    return RHOS[i], interior, pts


def f2(x):
    return "--" if x != x else f"{x:.2f}".replace("0.", ".", 1) if 0 <= x < 1 else f"{x:.2f}"


def pval(p):
    return "n/a (constant curve or scipy missing)" if p != p else ("<0.001" if p < 0.001 else f"={p:.3f}")


def main(ds):
    d = os.path.join(ROOT, "results", ds)
    S = load(d)
    L = []
    say = lambda s="": (print(s), L.append(s))

    say(f"% ===== Numbers for the report, dataset = {ds} =====")
    ncorp = None
    cache = [f for f in os.listdir(os.path.join(ROOT, "data"))
             if f.startswith(ds + "_q")] if os.path.isdir(os.path.join(ROOT, "data")) else []
    if cache:
        ncorp = len(json.load(open(os.path.join(ROOT, "data", cache[0])))["corpus"])
    say(f"% n (queries) = {S['n']}    N (corpus passages) = {ncorp}")
    say(f"% P_gen measured: {'yes' if S['gen'] else 'NO -> P_gen = 1 assumed (say so in Sec. V)'}")

    # ---- Table: P_ret ----
    say("\n% ---- Table tab:pret  (Recall@10, mean over seeds) ----")
    for rk in S["retrievers"]:
        say(f"% {RNAME[rk]}")
        for fam in FAMILIES:
            vals = [S["ret"][(rk, fam, rho)].mean() for rho in RHOS]
            say(f" & {NAMES[fam]} & " + " & ".join(f2(v) for v in vals) + r"\\")

    # ---- Table: epsilon ----
    say("\n% ---- Table tab:eps  (evasion, mean with 95% CI half-width) ----")
    for fam in FAMILIES:
        cells = []
        for rho in RHOS:
            m, lo, hi = boot_mean(_perq(S["eps"][(fam, rho)]))
            cells.append(f"{f2(m)}{{\\scriptsize$\\pm${f2((hi - lo) / 2)}}}")
        say(f"{NAMES[fam]} & " + " & ".join(cells) + r"\\")

    # ---- H1 / H2 statistics ----
    say("\n% ---- In-text statistics ----")
    for rk in S["retrievers"]:
        x, y = [], []
        for fam in SINGLE:
            for rho in RHOS:
                for s_val in S["ret"][(rk, fam, rho)].mean(axis=1):
                    x.append(rho); y.append(s_val)
        r, p = spearman(x, y)
        say(f"% H1 ({RNAME[rk]}, single-family pooled): r_s = {r:.2f}, p {pval(p)}")
    for fam in SINGLE + ["hybrid"]:
        for rk in S["retrievers"]:
            x = [rho for rho in RHOS for _ in range(S["ret"][(rk, fam, rho)].shape[0])]
            y = [v for rho in RHOS for v in S["ret"][(rk, fam, rho)].mean(axis=1)]
            r, p = spearman(x, y)
            say(f"%   H1 per family {RNAME[rk]}/{fam}: r_s = {r:.2f}, p {pval(p)}")
    x = [rho for fam in SINGLE for rho in RHOS for _ in range(S["eps"][(fam, rho)].shape[0])]
    y = [v for fam in SINGLE for rho in RHOS for v in S["eps"][(fam, rho)].mean(axis=1)]
    r, p = spearman(x, y)
    say(f"% H2 (single-family pooled): r_s = {r:.2f}, p {pval(p)}")

    # ---- Table: rho* ----
    say("\n% ---- Table tab:rhostar  (int. = CI-separated interior optimum) ----")
    say("% columns: " + " | ".join(RNAME[r] for r in S["retrievers"]))
    interior_rhos = []
    for fam in FAMILIES:
        cells = []
        for rk in S["retrievers"]:
            rs, inter, _ = rho_star(S, rk, fam)
            if inter:
                interior_rhos.append(rs)
            cells += [f"{rs:g}", "int." if inter else "bd."]
        say(f"{NAMES[fam]} & " + " & ".join(cells) + r"\\")
    if interior_rhos:
        vals, cnt = np.unique(interior_rhos, return_counts=True)
        say(f"% Most common interior rho* (use in Abstract): {vals[np.argmax(cnt)]:g}")
    else:
        say("% No CI-separated interior optimum found: H3 NOT supported -> rewrite Abstract/Sec. VI claims honestly.")

    say("\n% ASR(rho) point estimates [95% CI], for the text / appendix:")
    for rk in S["retrievers"]:
        for fam in FAMILIES:
            _, _, pts = rho_star(S, rk, fam)
            say(f"%   {RNAME[rk]}/{fam}: " + "  ".join(
                f"{rho:g}:{p[0]:.2f}[{p[1]:.2f},{p[2]:.2f}]" for rho, p in zip(RHOS, pts)))

    # ---- Table: defence ----
    if S["ret_def"] is not None and S["eps_def"] is not None:
        say("\n% ---- Table tab:defence  (attacker picks best rho; mean over retrievers) ----")
        def best(fams, defended):
            per_rk = []
            for rk in S["retrievers"]:
                per_rk.append(np.mean([max(np.prod(factors(S, rk, f, rho, defended))
                                           for rho in RHOS) for f in fams]))
            return float(np.mean(per_rk))
        def adaptive(defended):
            return float(np.mean([max(np.prod(factors(S, rk, f, rho, defended))
                                      for f in FAMILIES for rho in RHOS)
                                  for rk in S["retrievers"]]))
        util = ""
        if S["util"]:
            deltas = [100 * (float(u["recall10_defended"]) - float(u["recall10_raw"]))
                      for u in S["util"]]
            util = f"{np.mean(deltas):+.1f} pp"
            say("% clean Recall@10 raw -> defended: " + ", ".join(
                f"{RNAME[u['retriever']]} {float(u['recall10_raw']):.3f}->{float(u['recall10_defended']):.3f}"
                for u in S["util"]))
        say(f"Format carriers (Tags/ZW) & {f2(best(SINGLE[:2], False))} & "
            f"{f2(best(SINGLE[:2], True))} & \\multirow{{3}}{{*}}{{{util}}}\\\\")
        say(f"NFKC-invariant carriers   & {f2(best(SINGLE[2:], False))} & "
            f"{f2(best(SINGLE[2:], True))} & \\\\")
        say(f"Adaptive attacker         & {f2(adaptive(False))} & {f2(adaptive(True))} & \\\\")
    else:
        say("\n% (defence table: run --stage defence first)")

    open(os.path.join(d, "tables.tex"), "w", encoding="utf-8").write("\n".join(L) + "\n")
    print(f"\nSaved to {os.path.join(d, 'tables.tex')}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "nq")
