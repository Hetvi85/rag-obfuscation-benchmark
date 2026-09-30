r"""Control experiment: does the guardrail react to the *format* of obfuscated text
or to the *instruction* it hides?

Takes benign gold passages (no injection), applies each obfuscation family, and
reports the share the guardrail flags as injection.  If benign obfuscated text is
flagged, the classifier is keying on format (false-alarm risk); if it passes, any
flagging of the real payload comes from reading the instruction.

    python src/control_benign.py nq
"""
import csv, glob, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from guardrail import InjectionGuardrail
from obfuscation import OBFUSCATORS

FAMILIES = ["unicode_tags", "zero_width", "homoglyph", "typoglycemia", "base64"]
N_PASSAGES, SEED, RHOS = 60, 0, (0.0, 1.0)

ds = sys.argv[1] if len(sys.argv) > 1 else "nq"
root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
import re
cache = glob.glob(os.path.join(root, "data", f"{ds}_q*_c*.json"))
if not cache:
    sys.exit("Run the retrieval stage first so the dataset sample is cached in data/.")
# use the sample with the most queries (the full run, not a quick test)
cache.sort(key=lambda f: int(re.search(r"_q(\d+)_c", f).group(1)))
print("using sample:", os.path.basename(cache[-1]))
rows = json.load(open(cache[-1], encoding="utf-8"))["rows"]
texts = [r[1] for r in rows[:N_PASSAGES]]

G = InjectionGuardrail()
out_dir = os.path.join(root, "results", ds); os.makedirs(out_dir, exist_ok=True)
print(f"Benign passages flagged as injection (n={len(texts)}; lower = fewer false alarms)")
with open(os.path.join(out_dir, "control_benign.csv"), "w", newline="") as f:
    w = csv.writer(f); w.writerow(["family", "rho", "flagged_share"])
    for fam in FAMILIES:
        cells = []
        for rho in RHOS:
            obf = [OBFUSCATORS[fam](t, rho=rho, seed=SEED) for t in texts]
            share = float((G.p_injection(obf) >= 0.5).mean())
            w.writerow([fam, rho, share]); cells.append(f"rho={rho}: {100*share:.0f}%")
        print(f"  {fam:13s}", "   ".join(cells))
