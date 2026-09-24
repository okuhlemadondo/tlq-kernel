"""Runs R1-R4 exactly as pre-registered. Registry and rules untouched (v0.2)."""
import sys, time, json, pathlib
sys.path[:0] = [str(pathlib.Path(__file__).parents[2]), str(pathlib.Path(__file__).parent)]
from blind_case import blind
import tlq.search as SE
runs = {"R1": (dict(variant="team"), dict()),
        "R2": (dict(variant="team"), dict(cost_budget=2e6)),
        "R3": (dict(variant="game", concept="pbe"), dict()),
        "R4": (dict(variant="game", concept="nash"), dict(cost_budget=2e6)),
        # post-hoc (v0.3, corrected labels) -- reported separately from the pre-registered runs
        "P1": (dict(variant="team", distinct_labels=True), dict()),
        "P2": (dict(variant="team", distinct_labels=True), dict(cost_budget=2e6)),
        "P3": (dict(variant="game", concept="pbe", distinct_labels=True), dict()),
        "P4": (dict(variant="game", concept="nash", distinct_labels=True), dict(cost_budget=2e6))}
which = sys.argv[1:] or list(runs)
for name in which:
    kw, skw = runs[name]
    t = time.time()
    rep = SE.search(blind(**kw), **skw)
    out = {"run": name, "status": rep.status, "reasons": rep.reasons, "seconds": round(time.time() - t, 1), "nodes": rep.nodes,
           "found": [{"rules": [r.name for r in f.rules], "preserves": sorted(f.result.preserves), "classes": len(f.result.classes),
                      "values": sorted({round(x.get("value", x.get("vA", float("nan"))), 6) for x in f.result.solutions})} for f in rep.found],
           "blocked": [{"path": [r.name for r in b.path], "rule": b.rule, "stage": b.stage, "verdict": b.verdict,
                        "failed": [o for _, o, _ in b.failures], "detail": b.detail} for b in rep.blocked]}
    pathlib.Path(__file__).parent.joinpath("out", f"{name}.json").write_text(json.dumps(out, indent=1, default=str))
    print(json.dumps({k: out[k] for k in ("run", "status", "reasons", "seconds", "nodes")}), flush=True)
    for f in out["found"]: print("   FOUND", f)
    for b in out["blocked"]:
        if b["verdict"] != "UNESTABLISHED" or b["stage"] in ("budget", "error", "uncertified"):
            print("   BLOCKED", b["path"], b["rule"], b["stage"], b["verdict"], b["failed"][:2], b["detail"][:110])
