"""Distributional analysis of collective failure (Nexus R&R, round 2).

Four checks requested during the revision-strategy process:
  1. Collective-Brier variance decomposition  E[(1-p)^2] = (1-E[p])^2 + Var(p)
     for truth-normalized pooled probability p, log pool and arithmetic pool.
  2. Belief diversity D and mean individual error e_bar across the degree ladder,
     especially concordant evidence sharing (the high-local-load failure).
  3. Paired direction of movement from round 0 to terminal under conclusion sharing.
  4. Distribution shape of terminal pooled P(truth) (quantiles + histogram), so that
     "bimodal" is not asserted without inspection.

Population variance (ddof=0) is used throughout so the decomposition identity is exact.
System python3; reads code/data/<tag>/rounds.csv. Prints only; derives no manuscript text.
"""
import csv, collections, math, statistics as st
from pathlib import Path

HERE = Path(__file__).parent
TERMINAL = 3

DEGREE = {"circle": 2.0, "ring_shortcuts": 2.55, "lattice_k4": 4.0,
          "lattice_k6": 6.0, "lattice_k8": 8.0, "all_channel": 10.0}


def load(tag):
    f = HERE / "data" / tag / "rounds.csv"
    return list(csv.DictReader(open(f))) if f.exists() else []


def p_truth(row, pool):
    """Truth-normalized pooled probability under the requested pool."""
    coll = float(row["coll_logodds"] if pool == "log" else row["coll_mean"])
    return coll if row["true_state"] == "A" else 1.0 - coll


def cells(rows, regime, topology, content, pool, rnd):
    out = {}
    for r in rows:
        if r["regime"] != regime or r["topology"] != topology: continue
        if r["content_mode"] != content or int(r["round"]) != rnd: continue
        out[r["seed"]] = p_truth(r, pool)
    return out


def decompose(vals):
    """E[(1-p)^2] = (1-E[p])^2 + Var(p).  Returns the pieces plus diagnostics."""
    n = len(vals)
    m = sum(vals) / n
    var = sum((v - m) ** 2 for v in vals) / n
    brier = sum((1 - v) ** 2 for v in vals) / n
    return dict(n=n, mean=m, sd=math.sqrt(var), var=var, brier=brier,
                mean_term=(1 - m) ** 2, var_term=var,
                resid=brier - ((1 - m) ** 2 + var),
                wrong=sum(1 for v in vals if v < 0.5) / n,
                wrong_depth=(st.mean([v for v in vals if v < 0.5])
                             if any(v < 0.5 for v in vals) else float("nan")))


def q(vals, p):
    v = sorted(vals); k = (len(v) - 1) * p; f = int(k); c = min(f + 1, len(v) - 1)
    return v[f] + (v[c] - v[f]) * (k - f)


def check1(tag, rows, regime="misleading_majority", topology="all_channel"):
    print(f"\n{'='*104}\nCHECK 1 — collective-Brier decomposition | {tag} | {regime} / {topology}\n{'='*104}")
    for pool in ("log", "arith"):
        print(f"\n  [{pool} pool]")
        print(f"  {'condition':<22}{'n':>5}{'mean p':>9}{'SD':>8}{'Brier':>9}"
              f"{'(1-E[p])^2':>12}{'Var(p)':>9}{'var share':>11}{'wrong':>8}{'depth':>8}")
        base = None
        for content in ("none", "evidence_only", "conclusion_only", "both"):
            top = "baseline" if content == "none" else topology
            for rnd, lab in ((0, "r0"), (TERMINAL, "terminal")):
                v = list(cells(rows, regime, top, content, pool, rnd).values())
                if not v: continue
                if content == "none" and rnd != 0: continue
                d = decompose(v)
                share = d["var_term"] / d["brier"] if d["brier"] > 0 else float("nan")
                name = f"{content} {lab}" if content != "none" else "no communication"
                print(f"  {name:<22}{d['n']:>5}{d['mean']:>9.3f}{d['sd']:>8.3f}{d['brier']:>9.4f}"
                      f"{d['mean_term']:>12.4f}{d['var_term']:>9.4f}{share:>11.1%}"
                      f"{d['wrong']:>8.2f}{d['wrong_depth']:>8.3f}")
                if abs(d["resid"]) > 1e-12:
                    print(f"      !! identity residual {d['resid']:.2e}")
        # paired contrasts on shared seeds, terminal
        print(f"\n    paired terminal contrasts (shared seeds, {pool} pool):")
        ev = cells(rows, regime, topology, "evidence_only", pool, TERMINAL)
        cc = cells(rows, regime, topology, "conclusion_only", pool, TERMINAL)
        r0 = cells(rows, regime, topology, "conclusion_only", pool, 0)
        shared = sorted(set(ev) & set(cc))
        if shared:
            de = decompose([ev[s] for s in shared]); dc = decompose([cc[s] for s in shared])
            print(f"      evidence  n={de['n']:3d} Brier={de['brier']:.4f} = mean-term {de['mean_term']:.4f} + var {de['var_term']:.4f}")
            print(f"      conclusion n={dc['n']:3d} Brier={dc['brier']:.4f} = mean-term {dc['mean_term']:.4f} + var {dc['var_term']:.4f}")
            dB = dc['brier']-de['brier']
            print(f"      Δ Brier (concl−evid) = {dB:+.4f}  from mean-term {dc['mean_term']-de['mean_term']:+.4f} "
                  f"({(dc['mean_term']-de['mean_term'])/dB:.0%}) + variance {dc['var_term']-de['var_term']:+.4f} "
                  f"({(dc['var_term']-de['var_term'])/dB:.0%})")
        shared0 = sorted(set(r0) & set(cc))
        if shared0:
            d0 = decompose([r0[s] for s in shared0]); dc = decompose([cc[s] for s in shared0])
            dB = dc['brier']-d0['brier']
            print(f"      Δ Brier (conclusion terminal − its own round 0) = {dB:+.4f}  from mean-term "
                  f"{dc['mean_term']-d0['mean_term']:+.4f} ({(dc['mean_term']-d0['mean_term'])/dB:.0%})"
                  f" + variance {dc['var_term']-d0['var_term']:+.4f} ({(dc['var_term']-d0['var_term'])/dB:.0%})")


def check2(label, specs):
    print(f"\n{'='*104}\nCHECK 2 — diversity trajectory across the degree ladder | {label}\n{'='*104}")
    print(f"  {'degree':>7}  {'topology':<16}{'n':>5}{'e_bar r0':>10}{'e_bar term':>12}{'D r0':>9}{'D term':>9}{'ΔD':>9}{'acc':>7}")
    for tag, regime, content, tops in specs:
        rows = load(tag)
        for top in tops:
            sel = [r for r in rows if r["regime"] == regime and r["topology"] == top
                   and r["content_mode"] == content]
            r0 = [r for r in sel if int(r["round"]) == 0]
            rt = [r for r in sel if int(r["round"]) == TERMINAL]
            if not rt: continue
            e0 = st.mean(float(r["e_bar"]) for r in r0); et = st.mean(float(r["e_bar"]) for r in rt)
            d0 = st.mean(float(r["D"]) for r in r0);    dt = st.mean(float(r["D"]) for r in rt)
            acc = st.mean(int(r["acc_logodds"]) for r in rt)
            print(f"  {DEGREE.get(top,float('nan')):>7.2f}  {top:<16}{len(rt):>5}{e0:>10.3f}{et:>12.3f}"
                  f"{d0:>9.3f}{dt:>9.3f}{dt-d0:>+9.3f}{acc:>7.2f}")


def check3(tag, rows, regime="misleading_majority", topology="all_channel"):
    print(f"\n{'='*104}\nCHECK 3 — paired direction of movement, round 0 → terminal | {tag} | {regime}/{topology}\n{'='*104}")
    print("  Movement toward the misleading majority = a DECREASE in pooled P(truth).")
    for content in ("evidence_only", "conclusion_only"):
        a = cells(rows, regime, topology, content, "log", 0)
        b = cells(rows, regime, topology, content, "log", TERMINAL)
        s = sorted(set(a) & set(b))
        if not s: continue
        d = [b[k] - a[k] for k in s]
        tow = [x for x in d if x < 0]; awy = [x for x in d if x > 0]
        crossed = [k for k in s if a[k] >= 0.5 and b[k] < 0.5]
        print(f"\n  [{content}] n={len(s)}")
        print(f"    toward majority (Δ<0): {len(tow)/len(d):.0%}  mean Δ={st.mean(tow) if tow else 0:+.3f}  median Δ={st.median(tow) if tow else 0:+.3f}")
        print(f"    away   (Δ>0):          {len(awy)/len(d):.0%}  mean Δ={st.mean(awy) if awy else 0:+.3f}  median Δ={st.median(awy) if awy else 0:+.3f}")
        print(f"    net mean Δ = {st.mean(d):+.3f}   |Δ| mean = {st.mean(abs(x) for x in d):.3f}")
        print(f"    crossed to wrong side: {len(crossed)}/{len(s)} = {len(crossed)/len(s):.0%}")
        big = [k for k in s if b[k] < 0.5]
        if big:
            print(f"    terminal wrong-side trials: n={len(big)}  mean terminal P(truth)={st.mean(b[k] for k in big):.3f}"
                  f"  mean Δ={st.mean(b[k]-a[k] for k in big):+.3f}")


def check4(tag, rows, regime="misleading_majority", topology="all_channel"):
    print(f"\n{'='*104}\nCHECK 4 — terminal distribution shape of pooled P(truth) | {tag} | {regime}/{topology}\n{'='*104}")
    edges = [0,.05,.1,.2,.3,.4,.5,.6,.7,.8,.9,.95,1.0001]
    for content in ("evidence_only", "conclusion_only"):
        v = list(cells(rows, regime, topology, content, "log", TERMINAL).values())
        if not v: continue
        print(f"\n  [{content}] n={len(v)}")
        print("    quantiles  " + "  ".join(f"p{int(p*100):02d}={q(v,p):.3f}" for p in
                                            (.05,.10,.25,.50,.75,.90,.95)))
        h = collections.Counter()
        for x in v:
            for i in range(len(edges)-1):
                if edges[i] <= x < edges[i+1]: h[i] += 1; break
        print("    histogram:")
        for i in range(len(edges)-1):
            c = h[i]
            if c or True:
                print(f"      [{edges[i]:.2f},{min(edges[i+1],1.0):.2f})  {c:>3}  {'#'*c}")


if __name__ == "__main__":
    for tag in ("gpt-4.1-mini", "claude-sonnet-4-6"):
        rows = load(tag)
        check1(tag, rows); check3(tag, rows); check4(tag, rows)

    check2("concordant evidence sharing (failure 2), n=11 degree ladder", [
        ("gpt-4.1-mini_n11", "concordant", "evidence_only", ["circle", "ring_shortcuts", "all_channel"]),
        ("gpt-4.1-mini_overload", "concordant", "evidence_only", ["lattice_k4", "lattice_k6", "lattice_k8"]),
    ])
    check2("misleading-majority conclusion sharing (failure 1), n=11 degree ladder", [
        ("gpt-4.1-mini_n11", "misleading_majority", "conclusion_only", ["circle", "ring_shortcuts", "all_channel"]),
    ])
    check2("misleading-majority evidence sharing, n=11", [
        ("gpt-4.1-mini_n11", "misleading_majority", "evidence_only", ["circle", "ring_shortcuts", "all_channel"]),
    ])
    print("\n  n=11 high-fan-in cells, Brier decomposition:")
    check1("gpt-4.1-mini_n11", load("gpt-4.1-mini_n11"), "concordant", "all_channel")
