"""Two existing-data analyses for the PNAS Nexus R&R.

(A) NEIGHBOR FAN-IN (Reviewer 1 concern 6). The within-graph model shows an agent's OWN
    fan-in is associated with capitulation. Does a NEIGHBOR's fan-in add anything conditional
    on one's own? If not, the localization claim is cleaner: capitulation is a purely local
    function of incoming degree, with no measurable two-hop component.

(B) INBOX LOAD BEYOND DEGREE. Degree bundles several burdens at once (sender count, token
    volume, dependency tracking). This characterizes incoming message volume by degree and
    content mode, and asks whether volume varies WITHIN a degree level. Descriptive only:
    message length is endogenous to what agents choose to write.

System python3. Reads code/data/<tag>/{beliefs,rounds}.csv and full/transcripts.jsonl.
"""
import csv, json, collections, statistics as st, math
from pathlib import Path

HERE = Path(__file__).parent
TAG, RG, CM = "gpt-4.1-mini", "misleading_majority", "conclusion_only"
TOPOS = ["wheel", "hierarchy", "chain"]


def corr(xs, ys):
    if len(xs) < 3: return float("nan")
    mx, my = st.mean(xs), st.mean(ys)
    num = sum((x-mx)*(y-my) for x, y in zip(xs, ys))
    den = math.sqrt(sum((x-mx)**2 for x in xs) * sum((y-my)**2 for y in ys))
    return num/den if den else float("nan")


def partial(xs, ys, zs):
    """Correlation of x and y controlling for z, via residualization."""
    def resid(v, z):
        mz, mv = st.mean(z), st.mean(v)
        den = sum((a-mz)**2 for a in z)
        b = sum((a-mz)*(c-mv) for a, c in zip(z, v))/den if den else 0.0
        return [c - (mv + b*(a-mz)) for a, c in zip(z, v)]
    return corr(resid(xs, zs), resid(ys, zs))


def neighbor_fanin():
    print("="*100); print("(A) NEIGHBOR FAN-IN — does a neighbor's fan-in add anything beyond one's own?"); print("="*100)
    truth = {}
    for r in csv.DictReader(open(HERE/"data"/TAG/"rounds.csv")):
        if r["regime"] == RG and r["content_mode"] == CM and r["topology"] in TOPOS:
            truth[(r["topology"], r["seed"])] = r["true_state"]

    bel, fanin = collections.defaultdict(dict), {}
    for r in csv.DictReader(open(HERE/"data"/TAG/"beliefs.csv")):
        if r["regime"] != RG or r["content_mode"] != CM or r["topology"] not in TOPOS: continue
        k = (r["topology"], r["seed"], r["agent"]); rd = int(r["round"])
        bel[k][rd] = (float(r["belief_pA"]), float(r["llr"]))
        if rd == 1: fanin[k] = int(r["n_received"])

    # who each agent received from, round 1
    nbrs = {}
    for r in csv.DictReader(open(HERE/"data"/TAG/"messages.csv")):
        if r["regime"] != RG or r["content_mode"] != CM or r["topology"] not in TOPOS: continue
        if int(r["round"]) != 1: continue
        src = [s for s in (r.get("received_from") or "").split(";") if s]
        nbrs[(r["topology"], r["seed"], r["agent"])] = src

    own, nbr, cap, topo = [], [], [], []
    for k, byr in bel.items():
        tp, s, a = k
        if (tp, s) not in truth or k not in fanin or k not in nbrs: continue
        rs = sorted(byr); llr = byr[rs[0]][1]; bT = byr[rs[-1]][0]
        truth_A = truth[(tp, s)] == "A"
        if (llr > 0) != truth_A: continue          # strong-minority agents only
        nf = [fanin[(tp, s, x)] for x in nbrs[k] if (tp, s, x) in fanin]
        if not nf: continue
        own.append(fanin[k]); nbr.append(st.mean(nf))
        cap.append(1 if ((bT > 0.5) != truth_A) else 0); topo.append(tp)

    print(f"  strong-minority agents with resolvable neighbours: n={len(own)}")
    print(f"  own fan-in vs capitulation                    r = {corr(own, cap):+.3f}")
    print(f"  neighbour mean fan-in vs capitulation          r = {corr(nbr, cap):+.3f}")
    print(f"  neighbour fan-in vs capitulation | own fan-in  r = {partial(nbr, cap, own):+.3f}   <-- the test")
    print(f"  own fan-in vs capitulation | neighbour fan-in  r = {partial(own, cap, nbr):+.3f}")
    print(f"  collinearity: own vs neighbour fan-in          r = {corr(own, nbr):+.3f}")
    print("\n  within-topology (demeaned by graph):")
    for tp in TOPOS:
        idx = [i for i, t in enumerate(topo) if t == tp]
        if len(idx) < 10: continue
        o = [own[i] for i in idx]; nb = [nbr[i] for i in idx]; c = [cap[i] for i in idx]
        print(f"    {tp:<10} n={len(idx):>4}  own r={corr(o,c):+.3f}  nbr r={corr(nb,c):+.3f}  "
              f"nbr|own r={partial(nb,c,o):+.3f}")


def inbox_load():
    print("\n" + "="*100); print("(B) INBOX LOAD — incoming volume by degree and content mode"); print("="*100)
    for tag, regime in (("gpt-4.1-mini_n11", "misleading_majority"), ("gpt-4.1-mini_n11", "concordant")):
        f = HERE/"data"/tag/"full"/"transcripts.jsonl"
        if not f.exists(): print(f"  {tag}: no transcripts"); continue
        agg = collections.defaultdict(list)
        for line in open(f):
            o = json.loads(line)
            if o["regime"] != regime: continue
            for rd in o["rounds"]:
                if rd["round"] == 0: continue
                for ag in rd["agents"]:
                    rec = ag.get("received") or []
                    if not rec: continue
                    agg[(o["topology"], o["content_mode"])].append(
                        (len(rec), sum(len(x["text"]) for x in rec)))
        if not agg: continue
        print(f"\n  --- {tag} / {regime} ---")
        print(f"  {'topology':<18}{'content':<18}{'obs':>7}{'senders':>10}{'chars in':>11}{'chars/sender':>14}{'SD chars':>10}")
        for k in sorted(agg):
            v = agg[k]
            ns = [a for a, _ in v]; ch = [b for _, b in v]
            print(f"  {k[0]:<18}{k[1]:<18}{len(v):>7}{st.mean(ns):>10.1f}{st.mean(ch):>11.0f}"
                  f"{st.mean(ch)/max(st.mean(ns),1e-9):>14.0f}{st.pstdev(ch):>10.0f}")
        # within-degree variation
        print("\n  within-degree variation in incoming characters (same topology+content):")
        for k in sorted(agg):
            ch = [b for _, b in agg[k]]
            if len(ch) < 5: continue
            v = sorted(ch)
            print(f"    {k[0]:<18}{k[1]:<18} p10={v[len(v)//10]:>6} p50={v[len(v)//2]:>6} "
                  f"p90={v[int(len(v)*.9)]:>6}  CV={st.pstdev(ch)/max(st.mean(ch),1e-9):.2f}")


if __name__ == "__main__":
    neighbor_fanin()
    inbox_load()
