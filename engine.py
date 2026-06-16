"""Network message-passing engine: fixed rounds, no early stop, full transcript
logging (every signal, received message with source, generated message, and
belief, per round) so a human can review/code any trial end to end."""
from __future__ import annotations
from typing import Dict, List

import agents
from tasks import Trial


def receives_from(topology: str, n: int) -> Dict[int, List[int]]:
    """Map each node -> list of nodes whose messages it receives each round."""
    if topology == "baseline":
        return {i: [] for i in range(n)}
    if topology == "all_channel":
        return {i: [j for j in range(n) if j != i] for i in range(n)}
    if topology == "wheel":                      # node 0 = hub
        adj = {0: [j for j in range(1, n)]}
        for i in range(1, n):
            adj[i] = [0]
        return adj
    if topology == "circle":
        return {i: [(i - 1) % n, (i + 1) % n] for i in range(n)}
    if topology == "chain":                       # open line 0-1-...-(n-1), bidirectional
        adj = {}
        for i in range(n):
            nb = ([i - 1] if i > 0 else []) + ([i + 1] if i < n - 1 else [])
            adj[i] = nb
        return adj
    if topology == "hierarchy":                   # bidirectional tree (binary-heap layout; node 0 = root)
        adj = {i: [] for i in range(n)}
        for i in range(1, n):
            p = (i - 1) // 2
            adj[i].append(p); adj[p].append(i)
        return adj
    if topology == "ring_shortcuts":              # small-world: ring backbone + 3 fixed long-range shortcuts
        adj = {i: [(i - 1) % n, (i + 1) % n] for i in range(n)}
        for start in (0, n // 3, 2 * n // 3):     # 3 deterministic chords spanning ~half the ring
            a, b = start, (start + n // 2) % n
            if b not in adj[a]:
                adj[a].append(b); adj[b].append(a)
        return adj
    if topology.startswith("lattice_k"):
        # Ring lattice of even degree k: each node connects to k/2 nearest neighbors per side.
        # The "_ws" suffix applies deterministic degree-PRESERVING double-edge swaps, injecting
        # long-range edges (shorter path length) while holding every node's degree fixed. This
        # dissociates per-node degree (fan-in) from global path length: lattice_k4 vs lattice_k4_ws
        # have identical degree but different diameter; the k4/k6/k8 series varies degree.
        base = topology[len("lattice_k"):]
        ws = base.endswith("_ws")
        k = int(base[:-3] if ws else base)
        m = k // 2
        adj = {i: set((i + off) % n for off in range(-m, m + 1) if off != 0) for i in range(n)}
        if ws:
            swaps = [((0, 1), (n // 2, n // 2 + 1)),
                     ((2, 3), (n // 2 + 2, n // 2 + 3)),
                     ((4, 5), (n - 2, n - 1))]
            for (a, b), (c, d) in swaps:
                a, b, c, d = a % n, b % n, c % n, d % n
                if (b in adj[a] and d in adj[c] and len({a, b, c, d}) == 4
                        and d not in adj[a] and b not in adj[c]):
                    adj[a].discard(b); adj[b].discard(a)
                    adj[c].discard(d); adj[d].discard(c)
                    adj[a].add(d); adj[d].add(a)
                    adj[c].add(b); adj[b].add(c)
        return {i: sorted(adj[i]) for i in range(n)}
    raise ValueError(f"unknown topology: {topology}")


def run_trial(trial: Trial, topology: str, content_mode: str, r_comm: int = 3,
              model: str = None, seed_conclusion: str = None) -> dict:
    n = len(trial.signals)
    adj = receives_from(topology, n)
    sig_desc = [s.describe() for s in trial.signals]
    # Optional do-operator: a planted "first-mover" conclusion visible to all agents
    # from round 0, holding their private evidence fixed. Set to "A"/"B" to test
    # whether terminal collective belief causally follows the injected conclusion.
    seed_pairs = ([(-1, f"I am highly confident the hidden state is {seed_conclusion}.")]
                  if seed_conclusion else [])
    seed_dicts = [{"src": s, "text": t} for s, t in seed_pairs]
    seed_block = agents.format_received(seed_pairs)

    usage_total = {"prompt_tokens": 0, "completion_tokens": 0}
    def acc(u):
        usage_total["prompt_tokens"] += u.get("prompt_tokens", 0)
        usage_total["completion_tokens"] += u.get("completion_tokens", 0)

    rounds = []   # full transcript: list of {round, agents:[{id,belief,received,message}]}

    # --- round 0: private commitment (own signal only, no messages) ---
    r0_agents = []
    for i in range(n):
        b, u = agents.read_belief(sig_desc[i], seed_block, model); acc(u)
        r0_agents.append({"id": i, "belief": b, "received": list(seed_dicts), "message": None})
    rounds.append({"round": 0, "agents": r0_agents})

    has_comms = topology != "baseline" and r_comm > 0
    if has_comms:
        # initial messages from round-0 view
        messages = {}
        for i in range(n):
            m, u = agents.make_message(sig_desc[i], seed_block, content_mode, model); acc(u)
            messages[i] = m
        # record round-0 messages onto the round-0 agents (what they emitted)
        for a in r0_agents:
            a["message"] = messages[a["id"]]

        for r in range(1, r_comm + 1):
            inboxes = {i: seed_dicts + [{"src": s, "text": messages[s]} for s in adj[i]] for i in range(n)}
            r_agents = []
            for i in range(n):
                rb = agents.format_received([(d["src"], d["text"]) for d in inboxes[i]])
                b, u = agents.read_belief(sig_desc[i], rb, model); acc(u)
                r_agents.append({"id": i, "belief": b, "received": inboxes[i], "message": None})
            # generate next messages (skip on final round)
            if r < r_comm:
                new_msgs = {}
                for i in range(n):
                    rb = agents.format_received([(d["src"], d["text"]) for d in inboxes[i]])
                    m, u = agents.make_message(sig_desc[i], rb, content_mode, model); acc(u)
                    new_msgs[i] = m
                for a in r_agents:
                    a["message"] = new_msgs[a["id"]]
                messages = new_msgs
            rounds.append({"round": r, "agents": r_agents})

    agents_meta = [{
        "id": i, "signal_label": trial.signals[i].label,
        "p_a": trial.signals[i].p_a, "p_b": trial.signals[i].p_b,
        "llr": round(trial.signals[i].llr, 4),
        "private_post_a": round(trial.private_post_a(i), 4),
    } for i in range(n)]

    return {
        "seed": trial.seed, "regime": trial.regime, "topology": topology,
        "content_mode": content_mode, "n_agents": n, "r_comm": r_comm,
        "true_state": trial.true_state,
        "full_info_post_a": round(trial.full_info_post_a, 4),
        "naive_vote_a": round(trial.naive_vote_a, 4),
        "agents_meta": agents_meta,
        "rounds": rounds,
        "usage": usage_total,
    }


def beliefs_at(result: dict, round_idx: int) -> List[float]:
    return [a["belief"] for a in result["rounds"][round_idx]["agents"]]
