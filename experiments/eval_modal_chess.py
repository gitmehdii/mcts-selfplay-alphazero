#!/usr/bin/env python3
"""Évaluation locale du modèle d’échecs complets entraîné sur Modal (``modal_app.py``).

    modal volume get rl-alphazero runs/chess_standard    runs/modal/
    modal volume get rl-alphazero runs/chess_standard_v2 runs/modal/
    python experiments/eval_modal_chess.py --games 40

Produit dans ``results/chess_standard/`` : comparaison.md/.csv (duels), fins_de_partie.md
(comment l’agent gagne contre l’aléatoire), progression.png (scores d’évaluation au fil des runs).
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from multiprocessing import Pool

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from arena import match  # noqa: E402

GAME = "chess:standard"
RUNS = os.path.join(ROOT, "runs", "modal")
OUT = os.path.join(ROOT, "results", "chess_standard")
PURE = lambda n: f"mcts:sims={n},max_depth=20,rollout=captures"  # noqa: E731


def _ending(args):
    ckpt, sims, i = args
    import torch

    torch.set_num_threads(1)
    from agents import make_agent
    from arena import play_game
    from games import make_game

    g = make_game(GAME)
    az, rd = make_agent(f"az:ckpt={ckpt},sims={sims}", seed=i), make_agent("random", seed=10_000 + i)
    side = i % 2
    w = play_game(g, [az, rd] if side == 0 else [rd, az])
    b = g.board
    if b.is_checkmate():
        why = "mat"
    elif b.is_stalemate():
        why = "pat"
    elif b.is_insufficient_material():
        why = "matériel insuffisant"
    elif g.keys.count(g.keys[-1]) >= 3:
        why = "répétition"
    elif b.halfmove_clock >= 150:
        why = "75 coups"
    else:
        why = "arbitrage à 200 demi-coups"
    res = "victoire" if w == side else ("nulle" if w == -1 else "défaite")
    return res, why, g.ply


def read_metrics(path):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return list(csv.DictReader(f))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=40)
    ap.add_argument("--sims", type=int, default=64)
    ap.add_argument("--workers", type=int, default=14)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    v2 = os.path.join(RUNS, "chess_standard_v2")
    final = next(p for p in (os.path.join(v2, "final.pt"), os.path.join(v2, "latest.pt")) if os.path.exists(p))
    untrained = os.path.join(RUNS, "chess_standard", "iter_000.pt")
    S, G = args.sims, args.games
    az = f"az:ckpt={final},sims={S}"

    duels = [
        ("AZ vs aléatoire", az, "random"),
        (f"AZ vs MCTS pur, mêmes simulations ({S})", az, PURE(S)),
        (f"AZ vs MCTS pur, 5× plus de simulations ({5 * S})", az, PURE(5 * S)),
        (f"AZ vs MCTS pur, 10× plus de simulations ({10 * S})", az, PURE(10 * S)),
        ("Réseau seul (0 sim) vs aléatoire", f"az:ckpt={final},sims=0", "random"),
        (f"Réseau NON entraîné + PUCT vs MCTS pur ({S})", f"az:ckpt={untrained},sims={S}", PURE(S)),
        ("AZ entraîné vs AZ non entraîné", az, f"az:ckpt={untrained},sims={S}"),
    ]
    rows = []
    for label, a, b in duels:
        r = match(GAME, a, b, n_games=G, workers=args.workers, seed=2026)
        lo, hi = r.ci95()
        print(f"{label:48s} {r}", flush=True)
        rows.append({"duel": label, "wins": r.wins, "draws": r.draws, "losses": r.losses, "score": round(r.score, 3),
                     "ci_low": round(lo, 3), "ci_high": round(hi, 3), "elo": round(r.elo_diff())})
    with open(os.path.join(OUT, "comparaison.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    with open(os.path.join(OUT, "comparaison.md"), "w") as f:
        f.write(f"Modèle : `{os.path.relpath(final, ROOT)}` · {G} parties par duel, couleurs alternées\n\n")
        f.write("| Duel | V | N | D | Score [IC95] | Elo |\n|---|---|---|---|---|---|\n")
        for r in rows:
            f.write(f"| {r['duel']} | {r['wins']} | {r['draws']} | {r['losses']} | "
                    f"{r['score']:.0%} [{r['ci_low']:.0%}-{r['ci_high']:.0%}] | {r['elo']:+d} |\n")

    # comment se terminent les parties contre l’aléatoire
    with Pool(args.workers) as p:
        ends = p.map(_ending, [(final, S, i) for i in range(G)])
    tally = {}
    for res, why, _ in ends:
        tally[(res, why)] = tally.get((res, why), 0) + 1
    with open(os.path.join(OUT, "fins_de_partie.md"), "w") as f:
        f.write(f"AZ ({S} sims) contre l’aléatoire, {G} parties\n\n| Résultat | Fin de partie | Parties |\n|---|---|---|\n")
        for (res, why), n in sorted(tally.items(), key=lambda kv: -kv[1]):
            f.write(f"| {res} | {why} | {n} |\n")
    print("fins de partie :", tally, flush=True)

    # progression : évaluations des deux runs (le run 2 reprend les poids du run 1)
    m1 = read_metrics(os.path.join(RUNS, "chess_standard", "metrics.csv"))
    m2 = read_metrics(os.path.join(RUNS, "chess_standard_v2", "metrics.csv"))
    games1 = max((int(r["total_games"]) for r in m1), default=0)
    pts = {"random": [], "mcts": []}
    for rows_, offset in ((m1, 0), (m2, games1)):
        for r in rows_:
            for k in pts:
                if r.get(f"score_vs_{k}"):
                    pts[k].append((offset + int(r["total_games"]), float(r[f"score_vs_{k}"])))
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.plot(*zip(*pts["random"]), marker="o", label="contre l’aléatoire")
    ax.plot(*zip(*pts["mcts"]), marker="s", label=f"contre le MCTS pur ({S} sims)")
    ax.axvline(games1, ls="--", c="grey")
    ax.text(games1, 0.05, " correction de la règle de répétition", fontsize=9, color="grey")
    ax.set_ylim(0, 1.05)
    ax.axhline(0.5, ls=":", c="grey")
    ax.set_xlabel("parties de self-play (cumul)")
    ax.set_ylabel("score")
    ax.set_title("Échecs complets : AlphaZero entraîné sur Modal")
    ax.grid(alpha=.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "progression.png"), dpi=130)
    with open(os.path.join(OUT, "progression.csv"), "w") as f:
        f.write("adversaire,parties,score\n")
        for k, v in pts.items():
            for g_, s in v:
                f.write(f"{k},{g_},{s}\n")
    print("Résultats dans", OUT)


if __name__ == "__main__":
    main()
