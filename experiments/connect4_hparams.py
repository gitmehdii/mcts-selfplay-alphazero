#!/usr/bin/env python3
"""Niveau 1 — étude des hyperparamètres du MCTS pur à Puissance 4.

Expériences (résultats CSV + figures dans ``results/connect4/``) :
  E1  taux de victoire contre l'aléatoire et contre l'heuristique selon le nombre de simulations
  E2  influence de la constante d'exploration c (UCB1) contre un MCTS de référence (c = sqrt(2))
  E3  « courbe de force » : MCTS(n) contre un MCTS de référence à 1000 simulations
  E4  politique de rollout : aléatoire vs heuristique, à simulations égales puis à temps égal
  E5  avantage du premier joueur (MCTS contre lui-même)
  E6  coût : temps par coup selon le nombre de simulations

Usage :
    python experiments/connect4_hparams.py            # complet (~10-20 min sur 16 cœurs)
    python experiments/connect4_hparams.py --quick    # version rapide pour tester
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from agents import make_agent  # noqa: E402
from arena import match  # noqa: E402
from games import make_game  # noqa: E402

OUT = os.path.join(ROOT, "results", "connect4")
SQRT2 = math.sqrt(2)


def save_csv(name, rows):
    path = os.path.join(OUT, f"{name}.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    return path


def row(res, **params):
    lo, hi = res.ci95()
    return {**params, "wins": res.wins, "draws": res.draws, "losses": res.losses, "n": res.n,
            "score": round(res.score, 4), "ci_low": round(lo, 4), "ci_high": round(hi, 4),
            "mean_plies": round(sum(res.lengths) / max(len(res.lengths), 1), 1), "seconds": round(res.seconds, 1)}


def errplot(ax, xs, rows, label, marker="o"):
    ys = [r["score"] for r in rows]
    err = [[r["score"] - r["ci_low"] for r in rows], [r["ci_high"] - r["score"] for r in rows]]
    ax.errorbar(xs, ys, yerr=err, marker=marker, capsize=3, label=label)


def run(spec_a, spec_b, n, seed=0):
    res = match("connect4", spec_a, spec_b, n_games=n, seed=seed)
    print("  ", res)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--only", default="", help="ex. E1,E3")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    G = 40 if args.quick else 200          # parties par point de mesure
    only = set(args.only.upper().split(",")) - {""}
    want = lambda e: not only or e in only  # noqa: E731
    summary = []

    # ---------------------------------------------------------------- E1
    if want("E1"):
        print("E1 : MCTS(n) contre Random / Heuristic")
        sims = [1, 5, 10, 25, 50, 100, 200, 500, 1000]
        rows_r, rows_h = [], []
        for n in sims:
            rows_r.append(row(run(f"mcts:sims={n}", "random", G), sims=n, opponent="random"))
            rows_h.append(row(run(f"mcts:sims={n}", "heuristic", G), sims=n, opponent="heuristic"))
        save_csv("E1_sims_vs_baselines", rows_r + rows_h)
        fig, ax = plt.subplots(figsize=(6.5, 4))
        errplot(ax, sims, rows_r, "contre Random")
        errplot(ax, sims, rows_h, "contre Heuristic (gagne/bloque)", marker="s")
        ax.set_xscale("log"); ax.set_xlabel("simulations par coup"); ax.set_ylabel("score du MCTS (V=1, N=0.5)")
        ax.set_ylim(0, 1.05); ax.axhline(0.5, ls=":", c="grey"); ax.grid(alpha=.3); ax.legend()
        ax.set_title("E1 — Force du MCTS pur selon le budget de simulations")
        fig.tight_layout(); fig.savefig(os.path.join(OUT, "E1_sims_vs_baselines.png"), dpi=130); plt.close(fig)
        summary.append(("E1", rows_r + rows_h))

    # ---------------------------------------------------------------- E2
    if want("E2"):
        print("E2 : constante d'exploration c (300 sims) contre MCTS(c=sqrt 2, 300 sims)")
        cs = [0.0, 0.1, 0.25, 0.5, 1.0, SQRT2, 2.0, 3.0, 5.0]
        rows = []
        for c in cs:
            rows.append(row(run(f"mcts:sims=300,c={c}", f"mcts:sims=300,c={SQRT2}", G), c=round(c, 3), sims=300))
        save_csv("E2_exploration_constant", rows)
        fig, ax = plt.subplots(figsize=(6.5, 4))
        errplot(ax, cs, rows, "MCTS(c) vs MCTS(c=√2), 300 sims")
        ax.axvline(SQRT2, ls="--", c="grey", lw=.8); ax.axhline(0.5, ls=":", c="grey")
        ax.set_xlabel("constante d'exploration c"); ax.set_ylabel("score"); ax.set_ylim(0, 1.05)
        ax.grid(alpha=.3); ax.legend(); ax.set_title("E2 — Influence de la constante UCB1")
        fig.tight_layout(); fig.savefig(os.path.join(OUT, "E2_exploration_constant.png"), dpi=130); plt.close(fig)
        summary.append(("E2", rows))

    # ---------------------------------------------------------------- E3
    if want("E3"):
        print("E3 : MCTS(n) contre MCTS(1000)")
        sims = [50, 100, 250, 500, 1000, 2000, 4000, 8000]
        rows = [row(run(f"mcts:sims={n}", "mcts:sims=1000", G // 2 if n >= 4000 else G), sims=n) for n in sims]
        save_csv("E3_sims_vs_mcts1000", rows)
        fig, ax = plt.subplots(figsize=(6.5, 4))
        errplot(ax, sims, rows, "MCTS(n) vs MCTS(1000)")
        ax.set_xscale("log"); ax.axhline(0.5, ls=":", c="grey"); ax.axvline(1000, ls="--", c="grey", lw=.8)
        ax.set_xlabel("simulations par coup"); ax.set_ylabel("score"); ax.set_ylim(0, 1.05); ax.grid(alpha=.3)
        ax.legend(); ax.set_title("E3 — Rendements du budget de calcul")
        fig.tight_layout(); fig.savefig(os.path.join(OUT, "E3_sims_vs_mcts1000.png"), dpi=130); plt.close(fig)
        summary.append(("E3", rows))

    # ---------------------------------------------------------------- E4
    if want("E4"):
        print("E4 : rollouts heuristiques vs aléatoires")
        rows = []
        for n in (100, 500, 2000):
            rows.append(row(run(f"mcts:sims={n},rollout=heuristic", f"mcts:sims={n},rollout=random", G),
                            budget=f"{n} sims"))
        for t in (0.05, 0.2):
            rows.append(row(run(f"mcts:time_limit={t},rollout=heuristic", f"mcts:time_limit={t},rollout=random", G),
                            budget=f"{t}s / coup"))
        save_csv("E4_rollout_policy", rows)
        fig, ax = plt.subplots(figsize=(6.5, 4))
        xs = list(range(len(rows)))
        ax.bar(xs, [r["score"] for r in rows], yerr=[[r["score"] - r["ci_low"] for r in rows],
                                                     [r["ci_high"] - r["score"] for r in rows]], capsize=4)
        ax.set_xticks(xs, [r["budget"] for r in rows]); ax.axhline(0.5, ls=":", c="grey"); ax.set_ylim(0, 1.05)
        ax.set_ylabel("score du MCTS à rollouts heuristiques"); ax.grid(alpha=.3, axis="y")
        ax.set_title("E4 — Rollouts heuristiques vs aléatoires")
        fig.tight_layout(); fig.savefig(os.path.join(OUT, "E4_rollout_policy.png"), dpi=130); plt.close(fig)
        summary.append(("E4", rows))

    # ---------------------------------------------------------------- E5
    if want("E5"):
        print("E5 : avantage du premier joueur (MCTS contre lui-même)")
        rows = []
        for n in (100, 1000, 4000):
            # même agent des deux côtés : on regarde le taux de victoire de celui qui commence
            res = match("connect4", f"mcts:sims={n}", f"mcts:sims={n}", n_games=G, seed=1)
            first_wins = res.first_mover_wins
            rows.append({"sims": n, "n": res.n, "draws": res.draws, "first_player_wins": first_wins,
                         "first_player_win_rate": round(first_wins / res.n, 4)})
            print(f"   sims={n}: le premier joueur gagne {first_wins}/{res.n} parties, {res.draws} nuls")
        save_csv("E5_first_player", rows)
        summary.append(("E5", rows))

    # ---------------------------------------------------------------- E6
    if want("E6"):
        print("E6 : temps de calcul par coup (position initiale)")
        rows = []
        for n in (100, 1000, 10000):
            for pol in ("random", "heuristic"):
                a = make_agent(f"mcts:sims={n},rollout={pol}", seed=0)
                t0 = time.perf_counter()
                reps = 3
                for _ in range(reps):
                    a.select_move(make_game("connect4"))
                dt = (time.perf_counter() - t0) / reps
                rows.append({"sims": n, "rollout": pol, "seconds_per_move": round(dt, 4),
                             "sims_per_second": round(n / dt)})
                print(f"   {n} sims, rollout {pol}: {dt:.3f}s/coup ({n / dt:,.0f} sims/s)")
        save_csv("E6_timing", rows)
        summary.append(("E6", rows))

    print(f"\nRésultats écrits dans {OUT}")


if __name__ == "__main__":
    main()
