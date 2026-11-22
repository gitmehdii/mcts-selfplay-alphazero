#!/usr/bin/env python3
"""Niveau 2 — Comparer l'agent avec réseau (AlphaZero) et sans réseau (MCTS pur).

    python experiments/compare_az.py --game connect4 --run runs/connect4
    python experiments/compare_az.py --game chess:mini --run runs/chess_mini --games 40

Produit dans ``results/<jeu>/`` :
  - az_comparison.csv / .md : tableau des matchs ;
  - training_curves.png    : pertes, entropie de la policy et score d'évaluation au fil des itérations ;
  - checkpoint_progress.png: force de chaque checkpoint contre un adversaire fixe (effet du self-play).
"""
from __future__ import annotations

import argparse
import csv
import glob
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from arena import match  # noqa: E402

PURE = {
    "connect4": lambda n: f"mcts:sims={n}",
    "chess": lambda n: f"mcts:sims={n},max_depth=20,rollout=captures",
}


def read_metrics(run):
    path = os.path.join(run, "metrics.csv")
    with open(path) as f:
        return list(csv.DictReader(f))


def plot_training(run, out):
    rows = read_metrics(run)
    it = [int(r["iteration"]) for r in rows]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    axes[0].plot(it, [float(r["loss_policy"]) for r in rows], label="perte policy (CE)")
    axes[0].plot(it, [float(r["loss_value"]) for r in rows], label="perte value (MSE)")
    axes[0].set_title("Pertes d'entraînement"); axes[0].legend()
    axes[1].plot(it, [float(r["mean_len"]) for r in rows], c="C2", label="longueur moyenne")
    ax2 = axes[1].twinx()
    ax2.plot(it, [float(r["p1_win"]) for r in rows], c="C3", ls="--", label="victoires J1")
    ax2.plot(it, [float(r["draw"]) for r in rows], c="C4", ls=":", label="nuls")
    ax2.set_ylim(0, 1)
    axes[1].set_title("Parties de self-play"); axes[1].legend(loc="upper left"); ax2.legend(loc="upper right")
    ev = [(int(r["iteration"]), float(r["eval_score"])) for r in rows if r["eval_score"] not in ("", None)]
    if ev:
        axes[2].plot(*zip(*ev), marker="o")
    cfg_opp = ""
    try:
        import json

        cfg_opp = json.load(open(os.path.join(run, "config.json")))["eval_opponent"]
    except Exception:
        pass
    axes[2].set_ylim(0, 1.05); axes[2].axhline(0.5, ls=":", c="grey")
    axes[2].set_title(f"Score contre {cfg_opp}", fontsize=9)
    for a in axes:
        a.set_xlabel("itération"); a.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(os.path.join(out, "training_curves.png"), dpi=130); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", default="connect4")
    ap.add_argument("--run", default=None)
    ap.add_argument("--ckpt", default=None, help="checkpoint à évaluer (défaut : <run>/best.pt)")
    ap.add_argument("--games", type=int, default=100)
    ap.add_argument("--sims", type=int, default=100, help="simulations de l'agent AlphaZero")
    ap.add_argument("--progress-every", type=int, default=0, help="évaluer un checkpoint sur N (0 = non)")
    ap.add_argument("--workers", type=int, default=None)
    args = ap.parse_args()

    family = "chess" if args.game.startswith("chess") else "connect4"
    run = args.run or os.path.join(ROOT, "runs", args.game.replace(":", "_"))
    ckpt = args.ckpt or os.path.join(run, "best.pt")
    untrained = os.path.join(run, "iter_000.pt")
    out = os.path.join(ROOT, "results", args.game.replace(":", "_"))
    os.makedirs(out, exist_ok=True)
    pure = PURE[family]
    S, G = args.sims, args.games

    az = f"az:ckpt={ckpt},sims={S}"
    duels = [
        ("AZ vs aléatoire", az, "random"),
        ("AZ vs MCTS pur, mêmes simulations", az, pure(S)),
        ("AZ vs MCTS pur, 5x plus de simulations", az, pure(5 * S)),
        ("AZ vs MCTS pur, 10x plus de simulations", az, pure(10 * S)),
        ("Réseau seul (0 sim) vs aléatoire", f"az:ckpt={ckpt},sims=0", "random"),
        ("Réseau seul (0 sim) vs MCTS pur", f"az:ckpt={ckpt},sims=0", pure(S)),
        ("Réseau NON entraîné + PUCT vs MCTS pur", f"az:ckpt={untrained},sims={S}", pure(S)),
        ("AZ entraîné vs réseau NON entraîné", az, f"az:ckpt={untrained},sims={S}"),
    ]
    rows = []
    for label, a, b in duels:
        res = match(args.game, a, b, n_games=G, workers=args.workers, seed=123)
        print(f"{label:45s} {res}")
        lo, hi = res.ci95()
        rows.append({"duel": label, "agent_a": res.name_a, "agent_b": res.name_b, "wins": res.wins,
                     "draws": res.draws, "losses": res.losses, "score": round(res.score, 3),
                     "ci_low": round(lo, 3), "ci_high": round(hi, 3), "elo": round(res.elo_diff())})
    with open(os.path.join(out, "az_comparison.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    with open(os.path.join(out, "az_comparison.md"), "w") as f:
        f.write(f"| Duel ({G} parties, couleurs alternées) | V | N | D | Score A [IC95] | Elo A-B |\n|---|---|---|---|---|---|\n")
        for r in rows:
            f.write(f"| {r['duel']} | {r['wins']} | {r['draws']} | {r['losses']} | "
                    f"{r['score']:.0%} [{r['ci_low']:.0%}-{r['ci_high']:.0%}] | {r['elo']:+d} |\n")

    if os.path.exists(os.path.join(run, "metrics.csv")):
        plot_training(run, out)

    if args.progress_every:
        ckpts = sorted(glob.glob(os.path.join(run, "iter_*.pt")))
        ckpts = [c for c in ckpts if int(c[-6:-3]) % args.progress_every == 0]
        prog = []
        for c in ckpts:
            res = match(args.game, f"az:ckpt={c},sims={S}", pure(S), n_games=G // 2, workers=args.workers, seed=7)
            prog.append((int(c[-6:-3]), res))
            print(f"  checkpoint {c[-6:-3]} : {res}")
        fig, ax = plt.subplots(figsize=(6.5, 4))
        xs = [p[0] for p in prog]
        ys = [p[1].score for p in prog]
        lo = [p[1].score - p[1].ci95()[0] for p in prog]
        hi = [p[1].ci95()[1] - p[1].score for p in prog]
        ax.errorbar(xs, ys, yerr=[lo, hi], marker="o", capsize=3)
        ax.axhline(0.5, ls=":", c="grey"); ax.set_ylim(0, 1.05); ax.grid(alpha=.3)
        ax.set_xlabel("itération de self-play"); ax.set_ylabel(f"score contre {pure(S)}")
        ax.set_title(f"Effet du self-play : AZ({S} sims) vs MCTS pur ({S} sims)")
        fig.tight_layout(); fig.savefig(os.path.join(out, "checkpoint_progress.png"), dpi=130); plt.close(fig)
        with open(os.path.join(out, "checkpoint_progress.csv"), "w") as f:
            f.write("iteration,score,ci_low,ci_high\n")
            for i, r in prog:
                f.write(f"{i},{r.score:.3f},{r.ci95()[0]:.3f},{r.ci95()[1]:.3f}\n")
    print(f"Résultats dans {out}")


if __name__ == "__main__":
    main()
