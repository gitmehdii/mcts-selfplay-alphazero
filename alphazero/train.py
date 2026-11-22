#!/usr/bin/env python3
"""Boucle AlphaZero simplifiée : self-play -> entraînement -> évaluation, en boucle.

    python -m alphazero.train --game connect4 --out runs/c4
    python -m alphazero.train --game chess:mini --out runs/chess_mini --sims 64 --channels 64 --blocks 6

À chaque itération :
  1. le réseau courant génère ``--games`` parties en self-play (PUCT + bruit de Dirichlet) ;
  2. les exemples (s, π, z) rejoignent un replay buffer (fenêtre glissante) ;
  3. le réseau est entraîné ``--steps`` mini-lots : L = (z - v)² - π·log p + c||θ||² ;
  4. tous les ``--eval-every`` itérations, l'agent (réseau + PUCT) affronte un
     adversaire de référence (MCTS pur par défaut) ; le meilleur réseau est gardé
     dans ``best.pt``.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
import time
from collections import deque

import numpy as np
import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from alphazero.network import AZNet, NetEvaluator  # noqa: E402
from alphazero.selfplay import SelfPlayConfig, generate_games  # noqa: E402
from games import make_game  # noqa: E402

DEFAULTS = {
    # jeu : (sims, temp_moves, dirichlet_alpha, adversaire d'évaluation)
    "connect4": dict(sims=100, temp_moves=10, alpha=1.0, eval_opponent="mcts:sims=1000"),
    "chess": dict(sims=64, temp_moves=20, alpha=0.3, eval_opponent="mcts:sims=200,max_depth=20,rollout=captures"),
}


def make_batch(buffer, idx, game, device):
    planes = np.stack([buffer[i][0] for i in idx]).astype(np.float32)
    pi = np.zeros((len(idx), game.action_size), dtype=np.float32)
    z = np.array([buffer[i][3] for i in idx], dtype=np.float32)
    for k, i in enumerate(idx):
        pi[k, buffer[i][1]] = buffer[i][2]
    # augmentation par symétries (Puissance 4 : miroir gauche/droite)
    for k in range(len(idx)):
        syms = game.symmetries(planes[k], pi[k])
        if len(syms) > 1:
            planes[k], pi[k] = syms[random.randrange(len(syms))]
    to = lambda a: torch.from_numpy(a).to(device, non_blocking=True)  # noqa: E731
    return to(planes), to(pi), to(z)


def train_steps(net, opt, buffer, game, steps, batch_size, device):
    net.train()
    lp = lv = ent = 0.0
    for _ in range(steps):
        idx = np.random.randint(len(buffer), size=batch_size)
        x, pi, z = make_batch(buffer, idx, game, device)
        logits, v = net(x)
        logp = F.log_softmax(logits, dim=1)
        loss_p = -(pi * logp).sum(1).mean()
        loss_v = F.mse_loss(v, z)
        loss = loss_p + loss_v
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        lp += loss_p.item()
        lv += loss_v.item()
        ent += -(logp.exp() * logp).sum(1).mean().item()
    net.eval()
    return lp / steps, lv / steps, ent / steps


def evaluate(ckpt, game_name, opponent, n_games, sims, workers, seed):
    from arena import match

    res = match(game_name, f"az:ckpt={ckpt},sims={sims}", opponent, n_games=n_games, workers=workers, seed=seed)
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--game", default="connect4")
    ap.add_argument("--out", default=None)
    ap.add_argument("--iterations", type=int, default=30)
    ap.add_argument("--games", type=int, default=256, help="parties de self-play par itération")
    ap.add_argument("--sims", type=int, default=None, help="simulations PUCT par coup en self-play")
    ap.add_argument("--parallel", type=int, default=256, help="parties jouées simultanément (taille de lot)")
    ap.add_argument("--temp-moves", type=int, default=None)
    ap.add_argument("--alpha", type=float, default=None, help="paramètre du bruit de Dirichlet")
    ap.add_argument("--c-puct", type=float, default=1.5)
    ap.add_argument("--channels", type=int, default=64)
    ap.add_argument("--blocks", type=int, default=5)
    ap.add_argument("--buffer", type=int, default=150_000, help="taille max du replay buffer (positions)")
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--steps", type=int, default=400, help="pas de gradient par itération")
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--eval-every", type=int, default=3)
    ap.add_argument("--eval-games", type=int, default=40)
    ap.add_argument("--eval-sims", type=int, default=None)
    ap.add_argument("--eval-opponent", default=None)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--time-budget", type=float, default=None, help="arrêt après N minutes")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    family = "chess" if args.game.startswith("chess") else "connect4"
    d = DEFAULTS[family]
    sims = args.sims or d["sims"]
    eval_sims = args.eval_sims or sims
    opponent = args.eval_opponent or d["eval_opponent"]
    out = args.out or os.path.join(ROOT, "runs", args.game.replace(":", "_"))
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "config.json"), "w") as f:
        json.dump({**vars(args), "sims": sims, "eval_sims": eval_sims, "eval_opponent": opponent}, f, indent=2)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    game = make_game(args.game)
    latest = os.path.join(out, "latest.pt")
    start_iter = 0
    if args.resume and os.path.exists(latest):
        net = AZNet.load(latest, device)
        start_iter = net.meta.get("iteration", 0)
        print(f"Reprise depuis {latest} (itération {start_iter})")
    else:
        net = AZNet(game.obs_shape, game.action_size, args.channels, args.blocks).to(device)
        net.save(os.path.join(out, "iter_000.pt"), iteration=0)  # réseau aléatoire, pour comparaison
    n_params = sum(p.numel() for p in net.parameters())
    print(f"Jeu {args.game} | réseau {args.channels}x{args.blocks} ({n_params / 1e6:.2f} M paramètres) | {device}")

    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.wd)
    evaluator = NetEvaluator(net, device)
    cfg = SelfPlayConfig(sims=sims, parallel=args.parallel, temp_moves=args.temp_moves or d["temp_moves"],
                         c_puct=args.c_puct, dirichlet_alpha=args.alpha or d["alpha"])
    buffer: deque = deque(maxlen=args.buffer)
    log_path = os.path.join(out, "metrics.csv")
    new_log = not (args.resume and os.path.exists(log_path))
    best_score = -1.0
    t_start = time.perf_counter()

    for it in range(start_iter + 1, start_iter + args.iterations + 1):
        # 1) self-play
        t0 = time.perf_counter()
        net.eval()
        samples, st = generate_games(lambda: make_game(args.game), evaluator, args.games, cfg, rng)
        t_sp = time.perf_counter() - t0
        buffer.extend(samples)
        w = st["winners"]
        p1, p2, dr = w.count(0) / len(w), w.count(1) / len(w), w.count(-1) / len(w)

        # 2) entraînement
        t0 = time.perf_counter()
        # décroissance du taux d'apprentissage sur le dernier quart du run (en itérations ou en temps)
        late = it > start_iter + 0.75 * args.iterations or (
            args.time_budget and (time.perf_counter() - t_start) / 60 > 0.75 * args.time_budget)
        lr = args.lr * (0.1 if late else 1.0)
        for g in opt.param_groups:
            g["lr"] = lr
        lp, lv, ent = train_steps(net, opt, buffer, game, args.steps, args.batch_size, device)
        t_tr = time.perf_counter() - t0

        ckpt = os.path.join(out, f"iter_{it:03d}.pt")
        net.save(ckpt, iteration=it)
        net.save(latest, iteration=it)

        # 3) évaluation
        score = ""
        t_ev = 0.0
        if args.eval_every and it % args.eval_every == 0:
            t0 = time.perf_counter()
            res = evaluate(ckpt, args.game, opponent, args.eval_games, eval_sims, args.workers, seed=10_000 * it)
            t_ev = time.perf_counter() - t0
            score = round(res.score, 4)
            print(f"    éval : {res}")
            if res.score >= best_score:
                best_score = res.score
                net.save(os.path.join(out, "best.pt"), iteration=it, eval_score=res.score)

        row = {"iteration": it, "games": len(w), "new_samples": len(samples), "buffer": len(buffer),
               "mean_len": round(float(np.mean(st["lengths"])), 1), "p1_win": round(p1, 3), "p2_win": round(p2, 3),
               "draw": round(dr, 3), "loss_policy": round(lp, 4), "loss_value": round(lv, 4),
               "policy_entropy": round(ent, 3), "lr": lr, "t_selfplay": round(t_sp, 1), "t_train": round(t_tr, 1),
               "t_eval": round(t_ev, 1), "eval_score": score}
        with open(log_path, "a", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=list(row))
            if new_log:
                wr.writeheader()
                new_log = False
            wr.writerow(row)
        print(f"[it {it:3d}] {len(w)} parties ({t_sp:.0f}s, long. moy. {row['mean_len']}, "
              f"J1 {p1:.0%} / J2 {p2:.0%} / nul {dr:.0%}) | buffer {len(buffer)} | "
              f"L_pi {lp:.3f} L_v {lv:.3f} H {ent:.2f} ({t_tr:.0f}s)" + (f" | score éval {score}" if score != "" else ""),
              flush=True)

        if args.time_budget and (time.perf_counter() - t_start) / 60 > args.time_budget:
            print("Budget de temps atteint.")
            break

    if not os.path.exists(os.path.join(out, "best.pt")):
        net.save(os.path.join(out, "best.pt"), iteration=it)
    print(f"Terminé. Checkpoints dans {out}")


if __name__ == "__main__":
    main()
