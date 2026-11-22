"""AlphaZero distribué sur Modal : self-play sur des dizaines de cœurs CPU, entraînement sur GPU.

Le goulot du projet est le self-play (Python, un cœur). Ici, chaque itération :
  1. l’entraîneur (GPU L4) publie les poids courants dans un volume Modal ;
  2. ``workers`` conteneurs CPU jouent chacun ``games_per_worker`` parties de self-play
     en parallèle (``selfplay_worker.starmap``) et renvoient leurs exemples compressés ;
  3. l’entraîneur ajoute les exemples au replay buffer et fait ``steps`` pas de gradient ;
  4. régulièrement, des conteneurs CPU évaluent le réseau contre l’aléatoire et le MCTS pur.

Coupe-circuit : le coût est estimé en continu (tarifs Modal publics, majorés) et la boucle
s’arrête dès que l’estimation atteint 90 % du budget.

    modal run modal_app.py --run smoke --budget 0.4 --workers 4 --games-per-worker 2 --sims 16 --steps 20
    modal run --detach modal_app.py --run chess_standard --budget 25
    modal volume get rl-alphazero runs/chess_standard ./runs/modal/
"""
from __future__ import annotations

import io
import json
import os
import time

import modal

app = modal.App("rl-alphazero-chess")
vol = modal.Volume.from_name("rl-alphazero", create_if_missing=True)
VOL = "/vol"
CODE = ("games", "agents", "alphazero", "arena")

cpu_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("numpy==1.26.4", "chess==1.11.2")
    .pip_install("torch==2.5.1", index_url="https://download.pytorch.org/whl/cpu")
    .add_local_python_source(*CODE)
)
gpu_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("numpy==1.26.4", "chess==1.11.2", "torch==2.5.1")
    .add_local_python_source(*CODE)
)

# Tarifs publics Modal (https://modal.com/pricing), majorés par prudence.
PRICE_CPU_CORE_S = 0.0000131
PRICE_MEM_GIB_S = 0.00000667       # tarif le plus élevé affiché (estimation pessimiste)
PRICE_L4_S = 0.000222
WORKER_CPU, WORKER_MEM_GIB = 1.0, 1.0
TRAINER_CPU, TRAINER_MEM_GIB = 4.0, 8.0
OVERHEAD = 1.35                    # démarrages à froid, conteneurs inactifs avant arrêt, imprécisions
WORKER_RATE = (WORKER_CPU * PRICE_CPU_CORE_S + WORKER_MEM_GIB * PRICE_MEM_GIB_S) * OVERHEAD
TRAINER_RATE = (PRICE_L4_S + TRAINER_CPU * PRICE_CPU_CORE_S + TRAINER_MEM_GIB * PRICE_MEM_GIB_S) * OVERHEAD


def _pack(samples) -> bytes:
    import numpy as np

    planes = np.stack([s[0] for s in samples]).astype(np.float16)
    z = np.array([s[3] for s in samples], dtype=np.float32)
    offs = np.zeros(len(samples) + 1, dtype=np.int64)
    offs[1:] = np.cumsum([len(s[1]) for s in samples])
    acts = np.concatenate([s[1] for s in samples]).astype(np.int16)
    probs = np.concatenate([s[2] for s in samples]).astype(np.float32)
    buf = io.BytesIO()
    np.savez_compressed(buf, planes=planes, z=z, offs=offs, acts=acts, probs=probs)
    return buf.getvalue()


def _unpack(blob: bytes):
    import numpy as np

    d = np.load(io.BytesIO(blob))
    planes, z, offs, acts, probs = d["planes"], d["z"], d["offs"], d["acts"], d["probs"]
    return [(planes[i], acts[offs[i]:offs[i + 1]], probs[offs[i]:offs[i + 1]], z[i]) for i in range(len(z))]


@app.function(image=cpu_image, cpu=WORKER_CPU, memory=int(WORKER_MEM_GIB * 1024), volumes={VOL: vol},
              timeout=1800, scaledown_window=20, retries=1)
def selfplay_worker(weights: str, cfg: dict, n_games: int, seed: int) -> dict:
    t0 = time.time()
    import numpy as np
    import torch

    torch.set_num_threads(2)
    vol.reload()
    from alphazero.network import AZNet, NetEvaluator
    from alphazero.selfplay import SelfPlayConfig, generate_games
    from games import make_game

    ev = NetEvaluator(AZNet.load(os.path.join(VOL, weights), "cpu"), "cpu")
    sp = SelfPlayConfig(sims=cfg["sims"], parallel=n_games, temp_moves=cfg["temp_moves"], c_puct=1.5,
                        dirichlet_alpha=cfg["alpha"])
    samples, st = generate_games(lambda: make_game(cfg["game"]), ev, n_games, sp, np.random.default_rng(seed))
    return {"data": _pack(samples), "winners": st["winners"], "lengths": st["lengths"], "seconds": time.time() - t0}


@app.function(image=cpu_image, cpu=WORKER_CPU, memory=int(WORKER_MEM_GIB * 1024), volumes={VOL: vol},
              timeout=3600, scaledown_window=20, retries=1)
def eval_worker(weights: str, game: str, opponent: str, n_games: int, sims: int, seed: int) -> dict:
    t0 = time.time()
    import torch

    torch.set_num_threads(2)
    vol.reload()
    from arena import match

    res = match(game, f"az:ckpt={os.path.join(VOL, weights)},sims={sims}", opponent, n_games=n_games, workers=1,
                seed=seed)
    return {"wins": res.wins, "draws": res.draws, "losses": res.losses, "seconds": time.time() - t0}


@app.function(image=gpu_image, gpu="L4", cpu=TRAINER_CPU, memory=int(TRAINER_MEM_GIB * 1024), volumes={VOL: vol},
              timeout=8 * 3600)
def trainer(run: str, game: str, budget: float, workers: int, games_per_worker: int, sims: int, steps: int,
            batch: int, lr: float, buffer_size: int, eval_every: int, eval_games: int, channels: int, blocks: int,
            init_from: str = ""):
    import csv

    import numpy as np
    import torch

    from alphazero.network import AZNet
    from alphazero.train import train_steps
    from games import make_game

    t_start = time.time()
    spent_workers = 0.0
    cost = lambda: (time.time() - t_start) * TRAINER_RATE + spent_workers  # noqa: E731
    stop_at = 0.9 * budget

    out = os.path.join(VOL, "runs", run)
    os.makedirs(out, exist_ok=True)
    rel = lambda p: os.path.relpath(p, VOL)  # noqa: E731
    g = make_game(game)
    torch.manual_seed(0)
    np.random.seed(0)
    net = AZNet(g.obs_shape, g.action_size, channels, blocks).cuda()
    # Reprise automatique : si ce run a déjà un checkpoint (conteneur préempté puis relancé par Modal),
    # on repart de ses poids, de son itération et du coût déjà engagé.
    vol.reload()
    log_path = os.path.join(out, "metrics.csv")
    prev = []
    if os.path.exists(log_path):
        import csv as _csv

        prev = list(_csv.DictReader(open(log_path)))
    resumed = bool(prev) and os.path.exists(os.path.join(out, "latest.pt"))
    if resumed:
        net.load_state_dict(torch.load(os.path.join(out, "latest.pt"), map_location="cuda", weights_only=False)["state_dict"])
        spent_workers = float(prev[-1]["cost_estimate"])  # coût déjà engagé, compté une seule fois
        print(f"[{run}] reprise automatique à l’itération {prev[-1]['iteration']} (coût déjà engagé {spent_workers:.2f} $)",
              flush=True)
    elif init_from:  # reprise depuis un run précédent (mêmes architecture et encodage)
        vol.reload()
        net.load_state_dict(torch.load(os.path.join(VOL, init_from), map_location="cuda", weights_only=False)["state_dict"])
        print(f"[{run}] poids initiaux repris de {init_from}", flush=True)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    cfg = {"game": game, "sims": sims, "temp_moves": 30, "alpha": 0.3}
    json.dump({**cfg, "budget": budget, "workers": workers, "games_per_worker": games_per_worker, "steps": steps,
               "batch": batch, "lr": lr, "buffer": buffer_size, "channels": channels, "blocks": blocks},
              open(os.path.join(out, "config.json"), "w"), indent=2)
    if not resumed:
        net.save(os.path.join(out, "iter_000.pt"), iteration=0)  # point de départ du run (repris ou aléatoire)
        net.save(os.path.join(out, "latest.pt"), iteration=0)
        vol.commit()

    buf: list = []
    total_games = int(prev[-1]["total_games"]) if resumed else 0
    it = int(prev[-1]["iteration"]) if resumed else 0
    print(f"[{run}] {game} · réseau {channels}x{blocks} · {workers} workers × {games_per_worker} parties · "
          f"{sims} sims · budget {budget} $ (arrêt à {stop_at:.2f} $)", flush=True)
    while cost() < stop_at:
        it += 1
        t0 = time.time()
        # 1) self-play distribué avec les poids courants
        args = [(rel(os.path.join(out, "latest.pt")), cfg, games_per_worker, it * 100_000 + k) for k in range(workers)]
        winners, lengths, new = [], [], 0
        for res in selfplay_worker.starmap(args, return_exceptions=True):
            if isinstance(res, Exception):
                print("   worker en échec :", repr(res)[:200], flush=True)
                continue
            spent_workers += res["seconds"] * WORKER_RATE
            s = _unpack(res["data"])
            buf.extend(s)
            new += len(s)
            winners += res["winners"]
            lengths += res["lengths"]
        if len(buf) > buffer_size:
            del buf[: len(buf) - buffer_size]
        total_games += len(winners)
        t_sp = time.time() - t0

        # 2) entraînement sur GPU (taux d’apprentissage ÷10 sur le dernier quart du budget)
        t1 = time.time()
        cur_lr = lr * (0.1 if cost() > 0.75 * stop_at else 1.0)
        for grp in opt.param_groups:
            grp["lr"] = cur_lr
        lp, lv, ent = train_steps(net, opt, buf, g, steps, batch, "cuda") if buf else (0, 0, 0)
        t_tr = time.time() - t1
        net.save(os.path.join(out, "latest.pt"), iteration=it, games=total_games)
        if it % 5 == 0:
            net.save(os.path.join(out, f"iter_{it:03d}.pt"), iteration=it, games=total_games)
        vol.commit()

        # 3) évaluation distribuée
        ev = {}
        if eval_every and it % eval_every == 0:
            ck = f"iter_{it:03d}.pt"
            if not os.path.exists(os.path.join(out, ck)):
                net.save(os.path.join(out, ck), iteration=it, games=total_games)
                vol.commit()
            opps = {"random": "random", "mcts": f"mcts:sims={sims},max_depth=20,rollout=captures"}
            per = 4
            jobs = [(rel(os.path.join(out, ck)), game, spec, per, sims, it * 1000 + k)
                    for name, spec in opps.items() for k in range(max(1, eval_games // per))]
            agg = {name: [0, 0, 0] for name in opps}
            results = list(eval_worker.starmap(jobs, return_exceptions=True))  # consommer entièrement le flux
            for (w, _, spec, *_), r in zip(jobs, results):
                if isinstance(r, Exception):
                    continue
                spent_workers += r["seconds"] * WORKER_RATE
                name = "random" if spec == "random" else "mcts"
                agg[name][0] += r["wins"]; agg[name][1] += r["draws"]; agg[name][2] += r["losses"]
            for name, (wn, dr, ls) in agg.items():
                n = max(wn + dr + ls, 1)
                ev[f"score_vs_{name}"] = round((wn + 0.5 * dr) / n, 3)
                ev[f"n_vs_{name}"] = n

        wl = winners or [-1]
        row = {"iteration": it, "games": len(winners), "total_games": total_games, "new_samples": new,
               "buffer": len(buf), "mean_len": round(float(np.mean(lengths or [0])), 1),
               "white_win": round(wl.count(0) / len(wl), 3), "black_win": round(wl.count(1) / len(wl), 3),
               "draw": round(wl.count(-1) / len(wl), 3), "loss_policy": round(lp, 4), "loss_value": round(lv, 4),
               "entropy": round(ent, 3), "lr": cur_lr, "t_selfplay": round(t_sp, 1), "t_train": round(t_tr, 1),
               "cost_estimate": round(cost(), 3), **ev}
        new_file = not os.path.exists(log_path)
        with open(log_path, "a", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=["iteration", "games", "total_games", "new_samples", "buffer", "mean_len",
                                               "white_win", "black_win", "draw", "loss_policy", "loss_value", "entropy",
                                               "lr", "t_selfplay", "t_train", "cost_estimate", "score_vs_random",
                                               "n_vs_random", "score_vs_mcts", "n_vs_mcts"])
            if new_file:
                wr.writeheader()
            wr.writerow(row)
        vol.commit()
        print(f"[it {it:3d}] {len(winners)} parties ({t_sp:.0f}s, long. {row['mean_len']}, "
              f"B {row['white_win']:.0%}/N {row['black_win']:.0%}/nul {row['draw']:.0%}) | total {total_games} | "
              f"L_pi {lp:.3f} L_v {lv:.3f} ({t_tr:.0f}s) | coût estimé {cost():.2f} $"
              + (f" | vs aléatoire {ev['score_vs_random']:.0%}, vs MCTS {ev['score_vs_mcts']:.0%}" if ev else ""),
              flush=True)

    net.save(os.path.join(out, "final.pt"), iteration=it, games=total_games)
    vol.commit()
    print(f"[{run}] Arrêt : coût estimé {cost():.2f} $ ≥ {stop_at:.2f} $. {it} itérations, {total_games} parties.",
          flush=True)
    return {"iterations": it, "games": total_games, "cost_estimate": cost()}


@app.local_entrypoint()
def main(run: str = "chess_standard", game: str = "chess:standard", budget: float = 25.0, workers: int = 64,
         games_per_worker: int = 12, sims: int = 64, steps: int = 400, batch: int = 1024, lr: float = 2e-3,
         buffer_size: int = 300_000, eval_every: int = 5, eval_games: int = 32, channels: int = 64, blocks: int = 5,
         init_from: str = ""):
    if budget > 25.0:
        raise SystemExit("Budget refusé : plafond convenu de 25 $.")
    print(trainer.remote(run, game, budget, workers, games_per_worker, sims, steps, batch, lr, buffer_size,
                         eval_every, eval_games, channels, blocks, init_from))


def spawn_detached(**kw):
    """Lance l’entraînement sur l’app *déployée* (``modal deploy modal_app.py``) : l’appel vit côté serveur,
    indépendamment de l’ordinateur local, et Modal le relance en cas de préemption (reprise automatique)."""
    f = modal.Function.from_name("rl-alphazero-chess", "trainer")
    defaults = dict(game="chess:standard", workers=64, games_per_worker=12, sims=64, steps=400, batch=1024, lr=2e-3,
                    buffer_size=300_000, eval_every=5, eval_games=32, channels=64, blocks=5, init_from="")
    defaults.update(kw)
    if defaults["budget"] > 25.0:
        raise SystemExit("Budget refusé : plafond convenu de 25 $.")
    call = f.spawn(**defaults)
    print("appel lancé :", call.object_id)
    return call.object_id
