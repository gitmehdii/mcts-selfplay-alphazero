"""Génération de parties en self-play, avec évaluation par lots.

``parallel`` parties sont jouées simultanément : à chaque simulation, on prend
une feuille dans chacun des arbres et on les évalue toutes en un seul appel au
réseau (GPU). C'est ce qui rend le self-play praticable en Python.

Chaque position jouée donne un exemple d'entraînement (s, π, z) :
  s : encodage de la position,
  π : distribution des visites MCTS à la racine (cible de la policy),
  z : résultat final de la partie pour le joueur au trait en s (cible de la value).
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from .mcts import PUCTTree


@dataclass
class SelfPlayConfig:
    sims: int = 100
    parallel: int = 64
    temp_moves: int = 10        # température 1 pendant les premiers demi-coups, puis 0 (coup le plus visité)
    c_puct: float = 1.5
    dirichlet_alpha: float = 0.3
    noise_eps: float = 0.25


def generate_games(game_factory, evaluator, n_games: int, cfg: SelfPlayConfig, rng: np.random.Generator,
                   log_every: int = 0):
    samples = []
    stats = {"winners": [], "lengths": []}
    slots: list[dict] = []
    started = 0
    t0 = time.perf_counter()

    def new_slot():
        tree = PUCTTree(game_factory(), c_puct=cfg.c_puct, dirichlet_alpha=cfg.dirichlet_alpha,
                        noise_eps=cfg.noise_eps, add_noise=True, rng=rng)
        return {"tree": tree, "hist": []}

    while slots or started < n_games:
        while len(slots) < cfg.parallel and started < n_games:
            slots.append(new_slot())
            started += 1

        # --- cfg.sims simulations sur tous les arbres, feuilles évaluées en lot
        for _ in range(cfg.sims):
            paths = [s["tree"].select_leaf() for s in slots]
            need = [i for i, (s, p) in enumerate(zip(slots, paths)) if s["tree"].needs_eval(p)]
            if need:
                logits, values = evaluator([paths[i][-1].state for i in need])
                for k, i in enumerate(need):
                    slots[i]["tree"].expand_and_backup(paths[i], logits[k], float(values[k]))
            need_set = set(need)
            for i, (s, p) in enumerate(zip(slots, paths)):
                if i not in need_set:
                    s["tree"].expand_and_backup(p, None, 0.0)  # feuille terminale

        # --- chaque partie joue un coup
        still = []
        for s in slots:
            tree = s["tree"]
            game = tree.root.state
            acts, probs = tree.visit_policy(game.action_size)
            s["hist"].append((game.encode().astype(np.float16), acts.astype(np.int16), probs.astype(np.float32),
                              game.current_player))
            temp = 1.0 if len(s["hist"]) <= cfg.temp_moves else 0.0
            tree.advance(tree.choose(temp))
            game = tree.root.state
            if game.is_terminal():
                for planes, a, p, player in s["hist"]:
                    samples.append((planes, a, p, np.float32(game.result(player))))
                stats["winners"].append(game.winner())
                stats["lengths"].append(len(s["hist"]))
                if log_every and len(stats["lengths"]) % log_every == 0:
                    print(f"    self-play {len(stats['lengths'])}/{n_games} parties "
                          f"({time.perf_counter() - t0:.0f}s)", flush=True)
            else:
                still.append(s)
        slots = still
    return samples, stats
