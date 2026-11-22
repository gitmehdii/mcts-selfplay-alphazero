"""Agent de jeu AlphaZero : réseau entraîné + recherche PUCT (sans bruit, température 0)."""
from __future__ import annotations

import numpy as np
import torch

from agents.base import Agent
from games.base import Game
from .mcts import PUCTTree, run_search, softmax_legal
from .network import AZNet, NetEvaluator

_CACHE: dict[tuple[str, str], NetEvaluator] = {}


def load_evaluator(ckpt: str, device: str = "cpu") -> NetEvaluator:
    key = (ckpt, device)
    if key not in _CACHE:  # un seul chargement par processus
        _CACHE[key] = NetEvaluator(AZNet.load(ckpt, device), device)
    return _CACHE[key]


class AlphaZeroAgent(Agent):
    def __init__(self, ckpt: str, sims: int = 100, c_puct: float = 1.5, device: str = "cpu",
                 temperature: float = 0.0, seed: int | None = None, verbose: bool = False):
        if device == "cpu":
            torch.set_num_threads(1)
        self.evaluator = load_evaluator(ckpt, device)
        self.sims = int(sims)
        self.c_puct = c_puct
        self.temperature = temperature
        self.rng = np.random.default_rng(seed)
        self.verbose = verbose
        tag = ckpt.rsplit("/", 1)[-1].removesuffix(".pt")
        self.name = f"AZ({tag},sims={self.sims})" if self.sims else f"Net({tag})"

    def select_move(self, game: Game):
        if self.sims == 0:  # réseau seul : coup le plus probable selon la policy
            logits, v = self.evaluator([game])
            moves = game.legal_moves()
            p = softmax_legal(logits[0], [game.move_to_action(m) for m in moves])
            if self.verbose:
                print(f"  [{self.name}] v={v[0]:+.2f} p_max={p.max():.2f}")
            return moves[int(np.argmax(p))]
        tree = PUCTTree(game, c_puct=self.c_puct, rng=self.rng)
        run_search(tree, self.evaluator, self.sims)
        child = tree.choose(self.temperature)
        if self.verbose:
            kids = sorted(tree.root.children, key=lambda c: -c.N)[:6]
            print(f"  [{self.name}] " + ", ".join(
                f"{game.move_str(c.move)}: N={c.N} P={c.prior:.2f} Q={c.Q:+.2f}" for c in kids))
        return child.move
