"""MCTS « pur » (UCT, Kocsis & Szepesvári 2006) avec rollouts.

Les 4 phases d'une simulation :
1. **Sélection**  : on descend l'arbre en maximisant UCB1
       Q(s,a) + c * sqrt(ln N(s) / N(s,a))
   tant que le nœud est entièrement développé ;
2. **Expansion**  : on ajoute un enfant pour un coup non encore essayé ;
3. **Simulation** : rollout jusqu'à la fin de partie (ou jusqu'à ``max_depth``,
   auquel cas on utilise l'heuristique statique du jeu) ;
4. **Rétropropagation** : on remonte le résultat ; chaque nœud stocke la
   récompense du point de vue du joueur qui a joué le coup menant à ce nœud,
   ce qui permet au parent de simplement maximiser Q (formulation négamax).
"""
from __future__ import annotations

import math
import time

import numpy as np

from games.base import Game
from .base import Agent


class Node:
    __slots__ = ("move", "parent", "children", "untried", "N", "W", "player")

    def __init__(self, move, parent, untried, player):
        self.move = move
        self.parent: Node | None = parent
        self.children: list[Node] = []
        self.untried = untried          # coups pas encore développés
        self.N = 0                      # nombre de visites
        self.W = 0.0                    # somme des récompenses (point de vue de ``player``)
        self.player = player            # joueur qui a joué ``move`` pour arriver ici


class MCTSAgent(Agent):
    def __init__(self, sims: int = 1000, c: float = math.sqrt(2), rollout: str = "random",
                 max_depth: int | None = None, time_limit: float | None = None, seed: int | None = None,
                 verbose: bool = False):
        self.sims = int(sims)
        self.c = float(c)
        self.rollout_policy = rollout
        self.max_depth = max_depth          # None = rollout jusqu'à la fin
        self.time_limit = time_limit        # secondes par coup (prioritaire sur sims)
        self.rng = np.random.default_rng(seed)
        self.verbose = verbose
        self.last_root: Node | None = None
        budget = f"{time_limit:g}s" if time_limit else f"sims={self.sims}"
        self.name = f"MCTS({budget},c={self.c:g},{rollout})"

    # ------------------------------------------------------------------ API
    def select_move(self, game: Game):
        root = self.search(game)
        best = max(root.children, key=lambda ch: ch.N)  # « robust child »
        if self.verbose:
            self._print_stats(game, root)
        return best.move

    def search(self, game: Game) -> Node:
        root = Node(None, None, list(game.legal_moves()), 1 - game.current_player)
        deadline = time.perf_counter() + self.time_limit if self.time_limit else None
        n = 0
        while True:
            self._simulate(root, game.clone())
            n += 1
            if deadline is not None:
                if time.perf_counter() >= deadline:
                    break
            elif n >= self.sims:
                break
        self.last_root = root
        return root

    # ---------------------------------------------------------- simulation
    def _simulate(self, root: Node, state: Game) -> None:
        node = root
        c, log = self.c, math.log

        # 1) sélection (UCB1)
        while not node.untried and node.children:
            log_n = log(node.N)
            best, best_score = None, -math.inf
            for ch in node.children:
                score = ch.W / ch.N + c * math.sqrt(log_n / ch.N)
                if score > best_score:
                    best, best_score = ch, score
            node = best
            state.play(node.move)

        # 2) expansion
        if node.untried:
            i = self.rng.integers(len(node.untried))
            move = node.untried[i]
            node.untried[i] = node.untried[-1]
            node.untried.pop()
            player = state.current_player
            state.play(move)
            child = Node(move, node, list(state.legal_moves()), player)
            node.children.append(child)
            node = child

        # 3) simulation (rollout)
        depth = 0
        rng, policy, max_depth = self.rng, self.rollout_policy, self.max_depth
        while not state.is_terminal():
            if max_depth is not None and depth >= max_depth:
                break
            state.play(state.rollout_move(rng, policy))
            depth += 1
        if state.is_terminal():
            r0 = state.result(0)
        else:  # rollout tronqué : évaluation heuristique
            h = state.heuristic_value()
            r0 = h if state.current_player == 0 else -h

        # 4) rétropropagation
        while node is not None:
            node.N += 1
            node.W += r0 if node.player == 0 else -r0
            node = node.parent

    def _print_stats(self, game: Game, root: Node) -> None:
        kids = sorted(root.children, key=lambda ch: -ch.N)
        txt = ", ".join(f"{game.move_str(ch.move)}: N={ch.N} Q={ch.W / ch.N:+.2f}" for ch in kids[:8])
        print(f"  [{self.name}] {root.N} simulations -> {txt}")
