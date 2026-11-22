"""MCTS guidé par réseau (PUCT, AlphaZero).

Différences avec le MCTS pur (``agents/mcts.py``) :
- pas de rollout : la feuille est évaluée par la tête *value* du réseau ;
- la sélection utilise PUCT : Q(s,a) + c_puct * P(s,a) * sqrt(N(s)) / (1 + N(s,a)),
  où P est la *policy* du réseau (prior) ;
- tous les enfants sont créés d'un coup à l'expansion (avec leur prior) ;
- bruit de Dirichlet à la racine pendant le self-play pour garantir l'exploration.

L'arbre expose ``select_leaf`` / ``expand_and_backup`` séparément pour que le
self-play puisse évaluer en un seul lot GPU les feuilles de dizaines d'arbres.
"""
from __future__ import annotations

import math

import numpy as np

from games.base import Game


class AZNode:
    __slots__ = ("state", "move", "action", "prior", "N", "W", "children", "expanded")

    def __init__(self, state: Game | None, move=None, action: int = -1, prior: float = 1.0):
        self.state = state          # créé paresseusement à la première visite
        self.move = move
        self.action = action
        self.prior = prior
        self.N = 0
        self.W = 0.0                # du point de vue du joueur qui a joué ``move``
        self.children: list[AZNode] = []
        self.expanded = False

    @property
    def Q(self) -> float:
        return self.W / self.N if self.N else 0.0


def softmax_legal(logits: np.ndarray, actions: list[int]) -> np.ndarray:
    z = logits[actions].astype(np.float64)
    z -= z.max()
    p = np.exp(z)
    return p / p.sum()


class PUCTTree:
    def __init__(self, game: Game, c_puct: float = 1.5, dirichlet_alpha: float = 0.3, noise_eps: float = 0.25,
                 add_noise: bool = False, fpu: float = 0.0, rng: np.random.Generator | None = None):
        self.root = AZNode(game.clone())
        self.c_puct = c_puct
        self.alpha = dirichlet_alpha
        self.eps = noise_eps
        self.add_noise = add_noise
        self.fpu = fpu                     # valeur Q supposée d'un enfant jamais visité
        self.rng = rng or np.random.default_rng()

    # ------------------------------------------------------------ sélection
    def select_leaf(self) -> list[AZNode]:
        node, path = self.root, [self.root]
        while node.expanded and not node.state.is_terminal():
            node = self._best_child(node)
            if node.state is None:
                st = path[-1].state.clone()
                st.play(node.move)
                node.state = st
            path.append(node)
        return path

    def _best_child(self, node: AZNode) -> AZNode:
        c_sqrt = self.c_puct * math.sqrt(max(node.N, 1))
        fpu = self.fpu
        best, best_s = None, -math.inf
        for ch in node.children:
            q = ch.W / ch.N if ch.N else fpu
            s = q + c_sqrt * ch.prior / (1 + ch.N)
            if s > best_s:
                best, best_s = ch, s
        return best

    # ---------------------------------------------- expansion / backup
    def expand_and_backup(self, path: list[AZNode], logits: np.ndarray | None, value: float | None) -> None:
        """``value`` : évaluation du réseau pour le joueur au trait à la feuille.

        Si la feuille est terminale, on ignore le réseau et on utilise le vrai résultat.
        """
        leaf = path[-1]
        st = leaf.state
        if st.is_terminal():
            value = st.result(st.current_player)
        elif not leaf.expanded:
            moves = st.legal_moves()
            actions = [st.move_to_action(m) for m in moves]
            priors = softmax_legal(logits, actions)
            if leaf is self.root and self.add_noise:
                priors = self._noisy(priors)
            leaf.children = [AZNode(None, m, a, float(p)) for m, a, p in zip(moves, actions, priors)]
            leaf.expanded = True
        self.backup(path, value)

    @staticmethod
    def backup(path: list[AZNode], value: float) -> None:
        v = value  # point de vue du joueur au trait à la feuille
        for node in reversed(path):
            node.N += 1
            node.W -= v  # le nœud stocke le point de vue de celui qui y a mené
            v = -v

    def _noisy(self, priors: np.ndarray) -> np.ndarray:
        noise = self.rng.dirichlet([self.alpha] * len(priors))
        return (1 - self.eps) * priors + self.eps * noise

    def needs_eval(self, path: list[AZNode]) -> bool:
        leaf = path[-1]
        return not leaf.state.is_terminal() and not leaf.expanded

    # ---------------------------------------------------- choix du coup
    def visit_policy(self, action_size: int) -> tuple[np.ndarray, np.ndarray]:
        """(actions, probabilités N / ΣN) des enfants de la racine."""
        acts = np.array([ch.action for ch in self.root.children], dtype=np.int32)
        n = np.array([ch.N for ch in self.root.children], dtype=np.float64)
        return acts, n / n.sum()

    def choose(self, temperature: float) -> AZNode:
        kids = self.root.children
        n = np.array([ch.N for ch in kids], dtype=np.float64)
        if temperature <= 1e-3:
            return kids[int(self.rng.choice(np.flatnonzero(n == n.max())))]
        p = n ** (1.0 / temperature)
        return kids[int(self.rng.choice(len(kids), p=p / p.sum()))]

    def advance(self, child: AZNode) -> None:
        """Réutilise le sous-arbre du coup joué comme nouvelle racine."""
        if child.state is None:
            st = self.root.state.clone()
            st.play(child.move)
            child.state = st
        self.root = child
        if self.add_noise and child.expanded and child.children:
            pri = self._noisy(np.array([c.prior for c in child.children]))
            for c, p in zip(child.children, pri):
                c.prior = float(p)

    def advance_by_move(self, game: Game, move) -> None:
        for ch in self.root.children:
            if ch.move == move:
                self.advance(ch)
                return
        self.root = AZNode(game.clone())  # coup non exploré : nouvel arbre


def run_search(tree: PUCTTree, evaluator, n_sims: int) -> None:
    """Recherche séquentielle (une feuille évaluée à la fois) — utilisée par l'agent de jeu."""
    for _ in range(n_sims):
        path = tree.select_leaf()
        if tree.needs_eval(path):
            logits, values = evaluator([path[-1].state])
            tree.expand_and_backup(path, logits[0], float(values[0]))
        else:
            tree.expand_and_backup(path, None, 0.0)
