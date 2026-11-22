"""Puissance 4 (7 colonnes x 6 lignes) sur bitboards.

Représentation classique (cf. John Tromp / « Fhourstones ») : chaque colonne
occupe 7 bits (6 cases + 1 bit sentinelle), la case (col, ligne) correspond au
bit ``col * 7 + ligne`` (ligne 0 = bas). Un alignement de 4 se teste avec
quatre décalages/ET, ce qui rend les rollouts MCTS très rapides en Python.
"""
from __future__ import annotations

import numpy as np

from .base import Game

W, H = 7, 6
H1 = H + 1
_BOTTOM = [c * H1 for c in range(W)]
_TOP = [c * H1 + H - 1 for c in range(W)]  # bit de la case la plus haute


def _has_four(b: int) -> bool:
    for d in (1, H1, H, H + 2):  # vertical, horizontal, diagonale \, diagonale /
        m = b & (b >> d)
        if m & (m >> (2 * d)):
            return True
    return False


class Connect4(Game):
    action_size = W
    obs_shape = (3, H, W)
    name = "connect4"

    __slots__ = ("bb", "heights", "moves", "_winner")

    def __init__(self):
        self.bb = [0, 0]                 # pions du joueur 0 / du joueur 1
        self.heights = list(_BOTTOM)     # prochain bit libre de chaque colonne
        self.moves = 0
        self._winner: int | None = None

    def clone(self) -> "Connect4":
        g = Connect4.__new__(Connect4)
        g.bb = self.bb[:]
        g.heights = self.heights[:]
        g.moves = self.moves
        g._winner = self._winner
        return g

    # ------------------------------------------------------------------ règles
    @property
    def current_player(self) -> int:
        return self.moves & 1

    def legal_moves(self) -> list[int]:
        if self._winner is not None:
            return []
        return [c for c in range(W) if self.heights[c] <= _TOP[c]]

    def play(self, col: int) -> None:
        p = self.moves & 1
        b = self.bb[p] | (1 << self.heights[col])
        self.bb[p] = b
        self.heights[col] += 1
        self.moves += 1
        if _has_four(b):
            self._winner = p
        elif self.moves == W * H:
            self._winner = -1

    def is_terminal(self) -> bool:
        return self._winner is not None

    def winner(self) -> int | None:
        return self._winner

    def is_winning_move(self, col: int, player: int | None = None) -> bool:
        p = self.current_player if player is None else player
        return _has_four(self.bb[p] | (1 << self.heights[col]))

    # ----------------------------------------------------------- aides MCTS
    def rollout_move(self, rng: np.random.Generator, policy: str = "random") -> int:
        moves = self.legal_moves()
        if policy == "heuristic":
            # 1) gagner immédiatement si possible, 2) sinon bloquer l'adversaire
            me = self.current_player
            for c in moves:
                if self.is_winning_move(c, me):
                    return c
            for c in moves:
                if self.is_winning_move(c, 1 - me):
                    return c
        return moves[rng.integers(len(moves))]

    # ------------------------------------------------------- encodage réseau
    @staticmethod
    def _bits(b: int) -> np.ndarray:
        """Bitboard -> grille (H, W), ligne 0 = haut de l'écran."""
        bits = np.unpackbits(np.array([b], dtype=np.uint64).view(np.uint8), bitorder="little")
        grid = bits[: W * H1].reshape(W, H1)[:, :H].T  # (ligne depuis le bas, colonne)
        return grid[::-1]

    def encode(self) -> np.ndarray:
        p = self.current_player
        out = np.empty(self.obs_shape, dtype=np.float32)
        out[0] = self._bits(self.bb[p])
        out[1] = self._bits(self.bb[1 - p])
        out[2] = 1.0
        return out

    def move_to_action(self, move: int) -> int:
        return move

    def legal_actions(self) -> list[int]:
        return self.legal_moves()

    def symmetries(self, planes, pi):
        # symétrie gauche-droite du plateau (et donc des colonnes)
        return [(planes, pi), (planes[:, :, ::-1].copy(), pi[::-1].copy())]

    # ------------------------------------------------------------------ IO
    def parse_move(self, text: str) -> int:
        col = int(text.strip()) - 1  # l'humain saisit 1..7
        if col not in self.legal_moves():
            raise ValueError("colonne invalide ou pleine")
        return col

    def move_str(self, move: int) -> str:
        return str(move + 1)

    def key(self):
        return (self.bb[0], self.bb[1])

    def __str__(self) -> str:
        g0, g1 = self._bits(self.bb[0]), self._bits(self.bb[1])
        rows = []
        for r in range(H):
            rows.append("|" + "|".join("X" if g0[r, c] else "O" if g1[r, c] else " " for c in range(W)) + "|")
        rows.append("+" + "-+" * W)
        rows.append(" " + " ".join(str(c + 1) for c in range(W)))
        return "\n".join(rows)
