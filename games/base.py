"""Interface commune des jeux à deux joueurs, à information complète et à somme nulle.

Conventions :
- les joueurs sont numérotés 0 (premier joueur) et 1 ;
- toutes les valeurs (résultat, heuristique) sont exprimées du point de vue
  du joueur indiqué, dans [-1, 1] : +1 victoire, -1 défaite, 0 nul ;
- un « coup » (move) est l'objet natif du jeu (colonne pour Puissance 4,
  ``chess.Move`` pour les échecs) ; une « action » est son index entier dans
  l'espace d'actions fixe utilisé par le réseau de neurones.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Hashable

import numpy as np


class Game(ABC):
    #: taille de l'espace d'actions (sortie policy du réseau)
    action_size: int
    #: forme de l'encodage (C, H, W) donné au réseau
    obs_shape: tuple[int, int, int]
    name: str

    # ------------------------------------------------------------------ règles
    @abstractmethod
    def clone(self) -> "Game": ...

    @property
    @abstractmethod
    def current_player(self) -> int: ...

    @abstractmethod
    def legal_moves(self) -> list[Any]: ...

    @abstractmethod
    def play(self, move: Any) -> None: ...

    @abstractmethod
    def is_terminal(self) -> bool: ...

    @abstractmethod
    def winner(self) -> int | None:
        """0 ou 1 si la partie est gagnée, -1 si nulle, None si non terminée."""

    def result(self, player: int) -> float:
        """Résultat final du point de vue de ``player`` (+1 / 0 / -1)."""
        w = self.winner()
        if w is None or w == -1:
            return 0.0
        return 1.0 if w == player else -1.0

    # ------------------------------------------------- aides pour MCTS « pur »
    def rollout_move(self, rng: np.random.Generator, policy: str = "random") -> Any:
        """Coup utilisé pendant la simulation (rollout). Par défaut : aléatoire."""
        moves = self.legal_moves()
        return moves[rng.integers(len(moves))]

    def heuristic_value(self) -> float:
        """Évaluation statique dans [-1, 1] du point de vue du joueur au trait.

        Utilisée quand un rollout est tronqué (jeux longs comme les échecs).
        """
        return 0.0

    # --------------------------------------------- aides pour réseau / AlphaZero
    @abstractmethod
    def encode(self) -> np.ndarray:
        """Tenseur float32 de forme ``obs_shape``, vu du joueur au trait."""

    @abstractmethod
    def move_to_action(self, move: Any) -> int: ...

    def legal_actions(self) -> list[int]:
        return [self.move_to_action(m) for m in self.legal_moves()]

    def symmetries(self, planes: np.ndarray, pi: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
        """Augmentation de données par symétries (identité par défaut)."""
        return [(planes, pi)]

    # ------------------------------------------------------------- interface IO
    @abstractmethod
    def parse_move(self, text: str) -> Any:
        """Convertit une saisie humaine en coup (lève ValueError si invalide)."""

    def move_str(self, move: Any) -> str:
        return str(move)

    def key(self) -> Hashable:
        """Identifiant hashable de la position (debug / tests)."""
        return str(self)
