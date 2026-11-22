from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from games.base import Game


class Agent(ABC):
    name: str = "agent"

    @abstractmethod
    def select_move(self, game: Game): ...

    def reset(self) -> None:
        """Appelé au début de chaque partie."""

    def __repr__(self) -> str:
        return self.name


class RandomAgent(Agent):
    def __init__(self, seed: int | None = None):
        self.rng = np.random.default_rng(seed)
        self.name = "Random"

    def select_move(self, game: Game):
        moves = game.legal_moves()
        return moves[self.rng.integers(len(moves))]


class HeuristicAgent(Agent):
    """Politique de rollout « heuristique » du jeu, jouée directement.

    Puissance 4 : gagne si possible, sinon bloque, sinon aléatoire.
    Échecs : capture au hasard si possible, sinon coup aléatoire.
    """

    def __init__(self, seed: int | None = None):
        self.rng = np.random.default_rng(seed)
        self.name = "Heuristic"

    def select_move(self, game: Game):
        return game.rollout_move(self.rng, "heuristic")


class HumanAgent(Agent):
    def __init__(self, **_):
        self.name = "Human"

    def select_move(self, game: Game):
        while True:
            txt = input(f"Votre coup ({'colonne 1-7' if game.name == 'connect4' else 'UCI ou SAN, ex. e2e4 / Nf3'}) : ")
            if txt.strip().lower() in ("q", "quit", "exit"):
                raise KeyboardInterrupt
            try:
                return game.parse_move(txt)
            except (ValueError, IndexError) as e:
                print(f"  coup invalide ({e}), réessayez.")
