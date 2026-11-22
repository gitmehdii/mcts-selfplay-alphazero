"""Faire jouer deux agents l'un contre l'autre et mesurer les performances.

Les matchs alternent le premier joueur (la moitié des parties chacun) pour
neutraliser l'avantage du trait, et sont parallélisés sur plusieurs processus.
"""
from __future__ import annotations

import math
import os
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
import multiprocessing as mp

from agents import Agent, make_agent
from games import Game, make_game


def play_game(game: Game, agents: list[Agent], verbose: bool = False) -> int:
    """Joue une partie ; ``agents[i]`` joue le joueur i. Renvoie le gagnant (0, 1) ou -1 (nul)."""
    for a in agents:
        a.reset()
    if verbose:
        print(game, "\n")
    while not game.is_terminal():
        p = game.current_player
        t0 = time.perf_counter()
        move = agents[p].select_move(game)
        if verbose:
            print(f"{agents[p].name} (joueur {p + 1}) joue {game.move_str(move)}  [{time.perf_counter() - t0:.2f}s]")
        game.play(move)
        if verbose:
            print(game, "\n")
    return game.winner()


@dataclass
class MatchResult:
    name_a: str
    name_b: str
    wins: int = 0
    draws: int = 0
    losses: int = 0
    wins_as_first: int = 0
    games_as_first: int = 0
    first_mover_wins: int = 0      # parties gagnées par le joueur qui commence (quel qu'il soit)
    seconds: float = 0.0
    lengths: list = field(default_factory=list)

    @property
    def n(self) -> int:
        return self.wins + self.draws + self.losses

    @property
    def score(self) -> float:
        """Score de A : victoire = 1, nul = 0.5."""
        return (self.wins + 0.5 * self.draws) / max(self.n, 1)

    @property
    def win_rate(self) -> float:
        return self.wins / max(self.n, 1)

    def ci95(self) -> tuple[float, float]:
        """Intervalle de Wilson à 95 % sur le score."""
        n, p, z = self.n, self.score, 1.96
        if n == 0:
            return 0.0, 1.0
        den = 1 + z * z / n
        centre = (p + z * z / (2 * n)) / den
        half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
        return max(0.0, centre - half), min(1.0, centre + half)

    def elo_diff(self) -> float:
        s = min(max(self.score, 1e-3), 1 - 1e-3)
        return -400 * math.log10(1 / s - 1)

    def __str__(self) -> str:
        lo, hi = self.ci95()
        return (f"{self.name_a} vs {self.name_b} : {self.wins}V / {self.draws}N / {self.losses}D "
                f"sur {self.n} parties | score {self.score:.1%} [IC95 {lo:.1%}-{hi:.1%}] "
                f"| Elo {self.elo_diff():+.0f} | {self.seconds:.1f}s")


def _init_worker():
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    try:
        import torch

        torch.set_num_threads(1)
    except ImportError:
        pass


def _run_one(args) -> tuple[int, bool, int]:
    game_name, game_kwargs, spec_a, spec_b, a_first, seed = args
    game = make_game(game_name, **game_kwargs)
    a = make_agent(spec_a, seed=seed)
    b = make_agent(spec_b, seed=seed + 1_000_003)
    agents = [a, b] if a_first else [b, a]
    w = play_game(game, agents)
    plies = getattr(game, "ply", getattr(game, "moves", 0))
    if w == -1:
        return 0, a_first, plies
    a_won = (w == 0) == a_first
    return (1 if a_won else -1), a_first, plies


def match(game_name: str, spec_a: str, spec_b: str, n_games: int = 100, workers: int | None = None,
          seed: int = 0, game_kwargs: dict | None = None, progress: bool = False) -> MatchResult:
    """Match de ``n_games`` parties entre A et B (couleurs alternées)."""
    game_kwargs = game_kwargs or {}
    tasks = [(game_name, game_kwargs, spec_a, spec_b, i % 2 == 0, seed + 7919 * i) for i in range(n_games)]
    name_a, name_b = make_agent(spec_a).name, make_agent(spec_b).name if not spec_b.startswith("human") else "Human"
    res = MatchResult(name_a, name_b)
    t0 = time.perf_counter()
    workers = workers if workers is not None else min(os.cpu_count() or 1, n_games)

    def consume(outcome):
        r, a_first, plies = outcome
        res.lengths.append(plies)
        if a_first:
            res.games_as_first += 1
        if (r == 1 and a_first) or (r == -1 and not a_first):
            res.first_mover_wins += 1
        if r == 1:
            res.wins += 1
            res.wins_as_first += a_first
        elif r == 0:
            res.draws += 1
        else:
            res.losses += 1
        if progress:
            print(f"\r  {res.n}/{n_games}  V{res.wins} N{res.draws} D{res.losses}", end="", flush=True)

    if workers <= 1:
        _init_worker()
        for t in tasks:
            consume(_run_one(t))
    else:
        with ProcessPoolExecutor(workers, mp_context=mp.get_context("spawn"), initializer=_init_worker) as ex:
            for out in ex.map(_run_one, tasks):
                consume(out)
    if progress:
        print()
    res.seconds = time.perf_counter() - t0
    return res
