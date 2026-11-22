"""Construction d'agents à partir d'une chaîne de spécification.

Exemples :
    random
    heuristic
    human
    mcts:sims=1000,c=1.41,rollout=heuristic
    mcts:time_limit=2,max_depth=30          (échecs : rollouts tronqués)
    az:ckpt=runs/c4/best.pt,sims=200        (AlphaZero : réseau + PUCT)
    az:ckpt=runs/c4/best.pt,sims=0          (réseau seul : argmax de la policy)
"""
from __future__ import annotations

from .base import Agent, HeuristicAgent, HumanAgent, RandomAgent
from .mcts import MCTSAgent


def _parse_value(v: str):
    for cast in (int, float):
        try:
            return cast(v)
        except ValueError:
            pass
    if v.lower() in ("none", "null"):
        return None
    if v.lower() in ("true", "false"):
        return v.lower() == "true"
    return v


def parse_spec(spec: str) -> tuple[str, dict]:
    kind, _, rest = spec.partition(":")
    kwargs = {}
    for item in filter(None, rest.split(",")):
        k, _, v = item.partition("=")
        kwargs[k.strip()] = _parse_value(v.strip())
    return kind.strip().lower(), kwargs


def make_agent(spec: str, seed: int | None = None, **overrides) -> Agent:
    kind, kwargs = parse_spec(spec)
    kwargs.update(overrides)
    if kind == "random":
        return RandomAgent(seed=seed)
    if kind == "heuristic":
        return HeuristicAgent(seed=seed)
    if kind == "human":
        return HumanAgent()
    if kind == "mcts":
        return MCTSAgent(seed=seed, **kwargs)
    if kind in ("az", "alphazero"):
        from alphazero.agent import AlphaZeroAgent

        return AlphaZeroAgent(seed=seed, **kwargs)
    raise ValueError(f"agent inconnu : {spec}")


__all__ = ["Agent", "RandomAgent", "HeuristicAgent", "HumanAgent", "MCTSAgent", "make_agent", "parse_spec"]
