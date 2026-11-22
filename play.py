#!/usr/bin/env python3
"""Point d'entrée : faire jouer deux agents (ou un humain contre un agent).

Exemples :
    # humain contre MCTS à Puissance 4
    python play.py --game connect4 --p1 human --p2 mcts:sims=2000

    # 100 parties MCTS vs aléatoire, sur 16 processus
    python play.py --game connect4 --p1 mcts:sims=500 --p2 random --games 100

    # échecs (variante réduite) : AlphaZero vs MCTS pur
    python play.py --game chess:mini --p1 az:ckpt=runs/chess_mini/best.pt,sims=100 \
                   --p2 mcts:sims=200,max_depth=20,rollout=captures --games 20
"""
from __future__ import annotations

import argparse

from agents import make_agent
from arena import match, play_game
from games import make_game


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--game", default="connect4", help="connect4 | chess:mini | chess:micro | chess:standard")
    ap.add_argument("--p1", default="human", help="agent du joueur 1 (premier à jouer)")
    ap.add_argument("--p2", default="mcts:sims=1000", help="agent du joueur 2")
    ap.add_argument("--games", type=int, default=1, help="nombre de parties (couleurs alternées si > 1)")
    ap.add_argument("--workers", type=int, default=None, help="processus parallèles pour les matchs")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--verbose", action="store_true", help="afficher les statistiques MCTS à chaque coup")
    args = ap.parse_args()

    interactive = "human" in (args.p1 + args.p2) or args.games == 1
    if interactive:
        for i in range(args.games):
            extra = {"verbose": True} if args.verbose else {}
            agents = []
            for k, spec in enumerate((args.p1, args.p2)):
                kw = extra if spec.startswith(("mcts", "az")) else {}
                agents.append(make_agent(spec, seed=args.seed + 10 * i + k, **kw))
            game = make_game(args.game)
            try:
                w = play_game(game, agents, verbose=True)
            except KeyboardInterrupt:
                print("\nPartie interrompue.")
                return
            print("Match nul !" if w == -1 else f"Victoire de {agents[w].name} (joueur {w + 1}) !")
    else:
        res = match(args.game, args.p1, args.p2, n_games=args.games, workers=args.workers, seed=args.seed,
                    progress=True)
        print(res)
        print(f"  dont {res.wins_as_first}/{res.games_as_first} victoires de {res.name_a} quand il commence")


if __name__ == "__main__":
    main()
