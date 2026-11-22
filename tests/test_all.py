"""Tests rapides : ``python -m pytest -q`` (ou ``python tests/test_all.py``)."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents import make_agent  # noqa: E402
from alphazero.mcts import PUCTTree, run_search  # noqa: E402
from games import make_game  # noqa: E402


def c4(moves):
    g = make_game("connect4")
    for m in moves:
        g.play(m)
    return g


# ------------------------------------------------------------- Puissance 4
def test_c4_wins():
    assert c4([0, 6, 1, 6, 2, 6, 3]).winner() == 0            # horizontale
    assert c4([0, 1, 0, 1, 0, 1, 0]).winner() == 0            # verticale
    assert c4([0, 1, 1, 2, 2, 3, 2, 3, 3, 6, 3]).winner() == 0  # diagonale /
    assert c4([6, 5, 5, 4, 4, 3, 4, 3, 3, 0, 3]).winner() == 0  # diagonale \
    assert c4([0, 1, 0, 1, 0]).winner() is None


def test_c4_full_column_and_draw():
    g = c4([0] * 6)
    assert 0 not in g.legal_moves()
    # grille remplie sans alignement -> nul
    pattern = [0, 1, 0, 1, 0, 1, 2, 3, 2, 3, 2, 3, 1, 0, 1, 0, 1, 0, 3, 2, 3, 2, 3, 2, 4, 5, 4, 5, 4, 5,
               6, 4, 5, 4, 5, 6, 6, 6, 6, 6, 5, 4]
    g = c4(pattern)
    assert g.moves == 42 and g.winner() == -1 and g.legal_moves() == []


def test_c4_encode_symmetry():
    g = c4([0, 3])
    x = g.encode()
    assert x.shape == (3, 6, 7)
    assert x[0].sum() == 1 and x[1].sum() == 1   # joueur au trait = joueur 0 (son pion en colonne 0)
    assert x[0, 5, 0] == 1 and x[1, 5, 3] == 1
    (p2, pi2) = g.symmetries(x, np.arange(7.0))[1]
    assert p2[0, 5, 6] == 1 and pi2[0] == 6


def test_mcts_wins_and_blocks():
    a = make_agent("mcts:sims=400", seed=0)
    assert a.select_move(c4([0, 6, 0, 6, 0, 5])) == 0      # gagner en colonne 1
    assert a.select_move(c4([3, 0, 3, 0, 3])) == 3         # bloquer la verticale adverse


# ------------------------------------------------------------------ échecs
def test_chess_basics():
    import chess

    g = make_game("chess:standard")
    acts = [g.move_to_action(m) for m in g.legal_moves()]
    assert len(acts) == 20 and len(set(acts)) == 20
    g.play(chess.Move.from_uci("e2e4"))
    # vu des Noirs, e7e5 a le même index que e2e4 vu des Blancs
    assert g.move_to_action(chess.Move.from_uci("e7e5")) == chess.E2 * 64 + chess.E4
    assert g.encode().shape == (18, 8, 8)


def test_chess_mate_and_adjudication():
    g = make_game("chess:standard", fen="R5k1/5ppp/8/8/8/8/8/6K1 b - - 0 1")
    assert g.is_terminal() and g.winner() == 0
    g = make_game("chess:mini", max_plies=1)
    g.play(g.legal_moves()[0])
    assert g.is_terminal() and g.winner() == -1  # matériel égal à la limite -> nul
    assert make_agent("mcts:sims=300,max_depth=10", seed=0).select_move(
        make_game("chess:standard", fen="6k1/5ppp/8/8/8/8/8/R5K1 w - - 0 1")).uci() == "a1a8"


def test_chess_threefold_repetition_in_clones():
    """La triple répétition doit être vue aussi dans les copies (recherche et self-play)."""
    import chess

    g = make_game("chess:standard")
    for u in ["g1f3", "g8f6", "f3g1", "f6g8", "g1f3", "g8f6", "f3g1", "f6g8"]:
        g.play(chess.Move.from_uci(u))
    assert g.is_terminal() and g.winner() == -1
    assert g.clone().is_terminal()
    g2 = make_game("chess:standard")
    g2.play(chess.Move.from_uci("e2e4"))  # coup de pion : l’historique repart de zéro
    assert len(g2.keys) == 1


# --------------------------------------------------------------- PUCT
def test_puct_finds_win_with_uniform_net():
    """Avec un réseau « neutre » (policy uniforme, valeur 0), PUCT trouve un gain immédiat
    grâce aux vraies valeurs des feuilles terminales : vérifie les signes de la rétropropagation."""
    g = c4([0, 6, 0, 6, 0, 5])

    def uniform(states):
        return np.zeros((len(states), 7), np.float32), np.zeros(len(states), np.float32)

    tree = PUCTTree(g)
    run_search(tree, uniform, 200)
    assert max(tree.root.children, key=lambda c: c.N).move == 0
    assert tree.root.children[0].Q > 0.9


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
