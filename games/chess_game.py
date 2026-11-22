"""Échecs via ``python-chess`` (règles complètes), avec variantes allégées.

Choix de modélisation (AlphaZero simplifié) :
- espace d'actions fixe de 64 x 64 = 4096 indices ``from * 64 + to`` ;
  les sous-promotions sont retirées des coups légaux (on promeut toujours en
  dame), ce qui ne change pas la légalité des coups ;
- l'encodage et les actions sont exprimés du point de vue du joueur au trait
  (l'échiquier est retourné verticalement quand les Noirs jouent) ;
- les parties sont limitées à ``max_plies`` demi-coups ; à la limite, la partie
  est arbitrée au matériel (avantage >= ``adjudicate_margin``) sinon nulle.
  Cela fournit un signal d'apprentissage même avec peu de calcul.
"""
from __future__ import annotations

import math

import chess
import numpy as np

from .base import Game

VARIANTS = {
    # partie complète
    "standard": (chess.STARTING_FEN, 200),
    # « demi-échecs » : aile roi uniquement (R, F, C, T + 4 pions de chaque côté)
    "mini": ("4kbnr/4pppp/8/8/8/8/4PPPP/4KBNR w Kk - 0 1", 100),
    # encore plus petit : roi, tour et 3 pions
    "micro": ("4k2r/5ppp/8/8/8/8/5PPP/4K2R w Kk - 0 1", 80),
}

PIECE_VALUES = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0}
_PIECES = (chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING)


class ChessGame(Game):
    action_size = 64 * 64
    obs_shape = (18, 8, 8)

    __slots__ = ("board", "variant", "max_plies", "adjudicate_margin", "ply", "_legal", "_winner", "_status_done", "keys")

    def __init__(self, variant: str = "mini", max_plies: int | None = None, adjudicate_margin: int = 3,
                 fen: str | None = None):
        start_fen, default_plies = VARIANTS[variant]
        self.variant = variant
        self.board = chess.Board(fen or start_fen)
        self.max_plies = max_plies or default_plies
        self.adjudicate_margin = adjudicate_margin
        self.ply = 0
        self._legal: list[chess.Move] | None = None
        self._winner: int | None = None
        self._status_done = False
        # Historique des positions depuis le dernier coup irréversible (prise ou pion) : il voyage avec
        # les copies, pour que la recherche et le self-play voient la nulle par répétition.
        self.keys: tuple = (self.board._transposition_key(),)

    @property
    def name(self) -> str:
        return f"chess:{self.variant}"

    def clone(self) -> "ChessGame":
        g = ChessGame.__new__(ChessGame)
        g.board = self.board.copy(stack=False)  # pas d'historique : plus rapide
        g.variant, g.max_plies, g.adjudicate_margin = self.variant, self.max_plies, self.adjudicate_margin
        g.ply = self.ply
        g._legal, g._winner, g._status_done = self._legal, self._winner, self._status_done
        g.keys = self.keys  # tuple immuable : partagé sans copie
        return g

    # ------------------------------------------------------------------ règles
    @property
    def current_player(self) -> int:
        return 0 if self.board.turn == chess.WHITE else 1

    def _all_legal(self) -> list[chess.Move]:
        if self._legal is None:
            self._legal = [m for m in self.board.legal_moves if m.promotion in (None, chess.QUEEN)]
        return self._legal

    def legal_moves(self) -> list[chess.Move]:
        return [] if self.is_terminal() else self._all_legal()

    def play(self, move: chess.Move) -> None:
        self.board.push(move)
        self.ply += 1
        key = self.board._transposition_key()
        self.keys = (key,) if self.board.halfmove_clock == 0 else self.keys + (key,)
        self._legal = None
        self._winner = None
        self._status_done = False

    def material_diff(self) -> int:
        """Matériel (Blancs - Noirs)."""
        b, s = self.board, 0
        for pt in _PIECES[:-1]:
            v = PIECE_VALUES[pt]
            s += v * (b.pieces_mask(pt, chess.WHITE).bit_count() - b.pieces_mask(pt, chess.BLACK).bit_count())
        return s

    def _compute_status(self) -> None:
        self._status_done = True
        b = self.board
        if not self._all_legal():
            # mat : le joueur au trait a perdu ; sinon pat
            self._winner = (1 - self.current_player) if b.is_check() else -1
        elif b.is_insufficient_material() or b.halfmove_clock >= 150 or self.keys.count(self.keys[-1]) >= 3:
            # triple répétition traitée comme nulle (comme dans la recherche d’AlphaZero)
            self._winner = -1
        elif self.ply >= self.max_plies:
            d = self.material_diff()
            if d >= self.adjudicate_margin:
                self._winner = 0
            elif d <= -self.adjudicate_margin:
                self._winner = 1
            else:
                self._winner = -1
        else:
            self._winner = None

    def is_terminal(self) -> bool:
        if not self._status_done:
            self._compute_status()
        return self._winner is not None

    def winner(self) -> int | None:
        if not self._status_done:
            self._compute_status()
        return self._winner

    # ----------------------------------------------------------- aides MCTS
    def rollout_move(self, rng: np.random.Generator, policy: str = "random") -> chess.Move:
        moves = self._all_legal()
        if policy in ("captures", "heuristic"):
            caps = [m for m in moves if self.board.is_capture(m) or m.promotion]
            if caps:
                moves = caps
        return moves[rng.integers(len(moves))]

    def heuristic_value(self) -> float:
        d = self.material_diff()
        if self.current_player == 1:
            d = -d
        return math.tanh(d / 4.0)

    # ------------------------------------------------------- encodage réseau
    def encode(self) -> np.ndarray:
        b = self.board
        me = b.turn
        masks = [b.pieces_mask(pt, me) for pt in _PIECES] + [b.pieces_mask(pt, not me) for pt in _PIECES]
        if b.ep_square is not None:
            masks.append(1 << b.ep_square)
        else:
            masks.append(0)
        bits = np.unpackbits(np.array(masks, dtype=np.uint64).view(np.uint8).reshape(len(masks), 8),
                             axis=1, bitorder="little").reshape(len(masks), 8, 8)
        if me == chess.BLACK:
            bits = bits[:, ::-1, :]  # retournement vertical (case ^ 56)
        out = np.zeros(self.obs_shape, dtype=np.float32)
        out[0:12] = bits[0:12]
        out[12] = b.has_kingside_castling_rights(me)
        out[13] = b.has_queenside_castling_rights(me)
        out[14] = b.has_kingside_castling_rights(not me)
        out[15] = b.has_queenside_castling_rights(not me)
        out[16] = bits[12]
        out[17] = self.ply / self.max_plies  # proximité de l'arbitrage
        return out

    def move_to_action(self, move: chess.Move) -> int:
        f, t = move.from_square, move.to_square
        if self.board.turn == chess.BLACK:
            f, t = f ^ 56, t ^ 56
        return f * 64 + t

    # ------------------------------------------------------------------ IO
    def parse_move(self, text: str) -> chess.Move:
        text = text.strip()
        try:
            move = chess.Move.from_uci(text)
        except ValueError:
            move = self.board.parse_san(text)  # accepte aussi la notation algébrique (Nf3, O-O...)
        if move.promotion and move.promotion != chess.QUEEN:
            raise ValueError("seule la promotion en dame est autorisée")
        if move not in self._all_legal():
            raise ValueError("coup illégal")
        return move

    def move_str(self, move: chess.Move) -> str:
        return move.uci()

    def key(self):
        return self.board.fen()

    def __str__(self) -> str:
        lines = str(self.board).split("\n")
        out = [f"{8 - i} {line}" for i, line in enumerate(lines)]
        out.append("  a b c d e f g h")
        return "\n".join(out)
