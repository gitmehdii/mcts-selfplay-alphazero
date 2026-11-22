from .base import Game
from .connect4 import Connect4


def make_game(name: str, **kwargs) -> Game:
    """Fabrique de jeux à partir d'un nom (``connect4``, ``chess``, ``chess:mini``...)."""
    name = name.lower()
    if name in ("connect4", "c4", "puissance4"):
        return Connect4()
    if name.startswith("chess"):
        from .chess_game import ChessGame  # import paresseux (python-chess optionnel)

        variant = name.split(":", 1)[1] if ":" in name else kwargs.pop("variant", "mini")
        return ChessGame(variant=variant, **kwargs)
    raise ValueError(f"jeu inconnu : {name}")


__all__ = ["Game", "Connect4", "make_game"]
