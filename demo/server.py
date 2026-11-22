#!/usr/bin/env python3
"""Démo de stand : jouer contre nos agents dans le navigateur.

    python demo/server.py                 # puis ouvrir http://127.0.0.1:8000
    python demo/server.py --port 8080 --ckpt runs/connect4/best.pt --chess-ckpt runs/chess_mini/best.pt

Deux pages : Puissance 4 (``/``) et échecs, variante mini (``/echecs``).
Serveur sans dépendance supplémentaire (bibliothèque standard + notre code).
Le navigateur envoie la liste des coups joués ; le serveur reconstruit la
position, fait réfléchir l'agent demandé et renvoie son coup avec les
statistiques de sa recherche (visites, valeurs, intuition du réseau).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402

from agents.mcts import MCTSAgent  # noqa: E402
from alphazero.mcts import PUCTTree, run_search, softmax_legal  # noqa: E402
from games import make_game  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
LOCK = threading.Lock()  # un calcul à la fois (torch + arbres partagent le CPU)
RNG = np.random.default_rng()
NETS: dict[str, object] = {"connect4": None, "chess": None, "chess_std": None}

# Niveaux proposés aux visiteurs (les chiffres viennent de nos expériences).
C4_LEVELS = {
    "facile": {"label": "Facile", "kind": "mcts", "sims": 50,
               "desc": "MCTS pur : 50 parties simulées au hasard avant chaque coup."},
    "moyen": {"label": "Moyen", "kind": "mcts", "sims": 1000,
              "desc": "MCTS pur : 1 000 parties simulées au hasard avant chaque coup."},
    "instinct": {"label": "Instinct", "kind": "az", "sims": 0,
                 "desc": "Le réseau seul, sans aucune recherche : il joue à l’intuition apprise en self-play."},
    "expert": {"label": "Expert", "kind": "az", "sims": 200,
               "desc": "AlphaZero : réseau + 200 simulations guidées. Il bat le MCTS pur à 97 %."},
}
CHESS_LEVELS = {
    "instinct": {"label": "Instinct", "kind": "az", "sims": 0,
                 "desc": "Le réseau seul, sans recherche : ce qu’il a retenu de 4 608 parties contre lui-même."},
    "mcts": {"label": "MCTS pur", "kind": "mcts", "sims": 300,
             "desc": "300 simulations : rollouts qui privilégient les captures, puis comptage du matériel."},
    "expert": {"label": "AlphaZero", "kind": "az", "sims": 200,
               "desc": "Réseau + 200 simulations guidées. À simulations égales, il bat le MCTS pur à 70 %."},
}
CHESS_STD_LEVELS = {
    "instinct": {"label": "Instinct", "kind": "az", "sims": 0,
                 "desc": "Le réseau seul, sans recherche : il gagne du matériel mais sait mal conclure (souvent nul)."},
    "mcts": {"label": "MCTS pur", "kind": "mcts", "sims": 300,
             "desc": "300 simulations : rollouts qui privilégient les captures, puis comptage du matériel."},
    "expert": {"label": "AlphaZero", "kind": "az", "sims": 200,
               "desc": "Réseau entraîné en 24\u202f576 parties sur Modal + 200 simulations. Bat le MCTS pur à 90 %."},
}
LEVELS = {"connect4": C4_LEVELS, "chess": CHESS_LEVELS, "chess_std": CHESS_STD_LEVELS}
VARIANT_FAMILY = {"mini": "chess", "standard": "chess_std"}


def load_network(game: str, ckpt: str) -> None:
    if not os.path.exists(ckpt):
        print(f"[démo] Pas de réseau pour {game} ({ckpt}) : niveaux réseau désactivés.", flush=True)
        return
    import torch

    from alphazero.agent import load_evaluator

    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
    NETS[game] = load_evaluator(ckpt, "cpu")
    print(f"[démo] Réseau {game} chargé : {ckpt}", flush=True)


# ---------------------------------------------------------------- recherche commune
def search(game, family: str, level: str):
    """Fait réfléchir un agent. Renvoie (coup, stats par coup candidat, nb de simulations)."""
    spec = LEVELS[family][level]
    if spec["kind"] == "mcts":
        extra = {"max_depth": 20, "rollout": "captures"} if family.startswith("chess") else {}
        agent = MCTSAgent(sims=spec["sims"], seed=int(RNG.integers(1 << 31)), **extra)
        root = agent.search(game)
        stats = [(ch.move, ch.N, ch.W / ch.N, None) for ch in root.children]
        return max(root.children, key=lambda ch: ch.N).move, stats, root.N
    ev = NETS[family]
    if ev is None:
        raise RuntimeError("réseau non chargé")
    moves = game.legal_moves()
    logits, _ = ev([game])
    priors = softmax_legal(logits[0], [game.move_to_action(m) for m in moves])
    if spec["sims"] == 0:
        stats = [(m, None, None, float(p)) for m, p in zip(moves, priors)]
        return moves[int(np.argmax(priors))], stats, 0
    tree = PUCTTree(game, c_puct=1.5, rng=RNG)
    run_search(tree, ev, spec["sims"])
    stats = [(ch.move, ch.N, ch.Q if ch.N else None, ch.prior) for ch in tree.root.children]
    return tree.choose(0.0).move, stats, spec["sims"]


def value_for_first(game, family: str) -> float | None:
    """Évaluation dans [-1, 1] du point de vue du premier joueur (Jaune / Blancs)."""
    if game.is_terminal():
        return game.result(0)
    ev = NETS[family]
    if ev is None:
        return None
    _, v = ev([game])
    v = float(v[0])
    return v if game.current_player == 0 else -v


# ---------------------------------------------------------------- Puissance 4
def c4_build(moves):
    game = make_game("connect4")
    for m in moves:
        if not isinstance(m, int) or game.is_terminal() or m not in game.legal_moves():
            raise ValueError(f"coup invalide : {m}")
        game.play(m)
    return game


def c4_move(data):
    game = c4_build(data.get("moves", []))
    level = data.get("level", "expert")
    if level not in C4_LEVELS or game.is_terminal():
        raise ValueError("niveau inconnu ou partie terminée")
    t0 = time.perf_counter()
    move, stats, sims = search(game, "connect4", level)
    think = (time.perf_counter() - t0) * 1000
    cols = sorted(({"col": m, "visits": n, "q": q, "prior": p} for m, n, q, p in stats), key=lambda c: c["col"])
    game.play(move)
    return {"move": int(move), "level": level, "label": C4_LEVELS[level]["label"], "sims": int(sims),
            "think_ms": round(think), "columns": cols, "v_first": value_for_first(game, "connect4")}


def c4_eval(data):
    return {"v_first": value_for_first(c4_build(data.get("moves", [])), "connect4")}


# ---------------------------------------------------------------- échecs (variante mini)
def _variant(data) -> str:
    v = data.get("variant", "mini")
    if v not in VARIANT_FAMILY:
        raise ValueError("variante inconnue")
    return v


def chess_build(moves, variant="mini"):
    import chess

    game = make_game(f"chess:{variant}")
    for u in moves:
        try:
            m = chess.Move.from_uci(str(u))
        except ValueError:
            raise ValueError(f"coup invalide : {u}")
        if game.is_terminal() or m not in game.legal_moves():
            raise ValueError(f"coup illégal : {u}")
        game.play(m)
    return game


def chess_state(game, variant="mini") -> dict:
    import chess

    from games.chess_game import VARIANTS

    b = game.board
    over = game.is_terminal()
    reason = None
    if over:
        if not game._all_legal():
            reason = "échec et mat" if b.is_check() else "pat"
        elif b.is_insufficient_material():
            reason = "matériel insuffisant"
        elif b.halfmove_clock >= 150:
            reason = "règle des 75 coups"
        elif b.is_fivefold_repetition():
            reason = "répétition de la position"
        else:
            reason = "limite de 100 demi-coups : arbitrage au matériel"
    replay, san = chess.Board(VARIANTS[variant][0]), []
    for m in b.move_stack:
        san.append(replay.san(m))
        replay.push(m)
    king = b.king(b.turn)
    return {"fen": b.fen(), "turn": "w" if b.turn == chess.WHITE else "b",
            "legal": [m.uci() for m in game.legal_moves()], "check": b.is_check(),
            "king": chess.square_name(king) if king is not None else None,
            "last": b.move_stack[-1].uci() if b.move_stack else None, "san": san,
            "over": over, "winner": game.winner() if over else None, "reason": reason,
            "ply": game.ply, "max_plies": game.max_plies, "material": game.material_diff(),
            "v_white": value_for_first(game, VARIANT_FAMILY[variant]), "variant": variant}


def chess_move(data):
    variant = _variant(data)
    family = VARIANT_FAMILY[variant]
    game = chess_build(data.get("moves", []), variant)
    level = data.get("level", "expert")
    if level not in LEVELS[family] or game.is_terminal():
        raise ValueError("niveau inconnu ou partie terminée")
    t0 = time.perf_counter()
    move, stats, sims = search(game, family, level)
    think = (time.perf_counter() - t0) * 1000
    cands = [{"uci": m.uci(), "san": game.board.san(m), "visits": n, "q": q, "prior": p} for m, n, q, p in stats]
    cands.sort(key=lambda c: -(c["visits"] if c["visits"] is not None else c["prior"]))
    san = game.board.san(move)
    game.play(move)
    return {"move": move.uci(), "san_move": san, "level": level, "label": LEVELS[family][level]["label"],
            "sims": int(sims), "think_ms": round(think), "candidates": cands[:6],
            "n_legal": len(stats), "state": chess_state(game, variant)}


def chess_state_api(data):
    variant = _variant(data)
    return chess_state(chess_build(data.get("moves", []), variant), variant)


def info(family: str):
    levels = {}
    for k, v in LEVELS[family].items():
        levels[k] = {kk: v[kk] for kk in ("label", "kind", "sims", "desc")}
        levels[k]["available"] = v["kind"] == "mcts" or NETS[family] is not None
    return {"levels": levels}


ROUTES_POST = {
    "/api/move": c4_move, "/api/eval": c4_eval,
    "/api/chess/move": chess_move, "/api/chess/state": chess_state_api,
}
PAGES = {"/": "index.html", "/index.html": "index.html", "/echecs": "chess.html", "/chess.html": "chess.html"}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):  # journal discret
        pass

    def _send(self, code: int, body: bytes, ctype: str):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, code: int, obj):
        self._send(code, json.dumps(obj).encode(), "application/json; charset=utf-8")

    def do_GET(self):
        path = self.path.split("?", 1)[0]
        if path in PAGES:
            with open(os.path.join(HERE, PAGES[path]), "rb") as f:
                self._send(200, f.read(), "text/html; charset=utf-8")
        elif path == "/api/info":
            self._json(200, info("connect4"))
        elif path == "/api/chess/info":
            q = self.path.partition("?")[2]
            family = "chess_std" if "variant=standard" in q else "chess"
            self._json(200, info(family))
        else:
            self._send(404, b"introuvable", "text/plain; charset=utf-8")

    def do_POST(self):
        fn = ROUTES_POST.get(self.path)
        if fn is None:
            return self._send(404, b"introuvable", "text/plain; charset=utf-8")
        try:
            n = int(self.headers.get("Content-Length", 0))
            data = json.loads(self.rfile.read(n) or b"{}")
            with LOCK:
                out = fn(data)
            self._json(200, out)
        except (ValueError, RuntimeError, TypeError) as e:
            self._json(400, {"error": str(e)})


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--ckpt", default=os.path.join(ROOT, "runs", "connect4", "latest.pt"))
    ap.add_argument("--chess-ckpt", default=os.path.join(ROOT, "runs", "chess_mini", "latest.pt"))
    ap.add_argument("--chess-std-ckpt", default=os.path.join(ROOT, "runs", "modal", "chess_standard_v2", "final.pt"),
                    help="réseau entraîné sur Modal pour les échecs complets (repli : latest.pt)")
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args()

    load_network("connect4", args.ckpt)
    load_network("chess", args.chess_ckpt)
    std = args.chess_std_ckpt
    if not os.path.exists(std):
        std = os.path.join(os.path.dirname(std), "latest.pt")
    if os.path.exists(std):
        load_network("chess_std", std)
    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    url = f"http://{args.host}:{args.port}"
    print(f"[démo] Prêt : {url}  (Puissance 4)  ·  {url}/echecs  (échecs)   Ctrl+C pour arrêter", flush=True)
    if not args.no_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\n[démo] Arrêt.")


if __name__ == "__main__":
    main()
