Modèle : `runs/modal/chess_standard_v2/latest.pt` · 40 parties par duel, couleurs alternées

| Duel | V | N | D | Score [IC95] | Elo |
|---|---|---|---|---|---|
| AZ vs aléatoire | 36 | 4 | 0 | 95% [84%-99%] | +512 |
| AZ vs MCTS pur, mêmes simulations (64) | 32 | 8 | 0 | 90% [77%-96%] | +382 |
| AZ vs MCTS pur, 5× plus de simulations (320) | 29 | 8 | 3 | 82% [68%-91%] | +269 |
| AZ vs MCTS pur, 10× plus de simulations (640) | 27 | 7 | 6 | 76% [61%-87%] | +203 |
| Réseau seul (0 sim) vs aléatoire | 6 | 33 | 1 | 56% [41%-70%] | +44 |
| Réseau NON entraîné + PUCT vs MCTS pur (64) | 5 | 0 | 35 | 12% [6%-26%] | -338 |
| AZ entraîné vs AZ non entraîné | 29 | 9 | 2 | 84% [70%-92%] | +285 |
