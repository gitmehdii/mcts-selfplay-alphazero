"""Réseau policy + value façon AlphaZero (tronc résiduel convolutif, deux têtes)."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.c1 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.b1 = nn.BatchNorm2d(ch)
        self.c2 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.b2 = nn.BatchNorm2d(ch)

    def forward(self, x):
        y = F.relu(self.b1(self.c1(x)))
        y = self.b2(self.c2(y))
        return F.relu(x + y)


class AZNet(nn.Module):
    """f_θ(s) = (logits de la policy p, valeur v ∈ [-1, 1] pour le joueur au trait)."""

    def __init__(self, obs_shape: tuple[int, int, int], action_size: int, channels: int = 64, blocks: int = 4,
                 policy_channels: int = 16):
        super().__init__()
        c_in, h, w = obs_shape
        self.config = dict(obs_shape=tuple(obs_shape), action_size=action_size, channels=channels, blocks=blocks,
                           policy_channels=policy_channels)
        self.stem = nn.Sequential(nn.Conv2d(c_in, channels, 3, padding=1, bias=False), nn.BatchNorm2d(channels),
                                  nn.ReLU())
        self.tower = nn.Sequential(*[ResBlock(channels) for _ in range(blocks)])
        # tête policy
        self.p_conv = nn.Sequential(nn.Conv2d(channels, policy_channels, 1, bias=False),
                                    nn.BatchNorm2d(policy_channels), nn.ReLU())
        self.p_fc = nn.Linear(policy_channels * h * w, action_size)
        # tête value
        self.v_conv = nn.Sequential(nn.Conv2d(channels, 2, 1, bias=False), nn.BatchNorm2d(2), nn.ReLU())
        self.v_fc = nn.Sequential(nn.Linear(2 * h * w, 128), nn.ReLU(), nn.Linear(128, 1))

    def forward(self, x):
        x = self.tower(self.stem(x))
        p = self.p_fc(self.p_conv(x).flatten(1))
        v = torch.tanh(self.v_fc(self.v_conv(x).flatten(1))).squeeze(1)
        return p, v

    # ------------------------------------------------------------ sauvegarde
    def save(self, path: str, **meta) -> None:
        torch.save({"config": self.config, "state_dict": self.state_dict(), "meta": meta}, path)

    @classmethod
    def load(cls, path: str, device: str | torch.device = "cpu") -> "AZNet":
        ck = torch.load(path, map_location=device, weights_only=False)
        net = cls(**ck["config"])
        net.load_state_dict(ck["state_dict"])
        net.meta = ck.get("meta", {})
        return net.to(device).eval()


class NetEvaluator:
    """Évalue un lot de positions : renvoie (logits numpy (B, A), valeurs numpy (B,))."""

    def __init__(self, net: AZNet, device: str | torch.device = "cpu"):
        self.net = net.to(device).eval()
        self.device = torch.device(device)
        self.calls = 0

    @torch.no_grad()
    def __call__(self, states) -> tuple[np.ndarray, np.ndarray]:
        self.calls += 1
        x = torch.from_numpy(np.stack([s.encode() for s in states])).to(self.device, non_blocking=True)
        logits, v = self.net(x)
        return logits.float().cpu().numpy(), v.float().cpu().numpy()
