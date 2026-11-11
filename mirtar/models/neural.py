"""Neural predictors. All networks take token tensors (miRNA, reversed target).

M2  CrossAttentionNet   miRNA and target attend to each other; the attention is
                        biased by the local base-pairing pattern (pair_bias).
M3  ComplementarityNet  residual 2D CNN on a grid of pair types (G:C, A:U, G:U,
                        mismatch) in the antiparallel orientation.
    ExampleCNN          the course example network (16 ordered-pair channels + pad,
                        both sequences 5'->3'), re-implemented as the reference.
"""
from __future__ import annotations

from copy import deepcopy

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from ..data import MAX_MIRNA_LENGTH, TARGET_LENGTH
from ..pairing import PAIR_TABLE, encode_table
from .base import PairModel


# ----------------------------------------------------------------------------- M3
class ResidualBlock(nn.Module):
    def __init__(self, channels: int, dilation: int, dropout: float):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=dilation, dilation=dilation, bias=False)
        self.norm1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.norm2 = nn.BatchNorm2d(channels)
        self.dropout = nn.Dropout2d(dropout)

    def forward(self, x):
        y = self.dropout(F.relu(self.norm1(self.conv1(x))))
        return F.relu(x + self.norm2(self.conv2(y)))


class ComplementarityNet(nn.Module):
    """Input channels: one-hot pair type (pad, G:C, A:U, G:U, mismatch) + miRNA position.

    The position channel tells the network where the seed is; without it a
    globally pooled CNN could not distinguish a seed match from a 3' match.
    """

    def __init__(self, channels: int = 32, dilations=(1, 2, 1), dropout: float = 0.2, position_channel: bool = True):
        super().__init__()
        self.position_channel = position_channel
        self.register_buffer("pair_table", torch.from_numpy(PAIR_TABLE))
        self.register_buffer("position", torch.linspace(0, 1, MAX_MIRNA_LENGTH).view(1, 1, -1, 1))
        self.stem = nn.Sequential(
            nn.Conv2d(5 + int(position_channel), channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels), nn.ReLU(),
        )
        self.blocks = nn.Sequential(*[ResidualBlock(channels, d, dropout) for d in dilations])
        self.head = nn.Sequential(nn.Linear(2 * channels, 64), nn.ReLU(), nn.Dropout(dropout), nn.Linear(64, 1))

    def grid(self, mirna, target):
        pair_class = self.pair_table[mirna[:, :, None], target[:, None, :]]
        x = F.one_hot(pair_class, 5).permute(0, 3, 1, 2).float()
        if self.position_channel:
            x = torch.cat([x, self.position.expand(len(x), 1, -1, x.shape[-1])], dim=1)
        return x

    def forward(self, mirna, target):
        x = self.blocks(self.stem(self.grid(mirna, target)))
        pooled = torch.cat([x.amax(dim=(2, 3)), x.mean(dim=(2, 3))], dim=1)
        return self.head(pooled).squeeze(1)


# ----------------------------------------------------------------------------- M2
class BiasedAttention(nn.Module):
    """Multi-head attention with an optional additive bias on the attention logits."""

    def __init__(self, dim: int, heads: int, dropout: float):
        super().__init__()
        self.heads = heads
        self.query, self.key, self.value, self.out = (nn.Linear(dim, dim) for _ in range(4))
        self.dropout = nn.Dropout(dropout)
        self.last_attention = None

    def forward(self, x, context, key_padding=None, bias=None, edge=None):
        """``bias``: batch x heads x queries x keys, added to the attention logits.
        ``edge``: batch x queries x keys x dim, added to the value of each (query, key) pair,
        so a position also receives *how* it pairs with what it attends to."""
        batch, n, dim = x.shape
        split = lambda tensor: tensor.view(batch, -1, self.heads, dim // self.heads).transpose(1, 2)
        q, k, v = split(self.query(x)), split(self.key(context)), split(self.value(context))
        logits = q @ k.transpose(-1, -2) / (dim // self.heads) ** 0.5  # batch x heads x n x m
        if bias is not None:
            logits = logits + bias
        if key_padding is not None:
            logits = logits.masked_fill(key_padding[:, None, None, :], float("-inf"))
        attention = logits.softmax(dim=-1)
        self.last_attention = attention.detach()
        weights = self.dropout(attention)
        mixed = weights @ v
        if edge is not None:
            per_head = edge.view(batch, n, -1, self.heads, dim // self.heads)
            mixed = mixed + torch.einsum("bhnm,bnmhd->bhnd", weights, per_head)
        return self.out(mixed.transpose(1, 2).reshape(batch, n, dim))


class CrossBlock(nn.Module):
    def __init__(self, dim: int, heads: int, dropout: float):
        super().__init__()
        self.cross, self.self_attention = BiasedAttention(dim, heads, dropout), BiasedAttention(dim, heads, dropout)
        self.norm1, self.norm2, self.norm3 = nn.LayerNorm(dim), nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.feed_forward = nn.Sequential(nn.Linear(dim, 2 * dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(2 * dim, dim))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, context, x_padding=None, context_padding=None, bias=None, edge=None):
        x = x + self.dropout(self.cross(self.norm1(x), context, context_padding, bias, edge))
        normed = self.norm2(x)
        x = x + self.dropout(self.self_attention(normed, normed, x_padding))
        return x + self.dropout(self.feed_forward(self.norm3(x)))


class CrossAttentionNet(nn.Module):
    """Each miRNA position looks at all target positions and vice versa.

    ``pair_bias`` adds a term to the attention logits of every head, so that
    attending to a complementary base does not have to be discovered from 6,500
    examples:

    "none"    plain cross-attention (ablation)
    "scalar"  one learned number per pair type (G:C, A:U, G:U, mismatch)
    "conv"    a small 2D convolution over the pair-type grid, so the bias of a
              cell also depends on its neighbours: a base pair inside a stretch
              of consecutive pairs gets a different bias from an isolated one

    ``pair_values`` (needs a pair_bias): the same pairing information is also added
    to the attended values, like edge features in a graph network. The bias only
    steers *where* a position looks; with pair_values the position also learns
    whether what it looked at actually pairs with it.
    """

    def __init__(self, dim: int = 64, heads: int = 4, layers: int = 2, dropout: float = 0.2,
                 pair_bias: str = "conv", bias_channels: int = 16, pair_values: bool = True):
        super().__init__()
        if pair_bias not in ("none", "scalar", "conv"):
            raise ValueError(f"pair_bias must be none, scalar or conv, got {pair_bias!r}")
        self.pair_bias = pair_bias
        self.pair_values = pair_values and pair_bias != "none"
        if self.pair_values:
            self.edge = nn.Embedding(5, dim) if pair_bias == "scalar" else nn.Linear(bias_channels, dim)
        self.register_buffer("pair_table", torch.from_numpy(PAIR_TABLE))
        self.embed_mirna, self.embed_target = nn.Embedding(5, dim, padding_idx=0), nn.Embedding(5, dim, padding_idx=0)
        self.position_mirna = nn.Parameter(torch.randn(MAX_MIRNA_LENGTH, dim) * 0.02)
        self.position_target = nn.Parameter(torch.randn(TARGET_LENGTH, dim) * 0.02)
        # local context (neighbouring bases) before attention
        self.local_mirna = nn.Conv1d(dim, dim, 3, padding=1)
        self.local_target = nn.Conv1d(dim, dim, 3, padding=1)
        self.mirna_blocks = nn.ModuleList(CrossBlock(dim, heads, dropout) for _ in range(layers))
        self.target_blocks = nn.ModuleList(CrossBlock(dim, heads, dropout) for _ in range(layers))
        if pair_bias == "scalar":
            self.bias_mirna = nn.ModuleList(nn.Embedding(5, heads) for _ in range(layers))
            self.bias_target = nn.ModuleList(nn.Embedding(5, heads) for _ in range(layers))
        elif pair_bias == "conv":
            self.bias_trunk = nn.Sequential(
                nn.Conv2d(5, bias_channels, 3, padding=1), nn.ReLU(),
                nn.Conv2d(bias_channels, bias_channels, 3, padding=1), nn.ReLU(),
            )
            self.bias_mirna = nn.ModuleList(nn.Conv2d(bias_channels, heads, 1) for _ in range(layers))
            self.bias_target = nn.ModuleList(nn.Conv2d(bias_channels, heads, 1) for _ in range(layers))
        self.head = nn.Sequential(nn.LayerNorm(4 * dim), nn.Linear(4 * dim, 64), nn.GELU(), nn.Dropout(dropout), nn.Linear(64, 1))

    @staticmethod
    def _pool(x, padding=None):
        if padding is None:
            return torch.cat([x.mean(1), x.amax(1)], dim=1)
        keep = (~padding).unsqueeze(-1)
        mean = (x * keep).sum(1) / keep.sum(1).clamp(min=1)
        return torch.cat([mean, x.masked_fill(~keep, float("-inf")).amax(1)], dim=1)

    def forward(self, mirna, target):
        padding = mirna == 0
        m = self.embed_mirna(mirna) + self.position_mirna
        t = self.embed_target(target) + self.position_target
        m = m + self.local_mirna(m.transpose(1, 2)).transpose(1, 2)
        t = t + self.local_target(t.transpose(1, 2)).transpose(1, 2)
        pair_class = self.pair_table[mirna[:, :, None], target[:, None, :]]  # batch x miRNA x target
        if self.pair_bias == "conv":
            pair_map = self.bias_trunk(F.one_hot(pair_class, 5).permute(0, 3, 1, 2).float())
        edge = None  # batch x miRNA x target x dim
        if self.pair_values:
            edge = self.edge(pair_class if self.pair_bias == "scalar" else pair_map.permute(0, 2, 3, 1))
        for layer, (mirna_block, target_block) in enumerate(zip(self.mirna_blocks, self.target_blocks)):
            bias_m = bias_t = None  # batch x heads x queries x keys
            if self.pair_bias == "scalar":
                bias_m = self.bias_mirna[layer](pair_class).permute(0, 3, 1, 2)
                bias_t = self.bias_target[layer](pair_class).permute(0, 3, 2, 1)
            elif self.pair_bias == "conv":
                bias_m = self.bias_mirna[layer](pair_map)
                bias_t = self.bias_target[layer](pair_map).transpose(2, 3)
            new_m = mirna_block(m, t, x_padding=padding, bias=bias_m, edge=edge)
            t = target_block(t, m, context_padding=padding, bias=bias_t, edge=None if edge is None else edge.transpose(1, 2))
            m = new_m
        return self.head(torch.cat([self._pool(m, padding), self._pool(t)], dim=1)).squeeze(1)

    def attention_map(self, mirna, target):
        """miRNA -> target attention of the last layer, averaged over heads (batch x 28 x 50)."""
        self.eval()
        with torch.no_grad():
            self(mirna, target)
        return self.mirna_blocks[-1].cross.last_attention.mean(1)


# ---------------------------------------------------------------- course example
class ExampleCNN(nn.Module):
    """The network of exercise_10.ipynb: 16 ordered-pair channels + pad, no pairing knowledge."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(17, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2), nn.Flatten(),
            nn.Linear(32 * (MAX_MIRNA_LENGTH // 4) * (TARGET_LENGTH // 4), 32), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(32, 1),
        )

    def forward(self, mirna, target):
        target = target.flip(1)  # back to 5'->3', as in the notebook
        codes = 4 * (mirna - 1)[:, :, None] + (target - 1)[:, None, :]
        codes = torch.where(mirna[:, :, None] == 0, torch.full_like(codes, 16), codes)
        return self.net(F.one_hot(codes, 17).permute(0, 3, 1, 2).float()).squeeze(1)


NETWORKS = {"cross_attention": CrossAttentionNet, "complementarity_cnn": ComplementarityNet, "example_cnn": ExampleCNN}


# ------------------------------------------------------------------ training loop
class NeuralModel(PairModel):
    def __init__(self, network: str, seed: int = 0, rounds: int | None = None, max_epochs: int = 40,
                 patience: int = 8, batch_size: int = 64, learning_rate: float = 1e-3, weight_decay: float = 1e-2,
                 device: str | None = None, verbose: bool = True, **network_args):
        self.network, self.network_args = network, network_args
        self.seed, self.rounds, self.max_epochs, self.patience = seed, rounds, max_epochs, patience
        self.batch_size, self.learning_rate, self.weight_decay = batch_size, learning_rate, weight_decay
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.verbose = verbose
        self.history = []

    @staticmethod
    def _tensors(table):
        mirna, target = encode_table(table)
        return torch.from_numpy(mirna), torch.from_numpy(target)

    def _logits(self, mirna, target):
        outputs = []
        self.net.eval()
        with torch.no_grad():
            for start in range(0, len(mirna), 512):
                batch = slice(start, start + 512)
                outputs.append(self.net(mirna[batch].to(self.device), target[batch].to(self.device)).cpu())
        return torch.cat(outputs)

    def fit(self, train, validation=None):
        torch.manual_seed(self.seed)
        generator = torch.Generator().manual_seed(self.seed)
        self.net = NETWORKS[self.network](**self.network_args).to(self.device)
        optimizer = torch.optim.AdamW(self.net.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        mirna, target = self._tensors(train)
        labels = torch.tensor(train["label"].to_numpy(), dtype=torch.float32)
        early_stopping = validation is not None and self.rounds is None
        if early_stopping:
            validation_inputs = self._tensors(validation)
            validation_labels = torch.tensor(validation["label"].to_numpy(), dtype=torch.float32)
        epochs = self.max_epochs if early_stopping else (self.rounds or self.max_epochs)
        best_loss, best_state, self.best_rounds, self.history = float("inf"), None, epochs, []
        for epoch in range(1, epochs + 1):
            self.net.train()
            order, total = torch.randperm(len(labels), generator=generator), 0.0
            for start in range(0, len(order), self.batch_size):
                rows = order[start:start + self.batch_size]
                if len(rows) < 2:  # batch norm needs more than one example
                    continue
                optimizer.zero_grad()
                logits = self.net(mirna[rows].to(self.device), target[rows].to(self.device))
                loss = F.binary_cross_entropy_with_logits(logits, labels[rows].to(self.device))
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
                optimizer.step()
                total += loss.item() * len(rows)
            record = {"epoch": epoch, "train_loss": total / len(labels)}
            if early_stopping:
                loss = F.binary_cross_entropy_with_logits(self._logits(*validation_inputs), validation_labels).item()
                record["validation_loss"] = loss
                if loss < best_loss:
                    best_loss, self.best_rounds = loss, epoch
                    best_state = deepcopy(self.net.state_dict())
            self.history.append(record)
            if self.verbose:
                print("  " + "  ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in record.items()), flush=True)
            if early_stopping and epoch - self.best_rounds >= self.patience:
                break
        if best_state is not None:
            self.net.load_state_dict(best_state)
        self.net.eval()
        return self

    def predict_proba(self, table) -> np.ndarray:
        return torch.sigmoid(self._logits(*self._tensors(table))).numpy().astype(np.float64)
