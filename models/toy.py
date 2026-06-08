"""
AMK, self-contained toy decoder model (the M0 ground-truth oracle)
===================================================================

A tiny Llama-style decoder (RMSNorm + RoPE + grouped-query attention + SwiGLU MLP, no biases),
in pure PyTorch with no `transformers` dependency. It is:

  * the eager **correctness oracle** for the megakernel (logit equivalence vs this),
  * small enough to hand-lower into the IR for `vm/verify_vm.py` (proves the VM end-to-end),
  * the first target for the real graph importer/lowerer in `schedule/`.

Convention note: weights use torch `nn.Linear` layout `[out, in]`, RoPE uses the Llama
rotate-half convention, both chosen to match `instructions/reference.py` exactly so the
megakernel and eager agree.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ToyConfig:
    vocab: int = 256
    hidden: int = 64
    n_layers: int = 1
    n_heads: int = 4
    n_kv_heads: int = 2          # grouped-query attention
    head_dim: int = 16
    intermediate: int = 128
    rms_eps: float = 1e-6
    rope_theta: float = 10000.0
    max_seq: int = 256

    @property
    def q_dim(self) -> int:
        return self.n_heads * self.head_dim

    @property
    def kv_dim(self) -> int:
        return self.n_kv_heads * self.head_dim


def _rmsnorm(x, w, eps):
    x32 = x.float()
    return (x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + eps)).to(x.dtype) * w


def _rotate_half(x):
    half = x.shape[-1] // 2
    return torch.cat([-x[..., half:], x[..., :half]], dim=-1)


def _rope(x, pos, head_dim, theta):
    """x: [..., n_heads, head_dim]; pos: [seq] long. Llama rotate-half convention."""
    half = head_dim // 2
    inv = 1.0 / (theta ** (torch.arange(0, half, device=x.device, dtype=torch.float32) / half))
    ang = pos.float()[:, None] * inv[None, :]                  # [S, half]
    cos = torch.cat([ang.cos(), ang.cos()], dim=-1)            # [S, head_dim]
    sin = torch.cat([ang.sin(), ang.sin()], dim=-1)
    while cos.dim() < x.dim():
        cos = cos.unsqueeze(-2)
        sin = sin.unsqueeze(-2)
    return (x.float() * cos + _rotate_half(x.float()) * sin).to(x.dtype)


class ToyAttention(nn.Module):
    def __init__(self, cfg: ToyConfig):
        super().__init__()
        self.cfg = cfg
        self.q_proj = nn.Linear(cfg.hidden, cfg.q_dim, bias=False)
        self.k_proj = nn.Linear(cfg.hidden, cfg.kv_dim, bias=False)
        self.v_proj = nn.Linear(cfg.hidden, cfg.kv_dim, bias=False)
        self.o_proj = nn.Linear(cfg.q_dim, cfg.hidden, bias=False)

    def forward(self, x, pos):
        cfg = self.cfg
        S = x.shape[0]
        q = self.q_proj(x).view(S, cfg.n_heads, cfg.head_dim)
        k = self.k_proj(x).view(S, cfg.n_kv_heads, cfg.head_dim)
        v = self.v_proj(x).view(S, cfg.n_kv_heads, cfg.head_dim)
        q = _rope(q, pos, cfg.head_dim, cfg.rope_theta)
        k = _rope(k, pos, cfg.head_dim, cfg.rope_theta)
        rep = cfg.n_heads // cfg.n_kv_heads
        kk = k.repeat_interleave(rep, dim=1)
        vv = v.repeat_interleave(rep, dim=1)
        scale = cfg.head_dim ** -0.5
        scores = torch.einsum("qhd,khd->hqk", q.float(), kk.float()) * scale
        mask = torch.triu(torch.full((S, S), float("-inf"), device=scores.device,
                                     dtype=scores.dtype), diagonal=1)
        scores = scores + mask
        probs = scores.softmax(-1)
        out = torch.einsum("hqk,khd->qhd", probs, vv.float()).to(x.dtype).reshape(S, cfg.q_dim)
        return self.o_proj(out)


class ToyMLP(nn.Module):
    def __init__(self, cfg: ToyConfig):
        super().__init__()
        self.gate_proj = nn.Linear(cfg.hidden, cfg.intermediate, bias=False)
        self.up_proj = nn.Linear(cfg.hidden, cfg.intermediate, bias=False)
        self.down_proj = nn.Linear(cfg.intermediate, cfg.hidden, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class ToyLayer(nn.Module):
    def __init__(self, cfg: ToyConfig):
        super().__init__()
        self.cfg = cfg
        self.input_norm = nn.Parameter(torch.ones(cfg.hidden))
        self.post_norm = nn.Parameter(torch.ones(cfg.hidden))
        self.attn = ToyAttention(cfg)
        self.mlp = ToyMLP(cfg)

    def forward(self, x, pos):
        h = x + self.attn(_rmsnorm(x, self.input_norm, self.cfg.rms_eps), pos)
        h = h + self.mlp(_rmsnorm(h, self.post_norm, self.cfg.rms_eps))
        return h


class ToyLlama(nn.Module):
    def __init__(self, cfg: ToyConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab, cfg.hidden)
        self.layers = nn.ModuleList([ToyLayer(cfg) for _ in range(cfg.n_layers)])
        self.final_norm = nn.Parameter(torch.ones(cfg.hidden))
        self.lm_head = nn.Linear(cfg.hidden, cfg.vocab, bias=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Full prefill forward. input_ids: [S] long -> logits [S, vocab]."""
        pos = torch.arange(input_ids.shape[0], device=input_ids.device)
        x = self.embed(input_ids)
        for layer in self.layers:
            x = layer(x, pos)
        x = _rmsnorm(x, self.final_norm, self.cfg.rms_eps)
        return self.lm_head(x)

    @torch.no_grad()
    def weights_dict(self) -> dict[str, torch.Tensor]:
        """Flat name -> tensor map, the binding surface for the reference/CUDA VM."""
        return {k: v.detach().clone() for k, v in self.state_dict().items()}


def make_toy(seed: int = 0, dtype: torch.dtype = torch.float32, **overrides) -> ToyLlama:
    torch.manual_seed(seed)
    cfg = ToyConfig(**overrides)
    m = ToyLlama(cfg).to(dtype)
    m.eval()
    return m
