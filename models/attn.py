from inspect import isfunction
import torch
from torch import nn, einsum
from einops import rearrange, repeat


def exists(val):
    return val is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


class RetAugmentationLinearAttention(nn.Module):
    def __init__(self, in_dim, d, context_dim):
        super().__init__()

        self.cond_flag = False
        if context_dim != in_dim:
            self.cond_flag = True
            self.to_cond = nn.Linear(context_dim, in_dim, bias=False)

        self.linear_attn = nn.Linear(in_dim, d, bias=False)
        self.to_k = nn.Linear(in_dim, d, bias=False)
        self.to_v = nn.Linear(in_dim, d, bias=False)
        self.out = nn.Linear(d, in_dim, bias=False)

    def forward(self, h, h_retrieved):
        if self.cond_flag:
            h_retrieved = self.to_cond(h_retrieved)

        attn = self.linear_attn(h)
        attn = torch.softmax(attn, dim=-1)
        assert h.shape[-1] == h_retrieved.shape[-1]
        kv_in = torch.cat([h, h_retrieved], dim=1)
        k = self.to_k(kv_in)
        v = self.to_v(kv_in)
        f = torch.bmm(k.permute(0, 2, 1), v)

        h_aug = torch.bmm(attn, f)
        return h + self.out(h_aug)


class CrossAttention(nn.Module):
    def __init__(self, query_dim, context_dim=None, inner_dim=256, dropout=0.):
        super().__init__()
        context_dim = default(context_dim, query_dim)

        self.scale = inner_dim ** -0.5

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, context=None, mask=None):
        q = self.to_q(x)
        context = default(context, x)
        k = self.to_k(context)
        v = self.to_v(context)

        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale

        if exists(mask):
            mask = rearrange(mask, 'b ... -> b (...)')
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b j -> b () j')
            sim.masked_fill_(~mask, max_neg_value)

        attn = sim.softmax(dim=-1)

        out = einsum('b i j, b j d -> b i d', attn, v)
        return self.to_out(out) + x
