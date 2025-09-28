from functools import partial

import torch
import torch.nn.functional as F
from torch.nn import Module, ModuleList, Linear, RMSNorm, Identity

from einops import einsum
from einops.layers.torch import Rearrange

from rotary_embedding_torch import RotaryEmbedding

LinearNoBias = partial(Linear, bias = False)

# helper functions

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

# transformer-xl mask w/ flex attn

flex_attention = None

try:
    from torch.nn.attention.flex_attention import flex_attention, create_block_mask
    if torch.cuda.is_available():
        flex_attention = torch.compile(flex_attention)
except ImportError:
    pass

def create_xl_mask(
    seq_len,
    kv_seq_len,
    window_size,
    lookback_blocks = 1, # in transformer-xl, lookback is one window size block, but can be multiple for longer context
    device = None
):
    assert kv_seq_len >= seq_len
    assert window_size <= seq_len

    offset = kv_seq_len - seq_len

    def create_block_mask_fn(_, __, q, k):
        offset_q = q + offset
        block_q = offset_q // window_size
        block_k = k // window_size

        causal_mask = offset_q >= k

        # in transformer-xl, the previous segment is fully attended to - may just double the segments and make this sliding for ease of inference logic

        block_mask = (block_q >= block_k) & (block_q <= (block_k + lookback_blocks))

        return causal_mask & block_mask

    create_kwargs = dict(device = device) if exists(device) else dict()
    return create_block_mask(create_block_mask_fn, B = None, H = None, Q_LEN = seq_len, KV_LEN = kv_seq_len, _compile = True, **create_kwargs)

# transformer-xl with ppo

class Attention(Module):
    def __init__(
        self,
        dim,
        dim_head = 64,
        heads = 8,
        pre_rmsnorm = True
    ):
        super().__init__()
        self.norm = RMSNorm(dim) if pre_rmsnorm else Identity()
        self.scale = dim_head ** -0.5
        self.split_heads = Rearrange('b n (h d) -> b h n d', h = heads)
        self.merge_heads = Rearrange('b h n d -> b n (h d)')

        dim_inner = dim_head * heads
        self.to_q = LinearNoBias(dim, dim_inner)
        self.to_kv = LinearNoBias(dim, dim_inner * 2)
        self.to_out = LinearNoBias(dim_inner, dim)

    def forward(
        self,
        tokens,
    ):
        tokens = self.norm(tokens)

        q, k, v = (self.to_q(tokens), *self.to_kv(tokens).chunk(2, dim = -1))

        q, k, v = map(self.split_heads, (q, k, v))

        q = q * self.scale

        sim = einsum(q, k, 'b h i d, b h j d -> b h i j')

        attn = sim.softmax(dim = -1)

        out = einsum(attn, v, 'b h i j, b h j d -> b h i d')

        out = self.merge_heads(out)

        return self.to_out(out)

class FeedForward(Module):
    def __init__(
        self,
        dim,
        expansion_factor = 4.,
        pre_rmsnorm = True
    ):
        super().__init__()
        self.norm = RMSNorm(dim) if pre_rmsnorm else Identity()

        dim_inner = int(dim * expansion_factor * 2 / 3)

        self.proj_in = Linear(dim, dim_inner * 2)
        self.proj_out = Linear(dim_inner, dim)

    def forward(self, x):
        x = self.norm(x)

        x, gates = self.proj_in(x).chunk(2, dim = -1)

        x = x * F.gelu(gates)

        return self.proj_out(x)

class TransformerXL(Module):
    def __init__(
        self,
        dim,
        depth,
        dim_head = 64,
        heads = 8,
        expansion_factor = 4.,
        final_norm = True
    ):
        super().__init__()

        layers = ModuleList([])

        for _ in range(depth):
            attn = Attention(dim = dim, dim_head = dim_head, heads = heads)

            ff = FeedForward(dim = dim, expansion_factor = expansion_factor)

            layers.append(ModuleList([
                attn, ff
            ]))

        self.layers = layers
        self.norm = RMSNorm(dim) if final_norm else Identity()

    def forward(
        self,
        x
    ):

        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x

        return self.norm(x)

# class

class Actor(Module):
    def __init__(
        self
    ):
        super().__init__()
