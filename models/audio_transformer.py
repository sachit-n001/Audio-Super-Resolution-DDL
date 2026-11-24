"""
Full definition of a GPT Language Model, all of it in this single file.
References:
1) the official GPT-2 TensorFlow implementation released by OpenAI:
https://github.com/openai/gpt-2/blob/master/src/model.py
2) huggingface/transformers PyTorch implementation:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
"""


"""
Audio Super-Resolution Transformer.

Adapted from Andrej Karpathy's nanoGPT:
https://github.com/karpathy/nanoGPT

Original license: see LICENSE.nanoGPT.

Modifications:
- Bidirectional self-attention
- Continuous audio patch embeddings
- 8 kHz to 44.1 kHz waveform reconstruction
- Residual audio prediction
- L1 reconstruction loss

Rutgers ECE 534 - Distributed Deep Learning
"""



import math
import inspect
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F

class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)

class BidirectionalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()

        assert config.n_embd % config.n_head == 0

        self.c_attn = nn.Linear(
            config.n_embd, 3 * config.n_embd, bias=config.bias
        )

        self.c_proj = nn.Linear(
            config.n_embd, config.n_embd, bias=config.bias
        )

        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout

    def forward(self, x):
        B, T, C = x.size()

        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)

        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        # Bidirectional attention for audio reconstruction
        y = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=False
        )

        y = y.transpose(1, 2).contiguous().view(B, T, C)

        y = self.resid_dropout(self.c_proj(y))

        return y

class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = BidirectionalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

@dataclass
class GPTConfig:
    block_size: int = 200
    patch_size: int = 441
    low_sr: int = 8000
    high_sr: int = 44100
    n_layer: int = 6
    n_head: int = 8
    n_embd: int = 256
    dropout: float = 0.1
    bias: bool = True

class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        # Project audio patches into transformer embeddings
        self.input_proj = nn.Linear(
            config.patch_size, config.n_embd
        )

        self.transformer = nn.ModuleDict(dict(
            wpe=nn.Embedding(config.block_size, config.n_embd),
            drop=nn.Dropout(config.dropout),
            h=nn.ModuleList([
                Block(config) for _ in range(config.n_layer)
            ]),
            ln_f=LayerNorm(config.n_embd, bias=config.bias)
        ))

        # Predict waveform corrections for each patch
        self.output_proj = nn.Linear(
            config.n_embd, config.patch_size
        )

        self.apply(self._init_weights)

        # Start with the interpolated waveform as the baseline
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

            if module.bias is not None:
                nn.init.zeros_(module.bias)

        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, low_audio, targets=None):

        B, L = low_audio.shape

        # Upsample from 8 kHz to 44.1 kHz
        high_length = round(
            L * self.config.high_sr / self.config.low_sr
        )

        upsampled = F.interpolate(
            low_audio.unsqueeze(1),
            size=high_length,
            mode="linear",
            align_corners=False
        ).squeeze(1)

        # Divide waveform into audio patches
        assert high_length % self.config.patch_size == 0

        patches = upsampled.reshape(
            B, -1, self.config.patch_size
        )

        T = patches.size(1)
        assert T <= self.config.block_size

        # Audio and positional embeddings
        pos = torch.arange(T, device=low_audio.device)

        x = self.input_proj(patches)
        x = x + self.transformer.wpe(pos)
        x = self.transformer.drop(x)

        # Transformer blocks
        for block in self.transformer.h:
            x = block(x)

        x = self.transformer.ln_f(x)

        # Reconstruct missing audio information
        residual = self.output_proj(x)
        residual = residual.reshape(B, high_length)

        reconstructed = upsampled + residual

        # Audio reconstruction loss
        loss = None

        if targets is not None:
            loss = F.l1_loss(reconstructed, targets)

        return reconstructed, loss

    def get_num_params(self):
        return sum(p.numel() for p in self.parameters())