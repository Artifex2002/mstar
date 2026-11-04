"""Inference-only IRIS networks; attention and cache access are supplied by the engine."""

from typing import NamedTuple

import torch
from torch import nn
from torch.nn import functional as F


def _norm():
    return nn.GroupNorm(32, 64, eps=1e-6)


def _swish(x):
    return x * torch.sigmoid(x)


class Residual(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm1, self.norm2 = _norm(), _norm()
        self.conv1, self.conv2 = (nn.Conv2d(64, 64, 3, padding=1) for _ in range(2))

    def forward(self, x):
        return x + self.conv2(_swish(self.norm2(self.conv1(_swish(self.norm1(x))))))


class SpatialAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = _norm()
        self.q, self.k, self.v, self.proj_out = (nn.Conv2d(64, 64, 1) for _ in range(4))

    def forward(self, x, attend):
        normalized = self.norm(x)
        batch, _, height, width = x.shape
        q, k, v = (
            projection(normalized).flatten(2).transpose(1, 2).reshape(-1, 1, 64)
            for projection in (self.q, self.k, self.v)
        )
        output = attend(q, k, v, label=f"spatial_{height}")
        output = output.reshape(batch, height * width, 64).transpose(1, 2).reshape_as(x)
        return x + self.proj_out(output)


class Resize(nn.Module):
    def __init__(self, up):
        super().__init__()
        self.up = up
        self.conv = nn.Conv2d(64, 64, 3, stride=1 if up else 2, padding=1 if up else 0)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode="nearest") if self.up else F.pad(x, (0, 1, 0, 1))
        return self.conv(x)


class ImageStage(nn.Module):
    def __init__(self, level, decode):
        super().__init__()
        resolution = 64 // 2**level
        self.block = nn.ModuleList(Residual() for _ in range(3 if decode else 2))
        self.attn = nn.ModuleList(SpatialAttention() for _ in self.block) if resolution in (8, 16) else nn.ModuleList()
        if decode and level > 0:
            self.upsample = Resize(up=True)
        if not decode and level < 4:
            self.downsample = Resize(up=False)

    def forward(self, x, attend):
        for index, block in enumerate(self.block):
            x = block(x)
            if self.attn:
                x = self.attn[index](x, attend)
        if hasattr(self, "upsample"):
            x = self.upsample(x)
        if hasattr(self, "downsample"):
            x = self.downsample(x)
        return x


class ImageMiddle(nn.Module):
    def __init__(self):
        super().__init__()
        self.block_1, self.attn_1, self.block_2 = Residual(), SpatialAttention(), Residual()

    def forward(self, x, attend):
        return self.block_2(self.attn_1(self.block_1(x), attend))


class ImageNetwork(nn.Module):
    def __init__(self, decode=False):
        super().__init__()
        self.decode = decode
        self.conv_in = nn.Conv2d(512 if decode else 3, 64, 3, padding=1)
        stages = nn.ModuleList(ImageStage(level, decode) for level in range(5))
        if decode:
            self.up = stages
        else:
            self.down = stages
        self.mid, self.norm_out = ImageMiddle(), _norm()
        self.conv_out = nn.Conv2d(64, 3 if decode else 512, 3, padding=1)

    def forward(self, x, attend):
        x = self.conv_in(x)
        if self.decode:
            x = self.mid(x, attend)
            for stage in reversed(self.up):
                x = stage(x, attend)
        else:
            for stage in self.down:
                x = stage(x, attend)
            x = self.mid(x, attend)
        return self.conv_out(_swish(self.norm_out(x)))


class EncodedImage(NamedTuple):
    z: torch.Tensor
    z_quantized: torch.Tensor
    tokens: torch.Tensor


class Tokenizer(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder, self.decoder = ImageNetwork(), ImageNetwork(decode=True)
        self.pre_quant_conv, self.post_quant_conv = (nn.Conv2d(512, 512, 1) for _ in range(2))
        self.embedding = nn.Embedding(512, 512)

    def encode(self, image, attend):
        z = self.pre_quant_conv(self.encoder(image * 2 - 1, attend))
        vectors = z.permute(0, 2, 3, 1).reshape(-1, 512)
        table = self.embedding.weight
        distances = vectors.square().sum(1, keepdim=True) + table.square().sum(1) - 2 * vectors @ table.T
        tokens = distances.argmin(1).reshape(image.shape[0], 16)
        quantized = self.embedding(tokens).transpose(1, 2).reshape_as(z)
        return EncodedImage(z, quantized, tokens)

    def decode_tokens(self, tokens, attend):
        z = self.embedding(tokens).transpose(1, 2).reshape(-1, 512, 4, 4)
        return ((self.decoder(self.post_quant_conv(z), attend) + 1) / 2).clamp(0, 1)


class WorldAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.query, self.key, self.value, self.proj = (nn.Linear(256, 256) for _ in range(4))

    def forward(self, x, layer, attend):
        q, k, v = (projection(x).reshape(-1, 4, 64) for projection in (self.query, self.key, self.value))
        return self.proj(attend(layer, q, k, v).reshape_as(x))


class WorldBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(256), nn.LayerNorm(256)
        self.attn = WorldAttention()
        self.mlp = nn.Sequential(nn.Linear(256, 1024), nn.GELU(), nn.Linear(1024, 256))

    def forward(self, x, layer, attend):
        x = x + self.attn(self.ln1(x), layer, attend)
        return x + self.mlp(self.ln2(x))


class WorldOutput(NamedTuple):
    output_sequence: torch.Tensor
    logits_observations: torch.Tensor
    logits_rewards: torch.Tensor
    logits_ends: torch.Tensor


class WorldNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList(WorldBlock() for _ in range(10))
        self.ln_f, self.pos_emb = nn.LayerNorm(256), nn.Embedding(340, 256)
        self.action_embedding, self.observation_embedding = nn.Embedding(4, 256), nn.Embedding(512, 256)
        self.head_observations, self.head_rewards, self.head_ends = (
            nn.Sequential(nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, vocab)) for vocab in (512, 3, 2)
        )

    def forward(self, tokens, positions, call, attend):
        embedding = self.action_embedding if call.kind == "action" else self.observation_embedding
        x = embedding(tokens) + self.pos_emb(positions.reshape(tokens.shape))
        for layer, block in enumerate(self.blocks):
            x = block(x, layer, attend)
        x = self.ln_f(x)
        # The official heads omit position 15 mod 17; reward/end exist only on actions.
        observation_rows = x[:, :15] if call.kind == "prime" else x if call.sample_index is not None else x[:, :0]
        action_rows = x if call.kind == "action" else x[:, :0]
        return WorldOutput(
            x, self.head_observations(observation_rows), self.head_rewards(action_rows), self.head_ends(action_rows)
        )
