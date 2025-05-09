import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_

class AgentAttention(nn.Module):
    def __init__(self, dim, num_patches, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0., sr_ratio=1, agent_num=49):
        super(AgentAttention, self).__init__()

        assert dim % num_heads == 0, f"dim {dim} should be divisible by num_heads {num_heads}."

        self.dim = dim
        self.num_patches = num_patches
        self.num_heads = num_heads
        self.agent_num = agent_num
        self.sr_ratio = sr_ratio

        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        # Query, Key, and Value projections
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        
        # Dropout layers
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

        # Output projection
        self.proj = nn.Linear(dim, dim)

        # Spatial reduction if sr_ratio > 1
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

        # Depthwise convolution
        self.dwc = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, padding=1, groups=dim)

        # Biases for attention mechanisms
        window_size = int(num_patches ** 0.5)
        self.an_bias = nn.Parameter(torch.zeros(num_heads, agent_num, 7, 7))
        self.na_bias = nn.Parameter(torch.zeros(num_heads, agent_num, 7, 7))
        self.ah_bias = nn.Parameter(torch.zeros(1, num_heads, agent_num, window_size // sr_ratio, 1))
        self.aw_bias = nn.Parameter(torch.zeros(1, num_heads, agent_num, 1, window_size // sr_ratio))
        self.ha_bias = nn.Parameter(torch.zeros(1, num_heads, window_size, 1, agent_num))
        self.wa_bias = nn.Parameter(torch.zeros(1, num_heads, 1, window_size, agent_num))

        # Initialize biases
        trunc_normal_(self.an_bias, std=0.02)
        trunc_normal_(self.na_bias, std=0.02)
        trunc_normal_(self.ah_bias, std=0.02)
        trunc_normal_(self.aw_bias, std=0.02)
        trunc_normal_(self.ha_bias, std=0.02)
        trunc_normal_(self.wa_bias, std=0.02)

        # Pooling for agent tokens
        pool_size = int(agent_num ** 0.5)
        self.pool = nn.AdaptiveAvgPool2d(output_size=(pool_size, pool_size))

        # Softmax for attention
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, H, W):
        # print(x.size())
        B,C,H1,W1=x.shape
        x = x.view(B, H1 * W1, C)
        b, n, c = x.shape
        num_heads = self.num_heads
        head_dim = c // num_heads
        # print(x.size())
        # Query projection
        q = self.q(x)

        # Spatial reduction if needed
        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(b, c, H, W)
            x_ = self.sr(x_).reshape(b, c, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(b, -1, 2, c).permute(2, 0, 1, 3)
        else:
            kv = self.kv(x).reshape(b, -1, 2, c).permute(2, 0, 1, 3)

        k, v = kv[0], kv[1]

        agent_tokens = self.pool(q.reshape(b, H, W, c).permute(0, 3, 1, 2)).reshape(b, c, -1).permute(0, 2, 1)

        # Reshape Q, K, V for multi-head attention
        q = q.reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3)
        k = k.reshape(b, n // self.sr_ratio ** 2, num_heads, head_dim).permute(0, 2, 1, 3)
        v = v.reshape(b, n // self.sr_ratio ** 2, num_heads, head_dim).permute(0, 2, 1, 3)
        agent_tokens = agent_tokens.reshape(b, self.agent_num, num_heads, head_dim).permute(0, 2, 1, 3)

        # Position biases for agent-to-grid attention
        kv_size = (H // self.sr_ratio, W // self.sr_ratio)
        position_bias1 = nn.functional.interpolate(self.an_bias, size=kv_size, mode='bilinear').reshape(1, num_heads, self.agent_num, -1).repeat(b, 1, 1, 1)
        position_bias2 = (self.ah_bias + self.aw_bias).reshape(1, num_heads, self.agent_num, -1).repeat(b, 1, 1, 1)
        # print(position_bias1.size(),position_bias2.size())
        position_bias = position_bias1 + position_bias2

        # Agent-to-grid attention
        agent_attn = self.softmax((agent_tokens * self.scale) @ k.transpose(-2, -1) + position_bias)
        agent_attn = self.attn_drop(agent_attn)
        agent_v = agent_attn @ v

        # Position biases for grid-to-agent attention
        agent_bias1 = nn.functional.interpolate(self.na_bias, size=(H, W), mode='bilinear').reshape(1, num_heads, self.agent_num, -1).permute(0, 1, 3, 2).repeat(b, 1, 1, 1)
        agent_bias2 = (self.ha_bias + self.wa_bias).reshape(1, num_heads, -1, self.agent_num).repeat(b, 1, 1, 1)
        agent_bias = agent_bias1 + agent_bias2

        # Grid-to-agent attention
        q_attn = self.softmax((q * self.scale) @ agent_tokens.transpose(-2, -1) + agent_bias)
        q_attn = self.attn_drop(q_attn)
        x = q_attn @ agent_v

        # Reshape output and apply depthwise convolution
        x = x.transpose(1, 2).reshape(b, n, c)
        v = v.transpose(1, 2).reshape(b, H // self.sr_ratio, W // self.sr_ratio, c).permute(0, 3, 1, 2)
        if self.sr_ratio > 1:
            v = nn.functional.interpolate(v, size=(H, W), mode='bilinear')

        x = x + self.dwc(v).permute(0, 2, 3, 1).reshape(b, n, c)

        # Output projection and dropout
        x = self.proj(x)
        x = self.proj_drop(x)
        x = x.view(B, C, H1 ,W1)
        return x

  
