import torch
import torch.nn as nn
from einops import rearrange, repeat
import sys
import os
from transformers import BertModel, RobertaModel, AutoConfig, AutoTokenizer, DebertaV2Model
sys.path.append(os.path.abspath(os.path.join(__file__, "..", "..")))
from utils.utils import *

class SelfAttention(nn.Module):
    def __init__(self, dim: int, dim_head: int, heads: int):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        inner_dim = dim_head * heads

        self.layer_norm = nn.LayerNorm(dim)
        self.to_q = nn.Linear(dim, inner_dim)
        self.to_k = nn.Linear(dim, inner_dim)
        self.to_v = nn.Linear(dim, inner_dim)
        self.to_out = nn.Linear(inner_dim, dim)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): shape (b, n, d)
        """
        residual_x  = x
        x = self.layer_norm(x) # [b, n1, D]
        h = self.heads
        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)

        q = rearrange(q, "b n (h d) -> b h n d", h=h)
        k = rearrange(k, "b n (h d) -> b h n d", h=h)
        v = rearrange(v, "b n (h d) -> b h n d", h=h)
        q = q * self.scale

        # attention
        sim = torch.einsum("... i d, ... j d  -> ... i j", q, k)
        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        attn = sim.softmax(dim=-1)

        out = torch.einsum("... i j, ... j d -> ... i d", attn, v)
        out = rearrange(out, "b h n d -> b n (h d)", h=h)
        out = self.to_out(out) + residual_x

        return out

class QueryCrossAttention(nn.Module):
    def __init__(self, dim: int, dim_head: int, heads: int):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        inner_dim = dim_head * heads

        self.layer_norm_query = nn.LayerNorm(dim)
        self.layer_norm_kv = nn.LayerNorm(dim)
        self.to_q = nn.Linear(dim, inner_dim)
        self.to_k = nn.Linear(dim, inner_dim)
        self.to_v = nn.Linear(dim, inner_dim)
        self.to_out = nn.Linear(inner_dim, dim)


    def forward(self, query, kv: torch.Tensor) -> torch.Tensor:
        """
        Args:
            query, kv (torch.Tensor): shape (b, n, d)
        """
        kv = self.layer_norm_kv(kv)
        residual_query  = query
        query = self.layer_norm_query(query) # [b, n, D]
        h = self.heads
        q = self.to_q(query)
        kv_input = torch.cat((kv, query), dim=-2)

        k = self.to_k(kv_input)
        v = self.to_v(kv_input)

        q = rearrange(q, "b n (h d) -> b h n d", h=h)
        k = rearrange(k, "b n (h d) -> b h n d", h=h)
        v = rearrange(v, "b n (h d) -> b h n d", h=h)
        q = q * self.scale

        # attention
        sim = torch.einsum("... i d, ... j d  -> ... i j", q, k)
        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        attn = sim.softmax(dim=-1)

        out = torch.einsum("... i j, ... j d -> ... i d", attn, v)
        out = rearrange(out, "b h n d -> b n (h d)", h=h)
        out = self.to_out(out) + residual_query

        return out
    
class FFN(nn.Module):
    def __init__(self, dim: int, mult: int = 4):
        super().__init__()

        self.layer_norm = nn.LayerNorm(dim)
        self.dense1 = nn.Linear(dim, dim*mult, bias=True)
        self.gelu = nn.GELU()
        self.dense2 = nn.Linear(dim*mult, dim, bias=True)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): shape (b, n, D)
        """
        residual_x  = x
        x = self.layer_norm(x)
        x = self.dense1(x)
        x = self.gelu(x)
        x = self.dense2(x)
        
        return x + residual_x
# 增强版双向交互+门控模块（去掉了 SEBlock）
class BidirectionalCrossAttention(nn.Module):
    def __init__(self, dim: int, dim_head: int, heads: int):
        """
        在原有双向交互基础上：
         - 使用两层非线性门控 MLP 替代单层线性门控；
         - 增加额外的 FFN 进一步整合信息；
         - 不使用 SEBlock。
        """
        super().__init__()
        self.forward_cross = QueryCrossAttention(dim, dim_head, heads)
        self.reverse_cross = QueryCrossAttention(dim, dim_head, heads)
        self.gate_mlp = nn.Sequential(
            nn.Linear(2 * dim, 2 * dim),
            nn.ReLU(inplace=True),
            nn.Linear(2 * dim, dim),
            nn.Sigmoid()
        )
        self.fusion_linear = nn.Linear(2 * dim, dim)
        self.norm = nn.LayerNorm(dim)
        self.ffn = FFN(dim, mult=2)

    def forward(self, query, kv: torch.Tensor) -> torch.Tensor:
        # 正向交互：query tokens 关注图像信息
        forward_out = self.forward_cross(query, kv)  # (b, n, dim)
        # 反向交互：让图像关注 query tokens
        reverse_out = self.reverse_cross(kv, query)    # (b, m, dim)
        reverse_global = reverse_out.mean(dim=1, keepdim=True)  # (b, 1, dim)
        reverse_expanded = reverse_global.expand(query.size(0), query.size(1), query.size(2))  # (b, n, dim)
        fusion_input = torch.cat([forward_out, reverse_expanded], dim=-1)  # (b, n, 2*dim)
        gate = self.gate_mlp(fusion_input)  # (b, n, dim)
        gated_out = gate * forward_out + (1 - gate) * reverse_expanded
        fused = self.fusion_linear(torch.cat([gated_out, forward_out], dim=-1))
        fused = self.norm(fused + query)  # 残差连接
        fused = self.ffn(fused)
        return fused

# class BidirectionalCrossAttention(nn.Module):
#     def __init__(self, dim: int, dim_head: int, heads: int):
#         """
#         实现双向交互+门控融合：
#           - 正向分支：query tokens 利用图像信息更新自身（采用 QueryCrossAttention）。
#           - 反向分支：图像信息关注 query tokens，并取全局平均。
#           - 通过门控机制对两路信息进行动态加权融合。
#         """
#         super().__init__()
#         self.forward_cross = QueryCrossAttention(dim, dim_head, heads)
#         self.reverse_cross = QueryCrossAttention(dim, dim_head, heads)
#         # 用于计算门控权重，输入拼接正向与反向信息，输出与 dim 相同
#         self.gate_layer = nn.Linear(2 * dim, dim)
#         self.sigmoid = nn.Sigmoid()
#         # 融合层，将门控融合结果和正向结果进一步结合
#         self.fusion_linear = nn.Linear(2 * dim, dim)
#         self.norm = nn.LayerNorm(dim)

#     def forward(self, query, kv: torch.Tensor) -> torch.Tensor:
#         """
#         Args:
#             query: Tensor of shape (b, n, dim)，query tokens
#             kv: Tensor of shape (b, m, dim)，图像特征
#         Returns:
#             fused: Tensor of shape (b, n, dim)，经过双向交互和门控融合后的 query tokens
#         """
#         # 正向交互：query tokens 关注图像信息
#         forward_out = self.forward_cross(query, kv)  # (b, n, dim)
#         # 反向交互：让图像信息关注 query tokens
#         reverse_out = self.reverse_cross(kv, query)    # (b, m, dim)
#         # 对反向输出进行全局平均，得到 (b, 1, dim)
#         reverse_global = reverse_out.mean(dim=1, keepdim=True)
#         # 将全局反向信息扩展到每个 query token上，形状 (b, n, dim)
#         reverse_expanded = reverse_global.expand(query.size(0), query.size(1), query.size(2))
#         # 拼接正向和反向信息，计算门控权重
#         fusion_input = torch.cat([forward_out, reverse_expanded], dim=-1)  # (b, n, 2*dim)
#         gate = self.sigmoid(self.gate_layer(fusion_input))  # (b, n, dim)，取值范围 (0, 1)
#         # 融合两路信息：门控机制控制正向信息与反向信息的比重
#         gated_out = gate * forward_out + (1 - gate) * reverse_expanded
#         # 可选：进一步将 gated_out 与 forward_out 结合，得到最终输出
#         fused = self.fusion_linear(torch.cat([gated_out, forward_out], dim=-1))
#         fused = self.norm(fused + query)  # 残差连接并归一化
#         return fused

# class BidirectionalCrossAttention(nn.Module):
#     def __init__(self, dim: int, dim_head: int, heads: int):
#         """
#         实现双向交互：
#           - 正向：让 query tokens 关注 kv（图像）信息，类似 QueryCrossAttention。
#           - 反向：让 kv（图像）关注 query tokens的信息，并取全局平均。
#           - 最后融合两路信息更新 query tokens。
#         """
#         super().__init__()
#         self.forward_cross = QueryCrossAttention(dim, dim_head, heads)
#         self.reverse_cross = QueryCrossAttention(dim, dim_head, heads)
#         self.fusion = nn.Linear(2 * dim, dim)
#         self.norm = nn.LayerNorm(dim)

#     def forward(self, query, kv: torch.Tensor) -> torch.Tensor:
#         # 正向交互：query tokens 关注图像信息
#         forward_out = self.forward_cross(query, kv)  # (b, n, dim)
#         # 反向交互：让图像信息关注 query tokens
#         reverse_out = self.reverse_cross(kv, query)  # 输出 shape 为 (b, m, dim)
#         # 对反向输出做全局平均，得到 (b, 1, dim)
#         reverse_global = reverse_out.mean(dim=1, keepdim=True)
#         # 扩展为 (b, n, dim)
#         reverse_expanded = reverse_global.expand(query.shape[0], query.shape[1], query.shape[2])
#         # 融合两路信息
#         fusion_input = torch.cat([forward_out, reverse_expanded], dim=-1)
#         fused = self.fusion(fusion_input)
#         fused = self.norm(fused + query)  # 残差连接
#         return fused

class SamplerBlock(nn.Module):
    def __init__(self, dim: int, dim_head: int, heads: int, mult: int = 4):
        super().__init__()
        self.self_attn = SelfAttention(dim=dim, dim_head=dim_head, heads=heads)
        self.cross_attn = BidirectionalCrossAttention(dim=dim, dim_head=dim_head, heads=heads)
        # self.cross_attn = QueryCrossAttention(dim=dim, dim_head=dim_head, heads=heads)
        self.ffn = FFN(dim=dim, mult=mult)

    def forward(self, query_tokens, kv_input: torch.Tensor) -> torch.Tensor:
        """
        Args:
            query, kv (torch.Tensor): shape (b, n, d)
        """
        query_tokens = self.self_attn(query_tokens)
        query_tokens = self.cross_attn(query_tokens, kv_input)
        query_tokens = self.ffn(query_tokens)

        return query_tokens

class SamplerFormer(nn.Module):
    def __init__(
            self, 
            depth: int = 2, 
            dim: int = 768, 
            dim_head: int = int(768/12), 
            heads: int = 12, 
            mult: int = 4
    ):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                SamplerBlock(
                    dim=dim, dim_head=dim_head, heads=heads, mult=mult
                )
            )
        self.norm = nn.LayerNorm(dim)
        self.latents = nn.Parameter(torch.load("./experiments/query_tokens_vicuna.pth", map_location='cpu')[0]) # [num_latents, 768]
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): image features
                shape (b, n, D)
        Returns:
            shape (b, n, D) where n is self.num_latents
        """
        b, n, d = x.shape
        # blocks
        latents = repeat(self.latents, "n d -> b n d", b=b)
        for block in self.layers:
            latents = block(query_tokens=latents, kv_input=x)
        return self.norm(latents) 
    
class QAPrompting(nn.Module):
    def __init__(self, image_dim=1408):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained('microsoft/deberta-v3-base', use_fast=False)
        self.text_encoder = DebertaV2Model.from_pretrained('microsoft/deberta-v3-base')

        self.image_dense = nn.Linear(image_dim, 768)
        self.mha_layer = torch.nn.MultiheadAttention(embed_dim=768, kdim=768, vdim=768, num_heads=1, batch_first=True)
        self.gate_dense = nn.Linear(2*768, 768)
        self.sigmoid = nn.Sigmoid()

        self.decoder = SamplerFormer(depth=2, dim=768, dim_head= int(768/4), heads=4, mult=4)
        self.llm_proj = nn.Linear(768, 4096)
        self.llm_proj.load_state_dict(torch.load("./experiments/llm_proj_vicuna.pth", map_location='cpu'))

        for name, param in self.text_encoder.named_parameters():
            param.requires_grad = False
        self.text_encoder.eval()

    def encode_text(self, texts):
        text_inputs = self.tokenizer(texts, padding='longest', truncation=True, max_length=512, return_tensors="pt").to(self.device)
        text_outputs = self.text_encoder(input_ids=text_inputs.input_ids, attention_mask=text_inputs.attention_mask)
        word_text_embeds = text_outputs.last_hidden_state # [bs, seq_len, text_hidden_size]
        return word_text_embeds

    @property
    def device(self):
        return list(self.parameters())[0].device
    
    def forward(self, image_embeds, inputs_txt):
        hidden_states = self.encode_text(inputs_txt) # [bs, seq_len, 768]

        image_embedding = self.image_dense(image_embeds) # [bs, 257, 768]

        image_att, _ = self.mha_layer(hidden_states, image_embedding, image_embedding) # [bs, seq_len, 768]
        merge = torch.cat([hidden_states, image_att], dim=-1) # [bs, seq_len, 768*2]
        gate = self.sigmoid(self.gate_dense(merge)) # [bs, seq_len, 768]
        hidden_states = (1 - gate) * hidden_states + gate * image_att # [bs, seq_len, 768]

        query_tokens = self.decoder(hidden_states) # [bs, 32, 768]
        query_tokens = self.llm_proj(query_tokens) # [bs, 32, 4096]
        return query_tokens
    
# model = QAPrompting(1024)
# image_embeds = torch.randn(2, 576, 1024)
# inputs_txt = ['Question: What is the man wearing on his shoulders? Short answer: jacket', 'Question: What type of clothing is this man wearing? Short answer: wetsuit']
# query_tokens = model(image_embeds, inputs_txt)
# print(query_tokens.shape)