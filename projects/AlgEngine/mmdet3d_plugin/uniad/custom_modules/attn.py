from functools import partial

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn, einsum
from torch.utils.checkpoint import checkpoint

from mmdet3d_plugin.uniad.custom_modules.peft import (LoRALinear, ZeroAdapter, LoRACLAdapter, LoRAMoECLAdapter, MOELoRALinear,
    finetuning_detach, frozen_grad, peft_wrapper_forward, lora_wrapper)

# helper functions

def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


# regular attention

def attention(
        q, k, v,
        mask=None,
        causal=False,
        attn_bias=None,
        **kwargs
):
    scale = q.shape[-1] ** -0.5
    q = q * scale

    sim = einsum('b h i d, b h j d -> b h i j', q, k)

    if exists(attn_bias):
        sim = sim + attn_bias

    mask_value = -torch.finfo(sim.dtype).max

    if exists(mask):
        if mask.ndim == 2:
            mask = rearrange(mask, 'b j -> b 1 1 j')
        sim = sim.masked_fill(~mask, mask_value)

    if causal:
        i, j = sim.shape[-2:]
        mask = torch.ones(i, j, device=q.device, dtype=torch.bool).triu(j - i + 1)
        sim = sim.masked_fill(mask, mask_value)

    sim = sim - sim.amax(dim=-1, keepdim=True).detach()
    attn = sim.softmax(dim=-1)

    out = einsum('b h i j, b h j d -> b h i d', attn, v)
    return out


# memory efficient attention

def summarize_qkv_chunk(q, k, v, mask, attn_bias_chunk, causal, qk_start_indices, dropout):
    q_start_index, k_start_index, q_chunk_size, k_chunk_size, device = *qk_start_indices, q.shape[-2], k.shape[
        -2], q.device

    weight = einsum('b h i d, b h j d -> b h i j', q, k)

    if exists(attn_bias_chunk):
        weight = weight + attn_bias_chunk

    mask_value = -torch.finfo(weight.dtype).max

    if exists(mask):
        mask = rearrange(mask, 'b j -> b 1 1 j')
        weight = weight.masked_fill(~mask, mask_value)

    if causal and q_start_index < (k_start_index + k_chunk_size - 1):
        causal_mask = torch.ones((q_chunk_size, k_chunk_size), dtype=torch.bool, device=device).triu(
            q_start_index - k_start_index + 1)
        weight = weight.masked_fill(causal_mask, mask_value)

    weight_max = weight.amax(dim=-1, keepdim=True).detach()
    weight = weight - weight_max

    exp_weight = weight.exp()

    exp_weight = F.dropout(exp_weight, p=dropout)

    weighted_value = einsum('b h i j, b h j d -> b h i d', exp_weight, v)

    return exp_weight.sum(dim=-1), weighted_value, rearrange(weight_max, '... 1 -> ...')


checkpointed_summarize_qkv_chunk = partial(checkpoint, summarize_qkv_chunk)


def memory_efficient_attention(
        q, k, v,
        mask=None,
        causal=False,
        attn_bias=None,
        q_bucket_size=512,
        k_bucket_size=1024,
        eps=1e-8,
        dropout=0.,
        training=False
):
    scale = q.shape[-1] ** -0.5
    q = q * scale

    # function

    needs_backwards = q.requires_grad or k.requires_grad or v.requires_grad
    summarize_qkv_fn = checkpointed_summarize_qkv_chunk if needs_backwards else summarize_qkv_chunk

    # chunk all the inputs

    q_chunks = q.split(q_bucket_size, dim=-2)
    k_chunks = k.split(k_bucket_size, dim=-2)
    v_chunks = v.split(k_bucket_size, dim=-2)
    mask_chunks = mask.split(k_bucket_size, dim=-1) if exists(mask) else ((None,) * len(k_chunks))

    if exists(attn_bias):
        i, j = attn_bias.shape[-2:]
        attn_bias_chunks = attn_bias.split(q_bucket_size, dim=-2)
        attn_bias_chunks = list(map(lambda t: t.split(k_bucket_size, dim=-1), attn_bias_chunks))

    # loop through all chunks and accumulate

    out = []
    for q_index, q_chunk in enumerate(q_chunks):
        exp_weights = []
        weighted_values = []
        weight_maxes = []

        for k_index, (k_chunk, v_chunk, mask_chunk) in enumerate(zip(k_chunks, v_chunks, mask_chunks)):
            q_start_index = q_index * q_bucket_size
            k_start_index = k_index * k_bucket_size

            if causal and k_start_index > (q_start_index + q_chunk.shape[-2] - 1):
                # if chunk is to be all masked out causally, skip
                continue

            attn_bias_chunk = attn_bias_chunks[q_index][k_index] if exists(attn_bias) else None

            exp_weight_chunk, weighted_value_chunk, weight_max_chunk = summarize_qkv_fn(
                q_chunk,
                k_chunk,
                v_chunk,
                mask_chunk,
                attn_bias_chunk,
                causal,
                (q_start_index, k_start_index),
                dropout if training else 0.
            )

            exp_weights.append(exp_weight_chunk)
            weighted_values.append(weighted_value_chunk)
            weight_maxes.append(weight_max_chunk)

        weight_maxes = torch.stack(weight_maxes, dim=-1)

        weighted_values = torch.stack(weighted_values, dim=-1)
        exp_weights = torch.stack(exp_weights, dim=-1)

        global_max = weight_maxes.amax(dim=-1, keepdim=True)
        renorm_factor = (weight_maxes - global_max).exp().detach()

        exp_weights = exp_weights * renorm_factor
        weighted_values = weighted_values * rearrange(renorm_factor, '... c -> ... 1 c')

        all_values = weighted_values.sum(dim=-1)
        all_weights = exp_weights.sum(dim=-1)

        normalized_values = all_values / (rearrange(all_weights, '... -> ... 1') + eps)
        out.append(normalized_values)

    return torch.cat(out, dim=-2)


# main class

class Attention(nn.Module):
    def __init__(
            self,
            *,
            dim,
            heads=8,
            dim_head=64,
            dropout=0.,
            causal=False,
            memory_efficient=False,
            q_bucket_size=512,
            k_bucket_size=1024,
            use_lora=False,
            lora_rank=16,
            lora_drop=0.,
            moe_lora=False,
            num_task=6,
    ):
        super().__init__()
        self.heads = heads
        self.causal = causal
        self.dropout = dropout
        self.use_lora = use_lora
        inner_dim = heads * dim_head

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_k = nn.Linear(dim, inner_dim, bias=False)
        self.to_v = nn.Linear(dim, inner_dim, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

        if self.use_lora:
            lora_layer = MOELoRALinear if moe_lora else LoRALinear
            self.lora_to_q = lora_layer(dim, inner_dim, r=lora_rank, dropout=lora_drop, num_task=num_task)
            self.lora_to_k = lora_layer(dim, inner_dim, r=lora_rank, dropout=lora_drop, num_task=num_task)
            self.lora_to_v = lora_layer(dim, inner_dim, r=lora_rank, dropout=lora_drop, num_task=num_task)
            self.lora_to_out = lora_layer(inner_dim, dim, r=lora_rank, dropout=lora_drop, num_task=num_task)
            finetuning_detach(self)

        # memory efficient attention related parameters
        # can be overriden on forward
        self.memory_efficient = memory_efficient
        self.q_bucket_size = q_bucket_size
        self.k_bucket_size = k_bucket_size

    def forward(
            self,
            q, k, v,
            mask=None,
            attn_bias=None,
            memory_efficient=None,
            q_bucket_size=None,
            k_bucket_size=None,
            forward_origin=False,
            task_idx=None,
    ):
        memory_efficient = default(memory_efficient, self.memory_efficient)
        q_bucket_size = default(q_bucket_size, self.q_bucket_size)
        k_bucket_size = default(k_bucket_size, self.k_bucket_size)

        h = self.heads

        if forward_origin:
            q = self.to_q(q)
            k = self.to_k(k)
            v = self.to_v(v)
        else:
            if self.use_lora:
                q = self.to_q(q) + self.lora_to_q(q, i=task_idx)
                k = self.to_k(k) + self.lora_to_k(k, i=task_idx)
                v = self.to_v(v) + self.lora_to_v(v, i=task_idx)
            else:
                q = self.to_q(q)
                k = self.to_k(k)
                v = self.to_v(v)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), (q, k, v))

        attn_fn = attention if not memory_efficient else memory_efficient_attention

        out = attn_fn(q, k, v, mask=mask, attn_bias=attn_bias, causal=self.causal, q_bucket_size=q_bucket_size,
                      k_bucket_size=k_bucket_size, dropout=self.dropout, training=self.training)

        out = rearrange(out, 'b h n d -> b n (h d)')
        if forward_origin:
            return self.to_out(out)
        if self.use_lora:
           return self.to_out(out) + self.lora_to_out(out, i=task_idx) 
        return self.to_out(out)


class MemoryEffTransformer(nn.Module):
    def __init__(self,
        d_model,
        nhead,
        dim_feedforward=2048,
        dropout=0.1,
        activation=F.relu,
        layer_norm_eps=1e-5,
        use_lora=False,
        attn_use_lora=False,
        lora_rank=16,
        attn_lora_rank=16,
        moe_lora=False,
        num_task=6,
        ):
        super().__init__()
        dim_head = d_model // nhead
        self.self_attn = Attention(dim=d_model, 
                                   heads=nhead, 
                                   dim_head=dim_head, 
                                   memory_efficient=True,
                                   use_lora=attn_use_lora,
                                   lora_rank=attn_lora_rank,
                                   moe_lora=moe_lora,
                                   num_task=num_task)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm3 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.use_lora = use_lora
        if self.use_lora:
            lora_layer = MOELoRALinear if moe_lora else LoRALinear
            self.lora_linear1 = lora_layer(d_model, dim_feedforward, r=lora_rank, dropout=0., num_task=num_task)
            self.lora_linear2 = lora_layer(dim_feedforward, d_model,  r=lora_rank, dropout=0., num_task=num_task)

        self.activation = activation
    
    def forward_origin(self, x):
        tmp = self.self_attn(x, x, x, forward_origin=True)
        x = self.norm1(x + self.dropout1(tmp))
        x1 = self.linear1(x)
        x1 = self.dropout(self.activation(x1))
        tmp = self.linear2(x1)
        x = self.norm3(x + self.dropout3(tmp))
        return x

    def forward(self, x, forward_origin=False, task_idx=None):
        if forward_origin:
            return self.forward_origin(x)
        tmp = self.self_attn(x, x, x,task_idx=task_idx)
        x = self.norm1(x + self.dropout1(tmp))
        if self.use_lora:
            x1 = self.linear1(x) + self.lora_linear1(x,i=task_idx)
        else:
            x1 = self.linear1(x)
        x1 = self.dropout(self.activation(x1))
        if self.use_lora:
            tmp = self.linear2(x1) + self.lora_linear2(x1,i=task_idx)
        else:
            tmp = self.linear2(x1)
        x = self.norm3(x + self.dropout3(tmp))

        return x