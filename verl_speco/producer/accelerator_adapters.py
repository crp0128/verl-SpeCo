# Licensed under the Apache License, Version 2.0. See LICENSE.
"""Consolidated multi-node capabilities. Hardware adapters load only when explicitly selected."""
from __future__ import annotations
import torch
import triton
import triton.language as tl

@triton.jit
def mhc_triton_kernels_split(W, AP, BP, AO, BO, AR, BR, PRE, POST, MIX, TOKENS: tl.constexpr, EPS: tl.constexpr, GROUP: tl.constexpr):
    t = tl.program_id(0) * GROUP + tl.arange(0, GROUP)
    i = tl.arange(0, 4)
    mask = t < TOKENS
    pre = tl.load(W + t[:, None] * 24 + i[None, :], mask=mask[:, None], other=0)
    post = tl.load(W + t[:, None] * 24 + 4 + i[None, :], mask=mask[:, None], other=0)
    ap, ao, ar = (tl.load(AP).to(tl.float32), tl.load(AO).to(tl.float32), tl.load(AR).to(tl.float32))
    bp, bo = (tl.load(BP + i).to(tl.float32), tl.load(BO + i).to(tl.float32))
    pre = tl.sigmoid(pre * ap + bp[None, :]) + EPS
    post = 2 * tl.sigmoid(post * ao + bo[None, :])
    offsets = i[:, None] * 4 + i[None, :]
    mix = tl.load(W + t[:, None, None] * 24 + 8 + offsets[None, :, :], mask=mask[:, None, None], other=0)
    beta = tl.load(BR + offsets).to(tl.float32)
    mix = mix * ar + beta[None, :, :]
    mix = tl.exp(mix - tl.max(mix, axis=2)[:, :, None])
    mix = mix / tl.sum(mix, axis=2)[:, :, None] + EPS
    mix = mix / (tl.sum(mix, axis=1)[:, None, :] + EPS)
    for _ in tl.static_range(19):
        mix = mix / (tl.sum(mix, axis=2)[:, :, None] + EPS)
        mix = mix / (tl.sum(mix, axis=1)[:, None, :] + EPS)
    tl.store(PRE + t[:, None] * 4 + i[None, :], pre, mask=mask[:, None])
    tl.store(POST + t[:, None] * 4 + i[None, :], post, mask=mask[:, None])
    tl.store(MIX + t[:, None, None] * 16 + offsets[None, :, :], mix, mask=mask[:, None, None])

@triton.jit
def mhc_triton_kernels_weighted_pre(X, PRE, Y, TOKENS: tl.constexpr, H: tl.constexpr, GROUP: tl.constexpr, BLOCK: tl.constexpr):
    for tile in range(tl.program_id(0), triton.cdiv(TOKENS, GROUP) * triton.cdiv(H, BLOCK), tl.num_programs(0)):
        t = tile // triton.cdiv(H, BLOCK) * GROUP + tl.arange(0, GROUP)
        h = tile % triton.cdiv(H, BLOCK) * BLOCK + tl.arange(0, BLOCK)
        mask = (t[:, None] < TOKENS) & (h[None, :] < H)
        base = t[:, None] * 4 * H + h[None, :]
        x0 = tl.load(X + base + 0 * H, mask=mask, other=0).to(tl.float32)
        p0 = tl.load(PRE + t * 4 + 0, mask=t < TOKENS, other=0).to(tl.float32)[:, None] * x0
        x1 = tl.load(X + base + 1 * H, mask=mask, other=0).to(tl.float32)
        p1 = tl.load(PRE + t * 4 + 1, mask=t < TOKENS, other=0).to(tl.float32)[:, None] * x1
        x2 = tl.load(X + base + 2 * H, mask=mask, other=0).to(tl.float32)
        p2 = tl.load(PRE + t * 4 + 2, mask=t < TOKENS, other=0).to(tl.float32)[:, None] * x2
        x3 = tl.load(X + base + 3 * H, mask=mask, other=0).to(tl.float32)
        p3 = tl.load(PRE + t * 4 + 3, mask=t < TOKENS, other=0).to(tl.float32)[:, None] * x3
        y = p0 + p1 + p2 + p3
        tl.store(Y + t[:, None] * H + h[None, :], y, mask=(t[:, None] < TOKENS) & (h[None, :] < H))

@triton.jit
def mhc_triton_kernels_post(X, R, POST, MIX, Y, TOKENS: tl.constexpr, H: tl.constexpr, GROUP: tl.constexpr, BLOCK: tl.constexpr):
    for tile in range(tl.program_id(0), triton.cdiv(TOKENS, GROUP) * triton.cdiv(H, BLOCK), tl.num_programs(0)):
        t = tile // triton.cdiv(H, BLOCK) * GROUP + tl.arange(0, GROUP)
        h = tile % triton.cdiv(H, BLOCK) * BLOCK + tl.arange(0, BLOCK)
        mask = (t[:, None] < TOKENS) & (h[None, :] < H)
        base = t[:, None] * 4 * H + h[None, :]
        x = tl.load(X + t[:, None] * H + h[None, :], mask=mask, other=0).to(tl.float32)
        r0 = tl.load(R + base + 0 * H, mask=mask, other=0).to(tl.float32)
        r1 = tl.load(R + base + 1 * H, mask=mask, other=0).to(tl.float32)
        r2 = tl.load(R + base + 2 * H, mask=mask, other=0).to(tl.float32)
        r3 = tl.load(R + base + 3 * H, mask=mask, other=0).to(tl.float32)
        for j in tl.static_range(4):
            p0 = tl.load(MIX + t * 16 + 0 * 4 + j, mask=t < TOKENS, other=0).to(tl.float32)[:, None] * r0
            p1 = tl.load(MIX + t * 16 + 1 * 4 + j, mask=t < TOKENS, other=0).to(tl.float32)[:, None] * r1
            p2 = tl.load(MIX + t * 16 + 2 * 4 + j, mask=t < TOKENS, other=0).to(tl.float32)[:, None] * r2
            p3 = tl.load(MIX + t * 16 + 3 * 4 + j, mask=t < TOKENS, other=0).to(tl.float32)[:, None] * r3
            weighted = p0 + p1 + p2 + p3
            post = tl.load(POST + t * 4 + j, mask=t < TOKENS, other=0).to(tl.float32)
            y = post[:, None] * x + weighted
            tl.store(Y + base + j * H, y, mask=mask)

def mhc_triton_kernels_check(tensors):
    if any((t.device.type != 'npu' or not t.is_contiguous() for t in tensors)):
        raise ValueError('Ascend contiguous tensors required')

def split_sinkhorn(weight, parameters):
    mhc_triton_kernels_check([weight, *parameters])
    if weight.ndim != 2 or weight.shape[1] != 24 or weight.dtype != torch.float32:
        raise ValueError('FP32 four-stream mixes required')
    pre = weight.new_empty((weight.shape[0], 4))
    post = torch.empty_like(pre)
    mix = weight.new_empty((weight.shape[0], 4, 4))
    mhc_triton_kernels_split[triton.cdiv(weight.shape[0], 8),](weight, *parameters, pre, post, mix, weight.shape[0], 1e-06, 8, enable_fp_fusion=False)
    return (pre, post, mix)

def weighted_pre(x, pre, *, dtype):
    mhc_triton_kernels_check([x, pre])
    if x.ndim != 2 or x.shape[1] % 4 or pre.shape != (x.shape[0], 4) or (pre.dtype != torch.float32):
        raise ValueError('exact four-stream FP32 pre weights required')
    tokens, hidden = (x.shape[0], x.shape[1] // 4)
    y = torch.empty((tokens, hidden), device=x.device, dtype=dtype)
    mhc_triton_kernels_weighted_pre[min(4096, triton.cdiv(tokens, 4) * triton.cdiv(hidden, 256)),](x, pre, y, tokens, hidden, 4, 256, enable_fp_fusion=False)
    return y

def post(x, residual, h_post, h_res):
    mhc_triton_kernels_check([x, residual, h_post, h_res])
    if x.ndim != 2 or residual.shape != (x.shape[0], 4 * x.shape[1]) or h_post.shape != (x.shape[0], 4) or (h_res.shape != (x.shape[0], 4, 4)) or (h_post.dtype != torch.float32) or (h_res.dtype != torch.float32):
        raise ValueError('exact FP32 four-stream post shapes required')
    y = torch.empty_like(residual, dtype=x.dtype)
    mhc_triton_kernels_post[min(4096, triton.cdiv(x.shape[0], 4) * triton.cdiv(x.shape[1], 256)),](x, residual, h_post, h_res, y, x.shape[0], x.shape[1], 4, 256, enable_fp_fusion=False)
    return y
