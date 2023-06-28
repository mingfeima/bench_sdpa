import torch
import torch.autograd.profiler as profiler
from time import time


torch.manual_seed(1)

n_head = 25
n_embd = 1600

dropout = 0.5

default = False

def test_single_run(B, T, dtype=torch.float32, train=False, is_causal=False):

    niters = 100

    C = 3 * n_embd

    if train:
        B = B * 4
        niters = int(niters / 5)

    nwarmups = int(niters / 10)

    x = torch.randn(B, T, C)
    if train:
        x.requires_grad_(True)

    q, k, v  = x.split(n_embd, dim=2)
    k = k.view(B, T, n_head, n_embd // n_head).transpose(1, 2) # (B, nh, T, hs)
    q = q.view(B, T, n_head, n_embd // n_head).transpose(1, 2) # (B, nh, T, hs)
    v = v.view(B, T, n_head, n_embd // n_head).transpose(1, 2) # (B, nh, T, hs)

    for _ in range(nwarmups):
        if default:
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=dropout if train else 0, is_causal=is_causal)
        else:
            y, lse, _, _ = torch._scaled_dot_product_efficient_attention(q, k, v, compute_log_sumexp=train, is_causal=is_causal)

    t1 = time()
    for _ in range(niters):
        if default:
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=dropout if train else 0, is_causal=is_causal)
        else:
            y, lse, _, _ = torch._scaled_dot_product_efficient_attention(q, k, v, compute_log_sumexp=train, is_causal=is_causal)

        if train:
            y.sum().backward()
    t2 = time()
    tt = (t2 - t1) / niters * 1000 #ms

    print("is_causal: ", is_causal)
    if train:
        print("### scaled_dot_product_attention: training fwd + bwd: {:.3f} ms per iter".format(tt), "; ", q.size(), q.stride())
    else:
        print("### scaled_dot_product_attention: inference: {:.3f} ms per iter".format(tt), "; ", q.size(), q.stride())


dtype = torch.float32
def test_finetune(train=False):
    test_single_run(1, 1024, dtype, train=train, is_causal=False)
    test_single_run(1, 1024, dtype, train=train, is_causal=True)

def test_sample():
    for t in range(10, 112):
        test_single_run(1, t, dtype, False, False)

def bench(train, profile=False):
    with profiler.profile(enabled=profile) as prof:
        test_finetune(train=train)
        #test_sample()

    if profile:
        print(prof.key_averages().table(sort_by="cpu_time_total"))

profile = False
bench(False, profile=profile)
bench(True, profile=profile)

