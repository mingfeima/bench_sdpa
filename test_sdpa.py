import torch
import torch.autograd.profiler as profiler
from time import time


torch.manual_seed(1)

def cmp(t1, t2, msg='', atol=1e-05, rtol=1e-07):

    if t1.dtype == torch.bfloat16:
        atol = 1e-2
        rtol = 1e-2
    max_diff = (t1 - t2).abs().max().item()
    res = torch.allclose(t1, t2, atol=atol, rtol=rtol)
    print(msg, res, "; max diff: ", max_diff)

def test1(B, T, dtype=torch.float32, train=False):

    n_embd = 12
    n_head = 1

    C = 3 * n_embd

    x = torch.randn(B, T, C)
    if train:
        x.requires_grad_(True)

    q, k, v  = x.split(n_embd, dim=2)
    k = k.view(B, T, n_head, n_embd // n_head).transpose(1, 2) # (B, nh, T, hs)
    q = q.view(B, T, n_head, n_embd // n_head).transpose(1, 2) # (B, nh, T, hs)
    v = v.view(B, T, n_head, n_embd // n_head).transpose(1, 2) # (B, nh, T, hs)

    k2 = k.contiguous().view(B, T, n_embd)
    q2 = q.contiguous().view(B, T, n_embd)
    v2 = v.contiguous().view(B, T, n_embd)

    is_causal = False
    out = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0, is_causal=is_causal)
    out2, lse2 = torch._scaled_dot_product_efficient_attention(q2, k2, v2, compute_log_sumexp=False, is_causal=is_causal)

    cmp(out, out2, "\n## attn: ")

    #print(out)
    #print(out2)

#test1(1, 22)

def test2(B, T, dtype=torch.float32, causal=False, train=False):

    print("\nTesting efficient attention: ", dtype, "; causal: ", causal, "; train: ", train)
    n_embd = 825
    n_head = 25
    C = 3 * n_embd

    x = torch.randn(B, T, C, dtype=dtype)
    x2 = x.clone()

    if train:
        x.requires_grad_(True)
        x2.requires_grad_(True)

    q, k, v  = x.split(n_embd, dim=2)
    q2, k2, v2  = x2.split(n_embd, dim=2)

    if dtype is torch.bfloat16:
        q = q.float()
        k = k.float()
        v = v.float()

    k = k.view(B, T, n_head, n_embd // n_head).transpose(1, 2) # (B, nh, T, hs)
    q = q.view(B, T, n_head, n_embd // n_head).transpose(1, 2) # (B, nh, T, hs)
    v = v.view(B, T, n_head, n_embd // n_head).transpose(1, 2) # (B, nh, T, hs)
    k2 = k2.view(B, T, n_head, n_embd // n_head).transpose(1, 2) # (B, nh, T, hs)
    q2 = q2.view(B, T, n_head, n_embd // n_head).transpose(1, 2) # (B, nh, T, hs)
    v2 = v2.view(B, T, n_head, n_embd // n_head).transpose(1, 2) # (B, nh, T, hs)

    out = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0, is_causal=causal)
    out2, lse2, _, _ = torch._scaled_dot_product_efficient_attention(q2, k2, v2, compute_log_sumexp=train, is_causal=causal)

    if dtype is torch.bfloat16:
        out = out.bfloat16()

    cmp(out, out2, "## attn: ")

    if train:
        out.sum().backward()
        out2.sum().backward()

        grad_x, grad_x2 = x.grad, x2.grad
        grad_q, grad_k, grad_v = grad_x.split(n_embd, dim=2)
        grad_q2, grad_k2, grad_v2 = grad_x2.split(n_embd, dim=2)

        cmp(grad_q, grad_q2, "## grad_q: ", 5e-6)
        # the error is bigger for grad_k, it need accumulation on N
        cmp(grad_k, grad_k2, "## grad_k: ", 5e-6, 5e-6)
        cmp(grad_v, grad_v2, "## grad_v: ", 5e-6)


# inference test
test2(2, 1030, torch.float32, False, False)
test2(2, 1079, torch.float32, True, False)

# training test
test2(2, 276, torch.float32, False, True)
test2(2, 276, torch.float32, True, True)

test2(2, 1030, torch.bfloat16, False, False)
test2(2, 200, torch.bfloat16, True, False)
test2(2, 1030, torch.bfloat16, False, True)
test2(2, 20, torch.bfloat16, True, True)



