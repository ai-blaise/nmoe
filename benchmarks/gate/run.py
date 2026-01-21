import torch
from nmoe.csrc import rdep as _C

WARMUP, ITERS = 50, 200

def bench(n: int):
    dev = torch.device("cuda")
    stream = torch.cuda.current_stream(dev)
    lin = torch.randn(n, device=dev, dtype=torch.bfloat16)
    attn = torch.randn(n, device=dev, dtype=torch.bfloat16)
    out = torch.empty(n, device=dev, dtype=torch.bfloat16)
    gate = torch.empty(n, device=dev, dtype=torch.bfloat16)

    for _ in range(WARMUP):
        _C.g1_gate_fwd(lin.data_ptr(), attn.data_ptr(), out.data_ptr(), gate.data_ptr(), n, stream)
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(ITERS):
        _C.g1_gate_fwd(lin.data_ptr(), attn.data_ptr(), out.data_ptr(), gate.data_ptr(), n, stream)
    e.record()
    torch.cuda.synchronize()
    cust_us = s.elapsed_time(e) * 1000 / ITERS
    cust_out, cust_gate = out.clone(), gate.clone()

    for _ in range(WARMUP):
        g = torch.sigmoid(lin)
        o = attn * g
    torch.cuda.synchronize()
    s.record()
    for _ in range(ITERS):
        g = torch.sigmoid(lin)
        o = attn * g
    e.record()
    torch.cuda.synchronize()
    pt_us = s.elapsed_time(e) * 1000 / ITERS

    nbytes = 8 * n
    cust_gbs, pt_gbs = nbytes / (cust_us * 1e3), nbytes / (pt_us * 1e3)
    maxdiff = max((cust_out - o).abs().max().item(), (cust_gate - g).abs().max().item())
    return cust_us, pt_us, cust_gbs, pt_gbs, maxdiff

if __name__ == "__main__":
    print("| N        | Cust(us) | PT(us)  | Cust GB/s | PT GB/s | MaxDiff   |")
    print("|----------|----------|---------|-----------|---------|-----------|")
    for exp in range(16, 27):
        n = 1 << exp
        c_us, p_us, c_gb, p_gb, md = bench(n)
        print(f"| {n:<8} | {c_us:<8.2f} | {p_us:<7.2f} | {c_gb:<9.1f} | {p_gb:<7.1f} | {md:<9.2e} |")
