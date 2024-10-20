import pickle
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import csv
import gc
from einops import rearrange, repeat
from flash_attn.utils.benchmark import benchmark_all, benchmark_forward, benchmark_backward
from flash_attn.utils.benchmark import benchmark_fwd_bwd, benchmark_combined
from flash_attn import flash_attn_qkvpacked_func

try:
    from triton.ops.flash_attention import attention as attention_triton
except ImportError:
    attention_triton = None

try:
    import xformers.ops as xops
except ImportError:
    xops = None

def flops(batch, seqlen, headdim, nheads, causal, mode="fwd"):
    assert mode in ["fwd", "bwd", "fwd_bwd"]
    f = 4 * batch * seqlen**2 * nheads * headdim // (2 if causal else 1)
    return f if mode == "fwd" else (2.5 * f if mode == "bwd" else 3.5 * f)

def efficiency(flop, time):
    return (flop / time / 10**12) if not math.isnan(time) else 0.0

def attention_pytorch(qkv, dropout_p=0.0, causal=True):
    batch_size, seqlen, _, nheads, d = qkv.shape
    q, k, v = qkv.unbind(dim=2)
    q = rearrange(q, 'b t h d -> (b h) t d')
    k = rearrange(k, 'b s h d -> (b h) d s')
    softmax_scale = 1.0 / math.sqrt(d)
    scores = torch.empty(batch_size * nheads, seqlen, seqlen, dtype=qkv.dtype, device=qkv.device)
    scores = rearrange(torch.baddbmm(scores, q, k, beta=0, alpha=softmax_scale),
                       '(b h) t s -> b h t s', h=nheads)
    if causal:
        causal_mask = torch.triu(torch.full((seqlen, seqlen), -10000.0, device=scores.device), 1)
        scores = scores + causal_mask.to(dtype=scores.dtype)
    attention = torch.softmax(scores, dim=-1)
    attention_drop = F.dropout(attention, dropout_p)
    output = torch.einsum('bhts,bshd->bthd', attention_drop , v)
    return output.to(dtype=qkv.dtype)

def time_fwd_bwd(func, *args, **kwargs):
    time_f, time_b = benchmark_fwd_bwd(func, *args, **kwargs)
    return time_f[1].mean, time_b[1].mean

def ceil_div(a, b):
    return (a + b - 1) // b

repeats = 30
device = 'cuda'
dtype = torch.float16

bs_seqlen_vals = [(32, 512), (16, 1024), (8, 2048), (4, 4096), (2, 8192), (1, 16384)]
# bs_seqlen_vals = [(32, 512)]
causal_vals = [False, True]
# causal_vals = [False]
headdim_vals = [32, 64, 96, 128, 160, 192, 224, 256]
# headdim_vals = [64]
hidden_dim = [1024, 2048, 4096]
# hidden_dim = [1024]
dropout = [0.0, 0.2, 0.4]
# dropout = [0.0]

# Define the headers for the CSV file
headers = [
    "dropout", "causal", "hidden_dim", "headdim", "num_heads", "batch_size", "seqlen",
    "kBlockN", "num_n_blocks", "kBlockM", "num_m_blocks", "Layout_tile_multiply", "Nwarps",
    "gQ", "gK", "gV", "gP",
    "sQ", "sK", "sV", "sVt", "tSrQ", "tSrK", "tOrVt", "tSgS", "acc_o",
    "method", "fwd_TFLOPs/s", "bwd_TFLOPs/s", "fwd_bwd_TFLOPs/s",
    "mem_allocated_before", "mem_reserved_before", "mem_allocated_after", "mem_reserved_after"
]

methods = (["Flash2", "Pytorch"]
           + (["Triton"] if attention_triton is not None else [])
           + (["xformers.c"] if xops is not None else [])
           + (["xformers.f"] if xops is not None else []))

time_f = {}
time_b = {}
time_f_b = {}
speed_f = {}
speed_b = {}
speed_f_b = {}

kBlockM = 128
kBlockN = 64
Nwarps = 4
Layout_tile_multiply = 8


# Open the CSV file for writing
with open(f'attn_time_{kBlockM}_{kBlockN}_{Nwarps}_{Layout_tile_multiply}.csv', 'w', newline='') as csvfile:
    csvwriter = csv.writer(csvfile)
    # Write the headers
    csvwriter.writerow(headers)

    for causal in causal_vals:
        for dropout_p in dropout:
            for dim in hidden_dim:
                for headdim in headdim_vals:
                    for batch_size, seqlen in bs_seqlen_vals:
                        print("Memory allocated before:", torch.cuda.memory_allocated())
                        print("Memory reserved before:", torch.cuda.memory_reserved())


                        num_m_blocks = seqlen / kBlockM
                        n_block_max = ceil_div(seqlen, kBlockN)
                        num_n_blocks = min(n_block_max, ceil_div((num_m_blocks + 1) * kBlockM, kBlockN))
                        config = (causal, headdim, batch_size, seqlen)
                        nheads = dim // headdim
                        gQ = [kBlockM, headdim]
                        gK = [kBlockN, headdim]
                        gV = [kBlockN, headdim]
                        gP = [kBlockM, kBlockN]
                        sQ = [kBlockM, headdim]
                        sK = [kBlockN, headdim]
                        sV = [kBlockN, headdim]
                        sVt = [headdim, kBlockN]
                        tSrQ = [Nwarps * Layout_tile_multiply, 16, 16]
                        tSrK = [Nwarps * Layout_tile_multiply, 16, 16]
                        tOrVt = [Nwarps * Layout_tile_multiply, 16, 16]
                        tSgS = [Nwarps * Layout_tile_multiply, 16, 16]
                        acc_o = [Nwarps * Layout_tile_multiply, 16, 16]

                        mem_allocated_before = torch.cuda.memory_allocated(device)
                        mem_reserved_before = torch.cuda.memory_reserved(device)
                        qkv = torch.randn(batch_size, seqlen, 3, nheads, headdim, device=device, dtype=dtype,
                                          requires_grad=True)

                        try:
                            f, b = time_fwd_bwd(flash_attn_qkvpacked_func, qkv, dropout_p, causal=causal, repeats=repeats, verbose=False)
                            time_f[config, "Flash2"] = f
                            time_b[config, "Flash2"] = b
                        except Exception as e:
                            print(f"Error with Flash2: {e}")

                        print("Memory allocated after allocation:", torch.cuda.memory_allocated())
                        print("Memory reserved after allocation:", torch.cuda.memory_reserved())

                        try:
                            qkv = qkv.detach().requires_grad_(True)
                            f, b = time_fwd_bwd(attention_pytorch, qkv, dropout_p, causal=causal, repeats=repeats, verbose=False)
                            time_f[config, "Pytorch"] = f
                            time_b[config, "Pytorch"] = b
                        except Exception as e:
                            print(f"Error with Pytorch: {e}")
                            f, b = float('nan'), float('nan')

                        mem_allocated_after = torch.cuda.memory_allocated(device)
                        mem_reserved_after = torch.cuda.memory_reserved(device)
                        print(f"### causal={causal}, headdim={headdim}, batch_size={batch_size}, seqlen={seqlen} ###")
                        for method in methods:
                            try:
                                time_f_b[config, method] = time_f[config, method] + time_b[config, method]
                                speed_f[config, method] = efficiency(
                                    flops(batch_size, seqlen, headdim, nheads, causal, mode="fwd"),
                                    time_f[config, method]
                                )
                                speed_b[config, method] = efficiency(
                                    flops(batch_size, seqlen, headdim, nheads, causal, mode="bwd"),
                                    time_b[config, method]
                                )
                                speed_f_b[config, method] = efficiency(
                                    flops(batch_size, seqlen, headdim, nheads, causal, mode="fwd_bwd"),
                                    time_f_b[config, method]
                                )
                                print(
                                    f"{method} fwd: {speed_f[config, method]:.2f} TFLOPs/s, "
                                    f"bwd: {speed_b[config, method]:.2f} TFLOPs/s, "
                                    f"fwd + bwd: {speed_f_b[config, method]:.2f} TFLOPs/s"
                                )
                                # Write the row to the CSV file
                                csvwriter.writerow([
                                    dropout_p, causal, dim, headdim, nheads, batch_size, seqlen,
                                    kBlockN, num_n_blocks, kBlockM, num_m_blocks, Layout_tile_multiply, Nwarps,
                                    str(gQ), str(gK), str(gV), str(gP),
                                    str(sQ), str(sK), str(sV), str(sVt), str(tSrQ), str(tSrK), str(tOrVt), str(tSgS), str(acc_o), method,
                                    f"{speed_f[config, method]:.2f}", f"{speed_b[config, method]:.2f}", f"{speed_f_b[config, method]:.2f}",
                                    mem_allocated_before, mem_reserved_before, mem_allocated_after, mem_reserved_after
                                ])
                            except Exception as e:
                                print(f"Error processing {method}: {e}")

                        # Clear variables to avoid memory issues
                        del qkv, f, b
                    
                        print("Memory allocated after deletion:", torch.cuda.memory_allocated())
                        print("Memory reserved after deletion:", torch.cuda.memory_reserved())


