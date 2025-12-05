

import os
#os.environ["MIOPEN_ENABLE_LOGGING"] = "1"
#os.environ["MIOPEN_ENABLE_LOGGING_CMD"] = "1"
#os.environ["MIOPEN_LOG_LEVEL"] = "7"
#os.environ["MIOPEN_ENABLE_LOGGING_ELAPSED_TIME"] = "1"
#os.environ["MIOPEN_DEBUG_CONV_WINOGRAD"] = "1"
#os.environ["MIOPEN_DEBUG_CONV_GEMM"] = "0"
#os.environ["MIOPEN_DEBUG_CONV_DIRECT"] = "0"
#os.environ["MIOPEN_FIND_MODE"] = "2"


import torch
import torch.nn.functional as F
import numpy as np
import time

# ===== 設定 =====
device = "cuda"  # GPU

# ===== 多組卷積配置 =====
configs = [
    (1, 3, 8, 16, 16, 3, 1, 1),
    (1, 8, 16, 32, 32, 3, 1, 1),
    (2, 3, 8, 32, 32, 5, 2, 2),
    (1, 16, 32, 64, 64, 3, 1, 1),
    (1, 32, 64, 64, 64, 3, 2, 1),
    (2, 32, 64, 80, 80, 3, 1, 1),
    (2, 32, 128, 80, 80, 3, 2, 1),
    (4, 32, 64, 56, 56, 5, 1, 2),
    (1, 64, 128, 128, 128, 3, 1, 1),
    (1, 64, 128, 128, 128, 3, 2, 1),
    (2, 64, 256, 128, 128, 3, 2, 1),
    (2, 128, 256, 64, 64, 3, 1, 1),
    (4, 128, 256, 128, 128, 5, 2, 2),
    (1, 1, 1, 1, 1, 1, 1, 0),
    (1, 32, 64, 7, 7, 7, 1, 3),
    (1, 32, 64, 3, 3, 3, 2, 0),
    (2, 3, 64, 224, 224, 3, 2, 1),
]

# ===== 儲存表格資料 =====
results_fp32 = []
results_fp16 = []

# ===== 迴圈測試 =====
for idx, (N, C_in, C_out, H, W, kernel_size, stride, padding) in enumerate(configs):
    print(f"\n=== Config {idx+1} ===")
    print(f"N={N}, C_in={C_in}, C_out={C_out}, H={H}, W={W}, kernel_size={kernel_size}, stride={stride}, padding={padding}")

    for dtype, table in [(torch.float32, results_fp32), (torch.float16, results_fp16)]:
        dtype_name = "FP32" if dtype == torch.float32 else "FP16"

        # 隨機輸入
        x_cpu = torch.randn(N, C_in, H, W, dtype=dtype)
        weight_cpu = torch.randn(C_out, C_in, kernel_size, kernel_size, dtype=dtype)

        # CPU 計算
        start_cpu = time.time()
        out_cpu = F.conv2d(x_cpu, weight_cpu, stride=stride, padding=padding)
        cpu_time = time.time() - start_cpu

        # GPU 計算
        x_gpu = x_cpu.to(device)
        weight_gpu = weight_cpu.to(device)

        torch.cuda.synchronize()
        start_gpu = time.time()
        out_gpu = F.conv2d(x_gpu, weight_gpu, stride=stride, padding=padding)
        torch.cuda.synchronize()
        gpu_time = time.time() - start_gpu

        # GPU NaN / Inf 檢查
        gpu_has_nan = torch.isnan(out_gpu).any().item()
        gpu_has_inf = torch.isinf(out_gpu).any().item()
        print(f"{dtype_name} → GPU NaN: {gpu_has_nan}, Inf: {gpu_has_inf}")

        # CPU vs GPU 嚴謹比對
        gpu_np = out_gpu.cpu().numpy()
        cpu_np = out_cpu.numpy()
        diff = np.abs(gpu_np - cpu_np)
        valid_mask = np.isfinite(diff)

        if not np.any(valid_mask):
            print(f"{dtype_name} ❌ All diff values invalid")
            max_diff = float("nan")
            mean_diff = float("nan")
        else:
            max_diff = np.max(diff[valid_mask])
            mean_diff = np.mean(diff[valid_mask])
            print(f"{dtype_name} max_diff = {max_diff:.3e}, mean_diff = {mean_diff:.3e}")
            if max_diff > 1e-3:
                print(f"{dtype_name} ⚠️ GPU result deviates from CPU")
            else:
                print(f"{dtype_name} ✅ GPU matches CPU")

        # 儲存結果
        table.append({
            "idx": idx+1,
            "N": N, "C_in": C_in, "C_out": C_out, "H": H, "W": W,
            "kernel": kernel_size, "stride": stride, "padding": padding,
            "CPU_time": cpu_time, "GPU_time": gpu_time,
            "max_diff": max_diff, "mean_diff": mean_diff,
            "NaN": gpu_has_nan, "Inf": gpu_has_inf
        })

# ===== 最後輸出表格 =====
def print_table(results, title):
    print(f"\n\n=== Summary Table ({title}) ===")
    header = f"{'Idx':>3} | {'N':>2} | {'Cin':>3} | {'Cout':>4} | {'H':>3} | {'W':>3} | {'K':>2} | {'S':>2} | {'P':>2} | {'CPU(s)':>7} | {'GPU(s)':>7} | {'max_diff':>10} | {'mean_diff':>10} | {'NaN':>3} | {'Inf':>3}"
    print(header)
    print("-"*100)
    for r in results:
        print(f"{r['idx']:3d} | {r['N']:2d} | {r['C_in']:3d} | {r['C_out']:4d} | {r['H']:3d} | {r['W']:3d} | {r['kernel']:2d} | {r['stride']:2d} | {r['padding']:2d} | {r['CPU_time']:7.4f} | {r['GPU_time']:7.4f} | {r['max_diff']:10.3e} | {r['mean_diff']:10.3e} | {r['NaN']:3} | {r['Inf']:3}")

print_table(results_fp32, "FP32")
print_table(results_fp16, "FP16")
