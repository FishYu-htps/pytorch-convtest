
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

# ===== 設定 =====
device = "cuda"  # GPU
dtype = torch.float16

# ===== 多組卷積配置 =====
configs = [
    # 小尺寸卷積
    (1, 3, 8, 16, 16, 3, 1, 1),
    (1, 8, 16, 32, 32, 3, 1, 1),
    (2, 3, 8, 32, 32, 5, 2, 2),

    # 中等尺寸卷積
    (1, 16, 32, 64, 64, 3, 1, 1),
    (1, 32, 64, 64, 64, 3, 2, 1),
    (2, 32, 64, 80, 80, 3, 1, 1),
    (2, 32, 128, 80, 80, 3, 2, 1),
    (4, 32, 64, 56, 56, 5, 1, 2),

    # 大尺寸卷積
    (1, 64, 128, 128, 128, 3, 1, 1),
    (1, 64, 128, 128, 128, 3, 2, 1),
    (2, 64, 256, 128, 128, 3, 2, 1),
    (2, 128, 256, 64, 64, 3, 1, 1),
    (4, 128, 256, 128, 128, 5, 2, 2),

    # 特殊/邊界情況
    (1, 1, 1, 1, 1, 1, 1, 0),
    (1, 32, 64, 7, 7, 7, 1, 3),
    (1, 32, 64, 3, 3, 3, 2, 0),
    (2, 3, 64, 224, 224, 3, 2, 1),
]

# ===== 迴圈測試 =====
for idx, (N, C_in, C_out, H, W, kernel_size, stride, padding) in enumerate(configs):
    print(f"\n=== Config {idx+1} ===")
    print(f"N={N}, C_in={C_in}, C_out={C_out}, H={H}, W={W}, kernel_size={kernel_size}, stride={stride}, padding={padding}")

    # 隨機輸入
    x_cpu = torch.randn(N, C_in, H, W, dtype=dtype)
    weight_cpu = torch.randn(C_out, C_in, kernel_size, kernel_size, dtype=dtype)

    # CPU 計算
    out_cpu = F.conv2d(x_cpu, weight_cpu, stride=stride, padding=padding)

    # GPU 計算
    x_gpu = x_cpu.to(device)
    weight_gpu = weight_cpu.to(device)
    out_gpu = F.conv2d(x_gpu, weight_gpu, stride=stride, padding=padding)

    # GPU NaN / Inf 檢查
    gpu_has_nan = torch.isnan(out_gpu).any().item()
    gpu_has_inf = torch.isinf(out_gpu).any().item()
    print(f"GPU → NaN: {gpu_has_nan}, Inf: {gpu_has_inf}")

    # CPU vs GPU 嚴謹比對
    gpu_np = out_gpu.cpu().numpy()
    cpu_np = out_cpu.numpy()
    diff = np.abs(gpu_np - cpu_np)
    valid_mask = np.isfinite(diff)

    if not np.any(valid_mask):
        print("❌ All diff values are invalid (NaN/Inf everywhere)")
    else:
        max_diff = np.max(diff[valid_mask])
        mean_diff = np.mean(diff[valid_mask])
        print(f"max_diff  = {max_diff:.6e}, mean_diff = {mean_diff:.6e}")
        if max_diff > 1e-3:
            print("⚠️ WARNING: GPU result deviates significantly from CPU result!")
        else:
            print("✅ GPU result matches CPU result")