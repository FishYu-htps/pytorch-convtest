

import os
os.environ["MIOPEN_ENABLE_LOGGING"] = "1"
os.environ["MIOPEN_ENABLE_LOGGING_CMD"] = "1"
os.environ["MIOPEN_LOG_LEVEL"] = "5"
#os.environ["MIOPEN_ENABLE_LOGGING_ELAPSED_TIME"] = "1"
os.environ["HIPBLASLT_LOG_LEVEL"] = "5" 
os.environ["HIPBLASLT_LOG_MASK"] = "128" 
os.environ["HIPBLASLT_LOG_FILE"] = "hipblaslt.log"
os.environ["ROCBLAS_LAYER"] = "8"


os.environ["AMD_LOG_LEVEL"] = "5" 
os.environ["AMD_LOG_MASK"] = "-1" 
#os.environ["MIOPEN_CHECK_NUMERICS"] = "1" 
#os.environ["MIOPEN_DEBUG_CONV_WINOGRAD"] = "1"
#os.environ["MIOPEN_DEBUG_CONV_GEMM"] = "0"
#os.environ["MIOPEN_DEBUG_CONV_DIRECT"] = "0"
#os.environ["MIOPEN_FIND_MODE"] = "2"
#os.environ["HIPBLASLT_TENSILE_LIBPATH"] = "C:\\AI_model\\313venv\\Lib\\site-packages\\_rocm_sdk_libraries_gfx120X_all\\bin\\hipblaslt\\library"
#os.environ["HIPBLASLT_TENSILE_LIBPATH"] = "G:\\lada\\Therock\\dist\\rocm\\bin\\hipblaslt\\library"
import torch
import torch.nn.functional as F
import numpy as np
import time

# ===== 設定 =====
device = "cuda"  # GPU

# ===== 多組卷積配置 =====
configs = [
  #  (1, 3, 8, 16, 16, 3, 1, 1),
  #  (1, 8, 16, 32, 32, 3, 1, 1),
  #  (2, 3, 8, 32, 32, 5, 2, 2),
  #  (1, 16, 32, 64, 64, 3, 1, 1),
  #  (1, 32, 64, 64, 64, 3, 2, 1),
  #  (2, 32, 64, 80, 80, 3, 1, 1),
  #  (2, 32, 128, 80, 80, 3, 2, 1),
  #  (4, 32, 64, 56, 56, 5, 1, 2),
  #  (1, 64, 128, 128, 128, 3, 1, 1),
  #  (1, 64, 128, 128, 128, 3, 2, 1),
  #  (2, 64, 256, 128, 128, 3, 2, 1),
  #  (2, 128, 256, 64, 64, 3, 1, 1),
  #  (4, 128, 256, 128, 128, 5, 2, 2),
  #  (1, 1, 1, 1, 1, 1, 1, 0),
  #  (1, 32, 64, 7, 7, 7, 1, 3),
    (1, 32, 64, 3, 3, 3, 2, 0),
  #  (2, 3, 64, 224, 224, 3, 2, 1),
]

# ===== 儲存表格資料 =====
results_fp32 = []
results_fp16 = []

# ===== 迴圈測試 =====
for idx, (N, C_in, C_out, H, W, kernel_size, stride, padding) in enumerate(configs):
    print(f"\n=== Config {idx+1} ===")
    print(f"N={N}, C_in={C_in}, C_out={C_out}, H={H}, W={W}, kernel_size={kernel_size}, stride={stride}, padding={padding}")
    print(torch.__version__)
    #for dtype, table in [(torch.float32, results_fp32), (torch.float16, results_fp16)]:
    for dtype, table in [(torch.float32, results_fp32)]:
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
        print("========start move weight==============")
        weight_gpu = weight_cpu.to(device)
        torch.cuda.synchronize()
        print("========end move weight===========")
        

        start_gpu = time.time()
        print("========start calc conv==============")
        out_gpu = F.conv2d(x_gpu, weight_gpu, stride=stride, padding=padding)
        torch.cuda.synchronize()
        print("========end calc conv===========")
        gpu_time = time.time() - start_gpu


        x_cpu2 = x_gpu.to('cpu')
        weight_cpu2 = weight_gpu.to('cpu')
        out_cpu2 = F.conv2d(x_cpu2, weight_cpu2, stride=stride, padding=padding)


        # GPU NaN / Inf 檢查
        gpu_has_nan = torch.isnan(out_gpu).any().item()
        gpu_has_inf = torch.isinf(out_gpu).any().item()
        print(f"{dtype_name} → GPU NaN: {gpu_has_nan}, Inf: {gpu_has_inf}")

        # CPU vs GPU 嚴謹比對
        gpu_np = out_gpu.cpu().numpy()
        cpu_np = out_cpu.numpy()
        cpu2_np = out_cpu2.numpy()
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
            if dtype_name == 'FP32':
                if max_diff > 1e-3:
                    print(f"{dtype_name} ⚠️ GPU result deviates from CPU")
                else:
                    print(f"{dtype_name} ✅ GPU matches CPU")
            else:
                if max_diff > 1e-2:
                    print(f"{dtype_name} ⚠️ GPU result deviates from CPU")
                else:
                    print(f"{dtype_name} ✅ GPU matches CPU")

