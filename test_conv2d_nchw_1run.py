import torch
import time
import os
#torch.backends.cudnn.enabled = False

os.environ["MIOPEN_ENABLE_LOGGING_CMD"] = "1"
os.environ["MIOPEN_LOG_LEVEL"] = "5"
os.environ["MIOPEN_ENABLE_LOGGING_ELAPSED_TIME"] = "1"

# Convolution settings
batch_size = 32
height = 256
width = 256
in_channels = [64,8]
out_channels = [64,32]
stride = (1, 1)
padding = [1,3]
kernel = [3,7]


# Number of warmup and measured runs
warmup = 2
runs = 10

# Dtypes to test
dtypes = {
    "float32": torch.float32,
    "float16": torch.float16,
}

def benchmark(dtype_name, dtype, kernel,i):
    print(f"shape:[{in_channels[i]},{out_channels[i]},{kernel}]")
    print(f"\nTesting {dtype_name}...")
    kernel_size = (kernel, kernel) if isinstance(kernel, int) else kernel
    # 直接創建 NCHW / OIHW
    x = torch.randn((batch_size, in_channels[i], height, width), dtype=dtype, device="cuda")  # NCHW
    w = torch.randn((out_channels[i], in_channels[i], *kernel_size), dtype=dtype, device="cuda") # OIHW

    conv = torch.nn.functional.conv2d

    # Warmup
    for _ in range(warmup):
        y = conv(x, w, stride=stride, padding=padding[i])
        torch.cuda.synchronize()
    for _ in range(warmup):
        y = conv(x, w, stride=stride, padding=padding[i])
        torch.cuda.synchronize()

i=0
for kernel_size in kernel:
    print("\n===========================================\n")
    #print(f"kernel = {kernel_size}x{kernel_size}")
    for name, dtype in dtypes.items():
        try:
            benchmark(name, dtype,kernel_size,i)
        except Exception as e:
            print(f"Skipped {name} due to error: {e}")
    i=i+1