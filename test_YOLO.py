import time
import urllib.request
from ultralytics import YOLO
import torch
import os
os.environ["MIOPEN_ENABLE_LOGGING"] = "1"
os.environ["MIOPEN_ENABLE_LOGGING_CMD"] = "1"
os.environ["MIOPEN_LOG_LEVEL"] = "5"
os.environ["MIOPEN_ENABLE_LOGGING_ELAPSED_TIME"] = "1"
#os.environ["MIOPEN_DEBUG_CONV_WINOGRAD"] = "1"
#os.environ["MIOPEN_DEBUG_CONV_GEMM"] = "0"
#os.environ["MIOPEN_DEBUG_CONV_DIRECT"] = "0"
#os.environ["MIOPEN_FIND_MODE"] = "2"
# --------------------------------------------------------
# 0. Prepare Image
# --------------------------------------------------------
img_url = "https://ultralytics.com/images/bus.jpg"
img_path = "sample.jpg"

if not os.path.exists(img_path):
    print("Downloading sample image...")
    urllib.request.urlretrieve(img_url, img_path)
    print("Saved as", img_path)

# --------------------------------------------------------
# 1. Load Model
# --------------------------------------------------------
model = YOLO("yolov8s.pt")

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

print(f"Using device: {device}")

# --------------------------------------------------------
# 2. Inference
# --------------------------------------------------------
print("Running first inference...")
result = model(img_path, imgsz=640, half=True)
result[0].show()
