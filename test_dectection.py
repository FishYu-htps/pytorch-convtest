
import time
import torch
import os
import threading
import queue
import numpy as np

#os.environ["MIOPEN_ENABLE_LOGGING_CMD"] = "1"
#os.environ["MIOPEN_LOG_LEVEL"] = "5"
#os.environ["MIOPEN_ENABLE_LOGGING_ELAPSED_TIME"] = "1"
#torch.backends.cudnn.enabled = False
print(f"enabled cudnn :{torch.backends.cudnn.enabled}")

from lada.lib.mosaic_detection_model import MosaicDetectionModel
mosaic_detection_model_path = "C:/AI_model/lada/lada/model_weights/lada_mosaic_detection_model_v3.1_fast.pt"
mosaic_detection_model = MosaicDetectionModel(mosaic_detection_model_path, "cuda:0", classes=[0], conf=0.2)
model = mosaic_detection_model
print("=== Model structure ===")
from torchinfo import summary
#summary(model.model, input_data=x, depth=6, col_names=("input_size", "output_size", "num_params"))






x = torch.randn(4, 3, 384, 640).to("cuda:0")
#print(model.model)  # 這通常是 YOLO 主體 (nn.Module)
for i in range(10):
    x = torch.randn(4, 3, 384, 640).to("cuda:0") #.half()
    t0 = time.time()
    inference_results = model.inference(x) if x is not None else None
    end= time.time() -t0
    print(f"Batch Time: {end*1000} ms")



del mosaic_detection_model
torch.cuda.empty_cache()