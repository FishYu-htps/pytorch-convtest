
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

# ---- 共享隊列 ----
batch_queue = queue.Queue(maxsize=4)   # 模擬你 LADA pipeline 的 batch queue


# ---- preprocess (numpy → torch → GPU) ----
def preprocess_worker():
    for i in range(50):
        # 模擬影像
        imgs = np.random.randint(0, 255, (4, 384, 640, 3), dtype=np.uint8)

        t0 = time.time()

        # 模擬你的 preprocess 步驟
        im = imgs.astype(np.float32) / 255.0
        im = np.transpose(im, (0, 3, 1, 2))
        tensor = torch.from_numpy(im).contiguous().to("cuda:0")

        preprocess_t = (time.time() - t0) * 1000
        batch_queue.put((tensor, preprocess_t))   # 推給 inference thread


# ---- inference ----
def inference_worker():
    for i in range(50):
        tensor, prep_t = batch_queue.get()

        t0 = time.time()
        model.inference(tensor)
        infer_t = (time.time() - t0) * 1000

        print(f"[{i:02}] preprocess={prep_t:.3f} ms   inference={infer_t:.3f} ms")


# ---- 啟動 ----
t_pre = threading.Thread(target=preprocess_worker)
t_inf = threading.Thread(target=inference_worker)

t_pre.start()
t_inf.start()

t_pre.join()
t_inf.join()

del model
torch.cuda.empty_cache()

print("Done")