
import time
import torch
import os
import threading
import queue
import numpy as np

os.environ["MIOPEN_ENABLE_LOGGING_CMD"] = "1"
os.environ["MIOPEN_ENABLE_LOGGING"] = "1"
os.environ["MIOPEN_LOG_LEVEL"] = "5"
os.environ["MIOPEN_ENABLE_LOGGING_ELAPSED_TIME"] = "1"
#torch.backends.cudnn.enabled = False
print(f"enabled cudnn :{torch.backends.cudnn.enabled}")
lada_path = "C:/AI_model/lada/lada/"
from lada.lib.mosaic_detection_model import MosaicDetectionModel
mosaic_detection_model_path = lada_path + "model_weights/lada_mosaic_detection_model_v3.1_fast.pt"
mosaic_detection_model = MosaicDetectionModel(mosaic_detection_model_path, "cuda:0", classes=[0], conf=0.2)
model = mosaic_detection_model

''' Moedel Structure
#print("=== Model structure ===")
#from torchinfo import summary
#summary(model.model, input_data=x, depth=6, col_names=("input_size", "output_size", "num_params"))
'''


# ---- 共享隊列 ----
batch_queue = queue.Queue(maxsize=1)   # 模擬你 LADA pipeline 的 batch queue
pre_lock = threading.Lock()
inference_lock = threading.Lock()
# --- 同步工具 ---
ready_for_inference = threading.Event()   # preprocess 完成 → 通知 inference
ready_for_preprocess = threading.Event()  # inference 完成 → 通知 preprocess

ready_for_preprocess.set()   # 允許第一次 preprocess 開始

# ---- preprocess (numpy → torch → GPU) ----
def preprocess_worker():
    for i in range(3):

        # 等待 inference 完成
        ready_for_preprocess.wait()

        # 模擬影像
        imgs = np.random.randint(0, 255, (4, 384, 640, 3), dtype=np.uint8)

        t0 = time.time()

        # 模擬你的 preprocess 步驟
        im = imgs.astype(np.float32) / 255.0
        im = np.transpose(im, (0, 3, 1, 2))
        print(f"================pre:{i} start_transfer")
        tensor = torch.from_numpy(im).contiguous().to("cuda:0")
        preprocess_t = (time.time() - t0) * 1000
        print(f"================pre:{i} time={preprocess_t:.3f} ms")
        batch_queue.put((tensor, preprocess_t))   # 推給 inference thread

        # 通知 inference 可以開始
        ready_for_preprocess.clear()
        ready_for_inference.set()


# ---- inference ----
def inference_worker():
    for i in range(3):

        # 等待 preprocess 完成
        ready_for_inference.wait()
        tensor, prep_t = batch_queue.get()
        print(f"================inf:{i} start_inference")
        t0 = time.time()
        #with inference_lock:
            #model.inference(tensor)
        model.inference(tensor)
        torch.cuda.synchronize()
        infer_t = (time.time() - t0) * 1000
        print(f"================inf:{i} time={infer_t:.3f} ms")
        print(f"[{i:02}] preprocess={prep_t:.3f} ms   inference={infer_t:.3f} ms")

        # 通知 preprocess 可以下一輪
        ready_for_inference.clear()
        ready_for_preprocess.set()


# ---- 啟動 ----
t_pre = threading.Thread(target=preprocess_worker)
t_inf = threading.Thread(target=inference_worker)
start = time.time()
t_pre.start()
t_inf.start()

t_pre.join()
t_inf.join()
end = time.time() - start
del model
torch.cuda.empty_cache()

print(f"Done. Total time = {end}")