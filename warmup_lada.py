
import torch
import time
import os
import sys
os.add_dll_directory(r"C:\AI_model\lada\build\gtk\x64\release\bin")
from lada import MODEL_WEIGHTS_DIR, VERSION
from lada.cli import utils
from lada.utils import audio_utils
from lada.restorationpipeline.frame_restorer import FrameRestorer
from lada.restorationpipeline import load_models
from lada.utils.video_utils import get_video_meta_data, VideoWriter
from lada import LOG_LEVEL
from lada.utils import image_utils, video_utils, threading_utils, mask_utils
from lada.utils import visualization_utils
from lada.models.basicvsrpp.inference import load_model,get_default_gan_inference_config
from lada.utils.os_utils import gpu_has_tensor_cores


#os.environ["MIOPEN_ENABLE_LOGGING"] = "1"
#os.environ["MIOPEN_ENABLE_LOGGING_CMD"] = "1"
#os.environ["MIOPEN_LOG_LEVEL"] = "4"
#os.environ["MIOPEN_ENABLE_LOGGING_ELAPSED_TIME"] = "1"
os.environ["MIOPEN_USER_DB_PATH"] = "C:\\AI_model\\lada\\MIOPEN_USER_DB"
os.environ["MIOPEN_DEBUG_CONV_GEMM"] = "0"

mosaic_restoration_model_path = "C:/AI_model/lada/lada/model_weights/lada_mosaic_restoration_model_generic_v1.2.pth"
#torch.backends.cudnn.enabled = False
print(f"cudnn: {torch.backends.cudnn.enabled}")
config = get_default_gan_inference_config()
_model = load_model(config, mosaic_restoration_model_path, "cuda:0", fp16=True)




import torch
import threading
import queue
import random
import concurrent.futures

device = "cuda"
model = _model.to(device).half().eval()
mode = 8

def warmup_all_shapes(start_frame=1,max_frames=180):
    print("Warmup: building FindDB for all frame shapes (1..180)...")

    for t in range(start_frame, max_frames + 1):
        x = torch.randn(1, t, 3, 256, 256, device=device).half()
        torch.cuda.synchronize()
        start = time.time()
        with torch.inference_mode():
            y = model(inputs=x)
        torch.cuda.synchronize()
        print(f" Warmup T={t:3d}: done in {time.time() - start:.4f} sec")

    print("Warmup complete! All MIOpen FindDB entries ready.")
    return

if mode == 1:
    # 任務佇列（包含 1〜180）
    task_queue = queue.Queue()

    tasks = list(range(1, 181))
    random.shuffle(tasks)  # 隨機化任務順序

    for t in tasks:
        task_queue.put(t)

    def worker(worker_id):
        print(f"[Worker-{worker_id}] Started.")
        while True:
            try:
                t = task_queue.get_nowait()
            except queue.Empty:
                print(f"[Worker-{worker_id}] Finished.")
                break
            print(f"[Worker-{worker_id}] Warmup T={t} start   Remaining:{task_queue.qsize()}")
            x = torch.randn(1, t, 3, 256, 256, device=device).half()
            start = time.time()
            with torch.inference_mode():
                model(inputs=x)
            torch.cuda.synchronize()
            print(f"[Worker-{worker_id}] Warmup done T={t} in {time.time() - start:.4f} sec")
            task_queue.task_done()

    start1 = time.time()
    # 啟動三條線
    threads = []
    for i in range(3):
        t = threading.Thread(target=worker, args=(i+1,))
        t.start()
        threads.append(t)

    # 等全部任務完成
    for t in threads:
        t.join()

    print(f"Warmup for all shapes done in {time.time() - start1:.4f} sec")

elif mode == 2:
    def warmup_range(start_t, end_t):
        print(f"Warmup range {start_t}–{end_t} started...")
        for t in range(start_t, end_t + 1):
            x = torch.randn(1, t, 3, 256, 256, device=device).half()
            start = time.time()
            with torch.inference_mode():
                y = model(inputs=x)
            torch.cuda.synchronize()
            print(f"Warmup done T={t} in {time.time() - start:.4f} sec")
        print(f"Warmup range {start_t}–{end_t} finished.")

    # 三段平行執行
    ranges = [(1, 60), (61, 120), (121, 180)]

    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        futures = [executor.submit(warmup_range, s, e) for (s, e) in ranges]
        concurrent.futures.wait(futures)

    print("Warmup for all shapes done.")

elif mode == 3:
    # 執行 warmup
    warmup_all_shapes()


elif mode == 4:
    def warmup_range(start_t, end_t):
        print(f"Warmup range {start_t}–{end_t} started...")
        for t in range(start_t, end_t + 1):
            x = torch.randn(1, t, 3, 256, 256, device=device).half()
            start = time.time()
            with torch.inference_mode():
                y = model(inputs=x)
            torch.cuda.synchronize()
            print(f"Warmup done T={t} in {time.time() - start:.4f} sec")
        print(f"Warmup range {start_t}–{end_t} finished.")

    # 三段平行執行
    ranges = [(1, 90), (91, 180)]

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(warmup_range, s, e) for (s, e) in ranges]
        concurrent.futures.wait(futures)

    print("Warmup for all shapes done.")


elif mode == 5:
    START_FRAME = 42
    MAX_FRAMES = 180
    # 執行 warmup
    warmup_all_shapes(START_FRAME,MAX_FRAMES)


elif mode == 6:
    # 任務佇列（包含 1〜180）
    task_queue = queue.Queue()

    tasks = list(range(150, 181))
    random.shuffle(tasks)  # 隨機化任務順序

    for t in tasks:
        task_queue.put(t)

    def worker(worker_id):
        print(f"[Worker-{worker_id}] Started.")
        while True:
            try:
                t = task_queue.get_nowait()
            except queue.Empty:
                print(f"[Worker-{worker_id}] Finished.")
                break
            print(f"[Worker-{worker_id}] Warmup T={t} start   Remaining:{task_queue.qsize()}")
            x = torch.randn(1, t, 3, 256, 256, device=device).half()
            start = time.time()
            with torch.inference_mode():
                model(inputs=x)
            torch.cuda.synchronize()
            print(f"[Worker-{worker_id}] Warmup done T={t} in {time.time() - start:.4f} sec")
            task_queue.task_done()

    start1 = time.time()
    # 啟動三條線
    threads = []
    for i in range(2):
        t = threading.Thread(target=worker, args=(i+1,))
        t.start()
        threads.append(t)

    # 等全部任務完成
    for t in threads:
        t.join()

    print(f"Warmup for all shapes done in {time.time() - start1:.4f} sec")
else:
    pass

print("Test all shapes.")
s1 = time.time()
warmup_all_shapes()
print(f"Test for all shpaes done in {time.time() - s1:.4f} sec")
print('done.')

