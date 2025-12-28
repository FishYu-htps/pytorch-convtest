
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

#os.environ["MIOPEN_ENABLE_LOGGING"] = "1"
#os.environ["MIOPEN_ENABLE_LOGGING_CMD"] = "1"
#os.environ["HIPBLASLT_LOG_MASK"] = "32"
#os.environ["HIPBLASLT_LOG_FILE"] = "your_model_hipblaslt_log.txt"
#os.environ["TORCH_BLAS_PREFER_HIPBLASLT"] = "1"

#os.environ["MIOPEN_ENABLE_LOGGING"] = "1"
#os.environ["MIOPEN_ENABLE_LOGGING_CMD"] = "1"
#os.environ["MIOPEN_LOG_LEVEL"] = "4"
#os.environ["MIOPEN_ENABLE_LOGGING_ELAPSED_TIME"] = "1"
os.environ["MIOPEN_USER_DB_PATH"] = "C:\\AI_model\\lada\\MIOPEN_USER_DB"
#os.environ["MIOPEN_FIND_MODE"] = "2"
#os.environ["MIOPEN_DEBUG_CONV_GEMM"] = "0"
mosaic_restoration_model_path = "C:/AI_model/lada/lada/model_weights/lada_mosaic_restoration_model_generic_v1.2.pth"



import threading
import queue
import random
import concurrent.futures
import argparse
import psutil

p = psutil.Process(os.getpid())
#p.cpu_affinity([0,1,2,3,4,5,6,7])
#p.cpu_affinity([0,1,2,3])
#p.cpu_affinity([0,3])
#p.cpu_affinity([4,5,6,7])
#p.cpu_affinity([8,9,10,11])
#p.cpu_affinity([12,13,14,15])
#p.cpu_affinity([4,5,8,9])
p.cpu_affinity([1,3,5,7])
device = "cuda"
 

def init(mode = 5,fp16 = 1, cudnn = 1):
    if mode < 5:
        os.environ["MIOPEN_FIND_MODE"] = "1"
    else:
        os.environ["MIOPEN_FIND_MODE"] = "5"
    
    if cudnn != 1:
        torch.backends.cudnn.enabled = False

    print(f"cudnn: {torch.backends.cudnn.enabled}")
    config = get_default_gan_inference_config()
    enable_fp16 = fp16 == 1

    _model = load_model(config, mosaic_restoration_model_path, "cuda:0", fp16=enable_fp16)
    model = _model.to(device).eval()
    return model



def warmup_all_shapes(model, start_frame=1, max_frames=180, warmup=False, FP16=1):
    if warmup:
        print(f"Warmup: building FindDB for frame shapes ({start_frame}..{max_frames})...")
    else:
        print(f"Test: building FindDB for frame shapes ({start_frame}..{max_frames})...")

    for t in range(start_frame, max_frames + 1):
        x = torch.randn(1, t, 3, 256, 256, device=device)
        if FP16 == 1:
            x=x.half()
        torch.cuda.synchronize()
        start = time.time()
        with torch.inference_mode():
            y = model(inputs=x)
        torch.cuda.synchronize()
        print(f" Warmup T={t:3d}: done in {time.time() - start:.4f} sec")
        if not warmup:
            time.sleep(0.5)
        elif t>140:
            time.sleep(5)
        else:
            time.sleep(0.5)

def warmup_one_shape(model, start_frame=1, max_frames=2, warmup=False, FP16=1):
        if warmup:
            print(f"Warmup: building FindDB for frame shape {start_frame}")
            max_frames = 1
        else:
            print(f"Test: building FindDB for frame shape {start_frame}. {max_frames} times.")

        for i in range(max_frames):
            x = torch.randn(1, start_frame, 3, 256, 256, device=device)
            if FP16 == 1:
                x=x.half()
            torch.cuda.synchronize()
            start = time.time()
            with torch.inference_mode():
                y = model(inputs=x)
            torch.cuda.synchronize()
            print(f" Warmup T={start_frame:3d}: done in {time.time() - start:.4f} sec")
            time.sleep(0.5)


def run(mode: int, start_frame: int | None, max_frames: int | None, fp16: int | None, cudnn: int | None):
    print(f"mode = {mode}")
    print(f"START_FRAME = {start_frame}")
    print(f"MAX_FRAMES = {max_frames}")
    print(f"FP16 = {fp16}")
    model = init(mode=mode,fp16=fp16,cudnn=cudnn)
    warmup = mode < 5


    if mode == 1:
        run_mode_1(model, start_frame, max_frames,warmup,fp16)
    elif mode == 2:
        run_mode_2(model, start_frame, max_frames)
    elif mode == 3:
        run_mode_3(model, start_frame, max_frames)
    elif mode == 5:
        run_mode_5(model, start_frame, max_frames,warmup,fp16)
    elif mode == 6:
        run_mode_6(model, start_frame, max_frames,warmup,fp16)
    else:
        raise ValueError(f"Unsupported mode: {mode}")

def run_mode_1(model,start_frame, max_frames,warmup,fp16):
    print("Running mode 1")
    warmup_all_shapes(model, start_frame=start_frame,max_frames=max_frames,warmup=warmup,FP16=fp16)
    # 原本 mode = 1 的邏輯


def run_mode_2(model, start_frame, max_frames):
    print("Running mode 2")
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



def run_mode_3(model, start_frame, max_frames):
    print("Running mode 3")
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


def run_mode_5(model, start_frame, max_frames,warmup,fp16):
    print("Running mode 5: Test shapes")
    warmup_all_shapes(model, start_frame=start_frame,max_frames=max_frames,warmup=warmup,FP16=fp16)

def run_mode_6(model, start_frame, max_frames,warmup,fp16):
    print("Running mode 6: Test one shape")
    warmup_one_shape(model, start_frame=start_frame,max_frames=max_frames,warmup=warmup,FP16=fp16)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Example CLI for xxx.py"
    )

    parser.add_argument(
        "--mode",
        type=int,
        default=5,
        choices=[1, 2, 3,5,6],
        help="Execution mode (default: 1)"
    )

    parser.add_argument(
        "--START",
        type=int,
        default=1,
        help="Start frame index"
    )

    parser.add_argument(
        "--MAX",
        type=int,
        default=180,
        help="Maximum number of frames to process"
    )
    parser.add_argument(
        "--FP16",
        type=int,
        default=1,
        help="FP16 Enable"
    )

    parser.add_argument(
        "--cudnn",
        type=int,
        default=1,
        help="cudnn Enable"
    )



    return parser.parse_args()



def main():
    args = parse_args()
    run(
        mode=args.mode,
        start_frame=args.START,
        max_frames=args.MAX,
        fp16=args.FP16,
        cudnn=args.cudnn
    )


if __name__ == "__main__":
    main()
    print("Warmup complete! All MIOpen FindDB entries ready.")


