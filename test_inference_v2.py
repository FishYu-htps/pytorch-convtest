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

#torch.distributed = type('FakeDistributed', (), {'is_available': staticmethod(lambda: False)})()

#os.environ["MIOPEN_ENABLE_LOGGING"] = "1"
#os.environ["MIOPEN_ENABLE_LOGGING_CMD"] = "1"
#os.environ["MIOPEN_LOG_LEVEL"] = "5"
#os.environ["MIOPEN_ENABLE_LOGGING_ELAPSED_TIME"] = "1"



mosaic_restoration_model_path = "C:/AI_model/lada/lada/model_weights/lada_mosaic_restoration_model_generic_v1.2.pth"
#torch.backends.cudnn.enabled = False
print(f"cudnn: {torch.backends.cudnn.enabled}")
config = get_default_gan_inference_config()

_model = load_model(config, mosaic_restoration_model_path, "cuda:0", fp16=True)

#mosaic_restoration_model = load_model(config, mosaic_restoration_model_path, "cuda:0",fp16=True,clip_length=180)
#model = torch.compile(_model, backend="inductor")
#print(model)
randn_input = []
for i in range(3):
    randn_input.append(torch.randn(1, 180, 3, 256, 256))

for i in range(3):
    randn_input[i] = randn_input[i].to("cuda")
print("Loaded input:", randn_input[0].shape)

#model = torch.compile(model, backend="inductor")


half_model = _model.to("cuda").eval()


half_radn_input = randn_input[0].to("cuda").half()

batch_start = time.time()
print(half_radn_input.dtype)
torch.cuda.synchronize()
for i in range(2):
    start = time.time()
    with torch.inference_mode():
        output = half_model(inputs=half_radn_input.to("cuda"))
        #torch.cuda.synchronize()
        end = time.time()
    print(f"{i+1} Round: {end-start} seconds")
print("Output shape:", output.shape)
end = time.time()
print(f"Batch Time: {end-batch_start} seconds")






del _model
del half_model
torch.cuda.empty_cache()
