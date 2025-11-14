import torch
import time
import os
import sys
os.add_dll_directory(r"C:\AI_model\lada\build\gtk\x64\release\bin")
from lada.lib.frame_restorer import load_models
from lada.basicvsrpp.inference import load_model, get_default_gan_inference_config



mosaic_restoration_model_path = "C:/AI_model/lada/lada/model_weights/lada_mosaic_restoration_model_generic_v1.2.pth"

#torch.backends.cudnn.enabled = False
print(f"cudnn: {torch.backends.cudnn.enabled}")
config = get_default_gan_inference_config()
mosaic_restoration_model = load_model(config, mosaic_restoration_model_path, "cuda:0")
model = mosaic_restoration_model
#print(model)
randn_input = []
for i in range(4):
    randn_input.append(torch.randn(1, 180, 3, 256, 256))

for i in range(4):
    randn_input[i] = randn_input[i].to("cuda")
print("Loaded input:", randn_input[0].shape)


model.eval()

batch_start = time.time()
print(randn_input[0].dtype)
for i in range(4):
    start = time.time()
    with torch.no_grad():
        output = model(inputs=randn_input[i].to("cuda"))
        torch.cuda.synchronize()
        end = time.time()
    print(f"{i+1} Round: {end-start} seconds")
    #time.sleep(2)
print("Output shape:", output.shape)
end = time.time()
print(f"Batch Time: {end-batch_start} seconds")

half_model = model.to("cuda").eval().half()


half_radn_input = randn_input[0].to("cuda").half()

batch_start = time.time()
print(half_radn_input.dtype)
torch.cuda.synchronize()
for i in range(4):
    start = time.time()
    with torch.no_grad():
        output = half_model(inputs=half_radn_input.to("cuda"))
        torch.cuda.synchronize()
        end = time.time()
    print(f"{i+1} Round: {end-start} seconds")
print("Output shape:", output.shape)
end = time.time()
print(f"Batch Time: {end-batch_start} seconds")






del model
del mosaic_restoration_model
torch.cuda.empty_cache()
