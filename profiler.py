import onnx
import os
import onnxruntime
import numpy as np
from PIL import Image
import subprocess
import time



def get_gpu_mem():
    out = subprocess.check_output([
        "nvidia-smi",
        "--query-gpu=memory.used",
        "--format=csv,nounits,noheader"
    ])
    # take the first line (GPU 0)
    return int(out.split()[0])


def get_onnx_model_size(model):
    total_params = 0
    for tensor in model.graph.initializer:
        size = 1
        for dim in tensor.dims:
            size *= dim
        total_params += size
    return total_params


def inference(model_path):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    image = Image.open("/datasets/sutterra/manual/normal_iPh.png").convert("RGB")
    image = image.resize((288, 288))
    image = np.array(image).astype(np.float32) / 255.0

    image = (image - mean) / std
    image = image.transpose((2, 0, 1))
    image_batch = np.expand_dims(image, axis=0).astype(np.float32)
    image_batch = np.repeat(image_batch, 1, axis=0)


    # load model
    session = onnxruntime.InferenceSession(model_path, providers=['CUDAExecutionProvider'])
    inp_name = session.get_inputs()[0].name

    # check which provider(s) are actually in use
    used = session.get_providers()
    print("ONNX-Runtime execution providers:", used)
    if "CUDAExecutionProvider" not in used:
        print("⚠️  CUDAExecutionProvider not available — running on CPU")

    # give the driver a moment to settle
    time.sleep(0.1)
    base = get_gpu_mem()

    # prepare a dummy batch
    image = Image.open("/datasets/sutterra/manual/normal_iPh.png").convert("RGB")
    image = image.resize((288, 288))
    arr = (np.array(image).astype(np.float32)/255.0 - mean) / std
    batch = np.repeat(arr.transpose(2,0,1)[None,...], 1, axis=0).astype(np.float32)

    # warm-up run
    session.run(None, {inp_name: batch})
    time.sleep(0.1)

    # timed inference
    start = time.perf_counter()
    session.run(None, {inp_name: batch})
    end = time.perf_counter()
    elapsed_ms = (end - start) * 1000
    print(f"Inference time: {elapsed_ms:.1f} ms")

    # measure VRAM as before…
    peak = get_gpu_mem()
    print(f"Approx. VRAM used by inference: {peak - base} MB")

    # # segmentation output
    # output = np.expand_dims(output, axis=1)[0]
    # output = output.transpose((1, 2, 0))
    # output = cv2.resize(output, (288, 288), interpolation=cv2.INTER_LINEAR)
    # output = cv2.GaussianBlur(output, (33, 33), 4)

    # # binary or heatmap
    # # ret, mask = cv2.threshold(output, 0.5, 255, cv2.THRESH_BINARY)
    # mask = cv2.cvtColor(output, cv2.COLOR_GRAY2BGR)
    # mask = (mask * 255).astype('uint8')
    # mask = cv2.applyColorMap(mask, cv2.COLORMAP_JET)


import onnxruntime as ort
import torch
import sys, subprocess




#path = "/home/sutterra/GLASS/results/models/backbone_0/mvtec_mms_stretch_resnet18_v2/"
path = "/home/sutterra/GLASS/results/models/backbone_0/mvtec_mms_stretch_wideresnet/"

#find the the file with simplified.onnx in the name
simplified_onnx_path = path + [f for f in os.listdir(path) if "simplified.onnx" in f][0]
#find the file with onnx in the name and not simplified
onnx_path = path + [f for f in os.listdir(path) if "onnx" in f and "simplified" not in f][0]



simpliefied_model = onnx.load(simplified_onnx_path)
onnx_model = onnx.load(onnx_path)

simplified_model_size = get_onnx_model_size(simpliefied_model)
onnx_model_size = get_onnx_model_size(onnx_model)

inference(simplified_onnx_path)

print(f"Simplified Model Size: {simplified_model_size:,} parameters")
print(f"Original Model Size: {onnx_model_size:,} parameters")


#analyze to logfile in the folder 
logfile = os.path.join(path, [f for f in os.listdir(path) if ".log" in f][0])
print(logfile)  
