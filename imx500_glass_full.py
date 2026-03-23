from picamera2 import Picamera2, Preview
from picamera2.picamera2 import Picamera2
from picamera2.devices import IMX500

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import backbones
from glass import GLASS, IMAGENET_MEAN, IMAGENET_STD
import common
from torchvision import models
from torch.profiler import profile, record_function, ProfilerActivity
import os, ctypes, sys
import time
import cv2
import numpy as np
import scipy.ndimage as ndimage
import hashlib
import onnxruntime                                  # <<— add this at the top with your other imports

def load_image(img_path, device, input_size):
    tf = transforms.Compose([
        transforms.Resize(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    img = Image.open(img_path).convert("RGB")
    return tf(img).unsqueeze(0).to(device)  # (1,C,H,W)

class ResNet18Truncated(torch.nn.Module):
    def __init__(self, pretrained: bool = True):
        super().__init__()
        # Load the pretrained ResNet18
        base_model = models.resnet18(pretrained=pretrained)
        
        #print available layers
        available_layers = [name for name, _ in base_model.named_modules()]
        #print(f"Available layers in the resnet18: {available_layers}")
        
        # Copy the initial layers (conv1, bn1, relu, maxpool)
        self.conv1   = base_model.conv1
        self.bn1     = base_model.bn1
        self.relu    = base_model.relu
        self.maxpool = base_model.maxpool
        
        # Keep layer1 (as it's needed to feed into layer2)
        self.layer1 = base_model.layer1
        # Keep layer2 and layer3
        self.layer2 = base_model.layer2
        self.layer3 = base_model.layer3

    def forward(self, input: torch.Tensor):
        # Stem
        x = self.conv1(input)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        # Pass through layer1
        x = self.layer1(x)
        # After layer2
        x2 = self.layer2(x)
        # After layer3
        x3 = self.layer3(x2)
        
        # Return feature maps from layer2 and layer3
        return {"layer2": x2, "layer3": x3, "input": input}

def show_tensor_image(window_name: str, tensor):
    """
    Display a C×H×W or 1×C×H×W tensor (or numpy array) in OpenCV window.
    """
    # to numpy
    if isinstance(tensor, torch.Tensor):
        arr = tensor.detach().cpu().numpy()
    else:
        arr = tensor
    # remove batch dim if present
    if arr.ndim == 4:
        arr = arr[0]
    # C×H×W → H×W×C
    img = np.transpose(arr, (1, 2, 0))
    # RGB → BGR
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imshow(window_name, img)
    cv2.waitKey(1)   
    
    
def render_heatmap(mask, output_size=(288,288), label: str | None = None):
    """
    Display a H×W mask (torch Tensor or numpy) as a colored heatmap,
    optionally resizing to output_size=(H_out, W_out).
    """
    # pull to numpy
    if isinstance(mask, torch.Tensor):
        mask_np = mask.detach().cpu().numpy()
    else:
        mask_np = mask
    mask_arr = mask_np.squeeze()              # now H×W

    # resize if requested
    if output_size is not None and mask_arr.shape != output_size:
        # cv2.resize expects (width, height)
        mask_arr = cv2.resize(mask_arr,
                              (output_size[1], output_size[0]),
                              interpolation=cv2.INTER_LINEAR)

    # normalize to 0–255
    mask_uint8 = cv2.normalize(mask_arr, None, 0, 255,
                               cv2.NORM_MINMAX).astype(np.uint8)
    # apply colormap
    heatmap = cv2.applyColorMap(mask_uint8, cv2.COLORMAP_JET)
    if label:
        lines = label.split("\n")
        y = 18
        for ln in lines:
            cv2.putText(heatmap, ln, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            y += 16
    return heatmap
    cv2.waitKey(1)


def convert_to_segmentation(patch_scores, smoothing=4, size=288, device="cpu"):
    """Upsample patch scores to the full image resolution and smooth."""
    with torch.no_grad():
        if isinstance(patch_scores, np.ndarray):
            patch_scores = torch.from_numpy(patch_scores)
        scores = patch_scores.to(device).unsqueeze(1)
        scores = F.interpolate(scores, size=size, mode="bilinear", align_corners=False)
        scores = scores.squeeze(1)
        patch_scores = scores.cpu().numpy()
    return [ndimage.gaussian_filter(patch_score, sigma=smoothing) for patch_score in patch_scores]


def tensor_checksum(tensor):
    if tensor is None:
        return "None"
    return hashlib.md5(tensor.detach().cpu().numpy().tobytes()).hexdigest()[:16]


def render_imx_heatmap(imx_outputs, output_size=(256, 256), label: str | None = None):
    """Render a heatmap from the first IMX output tensor/array and return BGR image."""
    if not imx_outputs:
        return
    arr = imx_outputs[0]
    if arr is None:
        return
    arr_np = np.array(arr)
    # Squeeze batch/channel dims until <=2 dims remain
    while arr_np.ndim > 2:
        arr_np = np.squeeze(arr_np, axis=0)
    if arr_np.ndim == 1:
        side = int(np.ceil(np.sqrt(arr_np.size)))
        padded = np.zeros(side * side, dtype=arr_np.dtype)
        padded[: arr_np.size] = arr_np
        arr_np = padded.reshape(side, side)
    # normalize to 0–255
    arr_norm = cv2.normalize(arr_np, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    if output_size is not None:
        arr_norm = cv2.resize(arr_norm, (output_size[1], output_size[0]), interpolation=cv2.INTER_LINEAR)
    heatmap = cv2.applyColorMap(arr_norm, cv2.COLORMAP_JET)
    if label:
        lines = label.split("\n")
        y = 18
        for ln in lines:
            cv2.putText(heatmap, ln, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            y += 16
    return heatmap
    
    

def main():
    # ---- User inputs ----
    #image_path      = "/datasets/sutterra/mvtec/mms_stretch/test/good/IMG_0009.png"
    #image_path      = "/datasets/sutterra/mvtec_debug/mms_stretch/test/good/IMG_0009.png"
    image_path_good     = "/home/pi/glass/from_venus/IMG_0201.png"
    image_path_anomaly  = "/home/pi/glass/from_venus/IMG_0038.png"
    #image_path      ="/datasets/sutterra/mvtec_debug/mms_stretch/test/crack_hole/IMG_0039.png"

    checkpoint_path = "/home/pi/glass_imx/glass-main/results/models/backbone_0/mvtec_mms_rpi/ckpt_best_607.pth"
    device          = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backbone_name   = "resnet18"
    layers_to_use   = ("layer2", "layer3")
    # Use 256x256 to match GLASS's built-in patch grid (32->16) and avoid reshape mismatch
    input_shape     = (3, 256, 256)
    pretrain_embed_dimension = 384
    target_embed_dimension = 384
    
    # ---------------------

    # 1) load backbone
    backbone = backbones.load(backbone_name)
    backbone = backbone.to(device)

    # 2) init GLASS and load config (same call style as forward.py)
    glass = GLASS(device)
    glass.load(
        backbone=backbone,
        layers_to_extract_from=layers_to_use,
        device=device,
        input_shape=input_shape,
        pretrain_embed_dimension=pretrain_embed_dimension,
        target_embed_dimension=target_embed_dimension,
        patchsize=3,
        patchstride=1,
        skip_backbone=False,
    )

    # Hardcode patch grid sizes for input 256 (layer2: 32×32, layer3: 16×16)
    glass.static_patch_shapes = [(32, 32), (16, 16)]
    glass.trace_mode = False

    # 3) load checkpoint weights with Linear→Conv reshaping (as in forward.py)
    state = torch.load(checkpoint_path, map_location=device)
    if "discriminator" in state:
        d_state = state["discriminator"]
        if "tail.0.weight" in d_state and d_state["tail.0.weight"].ndim == 2:
            d_state["tail.0.weight"] = d_state["tail.0.weight"].unsqueeze(-1).unsqueeze(-1)
        if "body.block1.0.weight" in d_state and d_state["body.block1.0.weight"].ndim == 2:
            d_state["body.block1.0.weight"] = d_state["body.block1.0.weight"].unsqueeze(-1).unsqueeze(-1)
        glass.discriminator.load_state_dict(d_state, strict=False)
    else:
        raise RuntimeError("Discriminator weights missing.")

    if glass.pre_proj > 0 and "pre_projection" in state:
        glass.pre_projection.load_state_dict(state["pre_projection"])

    glass = glass.to(device).eval()

    # # — New: load your ONNX FA model —
    # onnx_session = onnxruntime.InferenceSession(
    #     "/home/pi/glass/mvtec_mms_rpi_FA_simplified_2out.onnx",
    #     providers=["CPUExecutionProvider"]           # or CUDA if built with GPU support
    # )
    # input_names = [inp.name for inp in onnx_session.get_inputs()]

    # 1. Load your model into the IMX500 and get its camera index
    #imx500 = IMX500("/home/pi/glass/network.rpk")
    imx500 = IMX500("/home/pi/glass_imx/glass-main/results/models/backbone_0/mvtec_mms_rpi/converted/network.rpk")

    input_hw = tuple(input_shape[-2:])

    # 2. Set up the camera pipeline using that camera index
    picam2 = Picamera2(imx500.camera_num)
    config = picam2.create_preview_configuration(
       # main   = {"size": (400, 400), "format": "BGR888"},
        main   = {"size": (256, 256), "format": "BGR888"},
        lores  = {"size": (5, 5), "format": "XBGR8888"}
    )
    picam2.configure(config)
    picam2.start()

    # Precompute transform for live frames
    live_tf = transforms.Compose([
        transforms.Resize(input_shape[-2:]),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    print("Starting main capture/inference loop... (press q to exit)")
    last_imx_heatmap = None
    
    CPU_INFERENCE = False   # Set to False to see only IMX NPU results
    IMX_INFERENCE = True  # Set to False to see only CPU GLASS results
    
    imx_fps_alpha = 0.9  # EWMA smoothing factor for IMX FPS
    imx_fps = 0.0
    prev_imx_t = time.time()
    try:
        while True:
            # Grab metadata for IMX NPU outputs
            metadata = picam2.capture_metadata()
            imx_outputs = imx500.get_outputs(metadata)
            now = time.time()
            dt = now - prev_imx_t
            prev_imx_t = now
            if dt > 0:
                inst_fps = 1.0 / dt
                imx_fps = imx_fps_alpha * imx_fps + (1 - imx_fps_alpha) * inst_fps if imx_fps > 0 else inst_fps

            # Grab RGB frame for CPU GLASS pass
            frame = picam2.capture_array("main")  # BGR
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_pil = Image.fromarray(frame_rgb)
            live_img_tensor = live_tf(frame_pil).unsqueeze(0).to(device)

            if CPU_INFERENCE:
                # CPU GLASS inference
                patch_scores = glass(live_img_tensor)
                row_max = torch.amax(patch_scores, dim=2)
                img_score = torch.amax(row_max, dim=1).item()
                mask = convert_to_segmentation(patch_scores, size=input_shape[-1], device=device)[0]
                classification = "GOOD" if img_score < 0.1 else "ANOMALY"
                cpu_label = f"CPU score: {img_score:.3f} ({classification})"

            # IMX score from first output if available
            if imx_outputs:
                imx_arr = np.array(imx_outputs[0])
                imx_score = float(np.max(imx_arr)) if imx_arr.size > 0 else 0.0
            else:
                imx_score = 0.0
            imx_cls = "GOOD" if imx_score < 0.1 else "ANOMALY"
            imx_label = f"IMX max: {imx_score:.3f} ({imx_cls})\nFPS: {imx_fps:5.1f}"

            # Visuals: stack raw frame, CPU heatmap, IMX heatmap side by side in one window
            if CPU_INFERENCE:
                cpu_heatmap = render_heatmap(mask, output_size=input_shape[-2:], label=cpu_label)
            
            imx_heatmap = render_imx_heatmap(imx_outputs, output_size=input_shape[-2:], label=imx_label)

            # Ensure all images are same size and BGR; reuse last IMX heatmap to avoid black flicker
            raw_bgr = cv2.resize(frame, (input_shape[-1], input_shape[-2]))
            if imx_heatmap is None or not imx_heatmap.any():
                if last_imx_heatmap is not None:
                    imx_heatmap = last_imx_heatmap
                else:
                    imx_heatmap = np.zeros_like(raw_bgr)
            else:
                last_imx_heatmap = imx_heatmap
            
            if CPU_INFERENCE and IMX_INFERENCE:
                mosaic = cv2.hconcat([raw_bgr, cpu_heatmap, imx_heatmap])
            elif CPU_INFERENCE:
                mosaic = cv2.hconcat([raw_bgr, cpu_heatmap])
            elif IMX_INFERENCE:
                mosaic = cv2.hconcat([raw_bgr, imx_heatmap])
            else:
                mosaic = raw_bgr
            cv2.imshow("Live view", mosaic)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()