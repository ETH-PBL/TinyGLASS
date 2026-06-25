"""
TinyGLASS demo for Sony IMX500.

Two modes:
  image  — run inference on one or more image files and save heatmaps
  live   — live camera loop using the IMX500 RPK on a Raspberry Pi

Usage examples:
  # Single image (any machine with a checkpoint):
  uv run python demo_imx500.py image \
      --checkpoint results/tinyglass_mms/models/backbone_0/mvtec_mms_rpi/ckpt_best_1.pth \
      --input path/to/image.png

  # Live IMX500 camera (Raspberry Pi only):
  uv run python demo_imx500.py live \
      --rpk results/tinyglass_mms/models/backbone_0/mvtec_mms_rpi/converted/network.rpk \
      --checkpoint results/tinyglass_mms/models/backbone_0/mvtec_mms_rpi/ckpt_best_1.pth

  # Download checkpoint from HuggingFace first:
  python -c "
  from huggingface_hub import hf_hub_download
  hf_hub_download('pietrobonazzi/TinyGLASS',
      'models/backbone_0/mvtec_mms_rpi/ckpt_best_1.pth', local_dir='checkpoints')
  "
"""

import argparse
import glob
import os
import sys

import cv2
import numpy as np
import scipy.ndimage as ndimage
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

import backbones
from glass import GLASS, IMAGENET_MEAN, IMAGENET_STD

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
INPUT_SHAPE = (3, 256, 256)
PRE_DIM = 384
TGT_DIM = 384
BACKBONE = "resnet18"
LAYERS = ("layer2", "layer3")
PATCH_SHAPES = [(32, 32), (16, 16)]   # layer2 / layer3 for 256×256 input


# ---------------------------------------------------------------------------
# Model helpers
# ---------------------------------------------------------------------------

def build_glass(device: torch.device) -> GLASS:
    backbone = backbones.load(BACKBONE).to(device)
    glass = GLASS(device)
    glass.load(
        backbone=backbone,
        layers_to_extract_from=LAYERS,
        device=device,
        input_shape=INPUT_SHAPE,
        pretrain_embed_dimension=PRE_DIM,
        target_embed_dimension=TGT_DIM,
        patchsize=3,
        patchstride=1,
        skip_backbone=False,
    )
    glass.static_patch_shapes = PATCH_SHAPES
    glass.trace_mode = False
    return glass


def load_checkpoint(glass: GLASS, ckpt_path: str, device: torch.device) -> GLASS:
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    if "discriminator" not in state:
        raise RuntimeError(f"No discriminator key in checkpoint: {ckpt_path}")
    d_state = state["discriminator"]
    # Reshape Linear→Conv1×1 if saved from an older export
    for key in ("tail.0.weight", "body.block1.0.weight"):
        if key in d_state and d_state[key].ndim == 2:
            d_state[key] = d_state[key].unsqueeze(-1).unsqueeze(-1)
    glass.discriminator.load_state_dict(d_state, strict=False)
    if glass.pre_proj > 0 and "pre_projection" in state:
        glass.pre_projection.load_state_dict(state["pre_projection"])
    return glass.to(device).eval()


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------

PREPROCESS = transforms.Compose([
    transforms.Resize(INPUT_SHAPE[-2:]),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


def infer(glass: GLASS, img_tensor: torch.Tensor, device: torch.device):
    """Return (image_score, segmentation_mask HxW numpy)."""
    with torch.no_grad():
        patch_scores = glass(img_tensor.to(device))          # [1, H, W]
        img_score = float(torch.amax(patch_scores).item())
        seg = F.interpolate(
            patch_scores.unsqueeze(1),
            size=INPUT_SHAPE[-2:],
            mode="bilinear",
            align_corners=False,
        ).squeeze().cpu().numpy()
    seg = ndimage.gaussian_filter(seg, sigma=4)
    return img_score, seg


def build_heatmap(orig_bgr: np.ndarray, seg: np.ndarray, score: float,
                  threshold: float) -> np.ndarray:
    """Overlay JET heatmap on the original BGR frame and add score text."""
    h, w = orig_bgr.shape[:2]
    seg_norm = cv2.normalize(seg, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    seg_resized = cv2.resize(seg_norm, (w, h), interpolation=cv2.INTER_LINEAR)
    heatmap = cv2.applyColorMap(seg_resized, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(orig_bgr, 0.5, heatmap, 0.5, 0)
    label = "ANOMALY" if score >= threshold else "GOOD"
    color = (0, 0, 220) if score >= threshold else (0, 180, 0)
    cv2.putText(overlay, f"{label}  score={score:.3f}",
                (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
    return overlay


# ---------------------------------------------------------------------------
# Image mode
# ---------------------------------------------------------------------------

def run_image(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    glass = load_checkpoint(build_glass(device), args.checkpoint, device)
    print(f"Loaded checkpoint: {args.checkpoint}")
    print(f"Device: {device}  |  Threshold: {args.threshold}")

    # Collect inputs (single file or glob)
    paths = sorted(glob.glob(args.input)) if "*" in args.input else [args.input]
    if not paths:
        sys.exit(f"No files found: {args.input}")

    os.makedirs(args.output_dir, exist_ok=True)
    for img_path in paths:
        img_pil = Image.open(img_path).convert("RGB")
        img_tensor = PREPROCESS(img_pil).unsqueeze(0)
        score, seg = infer(glass, img_tensor, device)

        orig_bgr = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        orig_bgr = cv2.resize(orig_bgr, INPUT_SHAPE[-2:][::-1])
        overlay = build_heatmap(orig_bgr, seg, score, args.threshold)

        stem = os.path.splitext(os.path.basename(img_path))[0]
        out_path = os.path.join(args.output_dir, f"{stem}_heatmap.png")
        cv2.imwrite(out_path, overlay)
        status = "ANOMALY" if score >= args.threshold else "GOOD"
        print(f"  {img_path}  →  score={score:.4f}  [{status}]  saved {out_path}")


# ---------------------------------------------------------------------------
# Live IMX500 mode (Raspberry Pi only)
# ---------------------------------------------------------------------------

def run_live(args):
    try:
        from picamera2 import Picamera2
        from picamera2.devices import IMX500
    except ImportError:
        sys.exit("picamera2 not found. Live mode requires a Raspberry Pi with picamera2 installed.")

    device = torch.device("cpu")   # RPi runs on CPU
    glass = load_checkpoint(build_glass(device), args.checkpoint, device)
    print(f"Loaded checkpoint: {args.checkpoint}")

    imx500 = IMX500(args.rpk)
    picam2 = Picamera2(imx500.camera_num)
    config = picam2.create_preview_configuration(
        main={"size": INPUT_SHAPE[-2:][::-1], "format": "BGR888"},
        lores={"size": (5, 5), "format": "XBGR8888"},
    )
    picam2.configure(config)
    picam2.start()

    imx_last = None
    print("Live view started — press q to quit.")
    try:
        while True:
            metadata = picam2.capture_metadata()
            imx_outputs = imx500.get_outputs(metadata)

            frame_bgr = picam2.capture_array("main")   # already BGR 256×256
            frame_pil = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
            img_tensor = PREPROCESS(frame_pil).unsqueeze(0)

            # ---- CPU GLASS inference ----
            score, seg = infer(glass, img_tensor, device)
            cpu_overlay = build_heatmap(frame_bgr.copy(), seg, score, args.threshold)

            # ---- IMX500 NPU output ----
            if imx_outputs and imx_outputs[0] is not None:
                imx_arr = np.array(imx_outputs[0])
                imx_score = float(np.max(imx_arr))
                # Treat NPU output as flat patch scores → reshape to grid
                side = int(np.ceil(np.sqrt(imx_arr.size)))
                padded = np.zeros(side * side, dtype=np.float32)
                padded[: imx_arr.size] = imx_arr.ravel()
                imx_seg = padded.reshape(side, side)
                imx_overlay = build_heatmap(frame_bgr.copy(), imx_seg, imx_score, args.threshold)
                imx_last = imx_overlay
            else:
                imx_overlay = imx_last if imx_last is not None else np.zeros_like(frame_bgr)

            mosaic = cv2.hconcat([frame_bgr, cpu_overlay, imx_overlay])
            cv2.putText(mosaic, "Raw | CPU GLASS | IMX500 NPU",
                        (10, mosaic.shape[0] - 8), cv2.FONT_HERSHEY_SIMPLEX,
                        0.45, (200, 200, 200), 1, cv2.LINE_AA)
            cv2.imshow("TinyGLASS - IMX500 Demo", mosaic)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cv2.destroyAllWindows()
        picam2.stop()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="TinyGLASS IMX500 demo")
    sub = p.add_subparsers(dest="mode", required=True)

    # ---- image mode ----
    img_p = sub.add_parser("image", help="Run inference on image file(s)")
    img_p.add_argument("--checkpoint", required=True,
                       help="Path to ckpt_best_*.pth")
    img_p.add_argument("--input", required=True,
                       help="Image path or glob (e.g. 'test/crack_hole/*.png')")
    img_p.add_argument("--output-dir", default="results/demo",
                       help="Where to save heatmap images (default: results/demo)")
    img_p.add_argument("--threshold", type=float, default=0.5,
                       help="Anomaly score threshold (default: 0.5)")

    # ---- live mode ----
    live_p = sub.add_parser("live", help="Live IMX500 camera demo (Raspberry Pi)")
    live_p.add_argument("--rpk", required=True,
                        help="Path to converted network.rpk")
    live_p.add_argument("--checkpoint", required=True,
                        help="Path to ckpt_best_*.pth for CPU fallback")
    live_p.add_argument("--threshold", type=float, default=0.5,
                        help="Anomaly score threshold (default: 0.5)")

    return p.parse_args()


def main():
    args = parse_args()
    if args.mode == "image":
        run_image(args)
    else:
        run_live(args)


if __name__ == "__main__":
    main()
