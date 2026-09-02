"""
Lean MCT post-training quantization for the TinyGLASS mms_rpi checkpoint.

Reuses the exact GLASS setup from mct_pt_full.py (resnet18, layer2/3,
target dim forced to 128, pre_proj disabled) but feeds MCT a simple
representative generator that reads raw images with the standard
ImageNet preprocessing — no MVTecDataset / DTD / masks needed.

Output: <base_dir>/qmodel.onnx  (quantized ONNX ready for imxconv-pt)

Run inside the `sony_env` conda env (Python 3.11, MCT 2.6, torch 2.6 CPU):
    python tools/quantize_mms.py \
        --ckpt checkpoints/models/backbone_0/mvtec_mms_rpi/ckpt_best_334.pth \
        --calib-dir /tmp/mmsdata/.../mms_rpi/test \
        --out checkpoints/models/backbone_0/mvtec_mms_rpi/qmodel.onnx \
        --n-calib 20
"""
import os
import sys
import glob
import argparse

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from PIL import Image
from torchvision import transforms

import backbones
from glass import GLASS, IMAGENET_MEAN, IMAGENET_STD
import model_compression_toolkit as mct

INPUT_SHAPE = (3, 256, 256)
BACKBONE = "resnet18"
LAYERS = ("layer2", "layer3")
PATCH_SHAPES = [(32, 32), (16, 16)]

PREPROCESS = transforms.Compose([
    transforms.Resize(INPUT_SHAPE[-2:]),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


def build_glass(device):
    backbone = backbones.load(BACKBONE).to(device)
    glass = GLASS(device)
    glass.load(
        backbone=backbone,
        layers_to_extract_from=LAYERS,
        device=device,
        input_shape=INPUT_SHAPE,
        pretrain_embed_dimension=384,   # internally forced to 128 by glass.py
        target_embed_dimension=384,
        patchsize=3,
        patchstride=1,
        skip_backbone=False,
    )
    glass.static_patch_shapes = PATCH_SHAPES
    glass.trace_mode = False
    return glass


def load_discriminator(glass, ckpt_path, device):
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    d_state = state["discriminator"]
    for key in ("tail.0.weight", "body.block1.0.weight"):
        if key in d_state and d_state[key].ndim == 2:
            d_state[key] = d_state[key].unsqueeze(-1).unsqueeze(-1)
    glass.discriminator.load_state_dict(d_state, strict=False)
    if hasattr(glass.discriminator, "_sync_conv_equivalents"):
        glass.discriminator._sync_conv_equivalents()
    return glass.to(device).eval()


def collect_images(calib_dir, n):
    paths = sorted(
        glob.glob(os.path.join(calib_dir, "**", "*.png"), recursive=True)
        + glob.glob(os.path.join(calib_dir, "**", "*.jpg"), recursive=True)
        + glob.glob(os.path.join(calib_dir, "**", "*.JPG"), recursive=True)
    )
    if not paths:
        sys.exit(f"No images found under {calib_dir}")
    # Evenly sample across the set so calibration sees varied content.
    if len(paths) > n:
        step = len(paths) / n
        paths = [paths[int(i * step)] for i in range(n)]
    print(f"Calibrating on {len(paths)} images from {calib_dir}")
    return paths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--calib-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-calib", type=int, default=20)
    args = ap.parse_args()

    device = torch.device("cpu")
    glass = load_discriminator(build_glass(device), args.ckpt, device)
    print(f"Loaded discriminator from {args.ckpt}")

    img_paths = collect_images(args.calib_dir, args.n_calib)
    tensors = [PREPROCESS(Image.open(p).convert("RGB")).unsqueeze(0) for p in img_paths]

    # sanity: run one forward through the fp32 model
    with torch.no_grad():
        out = glass(tensors[0].to(device))
    print(f"fp32 forward OK — patch-score map shape {tuple(out.shape)}")

    def representative_dataset_gen():
        for t in tensors:
            yield [t]

    tp = mct.get_target_platform_capabilities(tpc_version="1.0", device_type="imx500")
    q_model, q_info = mct.ptq.pytorch_post_training_quantization(
        in_module=glass,
        representative_data_gen=representative_dataset_gen,
        target_platform_capabilities=tp,
    )
    print("Quantization complete.")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    mct.exporter.pytorch_export_model(
        q_model,
        save_model_path=args.out,
        repr_dataset=representative_dataset_gen,
    )
    print(f"✅ Saved quantized ONNX: {args.out}")


if __name__ == "__main__":
    main()
