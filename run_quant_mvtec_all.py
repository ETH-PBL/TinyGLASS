"""
Post-training quantization evaluation for TinyGLASS on all available MVTec classes.
Results saved to results/table_mvtec_quantized.csv
"""
import os
import sys
import glob
import csv
import torch
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, average_precision_score

import torch.nn.functional as F
from scipy import ndimage

import backbones
import model_compression_toolkit as mct
from glass import GLASS, IMAGENET_MEAN, IMAGENET_STD
from datasets.mvtec import MVTecDataset, DatasetSplit

# ── CONFIG ────────────────────────────────────────────────────────────────────
DATA_PATH   = "/datasets/pbonazzi/tinyglass_mvtec"
AUG_PATH    = "/datasets/pbonazzi/tinyglass_mvtec/dtd"
CKPT_BASE   = "results/tinyglass_mvtec/models/backbone_0"
RESIZE      = 256
IMAGESIZE   = 256
NUM_WORKERS = 4
VAL_SPLIT   = 0.2   # fraction of test set used as calibration data
N_CALIB     = 10    # number of calibration batches for PTQ
BACKBONE    = "resnet18"
LAYERS      = ("layer2", "layer3")
PRE_DIM     = 384
TGT_DIM     = 384
OUT_CSV     = "results/table_mvtec_quantized.csv"

CLASSES = [
    "bottle", "carpet", "grid", "hazelnut", "leather",
    "metal_nut", "pill", "screw", "tile", "toothbrush",
    "transistor", "wood", "zipper",
]
# ─────────────────────────────────────────────────────────────────────────────


def get_loaders(classname, seed=0):
    full_test = MVTecDataset(
        DATA_PATH, AUG_PATH,
        classname=classname,
        resize=RESIZE, imagesize=IMAGESIZE,
        split=DatasetSplit.TEST, seed=seed,
    )
    test_loader = DataLoader(full_test, batch_size=1,
                             shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    # use a small portion as calibration set
    cal_size = max(1, int(VAL_SPLIT * len(full_test)))
    cal_ds, _ = random_split(
        full_test, [cal_size, len(full_test) - cal_size],
        generator=torch.Generator().manual_seed(seed),
    )
    cal_loader = DataLoader(cal_ds, batch_size=1,
                            shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    return test_loader, cal_loader


def load_checkpoint(glass, ckpt_path, device):
    state = torch.load(ckpt_path, map_location=device)
    if "discriminator" in state:
        d_state = state["discriminator"]
        if "tail.0.weight" in d_state and d_state["tail.0.weight"].ndim == 2:
            d_state["tail.0.weight"] = d_state["tail.0.weight"].unsqueeze(-1).unsqueeze(-1)
        if "body.block1.0.weight" in d_state and d_state["body.block1.0.weight"].ndim == 2:
            d_state["body.block1.0.weight"] = d_state["body.block1.0.weight"].unsqueeze(-1).unsqueeze(-1)
        glass.discriminator.load_state_dict(d_state, strict=False)
    if glass.pre_proj > 0 and "pre_projection" in state:
        p_state = state["pre_projection"]
        if "layers.0fc.weight" in p_state and p_state["layers.0fc.weight"].ndim == 2:
            p_state["layers.0fc.weight"] = p_state["layers.0fc.weight"].unsqueeze(-1).unsqueeze(-1)
        try:
            glass.pre_projection.load_state_dict(p_state, strict=False)
        except RuntimeError as e:
            print(f"  [warn] pre_projection load skipped: {e}")
    return glass


def build_glass(device):
    backbone = backbones.load(BACKBONE).to(device)
    glass = GLASS(device)
    glass.load(
        backbone=backbone,
        layers_to_extract_from=LAYERS,
        device=device,
        input_shape=(3, IMAGESIZE, IMAGESIZE),
        pretrain_embed_dimension=PRE_DIM,
        target_embed_dimension=TGT_DIM,
        patchsize=3,
        patchstride=1,
        skip_backbone=False,
    )
    glass.static_patch_shapes = [(32, 32), (16, 16)]
    return glass


def upsample_scores(patch_scores, target_size, smoothing=4):
    """Bilinear upsample [B,H,W] patch scores to target_size, then Gaussian smooth."""
    t = torch.from_numpy(patch_scores).unsqueeze(1).float()
    t = F.interpolate(t, size=target_size, mode="bilinear", align_corners=False)
    t = t.squeeze(1).numpy()
    return np.stack([ndimage.gaussian_filter(s, sigma=smoothing) for s in t])


def evaluate(model, loader, device):
    model.to(device).eval()
    all_img_scores, all_img_labels = [], []
    all_seg_maps, all_masks_gt = [], []

    with torch.no_grad():
        for data in tqdm(loader, desc="  eval", leave=False):
            imgs   = data.get("image", data.get("images"))
            labels = data.get("label", data.get("labels"))
            masks_gt = data.get("mask_gt")          # [B,1,H,W] or None
            if labels is None:
                paths = data.get("image_path", [])
                cls_names = [os.path.normpath(p).split(os.sep)[-2] for p in paths]
                labels = torch.tensor([0 if c == "good" else 1 for c in cls_names])

            imgs = imgs.to(device)
            out  = model(imgs)
            if isinstance(out, (tuple, list)):
                out = out[0]
            if out.dim() == 4:
                out = out.squeeze(1)            # → [B,H,W]

            # image-level score = max over spatial dims
            img_scores = out.amax(dim=(1, 2)).detach().cpu().numpy()
            patch_np   = out.detach().cpu().numpy()   # [B,H,W]

            all_img_scores.append(img_scores)
            all_img_labels.append(labels.reshape(-1).numpy())

            # pixel-level: upsample patch scores to full image size
            target_size = imgs.shape[-2:]           # (H_img, W_img)
            seg_maps = upsample_scores(patch_np, target_size)  # [B,H_img,W_img]
            all_seg_maps.append(seg_maps)
            if masks_gt is not None:
                all_masks_gt.append(masks_gt.squeeze(1).numpy())  # [B,H,W]

    all_img_scores = np.concatenate(all_img_scores)
    all_img_labels = np.concatenate(all_img_labels)
    img_auroc = roc_auc_score(all_img_labels, all_img_scores)
    img_ap    = average_precision_score(all_img_labels, all_img_scores)

    pix_auroc = None
    if all_masks_gt:
        seg_flat  = np.concatenate(all_seg_maps).ravel()
        mask_flat = np.concatenate(all_masks_gt).ravel().astype(int)
        if mask_flat.max() > 0:   # at least one positive pixel
            pix_auroc = roc_auc_score(mask_flat, seg_flat)

    return img_auroc, img_ap, pix_auroc


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    os.makedirs("results", exist_ok=True)
    rows = []

    for classname in CLASSES:
        ckpt_dir = os.path.join(CKPT_BASE, f"mvtec_{classname}")
        ckpts = sorted(glob.glob(os.path.join(ckpt_dir, "ckpt_best_*.pth")))
        if not ckpts:
            print(f"[SKIP] {classname}: no checkpoint found in {ckpt_dir}")
            continue
        ckpt_path = ckpts[-1]
        print(f"\n=== {classname}  (ckpt: {os.path.basename(ckpt_path)}) ===")

        # build & load model
        glass = build_glass(device)
        glass = load_checkpoint(glass, ckpt_path, device)
        if hasattr(glass.discriminator, "_sync_conv_equivalents"):
            glass.discriminator._sync_conv_equivalents()
        if glass.pre_proj > 0 and hasattr(glass.pre_projection, "_sync_conv_equivalents"):
            glass.pre_projection._sync_conv_equivalents()
        glass = glass.to(device).eval()
        glass.trace_mode = False

        test_loader, cal_loader = get_loaders(classname)

        # baseline (fp32)
        print("  [FP32] evaluating...")
        fp32_img_auroc, fp32_img_ap, fp32_pix_auroc = evaluate(glass, test_loader, device)
        pix_str = f"{fp32_pix_auroc:.4f}" if fp32_pix_auroc is not None else "N/A"
        print(f"  FP32  Img AUROC: {fp32_img_auroc:.4f}  Pix AUROC: {pix_str}")

        # calibration dataset generator for MCT
        def representative_dataset_gen():
            it = iter(cal_loader)
            for _ in range(N_CALIB):
                try:
                    batch = next(it)
                except StopIteration:
                    break
                img = batch[0] if not isinstance(batch, dict) else batch.get("image", next(iter(batch.values())))
                if img.dim() > 4:
                    img = img.squeeze(0)
                if img.shape[0] > 1:
                    img = img[:1]
                yield [img]

        # post-training quantization (IMX500 target)
        print("  [PTQ] quantizing...")
        tp = mct.get_target_platform_capabilities('pytorch', 'imx500', target_platform_version='v1')
        q_model, _ = mct.ptq.pytorch_post_training_quantization(
            in_module=glass,
            representative_data_gen=representative_dataset_gen,
            target_platform_capabilities=tp,
        )

        # quantized evaluation
        print("  [INT8] evaluating...")
        int8_img_auroc, int8_img_ap, int8_pix_auroc = evaluate(q_model, test_loader, device)
        pix_str = f"{int8_pix_auroc:.4f}" if int8_pix_auroc is not None else "N/A"
        pix_drop = (fp32_pix_auroc - int8_pix_auroc) * 100 if (fp32_pix_auroc and int8_pix_auroc) else None
        print(f"  INT8  Img AUROC: {int8_img_auroc:.4f} ({(fp32_img_auroc-int8_img_auroc)*100:+.2f}pp)  "
              f"Pix AUROC: {pix_str} ({pix_drop:+.2f}pp)" if pix_drop is not None
              else f"  INT8  Img AUROC: {int8_img_auroc:.4f} ({(fp32_img_auroc-int8_img_auroc)*100:+.2f}pp)  Pix AUROC: {pix_str}")

        rows.append({
            "class":            classname,
            "fp32_img_auroc":   round(fp32_img_auroc * 100, 2),
            "fp32_img_ap":      round(fp32_img_ap * 100, 2),
            "fp32_pix_auroc":   round(fp32_pix_auroc * 100, 2) if fp32_pix_auroc else "-",
            "int8_img_auroc":   round(int8_img_auroc * 100, 2),
            "int8_img_ap":      round(int8_img_ap * 100, 2),
            "int8_pix_auroc":   round(int8_pix_auroc * 100, 2) if int8_pix_auroc else "-",
            "img_auroc_drop":   round((fp32_img_auroc - int8_img_auroc) * 100, 2),
            "pix_auroc_drop":   round(pix_drop, 2) if pix_drop is not None else "-",
        })

        # free memory between classes
        del glass, q_model
        torch.cuda.empty_cache()

    # compute mean (only over numeric rows)
    if rows:
        num_fields = ["fp32_img_auroc", "fp32_img_ap", "fp32_pix_auroc",
                      "int8_img_auroc", "int8_img_ap", "int8_pix_auroc",
                      "img_auroc_drop", "pix_auroc_drop"]
        mean_row = {"class": "Mean"}
        for k in num_fields:
            vals = [r[k] for r in rows if isinstance(r[k], (int, float))]
            mean_row[k] = round(sum(vals) / len(vals), 2) if vals else "-"
        rows.append(mean_row)

    # save CSV
    fields = ["class",
              "fp32_img_auroc", "fp32_img_ap", "fp32_pix_auroc",
              "int8_img_auroc", "int8_img_ap", "int8_pix_auroc",
              "img_auroc_drop", "pix_auroc_drop"]
    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nResults saved to {OUT_CSV}")

    # print table
    print(f"\n{'Class':<15} {'FP32 ImgAUC':>11} {'FP32 PixAUC':>11} {'INT8 ImgAUC':>11} {'INT8 PixAUC':>11} {'ImgDrop':>8} {'PixDrop':>8}")
    print("-" * 80)
    for r in rows:
        print(f"{r['class']:<15} {str(r['fp32_img_auroc']):>11} {str(r['fp32_pix_auroc']):>11} "
              f"{str(r['int8_img_auroc']):>11} {str(r['int8_pix_auroc']):>11} "
              f"{str(r['img_auroc_drop']):>8} {str(r['pix_auroc_drop']):>8}")


if __name__ == "__main__":
    main()
