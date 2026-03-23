import os
import glob
import torch
import argparse
from PIL import Image
from torch.utils.data import DataLoader, random_split
from datasets.mvtec import MVTecDataset, DatasetSplit
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
from torchvision import transforms
from glass import GLASS, IMAGENET_MEAN, IMAGENET_STD
import torch.nn.functional as F
import scipy.ndimage as ndimage

import backbones
import model_compression_toolkit as mct
from tqdm import tqdm
import traceback

import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score   
import hashlib


DATA_PATH        = "/root/glass_imx/datasets"
AUG_PATH         = "/root/datasets/dtd"
IMAGE_PATH       = "/root/glass_imx/datasets/mms_rpi/test/crack_hole/IMG_0046.png"
IMAGE_PATH_def1  = "/root/glass_imx/datasets/mms_rpi_select/test/crack_hole/IMG_0038.png"
IMAGE_PATH_norm  = "/root/glass_imx/datasets/mms_rpi_select/test/good/IMG_0009.png"
IMAGE_PATH_def2  = "/root/glass_imx/datasets/mms_rpi_select/test/half/IMG_0003.png"
SUBDATASETS      = ["mms"] 
MODEL_FOLDER     = "mvtec_mms_rpi"    
RESIZE           = 256
IMAGESIZE        = 256
NUM_WORKERS      = 4
INPUT_SIZE       = (3, 256, 256)
VAL_SPLIT_RATIO  = 0.2  # portion of test set used for
PRE_DIM          = 384
TGT_DIM          = 384
BACKBONE         = "resnet18"
BATCH_SIZE       = 1


import matplotlib.pyplot as plt
import cv2
import numpy as np

def save_heatmap(image_path, patch_scores, output_path="heatmap_output.png", alpha=0.5):
    """
    Create and save an anomaly heatmap overlay on the original image.
    
    Args:
        image_path: Path to original image
        patch_scores: Tensor of shape [H, W] with anomaly scores per patch
        output_path: Where to save the result
        alpha: Transparency of heatmap overlay (0=transparent, 1=opaque)
    """
    # Load original image
    original_img = cv2.imread(image_path)
    original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
    h, w = original_img.shape[:2]
    
    # Convert patch scores to numpy and normalize to [0, 1]
    if isinstance(patch_scores, torch.Tensor):
        scores_np = patch_scores.squeeze().cpu().numpy()
    else:
        scores_np = np.array(patch_scores).squeeze()
    
    # Clamp to [0,1] without per-image normalization so absolute score magnitude is preserved
    scores_normalized = np.clip(scores_np*2, 0.0, 1.0)
    scores_max = scores_np.max()
    
    # Resize heatmap to match original image size
    heatmap = cv2.resize(scores_normalized, (w, h), interpolation=cv2.INTER_LINEAR)
    
    # Convert to uint8 and apply colormap
    heatmap_uint8 = (heatmap * 255).astype(np.uint8)
    heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    
    # Create overlay
    overlay = cv2.addWeighted(original_img, 1 - alpha, heatmap_colored, alpha, 0)
    
    # Create figure with subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original image
    axes[0].imshow(original_img)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # Heatmap only
    im = axes[1].imshow(heatmap, cmap='jet', vmin=0, vmax=1)
    axes[1].set_title(f'Anomaly Heatmap\n(max score: {scores_max:.4f})')
    axes[1].axis('off')
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
    
    # Overlay
    axes[2].imshow(overlay)
    axes[2].set_title('Overlay')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"   Heatmap saved to: {output_path}")
    print(f"   Score range: [{scores_np.min():.4f}, {scores_max:.4f}]")
    
    return heatmap, overlay



def convert_to_segmentation(self, patch_scores, smoothing=4, size=288, device='cpu'):
    with torch.no_grad():
        if isinstance(patch_scores, np.ndarray):
            patch_scores = torch.from_numpy(patch_scores)
        _scores = patch_scores.to(device)
        _scores = _scores.unsqueeze(1)
        _scores = F.interpolate(
            _scores, size=288, mode="bilinear", align_corners=False
        )
        _scores = _scores.squeeze(1)
        patch_scores = _scores.cpu().numpy()
    return [ndimage.gaussian_filter(patch_score, sigma=smoothing) for patch_score in patch_scores]

def tensor_checksum(tensor):
    """Compute MD5 checksum of a tensor."""
    if tensor is None:
        return "None"
    return hashlib.md5(tensor.detach().cpu().numpy().tobytes()).hexdigest()[:16]


device = 'cuda' if torch.cuda.is_available() else 'cpu'
base_dir = os.path.join("results","models","backbone_0", MODEL_FOLDER)
ckpts = sorted(glob.glob(os.path.join(base_dir, "ckpt_best_*.pth")))
assert ckpts, f"No checkpoints in {base_dir}"
ckpt_path = ckpts[-1]


#define a full glass model
device          = torch.device("cuda" if torch.cuda.is_available() else "cpu")
backbone_name   = "resnet18"
layers_to_use   = ("layer2", "layer3")
input_shape     = (3, 256, 256)
pretrain_embed_dimension = 384
target_embed_dimension = 384
backbone = backbones.load(backbone_name)
backbone = backbone.to(device)
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

# Hardcode patch grid sizes (layer2, layer3) for FX/MCT tracing
# For input 256: layer2 grid is 32x32, layer3 grid is 16x16 (ResNet18).
glass.static_patch_shapes = [(32, 32), (16, 16)]

glass.trace_mode = False

# 3) load checkpoint weights with Linear->Conv1d reshaping
state = torch.load(ckpt_path, map_location=device)

if "discriminator" in state:
    d_state = state["discriminator"]
    if "tail.0.weight" in d_state and d_state["tail.0.weight"].ndim == 2:
        d_state["tail.0.weight"] = d_state["tail.0.weight"].unsqueeze(-1).unsqueeze(-1)
    if "body.block1.0.weight" in d_state and d_state["body.block1.0.weight"].ndim == 2:
        d_state["body.block1.0.weight"] = d_state["body.block1.0.weight"].unsqueeze(-1).unsqueeze(-1)
    glass.discriminator.load_state_dict(d_state, strict=False)
else:
    raise RuntimeError("Discriminator weights missing.")

# Sync conv-equivalent weights once (avoid in-place copy_ during FX tracing).
# if hasattr(glass.discriminator, "_sync_conv_equivalents"):
#     glass.discriminator._sync_conv_equivalents()
# if glass.pre_proj > 0 and hasattr(glass.pre_projection, "_sync_conv_equivalents"):
#     glass.pre_projection._sync_conv_equivalents()

glass = glass.to(device).eval()

glass.print_stats()

tf = transforms.Compose([
    transforms.Resize(input_shape[-2:]),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

img = Image.open(IMAGE_PATH).convert("RGB")
img_tensor = tf(img).unsqueeze(0).to(device)  # (1,C,H,W) 

patch_scores = glass(img_tensor)

row_max = torch.amax(patch_scores, dim=2)  # [B, H]
score = torch.amax(row_max, dim=1)  # [B]

mask = convert_to_segmentation(glass, patch_scores, device=device)

# Save heatmap (output is patch scores with shape [H, W])
heatmap, overlay = save_heatmap(
    image_path=IMAGE_PATH, 
    patch_scores=mask,
    output_path="results/anomaly_heatmap.png",
    alpha=0.5
)

print(f"IMG score from forward pass: {score.item():.6f}")



img = Image.open(IMAGE_PATH_def1).convert("RGB")
img_tensor = tf(img).unsqueeze(0).to(device)  # (1,C,H,W) 
patch_scores = glass(img_tensor)
print(f"          Model output hash: {tensor_checksum(patch_scores)}")
row_max = torch.amax(patch_scores, dim=2)  # [B, H]
score = torch.amax(row_max, dim=1)  # [B]
print(f"IMG score from forward pass: {score.item():.6f}")


img = Image.open(IMAGE_PATH_norm).convert("RGB")
img_tensor = tf(img).unsqueeze(0).to(device)  # (1,C,H,W) 
patch_scores = glass(img_tensor)
print(f"          Model output hash: {tensor_checksum(patch_scores)}")
row_max = torch.amax(patch_scores, dim=2)  # [B, H]
score = torch.amax(row_max, dim=1)  # [B]
print(f"IMG score from forward pass: {score.item():.6f}")


img = Image.open(IMAGE_PATH_def2).convert("RGB")
img_tensor = tf(img).unsqueeze(0).to(device)  # (1,C,H,W) 
patch_scores = glass(img_tensor)
print(f"          Model output hash: {tensor_checksum(patch_scores)}")
row_max = torch.amax(patch_scores, dim=2)  # [B, H]
score = torch.amax(row_max, dim=1)  # [B]
print(f"IMG score from forward pass: {score.item():.6f}")

