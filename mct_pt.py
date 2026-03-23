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

import backbones
import model_compression_toolkit as mct
from tqdm import tqdm

import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score   # ← new

# ── CONFIGURE THESE TO MATCH YOUR SETUP ───────────────────────────────────────
DATA_PATH        = "/root/datasets/mms_mvtec_stretch"
AUG_PATH         = "/root/datasets/dtd"
SUBDATASETS      = ["mms"]
RESIZE           = 288
IMAGESIZE        = 288
VAL_SPLIT_RATIO  = 0.4
NUM_WORKERS      = 4
# ──────────────────────────────────────────────────────────────────────────────

def get_dataloaders(seed, batch_size):
    dataloaders = []
    for classname in SUBDATASETS:
        full_test = MVTecDataset(
            DATA_PATH, AUG_PATH,
            classname=classname,
            resize=RESIZE, imagesize=IMAGESIZE,
            split=DatasetSplit.TEST, seed=seed
        )
        if VAL_SPLIT_RATIO is not None:
            val_size  = int(VAL_SPLIT_RATIO * len(full_test))
            test_size = len(full_test) - val_size
            #test_ds, val_ds = random_split(
            #    full_test, [test_size, val_size],
            #    generator=torch.Generator().manual_seed(seed)
            #)
            test_ds = full_test
            val_ds  = full_test
        else:
            test_ds = val_ds = full_test
        test_loader = DataLoader(test_ds, batch_size=batch_size,
                                 shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
        val_loader  = DataLoader(val_ds, batch_size=batch_size,
                                 shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
        test_loader.name = f"mvtec_{classname}"
        val_loader.name  = test_loader.name + "_val"
        dataloaders.append({
            "training": val_loader, #use the validation set for quantization, because training is good-only
            "validation": val_loader,
            "testing":    test_loader,
        })
        #print the set sizes
        print(f"Dataset {classname}: {len(test_ds)} test, {len(val_ds)} val")
    return dataloaders

def load_image(img_path, device, input_size):
    tf = transforms.Compose([
        transforms.Resize(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    img = Image.open(img_path).convert("RGB")
    return tf(img).unsqueeze(0).to(device)  # (1,C,H,W)

    # helper to count and print parameters
def print_param_counts(mod, name):
    total = sum(p.numel() for p in mod.parameters())
    print(f"{name} total parameters: {total:,}")

def evaluate(model, loader, threshold: float = 0.0):
    """
    model: takes a batch of images and returns per-patch scores
    loader: yields (imgs, labels)
    threshold: cutoff on the image-level max score to call 'anomaly'
    returns: (acc, auroc, ap)
    """
    image_path      = "/root/datasets/mms_mvtec_stretch/mms/test/good/IMG_0009.png"
    checkpoint_path = "/root/rafeal_sutter_564/glass-main/results/models/backbone_0/mvtec_mms/ckpt.pth"
    device          = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backbone_name   = "resnet18"
    layers_to_use   = ["layer2", "layer3"]
    input_shape     = (3, 288, 288)
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
        skip_backbone=True,
    )
    # 3) load checkpoint weights
    state = torch.load(checkpoint_path, map_location=device)
    if "discriminator" in state:
        glass.discriminator.load_state_dict(state["discriminator"])
    if glass.pre_proj > 0 and "pre_projection" in state:
        glass.pre_projection.load_state_dict(state["pre_projection"])
    glass = glass.to(device).eval()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    
    print_param_counts(glass, "GLASS model")


    all_scores = []
    all_labels = []
    
    img_tensor = load_image(image_path, device, input_shape[-2:])


    with torch.no_grad():
        for data in tqdm(loader, desc="Evaluating", leave=False):
            # unpack
            if isinstance(data, dict):
                imgs = data.get("image", data.get("images"))
                labels = data.get("label", data.get("labels"))
                # if no labels key, infer from image_path
                if labels is None:
                    paths = data.get("image_path", [])
                    class_names = [os.path.normpath(p).split(os.sep)[-2] for p in paths]
                    labels = torch.tensor([0 if cn == "good" else 1 for cn in class_names], device=device)
            else:
                imgs, labels, *rest = data

            imgs  = imgs.to(device)
            labels = labels.to(device)
            
            #print dimension of imgs
            #print(f"imgs type: {imgs.dtype}, imgs shape: {imgs.shape}")
            #print(f"img_tensor type: {img_tensor.dtype}, img_tensor shape: {img_tensor.shape}")
            #print(f"img_path")

            # forward → per-patch scores
            feat = model(imgs)   # torch.Tensor [B, P] or [B*P] or [B,P,1]
            #feat = model(img_tensor)
            
            if isinstance(feat, list):
                feat = dict(zip(layers_to_use, feat))
                            
            out, _ = glass(feat)
                        
            # image‐level score = max patch score
            img_scores = out.squeeze()
            

            all_scores.append(img_scores.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    all_scores = np.array(all_scores).ravel()
    all_labels = np.array(all_labels).ravel()
    
    # metrics
    auroc = roc_auc_score(all_labels, all_scores)
    ap    = average_precision_score(all_labels, all_scores)
    preds = (all_scores > threshold).astype(int)
    acc   = (preds == all_labels).mean() * 100

    return acc, auroc, ap



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, required=True,
                        help="Subfolder in results/models/backbone_0/")
    parser.add_argument('--backbone',  type=str, default="wideresnet50")
    parser.add_argument('--layers',    nargs='+', default=["layer2","layer3"])
    parser.add_argument('--input_size',nargs=3, type=int, default=[3,288,288])
    parser.add_argument('--pre_dim',   type=int, default=1536)
    parser.add_argument('--tgt_dim',   type=int, default=1536)
    parser.add_argument('--batch_size',type=int, default=16)
    parser.add_argument('--n_iter',    type=int, default=10)
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    base_dir = os.path.join("results","models","backbone_0", args.model_dir)
    ckpts = sorted(glob.glob(os.path.join(base_dir, "ckpt_best_*.pth")))
    assert ckpts, f"No checkpoints in {base_dir}"
    ckpt_path = ckpts[-1]

    import classifier
    resnet18truncated = classifier.ResNet18Truncated().to(device).eval()
    model = resnet18truncated


    print_param_counts(model, "Original model")

    loaders = get_dataloaders(seed=0, batch_size=args.batch_size)
    dataloader = loaders[0]['validation']

    def representative_dataset_gen():
        it = iter(dataloader)
        for _ in range(args.n_iter):
            try:
                batch = next(it)
            except StopIteration:
                break
            img = batch[0] if not isinstance(batch, dict) else batch.get("image", next(iter(batch.values())))
            if img.dim()>4:
                img = img.squeeze(0)
            yield [img]

    # Post-training quantization
    tp = mct.get_target_platform_capabilities('pytorch', 'imx500', target_platform_version='v1')#'default')
    q_model, q_info = mct.ptq.pytorch_post_training_quantization(
        in_module=resnet18truncated,
        representative_data_gen=representative_dataset_gen,
        target_platform_capabilities=tp
    )

    # ←— INSERT: helper to print dtypes
    def print_dtype_info(mod, label):
        print(f"\n{label} model parameter dtypes:")
        for n, p in mod.state_dict().items():
            # bit width for integer dtypes
            bits = torch.iinfo(p.dtype).bits if p.dtype.is_floating_point == False else torch.finfo(p.dtype).bits
            print(f"  {n:40s} {str(p.dtype):10s} ({bits}-bit)")
    # print_dtype_info(model,   "Original")
    # print_dtype_info(q_model, "Quantized")
    print_param_counts(q_model, "Quantized model")

    acc, auroc, ap = evaluate(resnet18truncated, loaders[0]['testing'])
    print(f"Original Accuracy: {acc:.2f}%, AUROC: {auroc:.4f}, AP: {ap:.4f}")

    acc, auroc, ap = evaluate(q_model, loaders[0]['testing'])
    print(f"Q-Model Accuracy: {acc:.2f}%, AUROC: {auroc:.4f}, AP: {ap:.4f}")

    mct.exporter.pytorch_export_model(
        q_model,
        save_model_path=os.path.join(base_dir, "qmodel.onnx"),
        repr_dataset=representative_dataset_gen,
    )
    print(f"Quantized model saved to {os.path.join(base_dir, 'qmodel.onnx')}")


if __name__ == '__main__':
    main()


