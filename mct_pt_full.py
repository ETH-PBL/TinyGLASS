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
import traceback

import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score   
import hashlib



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

def load_checkpoint(glass, checkpoint_path, device):
    state = torch.load(checkpoint_path, map_location=device)
    if "discriminator" in state:
        # Remap Linear weights to Conv1d format if needed (keep learned weights)
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
            print(f"[warn] pre_projection weight load skipped due to shape mismatch: {e}")
    return glass

# helper to count and print parameters
def print_param_counts(mod, name):
    total = sum(p.numel() for p in mod.parameters())
    print(f"{name} total parameters: {total:,}")
    
    
def tensor_checksum(tensor):
    """Compute MD5 checksum of a tensor."""
    if tensor is None:
        return "None"
    return hashlib.md5(tensor.detach().cpu().numpy().tobytes()).hexdigest()[:16]

def evaluate(model, loader, threshold: float = 0.0):
    """
    model: takes a batch of images and returns per-patch scores
    loader: yields (imgs, labels)
    threshold: cutoff on the image-level max score to call 'anomaly'
    returns: (acc, auroc, ap)
    """
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    
    # print_param_counts(glass, "GLASS model")


    all_scores = []
    all_labels = []
    
    image_path = "/root/glass_imx/datasets/mms_rpi/test/crack_hole/IMG_0046.png"
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

            # forward → per-patch scores (same path as forward.py)
            out = model(imgs)
            #print(f"          Model output hash: {tensor_checksum(out)}")

            # If forward returns (scores, masks), grab scores
            if isinstance(out, (tuple, list)):
                out = out[0]

            # If patch map, reduce to image score by max as in GLASS forward use
            if out.dim() == 3:  # [B,H,W]
                img_scores = out.amax(dim=(1,2))
            elif out.dim() == 4:  # [B,1,H,W]
                img_scores = out.squeeze(1).amax(dim=(1,2))
            else:  # fallback
                img_scores = out.reshape(out.shape[0], -1).amax(dim=1)

            all_scores.append(img_scores.detach().cpu().numpy())
            all_labels.append(labels.detach().reshape(-1).cpu().numpy())

    all_scores = np.concatenate(all_scores, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    

    # metrics
    auroc = roc_auc_score(all_labels, all_scores)
    ap    = average_precision_score(all_labels, all_scores)
    preds = (all_scores > threshold).astype(int)
    acc   = (preds == all_labels).mean() * 100

    return acc, auroc, ap

# ── CONFIGURE THESE TO MATCH SETUP ───────────────────────────────────────
DATA_PATH        = "/root/glass_imx/datasets"
AUG_PATH         = "/root/glass_imx/datasets/dtd"
SUBDATASETS      = ["mms_rpi"] 
MODEL_FOLDER     = "mvtec_mms_rpi"    
RESIZE           = 256
IMAGESIZE        = 256
NUM_WORKERS      = 4
INPUT_SIZE       = (3, 256, 256)
VAL_SPLIT_RATIO  = 0.2  # portion of test set used for
PRE_DIM          = 384
TGT_DIM          = 384
BACKBONE         = "resnet18"
BATCH_SIZE       = 16

def main():
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--model_dir', type=str, required=True,
    #                     help="Subfolder in results/models/backbone_0/")
    # parser.add_argument('--backbone',  type=str, default="wideresnet50")
    # parser.add_argument('--layers',    nargs='+', default=["layer2","layer3"])
    # parser.add_argument('--input_size',nargs=3, type=int, default=[3,256,256])
    # parser.add_argument('--pre_dim',   type=int, default=1536)
    # parser.add_argument('--tgt_dim',   type=int, default=1536)
    # parser.add_argument('--batch_size',type=int, default=16)
    # parser.add_argument('--n_iter',    type=int, default=10)
    # parser.add_argument(
    #     '--trace_output',
    #     type=str,
    #     default='image',
    #     choices=['image', 'patch'],
    #     help="Trace/export output: 'image' returns scalar score; 'patch' returns per-patch score map.",
    # )
    # args = parser.parse_args()
    
 
    

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
    n_iter = 10
    
    # Hardcode patch grid sizes (layer2, layer3) for FX/MCT tracing
    # For input 256: layer2 grid is 32x32, layer3 grid is 16x16 (ResNet18).
    glass.static_patch_shapes = [(32, 32), (16, 16)]

    # 3) load checkpoint weights (reshape Linear -> Conv1d weights if needed)
    load_checkpoint(glass, ckpt_path, device)

    # Sync conv-equivalent weights once (avoid in-place copy_ during FX tracing).
    if hasattr(glass.discriminator, "_sync_conv_equivalents"):
        glass.discriminator._sync_conv_equivalents()
    if glass.pre_proj > 0 and hasattr(glass.pre_projection, "_sync_conv_equivalents"):
        glass.pre_projection._sync_conv_equivalents()
    glass = glass.to(device).eval()
    glass.trace_mode = False  # use unified path that produces flat 384-d patch features
    
    loaders = get_dataloaders(seed=0, batch_size=1)
    dataloader = loaders[0]['validation']
   
    glass.print_stats()

    def representative_dataset_gen():
        it = iter(dataloader)
        for _ in range(n_iter):
            try:
                batch = next(it)
            except StopIteration:
                break
            img = batch[0] if not isinstance(batch, dict) else batch.get("image", next(iter(batch.values())))
            if img.dim() > 4:
                img = img.squeeze(0)
            # Force batch=1
            if img.shape[0] > 1:
                print(f"Warning: forcing batch=1 for representative dataset, got batch size {img.shape[0]}")
                img = img[:1]
            yield [img]

    def representative_dataset_gen_for_export():
        it = iter(dataloader)
        for _ in range(n_iter):
            try:
                batch = next(it)
            except StopIteration:
                break
            img = batch[0] if not isinstance(batch, dict) else batch.get("image", next(iter(batch.values())))
            if img.dim() > 4:
                img = img.squeeze(0)
            yield (img,)

    def representative_dataset_gen_for_mct_export():
        """Representative dataset for MCT's ONNX exporter.

        MCT's exporter expects the representative dataset to yield a list/tuple of input tensors
        (even for a single-input model). Yielding `[img]` matches that expectation.
        """
        it = iter(dataloader)
        for _ in range(n_iter):
            try:
                batch = next(it)
            except StopIteration:
                break
            img = batch[0] if not isinstance(batch, dict) else batch.get("image", next(iter(batch.values())))
            if img.dim() > 4:
                img = img.squeeze(0)
            # Force batch=1
            if img.shape[0] > 1:
                print(f"Warning: forcing batch=1 for representative dataset for mct, got batch size {img.shape[0]}")
                img = img[:1]
            yield [img]

    def _get_one_input_from_loader(loader, device_for_export: torch.device):
        batch = next(iter(loader))
        img = batch[0] if not isinstance(batch, dict) else batch.get("image", next(iter(batch.values())))
        if img.dim() > 4:
            img = img.squeeze(0)
        # Export is most stable with batch=1 and float32.
        if img.shape[0] > 1:
            img = img[:1]
        img = img.to(device_for_export, dtype=torch.float32)
        return (img,)

    def _torch_onnx_export_legacy(model: torch.nn.Module, model_inputs: tuple, save_path: str):
        # IMPORTANT: MCT quantizers may keep tensors (e.g., scales) on the original device.
        # Forcing model.to('cpu') can leave some internal CUDA tensors behind and crash.
        # Instead, export on the model's current device and move inputs to match.
        try:
            model_device = next(model.parameters()).device
        except StopIteration:
            # No parameters - fall back to first buffer.
            buffers = list(model.buffers())
            model_device = buffers[0].device if len(buffers) > 0 else torch.device("cpu")

        model = model.eval()
        model_inputs = tuple(x.to(model_device) for x in model_inputs)

        # Use a modern opset when available (Torch 2.4+ defaults to 20 in MCT).
        opsets_to_try = [20, 17, 15]
        last_exc = None
        for opset in opsets_to_try:
            try:
                # torch>=2.1 supports dynamo flag. Force legacy tracer to avoid torch.export dynamic_shapes issues.
                torch.onnx.export(
                    model,
                    model_inputs,
                    save_path,
                    opset_version=opset,
                    input_names=["input"],
                    output_names=["output"],
                    do_constant_folding=True,
                    dynamo=False,
                )
                return
            except TypeError:
                # Older torch: no dynamo arg.
                try:
                    torch.onnx.export(
                        model,
                        model_inputs,
                        save_path,
                        opset_version=opset,
                        input_names=["input"],
                        output_names=["output"],
                        do_constant_folding=True,
                    )
                    return
                except Exception as e:
                    last_exc = e
            except Exception as e:
                last_exc = e

        raise RuntimeError(f"Legacy torch.onnx.export failed for all opsets {opsets_to_try}: {last_exc}")

    def _enable_mctq_custom_ops_for_onnx_export(model: torch.nn.Module):
        """Enable MCTQ custom-ops forward implementations in quantizers (similar to MCT exporter)."""
        from mct_quantizers import PytorchActivationQuantizationHolder, PytorchQuantizationWrapper
        from mct_quantizers import pytorch_quantizers

        for _, m in model.named_modules():
            if isinstance(m, PytorchActivationQuantizationHolder):
                assert isinstance(m.activation_holder_quantizer, pytorch_quantizers.BasePyTorchInferableQuantizer)
                m.activation_holder_quantizer.enable_custom_impl()

            if isinstance(m, PytorchQuantizationWrapper):
                for wq in m.weights_quantizers.values():
                    assert isinstance(wq, pytorch_quantizers.BasePyTorchInferableQuantizer)
                    wq.enable_custom_impl()

    def _mctq_onnx_export_legacy(model: torch.nn.Module, model_inputs: tuple, save_path: str):
        """Export using MCTQ custom ops but force legacy torch.onnx.export (dynamo=False)."""
        _enable_mctq_custom_ops_for_onnx_export(model)
        _torch_onnx_export_legacy(model, model_inputs, save_path)

    # Post-training quantization
    tp = mct.get_target_platform_capabilities('pytorch', 'imx500', target_platform_version='v1')#'default')
    q_model, q_info = mct.ptq.pytorch_post_training_quantization(
        in_module=glass,
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

    acc, auroc, ap = evaluate(glass, loaders[0]['testing'], threshold=0.1)
    print(f"Original Glass Accuracy: {acc:.2f}%, AUROC: {auroc:.4f}, AP: {ap:.4f}")

    acc, auroc, ap = evaluate(q_model, loaders[0]['testing'], threshold=0.1)
    print(f"Q-Model Accuracy: {acc:.2f}%, AUROC: {auroc:.4f}, AP: {ap:.4f}")

    onnx_path = os.path.join(base_dir, "qmodel.onnx")
    try:
        # MCT exporter produces an ONNX with quantization ops, but with Torch 2.5+ the dynamo-based
        # path can error due to dynamic_axes -> dynamic_shapes structure mismatches.
        mct.exporter.pytorch_export_model(
            q_model,
            save_model_path=onnx_path,
            repr_dataset=representative_dataset_gen_for_mct_export,
        )
        print(f"Quantized model saved to {onnx_path}")
    except Exception:
        print("\nMCT ONNX export failed; retrying legacy export with MCTQ custom ops (dynamo=False).")
        traceback.print_exc()
        export_inputs = _get_one_input_from_loader(dataloader, device_for_export=torch.device("cpu"))
        try:
            _mctq_onnx_export_legacy(q_model, export_inputs, onnx_path)
            print(f"(Fallback/MCTQ) Exported model saved to {onnx_path}")
        except Exception:
            print("\nMCTQ legacy export failed; falling back to plain legacy torch.onnx.export.")
            traceback.print_exc()
            _torch_onnx_export_legacy(q_model, export_inputs, onnx_path)
            print(f"(Fallback/plain) Exported model saved to {onnx_path}")


if __name__ == '__main__':
    main()


