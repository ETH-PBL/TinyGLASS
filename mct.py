import torch.fx

# Tell FX to treat len() (and list()) as atomic, so that
# len(proxy) and list(proxy_shape) calls get recorded
# instead of blowing up.
torch.fx.wrap('len')
torch.fx.wrap('list')
import model_compression_toolkit as mct
import torch
from torch.utils.data import DataLoader, random_split
import numpy as np
import random
from onnx2torch import convert
import os
from datasets.mvtec import MVTecDataset, DatasetSplit
from tqdm import tqdm


# ── CONFIGURE THESE TO MATCH YOUR main.py CALL ────────────────────────────────
data_path       = "/datasets/sutterra/mvtec"
aug_path        = "/datasets/sutterra/mvtec/aug"
subdatasets     = ["mms_stretch"]
batch_size      = 1
num_workers     = 4
resize          = 288
imagesize       = 288
val_split_ratio = 0.4
# ──────────────────────────────────────────────────────────────────────────────

def get_dataloaders(seed: int, test: bool):
    dataloaders = []
    for classname in subdatasets:
        # 1) load the full TEST split
        full_test = MVTecDataset(
            data_path,
            aug_path,
            classname=classname,
            resize=resize,
            imagesize=imagesize,
            split=DatasetSplit.TEST,
            seed=seed
        )

        # 2) apply same val‐split if requested
        if val_split_ratio is not None:
            val_size  = int(val_split_ratio * len(full_test))
            test_size = len(full_test) - val_size
            test_ds, val_ds = random_split(
                full_test,
                [test_size, val_size],
                generator=torch.Generator().manual_seed(seed)
            )
        else:
            test_ds, val_ds = full_test, full_test

        # 3) build PyTorch DataLoaders
        test_loader = DataLoader(
            test_ds, batch_size=batch_size,
            shuffle=False, num_workers=num_workers,
            pin_memory=True
        )
        val_loader = DataLoader(
            val_ds, batch_size=batch_size,
            shuffle=False, num_workers=num_workers,
            pin_memory=True
        )

        # 4) mimic your naming conventions
        test_loader.name = f"mvtec_{classname}"
        val_loader.name  = test_loader.name + "_val"

        # 5) we aren’t training here, so just return val/test
        dataloaders.append({
            "training": val_loader,    # you can treat val as “training” if you only eval
            "validation": val_loader,
            "testing": test_loader
        })

    return dataloaders




def evaluate(model, testloader):
    """
    Evaluate a model using a test loader.
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()  # Set the model to evaluation mode
    correct = 0
    total = 0
    with torch.no_grad():
        for data in tqdm(testloader):
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            # correct += (predicted == labels).sum().item()
    val_acc = (100 * correct / total)
    print('Accuracy: %.2f%%' % val_acc)
    return val_acc



float_model = convert("/home/sutterra/GLASS/results/models/backbone_0/mvtec_mms_stretch_resnet18_v2/mvtec_mms_stretch_resnet18_v2_simplified.onnx")

float_model.eval()
input_shape = (1, 3, imagesize, imagesize)
input_tensor = torch.randn(input_shape)

output = float_model(input_tensor)
print("Output shape:", output.shape)


seed = 0
use_test_flag = True

loaders = get_dataloaders(seed, use_test_flag)
for dset in loaders:
    print(f"{dset['testing'].name}: "
            f"test={len(dset['testing'].dataset)}, "
            f"val={len(dset['validation'].dataset)}")


batch_size = 16
n_iter = 10

dataloader = loaders[0]['validation']

def representative_dataset_gen():
    """
    Yields a list containing a single image tensor each iteration,
    matching what main.py would pass to your model.
    """
    dataloader_iter = iter(dataloader)
    for i in range(n_iter):
        try:
            batch = next(dataloader_iter)
        except StopIteration:
            break
        # batch could be a tuple (img, label) or a dict {'image': img, ...}
        if isinstance(batch, dict):
            img = batch.get("image", None)
            if img is None:
                # fallback to first value in dict
                img = next(iter(batch.values()))
        else:
            # assume tuple/list where first element is image
            img = batch[0]
        # now img is your [B,C,H,W] tensor – strip batch dim if >1
        if img.dim() > 4:
            img = img.squeeze(0)
        yield [img]


# Get a FrameworkQuantizationCapabilities object that models the hardware platform for the quantized model inference. Here, for example, we use the default platform that is attached to a Pytorch layers representation.
target_platform_cap = mct.get_target_platform_capabilities('pytorch', 'default')

import torch
from onnx2torch.node_converters import resize as _resize

_orig = _resize._onnx_mode_to_torch_mode

def _patched_mode(onnx_mode, dim_size):
    # if we got a Proxy, just pick a reasonable default:
    if isinstance(dim_size, torch.fx.Proxy):
        # assume a 2-D spatial resize → "bilinear"
        if onnx_mode == "linear":
            return "bilinear"
        # you could add 'cubic'→'bicubic' etc. here
    # otherwise fall back
    return _orig(onnx_mode, dim_size)

_resize._onnx_mode_to_torch_mode = _patched_mode


quantized_model, quantization_info = mct.ptq.pytorch_post_training_quantization(
        in_module=float_model,
        representative_data_gen=representative_dataset_gen,
        target_platform_capabilities=target_platform_cap
)


evaluate(float_model, loaders[0]['testing'])
evaluate(quantized_model, loaders[0]['testing'])


mct.exporter.pytorch_export_model(quantized_model, save_model_path='qmodel.onnx', repr_dataset=representative_dataset_gen)


