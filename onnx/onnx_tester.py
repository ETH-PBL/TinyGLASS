import os, sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import torch
import onnxruntime as ort
import numpy as np
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from torchvision import models


# import your dataset modules
import datasets.mvtec as mvtec_mod
import datasets.visa as visa_mod


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

def run_onnx_test():
    # ==== EDIT THESE PARAMETERS ====
    dataset_name = "mvtec"
    data_path    = "/datasets/sutterra/mvtec/mms_rpi/"
    aug_path     = "/datasets/sutterra/mvtec/fg_mask/"
    onnx_path    = "/home/sutterra/GLASS/results/models/backbone_0/mvtec_mms_rpi/mvtec_mms_rpi_FA_simplified.onnx"
    batch_size   = 1
    num_workers  = 4
    image_size   = 288
    seed         = 0
    # ================================

    # map name → (DatasetClass, SplitEnum)
    _DATASETS = {
        "mvtec": (mvtec_mod.MVTecDataset, mvtec_mod.DatasetSplit),
        "visa":  (visa_mod.VisADataset,    visa_mod.DatasetSplit),
        "mpdd":  (mvtec_mod.MVTecDataset, mvtec_mod.DatasetSplit),
        "wfdd":  (mvtec_mod.MVTecDataset, mvtec_mod.DatasetSplit),
    }
    DS, Split = _DATASETS[dataset_name]

    # 1. build test‐only loader
    test_ds = DS(
        data_path, aug_path,
        classname=None,
        resize=image_size,
        imagesize=image_size,
        split=Split.TEST,
        seed=seed,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    # 2. load trunk
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bb = ResNet18Truncated(pretrained=False)
    bb.to(device).eval()

    # 3. load ONNX session
    sess = ort.InferenceSession(onnx_path)
    input_name = sess.get_inputs()[0].name

    y_true, y_score = [], []

    # 4. inference
    with torch.no_grad():
        for batch in test_loader:
            img   = batch[0].to(device)
            label = batch[1].cpu().numpy()
            feats_np = bb(img).cpu().numpy()
            out = sess.run(None, {input_name: feats_np})[0]
            y_true.append(label)
            y_score.append(out.reshape(-1))

    y_true  = np.concatenate(y_true,  axis=0)
    y_score = np.concatenate(y_score, axis=0)
    auroc   = roc_auc_score(y_true, y_score)

    print(f"Image-level AUROC: {auroc*100:.2f}%")

if __name__ == "__main__":
    run_onnx_test()