import copy
import numpy as np
import scipy.ndimage as ndimage
import torch
import torch.nn.functional as F
from typing import Tuple


def reduce_channels_keep_spatial(
    x: torch.Tensor,
    in_channels: int,
    out_channels: int,
    spatial_hw: Tuple[int, int],
) -> torch.Tensor:
    """Reduce channel dimension with adaptive avg pooling while preserving spatial grid.

    This is used to avoid converting the patch grid (H*W) into the batch dimension,
    since the IMX500 converter only supports batch dim of 1/None.

    Args:
        x: Tensor of shape [B, in_channels, H, W]
        in_channels: Known input channel count.
        out_channels: Desired output channel count.
        spatial_hw: (H, W) as python ints (static for tracing/export).

    Returns:
        Tensor of shape [B, out_channels, H, W]
    """
    h, w = int(spatial_hw[0]), int(spatial_hw[1])
    group_size = int(in_channels // out_channels)
    assert in_channels % out_channels == 0, "in_channels must be divisible by out_channels"
    # Use -1 to infer batch so FX/Proxy tracing avoids reading x.shape[0]
    x = x.reshape(-1, out_channels, group_size, h, w)
    x = x.mean(dim=2)
    return x  # [B, Cout, H, W]

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
    

class Preprocessing(torch.nn.Module):
    def __init__(self, input_dims, output_dim):
        super(Preprocessing, self).__init__()
        self.input_dims = input_dims
        self.output_dim = output_dim

        self.preprocessing_modules = torch.nn.ModuleList()
        for _ in input_dims:
            module = MeanMapper(output_dim)
            self.preprocessing_modules.append(module)

    def forward(self, features):
        _features = []
        for module, feature in zip(self.preprocessing_modules, features):
            _features.append(module(feature))
        return torch.stack(_features, dim=1)


class MeanMapper(torch.nn.Module):
    def __init__(self, preprocessing_dim):
        super(MeanMapper, self).__init__()
        self.preprocessing_dim = preprocessing_dim

    def forward(self, features):
        # Keep this tensor-only for FX/MCT: avoid reading features.shape (creates Size/tuple nodes).
        features = features.flatten(1).unsqueeze(1).unsqueeze(-1)  # [B, 1, N, 1]
        # SDSP supports 2D pooling with 4D input (N,C,H,W). Treat N as H and use W=1.
        features = F.adaptive_avg_pool2d(features, (self.preprocessing_dim, 1))
        return features.squeeze(-1).squeeze(1)


class Aggregator(torch.nn.Module):
    def __init__(self, target_dim):
        super(Aggregator, self).__init__()
        self.target_dim = target_dim

    def forward(self, features):
        """Returns reshaped and average pooled features."""
        # Keep this tensor-only for FX/MCT: avoid reading features.shape (creates Size/tuple nodes).
        print(f"Aggregator target_dim: {self.target_dim}")
        features = features.flatten(1).unsqueeze(1).unsqueeze(-1)  # [B, 1, N, 1]
        features = F.adaptive_avg_pool2d(features, (self.target_dim, 1))
        return features.squeeze(-1).squeeze(1)  # [B, target_dim]


class RescaleSegmentor:
    def __init__(self, device, target_size=288):
        self.device = device
        self.target_size = target_size
        self.smoothing = 4

    def convert_to_segmentation(self, patch_scores):
        with torch.no_grad():
            if isinstance(patch_scores, np.ndarray):
                patch_scores = torch.from_numpy(patch_scores)
            _scores = patch_scores.to(self.device)
            _scores = _scores.unsqueeze(1)
            _scores = F.interpolate(
                _scores, size=self.target_size, mode="bilinear", align_corners=False
            )
            _scores = _scores.squeeze(1)
            patch_scores = _scores.cpu().numpy()
        return [ndimage.gaussian_filter(patch_score, sigma=self.smoothing) for patch_score in patch_scores]


class NetworkFeatureAggregator(torch.nn.Module):
    """Efficient extraction of network features."""

    def __init__(self, backbone, layers_to_extract_from, device, train_backbone=False):
        super(NetworkFeatureAggregator, self).__init__()
        """Extraction of network features.

        Runs a network only to the last layer of the list of layers where
        network features should be extracted from.

        Args:
            backbone: torchvision.model
            layers_to_extract_from: [list of str]
        """
        self.layers_to_extract_from = layers_to_extract_from
        self.backbone = backbone
        self.device = device
        self.train_backbone = train_backbone
        if not hasattr(backbone, "hook_handles"):
            self.backbone.hook_handles = []
        for handle in self.backbone.hook_handles:
            handle.remove()
        self.outputs = {}

        for extract_layer in layers_to_extract_from:
            self.register_hook(extract_layer)

        self.to(self.device)

    def forward(self, images, eval=True):
        self.outputs.clear()
        if self.train_backbone and not eval:
            self.backbone(images)
        else:
            with torch.no_grad():
                try:
                    _ = self.backbone(images)
                except LastLayerToExtractReachedException:
                    pass
        return self.outputs

    def feature_dimensions(self, input_shape):
        """Computes the feature dimensions for all layers given input_shape."""
        _input = torch.ones([1] + list(input_shape)).to(self.device)
        _output = self(_input)
        return [_output[layer].shape[1] for layer in self.layers_to_extract_from]

    def register_hook(self, layer_name):
        module = self.find_module(self.backbone, layer_name)
        if module is not None:
            forward_hook = ForwardHook(self.outputs, layer_name, self.layers_to_extract_from[-1])
            if isinstance(module, torch.nn.Sequential):
                hook = module[-1].register_forward_hook(forward_hook)
            else:
                hook = module.register_forward_hook(forward_hook)
            self.backbone.hook_handles.append(hook)
        else:
            raise ValueError(f"Module {layer_name} not found in the model")
    
    def find_module(self, model, module_name):
        for name, module in model.named_modules():
            if name == module_name:
                return module
            elif '.' in module_name:
                father, child = module_name.split('.', 1)
                if name == father:
                    return self.find_module(module, child)
        return None

class ForwardHook:
    def __init__(self, hook_dict, layer_name: str, last_layer_to_extract: str):
        self.hook_dict = hook_dict
        self.layer_name = layer_name
        self.raise_exception_to_break = copy.deepcopy(
            layer_name == last_layer_to_extract
        )

    def __call__(self, module, input, output):
        self.hook_dict[self.layer_name] = output
        return None

class LastLayerToExtractReachedException(Exception):
    pass
