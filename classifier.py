import torch
from PIL import Image
from torchvision import transforms
import backbones
from glass import GLASS, IMAGENET_MEAN, IMAGENET_STD
import common
from torchvision import models
from torch.profiler import profile, record_function, ProfilerActivity
import os, ctypes, sys
import time



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

    

def main():
    # ---- User inputs ----
    #image_path      = "/datasets/sutterra/mvtec/mms_stretch/test/good/IMG_0009.png"
    #image_path      = "/datasets/sutterra/mvtec_debug/mms_stretch/test/good/IMG_0009.png"
    #image_path      = "/datasets/sutterra/mvtec/mms_stretch/test/good/IMG_0201.png"
    image_path      = "/root/rafeal_sutter_564/datasets/mms_rpi/train/good/IMG_0001.png"
    #image_path      ="/datasets/sutterra/mvtec_debug/mms_stretch/test/crack_hole/IMG_0038.png"
    #image_path      ="/datasets/sutterra/mvtec_debug/mms_stretch/test/crack_hole/IMG_0039.png"

    checkpoint_path = "results/models/backbone_0/mvtec_mms/ckpt.pth"
    device          = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backbone_name   = "resnet18"
    layers_to_use   = ["layer2", "layer3"]
    input_shape     = (3, 288, 288)
    pretrain_embed_dimension = 384
    target_embed_dimension = 384
    
    # ---------------------

    # 1) load backbone
    backbone = backbones.load(backbone_name)
    backbone = backbone.to(device)

    # 2) init GLASS and load config
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
    glass.static_patch_shapes = [(36, 36), (18, 18)]
    
    # 3) load checkpoint weights
    state = torch.load(checkpoint_path, map_location=device)
    if "discriminator" in state:
        glass.discriminator.load_state_dict(state["discriminator"])
    if glass.pre_proj > 0 and "pre_projection" in state:
        glass.pre_projection.load_state_dict(state["pre_projection"])
    glass = glass.to(device).eval()

    # 4) prepare image tensor
    img_tensor = load_image(image_path, device, input_shape[-2:])
    
    feature_aggregator = common.NetworkFeatureAggregator(backbone, layers_to_use, device, train_backbone=False).eval()
    resNet18 = ResNet18Truncated().to(device).eval()
    
    
    #feats = feature_aggregator(img_tensor)
    #feats = resNet18(img_tensor)
    output = glass(img_tensor)
    
    
     # print shapes of whatever comes back
    print("Output[0] shape:", output[0].shape)
    print("Output[1] shape:", output[1].shape) 

    # 6) inspect
    image_scores, masks = output
    print("Image score:", image_scores)  # torch.Size([1])
    print("Mask shape:", masks.shape)          # torch.Size([1, 288, 288])
    
    glass.eval()
    input = {
        "layer2": torch.randn(1, 128, 36, 36).to(device),
        "layer3": torch.randn(1, 256, 18, 18).to(device),
    }
    torch.randn(1,288,288).to(device)

    
    print(f"shape of img_tensor: {img_tensor.shape}")
    #if the feats variable exists print its layer shapes
    if 'feats' in locals():
        print(f"shape of feats l2: {feats['layer2'].shape}")
        print(f"shape of feats l3: {feats['layer3'].shape}")
    print(f"shape of output: {output[0].shape}")
    print(f"shape of output: {output[1].shape}")
    
    print(f"output score: {output[0]}")
    
    print("using preprojection yes or no: ", glass.pre_proj)
    
    
    return 0

    # --- count parameters ---
    def count_params(m):
        #print(f"print model dict: {m}")
        return sum(p.numel() for p in m.parameters())

    print(f"ResNet18Truncated params:        {count_params(resNet18)}")
    print(f"FeatureAggregator params:        {count_params(feature_aggregator)}")
    print(f"GLASS model params (total):      {count_params(glass)}")
    prepro = glass.forward_modules["preprocessing"]
    print(f"GLASS.forward_modules[preprocessing] model params (total):{count_params(prepro)}")
    preadapt_aggregator = glass.forward_modules["preadapt_aggregator"]
    print(f"GLASS.forward_modules[preadapt_aggregator] model params (total):{count_params(preadapt_aggregator)}")
    pre_projection = glass.pre_projection
    print(f"GLASS.pre_projection model params (total):{count_params(pre_projection)}")
    discriminator = glass.discriminator
    print(f"GLASS.discriminator model params (total):{count_params(discriminator)}")

    # warm-up and timing settings
    num_warmup = 50
    num_runs   = 200
    times_feat  = []
    times_res   = []
    times_model = []

    # Warm-up (to stabilize CUDA, JIT, caches…)
    with torch.no_grad():
        for _ in range(num_warmup):
            feats = feature_aggregator(img_tensor)
            if device.type == "cuda": torch.cuda.synchronize()
            _ = resNet18(img_tensor)
            if device.type == "cuda": torch.cuda.synchronize()
            _ = glass(feats)
            if device.type == "cuda": torch.cuda.synchronize()

    # Timed runs
    with torch.no_grad():
        for _ in range(num_runs):
            # feature_aggregator
            t0 = time.perf_counter()
            _ = feature_aggregator(img_tensor)
            if device.type == "cuda": torch.cuda.synchronize()
            times_feat.append(time.perf_counter() - t0)

            # ResNet18Truncated
            t0 = time.perf_counter()
            feats = resNet18(img_tensor)
            if device.type == "cuda": torch.cuda.synchronize()
            times_res.append(time.perf_counter() - t0)

            # full GLASS model
            t0 = time.perf_counter()
            output = glass(feats)
            if device.type == "cuda": torch.cuda.synchronize()
            times_model.append(time.perf_counter() - t0)
            

    

    # compute & print stats (in ms)
    avg  = lambda lst: sum(lst) / len(lst)
    mn   = lambda lst: min(lst)
    mx   = lambda lst: max(lst)
    print(f"feature_aggregator over {num_runs} runs: avg {avg(times_feat)*1000:.2f} ms, "
          f"min {mn(times_feat)*1000:.2f} ms, max {mx(times_feat)*1000:.2f} ms")
    print(f"ResNet18Truncated over {num_runs} runs: avg {avg(times_res)*1000:.2f} ms, "
          f"min {mn(times_res)*1000:.2f} ms, max {mx(times_res)*1000:.2f} ms")
    print(f"GLASS model over {num_runs} runs: avg {avg(times_model)*1000:.2f} ms, "
          f"min {mn(times_model)*1000:.2f} ms, max {mx(times_model)*1000:.2f} ms")

     # print shapes of whatever comes back
    print("Output[0] shape:", output[0].shape)
    print("Output[1] shape:", output[1].shape) 

    # 6) inspect
    image_scores, masks = output
    print("Image score:", image_scores)  # torch.Size([1])
    print("Mask shape:", masks.shape)          # torch.Size([1, 288, 288])
    
    glass.eval()
    input = {
        "layer2": torch.randn(1, 128, 36, 36).to(device),
        "layer3": torch.randn(1, 256, 18, 18).to(device),
    }
    torch.randn(1,288,288).to(device)

    
    print(f"shape of img_tensor: {img_tensor.shape}")
    print(f"shape of feats l2: {feats['layer2'].shape}")
    print(f"shape of feats l3: {feats['layer3'].shape}")
    print(f"shape of output: {output[0].shape}")
    print(f"shape of output: {output[1].shape}")
    
    
    


    #cupti profiling;
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                 record_shapes=True) as prof:
        with record_function("inference"):
            _ = glass(input)
    print(prof.key_averages().table(sort_by="self_cuda_time_total"))
    


if __name__ == "__main__":
    main()