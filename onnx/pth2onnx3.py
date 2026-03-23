import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import argparse
import onnx
import os
from onnxsim import simplify
from glass import GLASS
import backbones
import glob
import time
from thop import profile

class GLASSOnnxWrapper(torch.nn.Module):
    def __init__(self, backbone_name, layers, input_shape, pre_dim, tgt_dim, ckpt_path, device, batchsize=1):
        super(GLASSOnnxWrapper, self).__init__()

        backbone = backbones.load(backbone_name)
        backbone.eval().to(device)

        self.glass = GLASS(device)
        self.glass.load(
            backbone=backbone,
            layers_to_extract_from=layers,
            device=device,
            input_shape=input_shape,
            pretrain_embed_dimension=pre_dim,
            target_embed_dimension=tgt_dim,
        )
        self.batchsize = batchsize

        state_dict = torch.load(ckpt_path, map_location=device)
        self.glass.pre_projection.load_state_dict(state_dict["pre_projection"])
        self.glass.discriminator.load_state_dict(state_dict["discriminator"])

    def forward(self, layer2, layer3):
        with torch.no_grad():
            # 1) get raw patch scores
            pf, ps = self.glass._embed_by_features(layer2, layer3, provide_patch_shapes=True, evaluation=True)
            pf = self.glass.pre_projection(pf)
            raw_scores = self.glass.discriminator(pf)

            # 2) unpatch to [B, H*W] then reshape to [B,H,W]
            flat = self.glass.patch_maker.unpatch_scores(raw_scores, batchsize=self.batchsize)
            H, W = ps[0]
            patch_map = flat.reshape(self.batchsize, H, W)

            # 3) image score = max over all patches
            img_score = self.glass.patch_maker.score(flat)

        return patch_map, img_score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, required=True, help="Subfolder name in results/models/backbone_0/")
    parser.add_argument('--backbone', type=str, default="wideresnet50")
    parser.add_argument('--layers', type=str, nargs='+', default=["layer2", "layer3"])
    parser.add_argument('--input_size', type=int, nargs=3, default=[3, 288, 288])
    parser.add_argument('--pre_dim', type=int, default=1536)
    parser.add_argument('--tgt_dim', type=int, default=1536)
    parser.add_argument('--batch_size', type=int, default=8)

    args = parser.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    base_dir = os.path.join("results", "models", "backbone_0", args.model_dir)
    assert os.path.isdir(base_dir), f"❌ Directory not found: {base_dir}"

    ckpt_files = sorted(glob.glob(os.path.join(base_dir, "ckpt_best_*.pth")))
    assert ckpt_files, f"❌ No ckpt_best_*.pth files found in {base_dir}"
    ckpt_path = ckpt_files[-1]  # use the latest by name

    dummy_input = (
        torch.randn([args.batch_size, 128, 36, 36]).to(device),
        torch.randn([args.batch_size, 256, 18, 18]).to(device)
    )

    model = GLASSOnnxWrapper(
        backbone_name=args.backbone,
        layers=args.layers,
        input_shape=tuple(args.input_size),
        pre_dim=args.pre_dim,
        tgt_dim=args.tgt_dim,
        ckpt_path=ckpt_path,
        device=device,
        batchsize=args.batch_size
    ).to(device)

    model.eval()

    onnx_path = os.path.join(base_dir, f"{args.model_dir}_FA.onnx")
    simp_path = os.path.join(base_dir, f"{args.model_dir}_FA_simplified.onnx")

    torch.onnx.export(
        model, dummy_input, onnx_path,
        input_names=["input"], output_names=["output"],
        verbose=True
    )

    model_onnx = onnx.load(onnx_path)
    model_simp, check = simplify(model_onnx)
    assert check, "Simplified ONNX model could not be validated"
    onnx.save(model_simp, simp_path)

    print(f"✅ Saved ONNX: {onnx_path}")
    print(f"✅ Saved simplified ONNX: {simp_path}")
    print(f"✅ Used checkpoint: {ckpt_path}")


    # PyTorch model inference
    model.eval()
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        start = time.perf_counter()
        pt_out = model(dummy_input[0].to(device), dummy_input[1].to(device))
        if device == 'cuda':
            torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - start) * 1000
    peak_bytes = torch.cuda.max_memory_allocated()
    peak_mb = peak_bytes / 1024**2

    print(f"PyTorch inference time: {elapsed_ms:.1f} ms")
    print(f"PyTorch peak VRAM usage: {peak_mb:.1f} MB")
    print(f"PyTorch output shape: {tuple(pt_out.shape)}")

    macs, params = profile(model, inputs=(dummy_input[0].to(device), dummy_input[1].to(device)))
    print(f"PyTorch MACs: {macs / 1e6:.1f} M")
    print(f"PyTorch params: {params / 1e6:.1f} M")
    

    # ONNX Runtime inference
    # ort_sess = onnxruntime.InferenceSession(onnx_path, providers=['CPUExecutionProvider','CUDAExecutionProvider'])
    # ort_inputs = {ort_sess.get_inputs()[0].name: dummy_input.cpu().numpy()}
    # ort_out = ort_sess.run(None, ort_inputs)[0]
    # print(f"ONNX Runtime output shape: {tuple(ort_out.shape)}")


if __name__ == '__main__':
    main()
