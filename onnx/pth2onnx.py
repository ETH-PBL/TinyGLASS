import torch
import onnx

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from onnxsim import simplify
from torchvision import models
from glass import GLASS


class GLASS_onnx(torch.nn.Module):
    def __init__(self):
        super(GLASS_onnx, self).__init__()
        self.backbone = models.wide_resnet50_2(pretrained=False).cuda()
        self.backbone.load_state_dict(torch.load('/home/sutterra/.cache/torch/hub/checkpoints/wide_resnet50_2-95faca4d.pth', map_location='cuda'))
        self.glass = GLASS('cuda')
        self.glass.load(self.backbone, ["layer2", "layer3"],
                        'cuda', (3, 288, 288), 1536, 1536)
        state_dict = torch.load('results/models/backbone_0/mvtec_mms_stretch_wideresnet/ckpt_best_164.pth', map_location='cuda')
        self.glass.pre_projection.load_state_dict(state_dict["pre_projection"])
        self.glass.discriminator.load_state_dict(state_dict["discriminator"])

    def forward(self, img):
        with torch.no_grad():
            patch_features, patch_shapes = self.glass._embed(img, provide_patch_shapes=True, evaluation=True)
            patch_features = self.glass.pre_projection(patch_features)
            patch_scores = self.glass.discriminator(patch_features)

            patch_scores = self.glass.patch_maker.unpatch_scores(patch_scores, batchsize=img.shape[0])
            scales = patch_shapes[0]
            patch_scores = patch_scores.reshape(img.shape[0], scales[0], scales[1])
        return patch_scores


if __name__ == '__main__':
    x = torch.randn(1, 3, 288, 288, device="cuda")

    torch.onnx.export(
        GLASS_onnx(),
        x,
        "results/models/backbone_0/mvtec_mms_stretch_wideresnet/glass_original.onnx",
        verbose=True,
        input_names=["input"],
        output_names=["output"]
    )

    # load your predefined ONNX model
    model = onnx.load("results/models/backbone_0/mvtec_mms_stretch_wideresnet/glass_original.onnx")

    # convert model
    model_simp, check = simplify(model)
    assert check, "Simplified ONNX model could not be validated"

    # use model_simp as a standard ONNX model object
    onnx.save(model_simp, "results/models/backbone_0/mvtec_mms_stretch_wideresnet/glass_original_simplified.onnx")
    print('finished exporting onnx')
