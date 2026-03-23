from loss import FocalLoss
from collections import OrderedDict
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from model import Discriminator, Projection, PatchMaker
from typing import Tuple
from torch.fx.proxy import Proxy
from torchvision import models

import numpy as np
import pandas as pd
import torch.nn.functional as F

import logging
import os
import math
import torch
import tqdm
import common
import metrics
import cv2
import utils
import glob
import shutil
import hashlib


LOGGER = logging.getLogger(__name__)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class TBWrapper:
    def __init__(self, log_dir):
        self.g_iter = 0
        self.logger = SummaryWriter(log_dir=log_dir)

    def step(self):
        self.g_iter += 1
        
        
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


class GLASS(torch.nn.Module):
    def __init__(self, device):
        super(GLASS, self).__init__()
        self.device = device
        self.trace_mode = False  


    @staticmethod
    def _group_mean_reduce_channels_spatial(
        x: torch.Tensor,
        out_channels: int,
        group_size: int,
        spatial_hw: Tuple[int, int],
    ) -> torch.Tensor:
        # Trace/export path uses fixed sizes (batch=1, spatial dims set via static_patch_shapes).
        h, w = int(spatial_hw[0]), int(spatial_hw[1])
        # Use -1 to infer batch without reading x.shape (FX/MCT friendly).
        return x.reshape(-1, out_channels, group_size, h, w).mean(dim=2)

    def load(
            self,
            backbone,
            layers_to_extract_from,
            device,
            input_shape,
            pretrain_embed_dimension,
            target_embed_dimension,
            patchsize=3,
            patchstride=1,
            meta_epochs=640,
            eval_epochs=1,
            dsc_layers=2,
            dsc_hidden=512,
            dsc_margin=0.5,
            train_backbone=False,
            pre_proj=1,
            mining=1,
            noise=0.015,
            radius=0.75,
            p=0.5,
            lr=0.0001,
            svd=0,
            step=20,
            limit=392,
            **kwargs,
    ):

        self.backbone = backbone.to(device)
        self.layers_to_extract_from = layers_to_extract_from
        self.input_shape = input_shape
        self.device = device
        
        self.skip_backbone = kwargs.get("skip_backbone", False)

        self.forward_modules = torch.nn.ModuleDict({})
        feature_aggregator = common.NetworkFeatureAggregator(
            self.backbone, self.layers_to_extract_from, self.device, train_backbone
        )
        feature_dimensions = feature_aggregator.feature_dimensions(input_shape)
        # Keep feature channel dims available for FX-safe patch extraction.
        self.feature_dimensions = list(feature_dimensions)
        self.forward_modules["feature_aggregator"] = feature_aggregator

        preprocessing = common.Preprocessing(feature_dimensions, pretrain_embed_dimension)
        self.forward_modules["preprocessing"] = preprocessing
        self.pretrain_embed_dimension = int(pretrain_embed_dimension)
        # Patch-grid path outputs 64+64 channels; keep discriminator width aligned
        self.target_embed_dimension = 128
        preadapt_aggregator = common.Aggregator(target_dim=target_embed_dimension)
        preadapt_aggregator.to(self.device)
        self.forward_modules["preadapt_aggregator"] = preadapt_aggregator

        self.meta_epochs = meta_epochs
        self.lr = lr
        self.train_backbone = train_backbone
        if self.train_backbone:
            self.backbone_opt = torch.optim.AdamW(self.forward_modules["feature_aggregator"].backbone.parameters(), lr)

        self.pre_proj = 0  # disable pre-projection for SDSP-safe path

        self.eval_epochs = eval_epochs
        self.dsc_layers = dsc_layers
        self.dsc_hidden = dsc_hidden
        self.discriminator = Discriminator(self.target_embed_dimension, n_layers=dsc_layers, hidden=dsc_hidden)
        self.discriminator.to(self.device)
        self.dsc_opt = torch.optim.AdamW(self.discriminator.parameters(), lr=lr * 2)
        self.dsc_margin = dsc_margin

        # Edge export head (trace/export only). Inputs will be 128 ch (64+64) at 16x16.
        self.edge_head = torch.nn.Sequential(
            torch.nn.Conv2d(128, 64, kernel_size=1),
            torch.nn.LeakyReLU(0.2),
            torch.nn.Conv2d(64, 1, kernel_size=1),
            torch.nn.Sigmoid(),
        ).to(self.device)

        self.c = torch.tensor(0)
        self.c_ = torch.tensor(0)
        self.p = p
        self.radius = radius
        self.mining = mining
        self.noise = noise
        self.svd = svd
        self.step = step
        self.limit = limit
        self.distribution = 0
        self.focal_loss = FocalLoss()

        self.patch_maker = PatchMaker(patchsize, stride=patchstride)
        # Pre-build patch extraction convs for the feature maps we will patchify.
        self.patch_maker.build_for_channels(self.feature_dimensions, device=self.device, dtype=torch.float32)
        self.anomaly_segmentor = common.RescaleSegmentor(device=self.device, target_size=input_shape[-2:])
        self.model_dir = ""
        self.dataset_name = ""
        self.logger = None
        
        self.resNet18 = ResNet18Truncated().to(device).eval()


    def set_model_dir(self, model_dir, dataset_name):
        self.model_dir = model_dir
        os.makedirs(self.model_dir, exist_ok=True)
        self.ckpt_dir = os.path.join(self.model_dir, dataset_name)
        os.makedirs(self.ckpt_dir, exist_ok=True)
        self.tb_dir = os.path.join(self.ckpt_dir, "tb")
        os.makedirs(self.tb_dir, exist_ok=True)
        self.logger = TBWrapper(self.tb_dir)



    def _embed(self, input, detach=True, provide_patch_shapes=False, evaluation=False):
        """Returns feature embeddings for images."""
        # Assert valid input type
        assert isinstance(input, (torch.Tensor, Proxy, dict)), (
            f"Expected input to be torch.Tensor, dict, or torch.fx.Proxy, got {type(input).__name__}"
        )

        # Extract features through backbone or use pre-extracted features
        if self.skip_backbone:
            assert isinstance(input, dict), (
                f"When skip_backbone=True, input must be a dict of features, got {type(input).__name__}"
            )
            features = [input[layer] for layer in self.layers_to_extract_from]
        else:
            assert isinstance(input, (torch.Tensor, Proxy)), (
                f"When skip_backbone=False, input must be torch.Tensor or Proxy, got {type(input).__name__}"
            )
            # FX/MCT: Run backbone step-by-step so every node output is a Tensor
            x = self.resNet18.conv1(input)
            x = self.resNet18.bn1(x)
            x = self.resNet18.relu(x)
            x = self.resNet18.maxpool(x)
            x = self.resNet18.layer1(x)
            x2 = self.resNet18.layer2(x)
            x3 = self.resNet18.layer3(x2)

            features = []
            for layer_name in self.layers_to_extract_from:
                if layer_name == "layer2":
                    features.append(x2)
                elif layer_name == "layer3":
                    features.append(x3)
                else:
                    raise ValueError(
                        f"Unsupported layer '{layer_name}' for ResNet18Truncated; "
                        "only 'layer2' and 'layer3' are supported in this path."
                    )

        # Patchify features - always use grid path for SDSP (batch=1)
        x2_grid = self.patch_maker.patchify_grid(x2, in_channels=128)  # [B, 1152, 32, 32]
        x3_grid = self.patch_maker.patchify_grid(x3, in_channels=256)  # [B, 2304, 16, 16]

        # Downsample layer2 grid to 16x16 to align with layer3 spatially
        x2_grid = F.avg_pool2d(x2_grid, kernel_size=2, stride=2)  # [B, 1152, 16, 16]
        ref_num_patches = (16, 16)

        # Reduce channels while keeping spatial grid (avoid batch blow-up)
        x2_grid = common.reduce_channels_keep_spatial(
            x2_grid, in_channels=1152, out_channels=64, spatial_hw=ref_num_patches
        )
        x3_grid = common.reduce_channels_keep_spatial(
            x3_grid, in_channels=2304, out_channels=64, spatial_hw=ref_num_patches
        )

        patch_grid = torch.cat([x2_grid, x3_grid], dim=1)  # [B, 128, 16, 16]
        if provide_patch_shapes:
            return patch_grid, [list(ref_num_patches)]
        return patch_grid

    def trainer(self, training_data, val_data, name):
        state_dict = {}
        ckpt_path = glob.glob(self.ckpt_dir + '/ckpt_best*')
        ckpt_path_save = os.path.join(self.ckpt_dir, "ckpt.pth")
        if len(ckpt_path) != 0:
            LOGGER.info("Start testing, ckpt file found!")
            return 0., 0., 0., 0., 0., -1.

        def update_state_dict():
            state_dict["discriminator"] = OrderedDict({
                k: v.detach().cpu()
                for k, v in self.discriminator.state_dict().items()})
            if self.pre_proj > 0:
                state_dict["pre_projection"] = OrderedDict({
                    k: v.detach().cpu()
                    for k, v in self.pre_projection.state_dict().items()})

        self.distribution = training_data.dataset.distribution
        xlsx_path = './datasets/excel/' + name.split('_')[0] + '_distribution.xlsx'
        try:
            if self.distribution == 1:  # rejudge by image-level spectrogram analysis
                self.distribution = 1
                self.svd = 1
            elif self.distribution == 2:  # manifold
                self.distribution = 0
                self.svd = 0
            elif self.distribution == 3:  # hypersphere
                self.distribution = 0
                self.svd = 1
            elif self.distribution == 4:  # opposite choose by file
                self.distribution = 0
                df = pd.read_excel(xlsx_path)
                self.svd = 1 - df.loc[df['Class'] == name, 'Distribution'].values[0]
            else:  # choose by file
                self.distribution = 0
                df = pd.read_excel(xlsx_path)
                self.svd = df.loc[df['Class'] == name, 'Distribution'].values[0]
        except:
            self.distribution = 1
            self.svd = 1

        # judge by image-level spectrogram analysis
        if self.distribution == 1:
            self.forward_modules.eval()
            with torch.no_grad():
                for i, data in enumerate(training_data):
                    img = data["image"]
                    img = img.to(torch.float).to(self.device)
                    batch_mean = torch.mean(img, dim=0)
                    if i == 0:
                        self.c = batch_mean
                    else:
                        self.c += batch_mean
                self.c /= len(training_data)

            avg_img = utils.torch_format_2_numpy_img(self.c.detach().cpu().numpy())
            self.svd = utils.distribution_judge(avg_img, name)
            os.makedirs(f'./results/judge/avg/{self.svd}', exist_ok=True)
            cv2.imwrite(f'./results/judge/avg/{self.svd}/{name}.png', avg_img)
            return self.svd

        pbar = tqdm.tqdm(range(self.meta_epochs), unit='epoch')
        pbar_str1 = ""
        best_record = None
        for i_epoch in pbar:
            self.forward_modules.eval()
            with torch.no_grad():  # compute center
                for i, data in enumerate(training_data):
                    img = data["image"]
                    img = img.to(torch.float).to(self.device)
                    if self.pre_proj > 0:
                        outputs = self.pre_projection(self._embed(img, evaluation=False))
                        outputs = outputs[0] if isinstance(outputs, (list, tuple)) else outputs
                    else:
                        outputs = self._embed(img, evaluation=False)
                    outputs = outputs[0] if isinstance(outputs, (list, tuple)) else outputs  # [B,C,H,W]

                    # Compute channel-wise center across all patches in the batch
                    outputs_flat = outputs.permute(0, 2, 3, 1).reshape(-1, outputs.shape[1])  # [B*P,C]
                    batch_mean = torch.mean(outputs_flat, dim=0)  # [C]
                    if i == 0:
                        self.c = batch_mean
                    else:
                        self.c += batch_mean
                self.c = (self.c / len(training_data)).to(self.device)

            pbar_str, pt, pf = self._train_discriminator(training_data, i_epoch, pbar, pbar_str1)
            update_state_dict()

            if (i_epoch + 1) % self.eval_epochs == 0:
                images, scores, segmentations, labels_gt, masks_gt, _ = self.predict(val_data)
                image_auroc, image_ap, pixel_auroc, pixel_ap, pixel_pro = self._evaluate(images, scores, segmentations,
                                                                                         labels_gt, masks_gt, name)

                self.logger.logger.add_scalar("i-auroc", image_auroc, i_epoch)
                self.logger.logger.add_scalar("i-ap", image_ap, i_epoch)
                self.logger.logger.add_scalar("p-auroc", pixel_auroc, i_epoch)
                self.logger.logger.add_scalar("p-ap", pixel_ap, i_epoch)
                self.logger.logger.add_scalar("p-pro", pixel_pro, i_epoch)

                eval_path = './results/eval/' + name + '/'
                train_path = './results/training/' + name + '/'
                if best_record is None:
                    best_record = [image_auroc, image_ap, pixel_auroc, pixel_ap, pixel_pro, i_epoch]
                    ckpt_path_best = os.path.join(self.ckpt_dir, "ckpt_best_{}.pth".format(i_epoch))
                    torch.save(state_dict, ckpt_path_best)
                    shutil.rmtree(eval_path, ignore_errors=True)
                    shutil.copytree(train_path, eval_path)
                    #log first best record with data
                    LOGGER.info(f"First best record: IAUC:{round(image_auroc * 100, 2)}")
                    

                #elif image_auroc + pixel_auroc > best_record[0] + best_record[2]:
                elif image_auroc  > best_record[0]:
                    best_record = [image_auroc, image_ap, pixel_auroc, pixel_ap, pixel_pro, i_epoch]
                    os.remove(ckpt_path_best)
                    ckpt_path_best = os.path.join(self.ckpt_dir, "ckpt_best_{}.pth".format(i_epoch))
                    torch.save(state_dict, ckpt_path_best)
                    shutil.rmtree(eval_path, ignore_errors=True)
                    shutil.copytree(train_path, eval_path)
                    #log best record with data
                    LOGGER.info(f"new Best record: IAUC:{round(image_auroc * 100, 2)}")
                
                else:
                    #Log "No improvment, this auroc: X best auroc Y message
                    LOGGER.info(f"No improvement, this auroc: {round(image_auroc * 100, 2)} best auroc: {round(best_record[0] * 100, 2)}")


                pbar_str1 = f" IAUC:{round(image_auroc * 100, 2)}({round(best_record[0] * 100, 2)})" \
                            f" IAP:{round(image_ap * 100, 2)}({round(best_record[1] * 100, 2)})" \
                            f" PAUC:{round(pixel_auroc * 100, 2)}({round(best_record[2] * 100, 2)})" \
                            f" PAP:{round(pixel_ap * 100, 2)}({round(best_record[3] * 100, 2)})" \
                            f" PRO:{round(pixel_pro * 100, 2)}({round(best_record[4] * 100, 2)})" \
                            f" E:{i_epoch}({best_record[-1]})"
                pbar_str += pbar_str1
                pbar.set_description_str(pbar_str)

            torch.save(state_dict, ckpt_path_save)
        return best_record

    def _train_discriminator(self, input_data, cur_epoch, pbar, pbar_str1):
        self.forward_modules.eval()
        if self.pre_proj > 0:
            self.pre_projection.train()
        self.discriminator.train()

        all_loss, all_p_true, all_p_fake, all_r_t, all_r_g, all_r_f = [], [], [], [], [], []
        sample_num = 0
        for i_iter, data_item in enumerate(input_data):
            self.dsc_opt.zero_grad()
            if self.pre_proj > 0:
                self.proj_opt.zero_grad()

            aug = data_item["aug"]
            aug = aug.to(torch.float).to(self.device)
            img = data_item["image"]
            img = img.to(torch.float).to(self.device)
            if self.pre_proj > 0:
                fake_feats = self.pre_projection(self._embed(aug, evaluation=False))
                fake_feats = fake_feats[0] if isinstance(fake_feats, (list, tuple)) else fake_feats
                true_feats = self.pre_projection(self._embed(img, evaluation=False))
                true_feats = true_feats[0] if isinstance(true_feats, (list, tuple)) else true_feats
            else:
                fake_feats = self._embed(aug, evaluation=False)
                true_feats = self._embed(img, evaluation=False)
            # Ensure leaf tensors for grad
            fake_feats = fake_feats.detach().requires_grad_(True)
            true_feats = true_feats.detach().requires_grad_(True)

            # Resize mask to match patch grid (16x16) and flatten to patch dimension
            mask_s_gt = data_item["mask_s"].to(torch.float32).to(self.device)
            if mask_s_gt.dim() == 2:
                mask_s_gt = mask_s_gt.unsqueeze(0)  # [1, H, W]
            mask_s_gt = F.interpolate(
                mask_s_gt.unsqueeze(1), size=true_feats.shape[-2:], mode="nearest"
            ).squeeze(1)  # [B, H, W]
            mask_s_flat = mask_s_gt.reshape(true_feats.shape[0], -1)  # [B, P]
            mask_s_gt = mask_s_flat  # reuse 2D mask for discriminator outputs

            noise = torch.normal(0, self.noise, true_feats.shape, device=self.device)
            gaus_feats = true_feats + noise

            # Flatten spatial dims for discriminator ([B, C, P, 1])
            bsz, ch, h, w = true_feats.shape
            true_seq = true_feats.reshape(bsz, ch, h * w, 1)
            fake_seq = fake_feats.reshape(bsz, ch, h * w, 1)
            gaus_seq = gaus_feats.reshape(bsz, ch, h * w, 1)

            # Flatten patch grids to [B*P, C] for distance computations
            true_flat = true_feats.permute(0, 2, 3, 1).reshape(-1, true_feats.shape[1])
            fake_flat = fake_feats.permute(0, 2, 3, 1).reshape(-1, fake_feats.shape[1])
            gaus_flat = gaus_feats.permute(0, 2, 3, 1).reshape(-1, gaus_feats.shape[1])
            center_flat = self.c.reshape(1, -1).repeat(true_flat.shape[0], 1)
            mask_flat = mask_s_flat.reshape(-1)  # [B*P]

            true_points = torch.concat([fake_flat[mask_flat == 0], true_flat], dim=0)
            c_t_points = torch.concat([center_flat[mask_flat == 0], center_flat], dim=0)
            dist_t = torch.norm(true_points - c_t_points, dim=1)
            r_t = torch.tensor([torch.quantile(dist_t, q=self.radius)]).to(self.device)

            for step in range(self.step + 1):
                gaus_seq = gaus_feats.reshape(bsz, ch, h * w, 1)
                gaus_flat = gaus_feats.permute(0, 2, 3, 1).reshape(-1, ch)
                scores = self.discriminator(torch.cat([true_seq, gaus_seq]))
                true_scores = scores[: len(true_feats)]
                gaus_scores = scores[len(true_feats) :]
                true_loss = torch.nn.BCELoss()(true_scores, torch.zeros_like(true_scores))
                gaus_loss = torch.nn.BCELoss()(gaus_scores, torch.ones_like(gaus_scores))
                bce_loss = true_loss + gaus_loss

                if step == self.step:
                    break
                elif self.mining == 0:
                    dist_g = torch.norm(gaus_flat - center_flat, dim=1)
                    r_g = torch.tensor([torch.quantile(dist_g, q=self.radius)]).to(self.device)
                    break

                grad = torch.autograd.grad(gaus_loss, [gaus_feats])[0]
                # Normalize per spatial location to keep shapes aligned with [B, C, H, W]
                grad_norm = torch.norm(grad, dim=1, keepdim=True)  # [B,1,H,W]
                grad_normalized = grad / (grad_norm + 1e-10)

                with torch.no_grad():
                    gaus_feats.add_(0.001 * grad_normalized)

                if (step + 1) % 5 == 0:
                    gaus_flat = gaus_feats.permute(0, 2, 3, 1).reshape(-1, ch)
                    dist_g = torch.norm(gaus_flat - center_flat, dim=1)
                    r_g = torch.tensor([torch.quantile(dist_g, q=self.radius)]).to(self.device)
                    proj_feats = center_flat if self.svd == 1 else true_flat
                    r = r_t if self.svd == 1 else 0.5

                    h_vec = gaus_flat - proj_feats
                    h_norm = dist_g if self.svd == 1 else torch.norm(h_vec, dim=1)
                    alpha = torch.clamp(h_norm, r, 2 * r)
                    proj = (alpha / (h_norm + 1e-10)).view(-1, 1)
                    h_vec = proj * h_vec
                    gaus_flat = proj_feats + h_vec
                    gaus_feats = gaus_flat.reshape_as(gaus_feats)

            fake_points = fake_flat[mask_flat == 1]
            true_points = true_flat[mask_flat == 1]
            c_f_points = center_flat[mask_flat == 1]
            dist_f = torch.norm(fake_points - c_f_points, dim=1)
            r_f = torch.tensor([torch.quantile(dist_f, q=self.radius)]).to(self.device)
            proj_feats = c_f_points if self.svd == 1 else true_points
            r = r_t if self.svd == 1 else 1

            if self.svd == 1:
                h_vec = fake_points - proj_feats
                h_norm = dist_f if self.svd == 1 else torch.norm(h_vec, dim=1)
                alpha = torch.clamp(h_norm, 2 * r, 4 * r)
                proj = (alpha / (h_norm + 1e-10)).view(-1, 1)
                h_vec = proj * h_vec
                fake_points = proj_feats + h_vec
                fake_flat[mask_flat == 1] = fake_points
                fake_feats = fake_flat.reshape_as(fake_feats)
                fake_seq = fake_feats.reshape(bsz, ch, h * w, 1)

            fake_scores = self.discriminator(fake_seq)
            if self.p > 0:
                fake_dist = (fake_scores - mask_s_gt) ** 2
                d_hard = torch.quantile(fake_dist, q=self.p)
                fake_scores_ = fake_scores[fake_dist >= d_hard].unsqueeze(1)
                mask_ = mask_s_gt[fake_dist >= d_hard].unsqueeze(1)
            else:
                fake_scores_ = fake_scores
                mask_ = mask_s_gt
            output = torch.cat([1 - fake_scores_, fake_scores_], dim=1)
            focal_loss = self.focal_loss(output, mask_)

            loss = bce_loss + focal_loss
            loss.backward()
            if self.pre_proj > 0:
                self.proj_opt.step()
            if self.train_backbone:
                self.backbone_opt.step()
            self.dsc_opt.step()

            pix_true = torch.concat([fake_scores.detach() * (1 - mask_s_gt), true_scores.detach()])
            pix_fake = torch.concat([fake_scores.detach() * mask_s_gt, gaus_scores.detach()])
            p_true = ((pix_true < self.dsc_margin).sum() - (pix_true == 0).sum()) / ((mask_s_gt == 0).sum() + true_scores.shape[0])
            p_fake = (pix_fake >= self.dsc_margin).sum() / ((mask_s_gt == 1).sum() + gaus_scores.shape[0])

            self.logger.logger.add_scalar(f"p_true", p_true, self.logger.g_iter)
            self.logger.logger.add_scalar(f"p_fake", p_fake, self.logger.g_iter)
            self.logger.logger.add_scalar(f"r_t", r_t, self.logger.g_iter)
            self.logger.logger.add_scalar(f"r_g", r_g, self.logger.g_iter)
            self.logger.logger.add_scalar(f"r_f", r_f, self.logger.g_iter)
            self.logger.logger.add_scalar("loss", loss, self.logger.g_iter)
            self.logger.step()

            all_loss.append(loss.detach().cpu().item())
            all_p_true.append(p_true.cpu().item())
            all_p_fake.append(p_fake.cpu().item())
            all_r_t.append(r_t.cpu().item())
            all_r_g.append(r_g.cpu().item())
            all_r_f.append(r_f.cpu().item())

            all_loss_ = np.mean(all_loss)
            all_p_true_ = np.mean(all_p_true)
            all_p_fake_ = np.mean(all_p_fake)
            all_r_t_ = np.mean(all_r_t)
            all_r_g_ = np.mean(all_r_g)
            all_r_f_ = np.mean(all_r_f)
            sample_num = sample_num + img.shape[0]

            pbar_str = f"epoch:{cur_epoch} loss:{all_loss_:.2e}"
            pbar_str += f" pt:{all_p_true_ * 100:.2f}"
            pbar_str += f" pf:{all_p_fake_ * 100:.2f}"
            pbar_str += f" rt:{all_r_t_:.2f}"
            pbar_str += f" rg:{all_r_g_:.2f}"
            pbar_str += f" rf:{all_r_f_:.2f}"
            pbar_str += f" svd:{self.svd}"
            pbar_str += f" sample:{sample_num}"
            pbar_str2 = pbar_str
            pbar_str += pbar_str1
            pbar.set_description_str(pbar_str)

            if sample_num > self.limit:
                break

        return pbar_str2, all_p_true_, all_p_fake_

    def tester(self, test_data, name, threshold=0.9):
        ckpt_path = glob.glob(self.ckpt_dir + '/ckpt_best*')
        if len(ckpt_path) != 0:
            state_dict = torch.load(ckpt_path[0], map_location=self.device)
            if 'discriminator' in state_dict:
                self.discriminator.load_state_dict(state_dict['discriminator'])
                if "pre_projection" in state_dict:
                    self.pre_projection.load_state_dict(state_dict["pre_projection"])
            else:
                self.load_state_dict(state_dict, strict=False)

            torch.cuda.reset_peak_memory_stats()
            # force CUDA to initialize
            dummy = torch.randn(1).to(self.device)

            images, scores, segmentations, labels_gt, masks_gt, class_names = self.predict(test_data)
            used_memory_MB = torch.cuda.max_memory_allocated(self.device) / 1024**2
            LOGGER.info(f"💾 Peak VRAM usage during inference: {used_memory_MB:.2f} MB")

            image_auroc, image_ap, pixel_auroc, pixel_ap, pixel_pro = self._evaluate(images, scores, segmentations,
                                                                                     labels_gt, masks_gt, name, path='eval')
            epoch = int(ckpt_path[0].split('_')[-1].split('.')[0])
                       
            # y_true = original class (e.g. "good", "crack", ...)
            y_true = class_names
            y_true_bin = [0 if cls == "good" else 1 for cls in class_names]  # binary ground truth

            #comupte best threshold for accuracy
            best_threshold = 0
            best_accuracy = 0
            for t in np.arange(0.1, 1.0, 0.01):
                y_pred_bin = [0 if s <= t else 1 for s in scores]
                accuracy = np.sum(np.array(y_true_bin) == np.array(y_pred_bin)) / len(y_true_bin)
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_threshold = t
            LOGGER.info("Best Threshold: " + str(best_threshold))
            threshold = best_threshold

            from sklearn.metrics import confusion_matrix
            # y_pred = binary prediction: "good" if score <= threshold else "anomaly"
            y_pred = ["good" if s <= threshold else "anomaly" for s in scores]
            y_pred_float = [s for s in scores]
            y_pred_bin = [0 if s <= threshold else 1 for s in scores]  # binary prediction

            #print accuracy
            accuracy = np.sum(np.array(y_true_bin) == np.array(y_pred_bin)) / len(y_true_bin)
            LOGGER.info("Accuracy: " + str(accuracy))

            # --- Plot ROC Curve ---
            from sklearn.metrics import roc_curve, auc
            import matplotlib.pyplot as plt

            fpr, tpr, thresholds = roc_curve(y_true_bin, scores)
            roc_auc = auc(fpr, tpr)


            plt.figure(figsize=(6, 6))
            plt.plot(fpr, tpr, color='darkorange', lw=2, label=f"ROC curve (AUC = {roc_auc:.2f})")
            plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'ROC Curve - {name}')
            plt.legend(loc='lower right')
            plt.grid(True)

            roc_plot_path = os.path.join(self.ckpt_dir, f"roc_curve_{name}.png")
            plt.savefig(roc_plot_path)
            LOGGER.info(f"📉 ROC curve saved to: {roc_plot_path}")
            plt.close()


            # Define consistent label order
            row_labels = sorted(set(y_true))  # e.g. ["crack", "faulty_imprint", "good", ...]
            col_labels = ["good", "anomaly"]  # binary output from model

           
            # Compute confusion matrix with string labels
            cm = confusion_matrix(y_true, y_pred, labels=col_labels+list(set(y_true)-set(col_labels)))  # to avoid shape mismatch

            # But to align the rows manually for display:
            counts = {label: {"good": 0, "anomaly": 0} for label in row_labels}
            for true, pred in zip(y_true, y_pred):
                counts[true][pred] += 1

            # Format into matrix
            header = "," + ",\t".join(col_labels)
            rows = [f"{label},\t" + ",\t".join(str(counts[label][pred]) for pred in col_labels) for label in row_labels]
            cm_str = "\n".join([header] + rows)
            LOGGER.info("🔍 Binary Confusion Matrix by Defect Type:\n" + cm_str)



            #print ytrue and ypred:
            LOGGER.info("y_true: " + str(y_true))
            LOGGER.info("y_pred: " + str(y_pred))   
            LOGGER.info("y_pred_float: " + str(y_pred_float))

            return image_auroc, image_ap, pixel_auroc, pixel_ap, pixel_pro, epoch, y_true, y_pred
        else:
            image_auroc, image_ap, pixel_auroc, pixel_ap, pixel_pro, epoch = 0., 0., 0., 0., 0., -1.    
            y_true, y_pred = [], []            
            LOGGER.info("No ckpt file found!")

        return image_auroc, image_ap, pixel_auroc, pixel_ap, pixel_pro, epoch, y_true, y_pred

    def _evaluate(self, images, scores, segmentations, labels_gt, masks_gt, name, path='training'):
        scores = np.squeeze(np.array(scores))
        image_scores = metrics.compute_imagewise_retrieval_metrics(scores, labels_gt, path)
        image_auroc = image_scores["auroc"]
        image_ap = image_scores["ap"]

        if len(masks_gt) > 0:
            segmentations = np.array(segmentations)
            pixel_scores = metrics.compute_pixelwise_retrieval_metrics(segmentations, masks_gt, path)
            pixel_auroc = pixel_scores["auroc"]
            pixel_ap = pixel_scores["ap"]
            if path == 'eval':
                try:
                    pixel_pro = metrics.compute_pro(np.squeeze(np.array(masks_gt)), segmentations)
                except:
                    pixel_pro = 0.
            else:
                pixel_pro = 0.
        else:
            pixel_auroc = -1.
            pixel_ap = -1.
            pixel_pro = -1.
            return image_auroc, image_ap, pixel_auroc, pixel_ap, pixel_pro

        defects = np.array(images)
        targets = np.array(masks_gt)
        for i in range(len(defects)):
            defect = utils.torch_format_2_numpy_img(defects[i])
            target = utils.torch_format_2_numpy_img(targets[i])

            mask = cv2.cvtColor(cv2.resize(segmentations[i], (defect.shape[1], defect.shape[0])),
                                cv2.COLOR_GRAY2BGR)
            mask = (mask * 255).astype('uint8')
            mask = cv2.applyColorMap(mask, cv2.COLORMAP_JET)

            img_up = np.hstack([defect, target, mask])
            img_up = cv2.resize(img_up, (256 * 3, 256))
            full_path = './results/' + path + '/' + name + '/'
            utils.del_remake_dir(full_path, del_flag=False)
            cv2.imwrite(full_path + str(i + 1).zfill(3) + '.png', img_up)

        return image_auroc, image_ap, pixel_auroc, pixel_ap, pixel_pro

    def predict(self, test_dataloader):
        """This function provides anomaly scores/maps for full dataloaders."""
        self.forward_modules.eval()

        img_paths = []
        images = []
        scores = []
        masks = []
        labels_gt = []
        masks_gt = []
        class_names = []


        with tqdm.tqdm(test_dataloader, desc="Inferring...", leave=False, unit='batch') as data_iterator:
            for data in data_iterator:
                if isinstance(data, dict):
                    #labels_gt.extend(data["is_anomaly"].numpy().tolist())
                    # Get true class name from path — assumes path ends in /<class>/<img>
                    for path in data["image_path"]:
                        class_name = os.path.normpath(path).split(os.sep)[-2]
                        labels_gt.append(0 if class_name == "good" else 1)
                        #save class name separately for confusion matrix
                        class_names.append(class_name)

                    if data.get("mask_gt", None) is not None:
                        masks_gt.extend(data["mask_gt"].numpy().tolist())
                    image = data["image"]
                    images.extend(image.numpy().tolist())
                    img_paths.extend(data["image_path"])
                _scores, _masks = self._predict(image)
                for score, mask in zip(_scores, _masks):
                    scores.append(score)
                    masks.append(mask)

        return images, scores, masks, labels_gt, masks_gt, class_names

        
    def print_stats(self):
        """Print detailed statistics about the GLASS model state for debugging."""
    
        def tensor_checksum(tensor):
            """Compute MD5 checksum of a tensor."""
            if tensor is None:
                return "None"
            return hashlib.md5(tensor.detach().cpu().numpy().tobytes()).hexdigest()[:16]
        
        def state_dict_checksum(state_dict):
            """Compute checksum of entire state dict."""
            combined = b""
            for k in sorted(state_dict.keys()):
                combined += state_dict[k].detach().cpu().numpy().tobytes()
            return hashlib.md5(combined).hexdigest()[:16]
        
        print("\n" + "="*80)
        print("GLASS MODEL STATISTICS")
        print("="*80)
        
        # Basic configuration
        print("\n--- Configuration ---")
        print(f"Device: {self.device}")
        print(f"Input shape: {self.input_shape}")
        print(f"Layers to extract: {self.layers_to_extract_from}")
        print(f"Feature dimensions: {self.feature_dimensions}")
        print(f"Pretrain embed dim: {self.pretrain_embed_dimension}")
        print(f"Target embed dim: {self.target_embed_dimension}")
        print(f"Pre-projection layers: {self.pre_proj}")
        print(f"DSC layers: {self.dsc_layers}")
        print(f"DSC hidden: {self.dsc_hidden}")
        print(f"DSC margin: {self.dsc_margin}")
        print(f"Skip backbone: {self.skip_backbone}")
        print(f"Trace mode: {self.trace_mode}")
        print(f"Static patch shapes: {getattr(self, 'static_patch_shapes', 'N/A')}")
        
        # Patch maker
        print("\n--- Patch Maker ---")
        print(f"Patch size: {self.patch_maker.patchsize}")
        print(f"Patch stride: {self.patch_maker.stride}")
        
        # Training parameters
        print("\n--- Training Parameters ---")
        print(f"Meta epochs: {self.meta_epochs}")
        print(f"Learning rate: {self.lr}")
        print(f"Mining: {self.mining}")
        print(f"Noise: {self.noise}")
        print(f"Radius: {self.radius}")
        print(f"p: {self.p}")
        print(f"SVD: {self.svd}")
        print(f"Step: {self.step}")
        print(f"Limit: {self.limit}")
        
        # Center tensor
        print("\n--- Center Tensor ---")
        if hasattr(self, 'c') and self.c is not None and self.c.numel() > 1:
            print(f"Shape: {self.c.shape}")
            print(f"Mean: {self.c.mean().item():.6f}")
            print(f"Std: {self.c.std().item():.6f}")
            print(f"Min: {self.c.min().item():.6f}")
            print(f"Max: {self.c.max().item():.6f}")
            print(f"Checksum: {tensor_checksum(self.c)}")
        elif hasattr(self, 'c') and self.c is not None:
            print(f"Shape: {self.c.shape}")
            print(f"Value: {self.c.item() if self.c.numel() == 1 else 'Scalar not initialized'}")
            print(f"Checksum: {tensor_checksum(self.c)}")
        else:
            print("Not initialized")
        
        # Backbone
        print("\n--- Backbone (ResNet18Truncated) ---")
        print(f"Type: {type(self.resNet18).__name__}")
        print(f"Training mode: {self.resNet18.training}")
        total_params = sum(p.numel() for p in self.resNet18.parameters())
        trainable_params = sum(p.numel() for p in self.resNet18.parameters() if p.requires_grad)
        print(f"Total params: {total_params:,}")
        print(f"Trainable params: {trainable_params:,}")
        print(f"State dict checksum: {state_dict_checksum(self.resNet18.state_dict())}")
        
        # Key layer checksums
        print(f"  conv1.weight: {tensor_checksum(self.resNet18.conv1.weight)}")
        print(f"  layer2.0.conv1.weight: {tensor_checksum(self.resNet18.layer2[0].conv1.weight)}")
        print(f"  layer3.0.conv1.weight: {tensor_checksum(self.resNet18.layer3[0].conv1.weight)}")
        
        # Pre-projection
        if self.pre_proj > 0:
            print("\n--- Pre-Projection ---")
            print(f"Type: {type(self.pre_projection).__name__}")
            print(f"Training mode: {self.pre_projection.training}")
            total_params = sum(p.numel() for p in self.pre_projection.parameters())
            trainable_params = sum(p.numel() for p in self.pre_projection.parameters() if p.requires_grad)
            print(f"Total params: {total_params:,}")
            print(f"Trainable params: {trainable_params:,}")
            print(f"State dict checksum: {state_dict_checksum(self.pre_projection.state_dict())}")
            
            # First layer checksum
            if hasattr(self.pre_projection, 'projection'):
                for i, layer in enumerate(self.pre_projection.projection):
                    if hasattr(layer, 'weight'):
                        print(f"  Layer {i} weight: {tensor_checksum(layer.weight)}")
        
        # Discriminator
        print("\n--- Discriminator ---")
        print(f"Type: {type(self.discriminator).__name__}")
        print(f"Training mode: {self.discriminator.training}")
        total_params = sum(p.numel() for p in self.discriminator.parameters())
        trainable_params = sum(p.numel() for p in self.discriminator.parameters() if p.requires_grad)
        print(f"Total params: {total_params:,}")
        print(f"Trainable params: {trainable_params:,}")
        print(f"State dict checksum: {state_dict_checksum(self.discriminator.state_dict())}")
        
        # Key layer checksums
        if hasattr(self.discriminator, 'discriminator'):
            for i, layer in enumerate(self.discriminator.discriminator):
                if hasattr(layer, 'weight'):
                    print(f"  Layer {i} weight: {tensor_checksum(layer.weight)}")
        
        # Forward modules
        print("\n--- Forward Modules ---")
        for name, module in self.forward_modules.items():
            print(f"{name}:")
            print(f"  Type: {type(module).__name__}")
            print(f"  Training mode: {module.training}")
            if hasattr(module, 'state_dict'):
                print(f"  State dict checksum: {state_dict_checksum(module.state_dict())}")
        
        # Model directory info
        print("\n--- Model Paths ---")
        print(f"Model dir: {self.model_dir}")
        print(f"Dataset name: {self.dataset_name}")
        print(f"Checkpoint dir: {getattr(self, 'ckpt_dir', 'N/A')}")
        
        # Check for loaded checkpoints
        if hasattr(self, 'ckpt_dir') and self.ckpt_dir:
            ckpts = glob.glob(os.path.join(self.ckpt_dir, "ckpt_best*.pth"))
            if ckpts:
                print(f"Best checkpoint: {ckpts[-1]}")
                # Load and show checkpoint info
                try:
                    state = torch.load(ckpts[-1], map_location='cpu')
                    print(f"Checkpoint keys: {list(state.keys())}")
                    if 'discriminator' in state:
                        print(f"  Discriminator checksum: {state_dict_checksum(state['discriminator'])}")
                    if 'pre_projection' in state:
                        print(f"  Pre-projection checksum: {state_dict_checksum(state['pre_projection'])}")
                except Exception as e:
                    print(f"Error loading checkpoint: {e}")
        
        print("\n" + "="*80)
        print("END OF STATISTICS")
        print("="*80 + "\n")



    def _predict(self, img):
        """Infer score and mask for a batch of images."""
        img = img.to(torch.float).to(self.device)
        self.forward_modules.eval()

        if self.pre_proj > 0:
            self.pre_projection.eval()
        self.discriminator.eval()

        with torch.no_grad():
            # Reuse forward() path to keep inference consistent
            patch_scores = self.forward(img)
            masks = self.anomaly_segmentor.convert_to_segmentation(patch_scores)

            image_scores = patch_scores.amax(dim=(1,2))
            if isinstance(image_scores, torch.Tensor):
                image_scores = image_scores.cpu().numpy()
                
        #self.print_stats()

        return list(image_scores), list(masks)
    
    
        
    def head(self, patch_features, batch_size=1, patch_shapes=[(32, 32), (16, 16)]):
        
        patch_features = self.pre_projection(patch_features)
        if isinstance(patch_features, (tuple, list)):
            patch_features = patch_features[0]
    
        # Get discriminator scores
        patch_scores = self.discriminator(patch_features)
        patch_scores = patch_scores.squeeze(1)  # [B, H, W]


        row_max = torch.amax(patch_scores, dim=2)  # [B, H]
        image_scores = torch.amax(row_max, dim=1, keepdim=True)  # [B, 1]
        
        
        output = {"image_scores": image_scores}
        output["patch_scores"] = patch_scores
        return output

    def forward(self, input):
        """
        Run single-batch inference. FX/MCT compatible.
        
        Args:
            input: Tensor of shape [B, C, H, W] or dict with pre-extracted features
            
        Returns:
            Patch-level scores of shape [B, H, W]
        """
        # Validate input
        assert isinstance(input, (torch.Tensor, Proxy, dict)), (
            f"Expected input to be torch.Tensor, dict, or torch.fx.Proxy, got {type(input).__name__}"
        )
        
        # Set to eval mode
        self.forward_modules.eval()
        if self.pre_proj > 0:
            self.pre_projection.eval()
        self.discriminator.eval()
        
        # Single unified path (works for Tensor or Proxy), grid-based to keep batch=1
        patch_grid, patch_shapes = self._embed(input, provide_patch_shapes=True, evaluation=True)  # [B,128,16,16]

        # Use trained discriminator to produce patch map directly
        bsz, ch, h, w = patch_grid.shape
        disc_in = patch_grid.reshape(bsz, ch, h * w, 1)
        patch_scores = self.discriminator(disc_in).reshape(bsz, h, w)
    
        return patch_scores
