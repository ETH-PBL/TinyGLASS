import torch
from torch.nn import Linear, BatchNorm2d, Conv2d
from torch.fx.proxy import Proxy
from typing import Optional


def init_weight(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_normal_(m.weight)
    if isinstance(m, torch.nn.BatchNorm2d):
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif isinstance(m, torch.nn.Conv2d):
        m.weight.data.normal_(0.0, 0.02)


class Discriminator(torch.nn.Module):
    """Per-patch discriminator keeping batch intact. Uses 1x1 Conv2d on [B, C, P, 1]."""

    def __init__(self, in_planes, n_layers=2, hidden=None):
        super(Discriminator, self).__init__()
        
        assert hidden is None or hidden == 512, "Only hidden=None (512) is supported currently, got"

        self.body = torch.nn.Sequential(
            torch.nn.Conv2d(in_planes, 512, kernel_size=1),
            torch.nn.BatchNorm2d(512),
            torch.nn.LeakyReLU(0.2),
        )
        self.tail = torch.nn.Sequential(
            torch.nn.Conv2d(512, 1, kernel_size=1, bias=False),
            torch.nn.Sigmoid(),
        )
        self.apply(init_weight)

    def forward(self, x):
        # Expect [B, C, P, 1]
        x = self.body(x)
        x = self.tail(x)
        # [B, 1, P, 1] -> [B, P]
        return x.squeeze(-1).squeeze(1)


class Projection(torch.nn.Module):
    """1x1 Conv2d projection on [B, C, P, 1] keeping batch intact."""

    def __init__(self, in_planes=384, out_planes=384, n_layers=1, layer_type=0):
        super(Projection, self).__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Conv2d(in_planes, out_planes, kernel_size=1),
        )
        self.apply(init_weight)

    def forward(self, x):
        # Expect [B, C, P, 1]
        x = self.layers(x)
        return x  # keep 4D; caller decides squeeze/permute


class PatchMaker(torch.nn.Module):
    def __init__(self, patchsize, top_k=0, stride=None):
        super().__init__()
        self.patchsize = patchsize
        self.stride = 1 if stride is None else stride
        self.top_k = top_k
        self.padding = (patchsize - 1) // 2

        # IMX500/SDSP note:
        # torch.nn.Unfold commonly exports to ONNX using Gather-based indexing.
        # The IMX500 converter rejects multi-index Gather patterns.
        # We therefore implement patch extraction via grouped Conv2d with fixed one-hot weights,
        # which exports as Conv + Reshape/Transpose (SDSP-friendly).
        self._patch_extract_convs = torch.nn.ModuleDict()

    def _get_patch_extract_conv(self, in_channels: int, device: torch.device, dtype: torch.dtype) -> torch.nn.Conv2d:
        key = str(int(in_channels))
        if key in self._patch_extract_convs:
            conv = self._patch_extract_convs[key]
            # IMPORTANT: during torch.fx tracing, `device`/`dtype` may be Proxies.
            # Avoid calling Module.to() with symbolic values.
            if isinstance(device, torch.device) and isinstance(dtype, torch.dtype):
                conv = conv.to(device=device, dtype=dtype)
            return conv

        ps = int(self.patchsize)
        out_channels = in_channels * ps * ps
        conv = torch.nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=ps,
            stride=int(self.stride),
            padding=int(self.padding),
            dilation=1,
            groups=in_channels,
            bias=False,
        )

        # One-hot weights to extract each (ky,kx) within each channel.
        # Weight shape for grouped conv: [C*ps*ps, 1, ps, ps]
        w = torch.zeros((out_channels, 1, ps, ps), dtype=torch.float32)
        for c in range(in_channels):
            base = c * ps * ps
            k = 0
            for ky in range(ps):
                for kx in range(ps):
                    w[base + k, 0, ky, kx] = 1.0
                    k += 1
        conv.weight = torch.nn.Parameter(w, requires_grad=False)
        if isinstance(device, torch.device) and isinstance(dtype, torch.dtype):
            conv = conv.to(device=device, dtype=dtype)
        self._patch_extract_convs[key] = conv
        return conv

    def build_for_channels(self, channels, device: torch.device, dtype: torch.dtype = torch.float32):
        """Pre-create patch extraction convs for known channel counts (FX-safe export/PTQ)."""
        for c in channels:
            self._get_patch_extract_conv(int(c), device=device, dtype=dtype)

    def patchify(self, features, return_spatial_info=False, static_spatial_info=None, in_channels: Optional[int] = None):
        """Convert a tensor into a tensor of respective patches.
        Args:
            features: [bs, c, h, w]
        Returns:
            unfolded_features: [bs, num_patches, c, patchsize, patchsize]
        """
        # Extract patches via grouped conv (see class docstring).
        if in_channels is None:
            # In eager mode, this is a Python int.
            if isinstance(features, Proxy):
                raise RuntimeError("PatchMaker.patchify requires 'in_channels' when tracing with torch.fx.")
            in_channels = int(features.shape[1])
            print(f"PatchMaker.patchify: inferred in_channels={in_channels}")

        if isinstance(features, Proxy):
            # FX tracing: convs must have been pre-built via build_for_channels().
            key = str(int(in_channels))
            if key not in self._patch_extract_convs:
                raise RuntimeError(
                    f"Missing prebuilt patch-extract conv for in_channels={in_channels}. "
                    "Call PatchMaker.build_for_channels([...]) before tracing."
                )
            conv = self._patch_extract_convs[key]
        else:
            conv = self._get_patch_extract_conv(int(in_channels), device=features.device, dtype=features.dtype)
        patches = conv(features)  # [B, C*ps*ps, H', W']

        ps = int(self.patchsize)
        # [B, C*ps*ps, H', W'] -> [B, C, ps, ps, H', W']
        patches = patches.unflatten(1, (int(in_channels), ps * ps))
        patches = patches.unflatten(2, (ps, ps))
        # [B, C, ps, ps, H', W'] -> [B, H', W', C, ps, ps]
        patches = patches.permute(0, 4, 5, 1, 2, 3)
        # [B, H', W', C, ps, ps] -> [B, L, C, ps, ps]
        unfolded_features = patches.flatten(1, 2)

        if return_spatial_info:
            if static_spatial_info is not None:
                return unfolded_features, list(static_spatial_info)

            h = int(features.shape[-2])
            w = int(features.shape[-1])
            n_h = int((h + 2 * self.padding - (self.patchsize - 1) - 1) / self.stride + 1)
            n_w = int((w + 2 * self.padding - (self.patchsize - 1) - 1) / self.stride + 1)
            return unfolded_features, [n_h, n_w]

        return unfolded_features

    def patchify_grid(
        self,
        features,
        in_channels: int,
    ):
        """Extract patch vectors as a channelized 4D tensor.

        Returns a tensor shaped [B, C*ps*ps, H', W'] without flattening patch locations
        into the batch dimension (IMX500/SDSP requirement).
        """
        if isinstance(features, Proxy):
            key = str(int(in_channels))
            if key not in self._patch_extract_convs:
                raise RuntimeError(
                    f"Missing prebuilt patch-extract conv for in_channels={in_channels}. "
                    "Call PatchMaker.build_for_channels([...]) before tracing."
                )
            conv = self._patch_extract_convs[key]
        else:
            conv = self._get_patch_extract_conv(int(in_channels), device=features.device, dtype=features.dtype)
        return conv(features)

    def unpatch_scores(self, x, batchsize):
        """
        FX-safe: no control flow on symbolic values.
        Assumes discriminator output x has shape [B*P, 1].
        Returns: [B, P, 1]
        """
        return x.reshape(batchsize, -1, x.shape[-1])

    def score(self, x):
        """
        FX-safe image-level score from [B, P, 1] -> [B].
        """
        x = x.squeeze(-1)           # [B, P]
        return torch.amax(x, dim=1) # [B]
