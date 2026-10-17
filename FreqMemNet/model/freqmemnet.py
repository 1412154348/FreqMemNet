"""RDANet model definition."""

from typing import Sequence

import torch
import torch.nn as nn

from model.modules.ramsad import RAMSAD
from model.modules.pgsm import PGSM


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        assert kernel_size in (3, 7), "kernel size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, use_atten=True):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        if stride != 1 or out_channels != in_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = None
        self.use_atten = use_atten
        if use_atten:
            self.ca = ChannelAttention(out_channels)
            self.sa = SpatialAttention()

    def forward(self, x):
        residual = x
        if self.shortcut is not None:
            residual = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.use_atten:
            out = self.ca(out) * out
            out = self.sa(out) * out
        out += residual
        out = self.relu(out)
        return out

#CALFP
##########################################
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Tuple
from pytorch_wavelets import DWTForward
from pytorch_wavelets import DWTInverse


class ConvModule(nn.Module):
    """Simplified ConvModule (Conv -> BN -> ReLU)"""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, norm=True, activation=True):
        super().__init__()
        layers = [nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=not norm)]
        if norm:
            layers.append(nn.BatchNorm2d(out_channels))
        if activation:
            layers.append(nn.ReLU(inplace=False))
        self.module = nn.Sequential(*layers)

    def forward(self, x):
        return self.module(x)

class ConvDWT(nn.Module):  # DWT: (B,C,H,W) -> (B,4C,H/2,W/2), no parameters are learnable
    def __init__(self, wave='haar', mode='zero'):
        super(ConvDWT, self).__init__()
        # one-level DWT
        self.dwt_forward = DWTForward(J=1, wave=wave, mode=mode)

    def forward(self, x):
        # input size: x (B, C, H, W)
        with torch.cuda.amp.autocast(enabled=False):
            if x.dtype != torch.float32:
                x = x.float()
            Yl, Yh = self.dwt_forward(x)
        b, c, h, w = x.shape
        # Yl (B, C, H/2, W/2) for low-frequency LL
        # List Yh for high-frequency from each level of DWT
        # Yh[0] (B, C, 3, H/2, W/2) for high-frequency LH,HL,HH

        Yh = Yh[0].transpose(1, 2).reshape(Yh[0].shape[0], -1, Yh[0].shape[3], Yh[0].shape[4])

        # output size: output (B, 4C, H/2, W/2)
        output = torch.cat((Yl, Yh), dim=1)
        output = F.interpolate(output, size=(h // 2, w // 2), mode='bilinear', align_corners=False)
        return output

class ConvIDWT(nn.Module):  # IDWT
    def __init__(self, wave='haar', mode='zero'):
        super(ConvIDWT, self).__init__()
        self.dwt_inverse = DWTInverse(wave=wave, mode=mode)

    def forward(self, low_freqs, high_freqs):
        # low_freqs: (B, C, H/2, W/2)
        # high_freqs: (B, 3C, H/2, W/2)
        B, C, H, W = low_freqs.shape

        high_freqs = high_freqs.reshape(B, C, 3, H, W)

        with torch.cuda.amp.autocast(enabled=False):
            reconstruction = self.dwt_inverse((low_freqs, [high_freqs.float()]))
        reconstruction = F.interpolate(reconstruction, size=(2 * H, 2 * W), mode='bilinear', align_corners=False)

        return reconstruction

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7, bn_before_sigmoid=False):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.bn_before_sigmoid = bn_before_sigmoid
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        if bn_before_sigmoid:
            self.bn = nn.BatchNorm2d(1)
            self.bn.bias.data.fill_(0)
            self.bn.bias.requires_grad = False
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)

        if self.bn_before_sigmoid:
            x = self.bn(x)
        return self.sigmoid(x)

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1   = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class LearnableGaussianFilterBank(nn.Module):
    def __init__(self, kernel_size, num_filters, num_channels):
        super(LearnableGaussianFilterBank, self).__init__()
        self.kernel_size = kernel_size
        self.num_filters = num_filters
        self.C = num_channels
        self.padding = kernel_size // 2  # Padding size to maintain input size

        # Create learnable parameters for sigmas
        self.sigmas = nn.ParameterList([nn.Parameter(torch.tensor([1.0])) for _ in range(num_filters)])

    def forward(self, x):
        # Apply Gaussian filters using convolution
        weights = [self._gaussian_kernel(self.kernel_size, sigma).repeat(self.C, 1, 1, 1) for sigma in self.sigmas]
        filtered_outputs = [F.conv2d(F.pad(x, (self.padding,self.padding,self.padding,self.padding), mode='replicate')
                                     , weight.to(x.device), groups=self.C) for weight in weights]
        return torch.cat(filtered_outputs, dim=1)

    def _gaussian_kernel(self, kernel_size, sigma):
        # Create a 2D Gaussian kernel with learnable sigma as tensors
        kernel = torch.zeros(1, 1, kernel_size, kernel_size)
        center = kernel_size // 2
        for i in range(kernel_size):
            for j in range(kernel_size):
                kernel[:, :, i, j] = torch.exp(-((i - center) ** 2 + (j - center) ** 2) / (2 * sigma ** 2))

        return kernel / kernel.sum() # normalization
# CALFP Module:
class wav_Enhance(nn.Module):
    def __init__(self, in_channels, out_channels=None, wave='haar', mode='symmetric',
                 with_gauss=True, gauss_gate=0.5):
        super(wav_Enhance, self).__init__()
        self.dwt = ConvDWT(wave=wave, mode=mode)
        self.idwt = ConvIDWT(wave=wave, mode=mode)
        self.with_gauss = with_gauss
        self.gauss_gate = gauss_gate

        self.out_channels = out_channels if out_channels is not None else in_channels

        self.attention = SpatialAttention()
        if self.with_gauss:
            self.gaussian_filter = LearnableGaussianFilterBank(
                kernel_size=3, num_filters=1, num_channels=3 * in_channels
            )

        # 通道变换
        if self.out_channels != in_channels:
            self.channel_conv = nn.Conv2d(in_channels, self.out_channels, kernel_size=1)
        else:
            self.channel_conv = None

    def forward(self, x):
        B, C, H, W = x.shape
        dwt_out = self.dwt(x)                    # (B, 4C, H/2, W/2)
        LL = dwt_out[:, :C, :, :]
        Yh = dwt_out[:, C:, :, :]
        att = self.attention(LL)
        Yh = Yh * att
        if self.with_gauss:
            Yh_blurred = self.gaussian_filter(Yh)
            mask = (Yh.abs() < self.gauss_gate).float()
            Yh = Yh * (1 - mask) + Yh_blurred * mask
        x_rec = self.idwt(LL, Yh)                # (B, C, H, W)

        if self.channel_conv is not None:
            x_rec = self.channel_conv(x_rec)     # (B, out_channels, H, W)
        return x_rec

#########################################




#####
# class freqmemnet(nn.Module):
#     def __init__(
#         self,
#         input_channels,
#         in_size,
#         layers=4,
#         factor=2,
#         block_factor=1,
#         block=ResidualBlock,
#         mem_slots: Sequence[int] = (256, 128, 64, 64),
#         patch: Sequence[int] = (16, 16, 8, 8),
#         topk: Sequence[int] = (16, 16, 8, 8),
#         drop_out: Sequence[float] = (0.0, 0.0, 0.0, 0.0),
#         use_overlap: Sequence[bool] = (True, True, False, False),
#     ):
#         super().__init__()
#         assert block is not None, "Please provide a residual block class"
#
#         base_channels = [8, 16, 32, 64, 128, 256]
#         assert layers <= len(base_channels), "layers exceeds the channel table"
#         self.block_factor = block_factor
#         chs = [c * factor for c in base_channels[:layers]]
#
#         self.skip_layers = nn.ModuleList(
#             PGSM(
#                 in_channels=chs[i],
#                 in_size=in_size // (2**i),
#                 patch=patch[i],
#                 mem_slots=mem_slots[i],
#                 topk=topk[i],
#                 mid_channels=max(16, chs[i] // 2),
#                 use_overlap=use_overlap[i],
#             )
#             for i in range(layers - 1)
#         )
#
#         self.down_layers = nn.ModuleList(
#             MSAD(in_ch=chs[i]) for i in range(layers - 1)
#         )
#         self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
#         self.conv_init = nn.Conv2d(input_channels, chs[0], kernel_size=1, stride=1)
#
#         self.encoder = nn.ModuleList()
#         self.encoder.append(self._make_layer(chs[0], chs[0], block))
#         for i in range(layers - 2):
#             self.encoder.append(
#                 self._make_layer(chs[i], chs[i + 1], block, block_num=2)
#             )
#
#         self.bottleneck = self._make_layer(
#             chs[layers - 2], chs[layers - 1], block, block_num=2
#         )
#
#         self.decoder = nn.ModuleList()
#         for i in range(layers - 2, 0, -1):
#             in_ch = chs[i] + chs[i + 1]
#             out_ch = chs[i]
#             self.decoder.append(
#                 self._make_layer(in_ch, out_ch, block, block_num=2)
#             )
#         self.decoder.append(
#             self._make_layer(chs[0] + chs[1], chs[0], block, block_num=1)
#         )
#         self.head = nn.Conv2d(chs[0], 1, 3, 1, 1)
#
#     def _make_layer(self, in_channels, out_channels, block, block_num=1):
#         block_num = block_num * self.block_factor
#         layers = [block(in_channels, out_channels)]
#         for _ in range(block_num - 1):
#             layers.append(block(out_channels, out_channels))
#         return nn.Sequential(*layers)
#
#     def forward(self, x):
#         skips = []
#         x = self.conv_init(x)
#         for idx, layer in enumerate(self.encoder):
#             x = layer(x)
#             skip = x
#             if idx < len(self.skip_layers):
#                 skips.append(self.skip_layers[idx](skip))
#             else:
#                 skips.append(skip)
#             x = self.down_layers[idx](x)
#
#         x = self.bottleneck(x)
#
#         for layer in self.decoder:
#             skip = skips.pop()
#             x = self.up(x)
#             x = torch.cat([skip, x], dim=1)
#             x = layer(x)
#
#         return self.head(x)
#####


####
class FreqMemNet(nn.Module):
    def __init__(
        self,
        input_channels,
        in_size,
        layers=4,
        factor=2,
        block_factor=1,
        # block=ResidualBlock,
        block=wav_Enhance,
        mem_slots: Sequence[int] = (256, 128, 64, 64),
        patch: Sequence[int] = (16, 16, 8, 8),
        topk: Sequence[int] = (16, 16, 8, 8),
        drop_out: Sequence[float] = (0.0, 0.0, 0.0, 0.0),
        use_overlap: Sequence[bool] = (True, True, False, False),
    ):
        super().__init__()
        assert block is not None, "Please provide a residual block class"

        base_channels = [8, 16, 32, 64, 128, 256]
        assert layers <= len(base_channels), "layers exceeds the channel table"
        self.block_factor = block_factor
        chs = [c * factor for c in base_channels[:layers]]

        self.skip_layers = nn.ModuleList(
            PGSM(
                in_channels=chs[i],
                in_size=in_size // (2**i),
                patch=patch[i],
                mem_slots=mem_slots[i],
                topk=topk[i],
                mid_channels=max(16, chs[i] // 2),
                use_overlap=use_overlap[i],
            )
            for i in range(layers - 1)
        )

        self.down_layers = nn.ModuleList(
            RAMSAD(in_ch=chs[i]) for i in range(layers - 1)
        )
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv_init = nn.Conv2d(input_channels, chs[0], kernel_size=1, stride=1)

        self.encoder = nn.ModuleList()
        self.encoder.append(self._make_layer(chs[0], chs[0], block))
        for i in range(layers - 2):
            self.encoder.append(
                self._make_layer(chs[i], chs[i + 1], block, block_num=1)
            )

        self.bottleneck = self._make_layer(
            chs[layers - 2], chs[layers - 1], block, block_num=1
        )

        self.decoder = nn.ModuleList()
        for i in range(layers - 2, 0, -1):
            in_ch = chs[i] + chs[i + 1]
            out_ch = chs[i]
            self.decoder.append(
                self._make_layer(in_ch, out_ch, block, block_num=1)
            )
        self.decoder.append(
            self._make_layer(chs[0] + chs[1], chs[0], block, block_num=1)
        )
        self.head = nn.Conv2d(chs[0], 1, 3, 1, 1)

    def _make_layer(self, in_channels, out_channels, block, block_num=1):
        block_num = block_num * self.block_factor
        layers = [block(in_channels, out_channels)]
        # layers = [block(in_channels)]
        for _ in range(block_num - 1):
            layers.append(block(out_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        skips = []
        x = self.conv_init(x)
        for idx, layer in enumerate(self.encoder):
            x = layer(x)
            skip = x
            if idx < len(self.skip_layers):
                skips.append(self.skip_layers[idx](skip))
            else:
                skips.append(skip)
            x = self.down_layers[idx](x)

        x = self.bottleneck(x)

        for layer in self.decoder:
            skip = skips.pop()
            x = self.up(x)
            x = torch.cat([skip, x], dim=1)
            x = layer(x)

        return self.head(x)
#####
__all__ = ["RDANet", "ResidualBlock", "ChannelAttention", "SpatialAttention"]


from thop import profile
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FreqMemNet(3,512).to(device)
    test_img = torch.randn(1, 3, 512, 512).to(device)

    # 计算 FLOPs、参数量
    flops, params = profile(model, inputs=(test_img,), verbose=False)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"总参数量: {total_params / 1e6:.6f} M")
    print(f"可训练参数量: {trainable_params / 1e6:.6f} M")
    print(f"计算量 FLOPs: {flops/1e9:.6f} G")
    print(f"总参数量: {params / 1e6:.6f} M")