"""Turn multichannel EEG into images on the GPU, and the CNN models used on them.

Image layout ("channel-stacked spectrogram"):
  rows  = 18 bipolar channels x 20 frequency rows (1-40 Hz, 2 Hz per row)
  cols  = 64 time columns covering the 8 s window (0.125 s each)
so row r belongs to channel r // 20 and frequency 1 + 2 * (r % 20) Hz. This
1-to-1 mapping lets Grad-CAM / occlusion maps be read back as
"which electrode, which rhythm, when".
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

FS = 128
N_CH, F_ROWS, T_COLS = 18, 20, 64
FREQS = [1 + 2 * i for i in range(F_ROWS)]  # lower edge of each 2 Hz row


def eeg_to_image(x: torch.Tensor) -> torch.Tensor:
    """x (B, 18, 1024) microvolts -> (B, 3, 360, 64) normalised log-spectrogram."""
    B, C, T = x.shape
    win = torch.hann_window(128, device=x.device)
    S = torch.stft(x.reshape(B * C, T).float(), n_fft=128, hop_length=16, window=win,
                   return_complex=True, center=False).abs() ** 2       # (B*C, 65, frames)
    S = S[:, 1:41]                                                     # 1..40 Hz
    S = S.reshape(B * C, F_ROWS, 2, -1).mean(2)                       # 2 Hz rows
    S = torch.log10(S + 1e-3)
    S = S.reshape(B, C * F_ROWS, -1)
    S = (S - S.mean((1, 2), keepdim=True)) / (S.std((1, 2), keepdim=True) + 1e-6)
    S = F.interpolate(S[:, None], size=(C * F_ROWS, T_COLS), mode="bilinear", align_corners=False)
    return S.expand(-1, 3, -1, -1).contiguous()


def resnet18(pretrained: bool) -> nn.Module:
    from torchvision.models import ResNet18_Weights, resnet18 as rn
    m = rn(weights=ResNet18_Weights.DEFAULT if pretrained else None)
    m.fc = nn.Linear(m.fc.in_features, 1)
    return m


class GradCAM:
    """Minimal Grad-CAM on a chosen layer (returns map at input resolution)."""

    def __init__(self, model: nn.Module, layer: nn.Module):
        self.model, self.acts, self.grads = model, None, None
        layer.register_forward_hook(lambda m, i, o: setattr(self, "acts", o))
        layer.register_full_backward_hook(lambda m, gi, go: setattr(self, "grads", go[0]))

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        self.model.zero_grad()
        logit = self.model(img).squeeze(-1)
        logit.sum().backward()
        w = self.grads.mean((2, 3), keepdim=True)
        cam = F.relu((w * self.acts).sum(1, keepdim=True))
        cam = F.interpolate(cam, size=img.shape[-2:], mode="bilinear", align_corners=False)[:, 0]
        cam = cam / (cam.amax((1, 2), keepdim=True) + 1e-8)
        return cam.detach()
