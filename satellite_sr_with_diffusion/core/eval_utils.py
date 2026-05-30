"""Shared evaluation utilities for diffusion-based super-resolution."""
import cv2
import numpy as np
import torch


def upsample_lr_to_hr(lr_tensor: torch.Tensor, hr_tensor: torch.Tensor, method: str) -> torch.Tensor:
    """
    Upsample an LR tensor to match the spatial size of HR using the specified method.
    Inputs are expected in [-1, 1] range with shape [C,H,W].
    Returns a tensor in [-1, 1] with shape [C,H,W].
    """
    # Convert to [0,255] uint8 HWC for OpenCV
    lr_clamped = lr_tensor.clamp(-1, 1)
    lr_01 = lr_clamped.mul(0.5).add(0.5)
    lr_hwc = lr_01.detach().cpu().numpy().transpose(1, 2, 0)
    lr_u8 = (lr_hwc * 255.0).round().astype(np.uint8)
    target_h, target_w = hr_tensor.shape[-2], hr_tensor.shape[-1]
    if method == 'bicubic':
        interp = cv2.INTER_CUBIC
    elif method == 'lanczos':
        interp = cv2.INTER_LANCZOS4
    else:
        raise ValueError(f"Unsupported upsampling method: {method}")
    # OpenCV uses BGR; convert RGB->BGR, resize, then back to RGB
    up_bgr = cv2.resize(lr_u8[:, :, ::-1], (target_w, target_h), interpolation=interp)
    up_rgb = up_bgr[:, :, ::-1]
    # Back to tensor [-1,1], CHW
    up_01 = up_rgb.astype(np.float32) / 255.0
    up_chw = torch.from_numpy(up_01.transpose(2, 0, 1))
    up = up_chw.mul(2.0).sub(1.0)
    return up.to(lr_tensor.device, dtype=lr_tensor.dtype)
