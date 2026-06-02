"""
Shared inference helpers.

Used by both entry points so the model-inference + image-saving logic lives in
one place: ``main.py`` (training / evaluation) and ``inference.py`` (inference
only) both build on these.
"""

import os

import torch
from torchmetrics.image.fid import FrechetInceptionDistance

import core.metrics as Metrics
from core.eval_utils import upsample_lr_to_hr


def build_fid(feature: int = 2048):
    """Create a FID metric on cuda > mps > cpu, with an MPS->CPU fallback."""
    device = (
        "cuda"
        if torch.cuda.is_available()
        else (
            "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu"
        )
    )
    if device == "mps":
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    try:
        return FrechetInceptionDistance(feature=feature).to(device)
    except (TypeError, RuntimeError):
        return FrechetInceptionDistance(feature=feature).to("cpu")


def compute_visuals(diffusion, val_data, val_upsampler="model"):
    """Return {'HR','SR','INF'} for one sample via the model or a baseline upsampler."""
    if val_upsampler != "model":
        hr_tensor = val_data["HR"]
        lr_tensor = val_data["SR"]
        if hr_tensor.dim() == 4:
            hr_tensor = hr_tensor.squeeze(0)
        if lr_tensor.dim() == 4:
            lr_tensor = lr_tensor.squeeze(0)
        sr_tensor = upsample_lr_to_hr(lr_tensor, hr_tensor, val_upsampler)
        return {"HR": hr_tensor, "SR": sr_tensor, "INF": lr_tensor}
    diffusion.feed_data(val_data)
    diffusion.test(continous=False)
    return diffusion.get_current_visuals(need_LR=False)


def save_triplet(visuals, result_path, idx):
    """Save the HR/SR/LR PNGs for one sample; return (sr_img, hr_img) as uint8 arrays."""
    hr_img = Metrics.tensor2img(visuals["HR"])
    sr_img = Metrics.tensor2img(visuals["SR"])
    inf_img = Metrics.tensor2img(visuals["INF"])
    Metrics.save_img(hr_img, f"{result_path}/{idx}_hr.png")
    Metrics.save_img(sr_img, f"{result_path}/{idx}_sr.png")
    Metrics.save_img(inf_img, f"{result_path}/{idx}_lr.png")
    return sr_img, hr_img
