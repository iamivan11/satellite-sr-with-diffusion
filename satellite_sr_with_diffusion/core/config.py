"""
Hydra/OmegaConf glue.

Turns a composed Hydra `DictConfig` into the plain ``opt`` dictionary the rest
of the codebase already expects, registering the custom resolvers that derive
resolution-dependent model parameters from a single ``hr_size`` knob.
"""

import os

from omegaconf import DictConfig, OmegaConf

import core.logger as Logger


def _attn_res(hr_size, div=1):
    """Attention resolutions for a 5-level U-Net: the two lowest feature maps.

    The U-Net (channel_mults of length 5) downsamples 4 times, so the spatial
    size at the two deepest levels is image_size/8 and image_size/16, where
    image_size = hr_size / div (div=2 for wavelet models, 1 for pixel models).
    Override ``model.unet.attn_res`` explicitly if you change ``channel_mults``.
    """
    image_size = int(hr_size) // int(div)
    return [image_size // 8, image_size // 16]


def register_resolvers():
    """Register OmegaConf resolvers used by the config files (idempotent)."""
    OmegaConf.register_new_resolver("floordiv", lambda a, b: int(a) // int(b), replace=True)
    OmegaConf.register_new_resolver("attn_res", _attn_res, replace=True)


# Register on import so they are available before any config is resolved.
register_resolvers()


def build_opt(cfg: DictConfig) -> dict:
    """Resolve a Hydra config into the runtime ``opt`` dict and set up the run.

    Handles interpolation/resolver resolution, GPU visibility, the
    ``distributed``/``enable_wandb`` flags, and experiment directory creation.
    """
    opt = OmegaConf.to_container(cfg, resolve=True)

    # GPU selection (mirrors the historical JSON behaviour).
    gpu_ids = opt.get("gpu_ids") or []
    opt["gpu_ids"] = list(gpu_ids)
    if opt["gpu_ids"]:
        gpu_list = ",".join(str(x) for x in opt["gpu_ids"])
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_list
        print("export CUDA_VISIBLE_DEVICES=" + gpu_list)
    opt["distributed"] = len(opt["gpu_ids"]) > 1

    opt["enable_wandb"] = bool((opt.get("wandb") or {}).get("enable", False))

    # Validate the upsampler choice (Hydra accepts any string, so check here).
    valid_upsamplers = ("model", "bicubic", "lanczos")
    if opt.get("val_upsampler", "model") not in valid_upsamplers:
        raise ValueError(
            f"val_upsampler must be one of {valid_upsamplers}, got {opt.get('val_upsampler')!r}"
        )

    # Create experiment/result/log directories and finalise the path block.
    Logger.setup_paths(opt)
    return opt
