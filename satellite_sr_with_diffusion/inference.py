"""
Pure inference script for diffusion-based super-resolution models.

This script performs inference without calculating metrics, only saving output images.
Based on the validation block from main.py but streamlined for pure inference.

Usage:
    python inference.py data=sen2venus_4x model=wave path.resume_state=/path/to/ckpt
"""

import logging
import os

import core.config as Config
import core.engine as Engine
import core.logger as Logger
import data as Data
import hydra
import model as Model
import torch
from data.image_pair_dataset import ImagePairDataset
from omegaconf import DictConfig


def setup_datasets(opt):
    """
    Initialize validation dataset.
    Returns:
        val_loader, val_set
    """
    val_loader, val_set = None, None

    for dataset_phase, dataset_opt in opt["datasets"].items():
        if dataset_phase == "val":
            val_set = ImagePairDataset(
                dataroot=dataset_opt["dataroot"],
                datatype=dataset_opt["datatype"],
                split="val",
                data_len=dataset_opt["data_len"],
            )
            val_loader = Data.create_dataloader(val_set, dataset_opt, "val")
            break

    return val_loader, val_set


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    """Main inference function (Hydra entry point)."""
    # Set MPS fallback for unsupported operations
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

    # Force phase to val for inference, then resolve the config into ``opt``.
    cfg.phase = "val"
    opt = Config.build_opt(cfg)
    opt = Logger.dict_to_nonedict(opt)

    # Setup CUDA optimization
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    # Setup logging
    Logger.setup_logger(
        "inference", opt["path"]["log"], "inference", level=logging.INFO, screen=True
    )
    logger = logging.getLogger("inference")
    logger.info(Logger.dict2str(opt))

    # Initialize datasets
    val_loader, _val_set = setup_datasets(opt)
    logger.info("Dataset initialization completed")

    # Initialize model
    diffusion = Model.create_model(opt)
    logger.info("Model initialization completed")

    # Set the correct noise schedule for validation
    diffusion.set_new_noise_schedule(opt["model"]["beta_schedule"]["val"], schedule_phase="val")

    # Setup results directory
    result_path = opt["path"]["results"]
    os.makedirs(result_path, exist_ok=True)

    logger.info("Starting inference...")
    idx = 0

    for _batch_idx, val_data in enumerate(val_loader):
        idx += 1
        visuals = Engine.compute_visuals(diffusion, val_data, opt["val_upsampler"])
        Engine.save_triplet(visuals, result_path, idx)

        if idx % 10 == 0:
            logger.info(f"Processed {idx} images")

    logger.info(f"Inference completed. Processed {idx} images.")
    logger.info(f"Results saved to: {result_path}")


if __name__ == "__main__":
    main()
