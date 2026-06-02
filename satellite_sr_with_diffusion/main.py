"""
Main training script for diffusion-based super-resolution models on satellite imagery datasets.

This script handles:
- Configuration parsing and validation
- Dataset initialization with generic HR/LR pair loader
- Model training with periodic validation (supports SR3 and Wave/DiWa architectures)
- Metrics calculation (PSNR, SSIM, LPIPS, FID)
- Checkpoint saving and loss visualization
- Weights & Biases integration
- Baseline upsampling comparison (bicubic/lanczos)
"""

import logging
import os
import random

import core.config as Config
import core.engine as Engine
import core.logger as Logger
import core.metrics as Metrics
import data as Data
import hydra
import model as Model
import torch
import wandb
from data.image_pair_dataset import ImagePairDataset
from omegaconf import DictConfig
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity


def setup_datasets(opt: dict, phase: str) -> tuple:
    """
    Initialize train and validation datasets.

    Returns:
        tuple: (train_loader, val_loader, val_indices) or appropriate subset
    """
    train_loader, val_loader, val_indices = None, None, None

    for dataset_phase, dataset_opt in opt["datasets"].items():
        if dataset_phase == "train" and phase == "train":
            train_set = ImagePairDataset(
                dataroot=dataset_opt["dataroot"],
                datatype=dataset_opt["datatype"],
                split=dataset_phase,
                data_len=dataset_opt["data_len"],
            )
            train_loader = Data.create_dataloader(train_set, dataset_opt, dataset_phase)

        elif dataset_phase == "val":
            val_set = ImagePairDataset(
                dataroot=dataset_opt["dataroot"],
                datatype=dataset_opt["datatype"],
                split="val",  # Always use 'val' split for validation dataset
                data_len=dataset_opt["data_len"],
            )
            val_loader = Data.create_dataloader(val_set, dataset_opt, "val")

            # Select random validation indices reproducibly (only for training phase)
            if phase == "train":
                n_val = len(val_set)
                if "data_len" in dataset_opt and dataset_opt["data_len"] > 0:
                    n_val = min(dataset_opt["data_len"], len(val_set))

                random.seed(42)  # Fixed seed for reproducible validation
                val_indices = random.sample(range(len(val_set)), n_val)
            else:
                val_indices = None  # Not needed for inference phase

    return train_loader, val_loader, val_indices, val_set if "val_set" in locals() else None


def run_validation(
    diffusion,
    val_set,
    val_indices: list[int],
    lpips_metric,
    result_path: str,
    current_step: int,
    logger,
) -> dict[str, float]:
    """
    Run validation and calculate metrics.

    Returns:
        dict: Contains avg_psnr, avg_ssim, avg_lpips
    """
    metrics = {"psnr": 0.0, "ssim": 0.0, "lpips": 0.0}
    idx = 0

    fid = Engine.build_fid()

    for val_idx in val_indices:
        val_data = val_set[val_idx]
        # Add batch dimension for model compatibility where needed
        val_data["SR"] = val_data["SR"].unsqueeze(0)
        val_data["HR"] = val_data["HR"].unsqueeze(0)
        idx += 1

        visuals = Engine.compute_visuals(diffusion, val_data)
        sr_img, hr_img = Engine.save_triplet(visuals, result_path, idx)

        # Calculate metrics
        metrics["psnr"] += Metrics.calculate_psnr(sr_img, hr_img)
        metrics["ssim"] += Metrics.calculate_ssim(sr_img, hr_img)

        # LPIPS expects [N,3,H,W] in [-1,1]
        sr_tensor = visuals["SR"]
        hr_tensor = visuals["HR"]
        if sr_tensor.dim() == 3:
            sr_tensor = sr_tensor.unsqueeze(0)
        if hr_tensor.dim() == 3:
            hr_tensor = hr_tensor.unsqueeze(0)
        if sr_tensor.dim() == 5:
            sr_tensor = sr_tensor.squeeze(0)
        if hr_tensor.dim() == 5:
            hr_tensor = hr_tensor.squeeze(0)
        sr_tensor = sr_tensor.clamp(-1, 1).to(lpips_metric.device)
        hr_tensor = hr_tensor.clamp(-1, 1).to(lpips_metric.device)
        metrics["lpips"] += lpips_metric(sr_tensor, hr_tensor).item()

        # sr_tensor = torch.from_numpy(sr_img.transpose(2,0,1)).unsqueeze(0)
        #     .to(diffusion.device, dtype=torch.uint8)
        # hr_tensor = torch.from_numpy(hr_img.transpose(2,0,1)).unsqueeze(0)
        #     .to(diffusion.device, dtype=torch.uint8)

        fid_dev = next(fid.parameters()).device
        sr_tensor = (
            torch.from_numpy(sr_img.transpose(2, 0, 1)).unsqueeze(0).to(fid_dev, dtype=torch.uint8)
        )
        hr_tensor = (
            torch.from_numpy(hr_img.transpose(2, 0, 1)).unsqueeze(0).to(fid_dev, dtype=torch.uint8)
        )
        fid.update(hr_tensor, real=True)
        fid.update(sr_tensor, real=False)

    # Average metrics
    for key in metrics:
        metrics[key] /= idx

    metrics["fid"] = fid.compute().to(torch.float32).item()

    logger.info(
        f"# Validation # PSNR: {metrics['psnr']:.4e}, "
        f"SSIM: {metrics['ssim']:.4e}, LPIPS: {metrics['lpips']:.4e}, FID: {metrics['fid']:.4e}"
    )

    return metrics


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    """Main training/validation function (Hydra entry point)."""
    # Resolve the composed config into the runtime ``opt`` dict.
    opt = Config.build_opt(cfg)
    opt = Logger.dict_to_nonedict(opt)

    # Initialize wandb if enabled
    if opt["enable_wandb"]:
        wandb.init(
            project=opt["wandb"].get("project", "satellite_sr"), name=opt["name"], config=opt
        )

    # Setup CUDA optimization
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    # Setup logging
    if opt["phase"] == "train":
        Logger.setup_logger(None, opt["path"]["log"], "train", level=logging.INFO, screen=True)
        Logger.setup_logger("val", opt["path"]["log"], "val", level=logging.INFO)
    else:
        Logger.setup_logger("val", opt["path"]["log"], "val", level=logging.INFO, screen=True)
    logger = logging.getLogger("base")
    logger.info(Logger.dict2str(opt))

    # Initialize datasets
    train_loader, val_loader, val_indices, val_set = setup_datasets(opt, opt["phase"])
    logger.info("Dataset initialization completed")

    # Initialize model
    diffusion = Model.create_model(opt)
    logger.info("Model initialization completed")

    # Set the correct noise schedule for the current phase
    if opt["phase"] == "val":
        diffusion.set_new_noise_schedule(opt["model"]["beta_schedule"]["val"], schedule_phase="val")

    # Initialize LPIPS metric for validation
    lpips_metric = LearnedPerceptualImagePatchSimilarity(net_type="vgg")
    lpips_metric = lpips_metric.to(diffusion.device).eval()

    # Set LPIPS in model if used for training loss
    if opt["phase"] == "train" and opt["train"].get("use_lpips", False):
        diffusion.set_lpips_metric(lpips_metric)

    if opt["phase"] == "train":
        # Initialize training state
        current_step = diffusion.begin_step
        current_epoch = diffusion.begin_epoch
        n_iter = opt["train"]["n_iter"]
        train_losses = []

        if opt["path"]["resume_state"]:
            logger.info(f"Resuming training from epoch: {current_epoch}, iter: {current_step}.")

        # Set noise schedule for training
        diffusion.set_new_noise_schedule(
            opt["model"]["beta_schedule"]["train"], schedule_phase="train"
        )

        logger.info(f"Starting training for {n_iter} iterations")

        while current_step < n_iter:
            current_epoch += 1

            for _i, train_data in enumerate(train_loader):
                current_step += 1
                if current_step > n_iter:
                    break

                # Training step
                diffusion.feed_data(train_data)
                diffusion.optimize_parameters(current_step)

                # Logging
                if current_step % opt["train"]["print_freq"] == 0:
                    logs = diffusion.get_current_log()
                    message = f"<epoch:{current_epoch:3d}, iter:{current_step:8,d}> "
                    for k, v in logs.items():
                        message += f"{k}: {v:.4e} "

                    if "l_pix" in logs:
                        train_losses.append(logs["l_pix"])

                    logger.info(message)

                    # Wandb logging
                    if opt["enable_wandb"]:
                        wandb_log = {"step": current_step}
                        if "l_pix" in logs:
                            wandb_log["train_loss"] = logs["l_pix"]
                        if "l_lpips" in logs:
                            wandb_log["train_lpips_loss"] = logs["l_lpips"]
                        wandb.log(wandb_log)

                # Checkpoint saving
                if current_step % opt["train"]["save_checkpoint_freq"] == 0:
                    logger.info("Saving checkpoint...")
                    diffusion.save_network(current_epoch, current_step)

                # Validation
                if current_step % opt["train"]["val_freq"] == 0:
                    logger.info("Running validation...")
                    result_path = f"{opt['path']['results']}/{current_epoch}"
                    os.makedirs(result_path, exist_ok=True)

                    # Switch to validation noise schedule
                    diffusion.set_new_noise_schedule(
                        opt["model"]["beta_schedule"]["val"], schedule_phase="val"
                    )

                    # Run validation
                    metrics = run_validation(
                        diffusion,
                        val_set,
                        val_indices,
                        lpips_metric,
                        result_path,
                        current_step,
                        logger,
                    )

                    # Log validation metrics
                    logger_val = logging.getLogger("val")
                    logger_val.info(
                        "<epoch:{:3d}, iter:{:8,d}> psnr: {:.4e}, ssim: {:.4e}, "
                        "lpips: {:.4e}, fid: {:.4e}".format(
                            current_epoch,
                            current_step,
                            metrics["psnr"],
                            metrics["ssim"],
                            metrics["lpips"],
                            metrics["fid"],
                        )
                    )

                    if opt["enable_wandb"]:
                        wandb.log(
                            {
                                "val_psnr": metrics["psnr"],
                                "val_ssim": metrics["ssim"],
                                "val_lpips": metrics["lpips"],
                                "val_fid": metrics["fid"],
                                "step": current_step,
                            }
                        )

                    # Switch back to training noise schedule
                    diffusion.set_new_noise_schedule(
                        opt["model"]["beta_schedule"]["train"], schedule_phase="train"
                    )

        logger.info("End of training.")

    else:
        # Validation/Inference phase
        logger.info("Starting inference...")

        # Setup results directory
        result_path = opt["path"]["results"]
        os.makedirs(result_path, exist_ok=True)

        # Run evaluation on the full validation set
        avg_psnr = 0.0
        avg_ssim = 0.0
        avg_lpips = 0.0
        idx = 0
        fid = Engine.build_fid()

        for _batch_idx, val_data in enumerate(val_loader):
            idx += 1
            visuals = Engine.compute_visuals(diffusion, val_data, opt["val_upsampler"])
            sr_img, hr_img = Engine.save_triplet(visuals, result_path, idx)

            avg_psnr += Metrics.calculate_psnr(sr_img, hr_img)
            avg_ssim += Metrics.calculate_ssim(sr_img, hr_img)

            sr_tensor = visuals["SR"].unsqueeze(0).clamp(-1, 1).to(lpips_metric.device)
            hr_tensor = visuals["HR"].squeeze(0).unsqueeze(0).clamp(-1, 1).to(lpips_metric.device)
            avg_lpips += lpips_metric(sr_tensor, hr_tensor).item()

            fid_dev = next(fid.parameters()).device
            fid.update(
                torch.from_numpy(hr_img.transpose(2, 0, 1))
                .unsqueeze(0)
                .to(fid_dev, dtype=torch.uint8),
                real=True,
            )
            fid.update(
                torch.from_numpy(sr_img.transpose(2, 0, 1))
                .unsqueeze(0)
                .to(fid_dev, dtype=torch.uint8),
                real=False,
            )

            if idx % 10 == 0:
                logger.info(
                    f"Progress {idx}: Avg PSNR: {avg_psnr / idx:.4f}, "
                    f"Avg SSIM: {avg_ssim / idx:.4f}, Avg LPIPS: {avg_lpips / idx:.4f}"
                )

        avg_psnr /= idx
        avg_ssim /= idx
        avg_lpips /= idx
        avg_fid = fid.compute().item()

        logger.info("=" * 50)
        logger.info("Final Results:")
        logger.info(f"Average PSNR: {avg_psnr:.4e}")
        logger.info(f"Average SSIM: {avg_ssim:.4e}")
        logger.info(f"Average LPIPS: {avg_lpips:.4e}")
        logger.info(f"Average FID: {avg_fid:.4e}")
        logger.info(f"Results saved to: {result_path}")

        # Save metrics to file
        with open(f"{result_path}/metrics.txt", "w") as f:
            f.write(f"Average PSNR: {avg_psnr:.4e}\n")
            f.write(f"Average SSIM: {avg_ssim:.4e}\n")
            f.write(f"Average LPIPS: {avg_lpips:.4e}\n")
            f.write(f"Average FID: {avg_fid:.4e}\n")
            f.write(f"Number of images: {idx}\n")

        logger.info("End of inference.")


if __name__ == "__main__":
    main()
