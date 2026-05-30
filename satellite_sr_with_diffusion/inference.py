"""
Pure inference script for diffusion-based super-resolution models.

This script performs inference without calculating metrics, only saving output images.
Based on the validation block from main.py but streamlined for pure inference.

Usage:
    python inference.py -c config/path/val_config.json
"""
import torch
import data as Data
import model as Model
import argparse
import logging
import core.logger as Logger
import core.metrics as Metrics
from core.eval_utils import upsample_lr_to_hr
import os
from data.image_pair_dataset import ImagePairDataset
import cv2
import numpy as np


def setup_datasets(opt):
    """
    Initialize validation dataset.
    Returns:
        val_loader, val_set
    """
    val_loader, val_set = None, None
    
    for dataset_phase, dataset_opt in opt['datasets'].items():
        if dataset_phase == 'val':
            val_set = ImagePairDataset(
                dataroot=dataset_opt['dataroot'],
                datatype=dataset_opt['datatype'],
                split='val',
                data_len=dataset_opt['data_len']
            )
            val_loader = Data.create_dataloader(val_set, dataset_opt, 'val')
            break
    
    return val_loader, val_set


def main():
    """Main inference function."""
    # Set MPS fallback for unsupported operations
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')
    
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, required=True, 
                        help='JSON file for configuration')
    parser.add_argument('-gpu', '--gpu_ids', type=str, default=None)
    parser.add_argument('--val_upsampler', type=str, choices=['model', 'bicubic', 'lanczos'], default='model',
                        help='Choose SR source: model, bicubic, or lanczos.')
    args = parser.parse_args()
    
    # Add required attributes for Logger compatibility
    args.phase = 'val'
    args.enable_wandb = False
    
    # Parse configuration
    opt = Logger.parse(args)
    opt = Logger.dict_to_nonedict(opt)
    
    # Force phase to val for inference
    opt['phase'] = 'val'
    
    # Setup CUDA optimization
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    
    # Setup logging
    Logger.setup_logger('inference', opt['path']['log'], 'inference', level=logging.INFO, screen=True)
    logger = logging.getLogger('inference')
    logger.info(Logger.dict2str(opt))
    
    # Initialize datasets
    val_loader, val_set = setup_datasets(opt)
    logger.info('Dataset initialization completed')
    
    # Initialize model
    diffusion = Model.create_model(opt)
    logger.info('Model initialization completed')

    # Set the correct noise schedule for validation
    diffusion.set_new_noise_schedule(
        opt['model']['beta_schedule']['val'], schedule_phase='val')

    # Setup results directory
    result_path = opt['path']['results']
    os.makedirs(result_path, exist_ok=True)
    
    logger.info('Starting inference...')
    idx = 0

    for batch_idx, val_data in enumerate(val_loader):
        idx += 1
        
        if args.val_upsampler != 'model':
            # Baseline upsampling path (bypass model)
            hr_tensor = val_data['HR']
            lr_tensor = val_data['SR']
            if hr_tensor.dim() == 4:
                hr_tensor = hr_tensor.squeeze(0)
            if lr_tensor.dim() == 4:
                lr_tensor = lr_tensor.squeeze(0)
            sr_tensor_baseline = upsample_lr_to_hr(lr_tensor, hr_tensor, args.val_upsampler)
            visuals = {
                'HR': hr_tensor,
                'SR': sr_tensor_baseline,
                'INF': lr_tensor,
            }
        else:
            # Run model inference
            diffusion.feed_data(val_data)
            diffusion.test(continous=False)
            visuals = diffusion.get_current_visuals(need_LR=False)
        
        # Convert tensors to images
        hr_img = Metrics.tensor2img(visuals['HR'])  # uint8
        sr_img = Metrics.tensor2img(visuals['SR'])  # uint8
        inf_img = Metrics.tensor2img(visuals['INF'])  # Input image
        
        # Save images
        Metrics.save_img(hr_img, f'{result_path}/{idx}_hr.png')
        Metrics.save_img(sr_img, f'{result_path}/{idx}_sr.png')
        Metrics.save_img(inf_img, f'{result_path}/{idx}_lr.png')
        
        if idx % 10 == 0:
            logger.info(f'Processed {idx} images')
    
    logger.info(f'Inference completed. Processed {idx} images.')
    logger.info(f'Results saved to: {result_path}')


if __name__ == "__main__":
    main()
