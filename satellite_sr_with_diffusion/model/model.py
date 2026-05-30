import logging
from collections import OrderedDict
import torch
import torch.nn as nn
import os
import model.networks as networks
from .base_model import BaseModel

logger = logging.getLogger('base')


class DDPM(BaseModel):
    """Denoising Diffusion Probabilistic Model"""
    def __init__(self, opt):
        super(DDPM, self).__init__(opt)
        self.netG = self.set_device(networks.define_G(opt))
        self.schedule_phase = None
        self.set_loss()
        # Always initialize with train schedule to match saved checkpoints
        self.set_new_noise_schedule(
            opt['model']['beta_schedule']['train'], schedule_phase='train')
        if self.opt['phase'] == 'train':
            self.netG.train()
            optim_params = list(self.netG.parameters())
            if opt['train']["optimizer"]["type"] == "adamw":
                self.optG = torch.optim.AdamW(
                    optim_params, lr=opt['train']["optimizer"]["lr"],
                    weight_decay=opt['train']["optimizer"]["weight_decay"])
            else:
                 self.optG = torch.optim.Adam(
                    optim_params, lr=opt['train']["optimizer"]["lr"])
            self.log_dict = OrderedDict()
        self.load_network()
        pytorch_total_params = sum(p.numel() for p in self.netG.parameters())
        logger.info(f"params: {pytorch_total_params}")


    def feed_data(self, data):
        self.data = self.set_device(data)


    def _compute_loss(self, l_dict, use_lpips):
        b, c, h, w = self.data['HR'].shape
        denom = int(b * c * h * w)
        
        raw_l_pix = l_dict if isinstance(l_dict, torch.Tensor) else l_dict['loss']
        l_pix = raw_l_pix.sum() / denom  # Now mean per pixel
        
        if use_lpips:
            x_recon = l_dict['x_recon'].clamp(-1, 1)
            l_lpips = self.lpips_metric(x_recon, self.data['HR'])
            lpips_weight = self.opt['train'].get('lpips_weight', 0.05)
            loss = l_pix + lpips_weight * l_lpips
            self.log_dict['l_lpips'] = l_lpips.item()
        else:
            loss = l_pix

        return loss, l_pix


    def optimize_parameters(self, current_step=0):
        self.optG.zero_grad()
        
        use_lpips = self.opt['train'].get('use_lpips', False) and hasattr(self, 'lpips_metric')
        
        l_dict = self.netG(self.data, return_reconstruction=use_lpips)
        
        loss, l_pix = self._compute_loss(l_dict, use_lpips)
        
        loss.backward()

        # Gradient clipping
        if self.opt['train'].get('gradient_clipping', False):
            clip_value = self.opt['train'].get('clip_value', 1.0)
            nn.utils.clip_grad_norm_(self.netG.parameters(), clip_value)

        self.optG.step()
        
        self.log_dict['l_pix'] = l_pix.item()


    def test(self, continous=False, sample_inter=None):
        """Performs inference on validation data."""
        self.netG.eval()
        with torch.no_grad():
            if isinstance(self.netG, nn.DataParallel):
                self.SR = self.netG.module.super_resolution(self.data['SR'], continous, sample_inter)
            else:
                self.SR = self.netG.super_resolution(self.data['SR'], continous, sample_inter)
        self.netG.train()


    def set_loss(self):
        """Sets the loss function for the model."""
        if isinstance(self.netG, nn.DataParallel):
            self.netG.module.set_loss(self.device)
        else:
            self.netG.set_loss(self.device)


    def set_new_noise_schedule(self, schedule_opt, schedule_phase='train'):
        """Sets the noise schedule for the diffusion process."""
        if self.schedule_phase is None or self.schedule_phase != schedule_phase:
            self.schedule_phase = schedule_phase
            if isinstance(self.netG, nn.DataParallel):
                self.netG.module.set_new_noise_schedule(schedule_opt, self.device)
            else:
                self.netG.set_new_noise_schedule(schedule_opt, self.device)


    def set_lpips_metric(self, lpips_metric):
        self.lpips_metric = lpips_metric.to(self.device)


    def get_current_log(self):
        return self.log_dict


    def get_current_visuals(self, need_LR=True, sample=False):
        out_dict = OrderedDict()
        out_dict['SR'] = self.SR.detach().float().cpu()
        out_dict['INF'] = self.data['SR'].detach().float().cpu()
        out_dict['HR'] = self.data['HR'].detach().float().cpu()
        return out_dict


    def print_network(self):
        s, n = self.get_network_description(self.netG)
        if isinstance(self.netG, nn.DataParallel):
            net_struc_str = f'{self.netG.__class__.__name__} - {self.netG.module.__class__.__name__}'
        else:
            net_struc_str = f'{self.netG.__class__.__name__}'
        logger.info(f'Network G structure: {net_struc_str}, with parameters: {n:,d}')


    def save_network(self, epoch, iter_step):
        """Saves the generator and optimizer states to disk with error handling."""
        gen_path = os.path.join(self.opt['path']['checkpoint'], f'I{iter_step}_E{epoch}_gen.pth')
        opt_path = os.path.join(self.opt['path']['checkpoint'], f'I{iter_step}_E{epoch}_opt.pth')
        
        # Ensure the checkpoint directory exists
        os.makedirs(self.opt['path']['checkpoint'], exist_ok=True)
        
        network = self.netG
        if isinstance(self.netG, nn.DataParallel):
            network = network.module
            
        state_dict = network.state_dict()
        for key, param in state_dict.items():
            state_dict[key] = param.cpu()
            
        opt_state = {'epoch': epoch, 'iter': iter_step, 'optimizer': self.optG.state_dict()}
        
        try:
            torch.save(state_dict, gen_path)
            torch.save(opt_state, opt_path)
            logger.info(f'Saved model in [{gen_path}] ...')
        except Exception as e:
            logger.warning(f"Could not save checkpoint at iteration {iter_step}: {e}")


    def load_network(self):
        """Loads a pretrained model and optimizer state from disk."""
        load_path = self.opt['path']['resume_state']
        if load_path is not None:
            logger.info(f'Loading pretrained model for G [{load_path}] ...')
            gen_path = f'{load_path}_gen.pth'
            opt_path = f'{load_path}_opt.pth'
            network = self.netG
            if isinstance(self.netG, nn.DataParallel):
                network = network.module
            network.load_state_dict(torch.load(gen_path), strict=(not self.opt['model']['finetune_norm']))
            if self.opt['phase'] == 'train':
                opt = torch.load(opt_path)
                self.optG.load_state_dict(opt['optimizer'])
                self.begin_step = opt['iter']
                self.begin_epoch = opt['epoch'] 