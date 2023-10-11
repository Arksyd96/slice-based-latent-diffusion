import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pathlib import Path 
import json

from modules.loss.perceivers import LPIPS
from modules.models.utils.conv_blocks import DownBlock, UpBlock, BasicBlock, BasicResBlock, UnetResBlock, UnetBasicBlock
from modules.loss.gan_losses import hinge_d_loss
from pytorch_msssim import SSIM, ssim

class VQGAN(pl.LightningModule):
    def __init__(
        self,
        in_channels     = 3, 
        out_channels    = 3, 
        spatial_dims    = 2,
        emb_channels    = 4,
        num_embeddings  = 8192,
        hid_chs         = [64, 128, 256, 512],
        kernel_sizes    = [3, 3, 3, 3],
        strides         = [1, 2, 2, 2],
        norm_name = ("GROUP", {'num_groups': 32, "affine": True}),
        act_name = ("Swish", {}),
        dropout         = 0.0,
        use_res_block   = True,
        deep_supervision = False,
        learnable_interpolation = True,
        use_attention   = 'none',
        beta            = 0.25,
        embedding_loss_weight = 1.0,
        perceiver       = LPIPS,
        perceiver_kwargs = {},
        perceptual_loss_weight = 1.0,
    
        start_gan_train_step = 50000, # NOTE step increase with each optimizer 
        gan_loss_weight = 1.0, # = discriminator  
        
        optimizer_vqvae = torch.optim.Adam, 
        optimizer_gan   = torch.optim.Adam, 
        optimizer_vqvae_kwargs = {'lr': 1e-6},
        optimizer_gan_kwargs = {'lr': 1e-6}, 
        lr_scheduler_vqvae = None, 
        lr_scheduler_vqvae_kwargs = {},
        lr_scheduler_gan = None, 
        lr_scheduler_gan_kwargs = {},

        pixel_loss      = torch.nn.L1Loss,
        pixel_loss_kwargs = {'reduction':'none'},
        gan_loss_fct    = hinge_d_loss,

        sample_every_n_steps = 1000

    ):
        super().__init__()
        self.sample_every_n_steps=sample_every_n_steps
        self.start_gan_train_step = start_gan_train_step
        self.gan_loss_weight = gan_loss_weight
        self.embedding_loss_weight = embedding_loss_weight

        self.optimizer_vqvae = optimizer_vqvae
        self.optimizer_gan = optimizer_gan
        self.optimizer_vqvae_kwargs = optimizer_vqvae_kwargs
        self.optimizer_gan_kwargs = optimizer_gan_kwargs
        self.lr_scheduler_vqvae = lr_scheduler_vqvae
        self.lr_scheduler_vqvae_kwargs = lr_scheduler_vqvae_kwargs
        self.lr_scheduler_gan = lr_scheduler_gan
        self.lr_scheduler_gan_kwargs = lr_scheduler_gan_kwargs

        self.pixel_loss_fct = pixel_loss(**pixel_loss_kwargs)
        self.gan_loss_fct = gan_loss_fct

        self.vqvae = VQVAE(in_channels, out_channels, spatial_dims, emb_channels, num_embeddings, hid_chs, kernel_sizes,
            strides, norm_name, act_name, dropout, use_res_block, deep_supervision, learnable_interpolation, use_attention, 
            beta, embedding_loss_weight, perceiver, perceiver_kwargs, perceptual_loss_weight)

        self.discriminator = nn.ModuleList([Discriminator(in_channels, spatial_dims, hid_chs, kernel_sizes, strides, 
                                            act_name, norm_name, dropout) for i in range(len(self.vqvae.outc_ver)+1)])
    
    def encode(self, x):
        return self.vqvae.encode(x) 
            
    def decode(self, z):
        return self.vqvae.decode(z)

    def forward(self, x):
        return self.vqvae.forward(x)
    
    
    def vae_img_loss(self, pred, target, dec_out_layer, step, discriminator, depth=0):
        # ------ VQVAE -------
        rec_loss =  self.vqvae.rec_loss(pred, [], target)

        # ------- GAN ----- 
        if step > self.start_gan_train_step:
            gan_loss = -torch.mean(discriminator[depth](pred))  
            lambda_weight = self.compute_lambda(rec_loss, gan_loss, dec_out_layer)
            gan_loss = gan_loss*lambda_weight

            with torch.no_grad():
                self.log(f"train/gan_loss_{depth}", gan_loss, on_step=True, on_epoch=True)
                self.log(f"train/lambda_{depth}", lambda_weight, on_step=True, on_epoch=True) 
        else:
            gan_loss = 0 #torch.tensor([0.0], requires_grad=True, device=target.device)
    
        return self.gan_loss_weight * gan_loss + rec_loss
    

    def gan_img_loss(self, pred, target, step, discriminators, depth):
        if (step > self.start_gan_train_step) and (depth<len(discriminators)):
            logits_real = discriminators[depth](target.detach())
            logits_fake = discriminators[depth](pred.detach())
            loss = self.gan_loss_fct(logits_real, logits_fake)
        else:
            loss = torch.tensor(0.0, requires_grad=True, device=target.device)
        
        with torch.no_grad():
            self.log(f"train/loss_1_{depth}", loss, on_step=True, on_epoch=True) 
        return loss 

    def _step(self, batch: dict, batch_idx: int, state: str, step: int, optimizer_idx:int):
        # ------------------------- Get Source/Target ---------------------------
        x = batch['source']
        target = x 

        # ------------------------- Run Model ---------------------------
        pred, pred_vertical, emb_loss = self(x)

        # ------------------------- Compute Loss ---------------------------
        interpolation_mode = 'area'
        weights = [1/2**i for i in range(1+len(pred_vertical))] # horizontal + vertical (reducing with every step down)
        tot_weight = sum(weights)
        weights = [w/tot_weight for w in weights]
        logging_dict = {}

        if optimizer_idx == 0:
            # Horizontal/Top Layer 
            img_loss = self.vae_img_loss(pred, target, self.vqvae.outc.conv, step, self.discriminator, 0)*weights[0]

            # Vertical/Deep Layer 
            for i, pred_i in enumerate(pred_vertical): 
                target_i = F.interpolate(target, size=pred_i.shape[2:], mode=interpolation_mode, align_corners=None)  
                img_loss += self.vae_img_loss(pred_i, target_i, self.vqvae.outc_ver[i].conv, step, self.discriminator, i+1)*weights[i+1]
            loss =  img_loss+self.embedding_loss_weight*emb_loss

            with torch.no_grad():
                logging_dict[f'img_loss'] = img_loss
                logging_dict[f'emb_loss'] = emb_loss
                logging_dict['loss_0'] = loss

        elif optimizer_idx == 1:
            # Horizontal/Top Layer 
            loss = self.gan_img_loss(pred, target, step, self.discriminator, 0)*weights[0]

            # Vertical/Deep Layer 
            for i, pred_i in enumerate(pred_vertical):  
                target_i = F.interpolate(target, size=pred_i.shape[2:], mode=interpolation_mode, align_corners=None)  
                loss += self.gan_img_loss(pred_i, target_i, step, self.discriminator, i+1)*weights[i+1]
            
            with torch.no_grad():
                logging_dict['loss_1'] = loss


        # --------------------- Compute Metrics  -------------------------------
        with torch.no_grad():
            logging_dict['loss'] = loss
            logging_dict[f'L2'] = torch.nn.functional.mse_loss(pred, x)
            logging_dict[f'L1'] = torch.nn.functional.l1_loss(pred, x)
            logging_dict['ssim'] = ssim((pred+1)/2, (target.type(pred.dtype)+1)/2, data_range=1)

        # ----------------- Log Scalars ----------------------
        for metric_name, metric_val in logging_dict.items():
            self.log(f"{state}/{metric_name}", metric_val, batch_size=x.shape[0], on_step=True, on_epoch=True)     

        # ----------------- Save Image ------------------------------
        if self.global_step != 0 and self.global_step % self.sample_every_n_steps == 0: # NOTE: step 1 (opt1) , step=2 (opt2), step=3 (opt1), ...
            
            log_step = self.global_step // self.sample_every_n_steps
            path_out = Path(self.logger.log_dir)/'images'
            path_out.mkdir(parents=True, exist_ok=True)
            # for 3D images use depth as batch :[D, C, H, W], never show more than 16+16 =32 images 
            def depth2batch(image):
                return (image if image.ndim<5 else torch.swapaxes(image[0], 0, 1))
            images = torch.cat([depth2batch(img)[:16] for img in (x, pred)]) 
            save_image(images, path_out/f'sample_{log_step}.png', nrow=x.shape[0], normalize=True)
        
        return loss 
    
    def configure_optimizers(self):
        opt_vqvae = self.optimizer_vqvae(self.vqvae.parameters(), **self.optimizer_vqvae_kwargs)
        opt_gan = self.optimizer_gan(self.discriminator.parameters(), **self.optimizer_gan_kwargs)
        schedulers = []
        if self.lr_scheduler_vqvae is not None:
            schedulers.append({
                'scheduler': self.lr_scheduler_vqvae(opt_vqvae, **self.lr_scheduler_vqvae_kwargs), 
                'interval': 'step',
                'frequency': 1
            })
        if self.lr_scheduler_gan is not None:
            schedulers.append({
                'scheduler': self.lr_scheduler_gan(opt_gan, **self.lr_scheduler_gan_kwargs),
                'interval': 'step',
                'frequency': 1
            })
        return [opt_vqvae, opt_gan], schedulers
    
    def compute_lambda(self, rec_loss, gan_loss, dec_out_layer, eps=1e-4):
        """Computes adaptive weight as proposed in eq. 7 of https://arxiv.org/abs/2012.09841"""
        rec_grads = torch.autograd.grad(rec_loss, dec_out_layer.weight, retain_graph=True)[0]
        gan_grads = torch.autograd.grad(gan_loss, dec_out_layer.weight, retain_graph=True)[0]
        d_weight = torch.norm(rec_grads) / (torch.norm(gan_grads) + eps) 
        d_weight = torch.clamp(d_weight, 0.0, 1e4)
        return d_weight.detach()