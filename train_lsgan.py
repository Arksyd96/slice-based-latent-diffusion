import numpy as np
import torch
import os
from datetime import datetime

from torch import nn
from torch import optim
from torch.nn import functional as F
from modules.data import BRATSDataModule
import wandb
from pytorch_lightning.loggers import wandb as wandb_logger
from tqdm import tqdm
import os
import wandb
from torchvision.utils import save_image


########### Architecture ##############

class Discriminator(nn.Module):
    def __init__(self, in_channels, num_channels = 512):
        super(Discriminator, self).__init__()        
        
        self.conv1 = nn.Conv3d(in_channels, num_channels // 16, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv3d(num_channels // 16, num_channels//8, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm3d(num_channels//8)
        self.conv3 = nn.Conv3d(num_channels//8, num_channels//4, kernel_size=4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm3d(num_channels//4)
        self.conv4 = nn.Conv3d(num_channels//4, num_channels//2, kernel_size=4, stride=2, padding=1)
        self.bn4 = nn.BatchNorm3d(num_channels//2)
        self.conv5 = nn.Conv3d(num_channels//2, num_channels, kernel_size=4, stride=2, padding=1)
        self.bn5 = nn.BatchNorm3d(num_channels)
        
        self.conv6 = nn.Conv3d(num_channels, 1, kernel_size=4, stride=2, padding=1)
        
    def forward(self, x):
        h1 = F.leaky_relu(self.conv1(x), negative_slope=0.2)
        h2 = F.leaky_relu(self.bn2(self.conv2(h1)), negative_slope=0.2)
        h3 = F.leaky_relu(self.bn3(self.conv3(h2)), negative_slope=0.2)
        h4 = F.leaky_relu(self.bn4(self.conv4(h3)), negative_slope=0.2)
        h5 = F.leaky_relu(self.bn5(self.conv5(h4)), negative_slope=0.2)
        h6 = self.conv6(h5)
        output = h6

        return output
    

class Generator(nn.Module):
    def __init__(self, latent_dim, out_channels, num_channels = 512):
        super(Generator, self).__init__()
        _c = num_channels
        self.latent_dim = latent_dim
        
        self.fc = nn.Linear(self.latent_dim, 512 * 6 * 6 * 3)
        self.bn1 = nn.BatchNorm3d(_c)
        
        self.tp_conv2 = nn.Conv3d(_c, _c // 2, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn2 = nn.BatchNorm3d(_c // 2)
        
        self.tp_conv3 = nn.Conv3d(_c // 2, _c // 4, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn3 = nn.BatchNorm3d(_c // 4)
        
        self.tp_conv4 = nn.Conv3d(_c // 4, _c // 8, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn4 = nn.BatchNorm3d(_c // 8)

        self.tp_conv5 = nn.Conv3d(_c // 8, _c // 16, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn5 = nn.BatchNorm3d(_c // 16)
        
        self.tp_conv6 = nn.Conv3d(_c // 16, out_channels, kernel_size=3, stride=1, padding=1, bias=True)
        
    def forward(self, noise):
        h = self.fc(noise)
        h = h.view(-1, 512, 6, 6, 3)
        h = F.relu(self.bn1(h))

        h = F.upsample(h,scale_factor = 2)
        h = self.tp_conv2(h)
        h = F.relu(self.bn2(h))
        
        h = F.upsample(h,scale_factor = 2)
        h = self.tp_conv3(h)
        h = F.relu(self.bn3(h))

        h = F.upsample(h,scale_factor = 2)
        h = self.tp_conv4(h)
        h = F.relu(self.bn4(h))

        h = F.upsample(h,scale_factor = 2)
        h = self.tp_conv5(h)
        h = F.relu(self.bn5(h))

        h = F.upsample(h,scale_factor = 2)
        h = self.tp_conv6(h)
        output = F.tanh(h)

        return output
    
def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


########### Training ##############

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    current_time = datetime.now().strftime("%Y_%m_%d_%H%M%S")
    save_dir = '{}/runs/LSGAN-{}'.format(os.path.curdir, str(current_time))
    os.makedirs(save_dir, exist_ok=True)

    logger = wandb_logger.WandbLogger(
        # project='slice-based-latent-diffusion',
        project='comparative-models', 
        name='LSGAN (3D)',
        save_dir=save_dir
    )

    max_epoch = 2000

    # setting latent variable sizes
    latent_dim = 2048

    discriminator = Discriminator(in_channels=1, num_channels=1024).to(device)
    generator = Generator(latent_dim=latent_dim, out_channels=1, num_channels=512).to(device)

    discriminator.apply(weights_init_normal)
    generator.apply(weights_init_normal)

    g_optimizer = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    d_optimizer = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

    adversarial_loss = torch.nn.MSELoss().to(device)

    datamodule = BRATSDataModule(
        data_dir        = './data/second_stage_dataset_192x192.npy',
        train_ratio     = 1.0,
        norm            = 'centered-norm', 
        batch_size      = 2,
        num_workers     = 6,
        shuffle         = True,
        # horizontal_flip = 0.5,
        # vertical_flip   = 0.5,
        # rotation        = (0, 90),
        # random_crop_size = (96, 96),
        dtype           = torch.float32,
        include_radiomics = False
    )

    datamodule.setup()
    train_loader = datamodule.train_dataloader()

    for epoch in range(max_epoch):
        g_loss_history = list()
        d_loss_history = list()
        d_real_loss_history = list()
        d_fake_loss_history = list()    

        progress = tqdm(train_loader, position=0, leave=True)
        for idx, (real_x,) in enumerate(train_loader):
            
            real_y = torch.ones(real_x.size(0), 1) - torch.randn(real_x.size(0), 1) * 0.05
            fake_y = torch.randn(real_x.size(0), 1) * 0.05
            
            real_y = real_y.to(device, dtype=torch.float32)
            fake_y = fake_y.to(device, dtype=torch.float32)

            real_x = real_x.to(device, dtype=torch.float32)

            # --------------------- Train Generator ---------------------
            g_optimizer.zero_grad(set_to_none=True)

            z = torch.randn(real_x.size(0), latent_dim).to(device, dtype=torch.float32)
            fake_x = generator(z)
            g_loss = adversarial_loss(discriminator(fake_x).mean(), real_y)

            g_loss.backward()
            torch.nn.utils.clip_grad_norm_(generator.parameters(), 5)
            g_optimizer.step()

            # --------------------- Train Discriminator ---------------------
            d_optimizer.zero_grad(set_to_none=True)

            d_real_loss = adversarial_loss(discriminator(real_x).mean(), real_y)
            d_fake_loss = adversarial_loss(discriminator(fake_x.detach()).mean(), fake_y)
            d_loss = 0.5 * (d_real_loss + d_fake_loss)

            d_loss.backward()
            torch.nn.utils.clip_grad_norm_(discriminator.parameters(), 5)
            d_optimizer.step()


            # step logging
            wandb.log({
                'D_real_loss': d_real_loss.item(),
                'D_fake_loss': d_fake_loss.item(),
                'D_loss': d_loss.item(),
                'G_loss': g_loss.item()
            })

            # history logging
            d_real_loss_history.append(d_real_loss.item())
            d_fake_loss_history.append(d_fake_loss.item())
            d_loss_history.append(d_loss.item())
            g_loss_history.append(g_loss.item())

            progress.set_description('Epoch: {}/{} | D_loss: {:.4f} | G_loss: {:.4f}'.format(
                epoch + 1, max_epoch, d_loss.item(), g_loss.item()
            ))
            progress.update(1)

        progress.close()
        del progress

        # epoch logging
        wandb.log({
            'D_real_loss_epoch': np.mean(d_real_loss_history),
            'D_fake_loss_epoch': np.mean(d_fake_loss_history),
            'D_loss_epoch': np.mean(d_loss_history),
            'G_loss_epoch': np.mean(g_loss_history)
        })

        ###############################################
        # Visualization
        ###############################################
        if (epoch + 1) % 10 == 0:
            with torch.no_grad():
                fake_x = fake_x[0].permute(3, 0, 1, 2)

                # selecting subset of the volume to display
                fake_x = fake_x[::4, ...] # 64 // 4 = 16

                fake_x = torch.cat([
                    torch.hstack([img for img in fake_x[:, idx, ...]]) for idx in range(fake_x.shape[1])
                ], dim=0)

                fake_x = fake_x.add(1).div(2).mul(255).clamp(0, 255).to(torch.uint8)
                
                wandb.log({
                    'Reconstruction examples': wandb.Image(
                        fake_x.cpu().numpy(), 
                        caption='[{}]'.format(epoch)#, format_condition(condition[0].cpu().numpy()))
                    )
                })
                    
        if (epoch + 1) % 10 == 0:
            torch.save(generator.state_dict(), save_dir + '/G_W_iter' + str(epoch + 1) + '.pth')
            torch.save(discriminator.state_dict(), save_dir + '/D_W_iter' + str(epoch + 1) + '.pth')
