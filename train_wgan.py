import numpy as np
import torch
import os
from datetime import datetime

from torch import nn
from torch import optim
from torch.nn import functional as F
from torch import autograd
from torch.autograd import Variable
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
        self.conv2 = nn.Conv3d(num_channels//16, num_channels//8, kernel_size=4, stride=2, padding=1)
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

########### Training ##############

LAMBDA = 10

def calc_gradient_penalty(netD, real_data, fake_data):    
    alpha = torch.rand(real_data.size(0), 1, 1, 1, 1)
    alpha = alpha.expand(real_data.size())
    
    alpha = alpha.cuda()

    interpolates = alpha * real_data + ((1 - alpha) * fake_data)

    interpolates = interpolates.cuda()
    interpolates = Variable(interpolates, requires_grad=True)

    disc_interpolates = netD(interpolates)

    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()).cuda(),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA
    return gradient_penalty

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


if __name__ == '__main__':
    current_time = datetime.now().strftime("%Y_%m_%d_%H%M%S")
    save_dir = '{}/runs/WGAN-{}'.format(os.path.curdir, str(current_time))
    os.makedirs(save_dir, exist_ok=True)

    logger = wandb_logger.WandbLogger(
        # project='slice-based-latent-diffusion',
        project='comparative-models', 
        name='WGAN (3D)',
        save_dir=save_dir
    )


    max_epoch = 2000
    lr = 0.0001

    #setting latent variable sizes
    latent_dim = 2048

    D = Discriminator(
        in_channels=1, num_channels=1024
    ).to(device)

    G = Generator(
        latent_dim=latent_dim, out_channels=1, num_channels=512
    ).to(device)

    g_optimizer = optim.Adam(G.parameters(), lr=0.0002)
    d_optimizer = optim.Adam(D.parameters(), lr=0.0002)

    # real_y = Variable(torch.ones((BATCH_SIZE, 1)).cuda())
    # fake_y = Variable(torch.zeros((BATCH_SIZE, 1)).cuda())
    loss_f = nn.BCELoss()

    d_real_losses = list()
    d_fake_losses = list()
    d_losses = list()
    g_losses = list()
    divergences = list()

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

    datamodule.prepare_data()
    datamodule.setup()
    loader = datamodule.train_dataloader()

    for epoch in range(max_epoch):
        g_loss_history = list()
        d_loss_history = list()
        d_real_loss_history = list()
        d_fake_loss_history = list()
        wasserstein_d_history = list()

        progress = tqdm(loader, position=0, leave=True)
        for idx, (real_images,) in enumerate(loader):
            ###############################################
            # Train D 
            ###############################################
            for p in D.parameters():  
                p.requires_grad = True 

            # real_images = gen_load.__next__()
            
            D.zero_grad()
            real_images = real_images.to(device, dtype=torch.float32)

            _batch_size = real_images.shape[0]


            y_real_pred = D(real_images)

            d_real_loss = y_real_pred.mean()
            
            noise = torch.randn(_batch_size, latent_dim).to(device, dtype=torch.float32)
            fake_images = G(noise)
            y_fake_pred = D(fake_images.detach())

            d_fake_loss = y_fake_pred.mean()

            gradient_penalty = calc_gradient_penalty(D, real_images, fake_images)
        
            d_loss = -d_real_loss + d_fake_loss + gradient_penalty
            d_loss.backward()
            Wasserstein_D = d_real_loss - d_fake_loss

            d_optimizer.step()

            ###############################################
            # Train G 
            ###############################################
            for p in D.parameters():
                p.requires_grad = False
                
            for iters in range(5):
                G.zero_grad()
                noise = torch.randn(_batch_size, latent_dim).to(device, dtype=torch.float32)
                fake_image = G(noise)
                y_fake_g = D(fake_image)

                g_loss = -y_fake_g.mean()

                g_loss.backward()
                g_optimizer.step()

            # step logging
            wandb.log({
                'D_real_loss': d_real_loss.item(),
                'D_fake_loss': d_fake_loss.item(),
                'D_loss': d_loss.item(),
                'G_loss': g_loss.item(),
                'Wasserstein_D': Wasserstein_D.item()
            })

            # history logging
            d_real_loss_history.append(d_real_loss.item())
            d_fake_loss_history.append(d_fake_loss.item())
            d_loss_history.append(d_loss.item())
            g_loss_history.append(g_loss.item())
            wasserstein_d_history.append(Wasserstein_D.item())

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
            'G_loss_epoch': np.mean(g_loss_history),
            'Wasserstein_D_epoch': np.mean(wasserstein_d_history)
        })

        ###############################################
        # Visualization
        ###############################################
        if (epoch + 1) % 10 == 0:
            with torch.no_grad():
                fake_image = fake_image[0].permute(3, 0, 1, 2)

                # selecting subset of the volume to display
                fake_image = fake_image[::4, ...] # 64 // 4 = 16

                fake_image = torch.cat([
                    torch.hstack([img for img in fake_image[:, idx, ...]]) for idx in range(fake_image.shape[1])
                ], dim=0)

                fake_image = fake_image.add(1).div(2).mul(255).clamp(0, 255).to(torch.uint8)
                
                wandb.log({
                    'Reconstruction examples': wandb.Image(
                        fake_image.cpu().numpy(), 
                        caption='[{}]'.format(epoch)#, format_condition(condition[0].cpu().numpy()))
                    )
                })
                    
        if (epoch + 1) % 10 == 0:
            torch.save(G.state_dict(), save_dir + '/G_W_iter' + str(epoch + 1) + '.pth')
            torch.save(D.state_dict(), save_dir + '/D_W_iter' + str(epoch + 1) + '.pth')

