# make a w-gan for the shoe dataset
# %%
# import pandas, Path, torch, torchvision, wandb
import matplotlib.pyplot as plt
from torch import nn
import torchvision
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import pandas as pd
import torch
# import pathlib
from pathlib import Path
# import lightening data module
from pytorch_lightning import LightningDataModule

# %%
base_path = Path('/home/garlan/git/shoe-data-set-and-classifier/Shoes_Dataset')
# %%
# get all the paths to the images
# print the current directory
print(os.getcwd())

image_paths = list(base_path.rglob("*.jpeg"))
# show the number of images and plot a few
print(len(image_paths))


# %%
# import the ShoeDataset class and ShoeDataModule class
from shoe_classifer import ShoeDataset, ShoeDataModule
# %%
# make a Critic class for the WGAN
class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
    def forward(self, x):
        residual = x
        # print(" x shape", x.shape, "next a layer of conv shape", self.conv1)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        # print("out shape", out.shape, "next a layer of conv shape", self.conv2)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample:
            # print("out shape", out.shape, "next a layer of downsample shape", self.downsample)
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        # print(" final out shape", out.shape)
        return out

class GAN_Critic(nn.Module):
    def __init__(self, in_channels=3, features_d=[32, 32, 64, 128, 256, 512]):
        super().__init__()
        # use resnet blocks
        self.initial = nn.Sequential(
            nn.Conv2d(in_channels, features_d[0], kernel_size=4, stride=2, padding=1, bias=False),
        )
        self.layer1 = self._make_layer(features_d[0], features_d[1], 4, stride=2)
        self.layer2 = self._make_layer(features_d[1], features_d[2], 4, stride=2)
        self.layer3 = self._make_layer(features_d[2], features_d[3], 4, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(features_d[3], 1)
    
    def _make_layer(self, in_channels, out_channels, num_residual_blocks, stride):
        downsample = None
        layers = []
        if stride != 1 or in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        layers.append(ResBlock(in_channels, out_channels, stride, downsample))
        for _ in range(num_residual_blocks - 1):
            layers.append(ResBlock(out_channels, out_channels))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.initial(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        return self.fc(x)
# %%
# test the critic with a random noise image
critic = GAN_Critic()
noise = torch.randn((1, 3, 256, 256))
print(critic(noise).shape)
            
# %%
# make a generator class for the WGAN
class GAN_Generator(nn.Module):
    # take in a latent vector and output an image
    def __init__(self, z_dim=100, in_channels=3, features_g=[512, 256, 128, 64, 32, 32]):
        super().__init__()
        self.z_dim = z_dim
        self.initial = nn.Sequential(
            nn.ConvTranspose2d(z_dim, features_g[0], kernel_size=4, stride=1, bias=False),
            nn.BatchNorm2d(features_g[0]),
            nn.ReLU(inplace=True)
        )
        self.conv_Transpose_layers = []
        for i in range(1, len(features_g)):
            group = nn.Sequential(
                nn.ConvTranspose2d(features_g[i-1], features_g[i], kernel_size=4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(features_g[i]),
                nn.ReLU(inplace=True))
            self.conv_Transpose_layers.append(group)
        self.convs_ = nn.Sequential(*self.conv_Transpose_layers)
            
        # self.layer1 = self._make_layer(features_g[0], features_g[1], 4, stride=2)
        # self.layer2 = self._make_layer(features_g[1], features_g[2], 4, stride=2)
        # self.layer3 = self._make_layer(features_g[2], features_g[3], 4, stride=2)
        # self.layer4 = self._make_layer(features_g[3], features_g[4], 4, stride=2)
        self.final = nn.Sequential(
            nn.ConvTranspose2d(features_g[4], in_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh()
        )
    def _make_layer(self, in_channels, out_channels, kernel_size, stride):
        layers = []
        layers.extend([
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)]
        )
        return nn.Sequential(*layers)
    def forward(self, x):
        x = self.initial(x)
        x = self.convs_(x)
       
        return self.final(x)
    def test_with_random_noise(self, n=1):
        z = torch.randn(n, self.z_dim, 1, 1)
        return self(z)
    
# %%
# test the generator with random noise
gen = GAN_Generator()
gen.test_with_random_noise().shape
    
    
# %%
import pytorch_lightning as pl
import torch
class WGAN(pl.LightningModule):
    def __init__(self, critic = None, generator = None, lr=2e-4, b1=0.5, b2=0.999,
                 batch_size=64, z_dim=100, in_channels=3,
                 data_mean = None, 
                 data_std = None,
                 **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.critic = critic if critic is not None else GAN_Critic(in_channels=in_channels)
        self.generator = generator if generator is not None else GAN_Generator(z_dim=z_dim, in_channels=in_channels)
        self.gen_opt = torch.optim.Adam(self.generator.parameters(), lr=self.hparams.lr, betas=(self.hparams.b1, self.hparams.b2))
        self.crit_opt = torch.optim.Adam(self.critic.parameters(), lr=self.hparams.lr, betas=(self.hparams.b1, self.hparams.b2))
        self.gen_loss = 0
        self.crit_loss = 0
        self.gen_steps = 0
        self.crit_steps = 0
        self.num_of_fixed_noise = 32
        self.fixed_noise = torch.randn(self.num_of_fixed_noise, self.hparams.z_dim, 1, 1)
        # self.example_input_array = torch.zeros(2, self.hparams.in_channels, 256, 256)
        
    def forward(self, x):
        return self.generator(x)
    
    def configure_optimizers(self):
        return [self.gen_opt, self.crit_opt], []
    # def adversarial_loss(self, y_hat, y):
    #     return self.criterion(y_hat, y)
    
    # at the start of training process, we need to set the generator and critic to train mode
    # and put them on the same device as the real images
    def on_train_start(self):
        self.generator.train()
        self.critic.train()
        self.generator =  self.generator.to(self.device)
        self.critic = self.critic.to(self.device)
        
    
    def training_step(self, batch, batch_idx, optimizer_idx):
        verbose = False
        real, _ = batch
        self.real = real
        batch_size = real.shape[0]
        if verbose:
            print('step. Batch idx: ', batch_idx, 'Opt idx: ', optimizer_idx, 'Batch size: ', batch_size, 'real shape: ', real.shape)
        # train generator
        if optimizer_idx == 0:
            if verbose: print('gen step')
            self.gen_steps += 1
            # generate noise
            z = torch.randn(batch_size, self.hparams.z_dim, 1, 1).to(real.device)
            # generate images
            if verbose: print("z shape and device: ", z.shape, z.device)
            fake = self(z)
            # pass images through critic
            if verbose: print("fake shape and device: ", fake.shape, fake.device)
            critic_pred = self.critic(fake)
            # calculate generator loss
            gen_loss = -torch.mean(critic_pred)
            self.log('gen_loss', gen_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
            return gen_loss
        # train critic
        if optimizer_idx == 1:
            if verbose: print('crit step')
            # train the critic more than the generator
            # use 5 steps for the critic and 1 step for the generator
            n_steps = 5
            for j in range(n_steps):
                self.crit_steps += 1
                self.crit_opt.zero_grad()
                # generate images
                # generate noise
                z = torch.randn(batch_size, self.hparams.z_dim, 1, 1).to(real.device)
                fake = self(z)
                # score both the real and fake images
                critic_real = self.critic(real)
                critic_fake = self.critic(fake.detach())
                # use the wgans loss function
                crit_loss = (torch.mean(critic_real) - torch.mean(critic_fake))
                # clip the weights of the critic
                for p in self.critic.parameters():
                    p.data.clamp_(-0.01, 0.01)
                # optimize
                self.log('crit_loss', crit_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
                if j == n_steps - 1:
                    return {"loss": crit_loss} #  py lightning will call backward() and optimizer.step() for us
                self.crit_opt.step()
                
    
    # plot some images
    def plot_images(self, batch, batch_idx, optimizer_idx, do_log=True, use_plt = False):
        print('plotting images. Optimizer idx: ', optimizer_idx)
        if optimizer_idx == 0:
            # log sampled images
            # use the fixed noise to generate images
            grid_genereated = torchvision.utils.make_grid(self(self.fixed_noise.to(self.device) )).cpu().detach()
            # get the same number of real images as the generated images
            # randomly sample the indices of the real images
            idx = torch.randint(0, self.real.shape[0], (self.num_of_fixed_noise,))
            real_temp = self.real[idx]
            grid_real = torchvision.utils.make_grid(real_temp).cpu().detach()
            if self.hparams.data_mean is not None and self.hparams.data_std is not None:
                mean = torch.tensor(self.hparams.data_mean).view(3, 1, 1)
                std = torch.tensor(self.hparams.data_std).view(3, 1, 1)
                grid_genereated = grid_genereated * std + mean
                grid_real = grid_real * std + mean

            if use_plt:
                fig, ax = plt.subplots(1, 2, figsize=(10, 5))
                ax[0].imshow(grid_genereated.permute(1, 2, 0))
                ax[0].set_title('Generated Images')
                ax[1].imshow(grid_real.permute(1, 2, 0))
                ax[1].set_title('Real Images')
                if do_log:
                    self.logger.experiment.log({'generated vs real': wandb.Image(fig)})
                    # print('logged generated vs real')
                    plt.close(fig)
                    # self.logger.experiment.log('generated vs real', fig, self.current_epoch)
                
    
    
    # at the end of the epoch, log the images
    def on_epoch_end(self):
        # print('on_epoch_end')
        self.plot_images(None, None, 0, use_plt=True)
    
    
    # def on_train_epoch_end(self):
    #     print('_on_epoch_end')
    #     self.plot_images(None, None, 0)
        
    # def on_epoch_start(self) -> None:
        # return super().on_epoch_start()
        
        
# %%
csv_file = Path('../../Shoes_Dataset/shoes.csv')
# check if the file exists, if not then assume running as a script
if csv_file.exists():
    print("ipython")
else:
    csv_file = Path("/Shoes_Dataset/shoes.csv")
    print("script")
    
df = pd.read_csv(csv_file)
dm  = ShoeDataModule(batch_size=64, df=df, filter_with_shoe_type_id=0)
# %%
model = WGAN(data_mean=dm.mean, data_std=dm.std)
# model = model.eval()
                
# %%
# show some example images
# get a real i mage from the dataloader
# dm.setup()
# real, _ = next(iter(dm.train_dataloader()))
# print(real.shape)
# model.real = real
# model.plot_images(None, None, 0, do_log=False, use_plt=True)    
# %%
# model.fixed_noise.shape
# grid = torchvision.utils.make_grid(model(model.fixed_noise))
# %%
# train the WGAN
use_wandb = True
if use_wandb:
    import wandb
    # also import wandb logger
    from pytorch_lightning.loggers import WandbLogger
    logger = WandbLogger(project='WGAN')
else:
    # use tenorboard logger
    from pytorch_lightning.loggers import TensorBoardLogger
    logger = TensorBoardLogger('lightning_logs', name='WGAN')

trainer = pl.Trainer(max_epochs=100, gpus=1, logger=logger)
trainer.fit(model, dm)
# %%
