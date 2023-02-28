import torch
import torch.nn as nn
import torch.nn.functional as F

class Upsampler(nn.Module):
    def __init__(self, in_channels=3, ngf=128):
        super(Upsampler, self).__init__()

        self.up = nn.Sequential(
            nn.ConvTranspose2d(in_channels, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d( ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d( ngf * 2, 3, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. 3 x 32 x 32
        )

    def forward(self, x):
        return self.up(x)

## From: https://github.com/LukeDitria/CNN-VAE

class ResDown(nn.Module):
    """
    Residual down sampling block for the encoder
    """

    def __init__(self, channel_in, channel_out, kernel_size=3):
        super(ResDown, self).__init__()
        self.conv1 = nn.Conv2d(channel_in, channel_out // 2, kernel_size, 2, kernel_size // 2)
        self.bn1 = nn.GroupNorm(channel_out//2, channel_out//2, affine=True)#nn.BatchNorm2d(channel_out // 2, eps=1e-4)
        self.conv2 = nn.Conv2d(channel_out // 2, channel_out, kernel_size, 1, kernel_size // 2)
        self.bn2 = nn.GroupNorm(channel_out, channel_out, affine=True)#nn.BatchNorm2d(channel_out, eps=1e-4)

        self.conv3 = nn.Conv2d(channel_in, channel_out, kernel_size, 2, kernel_size // 2)

        self.act_fnc = nn.ELU()

    def forward(self, x):
        skip = self.conv3(x)
        x = self.act_fnc(self.bn1(self.conv1(x)))
        x = self.conv2(x)
        return self.act_fnc(self.bn2(x + skip))
  
class ResUp(nn.Module):
    """
    Residual up sampling block for the decoder
    """

    def __init__(self, channel_in, channel_out, kernel_size=3, scale_factor=2):
        super(ResUp, self).__init__()

        self.conv1 = nn.Conv2d(channel_in, channel_out // 2, kernel_size, 1, kernel_size // 2)
        self.bn1 = nn.GroupNorm(channel_out//2, channel_out//2, affine=True)#nn.BatchNorm2d(channel_out // 2, eps=1e-4)
        self.conv2 = nn.Conv2d(channel_out // 2, channel_out, kernel_size, 1, kernel_size // 2)
        self.bn2 = nn.GroupNorm(channel_out, channel_out, affine=True)#nn.BatchNorm2d(channel_out, eps=1e-4)

        self.conv3 = nn.Conv2d(channel_in, channel_out, kernel_size, 1, kernel_size // 2)

        self.up_nn = nn.Upsample(scale_factor=scale_factor, mode="nearest")

        self.act_fnc = nn.ELU()

    def forward(self, x):
        x = self.up_nn(x)
        skip = self.conv3(x)
        x = self.act_fnc(self.bn1(self.conv1(x)))
        x = self.conv2(x)

        return self.act_fnc(self.bn2(x + skip))

class Encoder(nn.Module):
    """
    Encoder block
    Built for a 3x32x32 image and will result in a latent vector of size z x 1 x 1
    As the network is fully convolutional it will work for images LARGER than 64
    For images sized 64 * n where n is a power of 2, (1, 2, 4, 8 etc) the latent feature map size will be z x n x n
    When in .eval() the Encoder will not sample from the distribution and will instead output mu as the encoding vector
    and log_var will be None
    """
    def __init__(self, channels, ch=128, latent_channels=512, ae=False):
        super(Encoder, self).__init__()
        self.ae = ae
        self.conv_in = nn.Conv2d(channels, ch, 7, 1, 3)
        self.conv_label = nn.Conv2d(10, ch, 7, 1, 3)
        self.res_down_block1 = ResDown(ch*2, 4 * ch)
        self.res_down_block2 = ResDown(4 * ch, 8 * ch)
        self.res_down_block3 = ResDown(8 * ch, 16 * ch)
        if self.ae:
            self.conv_latent = nn.Conv2d(16 * ch, latent_channels, 4, 1)
        else:
            self.conv_mu = nn.Conv2d(16 * ch, latent_channels, 4, 1)
            self.conv_log_var = nn.Conv2d(16 * ch, latent_channels, 4, 1)
        self.act_fnc = nn.ELU()

        self.fill = torch.zeros([10, 10, 32, 32]).to('cuda')
        for i in range(10):
            self.fill[i, i, :, :] = 1

    def sample(self, mu, log_var):
        std = torch.exp(0.5*log_var)
        eps = torch.randn_like(std)
        return mu + eps*std
        
    def forward(self, x, y):
        y = self.fill[y]
        x = self.act_fnc(self.conv_in(x))
        y = self.act_fnc(self.conv_label(y))
        x = torch.cat([x, y], 1)
        x = self.res_down_block1(x)  # 16
        x = self.res_down_block2(x)  # 8
        x = self.res_down_block3(x)  # 4
        if self.ae:
            z = self.conv_latent(x)
            return z
        else:
            mu = self.conv_mu(x)  # 1
            log_var = self.conv_log_var(x)  # 1
            z = self.sample(mu, log_var)
            return mu, log_var, z

    
class Decoder(nn.Module):
    def __init__(self, channels, ch=128, latent_channels=512):
        super(Decoder, self).__init__()
        self.conv_t_up = nn.ConvTranspose2d(latent_channels, ch * 8, 4, 1)
        self.res_up_block1 = ResUp(ch * 8, ch * 4)
        self.res_up_block2 = ResUp(ch * 4, ch * 2)
        self.res_up_block3 = ResUp(ch * 2, ch)
        self.conv_out = nn.Conv2d(ch, channels, 3, 1, 1)
        self.act_fnc = nn.ELU()

    def forward(self, x):
        x = self.act_fnc(self.conv_t_up(x))  # 4
        x = self.res_up_block1(x)  # 8
        x = self.res_up_block2(x)  # 16
        x = self.res_up_block3(x)  # 32
        x = torch.tanh(self.conv_out(x))

        return x 

class ResVAE(nn.Module):
    def __init__(self, channel_in=3, ch=64, latent_size=32, ae=False):
        super(ResVAE, self).__init__()
        self.ae = ae
        self.encoder = Encoder(channel_in, ch=ch, latent_channels=latent_size, ae=self.ae)
        self.decoder = Decoder(channel_in, ch=ch, latent_channels=latent_size)

    def forward(self, x, y):
        if self.ae:
            z = self.encoder(x, y)
        else:
            mu, log_var, z = self.encoder(x, y)
        recon_img = self.decoder(z)
        if self.ae:
            return recon_img, z
        return recon_img, mu, log_var, z