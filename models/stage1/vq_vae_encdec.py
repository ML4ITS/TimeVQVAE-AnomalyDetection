"""
reference: https://github.com/nadavbh12/VQ-VAE/blob/master/vq_vae/auto_encoder.py
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils import timefreq_to_time, time_to_timefreq, SnakeActivation


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, frequency_indepence:bool, mid_channels=None, dropout:float=0.):
        super(ResBlock, self).__init__()

        if mid_channels is None:
            mid_channels = out_channels
        
        kernel_size = (1, 3) if frequency_indepence else (3, 3)
        padding = (0, 1) if frequency_indepence else (1, 1)

        layers = [
            SnakeActivation(in_channels, 2), #SnakyGELU(in_channels, 2), #SnakeActivation(in_channels, 2), #nn.LeakyReLU(), #SnakeActivation(),
            nn.Conv2d(in_channels, mid_channels,
                      kernel_size=kernel_size, stride=(1, 1), padding=padding),
            nn.BatchNorm2d(out_channels),
            SnakeActivation(out_channels, 2), #SnakyGELU(out_channels, 2), #SnakeActivation(out_channels, 2), #nn.LeakyReLU(), #SnakeActivation(),
            nn.Conv2d(mid_channels, out_channels,
                      kernel_size=kernel_size, stride=(1, 1), padding=padding),
            nn.Dropout(dropout)
        ]
        self.convs = nn.Sequential(*layers)
        self.proj = nn.Identity() if in_channels == out_channels else nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.proj(x) + self.convs(x)


class VQVAEEncBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 frequency_indepence:bool,
                 dropout:float=0.
                 ):
        super().__init__()
        
        kernel_size = (1, 4) if frequency_indepence else (3, 4)
        padding = (0, 1) if frequency_indepence else (1, 1)

        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=(1, 2), padding=padding,
                      padding_mode='replicate'),
            nn.BatchNorm2d(out_channels),
            SnakeActivation(out_channels, 2), #SnakyGELU(out_channels, 2), #SnakeActivation(out_channels, 2), #nn.LeakyReLU(), #SnakeActivation(),
            nn.Dropout(dropout))

    def forward(self, x):
        out = self.block(x)
        return out


class VQVAEDecBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 frequency_indepence:bool,
                 dropout:float=0.
                 ):
        super().__init__()
        
        kernel_size = (1, 4) if frequency_indepence else (3, 4)
        padding = (0, 1) if frequency_indepence else (1, 1)

        self.block = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=(1, 2), padding=padding),
            nn.BatchNorm2d(out_channels),
            SnakeActivation(out_channels, 2), #SnakyGELU(out_channels, 2), #SnakeActivation(out_channels, 2), #nn.LeakyReLU(), #SnakeActivation(),
            nn.Dropout(dropout))

    def forward(self, x):
        out = self.block(x)
        return out


class VQVAEEncoder(nn.Module):
    """
    following the same implementation from the VQ-VAE paper.
    """

    def __init__(self,
                 dim: int,
                 num_channels: int,
                 downsample_rate: int,
                 n_resnet_blocks: int,
                 n_fft:int,
                 frequency_indepence:bool,
                 dropout:float=0.3,
                 **kwargs):
        """
        :param d: hidden dimension size
        :param num_channels: channel size of input
        :param downsample_rate: should be a factor of 2; e.g., 2, 4, 8, 16, ...
        :param n_resnet_blocks: number of ResNet blocks
        :param bn: use of BatchNorm
        :param kwargs:
        """
        super().__init__()
        self.n_fft = n_fft

        d = 4
        enc_layers = [VQVAEEncBlock(num_channels, d, frequency_indepence),]
        d *= 2
        for _ in range(int(round(np.log2(downsample_rate))) - 1):
            enc_layers.append(VQVAEEncBlock(d//2, d, frequency_indepence))
            for _ in range(n_resnet_blocks):
                enc_layers.append(ResBlock(d, d, frequency_indepence, dropout=dropout))
            d *= 2
        enc_layers.append(ResBlock(d//2, dim, frequency_indepence, dropout=dropout))
        self.encoder = nn.Sequential(*enc_layers)

        self.is_num_tokens_updated = False
        self.register_buffer('num_tokens', torch.tensor(0))
        self.register_buffer('H_prime', torch.tensor(0))
        self.register_buffer('W_prime', torch.tensor(0))
    
    def forward(self, x):
        """
        :param x: (b c l)
        """
        in_channels = x.shape[1]
        x = time_to_timefreq(x, self.n_fft, in_channels)  # (b c h w)

        out = self.encoder(x)  # (b c h w)
        if not self.is_num_tokens_updated:
            self.H_prime = torch.tensor(out.shape[2])
            self.W_prime = torch.tensor(out.shape[3])
            self.num_tokens = self.H_prime * self.W_prime
            self.is_num_tokens_updated = True
        return out


class VQVAEDecoder(nn.Module):
    """
    following the same implementation from the VQ-VAE paper.
    """

    def __init__(self,
                 dim: int,
                 num_channels: int,
                 downsample_rate: int,
                 n_resnet_blocks: int,
                 input_length:int,
                 n_fft:int,
                 x_channels:int,
                 frequency_indepence:bool,
                 dropout:float=0.3,
                 **kwargs):
        """
        :param d: hidden dimension size
        :param num_channels: channel size of input
        :param downsample_rate: should be a factor of 2; e.g., 2, 4, 8, 16, ...
        :param n_resnet_blocks: number of ResNet blocks
        :param kwargs:
        """
        super().__init__()
        self.n_fft = n_fft
        self.x_channels = x_channels
        
        kernel_size = (1, 4) if frequency_indepence else (3, 4)
        padding = (0, 1) if frequency_indepence else (1, 1)
        
        d = int(4 * 2**(int(round(np.log2(downsample_rate))) - 1))  # enc_out_dim == dec_in_dim
        if round(np.log2(downsample_rate)) == 0:
            d = int(4 * 2**(int(round(np.log2(downsample_rate)))))
        
        dec_layers = [ResBlock(dim, d, frequency_indepence, dropout=dropout)]
        for _ in range(int(round(np.log2(downsample_rate))) - 1):
            for _ in range(n_resnet_blocks):
                dec_layers.append(ResBlock(d, d, frequency_indepence, dropout=dropout))
            d //= 2
            dec_layers.append(VQVAEDecBlock(2*d, d, frequency_indepence))
        dec_layers.append(nn.ConvTranspose2d(d, num_channels, kernel_size=kernel_size, stride=(1, 2), padding=padding))
        dec_layers.append(nn.ConvTranspose2d(num_channels, num_channels, kernel_size=kernel_size, stride=(1, 2), padding=padding))
        self.decoder = nn.Sequential(*dec_layers)

        self.interp = nn.Upsample(input_length, mode='linear')
        self.linear = nn.Linear(input_length, input_length)
        

    def forward(self, x):
        out = self.decoder(x)  # (b c h w)
        out = timefreq_to_time(out, self.n_fft, self.x_channels)  # (b c l)

        out = self.interp(out)  # (b c l)
        out = out + self.linear(out)  # (b c l)
        return out
        

if __name__ == '__main__':
    import numpy as np

    x = torch.rand(1, 2, 4, 128)  # (batch, channels, height, width)

    encoder = VQVAEEncoder(dim=32, num_channels=2, downsample_rate=4, n_resnet_blocks=2)
    decoder = VQVAEDecoder(dim=32, num_channels=2, downsample_rate=4, n_resnet_blocks=2)
    decoder.upsample_size = torch.IntTensor(np.array(x.shape[2:]))

    z = encoder(x)
    x_recons = decoder(z)

    print('x.shape:', x.shape)
    print('z.shape:', z.shape)
    print('x_recons.shape:', x_recons.shape)
