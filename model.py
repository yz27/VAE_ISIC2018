import torch
import torch.nn as nn
import torch.nn.functional as F
import math

ngf = 64
nz = 100
nc = 3
class VAE(nn.Module):
    def __init__(self, imageSize):
        super(VAE, self).__init__()
        self.nz = nz
        n = math.log2(imageSize)

        assert n == round(n), 'imageSize must be a power of 2'
        assert n >= 3, 'imageSize must be at least 8'
        n = int(n)

        self.conv_mu = nn.Conv2d(ngf * 2 ** (n - 3), nz, 4)
        self.conv_logvar = nn.Conv2d(ngf * 2 ** (n - 3), nz, 4)

        self.encoder = nn.Sequential()
        # input is (nc) x 64 x 64
        self.encoder.add_module('input-conv', nn.Conv2d(nc, ngf, 4, 2, 1, bias=False))
        self.encoder.add_module('input-relu', nn.LeakyReLU(0.2, inplace=True))
        for i in range(n - 3):
            # state size. (ngf) x 32 x 32
            self.encoder.add_module('pyramid_{0}-{1}_conv'.format(ngf * 2 ** i, ngf * 2 ** (i + 1)),
                                    nn.Conv2d(ngf * 2 ** (i), ngf * 2 ** (i + 1), 4, 2, 1, bias=False))
            self.encoder.add_module('pyramid_{0}_batchnorm'.format(ngf * 2 ** (i + 1)),
                                    nn.BatchNorm2d(ngf * 2 ** (i + 1)))
            self.encoder.add_module('pyramid_{0}_relu'.format(ngf * 2 ** (i + 1)), nn.LeakyReLU(0.2, inplace=True))

        self.decoder = nn.Sequential()
        # input is Z, going into a convolution
        self.decoder.add_module('input-conv', nn.ConvTranspose2d(nz, ngf * 2 ** (n - 3), 4, 1, 0, bias=False))
        self.decoder.add_module('input-batchnorm', nn.BatchNorm2d(ngf * 2 ** (n - 3)))
        self.decoder.add_module('input-relu', nn.LeakyReLU(0.2, inplace=True))

        # state size. (ngf * 2**(n-3)) x 4 x 4

        for i in range(n - 3, 0, -1):
            self.decoder.add_module('pyramid_{0}-{1}_conv'.format(ngf * 2 ** i, ngf * 2 ** (i - 1)),
                                    nn.ConvTranspose2d(ngf * 2 ** i, ngf * 2 ** (i - 1), 4, 2, 1, bias=False))
            self.decoder.add_module('pyramid_{0}_batchnorm'.format(ngf * 2 ** (i - 1)),
                                    nn.BatchNorm2d(ngf * 2 ** (i - 1)))
            self.decoder.add_module('pyramid_{0}_relu'.format(ngf * 2 ** (i - 1)), nn.LeakyReLU(0.2, inplace=True))

        self.decoder.add_module('ouput-conv', nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False))
        self.decoder.add_module('output-tanh', nn.Tanh())

    def encode(self, input):
        output = self.encoder(input)
        return [self.conv_mu(output), self.conv_logvar(output)]

    def reparameterize(self, mu, logvar):
        if not self.training:
            return mu
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar