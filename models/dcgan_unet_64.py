import torch
import torch.nn as nn
from torch.nn.functional import normalize
from models.time_distributed import TimeDistributed


affine = True
running_stats = True
momentum = 0.1
eps = 1e-5
normalize = True


class dcgan_conv(nn.Module):
    def __init__(self, nin, nout, stride=2, kernel_size=4):
        super(dcgan_conv, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(in_channels=nin, out_channels=nout, kernel_size=kernel_size,
                      stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(num_features=nout, momentum=momentum, affine=affine,
                           track_running_stats=running_stats, eps=eps),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.main.apply(self.init_weights)

    def init_weights(self, m):
        if type(m) == nn.Conv2d:
            m.weight.data.normal_(0.0, 0.02)
        if type(m) == nn.BatchNorm2d:
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)

    def forward(self, input):
        return self.main(input)


class dcgan_upconv(nn.Module):
    def __init__(self, nin, nout, stride=2, kernel_size=4):
        super(dcgan_upconv, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(in_channels=nin, out_channels=nout, kernel_size=kernel_size,
                               stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(num_features=nout, momentum=momentum, affine=affine,
                           track_running_stats=running_stats, eps=eps),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.main.apply(self.init_weights)

    def init_weights(self, m):
        if type(m) == nn.ConvTranspose2d:
            m.weight.data.normal_(0.0, 0.02)
        if type(m) == nn.BatchNorm2d:
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)

    def forward(self, input):
        return self.main(input)


class content_encoder(nn.Module):
    def __init__(self, content_dim, nc=1, act=nn.Tanh()):
        super(content_encoder, self).__init__()
        nf = 32
        # input is (nc) x 64 x 64
        self.c1 = dcgan_conv(nc, nf, stride=2)
        # state size. (nf) x 32 x 32
        self.c2 = dcgan_conv(nf, nf, stride=2)
        # state size. (nf*2) x 16 x 16
        self.c3 = dcgan_conv(nf, nf, stride=2)
        # state size. (nf*4) x 8 x 8
        self.c4 = dcgan_conv(nf, nf, stride=2)
        # state size. (nf*8) x 4 x 4
        self.conv5 = nn.Conv2d(in_channels=nf, out_channels=content_dim,
                               kernel_size=4, stride=2, padding=0, bias=False)
        self.bn5 = nn.BatchNorm2d(num_features=content_dim, momentum=momentum,
                                  affine=affine, track_running_stats=running_stats, eps=eps)
        self.actv5 = act

        self.conv5.apply(self.init_weights)
        self.bn5.apply(self.init_weights)

    def init_weights(self, m):
        if type(m) == nn.Conv2d:
            m.weight.data.normal_(0.0, 0.02)
        if type(m) == nn.BatchNorm2d:
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)

    def forward(self, _input, bn_output=True):
        
        # Convert from time dim to stacked frames
        bs = _input.size()[0]    
        n_frames = _input.size()[1]   
        _input = _input.view((bs, 1, 3*n_frames, 84, 84)).squeeze(1)
        
        _input = _input / 255.0 # - 0.5
        h1 = self.c1(_input)
        h2 = self.c2(h1)
        h3 = self.c3(h2)
        h4 = self.c4(h3)
        h5 = self.conv5(h4)
        
         # -- branch with BN
        h5_bn = self.bn5(h5)
        h5_norm = self.actv5(h5_bn)

        # -- branch without BN
        h5 = self.actv5(h5)

        h5 = torch.squeeze(torch.squeeze(h5, dim=-1), dim=-1)
        h5_norm = torch.squeeze(torch.squeeze(h5_norm, dim=-1), dim=-1)
        skips = [h1, h2, h3, h4]
        
        if normalize:
            h5 = torch.nn.functional.normalize(h5, p=2)
            h5_norm = torch.nn.functional.normalize(h5_norm, p=2)

        return h5, h5_norm, skips


class decoder(nn.Module):
    def __init__(self, h_dim, nc=1):
        super(decoder, self).__init__()
        nf = 32
        self.upc1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=h_dim, out_channels=nf,
                               kernel_size=5, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(num_features=nf, momentum=momentum,
                           affine=affine, track_running_stats=running_stats, eps=eps),
            nn.LeakyReLU(0.2, inplace=True)
        )
        # state size. (nf*8) x 4 x 4
        self.upc2 = dcgan_upconv(nf * 2, nf)
        # state size. (nf*4) x 8 x 8
        self.upc3 = dcgan_upconv(nf * 2, nf, kernel_size=5)
        # state size. (nf*2) x 16 x 16
        self.upc4 = dcgan_upconv(nf * 2, nf)
        # state size. (nf) x 32 x 32
        self.upc5 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=nf*2, out_channels=3,
                               kernel_size=4, stride=2, padding=1, bias=False),
            nn.Sigmoid()
        )

        self.upc1.apply(self.init_weights)
        self.upc5.apply(self.init_weights)

    def init_weights(self, m):
        if type(m) == nn.ConvTranspose2d:
            m.weight.data.normal_(0.0, 0.02)
        if type(m) == nn.BatchNorm2d:
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)

    def forward(self, x):
        h, skips = x
        h = torch.unsqueeze(h, axis=3)
        h = torch.unsqueeze(h, axis=4)
        d1 = TimeDistributed(self.upc1, to_list=False)(h)
        d2 = TimeDistributed(self.upc2, to_list=False)(torch.cat([d1, skips[3]], 2))
        d3 = TimeDistributed(self.upc3, to_list=False)(torch.cat([d2, skips[2]], 2))
        d4 = TimeDistributed(self.upc4, to_list=False)(torch.cat([d3, skips[1]], 2))
        output = TimeDistributed(self.upc5, to_list=False)(torch.cat([d4, skips[0]], 2))
        return output 
    
    
class decoder_no_skips(nn.Module):
    def __init__(self, h_dim, nc=3):
        super(decoder_no_skips, self).__init__()
        nf = 64
        self.upc1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=h_dim, out_channels=nf,
                               kernel_size=5, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(num_features=nf, momentum=momentum,
                           affine=affine, track_running_stats=running_stats, eps=eps),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        self.upc2 = dcgan_upconv(nf, nf)
        self.upc3 = dcgan_upconv(nf, nf, kernel_size=5)
        self.upc4 = dcgan_upconv(nf, nf)
        self.upc5 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=nf, out_channels=3,
                               kernel_size=4, stride=2, padding=1, bias=False),
            nn.Sigmoid()
        )

        self.upc1.apply(self.init_weights)
        self.upc5.apply(self.init_weights)

    def init_weights(self, m):
        if type(m) == nn.ConvTranspose2d:
            m.weight.data.normal_(0.0, 0.02)
        if type(m) == nn.BatchNorm2d:
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)

    def forward(self, h):
        h = torch.unsqueeze(h, axis=3)
        h = torch.unsqueeze(h, axis=4)
        d1 = TimeDistributed(self.upc1, to_list=False)(h)
        d2 = TimeDistributed(self.upc2, to_list=False)(d1)
        d3 = TimeDistributed(self.upc3, to_list=False)(d2)
        d4 = TimeDistributed(self.upc4, to_list=False)(d3)
        output = TimeDistributed(self.upc5, to_list=False)(d4)
        return output
    
