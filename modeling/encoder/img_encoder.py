import torch
import torch.nn as nn

import torch.nn.functional as F

import torchvision.models as models


class VAE(nn.Module):
    def __init__(self, in_ch, out_ch, hidden_ch=128):
        super(VAE, self).__init__()

        self.in_ch = in_ch
        self.out_ch = out_ch

        self.fc1 = nn.Linear(in_ch, hidden_ch)
        self.fc21 = nn.Linear(hidden_ch, 20)
        self.fc22 = nn.Linear(hidden_ch, 20)
        self.fc3 = nn.Linear(20, hidden_ch)
        self.fc4 = nn.Linear(hidden_ch, out_ch)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, self.in_ch))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


class SegEncoder(nn.Module):
    def __init__(   self, 
                    img_sidelength=128,
                    num_classes=19,
                    latent_dim=256,
                    kernel_size=7,
                    shortcut=None):
        
        super().__init__()

        self.num_classes = num_classes
        self.img_sidelength = img_sidelength
        self.latent_dim = latent_dim

        if shortcut is not None:
            self.shortcut = shortcut
        else:
            self.shortcut = torch.empty(self.latent_dim).cuda()
            nn.init.normal_(self.shortcut, mean=0.0, std=1.0)
        
        n_hidden = 128

        ks = kernel_size
        pw = ks // 2

        self._emb = nn.Sequential(
            nn.Conv2d(
                num_classes, n_hidden, kernel_size=ks, padding=pw),
            nn.InstanceNorm2d(num_features=n_hidden,affine=False),
            nn.ReLU()
        )

        self._gamma = nn.Sequential(
            nn.Conv2d(n_hidden, self.latent_dim, kernel_size=ks, padding=pw),
            nn.AdaptiveAvgPool2d(output_size=(1, 1)))

        self._beta = nn.Sequential( 
            nn.Conv2d(n_hidden, self.latent_dim, kernel_size=ks, padding=pw),
            nn.AdaptiveAvgPool2d(output_size=(1, 1)))

    def forward(self, x):
        seg = x
        # print('**** seg = ', seg.shape)
        seg = F.one_hot(seg.squeeze().long(), num_classes=self.num_classes).permute(0, 2, 1)
        # print('**** seg_one_hot = ', seg.shape)

        seg = seg.view(-1, self.num_classes, self.img_sidelength, self.img_sidelength).float()

        embedding = self._emb(seg)
        gamma = self._gamma(embedding).squeeze()
        beta = self._beta(embedding).squeeze()

        # print('*** gamma = ', gamma.shape)
        # print('*** beta = ', beta.shape)

        out = self.shortcut * (1 + gamma) + beta

        # print('**** out = ', out.shape)

        return out

class ConvEncoder(nn.Module):
    def __init__(   self, 
                    img_sidelength=128,
                    num_classes=19,
                    latent_dim=256):
        
        super().__init__()

        self.img_sidelength = img_sidelength
        self.latent_dim = latent_dim  
        self.num_classes = num_classes      

        self.encoder = models.resnet18(pretrained=False)
        self.activation = nn.ReLU()
        self.encoder.fc = nn.Identity()

        if latent_dim < 512:
            self.encoder.layer4= nn.Identity()
        
        self.encoder.conv1 = nn.Conv2d(num_classes, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.activation = nn.Tanh()

    def forward(self, x):
        seg = x
        seg = F.one_hot(seg.squeeze().long(), num_classes=self.num_classes).permute(0, 2, 1)
        seg = seg.view(-1, self.num_classes, self.img_sidelength, self.img_sidelength).float()

        embedding = self.encoder(seg)
        embedding = self.activation(embedding)

        return embedding
