import torchvision
from ..utils import common as util

from .pytorch_prototyping import Unet, UpsamplingNet, Conv2dSame
from .dimension_kernel import Trigonometric_kernel
from . import geometry

import torch
from torch import nn
import torch.nn.functional as F

import numbers
from . import hyperlayers
from torch.autograd import Variable


def init_recurrent_weights(self):
    for m in self.modules():
        if type(m) in [nn.GRU, nn.LSTM, nn.RNN]:
            for name, param in m.named_parameters():
                if 'weight_ih' in name:
                    nn.init.kaiming_normal_(param.data)
                elif 'weight_hh' in name:
                    nn.init.orthogonal_(param.data)
                elif 'bias' in name:
                    param.data.fill_(0)


def lstm_forget_gate_init(lstm_layer):
    for name, parameter in lstm_layer.named_parameters():
        if not "bias" in name: continue
        n = parameter.size(0)
        start, end = n // 4, n // 2
        parameter.data[start:end].fill_(1.)


def clip_grad_norm_hook(x, max_norm=10):
    total_norm = x.norm()
    total_norm = total_norm ** (1 / 2.)
    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1:
        return x * clip_coef


class DepthSampler(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self,
                xy,
                depth,
                cam2world,
                intersection_net,
                intrinsics):
        self.logs = list()

        batch_size, _, _ = cam2world.shape

        intersections = geometry.world_from_xy_depth(
            xy=xy, depth=depth, cam2world=cam2world, intrinsics=intrinsics)

        depth = geometry.depth_from_world(intersections, cam2world)

        if self.training:
            print(depth.min(), depth.max())

        return intersections, depth


class Raymarcher(nn.Module):
    def __init__(self,
                 num_feature_channels,
                 raymarch_steps,
                 orthogonal=False):
        super().__init__()

        self.n_feature_channels = num_feature_channels
        self.steps = raymarch_steps
        self.orthogonal = orthogonal

        self.dpt_min = -50
        self.dpt_max = 50

        self.dir_channels = 63
        self.lstm = hyperlayers.Linear(
            self.n_feature_channels, 1, 3, hidden_layer=3)

        self.counter = 0

        self.word_center = torch.tensor([0, 0.0, 0.0]).view(1, 1, 3).cuda()
        self.radius = 1.5


    def depth_init(self, world_coord, ray_dirs):
        dir_to_center = self.word_center.to(world_coord.device) - world_coord
        dir_to_center_len = torch.norm(dir_to_center, dim=2, keepdim=True)
        dir_to_center = dir_to_center / dir_to_center_len


        cos_theta = torch.sum(dir_to_center * ray_dirs,dim=2,keepdim=True)        
        sin_theta = torch.sqrt(1 - cos_theta**2+1e-6)
        sin_len,cos_len = sin_theta * dir_to_center_len, cos_theta * dir_to_center_len
        world_coord += ray_dirs * cos_len

        mask = (sin_len < self.radius).expand_as(world_coord)
        world_coord[mask] = world_coord[mask] - (ray_dirs*torch.sqrt(self.radius**2 - sin_len**2+1e-6))[mask]
        world_coord += ray_dirs * torch.zeros_like(cos_theta).normal_(mean=0.0, std=0.01).cuda()

        return world_coord

    def forward(self,
                cam2world,
                phi,
                uv,
                intrinsics,
                dpt=1e-6,
                dpt_scale=1.0,
                orthogonal=None,
                z=None, positionEncoder=None):

        batch_size, num_samples, _ = uv.shape
        log = list()

        orthogonal = self.orthogonal if orthogonal is None else orthogonal

        ray_dirs = geometry.get_ray_directions(uv,
                                               cam2world=cam2world,
                                               intrinsics=intrinsics,
                                               orthogonal=orthogonal)

        steps = self.steps = 10
        if isinstance(dpt, numbers.Number):
            initial_depth = torch.FloatTensor([dpt]).expand_as(uv)[:, :, :1]
            initial_depth = initial_depth.to(uv.device)

            world_coords = geometry.world_from_xy_depth(
                uv, initial_depth, intrinsics=intrinsics,
                cam2world=cam2world, orthogonal=orthogonal)

            return world_coords, initial_depth, log

        elif torch.is_tensor(dpt):
            assert dpt.shape[0] == batch_size
            initial_depth = dpt.view(batch_size, num_samples, 1).to(uv.device)

            world_coords = geometry.world_from_xy_depth(
                uv, initial_depth, intrinsics=intrinsics,
                cam2world=cam2world, orthogonal=orthogonal)
            steps = 4

        else:
            initial_depth = torch.zeros((batch_size, num_samples, 1)).normal_(mean=0.05, std=1e-3).cuda() * dpt_scale

        init_world_coords = geometry.world_from_xy_depth(uv,
                                                         initial_depth,
                                                         intrinsics=intrinsics,
                                                         cam2world=cam2world,
                                                         orthogonal=orthogonal)

        init_world_coords = self.depth_init(init_world_coords, ray_dirs)
        world_coords = [init_world_coords]
        depths = [initial_depth]
        states = [None]


        for step in range(self.steps):#
            if z is not None:
                if positionEncoder is not None:
                    v = phi(positionEncoder(world_coords[-1]), z)
                else:
                    v = phi(world_coords[-1], z)

            else:
                if positionEncoder is not None:
                    v = phi(positionEncoder(world_coords[-1]))
                else:
                    v = phi(world_coords[-1])

            signed_distance = self.lstm(v,ray_dirs)
            new_world_coords = world_coords[-1] + ray_dirs * signed_distance

            world_coords.append(new_world_coords)

        depth = geometry.depth_from_world(world_coords[-1], cam2world)
        depths.append(depth)
        log.append(('scalar', 'dmin_final', depths[-1].min(), 10))
        log.append(('scalar', 'dmax_final', depths[-1].max(), 10))

        if self.counter > 0 and (not self.counter % 1000):
            # Write tensorboard summary for each step of ray-marcher.
            drawing_depths = torch.stack(depths, dim=0)[:, 0, :, :]
            drawing_depths = util.lin2img(drawing_depths).repeat(1, 3, 1, 1)
            log.append(('image', 'raycast_progress',
                        torch.clamp(torchvision.utils.make_grid(drawing_depths, scale_each=False, normalize=True), 0.0,
                                    5),
                        100))

        self.counter += 1
        return world_coords[-1], depths[-1], log


class DeepvoxelsRenderer(nn.Module):
    def __init__(self,
                 nf0,
                 in_channels,
                 input_resolution,
                 img_sidelength,
                 out_channels=3):
        super().__init__()

        self.nf0 = nf0
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.input_resolution = input_resolution
        self.img_sidelength = img_sidelength

        self.num_down_unet = util.num_divisible_by_2(input_resolution)
        self.num_upsampling = util.num_divisible_by_2(img_sidelength) - self.num_down_unet

        self.build_net()

    def build_net(self):
        self.net = [
            Unet(in_channels=self.in_channels,
                 out_channels=self.out_channels if self.num_upsampling <= 0 else 4 * self.nf0,
                 outermost_linear=True if self.num_upsampling <= 0 else False,
                 use_dropout=True,
                 dropout_prob=0.1,
                 nf0=self.nf0 * (2 ** self.num_upsampling),
                 norm=nn.BatchNorm2d,
                 max_channels=8 * self.nf0,
                 num_down=self.num_down_unet)
        ]

        if self.num_upsampling > 0:
            self.net += [
                UpsamplingNet(per_layer_out_ch=self.num_upsampling * [self.nf0],
                              in_channels=4 * self.nf0,
                              upsampling_mode='transpose',
                              use_dropout=True,
                              dropout_prob=0.1),
                Conv2dSame(self.nf0, out_channels=self.nf0 // 2, kernel_size=3, bias=False),
                nn.BatchNorm2d(self.nf0 // 2),
                nn.ReLU(True),
                Conv2dSame(self.nf0 // 2, out_channels=self.out_channels, kernel_size=3)
            ]

        self.net += [nn.Tanh()]
        self.net = nn.Sequential(*self.net)

    def forward(self, input):
        batch_size, _, ch = input.shape
        input = input.permute(0, 2, 1).view(batch_size, ch, self.img_sidelength, self.img_sidelength)
        out = self.net(input)
        return out.view(batch_size, self.out_channels, -1).permute(0, 2, 1)


class ImAEDecoder(nn.Module):
    def __init__(self, in_dim, out_dim, pos_dim=3, gf_dim=128, pos_encoder=True):
        super(ImAEDecoder, self).__init__()

        self.in_dim = in_dim
        self.pos_dim = pos_dim
        self.out_dim = out_dim
        self.gf_dim = gf_dim

        if pos_encoder:
            self.pos_encoder = Trigonometric_kernel(input_dim=self.pos_dim)
            self.pos_dim = self.pos_encoder.calc_dim(self.pos_dim)

        self.linear_1 = nn.Linear(self.in_dim + self.pos_dim, self.gf_dim * 8, bias=True)
        self.linear_2 = nn.Linear(self.gf_dim * 8, self.gf_dim * 8, bias=True)
        self.linear_3 = nn.Linear(self.gf_dim * 8, self.gf_dim * 8, bias=True)
        self.linear_4 = nn.Linear(self.gf_dim * 8, self.gf_dim * 4, bias=True)
        self.linear_5 = nn.Linear(self.gf_dim * 4, self.gf_dim * 2, bias=True)
        self.linear_6 = nn.Linear(self.gf_dim * 2, self.gf_dim * 1, bias=True)
        self.linear_7 = nn.Linear(self.gf_dim * 1, self.out_dim, bias=True)

        nn.init.normal_(self.linear_1.weight, mean=0.0, std=0.02)
        nn.init.constant_(self.linear_1.bias, 0)
        nn.init.normal_(self.linear_2.weight, mean=0.0, std=0.02)
        nn.init.constant_(self.linear_2.bias, 0)
        nn.init.normal_(self.linear_3.weight, mean=0.0, std=0.02)
        nn.init.constant_(self.linear_3.bias, 0)
        nn.init.normal_(self.linear_4.weight, mean=0.0, std=0.02)
        nn.init.constant_(self.linear_4.bias, 0)
        nn.init.normal_(self.linear_5.weight, mean=0.0, std=0.02)
        nn.init.constant_(self.linear_5.bias, 0)
        nn.init.normal_(self.linear_6.weight, mean=0.0, std=0.02)
        nn.init.constant_(self.linear_6.bias, 0)
        nn.init.normal_(self.linear_7.weight, mean=1e-5, std=0.02)
        nn.init.constant_(self.linear_7.bias, 0)

    def forward(self, in_feat):

        B, L, C = in_feat.shape
        in_feat = in_feat.view(-1, C)  # (B, H*W, C) -> (B*H*W, C)

        if hasattr(self, 'pos_encoder'):
            pts = self.pos_encoder(in_feat[:, :3])
            in_feat = in_feat[:, 3:]
            in_feat = torch.cat([pts, in_feat], dim=-1)

        l1 = self.linear_1(in_feat)
        l1 = F.leaky_relu(l1, negative_slope=0.02, inplace=True)

        l2 = self.linear_2(l1)
        l2 = F.leaky_relu(l2, negative_slope=0.02, inplace=True)

        l3 = self.linear_3(l2)
        l3 = F.leaky_relu(l3, negative_slope=0.02, inplace=True)

        l4 = self.linear_4(l3)
        l4 = F.leaky_relu(l4, negative_slope=0.02, inplace=True)

        l5 = self.linear_5(l4)
        l5 = F.leaky_relu(l5, negative_slope=0.02, inplace=True)

        l6 = self.linear_6(l5)
        l6 = F.leaky_relu(l6, negative_slope=0.02, inplace=True)

        l7 = self.linear_7(l6)

        l7 = l7.view(B, L, self.out_dim)

        return l7

import math
import numpy as np
class PosEncodingNeRF(nn.Module):
    '''Module to add positional encoding as in NeRF [Mildenhall et al. 2020].'''
    def __init__(self, in_features, sidelength=None, fn_samples=None, use_nyquist=True):
        super().__init__()

        self.in_features = in_features

        if self.in_features == 3:
            self.num_frequencies = 10
        elif self.in_features == 2:
            assert sidelength is not None
            if isinstance(sidelength, int):
                sidelength = (sidelength, sidelength)
            self.num_frequencies = 4
            if use_nyquist:
                self.num_frequencies = self.get_num_frequencies_nyquist(min(sidelength[0], sidelength[1]))
        elif self.in_features == 1:
            assert fn_samples is not None
            self.num_frequencies = 4
            if use_nyquist:
                self.num_frequencies = self.get_num_frequencies_nyquist(fn_samples)

        self.out_dim = in_features + 2 * in_features * self.num_frequencies

    def get_num_frequencies_nyquist(self, samples):
        nyquist_rate = 1 / (2 * (2 * 1 / samples))
        return int(math.floor(math.log(nyquist_rate, 2)))

    def forward(self, coords):
        coords = coords.view(coords.shape[0], -1, self.in_features)

        coords_pos_enc = coords
        for i in range(self.num_frequencies):
            for j in range(self.in_features):
                c = coords[..., j]

                sin = torch.unsqueeze(torch.sin((2 ** i) * np.pi * c), -1)
                cos = torch.unsqueeze(torch.cos((2 ** i) * np.pi * c), -1)

                coords_pos_enc = torch.cat((coords_pos_enc, sin, cos), axis=-1)

        return coords_pos_enc.reshape(coords.shape[0], -1, self.out_dim)