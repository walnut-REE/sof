import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import bisect

import torchvision
import util

import skimage.measure
from torch.nn import functional as F


from layers.pytorch_prototyping import *
from layers import custom_layers
from layers import geometry
from layers import hyperlayers

_CAM_CENTER = (0., 0.0, 1.1)
_OBJ_CENTER = (0., 0.0, 0.1)


def _campos2matrix(
        camera_position, at=(_OBJ_CENTER,), up=((0., 1., 0.),), device: str = "cpu", fronto=False
) -> torch.Tensor:
    at = at if not fronto else (camera_position - torch.Tensor(
        [0., 0., 1.0]).to(camera_position.device))

    if not torch.is_tensor(camera_position):
        camera_position = torch.FloatTensor(camera_position).cuda()

    at = torch.FloatTensor(at).expand_as(camera_position).to(camera_position.device)
    up = torch.FloatTensor(up).expand_as(camera_position).to(camera_position.device)

    z_axis = F.normalize(at - camera_position, eps=1e-5)
    x_axis = F.normalize(torch.cross(z_axis, up), eps=1e-5)
    y_axis = F.normalize(torch.cross(x_axis, z_axis), eps=1e-5)
    R = torch.cat((x_axis[:, None, :], -y_axis[:, None, :], z_axis[:, None, :]), dim=1)

    return R


class SOFModel(nn.Module):
    def __init__(self,
                 num_instances,
                 latent_dim,
                 tracing_steps,
                 orthogonal=True,
                 hidden_dim=256,
                 feat_dim=None,
                 renderer='FC',
                 use_encoder=False,
                 freeze_networks=False,
                 animated=False,
                 sample_frames=None,
                 opt_cam=False,
                 out_channels=3,
                 img_sidelength=128,
                 output_sidelength=128):
        super().__init__()

        self.latent_dim = latent_dim
        self.animated = animated
        self.sample_frames = sample_frames
        self.opt_cam = opt_cam

        self.num_hidden_units_phi = hidden_dim
        self.phi_layers = 3  # includes the in and out layers
        self.rendering_layers = 3  # includes the in and out layers
        self.sphere_trace_steps = tracing_steps
        self.freeze_networks = freeze_networks
        self.out_channels = out_channels
        self.img_sidelength = img_sidelength
        self.output_sidelength = output_sidelength
        self.num_instances = num_instances
        self.orthogonal = orthogonal

        # List of logs
        self.logs = list()

        # Auto-decoder: each scene instance gets its own code vector z
        if use_encoder:
            self.mapping_fn = nn.Sequential(
                nn.Linear(feat_dim, self.latent_dim),
                nn.Tanh()
            ).cuda()

            self.get_embedding = lambda x: self.mapping_fn(x['params'].cuda())
            print('[INIT embedding] encoder.', feat_dim, self.latent_dim)

        elif self.animated > 0:
            self.sample_frames = torch.Tensor(sorted(sample_frames)).cuda()
            print('[INIT embedding] animation.',
                  self.sample_frames, self.animated)

            self.latent_codes = nn.Embedding(
                self.num_instances, self.latent_dim - self.animated).cuda()
            nn.init.normal_(self.latent_codes.weight, mean=0, std=0.01)
            self.ami_code = nn.Embedding(
                len(self.sample_frames), self.animated).cuda()
            nn.init.normal_(self.ami_code.weight, mean=0, std=0.01)

            self.get_embedding = lambda x: self.get_frame_embedding(
                x['instance_idx'].long().cuda(), x['observation_idx'].long().cuda())

        elif self.opt_cam:

            def reset_cam():
                print('num instances = ', self.num_instances)
                self.latent_codes = nn.Embedding(
                    self.num_instances, self.latent_dim + 3).cuda()
                nn.init.normal_(self.latent_codes.weight, mean=0, std=0.01)

                self.latent_codes.weight.data[:, :3].copy_(
                    torch.Tensor(_CAM_CENTER).expand(self.num_instances, -1))

            self.reset_cam = reset_cam

            reset_cam()
            self.get_embedding = lambda x: self.latent_codes(
                x['instance_idx'].long().cuda())

            print('[INIT embedding] optimize camera.')

        else:
            # print('*** Init embedding: (%d, %d)'%(self.num_instances, self.latent_dim))
            self.latent_codes = nn.Embedding(
                self.num_instances, self.latent_dim).cuda()
            nn.init.normal_(self.latent_codes.weight, mean=0, std=0.01)
            self.get_embedding = lambda x: self.latent_codes(
                x['instance_idx'].long().cuda())

            self.logs.append(("embedding", "", self.latent_codes.weight, 1000))
            print('[INIT embedding] variable.')


        # self.hyper_phi = hyperlayers.Linear(3, 256, self.latent_dim)
        self.hyper_phi = hyperlayers.HyperFC(hyper_in_ch=self.latent_dim,
                                             hyper_num_hidden_layers=1,
                                             hyper_hidden_ch=self.latent_dim,
                                             hidden_ch=self.num_hidden_units_phi,
                                             num_hidden_layers=self.phi_layers - 2,
                                             in_ch=3,
                                             out_ch=self.num_hidden_units_phi)

        self.ray_marcher = custom_layers.Raymarcher(num_feature_channels=self.num_hidden_units_phi,
                                                    raymarch_steps=self.sphere_trace_steps,
                                                    orthogonal=self.orthogonal)

        # self.positionEncoder = custom_layers.PosEncodingNeRF(3)

        self.renderer = renderer
        if renderer == 'Deepvoxels':
            print('[INIT renderer] Deepvoxels, with renderer = %s' % (renderer))
            self.pixel_generator = custom_layers.DeepvoxelsRenderer(
                nf0=32, in_channels=self.num_hidden_units_phi,
                input_resolution=self.img_sidelength, img_sidelength=self.img_sidelength,
                out_channels=self.out_channels)

        elif renderer == 'ImAE':
            print('[INIT renderer] ImAE, with renderer = %s' % (renderer))
            self.pixel_generator = custom_layers.ImAEDecoder(self.num_hidden_units_phi, self.out_channels)

        else:
            print('[INIT renderer] FC, with renderer = %s' % (renderer))
            self.pixel_generator = FCBlock(hidden_ch=self.num_hidden_units_phi,
                                           num_hidden_layers=self.rendering_layers - 1,
                                           in_features=self.num_hidden_units_phi,
                                           out_features=self.out_channels,
                                           outermost_linear=True)

        if self.freeze_networks:
            all_network_params = (list(self.pixel_generator.parameters())
                                  + list(self.ray_marcher.parameters())
                                  + list(self.hyper_phi.parameters()))

            for param in all_network_params:
                param.requires_grad = False

        # Losses
        self.l2_loss = nn.MSELoss(reduction="mean")

        # print(self)
        # print("Number of parameters:")
        # util.print_network(self)

    def reset_net_status(self):
        if self.freeze_networks:
            all_network_params = (list(self.pixel_generator.parameters())
                                  + list(self.ray_marcher.parameters())
                                  + list(self.hyper_phi.parameters()))

            for param in all_network_params:
                param.requires_grad = True
            print('[DONE] reset network status.')

    def get_frame_embedding(self, instance_idx, frame_idx):
        instance_emb = self.latent_codes(instance_idx)
        num_samples = len(self.sample_frames)
        sample_frames = self.sample_frames.unsqueeze(0)

        # print('*** frame_idx: ', frame_idx)
        # print('*** sample_frames: ', self.sample_frames)

        emb_idx = torch.sum(frame_idx.unsqueeze(
            1) >= sample_frames, axis=1).long().cuda()
        # print('*** emb_idx: ', emb_idx)
        emb_idx = (emb_idx - 1 + num_samples) % num_samples
        # print('*** emb_idx: ', emb_idx, emb_idx.shape, self.ami_code(emb_idx).shape)

        weights = torch.abs(frame_idx - sample_frames[:, emb_idx]) / (
                sample_frames[:, (emb_idx + 1) % num_samples] - sample_frames[:, emb_idx])
        # print('*** weights = ', weights.shape, weights)
        weights = weights.permute((1, 0))
        # print('*** weights = ', weights.shape, weights)
        ami_emb = self.ami_code(emb_idx) * (1.0 - weights) + \
                  self.ami_code((emb_idx + 1) % num_samples) * weights

        if self.animated == self.latent_dim:
            emb = instance_emb + ami_emb
        else:
            emb = torch.cat([instance_emb, ami_emb], axis=1)

        # print('*** emb = ', emb.shape)
        return emb

    def get_loss(self, prediction, ground_truth, opt):

        if opt.out_channels in [1, 3]:
            dist_loss = self.get_image_loss(prediction, ground_truth)
        else:
            dist_loss = self.get_cls_loss(prediction, ground_truth)

        reg_loss = self.get_regularization_loss(prediction, ground_truth)

        weighted_dist_loss = opt.l1_weight * dist_loss
        weighted_reg_loss = opt.reg_weight * reg_loss

        total_loss = (weighted_dist_loss + weighted_reg_loss)

        self.logs.append(("scalar", "Loss/total", total_loss, 100))
        self.logs.append(("scalar", "Loss/dist", weighted_dist_loss, 100))
        self.logs.append(("scalar", "Loss/reg", weighted_reg_loss, 100))

        if hasattr(self, 'latent_reg_loss'):
            latent_loss = self.get_latent_loss() * opt.kl_weight
            self.logs.append(("scalar", "Loss/latent", latent_loss, 100))

        if opt.geo_weight > 0:
            geo_loss = model.get_geo_loss(prediction, ground_truth)
            weighted_geo_loss = opt.geo_weight * geo_loss
            total_loss += weighted_geo_loss
            self.logs.append(("scalar", "Loss/reg", weighted_geo_loss, 100))

        return total_loss

    def get_regularization_loss(self, prediction, ground_truth):
        """Computes regularization loss on final depth map (L_{depth} in eq. 6 in paper)

        :param prediction (tuple): Output of forward pass.
        :param ground_truth: Ground-truth (unused).
        :return: Regularization loss on final depth map.
        """
        _, depth = prediction

        # print('*** depth = ', depth.shape, torch.zeros_like(depth).shape)
        # print(torch.max(depth), torch.min(depth))

        neg_penalty = (torch.min(depth, torch.zeros_like(depth)) ** 2)
        return torch.mean(neg_penalty) * 10000

    def get_image_loss(self, prediction, ground_truth):
        """Computes loss on predicted image (L_{img} in eq. 6 in paper)

        :param prediction (tuple): Output of forward pass.
        :param ground_truth: Ground-truth (unused).
        :return: image reconstruction loss.
        """
        pred_imgs, _ = prediction
        trgt_imgs = ground_truth['rgb']

        trgt_imgs = trgt_imgs.cuda()

        loss = self.l2_loss(pred_imgs, trgt_imgs)
        return loss

    def get_cls_loss(self, prediction, ground_truth):

        xent_loss = nn.CrossEntropyLoss(reduction='mean')

        # print('**** get_cls_loss: ')
        # print(ground_truth['rgb'].shape, np.unique(ground_truth['rgb'].detach().cpu().numpy()))
        # print(prediction[0].shape, np.unique(prediction[0].detach().cpu().numpy()))

        pred_imgs, _ = prediction

        pred_imgs = pred_imgs.permute(0, 2, 1)  # (B, C, H*W) in [0, 1], softmax prob

        trgt_imgs = ground_truth['rgb'].squeeze(-1).long()
        trgt_imgs = trgt_imgs.cuda()  # (B, H*W) in [0, 18] class label

        # compute softmax_xent loss
        loss = xent_loss(pred_imgs, trgt_imgs)

        # calc mIoU
        pred_onehot = F.one_hot(torch.argmax(
            pred_imgs, dim=1, keepdim=False).long(),
                                num_classes=self.out_channels)
        trgt_onehot = F.one_hot(
            trgt_imgs, num_classes=self.out_channels)

        IoU = torch.div(
            torch.sum(pred_onehot & trgt_onehot, dim=1).float(),
            torch.sum(pred_onehot | trgt_onehot, dim=1).float() + 1e-6)
        mIoU = IoU[IoU!=0].mean()

        # print('*** mIoU = ', pred_onehot.shape, trgt_onehot.shape, mIoU)

        self.logs.append(("scalar", "mIoU", mIoU, 1))

        return loss

    def get_geo_loss(self, prediction, ground_truth):
        assert 'depth' in ground_truth.keys(), 'GT depth does not exist.'

        geo_loss = nn.SmoothL1Loss(reduction='none')

        trgt_depth = ground_truth['depth'].cuda()

        pred_img, pred_depth = prediction

        nonzero_cnt = 1.0

        if pred_img.shape[-1] in [1, 3]:
            pred_img = torch.argmax(pred_img, dim=2).long().unsqueeze(2)
            # normalize pred_img
            pred_depth = pred_depth * (pred_img != 0).float()

        # print('*** geo_dpt_pred: ', z_range.shape, pred_depth.shape, torch.min(pred_depth), torch.max(pred_depth))
        # print('*** geo_dpt_trgt: ', z_range.shape, trgt_depth.shape, torch.min(trgt_depth), torch.max(trgt_depth))

        loss = geo_loss(pred_depth, trgt_depth)
        nonzero_cnt = (loss > 1e-12).sum(axis=1).clamp(min=1).float()

        # return torch.mean(loss.sum(axis=1) / nonzero_cnt)
        return torch.mean(loss.sum() / nonzero_cnt)

    def get_latent_loss(self):
        """Computes loss on latent code vectors (L_{latent} in eq. 6 in paper)
        :return: Latent loss.
        """

        # if hasattr(self, 'mapping_fn'):
        #     self.latent_reg_loss += -0.5 * \
        #         torch.sum(1 + self.logvar -
        #                   (self.mu).pow(2) - self.logvar.exp())

        return self.latent_reg_loss

    def get_psnr(self, prediction, ground_truth):
        """Compute PSNR of model image predictions.

        :param prediction: Return value of forward pass.
        :param ground_truth: Ground truth.
        :return: (psnr, ssim): tuple of floats
        """
        pred_imgs, _ = prediction
        trgt_imgs = ground_truth['rgb']

        trgt_imgs = trgt_imgs.cuda()
        batch_size = pred_imgs.shape[0]

        if not isinstance(pred_imgs, np.ndarray):
            pred_imgs = util.lin2img(pred_imgs).detach().cpu().numpy()

        if not isinstance(trgt_imgs, np.ndarray):
            trgt_imgs = util.lin2img(trgt_imgs).detach().cpu().numpy()

        psnrs, ssims = list(), list()
        for i in range(batch_size):
            p = pred_imgs[i].squeeze().transpose(1, 2, 0)
            trgt = trgt_imgs[i].squeeze().transpose(1, 2, 0)

            p = (p / 2.) + 0.5
            p = np.clip(p, a_min=0., a_max=1.)

            trgt = (trgt / 2.) + 0.5

            ssim = skimage.measure.compare_ssim(
                p, trgt, multichannel=True, data_range=1)
            psnr = skimage.measure.compare_psnr(p, trgt, data_range=1)

            psnrs.append(psnr)
            ssims.append(ssim)

        return psnrs, ssims

    def get_comparisons(self, model_input, prediction, ground_truth=None):
        predictions, depth_maps = prediction

        batch_size = predictions.shape[0]

        # Parse model input.
        intrinsics = model_input["intrinsics"].cuda()
        uv = model_input["uv"].cuda().float()

        x_cam = uv[:, :, 0].view(batch_size, -1)
        y_cam = uv[:, :, 1].view(batch_size, -1)
        z_cam = depth_maps.view(batch_size, -1)

        normals = geometry.compute_normal_map(
            x_img=x_cam, y_img=y_cam, z=z_cam,
            intrinsics=intrinsics, orthogonal=self.orthogonal)
        normals = F.pad(normals, pad=(1, 1, 1, 1), mode="constant", value=1.)

        predictions = util.lin2img(predictions)

        if ground_truth is not None:
            trgt_imgs = ground_truth["rgb"]
            trgt_imgs = util.lin2img(trgt_imgs)

            return torch.cat((normals.cpu(), predictions.cpu(), trgt_imgs.cpu()), dim=3).numpy()
        else:
            return torch.cat((normals.cpu(), predictions.cpu()), dim=3).numpy()

    def get_output_img(self, prediction):
        pred_imgs, _ = prediction
        return util.lin2img(pred_imgs)

    def write_updates(self, writer, predictions, ground_truth=None, iter=0, mode="", color_map=None):
        """Writes tensorboard summaries using tensorboardx api.

        :param writer: tensorboardx writer object.
        :param predictions: Output of forward pass.
        :param ground_truth: Ground truth.
        :param iter: Iteration number.
        :param prefix: Every summary will be prefixed with this string.
        """
        predictions, depth_maps = predictions
        batch_size, _, channels = predictions.shape

        if not channels in [1, 3]:
            # classification
            predictions = torch.argmax(predictions, dim=-1).long().unsqueeze(2)

        if ground_truth is not None:
            trgt_imgs = ground_truth['rgb']
            trgt_imgs = trgt_imgs.cuda()
            if not channels in [1, 3]:
                trgt_imgs = trgt_imgs.long()
            assert predictions.shape == trgt_imgs.shape
        else:
            trgt_imgs = None

        prefix = mode + '/'

        # Module"s own log
        for type, name, content, every_n in self.logs:
            name = prefix + name

            if (iter % every_n == 0):
                if type == "image":
                    writer.add_image(
                        name, content.detach().cpu().numpy(), iter)
                elif type == "figure":
                    writer.add_figure(name, content, iter, close=True)
                elif type == "histogram":
                    writer.add_histogram(
                        name, content.detach().cpu().numpy(), iter)
                elif type == "scalar":
                    writer.add_scalar(
                        name, content.detach().cpu().numpy(), iter)
                elif type == "embedding" and (mode == 'train'):
                    writer.add_embedding(mat=content, global_step=iter)
                elif type == "mesh":
                    vert, color = util.mat2mesh(content.detach().cpu().numpy())
                    writer.add_mesh(name, vertices=vert, colors=color)

        if (iter % 100 == 0) or (mode == 'test'):
            output_vs_gt = torch.cat(
                (predictions, trgt_imgs), dim=0) if trgt_imgs is not None else predictions
            output_vs_gt = util.lin2img(output_vs_gt, color_map)

            # print('*** output_vs_gt = ', output_vs_gt.shape, output_vs_gt.dtype)

            writer.add_image(prefix + "Output_vs_gt",
                             torchvision.utils.make_grid(output_vs_gt,
                                                         scale_each=False,
                                                         normalize=True).cpu().detach().numpy(),
                             iter)

            if trgt_imgs is not None:
                rgb_loss = ((predictions.float().cuda(
                ) - trgt_imgs.float().cuda()) ** 2).mean(dim=2, keepdim=True)
                rgb_loss = util.lin2img(rgb_loss)

                fig = util.show_images([rgb_loss[i].detach().cpu().numpy().squeeze()
                                        for i in range(batch_size)])
                writer.add_figure(prefix + "rgb_error_fig",
                                  fig,
                                  iter,
                                  close=True)

                # writer.add_scalar(prefix + "trgt_min", trgt_imgs.min(), iter)
                # writer.add_scalar(prefix + "trgt_max", trgt_imgs.max(), iter)

            depth_maps_plot = util.lin2img(depth_maps)
            writer.add_image(prefix + "pred_depth",
                             torchvision.utils.make_grid(depth_maps_plot.detach().cpu().repeat(1, 3, 1, 1),
                                                         scale_each=True,
                                                         normalize=True).numpy(),
                                                         iter)
            if 'depth' in ground_truth.keys():
                trgt_depth = ground_truth['depth'].float().cuda()
                pred_depth = (depth_maps * (predictions != 0)).float().cuda()
                geo_loss = torch.abs(pred_depth - trgt_depth)

                geo_loss = util.lin2img(geo_loss)
                fig = util.show_images([geo_loss[i].detach().cpu().numpy().squeeze()
                                        for i in range(batch_size)])
                writer.add_figure(prefix + "depth_error_fig",
                                  fig,
                                  iter,
                                  close=True)

        # writer.add_scalar(prefix + "out_min", predictions.min(), iter)
        # writer.add_scalar(prefix + "out_max", predictions.max(), iter)

        if hasattr(self, 'latent_reg_loss'):
            writer.add_scalar(prefix + "latent_reg_loss",
                              self.latent_reg_loss, iter)

    def forward(self, pose, z, intrinsics, uv, dpt=None, dpt_scale=1.0, device=None, auc_input=None, orthogonal=None):

        # Parse model input.
        # instance_idcs = input["instance_idx"].long().cuda()
        # pose = input["pose"].cuda()
        # intrinsics = input["intrinsics"].cuda()
        # uv = input["uv"].cuda().float()

        intrinsics = intrinsics.cuda()
        uv = uv.cuda().float()
        self.z = z.cuda()
        cam_pose = pose.cuda()

        # print(pose.shape, intrinsics.shape, z.shape)

        orthogonal = self.orthogonal if orthogonal is None else orthogonal

        #         print('*** opt_cam = ', self.opt_cam)

        if self.opt_cam:
            # parse
            #             print('*** training = ', self.training)
            if self.training:
                cam_pose = torch.eye(4, requires_grad=True).unsqueeze(
                    0).expand(self.z.shape[0], -1, -1).cuda()

                cam_pose[:, :3, :3] = _campos2matrix(self.z[:, :3]).cuda()
                cam_pose[:, :3, 3] = self.z[:, :3]

            self.z = self.z[:, 3:]

        self.latent_reg_loss = torch.mean(self.z ** 2)
        # Forward pass through hypernetwork yields a (callable) SOF.
        phi = self.hyper_phi(self.z)

        # print('*** 2: ray_marcher.')

        # print('*** forward - init ', torch.cuda.memory_allocated() - cur_mem)
        # cur_mem = torch.cuda.memory_allocated()


        # self.z = z[:,None].expand(-1, uv.shape[1], -1)
        # Raymarch SOF phi along rays defined by camera pose, intrinsics and uv coordinates.
        points_xyz, depth_maps, log = self.ray_marcher(cam2world=cam_pose,
                                                       intrinsics=intrinsics,
                                                       uv=uv,
                                                       dpt=dpt,
                                                       dpt_scale=dpt_scale,
                                                       phi=phi,#self.hyper_phi,
                                                       orthogonal=orthogonal,
                                                       #z=self.z,
                                                       # positionEncoder=self.positionEncoder)

                                                       )
        # print(uv)
        # print(points_xyz)
        # print('*** forward - ray_marcher ', torch.cuda.memory_allocated()-cur_mem)
        # cur_mem = torch.cuda.memory_allocated()

        # Sapmle phi a last time at the final ray-marched world coordinates.
        # v = self.hyper_phi(torch.cat((self.z,self.positionEncoder(points_xyz)),dim=2))
        v = phi(points_xyz)

        # print('*** forward - hyper ', torch.cuda.memory_allocated())
        # cur_mem = torch.cuda.memory_allocated()
        # Translate features at ray-marched world coordinates to RGB colors.

        ############ with implicit encoder #############
        if self.renderer == 'ImAE':
            # print('*** v = ', v.shape, points_xyz.shape)
            v = torch.cat([v, points_xyz], dim=2)

        novel_views = self.pixel_generator(v)

        # print('***** novel_views = ', novel_views.shape)

        # print('*** forward - pixel_generator ', torch.cuda.memory_allocated()-cur_mem)
        # cur_mem = torch.cuda.memory_allocated()

        if self.output_sidelength != self.img_sidelength:
            novel_views = novel_views.permute(0, 2, 1).view(
                -1, self.out_channels, self.img_sidelength, self.img_sidelength)

            novel_views = F.interpolate(
                F.softmax(novel_views, dim=1),
                size=(self.output_sidelength, self.output_sidelength),
                mode='bilinear', align_corners=True)

            novel_views = novel_views.view(-1, self.out_channels, self.output_sidelength ** 2).permute(0, 2, 1)

            # print('*** forward - up_scaling ', torch.cuda.memory_allocated()-cur_mem)
            # cur_mem = torch.cuda.memory_allocated()

        # Calculate normal map
        if self.training:
            # log saves tensors that"ll receive summaries when model"s write_updates function is called
            self.logs = list()
            self.logs.extend(log)

            with torch.no_grad():
                batch_size = uv.shape[0]
                x_cam = uv[:, :, 0].view(batch_size, -1)
                y_cam = uv[:, :, 1].view(batch_size, -1)
                z_cam = depth_maps.view(batch_size, -1)

                normals = geometry.compute_normal_map(
                    x_img=x_cam, y_img=y_cam, z=z_cam,
                    intrinsics=intrinsics, orthogonal=orthogonal)
                self.logs.append(("image", "normals",
                                  torchvision.utils.make_grid(normals, scale_each=True, normalize=True), 100))

            self.logs.append(("histogram", "embedding", self.z, 1000))
            # if self.opt_cam:
            #     self.logs.append(("mesh", "", self.cam_pose.weight, 100))

            # self.logs.append(
            #     ("embedding", "", self.latent_codes.weight, 1000))

            # self.logs.append(("scalar", "embed_min", self.z.min(), 1))
            # self.logs.append(("scalar", "embed_max", self.z.max(), 1))
        # print('***** depth_map = ', depth_maps.shape)

        return novel_views, depth_maps
