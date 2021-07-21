from __future__ import print_function

import os, struct, math
import numpy as np
import torch
from glob import glob
from typing import List, Tuple

import cv2
import torch.nn.functional as F

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.spatial.transform import Rotation as R
from mpl_toolkits.mplot3d import Axes3D

from matplotlib import cm
import re


def get_latest_file(root_dir):
    """Returns path to latest file in a directory."""
    list_of_files = glob.glob(os.path.join(root_dir, '*'))
    latest_file = max(list_of_files, key=os.path.getctime)
    return latest_file


def parse_num_range(s):
    '''Accept either a comma separated list of numbers 'a,b,c' or a range 'a-c' and return as a list of ints.'''

    range_re = re.compile(r'^(\d+)-(\d+)$')
    m = range_re.match(s)
    if m:
        return list(range(int(m.group(1)), int(m.group(2))+1))
    vals = s.split(',')
    return [int(x) for x in vals]

def convert_image(img):
    if not isinstance(img, np.ndarray):
        img = np.array(img.cpu().detach().numpy())

    img = img.squeeze()
    img = img.transpose(1,2,0)
    img += 1.
    img /= 2.
    img *= 2**8 - 1
    img = img.round().clip(0, 2**8-1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def write_img(img, path):
    cv2.imwrite(path, img.astype(np.uint8))

def in_out_to_param_count(in_out_tuples):
    return np.sum([np.prod(in_out) + in_out[-1] for in_out in in_out_tuples])

def lin2img(tensor, color_map=None):
    batch_size, num_samples, channels = tensor.shape
    sidelen = np.sqrt(num_samples).astype(int)

    if color_map is not None:
        tensor = tensor.squeeze(2).long()
        output_img = torch.cat([color_map[p].unsqueeze(0) for p in tensor], dim=0)
        channels = output_img.shape[-1]
 
    else:
        output_img = tensor

    return output_img.permute(0,2,1).view(batch_size, channels, sidelen, sidelen)

    

def num_divisible_by_2(number):
    i = 0
    while not number%2:
        number = number // 2
        i += 1

    return i

def cond_mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def load_pose(filename):
    assert os.path.isfile(filename)
    lines = open(filename).read().splitlines()
    assert len(lines) == 4
    lines = [[x[0],x[1],x[2],x[3]] for x in (x.split(" ") for x in lines)]
    return torch.from_numpy(np.asarray(lines).astype(np.float32))


def normalize(img):
    return (img - img.min()) / (img.max() - img.min())


def write_image(writer, name, img, iter):
    writer.add_image(name, normalize(img.permute([0,3,1,2])), iter)


def print_network(net):
    model_parameters = filter(lambda p: p.requires_grad, net.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print("%d"%params)


def custom_load(model, path, discriminator=None, overwrite_embeddings=False, overwrite_renderer=False, overwrite_cam=True, optimizer=None):

    if os.path.isdir(path):
        checkpoint_path = sorted(glob(os.path.join(path, "*.pth")))[-1]
    else:
        checkpoint_path = path

    whole_dict = torch.load(checkpoint_path)

    if overwrite_embeddings:
        del whole_dict['model']['latent_codes.weight']
    if overwrite_cam:
        if 'cam_pose.weight' in whole_dict['model']:
            del whole_dict['model']['cam_pose.weight']

    if overwrite_renderer:
        keys_to_remove = [key for key in whole_dict['model'].keys() if 'rendering_net' in key]
        for key in keys_to_remove:
            print(key)
            whole_dict['model'].pop(key, None)

    state = model.state_dict()
    state.update(whole_dict['model'])
    model.load_state_dict(state)

    if discriminator:
        discriminator.load_state_dict(whole_dict['discriminator'])

    if optimizer:
        optimizer.load_state_dict(whole_dict['optimizer'])


def custom_save(model, path, discriminator=None, optimizer=None):
    whole_dict = {'model':model.state_dict()}
    if discriminator:
        whole_dict.update({'discriminator':discriminator.state_dict()})
    if optimizer:
        whole_dict.update({'optimizer':optimizer.state_dict()})

    torch.save(whole_dict, path)


def show_images(images, titles=None):
    """Display a list of images in a single figure with matplotlib.
    Parameters
    ---------
    images: List of np.arrays compatible with plt.imshow.

    cols (Default = 1): Number of columns in figure (number of rows is
                        set to np.ceil(n_images/float(cols))).

    titles: List of titles corresponding to each image. Must have
            the same length as titles.
    """
    assert ((titles is None) or (len(images) == len(titles)))
    cols = np.ceil(np.sqrt(len(images))).astype(int)

    n_images = len(images)
    if titles is None: titles = ['Image (%d)' % i for i in range(1, n_images + 1)]
    fig = plt.figure()
    for n, (image, title) in enumerate(zip(images, titles)):
        a = fig.add_subplot(np.ceil(n_images / float(cols)), cols, n + 1)
        im = a.imshow(image)

        a.get_xaxis().set_visible(False)
        a.get_yaxis().set_visible(False)

        if len(images) < 10:
            divider = make_axes_locatable(a)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            fig.colorbar(im, cax=cax, orientation='vertical')


    plt.tight_layout()

    # fig.set_size_inches(np.array(fig.get_size_inches()) * n_images)
    return fig

# def transform_points(self, points, eps: Optional[float] = None):
#     """
#     Use this transform to transform a set of 3D points. Assumes row major
#     ordering of the input points.
#     Args:
#         points: Tensor of shape (P, 3) or (N, P, 3)
#         eps: If eps!=None, the argument is used to clamp the
#             last coordinate before peforming the final division.
#             The clamping corresponds to:
#             last_coord := (last_coord.sign() + (last_coord==0)) *
#             torch.clamp(last_coord.abs(), eps),
#             i.e. the last coordinates that are exactly 0 will
#             be clamped to +eps.
#     Returns:
#         points_out: points of shape (N, P, 3) or (P, 3) depending
#         on the dimensions of the transform
#     """
#     points_batch = points.clone()
#     if points_batch.dim() == 2:
#         points_batch = points_batch[None]  # (P, 3) -> (1, P, 3)
#     if points_batch.dim() != 3:
#         msg = "Expected points to have dim = 2 or dim = 3: got shape %r"
#         raise ValueError(msg % repr(points.shape))

#     N, P, _3 = points_batch.shape
#     ones = torch.ones(N, P, 1, dtype=points.dtype, device=points.device)
#     points_batch = torch.cat([points_batch, ones], dim=2)

#     composed_matrix = self.get_matrix()
#     points_out = _broadcast_bmm(points_batch, composed_matrix)
#     denom = points_out[..., 3:]  # denominator
#     if eps is not None:
#         denom_sign = denom.sign() + (denom == 0.0).type_as(denom)
#         denom = denom_sign * torch.clamp(denom.abs(), eps)
#     points_out = points_out[..., :3] / denom

#     # When transform is (1, 4, 4) and points is (P, 3) return
#     # points_out of shape (P, 3)
#     if points_out.shape[0] == 1 and points.dim() == 2:
#         points_out = points_out.reshape(points.shape)

#     return points_out


def plot_camera_scene(  cam_pose, 
                        scale: float = 0.3, 
                        margin: float = 0.5, 
                        fig_size: Tuple[float] = (10, 5)):
    """
    Plots a set of predicted cameras `cameras` and their corresponding
    ground truth locations `cameras_gt`. The plot is named with
    a string passed inside the `status` argument.
    """    
    
    def _get_camera_wireframe(scale: float = 0.3):
        """
        Returns a wireframe of a 3D line-plot of a camera symbol.
        """
        # a = 0.5 * np.array([-2, 1.5, 4])
        # b = 0.5 * np.array([2, 1.5, 4])
        # c = 0.5 * np.array([-2, -1.5, 4])
        # d = 0.5 * np.array([2, -1.5, 4])
        # C = np.zeros(3)
        # F = np.array([0, 0, 3])
        # camera_points = [a, b, d, c, a, C, b, d, C, c, C, F]
        # lines = np.stack([x.astype(np.float) for x in camera_points]) * scale

        C = np.zeros(3)
        x = np.array([1.0, 0.0, 0.0])
        y = np.array([0.0, -2.0, 0.0])
        z = np.array([0.0, 0.0, 5.0])

        camera_points = [C, x, C, y, C, z]

        lines = np.stack([x.astype(np.float) for x in camera_points]) * scale
        return lines


    def _plot_cameras(ax, cam_pose):
        """
        Plots a set of `cameras` objects into the maplotlib axis `ax` with
        color `color`.
        """

        color = ['red', 'red', 'green', 'green', 'blue', 'blue']

        cam_wires_canonical = _get_camera_wireframe(scale=scale)
        cam_wires_canonical = np.expand_dims(cam_wires_canonical.T, axis=0)

        cam_R = cam_pose[:, :3, :3]
        cam_T = cam_pose[:, :3, 3:]

        cam_wires_trans = cam_R@cam_wires_canonical
        cam_wires_trans = cam_wires_trans + cam_T
        plot_handles = []

        print(cam_wires_trans.shape, cam_wires_canonical.shape)

        for idx in range(cam_wires_trans.shape[0]):
            # the Z and Y axes are flipped intentionally here!
            x_, z_, y_ = cam_wires_trans[idx].astype(float)

            (h,) = ax.plot(x_, y_, z_, color=color[idx%6], linewidth=0.3)
            plot_handles.append(h)

        return plot_handles
    
    fig = plt.figure(figsize=(fig_size))
    ax = fig.gca(projection="3d")
    ax.clear()

    handle_cam = _plot_cameras(ax, cam_pose)
    ax.set_xlim3d([np.min(cam_pose[:, 0, 3]) - margin, np.max(cam_pose[:, 0, 3])+margin])
    ax.set_ylim3d([np.min(cam_pose[:, 2, 3]) - margin, np.max(cam_pose[:, 2, 3])+margin])
    ax.set_zlim3d([np.min(cam_pose[:, 1, 3]) - margin, np.max(cam_pose[:, 1, 3])+margin])
    
    ax.set_xlabel("x")
    ax.set_ylabel("z")
    ax.set_zlabel("y")
    
#     plt.show()
    return fig
