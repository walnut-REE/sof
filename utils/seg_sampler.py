import sys
import os
import configargparse

SOF_ROOT_DIR = 'sofgan_test/modules/SOF'
sys.path.append(SOF_ROOT_DIR)

import torch
import torch.nn
import torch.nn.functional as F
import numpy as np
from sklearn import mixture

from utils.volRenderer import render_scene_cam
from utils.common import custom_load

from modeling import SOFModel
from dataset.face_dataset import _campos2matrix
from glob import glob


_FOCAL = 36
_IMG_SIZE = 128
_OUT_SIZE = 512
_DEFAULT_MODEL_PATH = os.path.join(
    SOF_ROOT_DIR, 'checkpoints/epoch_0250_iter_050000.pth')
_DEFAULT_INT = np.array(
    [[_FOCAL,0,_IMG_SIZE//2],
    [0,_FOCAL,_IMG_SIZE//2],
    [0,0,1]])


def _parse_intrinsics(filepath, trgt_sidelength=None, invert_y=False):
    # Get camera intrinsics
    with open(filepath, 'r') as file:
        f, cx, cy, _ = map(float, file.readline().split())
        file.readline()
        scale = float(file.readline())
        height, width = map(float, file.readline().split())

    if trgt_sidelength is not None:
        cx = cx/width * trgt_sidelength
        cy = cy/height * trgt_sidelength
        f = trgt_sidelength / height * f

    fx = f
    if invert_y:
        fy = -f
    else:
        fy = f

    # Build the intrinsic matrices
    full_intrinsic = np.array([[fx, 0., cx, 0.],
                               [0., fy, cy, 0],
                               [0., 0, 1, 0],
                               [0, 0, 0, 1]])

    return full_intrinsic


def _rand_cam_sphere(   R=1.5, 
                        num_samples=15, 
                        random=False, 
                        cam_center=None,
                        look_at=None, 
                        sample_range=None):
    side_len = np.ceil(np.sqrt(num_samples)).astype(np.uint8)
    cam_pos = []

    _PHI_RANGE = sample_range[0] if sample_range is not None else [np.pi/2-0.6, np.pi/2+0.6]
    _THETA_RANGE = sample_range[1] if sample_range is not None else [np.pi/2-0.3, np.pi/2+0.3]

    if random:
        _theta = np.random.random_sample((side_len,)) * (_THETA_RANGE[1]-_THETA_RANGE[0]) + _THETA_RANGE[0]
        _phi = np.random.random_sample((side_len,)) * (_PHI_RANGE[1]-_PHI_RANGE[0]) + _PHI_RANGE[0]
    else:
        _theta = np.linspace(_THETA_RANGE[0], _THETA_RANGE[1], num=side_len)
        _phi = np.linspace(_PHI_RANGE[0], _PHI_RANGE[1], num=side_len)
    
    _p = 1
    _idx = 0

    for theta in _theta:
        for i in range(len(_phi)):
            _cur_idx = int(_idx+i*_p)
            phi = _phi[_cur_idx]
            x = R * np.sin(theta) * np.cos(phi)
            y = R * np.sin(theta) * np.sin(phi)
            z = R * np.cos(theta)
            
            cam_pos.append(np.array([x, z, y])+cam_center)
        
        _p *= -1
        _idx = _cur_idx
    
    cam_pos = cam_pos[:num_samples]
    cam_pos = np.asarray(cam_pos)
    
    return cam_pos


def _rand_cam_plane(R=1.2, 
                    num_samples=15, 
                    cam_center=None,
                    look_at=None, 
                    sample_range=None):
    side_len = np.ceil(np.sqrt(num_samples)).astype(np.uint8)
    cam2world = []

    _X_RANGE = sample_range[0] if sample_range is not None else [0.05, -0.05]
    _Y_RANGE = sample_range[1] if sample_range is not None else [0.13, 0.12]

    _x = np.linspace(_X_RANGE[0], _X_RANGE[1], side_len)
    _y = np.linspace(_Y_RANGE[0], _Y_RANGE[1], side_len)

    _p = 1
    _idx = 0

    for x in _x:
        for i in range(len(_y)):
            _cur_idx = int(_idx + i*_p)
            y = _y[_cur_idx]

            cam2world.append(_campos2matrix(
                np.array([x, y, R]+cam_center), 
                (np.array([x, y, R-1.0])+cam_center).reshape((1, 3))))
        _p *= -1
        _idx = _cur_idx

    cam2world = np.asarray(cam2world)
    
    return cam2world


def _rand_cam_uniform(  R=1.2, 
                        num_samples=15, 
                        cam_center=None,
                        look_at=None,
                        sample_range=None):
    
    cam2world = []
    
    theta = []
    _THETA_RANGE = sample_range if sample_range is not None else [-0.55, 0.55]

    for i in range(len(_THETA_RANGE)-1):
        theta.append( np.linspace(
            _THETA_RANGE[i],_THETA_RANGE[i+1], num=num_samples))
        
    ys = np.linspace(0.3,-0.2,5,endpoint=True)
    
    theta = np.concatenate(theta)
    x = R*np.sin(theta)
    z = R*np.cos(theta)
    
    for y in ys:        
        cam_T = np.stack([x,np.ones_like(x)*y,z],axis=1) + cam_center.reshape((1,3))
        for i in range(len(theta)):
            cam_pose = _campos2matrix(cam_T[i], cam_center)
            cam2world.append(cam_pose)        
    
    cam2world = np.asarray(cam2world)
    return cam2world


def _rand_cam_spiral(   R=1.2, 
                        num_samples=15, 
                        cam_center=None,
                        look_at=None,
                        sample_range=None):

    cam2world = []

    # ROTATE
    spiral_R = np.linalg.norm(cam_center-look_at) + R

    theta = []
    theta_range = [0.0, -0.45, 0.45, 0.0]
    for i in range(len(theta_range)-1):
        theta.append( np.linspace(theta_range[i],theta_range[i+1], num=num_samples))
    theta = np.concatenate(theta)
    x = spiral_R*np.sin(theta)
    y = np.zeros_like(x)
    z = spiral_R*np.cos(theta)
    cam_T = np.stack([x,y,z],axis=1) + look_at.reshape((1,3))

     # SPPIRAL PATH
    t = np.linspace(0, 4*np.pi, num_samples, endpoint=True)
    for k in range(len(t)):
        cam_T = np.array([np.cos(t[k]), -np.sin(t[k]), -np.sin(0.5*t[k])]) * R
        cam_T = cam_T[[1,2,0]] + cam_center
        cam_pose = _campos2matrix(cam_T, look_at)
        cam2world.append(cam_pose)

    cam2world = np.asarray(cam2world)
    return cam2world


def _get_random_poses(
    sample_radius, num_samples, mode, 
    cam_center=None, look_at=None, 
    cam_pos=None, sample_range=None):
    if cam_pos is None:
        if mode == 'sphere':
            cam_pos = _rand_cam_sphere(
                sample_radius, num_samples, 
                cam_center=cam_center, 
                look_at=look_at, 
                sample_range=sample_range)            
        elif mode == 'plane':
            return _rand_cam_plane(
                sample_radius, num_samples, 
                cam_center=cam_center, 
                look_at=look_at, 
                sample_range=sample_range)
        elif mode == 'spiral':
            return _rand_cam_spiral(
                sample_radius, num_samples, 
                cam_center=cam_center, 
                look_at=look_at, 
                sample_range=sample_range)
        elif mode == 'uniform':
            return _rand_cam_uniform(
                sample_radius, num_samples, 
                cam_center=cam_center, 
                look_at=look_at, 
                sample_range=sample_range)
        else:
            raise ValueError('Unsupported camera path: %s, \
                must be one in [sphere, plane, spiral, uniform].'%(mode))
    
    assert cam_pos is not None, 'Campose not specified'
    cam2world = []

    for i in range(num_samples):
        cam2world.append(_campos2matrix(cam_pos[i], np.asarray(cam_center).reshape((1, 3))))

    cam2world = np.asarray(cam2world)

    return cam2world

class FaceSegSampler():
    def __init__(   self,
                    model_path=_DEFAULT_MODEL_PATH,
                    img_size=64,
                    num_instances=221,
                    num_poses=25,
                    sample_mode='spiral',
                    sample_radius=1.2):
        super().__init__()
        # init SOF model
        self.num_instances = num_instances
        self.num_poses = num_poses
        self.model_path = model_path
        self.sample_mode = sample_mode
        self.sample_radius = sample_radius
        self.img_size = img_size
        
        self.model = SOFModel(num_instances=self.num_instances,
                              latent_dim=256,
                              renderer='FC',
                              tracing_steps=10,
                              freeze_networks=True,
                              out_channels=20,
                              img_sidelength=128,
                              output_sidelength=128,
                              opt_cam=False,
                              orthogonal=True
                             )

        print('DONE Load model.')

        custom_load(self.model,
                   path=self.model_path,
                   discriminator=None,
                   overwrite_embeddings=False)
        
        print('DONE Load ckpt.')

        self.model.eval()
        self.model.cuda()

        num_comp = 16
        self.gmm = mixture.GaussianMixture(
            n_components=num_comp, covariance_type='full')
        self.gmm.fit(self.model.latent_codes.weight.data.cpu().numpy())

        print('DONE Build sampling space.')        

        self.intrinsics = torch.from_numpy(_DEFAULT_INT).float().unsqueeze(0)
        
        self.uv = np.mgrid[0:128, 0:128].astype(np.int32)
        self.uv = torch.from_numpy(np.flip(self.uv, axis=0).copy()).long()
        self.uv = self.uv.reshape(2, -1).transpose(1, 0)
        

    def sample_ins(self, num_samples=100, cam2world=None, return_feat=False):
        """ Sample num_samples random cameras for random instance. 
            If cam2world is given, sample instances with the given pose.
        """
        with torch.no_grad():
            
            src_emb = torch.from_numpy(self.gmm.sample(1)[0]).float()
            trgt_emb = torch.from_numpy(self.gmm.sample(1)[0]).float()

            weights = torch.rand((num_samples,)).unsqueeze(1).to(src_emb.device)

            z_interp = (src_emb * weights + trgt_emb * (1.0 - weights))

            intrinsics = self.intrinsics.repeat(num_samples, 1, 1)
            uv = self.uv.repeat(num_samples, 1, 1)

            if cam2world is not None:
                cam2world = cam2world.repeat(num_samples, 1, 1)
            else:
                cam2world = _get_random_poses(
                    self.sample_radius, num_samples, self.sample_mode)
                cam2world = torch.from_numpy(cam2world).float()

            predictions, _ = self.model(
                cam2world, z_interp, intrinsics, uv)
            predictions = predictions.view(num_samples, 128, 128, -1)
            predictions = F.interpolate(
                predictions.permute(0,3,1,2), 
                size=(self.img_size, self.img_size), 
                mode='bilinear', 
                align_corners=True).permute(0,2,3,1).view(1,-1,predictions.shape[-1])

            N, H, W = num_samples, self.img_size, self.img_size

            if return_feat:
                pred = F.log_softmax(predictions, dim=2).permute(0, 2, 1).view(
                    N, -1, H, W).cpu().numpy()
            else:
                pred = torch.argmax(predictions, dim=2, keepdim=True).permute(0, 2, 1).view(
                    N, 1, H, W).cpu().numpy()
        
        return pred


    def sample_pose(self, cam_center, look_at=None, num_samples=25, emb=None, return_feat=False):
        """ Sample num_samples random cameras for one instance
        """
        with torch.no_grad():
            if emb is None:
                z_src = torch.from_numpy(self.gmm.sample(1)[0]).float()
                z_trgt = torch.from_numpy(self.gmm.sample(1)[0]).float()
                weight = torch.rand((1,)).to(z_src.device)

                emb = z_src * weight + z_trgt * (1.0 - weight)

            cam2world = _get_random_poses(
                self.sample_radius, num_samples, self.sample_mode,
                cam_center=cam_center, look_at=look_at)
            cam2world = torch.from_numpy(cam2world).float()

            emb = emb.repeat(num_samples, 1)
            intrinsics = self.intrinsics.repeat(num_samples, 1, 1)
            uv = self.uv.repeat(num_samples, 1, 1)

            predictions, _ = self.model(
                cam2world, emb, intrinsics, uv)
            predictions = predictions.view(num_samples, 128, 128, -1)
            predictions = F.interpolate(
                predictions.permute(0,3,1,2), 
                size=(self.img_size, self.img_size), 
                mode='bilinear', 
                align_corners=True).permute(0,2,3,1).view(1,-1,predictions.shape[-1])

            N, H, W = num_samples, self.img_size, self.img_size

            if return_feat:
                pred = F.log_softmax(predictions, dim=2).permute(0, 2, 1).view(
                    N, -1, H, W).cpu().numpy()
            else:
                pred = torch.argmax(predictions, dim=2, keepdim=True).permute(0, 2, 1).view(
                    N, 1, H, W).cpu().numpy()
                    
            return pred


    def sample_ins_fix_pose(self, num_samples=25, return_feat=False):
        """ Sample num_samples instances with fix camera pose
        """
        cam2world = _get_random_poses(
            self.sample_radius, 1, self.sample_mode)
        cam2world = torch.from_numpy(cam2world).float()

        return self.sample_ins(num_samples, cam2world, return_feat=return_feat)


if __name__ == "__main__":
    _NUM_SAMPLES = 100

    device = torch.cuda.current_device()

    # ONLY init model once for all samples.
    # The model is quite big...
    print(torch.cuda.memory_allocated(device))
    sampler = FaceSegSampler()  
    print('DONE: init model.')
    print(torch.cuda.memory_allocated(device))

    # sampling
    smp_ins = sampler.sample_ins(_NUM_SAMPLES)
    print('Sampling ', smp_ins.shape)
    print(torch.cuda.memory_allocated(device))

    look_at = np.asarray([0, 0.1, 0.0])
    cam_center =  np.asarray([0, 0.1, 4.5])
    smp_pose = sampler.sample_pose(cam_center, look_at, _NUM_SAMPLES)

    smp_pose = sampler.sample_ins_fix_pose(_NUM_SAMPLES)
