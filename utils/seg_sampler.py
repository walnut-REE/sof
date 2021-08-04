import sys
import os
import configargparse

SOF_ROOT_DIR = 'sofgan_test/modules/SOF'
sys.path.append(SOF_ROOT_DIR)

import torch
import torch.nn
import torch.nn.functional as F
import numpy as np

from modeling import SOFModel
from glob import glob


_DEFAULT_MODEL_PATH = os.path.join(
    SOF_ROOT_DIR, 'checkpoints/epoch_0018_iter_070000.pth')
_DEFAULT_INT = os.path.join(
    SOF_ROOT_DIR, 'data/intrinsics.txt')


def _campos2matrix(cam_pos, cam_center=None, cam_up=None):
    _cam_target = np.asarray([0,0.11,0.1]).reshape((1, 3)) if cam_center is None else cam_center

    _cam_up = np.asarray([0.0, 1.0, 0.0]) if cam_up is None else cam_up

    cam_dir = (_cam_target-cam_pos)
    cam_dir = cam_dir / np.linalg.norm(cam_dir)
    cam_right = np.cross(cam_dir, _cam_up)
    cam_right = cam_right / np.linalg.norm(cam_right)
    cam_up = np.cross(cam_right, cam_dir)

    cam_R = np.concatenate([cam_right.T, -cam_up.T, cam_dir.T], axis=1)

    cam_P = np.eye(4)
    cam_P[:3, :3] = cam_R
    cam_P[:3, 3] = cam_pos

    return cam_P


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
                        sample_range=None):

    cam2world = []

    theta = []
    _THETA_RANGE = sample_range if sample_range is not None else [0.0, -0.55, 0.55, 0.0]

    for i in range(len(_THETA_RANGE)-1):
        theta.append(np.linspace(
            _THETA_RANGE[i],_THETA_RANGE[i+1], num=num_samples))

    theta = np.concatenate(theta)
    x = R*np.sin(theta)
    y = np.zeros_like(x)
    z = R*np.cos(theta)
    cam_T = np.stack([x,y,z],axis=1) + cam_center.reshape((1,3))

    vis_outputs,out_segs = [],[]
    for i in range(len(theta)):
        cam_pose = _campos2matrix(cam_T[i], cam_center)     
        cam2world.append(cam_pose)   

    # SPPIRAL PATH
    t = np.linspace(0, 4*np.pi, num_samples*4, endpoint=True)
    for k in range(len(t)):
        cam_T = np.array([np.cos(t[k]), -np.sin(t[k]), -np.sin(0.5*t[k])]) * R
        cam_T = cam_T[[1,2,0]] + cam_center
        cam_pose = _campos2matrix(cam_T, cam_center)

        cam2world.append(cam_pose)

    cam2world = np.asarray(cam2world)
    return cam2world


def _get_random_poses(sample_radius, num_samples, mode, cam_center=None, cam_pos=None, sample_range=None):
    if cam_pos is None:
        if mode == 'sphere':
            cam_pos = _rand_cam_sphere(
                sample_radius, num_samples, cam_center=cam_center, sample_range=sample_range)            
        elif mode == 'plane':
            return _rand_cam_plane(
                sample_radius, num_samples, cam_center=cam_center, sample_range=sample_range)
        elif mode == 'spiral':
            return _rand_cam_spiral(
                sample_radius, num_samples, cam_center=cam_center, sample_range=sample_range)
        elif mode == 'uniform':
            return _rand_cam_uniform(
                sample_radius, num_samples, cam_center=cam_center, sample_range=sample_range)
        else:
            raise ValueError('Unsupported camera path: %s, \
                must be one in [sphere, plane, spiral, uniform].'%(mode))
    
    assert cam_pos is not None, 'Campose not specified'
    cam2world = []

    for i in range(num_samples):
        cam2world.append(_campos2matrix(cam_pos[i], np.asarray(cam_center).reshape((1, 3))))

    cam2world = np.asarray(cam2world)

    return cam2world


def load_model(model, path, discriminator=None, overwrite_embeddings=False, overwrite_renderer=False, optimizer=None):
    if os.path.isdir(path):
        checkpoint_path = sorted(glob(os.path.join(path, "*.pth")))[-1]
    else:
        checkpoint_path = path

    whole_dict = torch.load(checkpoint_path)

    if overwrite_embeddings:
        del whole_dict['model']['latent_codes.weight']

    if overwrite_renderer:
        keys_to_remove = [
            key for key in whole_dict['model'].keys() if 'rendering_net' in key]
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


class FaceSegSampler():
    def __init__(   self,
                    model_path=_DEFAULT_MODEL_PATH,
                    img_size=64,
                    num_instances=1494,
                    num_poses=25,
                    sample_mode='sphere',
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
                               fit_single_SOF=False,
                               use_unet_renderer=False,
                               tracing_steps=10,
                               freeze_networks=True,
                               out_channels=19,
                               img_sidelength=128,
                               output_sidelength=self.img_size,
                               )

        print('DONE Load model.')
        print(torch.cuda.memory_allocated(torch.cuda.current_device()))

        load_model(self.model,
                   path=self.model_path,
                   discriminator=None,
                   overwrite_embeddings=False)
        
        print('DONE Load ckpt.')
        print(torch.cuda.memory_allocated(torch.cuda.current_device()))

        self.model.eval()
        self.model.cuda()

        self.intrinsics = torch.Tensor(_parse_intrinsics(
            _DEFAULT_INT, 128)).float()
        
        self.uv = np.mgrid[0:128, 0:128].astype(np.int32)
        self.uv = torch.from_numpy(np.flip(self.uv, axis=0).copy()).long()
        self.uv = self.uv.reshape(2, -1).transpose(1, 0)
        

    def sample_ins(self, num_samples=100, cam2world=None, return_feat=False):
        """ Sample num_samples random cameras for random instance. 
            If cam2world is given, sample instances with the given pose.
        """
        with torch.no_grad():
            src_idx, trgt_idx = torch.randint(
                0, self.num_instances, (2, num_samples)).squeeze().cuda()

            src_emb = self.model.latent_codes(src_idx)
            trgt_emb = self.model.latent_codes(trgt_idx)

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

            N, H, W = num_samples, self.img_size, self.img_size

            if return_feat:
                pred = F.log_softmax(predictions, dim=2).permute(0, 2, 1).view(
                    N, -1, H, W).cpu().numpy()
            else:
                pred = torch.argmax(predictions, dim=2, keepdim=True).permute(0, 2, 1).view(
                    N, 1, H, W).cpu().numpy()
        
        return pred


    def sample_pose(self, num_samples=25, emb=None, return_feat=False):
        """ Sample num_samples random cameras for one instance
        """
        with torch.no_grad():
            if emb is None:
                src_idx, trgt_idx = torch.randint(
                    0, self.num_instances, (2,)).squeeze().cuda()

                z_src = self.model.latent_codes(src_idx)
                z_trgt = self.model.latent_codes(trgt_idx)

                weight = torch.rand((1,)).to(z_src.device)

                emb = z_src * weight + z_trgt * (1.0 - weight)
            
            cam2world = _get_random_poses(self.sample_radius, num_samples, self.sample_mode)
            cam2world = torch.from_numpy(cam2world).float()

            emb = emb.unsqueeze(0).repeat(num_samples, 1)
            intrinsics = self.intrinsics.repeat(num_samples, 1, 1)
            uv = self.uv.repeat(num_samples, 1, 1)

            predictions, _ = self.model(
                cam2world, emb, intrinsics, uv)

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
    smp_pose = sampler.sample_pose(_NUM_SAMPLES)
    smp_pose = sampler.sample_ins_fix_pose(_NUM_SAMPLES)
