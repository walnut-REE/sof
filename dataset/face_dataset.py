import os
from pathlib import Path
import torch
import numpy as np
from glob import glob
import random

import cv2
from dataset import data_util

import utils.common as util


_COLOR_MAP = np.asarray([[0, 0, 0], [204, 0, 0], [76, 153, 0], [204, 204, 0], 
                        [51, 51, 255], [204, 0, 204], [0, 255, 255], [255, 204, 204], 
                        [102, 51, 0], [255, 0, 0], [102, 204, 0], [255, 255, 0], 
                        [0, 0, 153], [0, 0, 204], [255, 51, 153], [0, 204, 204], 
                        [0, 51, 0], [255, 153, 51], [0, 204, 0], [0, 204, 153]])

_NUM_CLASSES = 20

def _id_remapping(seg):
    """ Map CelebAMask classes to required classes.
        label_cls = [
            'background', 'skin', 'l_brow', 'r_brow', 
            'l_eye', 'r_eye', 'eye_g', 'l_ear', 'r_ear', 'ear_r',
            'nose', 'mouth', 'u_lip', 'l_lip', 'neck', 'neck_l', 
            'cloth', 'hair', 'hat']
    """
    remap_list = np.array([0,1,2,2,3,3,4,5,6,7,8,9,9,10,11,12,13,14,15,16])
    return remap_list[seg.astype(np.uint8)]
    

def _campos2matrix(cam_pos, cam_center=None, cam_up=None):
    _cam_target = np.asarray([0,0.11,0.1]) if cam_center is None else cam_center
    _cam_target = _cam_target.reshape((1, 3))

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


def _rand_cam_spiral(R=1.0, num_samples=100):
    _v = 1.0
    _omega = 20

    t = np.linspace(0, R * 2, num=num_samples)
    z = -R + t * _v

    r = np.sqrt(R**2-z**2)
    x = r * np.cos(_omega * t)
    y = r * np.sin(_omega * t)
    cam_pos = (np.stack([x, z, y])).T

    return cam_pos

def _rand_cam_cube(Rx=0.5, Ry=0.5, Rz=1.0, num_samples=15):
    x = (np.random.rand(num_samples, 1) * 2.0 - 1.0) * Rx
    y = (np.random.rand(num_samples, 1) * 2.0 - 1.0) * Ry
    z = -(np.random.rand(num_samples, 1)) * Rz

    cam_pos = np.concatenate([x, z, y], axis=1)
    return cam_pos

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
                    cam_center=[0, 0.11, 0.1], 
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
                np.array([x, y, R]+cam_center), (np.array([x, y, R-1.0])+np.array(cam_center)).reshape((1, 3))))
        _p *= -1
        _idx = _cur_idx

    cam2world = np.asarray(cam2world)
    
    return cam2world


def _get_random_poses(sample_radius, num_samples, mode, cam_center=None, cam_pos=None, sample_range=None):
    if cam_pos is None:
        if mode == 'spiral':
            cam_pos = _rand_cam_spiral(sample_radius, num_samples)
        elif mode == 'sphere':
            cam_pos = _rand_cam_sphere(sample_radius, num_samples, cam_center=cam_center, sample_range=sample_range)
        elif mode == 'load':
            _DEFAULT_POSE = '/data/anpei/facial-data/seg_face_syn/779/cam2world.npy'
            cam_pos = np.load(_DEFAULT_POSE)
            cam_pos = cam_pos[:num_samples, :3, 3].squeeze()
        elif mode == 'plane':
            return _rand_cam_plane(sample_radius, num_samples, cam_center=cam_center, sample_range=sample_range)
        else:
            cam_pos = _rand_cam_cube(num_samples=num_samples)
    
    assert cam_pos is not None, 'Campose not specified'
    cam2world = []

    for i in range(num_samples):
        cam2world.append(_campos2matrix(cam_pos[i], np.asarray(cam_center).reshape((1, 3))))

    cam2world = np.asarray(cam2world)

    return cam2world
    

class FaceInstanceDataset():
    """This creates a dataset class for a single object instance (such as a single car)."""

    def __init__(self,
                 instance_idx,
                 cam2world,
                 intrinsics,
                 param='params.npy',
                 data_type='seg',
                 instance_path=None,
                 load_depth=False,
                 img_sidelength=None,
                 sample_observations=None,
                 remap_fn=None,
                 shuffle=True):

        self.data_root = os.path.dirname(instance_path)
        self.data_id = os.path.basename(instance_path).split('.')[0]

        self.instance_idx = instance_idx
        self.img_sidelength = img_sidelength
        self.data_type = data_type
        self.load_depth = load_depth

        self.remap_fn = remap_fn
        self.color_paths = sorted(glob(instance_path))

        if isinstance(cam2world, str):
            try:
                self.poses = np.load(cam2world)
            except ValueError:
                self.poses = np.load(cam2world, allow_pickle=True).item()

        elif isinstance(cam2world, (np.ndarray, np.generic)):
            self.poses = cam2world
        else:
            raise ValueError('Invalid camera2world.')

        if isinstance(intrinsics, str):
            try:
                self.intrinsics = np.load(intrinsics)
            except ValueError:
                self.intrinsics = np.load(intrinsics, allow_pickle=True).item()

        elif isinstance(intrinsics, (np.ndarray, np.generic)):
            self.intrinsics = intrinsics
        else:
            raise ValueError('Invalid intrinsics.')

        self.z_range = None
        z_range_fp = os.path.join(self.data_root, 'zRange.npy')
        if load_depth and os.path.exists(z_range_fp):
            self.z_range = np.load(z_range_fp, allow_pickle=True).item()

        if param is not None:
            param_fp = os.path.join(self.data_root, param)
            if os.path.exists(param_fp):
                self.param = np.load(param_fp, allow_pickle=True).item()

        if shuffle:
            idxs = np.random.permutation(len(self.color_paths))
            self.color_paths = [self.color_paths[x] for x in idxs]
            if self.z_range is not None: self.z_range = self.z_range[idxs]

        if sample_observations is not None:
            sample_observations = [x for x in sample_observations if x < len(self.color_paths)]

            self.color_paths = [self.color_paths[idx] for idx in sample_observations]
            if self.z_range is not None: self.z_range = self.z_range[sample_observations]

    def set_img_sidelength(self, new_img_sidelength):
        """For multi-resolution training: Updates the image sidelength with whichimages are loaded."""
        self.img_sidelength = new_img_sidelength        

    def set_intrinsics(self, cam_int):
        self.intrinsics = cam_int

    def __len__(self):
        return len(self.color_paths)
        
    def __getitem__(self, idx):

        pose = self.poses[idx]
        cam_int = self.intrinsics[idx]

        uv = np.mgrid[0:self.img_sidelength, 0:self.img_sidelength].astype(np.int32)
        uv = torch.from_numpy(np.flip(uv, axis=0).copy()).long()
        uv = uv.reshape(2, -1).transpose(1, 0)

        sample = {
            'instance_idx': torch.Tensor([self.instance_idx]).squeeze(),
            'data_id': self.data_id,
            'observation_idx': idx,
            'pose': torch.from_numpy(pose).float(),
            'uv': uv,
        }

        if hasattr(self, 'param'):
            sample['params'] = torch.from_numpy(self.param).float()

        if hasattr(self, 'color_paths'):
            img_fp = self.color_paths[idx].replace('_vis', '')
            if self.data_type == 'seg':
                seg_img, cam_int = data_util.load_seg_map(
                    img_fp, sidelength=self.img_sidelength, 
                    num_classes=_NUM_CLASSES, remap_fn=self.remap_fn,
                    cam_int=cam_int)

                sample['rgb'] = torch.from_numpy(seg_img).float()
                sample['intrinsics'] = torch.from_numpy(cam_int).float()
            
            elif self.data_type == 'rgb':
                render_img, cam_int = data_util.load_rgb(
                    img_fp, sidelength=self.img_sidelength, 
                    cam_int=cam_int)
                render_img = render_img.reshape(3, -1).transpose(1, 0)
                sample['rgb'] = torch.from_numpy(render_img).float()
                sample['intrinsics'] = torch.from_numpy(cam_int).float()


        if self.z_range is not None:
            sample['z_range'] = torch.FloatTensor(self.z_range[idx])

        if hasattr(self, 'load_depth') and self.load_depth:
            depth_path = os.path.join(
                self.data_root,
                os.path.basename(img_fp).replace(self.data_type, 'depth'))

            zRange = self.z_range[idx] if self.z_range is not None else None

            depth_img = data_util.load_depth(
                depth_path, sidelength=self.img_sidelength, zRange=zRange)

            depth_img = depth_img.reshape(1, -1).transpose(1, 0)
            sample['depth'] = torch.from_numpy(depth_img).float()

        return sample


class FaceInstanceRandomPose(FaceInstanceDataset):
    """This creates a dataset class for a single object instance (such as a single car)."""

    def __init__(self,
                 instance_idx,
                 intrinsics,
                 num_observations=15,
                 sample_radius=2.0,
                 img_sidelength=128,
                 cam_center=None,
                 mode='cube'):

        self.instance_idx = instance_idx
        self.intrinsics = intrinsics
        self.img_sidelength = img_sidelength
        
        self.z_range = None

        self.poses = _get_random_poses(
            sample_radius, num_samples=num_observations, cam_center=cam_center, mode=mode)
            
        # transform to desired dictionary format
        self.color_paths = list(range(self.poses.shape[0]))

        self.poses = [
            {self.color_paths[idx], self.poses[idx]} 
            for idx in range(len(self.color_paths))]

        self.intrinsics = [
            {self.color_paths[idx], intrinsics} 
            for idx in range(len(self.color_paths))]

        self.load_depth = False


class FaceClassDataset(torch.utils.data.Dataset):
    """Dataset for 1000 face segs with 25 views each, where each datapoint is a FaceInstanceDataset.
    
    Face Labels :

    0: 'background'	1: 'skin'	2: 'l_nose'
    3: 'eye_g'	4: 'l_eye'	5: 'r_eye'
    6: 'l_brow'	7: 'r_brow'	8: 'l_ear'
    9: 'r_ear'	10: 'mouth'	11: 'u_lip'
    12: 'l_lip'	13: 'hair'	14: 'hat'
    15: 'ear_r'	16: 'neck_l'	17: 'neck'
    18: 'cloth'	19: r_nose
    
    """

    def __init__(self,
                 root_dir,
                 ckpt_dir='',
                 data_type='seg',
                 cam2world_fp='cam2world.npy',
                 intrinsics_fp='intrinsics.npy',
                 img_sidelength=None,
                 sample_instances=None,
                 sample_observations=None,
                 load_depth=False):
        
        tot_instances = sorted(glob(os.path.join(root_dir, '*/')))

        if ckpt_dir:
            idx_fp = os.path.join(ckpt_dir, 'indexing.txt')
            if os.path.exists(idx_fp):
                print('> Load indexing file from: ', idx_fp)
                with open(idx_fp, 'r') as f: tot_instances = [x.strip() for x in f.readlines()]
            else:  
                print('> Save indexing file to: ', idx_fp)          
                with open(idx_fp, 'w') as f: f.writelines('\n'.join(tot_instances))

        if sample_instances is not None:
            if isinstance(sample_instances, int): 
                tot_instances = random.choices(
                    tot_instances, k=min(sample_instances, len(tot_instances)))
            elif isinstance(sample_instances, list):
                tot_instances=[tot_instances[idx] for idx in sample_instances \
                    if idx < len(tot_instances)]
            else:
                raise ValueError('Invalid sample_instances, should be int or list.')

        self.num_instances = len(tot_instances)

        img_fp = data_type + '_vis_[0-9]*.png'
        
        tot_imgs = [os.path.join(instance_dir, img_fp) for instance_dir in tot_instances]
        cam2world = [os.path.join(instance_dir, cam2world_fp) for instance_dir in tot_instances]
        intrinsics = [os.path.join(instance_dir, intrinsics_fp) for instance_dir in tot_instances]

        self.all_instances = [FaceInstanceDataset(  instance_idx=instance_idx,
                                                    instance_path=instance_fp,
                                                    data_type=data_type,
                                                    cam2world=cam2world[instance_idx],
                                                    intrinsics=intrinsics[instance_idx],
                                                    img_sidelength=img_sidelength,
                                                    sample_observations=sample_observations,
                                                    load_depth=load_depth)
                              for instance_idx, instance_fp in enumerate(tot_imgs)]

        assert len(self.all_instances) == self.num_instances

        self.num_per_instance_observations =  [len(obj) for obj in self.all_instances]
        self.num_instances = len(self.all_instances)
        self.load_depth = load_depth

        if data_type[:3] == 'seg':
            self.color_map = torch.tensor(_COLOR_MAP, dtype=torch.float32) / 255.0
        else:
            self.color_map = None

        print('> Load %d instances from %s.'%(self.num_instances, root_dir))

    def set_img_sidelength(self, new_img_sidelength):
        """For multi-resolution training: Updates the image sidelength with whichimages are loaded."""
        for instance in self.all_instances:
            instance.set_img_sidelength(new_img_sidelength)

    def set_cam_int(self, new_cam_int, ins_idx=None):
        ins_idx = list(range(len(self.all_instances))) if ins_idx is None else ins_idx
        for idx in ins_idx:
            self.all_instances[idx].set_cam_int(new_cam_int)

    def __len__(self):
        return np.sum(self.num_per_instance_observations).astype(np.int)

    def get_instance_idx(self, idx):
        """Maps an index into all tuples of all objects to the idx of the tuple relative to the other tuples of that
        object
        """
        obj_idx = 0
        while idx >= 0:
            idx -= self.num_per_instance_observations[obj_idx]
            obj_idx += 1
        return obj_idx - 1, int(idx + self.num_per_instance_observations[obj_idx - 1])

    def collate_fn(self, batch_list):
        batch_list = zip(*batch_list)

        all_parsed = []
        for entry in batch_list:
            # make them all into a new dict
            ret = {}
            for k in entry[0][0].keys():
                ret[k] = []
            # flatten the list of list
            for b in entry:
                for k in entry[0][0].keys():
                    ret[k].extend( [bi[k] for bi in b])
            for k in ret.keys():
                if type(ret[k][0]) == torch.Tensor:
                   ret[k] = torch.stack(ret[k])
            all_parsed.append(ret)

        return tuple(all_parsed)

    def __getitem__(self, idx):
        """Each __getitem__ call yields a list of self.samples_per_instance observations of a single scene (each a dict),
        as well as a list of ground-truths for each observation (also a dict)."""
        obj_idx, rel_idx = self.get_instance_idx(idx)

        observations = [self.all_instances[obj_idx][rel_idx]]

        if hasattr(self, 'load_depth') and self.load_depth:
            ground_truth = [{
                'rgb':ray_bundle['rgb'] if 'rgb' in ray_bundle else None,
                'depth':ray_bundle['depth']} for ray_bundle in observations]

        else:
            ground_truth = [
                {'rgb':ray_bundle['rgb'] if 'rgb' in ray_bundle else None} 
                for ray_bundle in observations]

        return observations, ground_truth


class FaceRandomPoseDataset(FaceClassDataset):

    def __init__(self,
                 intrinsics=None,
                 sample_radius=1.1,
                 img_sidelength=128,
                 num_instances=100,
                 num_observations=15,
                 cam_center=None,
                 mode='load'
                 ):

        _DEFAULT_INT = intrinsics if intrinsics is not None else '/data/anpei/facial-data/seg_face_0417/intrinsics.txt'
        print('Load cam_int from: ', _DEFAULT_INT)
        # print('Camera center: ', cam_center)

        if isinstance(num_instances, int):
            num_instances = list(range(num_instances))

        intrinsics = data_util.parse_intrinsics(_DEFAULT_INT, trgt_sidelength=img_sidelength)

        self.all_instances = [FaceInstanceRandomPose(   instance_idx=idx,
                                                        intrinsics=intrinsics,
                                                        num_observations=num_observations,
                                                        sample_radius=sample_radius,
                                                        img_sidelength=128,
                                                        cam_center=cam_center,
                                                        mode=mode)

                                for idx in num_instances]

        self.color_map = torch.tensor(_COLOR_MAP, dtype=torch.float32) / 255.0
        self.num_per_instance_observations = [len(obj) for obj in self.all_instances]
        self.load_depth = False


class CelebAMaskDataset(FaceClassDataset):

    def __init__(self,
                root_dir,
                ckpt_dir=None,
                data_type='seg',
                cam2world_fp='cam2world.npy',
                intrinsics_fp='intrinsics.npy',
                img_sidelength=None,
                sample_instances=None,
                sample_observations=None,
                load_depth=False):
        _ = data_type, cam2world_fp, intrinsics_fp, sample_observations, load_depth

        _DEFAULT_INT = np.array([
            [[2000, 0, 256], [0, 2000, 256], [0, 0, 1]]], dtype=np.float32)

        _DEFAULT_CAM = np.array([[[1., 0.,  0.,  0.],
                                [0., 1., 0., 0.],
                                [0., 0., 1., 1.1],
                                [0., 0., 0., 1.]]], dtype=np.float32)

        idx_fp = os.path.join(ckpt_dir, 'indexing.txt')

        if os.path.exists(idx_fp):
            print('> Load indexing file from: ', idx_fp)
            with open(idx_fp, 'r') as f:
                tot_imgs = [x.strip() for x in f.readlines()]

        else:
            tot_imgs = sorted(glob(os.path.join(root_dir, '*.png')))
            if len(tot_imgs) > 0:
                if sample_instances > 0 and sample_instances < len(tot_imgs):
                    tot_imgs = random.choices(tot_imgs, k=sample_instances)
                # save indexing dict
                with open(idx_fp, 'w') as f:
                    f.writelines('\n'.join(tot_imgs))

        print('Load CelebA, num_instances = ', len(tot_imgs))

        self.all_instances = [FaceInstanceDataset(instance_idx=instance_idx,
                                                  instance_path=instance_fp,
                                                  data_type='seg',
                                                  cam2world=_DEFAULT_CAM,
                                                  intrinsics=_DEFAULT_INT,
                                                  img_sidelength=img_sidelength)
                              for instance_idx, instance_fp in enumerate(tot_imgs)]
        
        self.num_instances = len(self.all_instances)

        self.color_map = torch.tensor(_COLOR_MAP, dtype=torch.float32) / 255.0
        self.num_per_instance_observations = [len(obj) for obj in self.all_instances]
        
        print('> Load %d instances from %s.' % (self.num_instances, root_dir))


class FaceRealDataset(FaceClassDataset):

    def __init__(self,
                 root_dir,
                 ckpt_dir='',
                 img_sidelength=128,
                 num_instances=100
                 ):


        all_imgs = glob(os.path.join(root_dir, '*.npy'))
        all_instances = set([x.split('_')[0] for x in all_imgs])

        _DEFAULT_INT = np.array([
            [[1600, 0, 256], [0, 1600, 256], [0, 0, 1]]])

        _DEFAULT_CAM = np.array([[[1., - 0.,  0.,  0.],
                                 [-0., - 1., 0., 0.],
                                 [0., - 0., - 1.,1.1],
                                 [0., 0., 0., 1.]]])

        idx_fp = os.path.join(ckpt_dir, 'indexing.txt')
            
        if ckpt_dir and os.path.exists(idx_fp):
            print('> Load indexing file from: ', idx_fp)
            with open(idx_fp, 'r') as f:
                tot_imgs = [x.strip() for x in f.readlines()]

        else:
            tot_imgs = sorted(glob(os.path.join(root_dir, '*.png')))

            if num_instances > 0 and num_instances < len(tot_imgs):
                tot_imgs = random.choices(tot_imgs, k=num_instances)
             # save indexing dict
            with open(idx_fp, 'w') as f:
                f.writelines('\n'.join(tot_imgs))

        print('Load portraits, num_instances = ', len(tot_imgs))

        if isinstance(num_instances, int):
            num_instances = list(range(num_instances))

        intrinsics = data_util.parse_intrinsics(
            _DEFAULT_INT, trgt_sidelength=img_sidelength)

        cam2world = np.load(_DEFAULT_CAM)

        self.all_instances = [FaceInstanceDataset(instance_idx=instance_idx,
                                                  instance_path=instance_fp,
                                                  data_type='seg',
                                                  cam2world=cam2world,
                                                  intrinsics=intrinsics,
                                                  img_sidelength=img_sidelength)
                              for instance_idx, instance_fp in enumerate(tot_imgs)]
        
        self.num_instances = len(self.all_instances)

        self.color_map = torch.tensor(_COLOR_MAP, dtype=torch.float32) / 255.0
        self.num_per_instance_observations = [len(obj) for obj in self.all_instances]
        
        print('> Load %d instances from %s.' % (self.num_instances, root_dir))
