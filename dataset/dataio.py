import os
import torch
import numpy as np
from glob import glob
import data_util
from ..utils import common as util


def pick(list, item_idcs):
    if not list:
        return list
    return [list[i] for i in item_idcs]


class SceneInstanceDataset():
    """This creates a dataset class for a single object instance (such as a single car)."""

    def __init__(self,
                 instance_idx,
                 instance_dir,
                 specific_observation_idcs=None,  # For few-shot case: Can pick specific observations only
                 img_sidelength=None,
                 num_images=-1):
        self.instance_idx = instance_idx
        self.img_sidelength = img_sidelength
        self.instance_dir = instance_dir

        color_dir = os.path.join(instance_dir, "render_*.png")
        pose_dir = os.path.join(instance_dir, "pose")
        param_dir = os.path.join(instance_dir, "params")

        if not os.path.isdir(color_dir):
            print("Error! root dir %s is wrong" % instance_dir)
            return

        self.has_params = os.path.isdir(param_dir)
        self.color_paths = sorted(data_util.glob_imgs(color_dir))
        self.pose_paths = sorted(glob(os.path.join(pose_dir, "*.txt")))

        if self.has_params:
            self.param_paths = sorted(glob(os.path.join(param_dir, "*.txt")))
        else:
            self.param_paths = []

        if specific_observation_idcs is not None:
            self.color_paths = pick(self.color_paths, specific_observation_idcs)
            self.pose_paths = pick(self.pose_paths, specific_observation_idcs)
            self.param_paths = pick(self.param_paths, specific_observation_idcs)
        elif num_images != -1:
            idcs = np.linspace(0, stop=len(self.color_paths), num=num_images, endpoint=False, dtype=int)
            self.color_paths = pick(self.color_paths, idcs)
            self.pose_paths = pick(self.pose_paths, idcs)
            self.param_paths = pick(self.param_paths, idcs)

    def set_img_sidelength(self, new_img_sidelength):
        """For multi-resolution training: Updates the image sidelength with whichimages are loaded."""
        self.img_sidelength = new_img_sidelength

    def __len__(self):
        return len(self.pose_paths)

    def __getitem__(self, idx):
        intrinsics = data_util.parse_intrinsics(os.path.join(self.instance_dir, "intrinsics.txt"),
                                                                  trgt_sidelength=self.img_sidelength)
        intrinsics = torch.Tensor(intrinsics).float()

        rgb = data_util.load_rgb(self.color_paths[idx], sidelength=self.img_sidelength)
        rgb = rgb.reshape(3, -1).transpose(1, 0)

        pose = data_util.load_pose(self.pose_paths[idx])

        if self.has_params:
            params = data_util.load_params(self.param_paths[idx])
        else:
            params = np.array([0])

        uv = np.mgrid[0:self.img_sidelength, 0:self.img_sidelength].astype(np.int32)
        uv = torch.from_numpy(np.flip(uv, axis=0).copy()).long()
        uv = uv.reshape(2, -1).transpose(1, 0)

        sample = {
            "instance_idx": torch.Tensor([self.instance_idx]).squeeze(),
            "rgb": torch.from_numpy(rgb).float(),
            "pose": torch.from_numpy(pose).float(),
            "uv": uv,
            "param": torch.from_numpy(params).float(),
            "intrinsics": intrinsics
        }
        return sample


class SceneClassDataset(torch.utils.data.Dataset):
    """Dataset for a class of objects, where each datapoint is a SceneInstanceDataset."""

    def __init__(self,
                 root_dir,
                 img_sidelength=None,
                 max_num_instances=-1,
                 max_observations_per_instance=-1,
                 specific_observation_idcs=None,  # For few-shot case: Can pick specific observations only
                 samples_per_instance=2):

        self.samples_per_instance = samples_per_instance
        self.instance_dirs = sorted(glob(os.path.join(root_dir, "*/")))

        assert (len(self.instance_dirs) != 0), "No objects in the data directory"

        if max_num_instances != -1:
            self.instance_dirs = self.instance_dirs[:max_num_instances]

        self.all_instances = [SceneInstanceDataset(instance_idx=idx,
                                                   instance_dir=dir,
                                                   specific_observation_idcs=specific_observation_idcs,
                                                   img_sidelength=img_sidelength,
                                                   num_images=max_observations_per_instance)
                              for idx, dir in enumerate(self.instance_dirs)]

        self.num_per_instance_observations = [len(obj) for obj in self.all_instances]
        self.num_instances = len(self.all_instances)

    def set_img_sidelength(self, new_img_sidelength):
        """For multi-resolution training: Updates the image sidelength with whichimages are loaded."""
        for instance in self.all_instances:
            instance.set_img_sidelength(new_img_sidelength)

    def __len__(self):
        return np.sum(self.num_per_instance_observations)

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

        observations = []
        observations.append(self.all_instances[obj_idx][rel_idx])

        for i in range(self.samples_per_instance - 1):
            observations.append(self.all_instances[obj_idx][np.random.randint(len(self.all_instances[obj_idx]))])

        ground_truth = [{'rgb':ray_bundle['rgb']} for ray_bundle in observations]

        return observations, ground_truth


def _campos2matrix(cam_pos, cam_center=None, cam_up=None):
    _cam_target = np.asarray([0,0.11,0.1]).reshape((1, 3)) if cam_center is None else cam_center
    _cam_up = np.asarray([0.0, 1.0, 0.0]) if cam_up is None else cam_up

    cam_dir = (_cam_target-cam_pos)
    cam_dir = cam_dir / np.linalg.norm(cam_dir)
    cam_right = np.cross(cam_dir, _cam_up)
    cam_right = cam_right / np.linalg.norm(cam_right)
    cam_up = np.cross(cam_right, cam_dir)

    cam_R = np.concatenate([cam_right, -cam_up, cam_dir], axis=0).T

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

def _rand_cam_sphere(R=2.0, num_samples=15, random=False):
    side_len = np.ceil(np.sqrt(num_samples)).astype(np.uint8)
    cam_pos = []

    _PHI_RANGE = [np.pi/2-0.6, np.pi/2+0.6]
    _THETA_RANGE = [np.pi/2-0.3, np.pi/2+0.3]

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
            
            cam_pos.append(np.array([x, z, y]))
        
        _p *= -1
        _idx = _cur_idx
    
    cam_pos = cam_pos[:num_samples]
    cam_pos = np.asarray(cam_pos)
    
    return cam_pos

def _rand_cam_plane(R=1.2, num_samples=15):
    side_len = np.ceil(np.sqrt(num_samples)).astype(np.uint8)
    cam2world = []

    _X_RANGE = [0.05, -0.05]
    _Y_RANGE = [0.13, 0.12]

    _x = np.linspace(_X_RANGE[0], _X_RANGE[1], side_len)
    _y = np.linspace(_Y_RANGE[0], _Y_RANGE[1], side_len)

    _p = 1
    _idx = 0

    for x in _x:
        for i in range(len(_y)):
            _cur_idx = int(_idx + i*_p)
            y = _y[_cur_idx]
            cam2world.append(_campos2matrix(
                np.array([x, y, R]), np.array([x, y, R-1.0]).reshape((1, 3))))
        _p *= -1
        _idx = _cur_idx

    cam2world = np.asarray(cam2world)
    
    return cam2world

def _get_random_poses(sample_radius, num_samples, mode):    
    if mode == 'spiral':
        cam_pos = _rand_cam_spiral(sample_radius, num_samples)
    elif mode == 'sphere':
        cam_pos = _rand_cam_sphere(sample_radius, num_samples)
    elif mode == 'load':
        _DEFAULT_POSE = '/data/anpei/facial-data/seg_face_8000/cam2world.npy'
        cam_pos = np.load(_DEFAULT_POSE)
        cam_pos = cam_pos[:num_samples, :3, 3].squeeze()
    elif mode == 'plane':
        return _rand_cam_plane(sample_radius, num_samples)
    else:
        cam_pos = _rand_cam_cube(num_samples=num_samples)
    
    cam2world = []

    for i in range(num_samples):
        cam2world.append(_campos2matrix(cam_pos[i]))

    cam2world = np.asarray(cam2world)

    return cam2world

class FaceInstanceRandomPose():
    """This creates a dataset class for a single object instance (such as a single car)."""

    def __init__(self,
                 instance_idx,
                 intrinsics,
                 num_observations=15,
                 sample_radius=2.0,
                 img_sidelength=128,
                 mode='cube'):

        self.instance_idx = instance_idx
        self.num_observations = num_observations
        self.intrinsics = intrinsics
        self.img_sidelength = img_sidelength

        self.poses = _get_random_poses(
            sample_radius, num_samples=num_observations, mode=mode)

    def __len__(self):
        return self.poses.shape[0]

    def set_img_sidelength(self, new_img_sidelength):
        """For multi-resolution training: Updates the image sidelength with whichimages are loaded."""
        self.img_sidelength = new_img_sidelength

    def __getitem__(self, idx):
        intrinsics = torch.Tensor(self.intrinsics).float()

        pose = self.poses[idx]

        uv = np.mgrid[0:self.img_sidelength, 0:self.img_sidelength].astype(np.int32)
        uv = torch.from_numpy(np.flip(uv, axis=0).copy()).long()
        uv = uv.reshape(2, -1).transpose(1, 0)

        sample = {
            "instance_idx": torch.Tensor([self.instance_idx]).squeeze(),
            "observation_idx": idx,
            "rgb": None,
            "pose": torch.from_numpy(pose).float(),
            "uv": uv,
            "intrinsics": intrinsics,
        }
        return sample

class FaceRandomPoseDataset(torch.utils.data.Dataset):
    """Dataset for 1000 face segs with 25 views each, where each datapoint is a FaceInstanceDataset.
    
    Face Labels :

    0: 'background'	1: 'skin'	2: 'nose'
    3: 'eye_g'	4: 'l_eye'	5: 'r_eye'
    6: 'l_brow'	7: 'r_brow'	8: 'l_ear'
    9: 'r_ear'	10: 'mouth'	11: 'u_lip'
    12: 'l_lip'	13: 'hair'	14: 'hat'
    15: 'ear_r'	16: 'neck_l'	17: 'neck'
    18: 'cloth'	
    
    """

    def __init__(self,
                 sample_radius=1.1,
                 img_sidelength=128,
                 num_instances=100,
                 num_observations=15,
                 mode='load'
                 ):

        _DEFAULT_INT = '/data/anpei/facial-data/seg_face_8000/intrinsics.txt'

        if isinstance(num_instances, int):
            num_instances = list(range(num_instances))

        intrinsics = data_util.parse_intrinsics(_DEFAULT_INT, trgt_sidelength=img_sidelength)

        self.all_instances = [FaceInstanceRandomPose(   instance_idx=idx,
                                                        intrinsics=intrinsics,
                                                        num_observations=num_observations,
                                                        sample_radius=sample_radius,
                                                        img_sidelength=128,
                                                        mode=mode)

                                for idx in num_instances]

        self.num_per_instance_observations = [len(obj) for obj in self.all_instances]

    def set_img_sidelength(self, new_img_sidelength):
        """For multi-resolution training: Updates the image sidelength with whichimages are loaded."""
        for instance in self.all_instances:
            instance.set_img_sidelength(new_img_sidelength)

    def __len__(self):
        return np.sum(self.num_per_instance_observations)

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

        # print('obj_idx = ', obj_idx, 'rel_idx = ', rel_idx)

        observations = []
        observations.append(self.all_instances[obj_idx][rel_idx])

        ground_truth = [{'rgb':ray_bundle['rgb']} for ray_bundle in observations]

        return observations, ground_truth


class FaceInstanceDataset():
    """This creates a dataset class for a single object instance (such as a single car)."""

    def __init__(self,
                 instance_idx,
                 cam2world,
                 intrinsics,
                 data_type='seg',
                 instance_path=None,
                 load_depth=False,
                 img_sidelength=None,
                 sample_observations=None,
                 shuffle=True):

        self.data_root = os.path.dirname(instance_path)
        self.instance_idx = instance_idx
        self.img_sidelength = img_sidelength
        self.poses = np.load(cam2world) if isinstance(cam2world, str) else cam2world
        self.data_type = data_type
        self.load_depth = load_depth

        self.intrinsics = intrinsics

        if instance_path is not None:
            self.color_paths = sorted(glob(instance_path))
        else:
            self.color_paths = []

        self.z_range = None
        z_range_fp = os.path.join(self.data_root, 'zRange.npy')
        if os.path.exists(z_range_fp):
            self.z_range = np.load(z_range_fp)

        if shuffle:
            idxs = np.random.permutation(len(self.color_paths))
            if self.color_paths: self.color_paths = [self.color_paths[x] for x in idxs]
            self.poses = self.poses[idxs]
            if self.z_range is not None: self.z_range = self.z_range[idxs]

        if sample_observations is not None:
            if self.color_paths: self.color_paths = [self.color_paths[idx] for idx in sample_observations]
            self.poses = self.poses[sample_observations, :, :]
            if self.z_range is not None: self.z_range = self.z_range[sample_observations, :]

        # print('\t > [DONE] Init instance #%04d with %02d observations.'%(self.instance_idx, self.poses.shape[0]))

    def set_img_sidelength(self, new_img_sidelength):
        """For multi-resolution training: Updates the image sidelength with whichimages are loaded."""
        self.img_sidelength = new_img_sidelength

    def __len__(self):
        return self.poses.shape[0]

    def __getitem__(self, idx):
        
        intrinsics = torch.Tensor(self.intrinsics).float()
        pose = self.poses[idx]

        uv = np.mgrid[0:self.img_sidelength, 0:self.img_sidelength].astype(np.int32)
        uv = torch.from_numpy(np.flip(uv, axis=0).copy()).long()
        uv = uv.reshape(2, -1).transpose(1, 0)

        sample = {
            'instance_idx': torch.Tensor([self.instance_idx]).squeeze(),
            'rgb': None,
            'pose': torch.from_numpy(pose).float(),
            'uv': uv,
            'intrinsics': intrinsics,
        }

        img_fp = self.color_paths[idx]

        if self.color_paths:
            if self.data_type == 'seg':
                seg_img = data_util.load_gray(img_fp, sidelength=self.img_sidelength)
                
                if np.max(seg_img) > 19:
                    seg_img /= 10.0
                
                img_channels = seg_img.shape[0]
                seg_img = seg_img.reshape(img_channels, -1).transpose(1, 0)
                sample['rgb'] = torch.from_numpy(seg_img).float()
            elif self.data_type == 'render':
                render_img = data_util.load_rgb(img_fp, sidelength=self.img_sidelength)
                render_img = render_img.reshape(3, -1).transpose(1, 0)
                sample['rgb'] = torch.from_numpy(render_img).float()

        if self.load_depth:
            depth_path = os.path.join(
                self.data_root,
                os.path.basename(img_fp).replace(self.data_type, 'depth'))

            if self.z_range is not None:
                depth_img = data_util.load_depth(
                    depth_path, sidelength=self.img_sidelength, zRange=self.z_range[idx])
            else:
                depth_img = data_util.load_depth(depth_path, sidelength=self.img_sidelength)
            depth_img = depth_img.reshape(1, -1).transpose(1, 0)
            sample['depth'] = torch.from_numpy(depth_img).float()

        return sample


class FaceClassDataset(torch.utils.data.Dataset):
    """Dataset for 1000 face segs with 25 views each, where each datapoint is a FaceInstanceDataset.
    
    Face Labels :

    0: 'background'	1: 'skin'	2: 'nose'
    3: 'eye_g'	4: 'l_eye'	5: 'r_eye'
    6: 'l_brow'	7: 'r_brow'	8: 'l_ear'
    9: 'r_ear'	10: 'mouth'	11: 'u_lip'
    12: 'l_lip'	13: 'hair'	14: 'hat'
    15: 'ear_r'	16: 'neck_l'	17: 'neck'
    18: 'cloth'	
    
    """

    def __init__(self,
                 root_dir,
                 data_type='seg',
                 cam2world_fp='cam2world.npy',
                 intrinsics_fp='intrinsics.txt',
                 img_sidelength=None,
                 sample_instances=None,
                 sample_observations=None,
                 load_depth=False):

        tot_instances = glob(os.path.join(root_dir, '*/'))
        
        if sample_instances is not None:
            tot_instances=[tot_instances[idx] for idx in sample_instances if idx < len(tot_instances)]

        self.num_instances = len(tot_instances)

        img_fp = data_type + '_*.png'

        tot_imgs = [os.path.join(instance_dir, img_fp) for instance_dir in tot_instances]
        cam2world = [os.path.join(instance_dir, cam2world_fp) for instance_dir in tot_instances]
        intrinsics = data_util.parse_intrinsics(
            os.path.join(root_dir, intrinsics_fp), 
            trgt_sidelength=img_sidelength)

        self.all_instances = [FaceInstanceDataset(  instance_idx=instance_idx,
                                                    instance_path=instance_fp,
                                                    data_type=data_type,
                                                    cam2world=cam2world[instance_idx],
                                                    intrinsics=intrinsics,
                                                    img_sidelength=img_sidelength,
                                                    sample_observations=sample_observations,
                                                    load_depth=load_depth)
                              for instance_idx, instance_fp in enumerate(tot_imgs)]

        assert len(self.all_instances) == self.num_instances

        self.num_per_instance_observations =  [len(obj) for obj in self.all_instances]
        self.num_instances = len(self.all_instances)
        self.load_depth = load_depth

        print('> Load %d instances from %s.'%(self.num_instances, root_dir))

    def set_img_sidelength(self, new_img_sidelength):
        """For multi-resolution training: Updates the image sidelength with whichimages are loaded."""
        for instance in self.all_instances:
            instance.set_img_sidelength(new_img_sidelength)

    def __len__(self):
        return np.sum(self.num_per_instance_observations)

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

        # print('obj_idx = ', obj_idx, 'rel_idx = ', rel_idx)

        observations = []
        observations.append(self.all_instances[obj_idx][rel_idx])

        if self.load_depth:
            ground_truth = [{
                'rgb':ray_bundle['rgb'],
                'depth':ray_bundle['depth']} for ray_bundle in observations]

        else:
            ground_truth = [{'rgb':ray_bundle['rgb']} for ray_bundle in observations]
            # print('** ground_truth = ', ground_truth[0]['rgb'].shape)

        return observations, ground_truth