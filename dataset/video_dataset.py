import os
from pathlib import Path
import torch
import numpy as np
from glob import glob

import cv2
from dataset import data_util

import utils.common as util

_COLOR_MAP = np.asarray([[0, 0, 0], [204, 0, 0], [76, 153, 0], [204, 204, 0], [51, 51, 255], [204, 0, 204], [0, 255, 255], [255, 204, 204], [102, 51, 0], [
                        255, 0, 0], [102, 204, 0], [255, 255, 0], [0, 0, 153], [0, 0, 204], [255, 51, 153], [0, 204, 204], [0, 51, 0], [255, 153, 51], [0, 204, 0], [0, 204, 153]])
_NUM_CLASSES = 20


def load_calib(data_dir, cam2world='cam2world.npy', intrinsics='intrinsics.txt', img_sidelength=128):
    cams = None

    if cam2world is not None:
        print('> load cam2world from ', cam2world)
        cam2world = np.load(os.path.join(data_dir, cam2world))
    else:
        cam_fp = os.path.join(data_dir, 'cams.npy')
        cams = np.load(cam_fp, allow_pickle=True)

        cam2world = np.asarray(
            [np.linalg.inv(cam['extrinsic']) for cam in cams], dtype=np.float32)
        cam2world[:, :3, 3] /= 100.0
    
    def _scale2intrinsic(scale, img_size=(512, 512)):
            fx, fy = scale
            H, W = img_size
            cx = W // 2
            cy = H // 2
            return np.array([[fx, 0., cx], [0., fy, cy], [0., 0., 1.]])

    if intrinsics is not None:
        intrinsics = np.expand_dims(data_util.parse_intrinsics(
            os.path.join(data_dir, intrinsics),
            trgt_sidelength=img_sidelength), axis=0)
    else:
        intrinsics = np.asarray([_scale2intrinsic(cam['scale'], (img_sidelength, img_sidelength)) for cam in cams])

    return cam2world, intrinsics


def _load_video_frame(img_fp, sidelength=None):
    img = cv2.imread(img_fp, cv2.IMREAD_UNCHANGED).astype(np.float32)

    _, W, _ = img.shape
    seg = img[:, W//2:, 0]

    if sidelength is not None:
        seg = cv2.resize(seg, (sidelength, sidelength),
                         interpolation=cv2.INTER_NEAREST)

    seg = seg.reshape(1, -1).transpose(1, 0)

    return seg


class FaceFrameDataset():
    """This creates a dataset class for a single object instance (such as a single car)."""

    def __init__(self,
                 instance_idx,
                 instance_path,
                 data_type='seg',
                 fix_camera = True,
                 img_sidelength=None,
                 sample_observations=None,
                 shuffle=False):

        self.data_root = instance_path
        self.instance_idx = instance_idx
        self.img_sidelength = img_sidelength
        self.data_type = data_type
        self.fix_camera = fix_camera

        self.poses, self.intrinsics = load_calib(
            self.data_root, img_sidelength = self.img_sidelength)

        print('Data root = ', self.data_root)
        self.color_paths = sorted(
            glob(os.path.join(self.data_root, '%s_*.png' % (data_type))))

        if shuffle:
            idxs = np.random.permutation(len(self.color_paths))
            if hasattr(self, 'color_paths'): self.color_paths = [self.color_paths[x] for x in idxs]
            if not self.fix_camera:
                self.poses = self.poses[idxs]
            if not self.fix_camera:
                self.intrinsics = self.intrinsics[idxs]

        if sample_observations is not None:
            sample_observations = [x for x in sample_observations if x < len(self.color_paths)]
            if hasattr(self, 'color_paths'): 
                self.color_paths = [
                    self.color_paths[idx] for idx in sample_observations]
            if not self.fix_camera: self.poses = self.poses[sample_observations, :, :]
            if not self.fix_camera:
                self.intrinsics = self.intrinsics[sample_observations, :, :]

        print('> [DONE] Init instance #%04d with %02d observations.'%(self.instance_idx, self.poses.shape[0]))

    def set_img_sidelength(self, new_img_sidelength):
        """For multi-resolution training: Updates the image sidelength with whichimages are loaded."""
        self.img_sidelength = new_img_sidelength        

    def __len__(self):
        return len(self.color_paths)

    def __getitem__(self, idx):
        
        intrinsics = self.intrinsics[0] if self.fix_camera else self.intrinsics[idx]
        pose = self.poses[0] if self.fix_camera else self.poses[idx]

        uv = np.mgrid[0:self.img_sidelength, 0:self.img_sidelength].astype(np.int32)
        uv = torch.from_numpy(np.flip(uv, axis=0).copy()).long()
        uv = uv.reshape(2, -1).transpose(1, 0)

        sample = {
            'instance_idx': torch.Tensor([self.instance_idx]).squeeze(),
            'observation_idx': torch.Tensor([idx]).squeeze(),
            'pose': torch.from_numpy(pose).float(),
            'uv': uv,
            'intrinsics': torch.Tensor(intrinsics).float(),
        }

        if hasattr(self, 'color_paths'):
            img_fp = self.color_paths[idx]
            if self.data_type == 'seg':
                seg_img = _load_video_frame(
                    img_fp, sidelength=self.img_sidelength)
                if np.max(seg_img) > _NUM_CLASSES:
                    seg_img /= 10.0
                sample['rgb'] = torch.from_numpy(seg_img).float()
            
            elif self.data_type == 'render':
                render_img = data_util.load_rgb(img_fp, sidelength=self.img_sidelength)
                render_img = render_img.reshape(3, -1).transpose(1, 0)
                sample['rgb'] = torch.from_numpy(render_img).float()

        return sample


class FaceVideoDataset(torch.utils.data.Dataset):
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
                 fix_camera=True,
                 img_sidelength=None,
                 sample_instances=None,
                 sample_observations=None):

        tot_instances = sorted(glob(os.path.join(root_dir, '[0-9]*/')))
        
        if sample_instances is not None:
            tot_instances=[tot_instances[idx] for idx in sample_instances if idx < len(tot_instances)]

        self.num_instances = len(tot_instances)

        self.all_instances = [FaceFrameDataset(     instance_idx=instance_idx,
                                                    instance_path=instance_fp,
                                                    data_type=data_type,
                                                    fix_camera=fix_camera,
                                                    img_sidelength=img_sidelength,
                                                    sample_observations=sample_observations)
                              for instance_idx, instance_fp in enumerate(tot_instances)]

        assert len(self.all_instances) == self.num_instances

        self.num_per_instance_observations =  [len(obj) for obj in self.all_instances]
        self.num_instances = len(self.all_instances)

        if data_type[:3] == 'seg':
            self.color_map = torch.tensor(_COLOR_MAP, dtype=torch.float32) / 255.0
        else:
            self.color_map = None

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

        observations = []
        observations.append(self.all_instances[obj_idx][rel_idx])

        ground_truth = [{'rgb':ray_bundle['rgb']} for ray_bundle in observations]

        return observations, ground_truth
