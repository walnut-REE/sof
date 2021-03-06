import torch
import torch.nn as nn
import torch.nn.functional as F

import os, sys
import numbers
import numpy as np

from .common import lin2img

from skimage.measure import marching_cubes

import meshplot as mp
import trimesh

import matplotlib.pyplot as plt
import cv2

import pyrender
os.environ['PYOPENGL_PLATFORM'] = 'egl'


_LABEL = ['background', 'skin', 'l_brow', 'r_brow', 'l_eye', 'r_eye','r_nose', 'l_nose', 'mouth', 'u_lip',
'l_lip', 'l_ear', 'r_ear', 'ear_r', 'eye_g', 'neck', 'neck_l', 'cloth', 'hair', 'hat']


_CMAP = np.asarray([[0, 0, 0], [127, 212, 255], [255, 255, 127], [255, 255, 127], # 'background','skin', 'l_brow', 'r_brow'
                    [255, 255, 170], [255, 255, 170], [240, 157, 240], [255, 212, 255], #'l_eye', 'r_eye', 'r_nose', 'l_nose',
                    [31, 162, 230], [127, 255, 255], [127, 255, 255], #'mouth', 'u_lip', 'l_lip'
                    [0, 255, 85], [0, 255, 85], [0, 255, 170], [255, 255, 170], #'l_ear', 'r_ear', 'ear_r', 'eye_g'
                    [127, 170, 255], [85, 0, 255], [255, 170, 127], #'neck', 'neck_l', 'cloth'
                    [212, 127, 255], [0, 170, 255]#, 'hair', 'hat'
                    ])

_CMAP =torch.tensor(_CMAP, dtype=torch.float32) / 255.0

_CV2GL = np.asarray([
    [ 1.,  0.,  0.,  0.],
    [ 0., -1.,  0.,  0.],
    [ 0.,  0., -1.,  0.],
    [ 0.,  0.,  0.,  1.]])


def world_from_xy_depth(xy, depth, cam2world, intrinsics, orthogonal=False):
    '''Translates meshgrid of xy pixel coordinates plus depth to  world coordinates.
    '''
        
    x_cam = xy[:, 0]
    y_cam = xy[:, 1]

    fx, fy, cx, cy = intrinsics[0, 0], intrinsics[1, 1], intrinsics[0, 2], intrinsics[1, 2]
    
    if orthogonal:
        x_lift = (x_cam - cx) / fx
        y_lift = (y_cam - cy) / fy
        
    else:
        x_lift = (x_cam - cx) / fx * depth
        y_lift = (y_cam - cy) / fy * depth
        
    pix_coord_homo = np.stack([x_lift, y_lift, depth, np.ones_like(depth)], axis=0)
    world_coords = np.matmul(cam2world, pix_coord_homo)[:3, :]
        
    return world_coords.T



def render_scene_cam(model, z, cam2world, cam_int, img_sidelength, output_sidelength=512, dpt=None, cmap=_CMAP):
    
    with torch.no_grad():
        pose = torch.from_numpy(cam2world).float().unsqueeze(0)
        cam_int = torch.from_numpy(cam_int).float().unsqueeze(0)
        
        uv = np.mgrid[0:img_sidelength, 0:img_sidelength].astype(np.int32)
        uv = torch.from_numpy(np.flip(uv, axis=0).copy()).long()
        uv = uv.reshape(2, -1).transpose(1, 0).unsqueeze(0)
        
        predictions, depth_maps = model(pose, z, cam_int, uv, dpt=dpt)
        predictions = predictions.view(1, img_sidelength, img_sidelength, -1)
        
        predictions = F.interpolate(
            predictions.permute(0,3,1,2), 
            size=(output_sidelength, output_sidelength), 
            mode='bilinear', 
            align_corners=True).permute(0,2,3,1).view(1,-1,predictions.shape[-1])
        
        pred = torch.argmax(predictions, dim=2, keepdim=True)
        prob = F.softmax(predictions, dim=2)

        out_img = lin2img(pred, color_map=cmap).cpu().numpy()
        out_seg = pred.view(output_sidelength, output_sidelength, 1).cpu().numpy()
        
        depth_maps = depth_maps.view(img_sidelength, img_sidelength).cpu().numpy()
        
        out_img = (out_img.squeeze().transpose(1, 2, 0)) * 255.0
        out_img = out_img.round().clip(0, 255).astype(np.uint8)
        out_seg = out_seg.squeeze().astype(np.uint8)
        
        out_prob = prob.squeeze(0).view(output_sidelength, output_sidelength, -1).cpu().numpy()
    
        return out_img, out_seg, out_prob, depth_maps
    

def render_perp(mesh,intrinsics,cam=np.eye(4),model=np.eye(4),resolution=[512,512]):

    fx, cx, fy, cy = intrinsics[0,0], intrinsics[0,2], intrinsics[1,1], intrinsics[1,2]
    
    obj = pyrender.Mesh.from_trimesh(mesh, smooth=True)
    fovY = 2*np.arctan(resolution[0]/2/fy)
    
    # compose scene
    scene = pyrender.Scene(ambient_light=[0.1, 0.1, 0.1], bg_color=[0, 0, 0])
    camera = pyrender.PerspectiveCamera( yfov=fovY)
    light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=8)

    scene.add(obj, pose=  model)
    scene.add(light, pose=  model)
    scene.add(camera, pose = cam)
    
    # render scene
    r = pyrender.OffscreenRenderer(resolution[1], resolution[0])
    color, depth = r.render(scene)
    
    return color,depth


def render_ortho(mesh, cam=np.eye(4), model=np.eye(4), scale=[1,1], resolution=[512, 512]):

    obj = pyrender.Mesh.from_trimesh(mesh, smooth=False)

    scene = pyrender.Scene(ambient_light=[1.0, 1.0, 1.0], bg_color=[0, 0, 0])
    camera = pyrender.OrthographicCamera(xmag=1, ymag=resolution[1]/2/scale[1], znear=1e-3, zfar=10000.0)
    light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=2.0)

    scene.add(obj, pose=  model)
    scene.add(light, pose=  model)
    scene.add(camera, pose = cam)

    # render scene
    r = pyrender.OffscreenRenderer(resolution[1], resolution[0])
    color, depth = r.render(scene)
            
    return color, depth



class VolRender:
    def __init__(self, model, latent, cam_P, cam_K, dpt_range, 
                 ortho=False, level_set=0., resolution=(128, 128, 128)):
                    
        # load model
        self.level_set = level_set
        self.model = model
        self.latent = latent
        
        self.cam_P = cam_P
        self.cam_K = cam_K
        self.dpt_range = dpt_range
        self.reso = resolution
        self.ortho = ortho
        
        self.build_mesh()
        
        
    def _render_vol(self):
        
        # render volume
        full_prob = []
        
        for dpt in np.linspace(self.dpt_range[0], self.dpt_range[1], self.reso[-1]):
            _, _, prob, _ = render_scene_cam(
                self.model, self.latent, self.cam_P, self.cam_K, self.reso[0], dpt=dpt)
            full_prob.append(np.expand_dims(prob, 2))

        full_prob = np.concatenate(full_prob, axis=2)
        
        bg_prob = full_prob[:, :, :, 0]
        fg_prob = np.max(full_prob[:, :, :, 1:], axis=-1)

        vol_data = bg_prob - fg_prob
        
        return vol_data
        
        
    def build_mesh(self, vis=False, export_fp=None):
        
        vol_data = self._render_vol()
        
        v, f, vn, _ = marching_cubes(vol_data, self.level_set)
        
        c = v / vol_data.shape[2]
        
        
        v[:, 2] = (v[:, 2] / vol_data.shape[2]) * (self.dpt_range[1] - self.dpt_range[0]) + self.dpt_range[0]
        
        world_v = world_from_xy_depth(
            v[:, [1, 0]], v[:, 2], self.cam_P, self.cam_K, self.ortho).astype(np.float32)
        
        self.mesh = trimesh.Trimesh(vertices=world_v, faces=f[:,[2,1,0]],preprocess=False)
        self.world_v,self.f,self.color = world_v,f,c
        
        if export_fp is not None:
            self.mesh.export(export_fp)

        if vis:
        
            m = np.min(world_v, axis=0)
            ma = np.max(world_v, axis=0)
            # Corners of the bounding box
            v_box = np.array([[m[0], m[1], m[2]], 
                              [ma[0], m[1], m[2]], 
                              [ma[0], ma[1], m[2]], 
                              [m[0], ma[1], m[2]],
                              [m[0], m[1], ma[2]], 
                              [ma[0], m[1], ma[2]], 
                              [ma[0], ma[1], ma[2]], 
                              [m[0], ma[1], ma[2]]])
            
            f_box = np.array([[0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6], [6, 7], 
                              [7, 4], [0, 4], [1, 5], [2, 6], [7, 3]], dtype=np.int)

            # origin
            scale = np.linalg.norm(ma - m) * 0.5
            v_origin = np.array([
                [0., 0., 0.],
                [1., 0., 0.],
                [0., 1., 0.],
                [0., 0., 1.]]) * scale
            f_origin = np.array([[0, 1], [0, 2], [0, 3]], dtype=np.int) 

            p = mp.plot(world_v, f, c, return_plot=True)

        
    def render(self, cam_P, cam_K, resolution=None):
        
        fx, cx, fy, cy = cam_K[0,0], cam_K[0,2], cam_K[1,1], cam_K[1,2]
        
        render_reso = resolution if resolution is not None else self.reso[:2]
                        
        if self.ortho:
            render_cam_P = np.linalg.inv(_CV2GL@cam_P)
            _, depth = render_ortho(
                self.mesh, cam=render_cam_P, scale=[fx,fy], resolution=render_reso)
        else:
            render_cam_P = np.array(cam_P)
            render_cam_P[:3, [1,2]] = -render_cam_P[:3,[1,2]]
            render_cam_P = np.linalg.inv(render_cam_P)
            _, depth_piror = render_perp(
                self.mesh, cam_K, model=render_cam_P, resolution=render_reso)
            
        rgb, out_seg, _, pred_dpt = render_scene_cam(
            self.model, self.latent, cam_P, cam_K, self.reso[0])
                
        rgb2, out_seg2, _, pred_dpt_with_init = render_scene_cam(
            self.model, self.latent, cam_P, cam_K, self.reso[0], 
            dpt=torch.from_numpy(depth_piror.copy()).unsqueeze(0))
        
        return rgb, rgb2, out_seg,out_seg2,pred_dpt, pred_dpt_with_init, depth_piror