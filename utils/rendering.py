import numpy as np
from dataset.face_dataset import _campos2matrix
import cv2
import random,os,imageio
from scipy.stats import norm

from sklearn import mixture
from skimage import morphology
from matplotlib import pyplot as plt

import utils
import torch


_CMAP =torch.tensor(np.asarray([[0, 0, 0], [127, 212, 255], [255, 255, 127], [255, 255, 127], # 'background','skin', 'l_brow', 'r_brow'
                    [255, 255, 170], [255, 255, 170], [240, 157, 240], [255, 212, 255], #'l_eye', 'r_eye', 'r_nose', 'l_nose',
                    [31, 162, 230], [127, 255, 255], [127, 255, 255], #'mouth', 'u_lip', 'l_lip'
                    [0, 255, 85], [0, 255, 85], [0, 255, 170], [255, 255, 170], #'l_ear', 'r_ear', 'ear_r', 'eye_g'
                    [127, 170, 255], [85, 0, 255], [255, 170, 127], #'neck', 'neck_l', 'cloth'
                    [212, 127, 255], [0, 170, 255]#, 'hair', 'hat'
                    ]), dtype=torch.float32) / 255.0


def _build_cam_int(focal, H, W):
    return np.array([  [focal, 0., W // 2, 0.],
                       [0., focal, H // 2, 0],
                       [0., 0, 1, 0],
                       [0, 0, 0, 1]])


def _render_scene(model, pose, z, focal, img_sidelength):
    with torch.no_grad():
        pose = torch.from_numpy(pose).float().unsqueeze(0)
        cam_int = torch.from_numpy(
            _build_cam_int(focal, img_sidelength, img_sidelength)).float().unsqueeze(0)

        uv = np.mgrid[0:img_sidelength, 0:img_sidelength].astype(np.int32)
        uv = torch.from_numpy(np.flip(uv, axis=0).copy()).long()
        uv = uv.reshape(2, -1).transpose(1, 0).unsqueeze(0)

        predictions, depth_maps = model(pose, z, cam_int, uv)

        pred = torch.argmax(predictions, dim=2, keepdim=True)
        out_img = utils.common.lin2img(pred, color_map=_CMAP).cpu().numpy()
        out_seg = pred.view(img_sidelength, img_sidelength, 1).cpu().numpy()
        
        out_img = (out_img.squeeze().transpose(1, 2, 0)) * 255.0
        out_img = out_img.round().clip(0, 255).astype(np.uint8)

        out_seg = out_seg.squeeze().astype(np.uint8)
        out_seg = morphology.area_closing(out_seg, area_threshold=6000)

        return out_img, out_seg
    
    
def _render_scene_cam(model, pose, z, cam_int, img_sidelength,orthogonal=True):
    
    with torch.no_grad():
        pose = torch.from_numpy(pose).float().unsqueeze(0)
        cam_int = torch.from_numpy(cam_int).float().unsqueeze(0)
        
        uv = np.mgrid[0:img_sidelength, 0:img_sidelength].astype(np.int32)
        uv = torch.from_numpy(np.flip(uv, axis=0).copy()).long()
        uv = uv.reshape(2, -1).transpose(1, 0).unsqueeze(0)

        predictions, depth_maps = model(pose, z, cam_int, uv, orthogonal=orthogonal)

        pred = torch.argmax(predictions, dim=2, keepdim=True)

        out_img = utils.common.lin2img(pred, color_map=_CMAP).cpu().numpy()
        out_seg = pred.view(img_sidelength, img_sidelength, 1).cpu().numpy()
        
        out_img = (out_img.squeeze().transpose(1, 2, 0)) * 255.0
        out_img = out_img.round().clip(0, 255).astype(np.uint8)

        out_seg = out_seg.squeeze().astype(np.uint8)
        return out_img, out_seg


def _render_spiral_path(model,
                        cam_center,
                        lookat,
                        radii,
                        cam_int,
                        src_latent,
                        trgt_latent,
                        output_size=128,
                        steps=10,
                        orthogonal=False):
    # ROTATE
    R = np.linalg.norm(cam_center-lookat) + radii[0]

    theta = []
    theta_range = [0.0, -0.55, 0.55, 0.0]
    for i in range(len(theta_range)-1):
        theta.append( np.linspace(theta_range[i],theta_range[i+1], num=steps))
#         theta.append(np.logspace(0.0, 1, 10, endpoint=False)[::-1]/100)
    theta = np.concatenate(theta)
    x = R*np.sin(theta)
    y = np.zeros_like(x)
    z = R*np.cos(theta)
    cam_T = np.stack([x,y,z],axis=1) + lookat.reshape((1,3))

    vis_outputs,out_segs = [],[]
    for i in range(len(theta)):
        cam_pose = _campos2matrix(cam_T[i], lookat)        
        out_img, out_seg = _render_scene_cam(
            model, cam_pose, src_latent, cam_int, output_size, orthogonal=orthogonal)  

        vis_outputs.append(out_img)
        out_segs.append(out_seg)


    # SPPIRAL PATH
    t = np.linspace(0, 4*np.pi, steps*4, endpoint=True)
    for k in range(len(t)):
        cam_T = np.array([np.cos(t[k]), -np.sin(t[k]), -np.sin(0.5*t[k])]) * radii
        cam_T = cam_T[[1,2,0]] + cam_center
        cam_pose = _campos2matrix(cam_T, lookat)
        out_img, out_seg = _render_scene_cam(
            model, cam_pose, src_latent, cam_int, output_size, orthogonal=orthogonal)  
        vis_outputs.append(out_img)
        out_segs.append(out_seg)

    cdf_scale = 1.0/(1.0-norm.cdf(-steps//2,0,6)*2)
    for idx in range(-steps//2,steps//2+1):
        
        _w = (norm.cdf(idx,0,6)-norm.cdf(-steps//2,0,6))*cdf_scale
        latent = (1.0-_w)*src_latent + _w*trgt_latent
        
        out_img, out_seg = _render_scene_cam(
            model, cam_pose, latent, cam_int, output_size, orthogonal=orthogonal) 
        vis_outputs.append(out_img)
        out_segs.append(out_seg)


    for k in range(len(t)):
        cam_T = np.array([np.cos(t[k]), -np.sin(t[k]), -np.sin(0.5*t[k])]) * radii
        cam_T = cam_T[[1,2,0]] + cam_center
        cam_pose = _campos2matrix(cam_T, lookat)
        out_img, out_seg = _render_scene_cam(
            model, cam_pose, trgt_latent, cam_int, output_size, orthogonal=orthogonal)  
        vis_outputs.append(out_img)
        out_segs.append(out_seg)

    for idx in range(-steps//2,steps//2+1):
        _w = (norm.cdf(idx,0,6)-norm.cdf(-steps//2,0,6))*cdf_scale
        latent = (1.0-_w)*trgt_latent + _w*src_latent
        out_img, out_seg = _render_scene_cam(
            model, cam_pose, latent, cam_int, output_size, orthogonal=orthogonal) 
        vis_outputs.append(out_img)
        out_segs.append(out_seg)

    return out_segs, vis_outputs

def _render_uniform_path(   model, 
                            cam_center,
                            lookat,radii,
                            cam_int,
                            src_latent,
                            trgt_latent=None,
                            output_size=128,
                            steps=10,
                            orthogonal=False):
    # ROTATE
    R = np.linalg.norm(cam_center-lookat) + radii[0]

    theta = []
    theta_range = [-0.55, 0.55]
    for i in range(len(theta_range)-1):
        theta.append( np.linspace(theta_range[i],theta_range[i+1], num=steps))
        
    ys = np.linspace(0.3,-0.2,5,endpoint=True)
    
    theta = np.concatenate(theta)
    x = R*np.sin(theta)
    z = R*np.cos(theta)
    
    vis_outputs,out_segs = [],[]
    for y in ys:
        
        cam_T = np.stack([x,np.ones_like(x)*y,z],axis=1) + lookat.reshape((1,3))

        for i in range(len(theta)):
            cam_pose = _campos2matrix(cam_T[i], lookat)        
            out_img, out_seg = _render_scene_cam(
                model, cam_pose, src_latent, cam_int, output_size, orthogonal=orthogonal)  
            vis_outputs.append(out_img)
            out_segs.append(out_seg)

    return out_segs,vis_outputs


def rendering_fvv(model, num_instrances, cam_int, img_size, render_mode='spiral'):

    f = 500
    _DEFAULT_CAM_INT = np.array([[f,0,img_size//2],[0,f,img_size//2],[0,0,1]])

    lookat = np.asarray([0,0.0,0.1])
    print('*** lookat = ', lookat)

    cam_center =  lookat + np.asarray([0, 0, 0.9])
    cam_up = np.asarray([0.0, 1.0, 0.0])


    radii, focus_depth = np.asarray([0.1,0.3,0.2]), 4.5 # z,x,y

    _LOG_ROOT = os.path.join('../log/mv')
    os.makedirs(_LOG_ROOT, exist_ok=True)
    os.makedirs(os.path.join(_LOG_ROOT, 'vis'), exist_ok=True)
    print('Logging root = ', _LOG_ROOT)

    _INTERP_STEPS = 3
    all_instances = list(range(num_instrances))
    random.shuffle(all_instances)

    _ORTHO=False
    for num_comp in [16]:#, 2, 4, 8, 16
        gmm = mixture.GaussianMixture(
            n_components=num_comp, covariance_type='full')
        gmm.fit(model.latent_codes.weight.data.cpu().numpy())

        print('Processing gmm :', num_comp)
        
        for i in range(1000):
            
            src_latent = torch.from_numpy(gmm.sample(1)[0]).float()
            trgt_latent = torch.from_numpy(gmm.sample(1)[0]).float()
        
        
            output_dir = os.path.join(
                _LOG_ROOT, 'gmm_%02d'%(num_comp), '%03d'%(i))
            os.makedirs(os.path.join(output_dir), exist_ok=True)

            if render_mode == 'uniform':
                out_segs,vis_outputs = _render_uniform_path(
                    model, cam_center,lookat,radii,cam_int,src_latent,trgt_latent,img_size,_INTERP_STEPS,_ORTHO)
            elif render_mode == 'spiral':
                out_segs,vis_outputs = _render_spiral_path(
                    model, cam_center,lookat,radii,cam_int,src_latent,trgt_latent,img_size,_INTERP_STEPS,_ORTHO)

            for k,out_seg in enumerate(out_segs):
                output_fp = os.path.join(output_dir, '%04d.png'%k)
                utils.common.write_img(out_seg, output_fp)       

            vis_fp = os.path.join(_LOG_ROOT, 'vis', '%02d_'%(num_comp)+os.path.basename(output_dir)+'.gif')
            print('\t [DONE] save vis to : ', vis_fp)
            imageio.mimsave(vis_fp, vis_outputs, fps=15.0)