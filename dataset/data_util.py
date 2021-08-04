import functools
import cv2
import numpy as np
import imageio
from glob import glob
import os
import shutil
import skimage
import pandas as pd

def parse_intrinsics(filepath, trgt_sidelength=None, invert_y=False):
    # Get camera intrinsics
    if os.path.splitext(filepath)[-1] == '.npy':
        cam_int = np.load(filepath)

        cx, cy = cam_int[0, 2], cam_int[1, 2]
        fx, fy = cam_int[0, 0], cam_int[1, 1]

    else:
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

def load_rgb(path, sidelength=None, cam_int=None):
    # print(path)
    img = imageio.imread(path)
    # print(img.shape, np.max(img), np.min(img), np.unique(img), img.dtype)

    img = skimage.img_as_float32(img)

    img = square_crop_img(img)

    if sidelength is not None:
        height, width = img.shape
        img = cv2.resize(img, (sidelength, sidelength), interpolation=cv2.INTER_AREA)

        if cam_int is not None:
            cam_int = cam_int.copy()
            cam_int[0, 2] *= (sidelength / width)
            cam_int[1, 2] *= (sidelength / height)
            cam_int[0, 0] *= (sidelength / width)
            cam_int[1, 1] *= (sidelength / height)

    img -= 0.5
    img *= 2.
    img = img.transpose(2, 0, 1)
    return img, cam_int


def load_gray(path, sidelength=None, cam_int=None):

    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE).astype(np.float32)
    assert (len(img.shape) == 2 or img.shape[2] == 1)

    if sidelength is not None:
        height, width = img.shape
        img = cv2.resize(img, (sidelength, sidelength), interpolation=cv2.INTER_NEAREST)

        if cam_int is not None:
            cam_int = cam_int.copy()
            cam_int[0, 2] *= (sidelength / width)
            cam_int[1, 2] *= (sidelength / height)
            cam_int[0, 0] *= (sidelength / width)
            cam_int[1, 1] *= (sidelength / height)

    
    img = img.reshape(1, -1).transpose(1, 0)

    return img, cam_int


def load_seg_map(path, sidelength=None, num_classes=20, remap_fn=None, cam_int=None):

    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    assert (len(img.shape) == 2 or img.shape[2] == 1), path

    if sidelength is not None:
        height, width = img.shape
        img = cv2.resize(img, (sidelength, sidelength), interpolation=cv2.INTER_NEAREST)

        if cam_int is not None:
            cam_int = cam_int.copy()
            cam_int[0, 2] *= (sidelength / width)
            cam_int[1, 2] *= (sidelength / height)
            cam_int[0, 0] *= (sidelength / width)
            cam_int[1, 1] *= (sidelength / height)
    
    img = img.reshape(1, -1).transpose(1, 0)

    if np.max(img) > num_classes:
        scaler = np.max(img)/(num_classes-1)
        img = img // scaler

    if remap_fn is not None:
        img = remap_fn(img)

    return img.astype(np.uint8), cam_int

def load_video_seg(path, sidelength=None):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED).astype(np.float32)

    H, W, C = img.shape    
    seg = img[:, W//2:, 0]

    if sidelength is not None:
        seg = cv2.resize(seg, (sidelength, sidelength), interpolation=cv2.INTER_NEAREST)
    
    seg = seg.reshape(1, -1).transpose(1, 0)

    return seg


def load_depth(path, sidelength=None, zRange=None):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED).astype(np.float32)

    if sidelength is not None:
        img = cv2.resize(img, (sidelength, sidelength), interpolation=cv2.INTER_LINEAR)

    if len(img.shape) == 3:
        img = img[:, :, :1]
        img = img.transpose(2, 0, 1)
    else:
        img = img[None, :, :]

    img = img / 255.0

    if zRange is not None:
        mask = (img > 1e-12).astype(np.float32)
        img = img * (zRange[1] - zRange[0]) + zRange[0]
        img = img * mask      
    
    return img


def load_pose(filename):
    lines = open(filename).read().splitlines()
    if len(lines) == 1:
        pose = np.zeros((4, 4), dtype=np.float32)
        for i in range(16):
            pose[i // 4, i % 4] = lines[0].split(" ")[i]
        return pose.squeeze()
    else:
        lines = [[x[0], x[1], x[2], x[3]] for x in (x.split(" ") for x in lines[:4])]
        return np.asarray(lines).astype(np.float32).squeeze()


def load_params(filename):
    lines = open(filename).read().splitlines()

    params = np.array([float(x) for x in lines[0].split()]).astype(np.float32).squeeze()
    return params


def square_crop_img(img):
    min_dim = np.amin(img.shape[:2])
    center_coord = np.array(img.shape[:2]) // 2
    img = img[center_coord[0] - min_dim // 2:center_coord[0] + min_dim // 2,
          center_coord[1] - min_dim // 2:center_coord[1] + min_dim // 2]
    return img


def train_val_split(object_dir, train_dir, val_dir):
    dirs = [os.path.join(object_dir, x) for x in ['pose', 'rgb', 'depth']]
    data_lists = [sorted(glob(os.path.join(dir, x)))
                  for dir, x in zip(dirs, ['*.txt', "*.png", "*.png"])]

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)


    [os.makedirs(os.path.join(train_dir, x), exist_ok=True) for x in ['pose', 'rgb', 'depth']]
    [os.makedirs(os.path.join(val_dir, x), exist_ok=True) for x in ['pose', 'rgb', 'depth']]

    shutil.copy(os.path.join(object_dir, 'intrinsics.txt'), os.path.join(val_dir, 'intrinsics.txt'))
    shutil.copy(os.path.join(object_dir, 'intrinsics.txt'), os.path.join(train_dir, 'intrinsics.txt'))

    for data_name, data_ending, data_list in zip(['pose', 'rgb', 'depth'], ['.txt', '.png', '.png'], data_lists):
        val_counter = 0
        train_counter = 0
        for i, item in enumerate(data_list):
            if not i % 3:
                shutil.copy(item, os.path.join(train_dir, data_name, "%06d" % train_counter + data_ending))
                train_counter += 1
            else:
                shutil.copy(item, os.path.join(val_dir, data_name, "%06d" % val_counter + data_ending))
                val_counter += 1


def glob_imgs(path):
    imgs = []
    for ext in ['*.png', '*.jpg', '*.JPEG', '*.JPG']:
        imgs.extend(glob(os.path.join(path, ext)))
    return imgs


def read_view_direction_rays(direction_file):
    img = cv2.imread(direction_file, cv2.IMREAD_UNCHANGED).astype(np.float32)
    img -= 40000
    img /= 10000
    return img


def shapenet_train_test_split(shapenet_path, synset_id, name, csv_path):
    '''

    :param synset_id: synset ID as a string.
    :param name:
    :param csv_path:
    :return:
    '''
    parsed_csv = pd.read_csv(filepath_or_buffer=csv_path)
    synset_df = parsed_csv[parsed_csv['synsetId'] == int(synset_id)]

    train = synset_df[synset_df['split'] == 'train']
    val = synset_df[synset_df['split'] == 'val']
    test = synset_df[synset_df['split'] == 'test']
    print(len(train), len(val), len(test))

    train_path, val_path, test_path = [os.path.join(shapenet_path, str(synset_id) + '_' + name + '_' + x)
                                       for x in ['train', 'val', 'test']]
    
    os.makedirs(train_path, exist_ok=True)
    os.makedirs(val_path, exist_ok=True)
    os.makedirs(test_path, exist_ok=True)

    for split_df, trgt_path in zip([train, val, test], [train_path, val_path, test_path]):
        for row_no, row in split_df.iterrows():
            try:
                shutil.copytree(os.path.join(shapenet_path, str(synset_id), str(row.modelId)),
                                os.path.join(shapenet_path, trgt_path, str(row.modelId)))
            except FileNotFoundError:
                print("%s does not exist" % str(row.modelId))


def transform_viewpoint(v):
    """Transforms the viewpoint vector into a consistent representation"""

    return np.concatenate([v[:, :3],
                           np.cos(v[:, 3:4]),
                           np.sin(v[:, 3:4]),
                           np.cos(v[:, 4:5]),
                           np.sin(v[:, 4:5])], 1)


def euler2mat(z=0, y=0, x=0):
    Ms = []
    if z:
        cosz = np.cos(z)
        sinz = np.sin(z)
        Ms.append(np.array(
            [[cosz, -sinz, 0],
             [sinz, cosz, 0],
             [0, 0, 1]]))
    if y:
        cosy = np.cos(y)
        siny = np.sin(y)
        Ms.append(np.array(
            [[cosy, 0, siny],
             [0, 1, 0],
             [-siny, 0, cosy]]))
    if x:
        cosx = np.cos(x)
        sinx = np.sin(x)
        Ms.append(np.array(
            [[1, 0, 0],
             [0, cosx, -sinx],
             [0, sinx, cosx]]))
    if Ms:
        return functools.reduce(np.dot, Ms[::-1])
    return np.eye(3)


def look_at(vec_pos, vec_look_at):
    z = vec_look_at - vec_pos
    z = z / np.linalg.norm(z)

    x = np.cross(z, np.array([0., 1., 0.]))
    x = x / np.linalg.norm(x)

    y = np.cross(x, z)
    y = y / np.linalg.norm(y)

    view_mat = np.zeros((3, 3))

    view_mat[:3, 0] = x
    view_mat[:3, 1] = y
    view_mat[:3, 2] = z

    return view_mat
