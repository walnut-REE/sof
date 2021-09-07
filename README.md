# Semantic Occupancy Field

This repository contains the code for training/generating SOF (semantic occupancy field) as part of the TOG submission: [SofGAN: A Portrait Image Generator with Dynamic Styling](https://arxiv.org/abs/2007.03780).

## Installation
Clone the main SofGAN repo by `git clone --recursive https://github.com/apchenstu/softgan_test.git`. This repo will be automatically included in `softgan_test/modules`.

### Data preparation

Create a root directory (e.g. `data`), and for each instance (e.g. `00000`) create a folder with seg images and calibrated camera poses. The folder structure looks like:

```bash
└── data   # instance id
    └── 00000
    │   ├── cam2world.npy       # camera extrinsics
    │   ├── cameras.npy            
    │   ├── intrinsic.npy       # camera intrinsics
    │   ├── zRange.npy          # optional only when use depth for training
    │   ├── 00000.png
    │   ...
    │   └── 00029.png
    ├── 00001
    │   └── ...
    ...
    └── xxxxx
        └── ...
```

Download the example data from [here](https://drive.google.com/file/d/1yW5YAqhZKRPkyzdX7LPLR5K42YOiUDzk/view?usp=sharing). We provide a [notebook](https://github.com/walnut-REE/sof/blob/main/scripts/DataPreprocess.ipynb) for data preprocessing.

Ideally, `SOF` could be trained with your own datasets with multi-view face segmentation maps. Similar to [SRNs](https://github.com/vsitzmann/scene-representation-networks.git) we uses an "OpenCV" style camera coordinate system, where the Y-axis points downwards (the up-vector points in the negative Y-direction), the X-axis points right, and the Z-axis points into the image plane. Camera poses are assumed to be in a "camera2world" format, i.e., they denote the matrix transform that transforms camera coordinates to world coordinates. Please specify `--orthogonal` during training if you're using orthogonal projection for your own data. Please also notice that you might need to change the `sample_instances_*` and `sample_observations_*` parameter according to the number of instances and views of your own dataset.

As the accuracy of camera parameters might largly affect the training, you can specify `--opt_cam` during training to automatically optimize the camera parameters.

## Training
### STEP 1: Training network parameters
The training is done following two phrases. Firstly, please train the network parameters with multiview segmaps:
```
python train.py --config_filepath=./configs/face_seg_real.yml 
```
Training might take 1 to 3 days depends on the dataset size and quality.

### STEP 2 (optional): Inverse rendering
We use inverse rendering to expand the trained geometric sampling space with single view segmaps collected from [CelebAMaskHQ](https://github.com/switchablenorms/CelebAMask-HQ). The example config file is provided in `./configs/face_seg_single_view.yml`, notice that we set `--overwrite_embeddings` and `--freeze_networks` to `True`, and specify `--checkpoint_path` as the trained checkpoint in [STEP 1](https://github.com/walnut-REE/sof#training). After training, you can access the corresponding latent code for each portrait by loading the checkpoint.

```
python train.py --config_filepath=./configs/face_seg_single_view.yml 
```
Similar process could be used to back project in-the-wild portrait images into a latent vector in `SOF` geometric sampling space, and used for mutiview portrait generation.


## Pretrained Checkpoints
Please download the pre-trained checkpoint from either [GoogleDrive](https://drive.google.com/drive/folders/1Ursk30iAZY6cdPKE8TznYhOYHuFAzyK8?usp=sharing) or [BaiduDisk](https://pan.baidu.com/s/1KXUDwEhPt3YEarWI85Hufw) (password: k0b8) and save to `./checkpoints`.

## Inference
Please follow [`renderer.ipynb`](https://github.com/apchenstu/sofgan/blob/master/renderer.ipynb) in the SofGAN repo for free-view portrait generation.

Once trained, SOF could be used for generating free-view segmentation maps for arbitrary instances in the geometric space. The inference codes are provided in notebooks in `scripts`:
* Most testing codes are included in `scripts/TestAll.ipynb`, e.g. generating multiview images, modify attributes, visualize depth layers and build depth prior with marching cube.
* To generate sampling free-view portrait segmentations from the geometry space, please refer to `scripts/Test_MV_Inference.ipynb`.
* To visulalize a trained SOF volume as in Fig.5, please use `scripts/Test_Slicing.ipynb`.
* To calculat mIOU during SOF training (Fig.9), please modify the model checkpoint directory and run `scripts/Test_mIoU.ipynb`.
* We also provide `scripts/Test_GMM.ipynb` for miscs like fitting GMM model to the geometric space.

# Acknowledgment
Thanks [vsitzmann](https://github.com/vsitzmann) for sharing the awesome idea of [SRNs](https://github.com/vsitzmann/scene-representation-networks.git), which has greatly inspired our design of SOF.
