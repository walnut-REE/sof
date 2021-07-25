# Semantic Occupancy Field

This repository contains the code for training/generating SOF (semantic occupancy field) as part of the TOG submission: [SofGAN: A Portrait Image Generator with Dynamic Styling](https://arxiv.org/abs/2007.03780).

## Installation
Clone the main SofGAN repo by `git clone --recursive https://github.com/apchenstu/softgan_test.git`. This repo will be automatically included in `softgan_test/modules`.

## Training
We train 
### Data preparation

Create a root directory (e.g. `data`), and for each instance (e.g. `00000`) create a folder with seg images and calibrated camera poses. The folder structure looks like:

```bash
└── data   # instance id
    └── 00000
    │   ├── cam2world.npy
    │   ├── cameras.npy
    │   ├── 00000.png
    │   ...
    │   └── 00029.png
    ├── 00001
    │   └── ...
    ...
    └── xxxxx
        └── ...
```

Download the data from [here](). We also provide a [notebook](https://github.com/walnut-REE/sof/blob/main/scripts/DataPreprocess.ipynb) for data preprocessing.

## Training
```
python train.py --config_filepath=./configs/face_seg_real.yml 
```

## Inference
Once trained, SOF could be used for generating free-view segmentation maps for arbitrary instances in the geometric space. The inference codes are provided in notebooks in `scripts`:
* To generate sampling free-view portrait segmentations from the geometry space, please refer to `scripts\Test_MV_Inference.ipynb`.
* To visulalize a trained SOF volume as in Fig.5, please use `scripts\Test_Slicing.ipynb`.
* To calculat mIOU during SOF training (Fig.9), please modify the model checkpoint directory and run `scripts\Test_mIoU.ipynb`.
* To generate the portrait geometry proxy in Fig.20 with marching cube, please refer to `scripts\Test_MCube.ipynb`.
* We also provide `scripts\Test_GMM.ipynb` for miscs like fitting GMM model to the geometric space.

# Acknowledgment
Thanks [vsitzmann](https://github.com/vsitzmann) for sharing the awesome idea of [SRNs](https://github.com/vsitzmann/scene-representation-networks.git), which has greatly inspired our design of SOF.