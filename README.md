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

Download the data from [here]().

## Training
```
python train.py --config_filepath=./configs/face_seg_real.yml 
```

## Inference
TBD.
<!-- Modify `configs/test.yml` with corredct `data_root` and `checkpoint_path`, use the following command to test your model:

```
python test.py --data_root [path to directory with dataset] ] \
               --logging_root [path to directoy where test output should be written to] \
               --num_instances [number of instances in training set (for instance, 2433 for shapenet cars)] \
               --checkpoint [path to checkpoint]
```

We also provide testing scripts in `scripts`, please find the instruction [here](https://github.com/walnut-REE/sof/scripts).  -->


# Acknowledgment
Thanks [vsitzmann](https://github.com/vsitzmann) for sharing the awesome idea of [SRNs](https://github.com/vsitzmann/scene-representation-networks.git), which has greatly inspired our design of SOF.