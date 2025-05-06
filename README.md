# Domain-Adaptive Full-Face Gaze Estimation via Novel-View-Synthesis and Feature Disentanglement
[[arXiv 2305.16140](https://arxiv.org/abs/2305.16140)]

<a href="https://jqin-home.github.io/">Jiawei Qin</a><sup>1</sup>,
Takuru Shimoyama<sup>1</sup>,
<a href="https://www.ccmitss.com/zhang">Xucong Zhang</a><sup>2</sup>,
<a href="https://www.yusuke-sugano.info/">Yusuke Sugano</a><sup>1</sup><br>
<sup>1</sup>The University of Tokyo   
<sup>2</sup>Delft University of Technology

---

# ETH-XGaze 3D

[ETH-XGaze](https://ait.ethz.ch/xgaze) is a large‑scale full‑face gaze dataset captured with 18 synchronized cameras.  
However, its original camera parameters contain noise, and 2D landmark annotations contain mismatches.  
- We use [Agisoft Metashape](https://www.agisoft.com/) to **re-calibrate** the camera extrinsic parameters for each frame; then we compute averaged camera extrinsic parameters for each subject 
- We re-detected the 2D landmarks, and compute a 3D landmarks by optimizing for the whole 18 images.

We directly provide the updated camera parameters & updated annotations, which can be used to re-normalize the XGaze (not included in this code).

With refined cameras and landmarks, the normalized XGaze dataset are less noisy, as the comparison illustrated below:

#### Original normalized XGaze
<img src=./assets/old_xgaze.jpg width=700>

#### Updated normalized XGaze
<img src=./assets/new_xgaze.jpg width=700>



# Overview
This repo contains:

1. **Accurate multi‑view 3D reconstruction** for every frame (via Agisoft Metashape).  
2. **Photo‑realistic novel‑view rendering** with PyTorch3D, giving synthetic images. The synthetic images show comparable performance with the real data under same head pose/gaze.


---


## Installation
> Tested on Ubuntu 20.04, CUDA 12.2, Python 3.8, PyTorch3D 0.7.8

```bash
git clone https://github.com/ut-vision/XGaze3D.git
cd XGaze3D

## Conda 
conda create -n xgaze3d python=3.8
conda activate xgaze3d


pip install torch==2.2.0 torchvision==0.17.0 --index-url https://download.pytorch.org/whl/cu121
pip install "git+https://github.com/facebookresearch/pytorch3d.git"
pip install -r requirements.txt
pip install face_alignment
```
### Metashape
- Download the Python 3 Module from https://www.agisoft.com/downloads/installer/
- Install and activate/de-activate:
```bash
pip install Metashape-2.2.1-cp37.cp38.cp39.cp310.cp311-abi3-linux_x86_64.whl
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libffi.so.7 python3 -c "import Metashape; Metashape.license.activate("<YOUR-LICENSE-KEY>"); print('Activated Metashape')"
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libffi.so.7 python3 -c "import Metashape; Metashape.license.deactivate(); print('De-activated Metashape')"
```
`LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libffi.so.7` is needed if you used a Conda environment.


## Data Preparation
- **ETH‑XGaze**  
  Download the raw dataset from the [ETH-XGaze](https://ait.ethz.ch/xgaze).

- **Updated files (cameras & annotations)**  
Download the updated files from [Google Drive](https://drive.google.com/drive/folders/1SQp0O-hZvVBwH5XyAP2ZQmPJ2e0QqMFk?usp=sharing).
  - unzip `avg_cams_final.zip` and `annotation_updated.zip`
  - Placing the files as follows:
  ```yaml
  ETH-XGaze/
  ├─ calibration/cam_calibration/
  ├─ avg_cams_final/            # ★ re‑calibrated cameras
  ├─ data/
  │  ├─ train/
  │  ├─ annotation_train/       # original annotations
  │  └─ annotation_updated/     # ★ refined annotations
  └─ light_meta.yaml            # ★ The information of the lighting condition: we only use the full-light frames

- **Places365**: Download the validation split (val_256) from the [Places365](http://places2.csail.mit.edu/).

## 1. Run 3D Reconstruction
- The raw images are pre-processed: cropping, resizing, background-removing.
- `--resize 1200`: we resize the cropped face to 1200x1200 to reduce the processing time without harming the quality, you can adjust this size to find a better tradeoff.

```bash
cd src
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libffi.so.7 python3 main_reconstruct.py \
  --xgaze_basedir <PATH_TO_ETH-XGaze> \
  --output_path <SAVE_DIR> \
  --resize 1200 \             # trade‑off between quality & speed
  --grp configs/group.yaml    # subjects to process
```

####  Output Format
```yaml
<SAVE_DIR>/
└─ yyyy‑mm‑dd/hh‑mm‑ss/
   └─ 1200_data/train/
      └─ subject0000/frame0000/
         └─ final_obj/
            ├─ mvs_3d.obj
            ├─ mvs_3d.mtl
            └─ mvs_3d.jpg
```
We included one 3D sample in [assets/subject0000/frame0000/final_obj](./assets/subject0000/frame0000/final_obj)

<img src=./assets/meshlab.gif width=700>

## 2. Run Rendering
```bash
python3 main_render.py \
  --renderer_split_verts 2 \               # split meshes if GPU RAM is low
  --xgaze_basedir <PATH_TO_ETH-XGaze> \
  --xgaze_3d_basedir <PATH_TO_THE_3D_OUTPUT> \
  --output_path <RENDER_OUT> \
  --grp configs/group.yaml \
  --place365_dir <PATH_TO_Places365_val_256>
```


#### Output and Visualization
The rendering code will produce two versions
- Rendered with green background and full light
  - <img src=./assets/sample.jpg width=500>
- aug: Rendered with random images from Places365 as background and random low light
  - <img src=./assets/sample_aug.jpg width=500>

```yaml
<RENDER_OUT>/render_<time>/
├─ subject0000.h5
├─ subject0003.h5
├─ ...
└─ aug/
    ├─ subject0000.h5
    ├─ subject0003.h5
    └─ ...
         
```

| Key               | Shape                            | Description                                          |
|-------------------|----------------------------------|------------------------------------------------------|
| face_gaze         | (N, 2)                    | Gaze angles (pitch, yaw)                             |
| face_head_pose    | (N, 3)                    | Head pose (roll, pitch, yaw)                         |
| face_patch        | (N, 448 × 448 × 3)        | Rendered face                                        |
| rotation_matrix   | (N, 3 × 3)                | Source → target rotation                             |
| face_mat_norm     | (N, 3 × 3)                | Camera normalization matrix                          |
| landmarks_norm    | (N, 68, 2)                | 2D landmark positions in normalized space            |


```bash 
python3 print_result.py --data_dir <SAVE_DIR>
```




#### The updated annotation formats
`subject0000_updated.csv`:
  - column 1: frame folder name
  - column 2: image file name
  - column 3-4: gaze point in the screen coordinate system, it is the same for all samples in the same frame folder
  - column 5-7: gaze point location in the current camera coordinate system. 
  - column 8-10: (UPDATED) head pose rotation in the current camera coordinate system, which is estimated from detected 2D facial landmarks.
  - column 11-13: (UPDATED) head pose translation in the current camera coordinate system, which is estimated from detected 2D facial landmarks.
  - column 14-150: the 68 detected 2D landmarks,
  - column 14-114: (ADDED) reprojected 2D facial landmarks, and only has 50 landmarks







# Citation
If you find the dataset useful for your research, please consider citing:
```
@article{qin2023domain,
  title={Domain-adaptive full-face gaze estimation via novel-view-synthesis and feature disentanglement},
  author={Qin, Jiawei and Shimoyama, Takuru and Zhang, Xucong and Sugano, Yusuke},
  journal={arXiv preprint arXiv:2305.16140},
  year={2023}
}
```
:fire: **Huge thanks to Xucong Zhang** for contributing to the Metashape multi‑view reconstruction scripts!

### License
ETH‑XGaze, Metashape, and Places365 are subject to their respective licenses—please comply with their terms.



## Contact
If you have any questions, feel free to contact Jiawei Qin at jqin@iis.u-tokyo.ac.jp.
