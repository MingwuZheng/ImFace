# ImFace: A Nonlinear 3D Morphable Face Model with Implicit Neural Representations

###  [Paper](https://arxiv.org/abs/2203.14510) | [Data](https://facescape.nju.edu.cn/)

Official code for CVPR 2022 paper ImFace: A Nonlinear 3D Morphable Face Model with Implicit Neural Representations.

The paper presents a novel 3D morphable face model, namely ImFace, to learn a nonlinear and continuous space with implicit neural representations.
ImFace explicitly disentangles facial shape morphs into two separate deformation fields associated with identity and expression, respectively. Then, we can interoperate identity and expression embeddings to generate high quality 3D faces.


## Installation Requirmenets
The code is compatible with python 3.8.10 and pytorch 1.9.0.
You can create an anaconda environment called `imface` with the required dependencies by running:

```
conda create -n imface
conda activate imface
pip install -r requirement.txt
```

## Usage
### Data
We sample 5,323 face scans from 355 persons with 15 expressions from the <a href="https://facescape.nju.edu.cn/" target="_blank">FaceScape Dataset</a>. 
The data contains each individual's mesh and key points, which can be used in our experiment.
#### Data Preprocessing
First, you have to extracted the TU mesh downloaded from FaceScape to `dataset/FacescapeOriginData`. Then, to get pseudo watertight mesh and samples from the face, run:
```
python data_preprocess/preprocess.py
```
For more information on the preprocessing, we recommend you to our paper.

After preprocessing, the dataset folder sturture is as follow:
```
dataset
  |————FacescapeOriginData         # data extracted from FaceScape
  |       └——————1                 # person identity
  |         └——————1_neutral.obj   # TU mesh
  |         └——————2_smile.obj     # TU mesh
  |         └——————...
  |       └——————2                 # person identity
  |         └——————...
  |————Facescape
  |       └——————1                      # person identity
  |       	└——————1_free_grd.npy  # (15000, 3) sphere points gradient vector with neutral expression (number 1)
  |       	└——————1_free_pcl.npy  # (15000, 3) sphere points position 
  |       	└——————1_free_sdf.npy  # (15000, 1) sphere points SDF 
  |       	└——————1_surf_nor.npy  # (250000, 3) surface points normal
  |       	└——————1_surf_pcl.npy  # (250000, 3) surface points position
  |       	└——————1.bnd           # (68, 3) key points
  |       	└——————1.obj           # pseudo watertight mesh
  |             └——————2_free_grd.npy  # (15000, 3) sphere points gradient vector with smile expression (number 2)
  |       	└——————2_free_pcl.npy  # (15000, 3) sphere points position 
  |       	└——————2_free_sdf.npy  # (15000, 1) sphere points SDF 
  |       	└——————2_surf_nor.npy  # (250000, 3) surface points normal
  |       	└——————2_surf_pcl.npy  # (250000, 3) surface points position
  |       	└——————2.bnd           # (68, 3) key points
  |       	└——————2.obj           # pseudo watertight mesh
  |       	└——————...
  |       └——————2                     # person identity
  |       	└——————...
```
### Train
For training ImFace, run:
```
python run/train.py [--config file's name]
```
Please check pathes in your config file are both correct. Results can be found in `result/<timestamp>/`.
Our trained model can be downloaded as follow:

| Trained Model            | Description  |
|-------------------|-------------------------------------------------------------|
| <a href="" target="_blank">ImFace(SE3)</a> (coming soon) | deformation with a SE(3) field|
| <a href="" target="_blank">ImFace(translation)</a> (coming soon) | common translation deformation |

### Evaluation
For evaluating chamfer distance and f1_score of testing set with trained model, run:
```
python run/fit.py [--config file's name]
```
Please modify the `load_path`  in config file with your checkpoint's path. Results can be found in `result/<timestamp>/fit`.

### Fit with one 3D face
For fitting one sample, which is generated from FaceScape <a href="https://nbviewer.org/github/zhuhao-nju/facescape/blob/master/toolkit/demo_bilinear_basic.ipynb" target="_blank">toolkit</a>, run:
```
# preprocess demo data
python demo/process.py
# fit demo
python run/fit_one_sample.py [--config file's name]
```
Please modify the `load_path` in config file with your checkpoint's path. Results can be found in `demo/fit`.


## Citation
If you find our work useful in your research, please consider citing:

	@inproceedings{zheng2022imface,
	title={ImFace: A Nonlinear 3D Morphable Face Model with Implicit Neural Representations},
	author={Zheng, Mingwu and Yang, Hongyu and Huang, Di and Chen, Liming},
	booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
	pages={20343--20352},
	year={2022}
	}

