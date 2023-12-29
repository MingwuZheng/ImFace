# ImFace: A Nonlinear 3D Morphable Face Model with Implicit Neural Representations

###  [Paper](https://arxiv.org/abs/2203.14510) | [Data](https://facescape.nju.edu.cn/)

<img src="./media/1.gif" width=20%><img src="./media/2.gif" width=20%><img src="./media/3.gif" width=20%><img src="./media/4.gif" width=20%><img src="./media/5.gif" width=20%>


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

## [2023.12] ImFace++
Release the ImFace++ trained on FaceScape dataset. We propose a refinement displacement field which can faithfully encode high-frequency details, thereby enhancing the representation capabilities of ImFace++. ImFace++ can reconstruct and synthesize 3D faces spanning a wide range of facial expressions and it renders more vivid results. 

<img src="./media/6.gif" width=20%><img src="./media/7.gif" width=20%><img src="./media/8.gif" width=20%><img src="./media/9.gif" width=20%><img src="./media/10.gif" width=20%>

Technical details of ImFace++ can be found in this research paper:
<a href="https://arxiv.org/abs/2312.04028" target="_blank">ImFace++: A Sophisticated Nonlinear 3D Morphable Face Model with Implicit Neural Representations</a>

Checkout to the branch <a href="https://github.com/MingwuZheng/ImFace/tree/imface%2B%2B" target="_blank">ImFace++</a> for more details of generation and fitting.

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
| <a href="https://drive.google.com/drive/folders/1wKvWZHhuLO6ISc0jML8DfM_u4de2m2-y?usp=sharing" target="_blank">ImFace(SE3)</a> | deformation with a SE(3) field|
| <a href="https://drive.google.com/drive/folders/1QIkyvSdNmwPudC2cm4se-UoBNTh5wBjU?usp=sharing" target="_blank">ImFace(translation)</a> | common translation deformation |

### Evaluation
For evaluating chamfer distance and f1_score of testing set with trained model, run:
```
python run/fit.py [--config file's name]
```
Please modify the `load_path` in config file with your checkpoint's path and make sure `warp_type` matches the trained model. Results can be found in `result/<timestamp>/fit`.

### Fit with one 3D face
For fitting one sample, which is generated from FaceScape <a href="https://nbviewer.org/github/zhuhao-nju/facescape/blob/master/toolkit/demo_bilinear_basic.ipynb" target="_blank">toolkit</a>, run:
```
# preprocess demo data
python demo/process.py
# fit demo
python run/fit_one_sample.py [--config file's name]
```
Please modify the `load_path` in config file with your checkpoint's path and make sure `warp_type` matches the trained model. Results can be found in `demo/fit`.

### Related Projects

<a href="https://github.com/aejion/NeuFace" target="_blank">NeuFace: Realistic 3D Neural Face Rendering from Multi-view Images (CVPR 2023)</a>

<a href="https://arxiv.org/abs/2312.04028" target="_blank">ImFace++: A Sophisticated Nonlinear 3D Morphable Face Model with Implicit Neural Representations (code coming soon)</a>


## Citation
If you find our work useful in your research, please consider citing:

	@inproceedings{zheng2022imface,
	  title={ImFace: A Nonlinear 3D Morphable Face Model with Implicit Neural Representations},
	  author={Zheng, Mingwu and Yang, Hongyu and Huang, Di and Chen, Liming},
	  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
	  pages={20343--20352},
	  year={2022}
	}
 	@inproceedings{zheng2023neuface,
	  title={NeuFace: Realistic 3D Neural Face Rendering from Multi-view Images},
	  author={Zheng, Mingwu and Zhang, Haiyu and Yang, Hongyu and Huang, Di},
	  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
	  pages={16868--16877},
	  year={2023}
	}
	 @article{zheng2023imface++,
	  title={ImFace++: A Sophisticated Nonlinear 3D Morphable Face Model with Implicit Neural Representations},
	  author={Zheng, Mingwu and Zhang, Haiyu and Yang, Hongyu and Chen, Liming and Huang, Di},
	  journal={arXiv preprint arXiv:2312.04028},
	  year={2023}
	}
 	

