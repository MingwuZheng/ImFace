import numpy as np
import warnings
import os, h5py
import torch
from torch.utils.data import Dataset
from glob import glob
import random
from utils import sample
from multiprocessing import Manager
from copy import deepcopy


def get_keypoints(bnds, keypoint_type):
    bnds = np.array(bnds)
    keypoint_types = {
        'corner_9': np.array([36, 39, 42, 45, 30, 31, 35, 48, 54]),
        'corner_11': np.array([36, 39, 42, 45, 31, 30, 35, 48, 51, 54, 57]),
    }
    if keypoint_type == 'full':
        return bnds
    if keypoint_type == 'center_5':
        bnd5 = np.zeros((5, 3))
        bnd5[0] = (bnds[36] + bnds[39]) / 2.
        bnd5[1] = (bnds[42] + bnds[45]) / 2.
        bnd5[2] = bnds[30]
        bnd5[3] = bnds[48]
        bnd5[4] = bnds[54]
        return bnd5
    if keypoint_type == 'center_10':
        bnd10 = np.zeros((10, 3))
        bnd10[0] = (bnds[36] + bnds[39]) / 2.
        bnd10[1] = bnds[27]
        bnd10[2] = (bnds[42] + bnds[45]) / 2.
        bnd10[3] = bnds[3]
        bnd10[4] = bnds[30]
        bnd10[5] = bnds[13]
        bnd10[6] = bnds[48]
        bnd10[7] = bnds[51]
        bnd10[8] = bnds[54]
        return bnd10
    else:
        return bnds[keypoint_types[keypoint_type]]


class NormalDataset(Dataset):
    def __init__(self, root_path, sample_num, sample_func, keypoint_type='full'):
        self.root_path = root_path
        self.keypoint_type = keypoint_type

        self.sample_num = sample_num
        self.sample_func = sample_func
        self.size = 1
        self._get_nu_bnd()
        print('FaceScape dataset initialized.\n')

    def _get_nu_bnd(self):
        bnds = {}
        bnd_array = []
        nu_bnd_file = os.path.join(self.root_path, 'demo.bnd')
        with open(nu_bnd_file, 'r') as bf:
            for line in bf:
                key_point = line.split()
                bnd_array.append([float(key_point[0]), float(key_point[1]), float(key_point[2])])
        bnd_array = get_keypoints(bnd_array, self.keypoint_type)
        bnds[0] = bnd_array
        self.nu_bnds = bnds

    def get_template_kpts(self):
        # template mesh's key points
        template_kpts = np.loadtxt('dataset/Facescape/FacescapeNormal.bnd')
        return torch.tensor(get_keypoints(template_kpts, self.keypoint_type)).float()

    def _load_data(self):
        surf_pcl_file = os.path.join(self.root_path, 'demo_surf_pcl.npy')
        assert os.path.exists(surf_pcl_file), '{} not exist!'.format(surf_pcl_file)
        bnds = []
        with open(surf_pcl_file.replace('_surf_pcl.npy', '.bnd'), 'r') as bf:
            for line in bf:
                key_point = line.split()
                bnds.append([float(key_point[0]), float(key_point[1]), float(key_point[2])])
        bnds = get_keypoints(bnds, self.keypoint_type)

        surf_points = torch.tensor(np.load(surf_pcl_file).astype(np.float32)).float()
        surf_normals = torch.tensor(np.load(surf_pcl_file.replace('_surf_pcl.', '_surf_nor.')).astype(np.float32)).float()
        free_points = torch.tensor(np.load(surf_pcl_file.replace('_surf_pcl.', '_free_pcl.')).astype(np.float32)).float()
        free_points_grad = torch.tensor(np.load(surf_pcl_file.replace('_surf_pcl.', '_free_grd.')).astype(np.float32)).float()
        free_points_sdfs = torch.tensor(np.load(surf_pcl_file.replace('_surf_pcl.', '_free_sdf.')).astype(np.float32)).float()

        points = torch.cat([surf_points, free_points], dim=0)
        sdfs = torch.cat([torch.zeros(len(surf_points)), free_points_sdfs])
        normals = torch.cat([surf_normals, free_points_grad], dim=0)
        p_sdf_grad = torch.cat([points, sdfs.unsqueeze(1), normals], dim=1)

        sample_data = {
            'p_sdf_grad': p_sdf_grad,
            'bnd': torch.tensor(bnds).float(),
            'file': surf_pcl_file
        }

        return sample_data

    def __len__(self):
        return self.size

    def _get_item(self, index):
        sample_data = self._load_data()
        sdf_file = sample_data['file']
        p_sdf_grad = sample_data['p_sdf_grad']
        key_pts = sample_data['bnd']

        samples = self.sample_func(p_sdf_grad, self.sample_num)

        data_dict = {
            'xyz': samples[:, :3], 'gt_sdf': samples[:, 3], 'grad': samples[:, 4:7],
            'exp': 0, 'id': 0, 'key_pts': key_pts,
            'key_pts_nu': torch.tensor(self.nu_bnds[0]).float()
        }
        return data_dict, sdf_file

    def __getitem__(self, index):
        return self._get_item(index)

