import numpy as np
import os, lmdb, pickle, re
from torch.utils.data import Dataset
import torch


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
        bnd5[0] = bnds[36]
        bnd5[1] = bnds[45]
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
    def __init__(self, root_path, meta_info, ids, exps, sample_num, sample_func, keypoint_type='full', id2idx=None,
                 exp2idx=None, cache_size=10000):
        self.root_path = root_path
        self.keypoint_type = keypoint_type
        self.pcls = []
        self.data_paths = meta_info['keys']
        surf_pcl_pattern = re.compile(r'.*/(\d+)/(\d+)_surf_pcl.npy')
        for data_path in self.data_paths.keys():
            match = re.match(surf_pcl_pattern, data_path)
            if match is not None:
                id_name = match.group(1)
                exp_type = int(match.group(2))
                if (id_name in ids) and (exp_type in exps):
                    self.pcls.append(data_path)
        self.lmdb_env = lmdb.open(root_path, readonly=True, lock=False, readahead=False, meminit=False)
        self.size = len(self.pcls)
        ids.sort()
        exps.sort()
        self.ids, self.exps = ids, exps
        self.id_num, self.exp_num = len(ids), len(exps)

        if exp2idx is None:
            self.exp2idx = {}
            for i, exp_type in enumerate(exps):
                self.exp2idx[int(exp_type)] = i
        else:
            self.exp2idx = exp2idx
        if id2idx is None:
            self.id2idx = {}
            for i, id_name in enumerate(ids):
                self.id2idx[int(id_name)] = i
        else:
            self.id2idx = id2idx

        self.cache_size = cache_size
        self.sample_num = sample_num
        self.sample_func = sample_func
        self._get_nu_bnd()
        print('FaceScape LMDB dataset initialized.\n')

    def get_template_kpts(self):
        template_kpts = np.loadtxt('/home/zhengmingwu_2020/ImFace-LIF/dataset/FacescapeNormal.bnd')
        return torch.tensor(get_keypoints(template_kpts, self.keypoint_type)).float()

    def _read_lmdb(self, key):
        with self.lmdb_env.begin(write=False) as txn:
            buf = txn.get(key.encode('ascii'))
        if buf is None:
            return None
        data_flat = np.frombuffer(buf, dtype=np.float32)
        return torch.tensor(data_flat.reshape(self.data_paths[key])).float()

    def _load_data(self, index):

        surf_pcl_file = self.pcls[index]

        bnds = self._read_lmdb(surf_pcl_file.replace('_surf_pcl.npy', '.bnd'))
        bnds = get_keypoints(bnds, self.keypoint_type)

        surf_points = self._read_lmdb(surf_pcl_file)
        surf_normals = self._read_lmdb(surf_pcl_file.replace('_surf_pcl.', '_surf_nor.'))
        free_points = self._read_lmdb(surf_pcl_file.replace('_surf_pcl.', '_free_pcl.'))
        free_points_grad = self._read_lmdb(surf_pcl_file.replace('_surf_pcl.', '_free_grd.'))
        free_points_sdfs = self._read_lmdb(surf_pcl_file.replace('_surf_pcl.', '_free_sdf.'))

        points = torch.cat([surf_points, free_points], dim=0)
        sdfs = torch.cat([torch.zeros(len(surf_points)), free_points_sdfs])
        normals = torch.cat([surf_normals, free_points_grad], dim=0)
        p_sdf_grad = torch.cat([points, sdfs.unsqueeze(1), normals], dim=1)
        # p_sdf_grad = p_sdf_grad[torch.randperm(p_sdf_grad.shape[0])]

        sample_data = {
            'p_sdf_grad': p_sdf_grad,
            'bnd': torch.tensor(bnds).float(),
            'file': surf_pcl_file
        }
        return sample_data

    def __len__(self):
        return self.size

    def _get_nu_bnd(self):
        bnds = {}
        for id_name in self.ids:
            bnd_array = self._read_lmdb('/{}/1.bnd'.format(id_name))
            if bnd_array is None:
                continue
            bnd_array = get_keypoints(bnd_array, self.keypoint_type)
            bnds[id_name] = bnd_array
        self.nu_bnds = bnds

    def _get_item(self, index):
        sample_data = self._load_data(index)
        sdf_file = sample_data['file']
        exp_type = int(os.path.basename(sdf_file).split('.')[0].split('_')[0])
        exp_idx = self.exp2idx[exp_type]
        id_name = int(sdf_file.split('/')[-2])
        id_idx = self.id2idx[id_name]

        p_sdf_grad = sample_data['p_sdf_grad']
        key_pts = sample_data['bnd']

        samples = self.sample_func(p_sdf_grad, self.sample_num)

        data_dict = {
            'xyz': samples[:, :3], 'gt_sdf': samples[:, 3], 'grad': samples[:, 4:7],
            'exp': exp_idx, 'id': id_idx, 'key_pts': key_pts,
            'key_pts_nu': torch.tensor(self.nu_bnds[str(id_name)]).float(),
        }
        return data_dict, sdf_file

    def __getitem__(self, index):
        return self._get_item(index)


if __name__ == '__main__':
    import torch
    import copy
    import time
    from utils import sample, fileio, visualization
    from utils.sample import SAMPLE_FUNCTIONS
    from utils import visualization
    from tqdm import tqdm

    # torch.multiprocessing.set_sharing_strategy('file_system')
    stat = []
    # data_path = '/home/zhengmingwu_2020/ImFace-LIF/dataset/FacescapeNormalMem/FacescapeNormal.lmdb'

    data_path = '/home/zhengmingwu_2020/ImFace-LIF/dataset/FacescapeNormal.lmdb'

    meta_info = pickle.load(open(os.path.join(data_path, 'meta_info.pkl'), 'rb'))
    total_ids = meta_info['ids']
    t0 = time.time()
    train_data = NormalDataset(data_path, meta_info, total_ids[:10],
                               [1, 2, 3, 4, 5], 16384,
                               SAMPLE_FUNCTIONS['random_sample_full'], 'corner_11')
    # quit()
    trainDataLoader = torch.utils.data.DataLoader(train_data, batch_size=20, shuffle=True,
                                                  num_workers=8)
    print('Init takes {}.'.format(time.time() - t0))
    total_t = 0.
    times = 1
    for i in range(times):
        t0 = time.time()
        for data, file in tqdm(trainDataLoader):
            # visualization.show_scatter(data['key_pts'][0])
            pass
        t_i = time.time() - t0
        total_t += t_i
        print('Epoch {} takes {}.'.format(i + 1, t_i))
    print('Avg time {}s.'.format(total_t / times))
    # RAM 0.443s ;FULL 0.626s
