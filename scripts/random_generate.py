import torch, os, datetime, shutil, random, numpy as np, sys, yaml
root_dir = os.path.abspath(os.path.join(__file__, os.pardir, os.pardir))
sys.path.append(root_dir)

from glob import glob
import torch.utils.data
import torch.optim.lr_scheduler
from easydict import EasyDict as edict
from utils import fileio, summary
from models import imface
import argparse

def load_test_config(path):
    with open(path, 'r') as f:
        config = f.read()
    config = edict(yaml.load(config, Loader=yaml.FullLoader))
    train_config = glob(os.path.join(config.LOAD_PATH, '*.yaml'))
    assert len(train_config) == 1
    train_config = train_config[0]
    with open(train_config, 'r') as f:
        train_config = f.read()
    train_config = edict(yaml.load(train_config, Loader=yaml.FullLoader))
    id2idx = fileio.read_dict(os.path.join(config.LOAD_PATH, 'id2idx.json'))
    exp2idx = fileio.read_dict(os.path.join(config.LOAD_PATH, 'exp2idx.json'))
    return config, train_config, id2idx, exp2idx


def test_imface():
    # --------------------- Prepare Environment --------------------- #
    parser = argparse.ArgumentParser()
    parser.add_argument("--generate_type", type=str, required=True, choices=["train", "test"],)
    args = parser.parse_args()

    config_path = os.path.join(root_dir, 'config/test_imface++.yaml')
    test_config, train_config, train_id2idx, train_exp2idx = load_test_config(config_path)

    os.environ["CUDA_VISIBLE_DEVICES"] = test_config.GPUS

    valid_train_data = fileio.read_dict('valid_train_data.json', key2int=False) # train data used to train imface++

    print('Loaded from \'{}\'.\n'.format(test_config.LOAD_PATH))

    work_dir = os.path.join(test_config.LOAD_PATH, 'generate')
    os.makedirs(work_dir, exist_ok=True)

    # ----------------------- Build Test Dataset -----------------------#
    test_id_num = 0
    test_ids = []
    with open(test_config.FIT_LIST) as f:
        for line in f:
            test_ids.append(line.strip())
            test_id_num += 1
    test_exps = test_config.EXP_TYPES
    test_exp_num = len(test_config.EXP_TYPES)

    test_exp2idx = {}
    for i, exp_type in enumerate(test_exps):
        test_exp2idx[int(exp_type)] = i
    test_id2idx = {}
    for i, id_name in enumerate(test_ids):
        test_id2idx[int(id_name)] = i

    # ----------------------- Build Train Dataset -----------------------#
    train_id_num = 0
    train_ids = []
    with open(test_config.TRAIN_LIST) as f:
        for line in f:
            train_ids.append(line.strip())
            train_id_num += 1
    train_exps = test_config.EXP_TYPES
    train_exp_num = len(test_config.EXP_TYPES)

    # ----------------------- Build Model ----------------------- #
    train_config.DATA_CONFIG.ID_NUM = len(train_id2idx)
    train_config.DATA_CONFIG.EXP_NUM = len(train_exp2idx)
    train_config.DATA_CONFIG.TEMPLATE_KPTS_ALL = torch.zeros((13008, 3))
    train_config.DATA_CONFIG.TEMPLATE_KPTS = torch.zeros((68, 3))

    model = imface.ImFaceSDF(train_config.MODEL, train_config.DATA_CONFIG)
    envs = torch.load(os.path.join(test_config.LOAD_PATH, 'pth', 'envs_last.pth'), map_location='cpu')
    model.load_state_dict(envs['model'], strict=True)

    # ----------------------- Generate Face from Test Dataset ----------------------- #
    if args.generate_type == 'test':
        model.embeddings = imface.Embeddings(test_id_num * test_exp_num,
                                             test_exp_num * test_id_num,
                                             train_config.DATA_CONFIG.ID_DIM,
                                             train_config.DATA_CONFIG.EXP_DIM,
                                             initial_std=test_config.INITIAL_STD,
                                             extend=False)
        model.detail_embeddings = imface.DetailEmbeddings(test_id_num * test_exp_num,
                                             test_exp_num * test_id_num,
                                             train_config.DATA_CONFIG.DETAIL_DIM,
                                             extend=False)

        print(f'Loaded fitting results from {test_config.RESUME}.')
        envs = torch.load(test_config.RESUME, map_location='cpu')
        model.load_state_dict(envs['model'])
        # ----------------------- Get ID & EXP (test dataset)-----------------------#
        while True:
            generate_id = test_ids[random.randint(0, len(test_ids) - 1)]
            generate_exp = test_exps[random.randint(0, test_exp_num - 1)]
            # FaceScape has not provided the dataset for id=662, exp=5
            if generate_exp == 5 and generate_id == 662:
                continue
            break
        print('generate face: id={0} exp={1}......'.format(generate_id, generate_exp))
        input_id = test_id2idx[int(generate_id)] + test_exp2idx[int(generate_exp)] * len(test_id2idx)
        input_exp = input_id
    else:
        # ----------------------- Generate Face from Train Dataset ----------------------- #
        while True:
            generate_id = train_ids[random.randint(0, len(train_ids) - 1)]
            generate_exp = train_exps[random.randint(0, train_exp_num - 1)]
            if valid_train_data['id_{0}_exp_{1}'.format(generate_id, generate_exp)] == 1:
                break
        print('generate face: id={0} exp={1}......'.format(generate_id, generate_exp))
        input_id = train_id2idx[int(generate_id)]
        input_exp = train_exp2idx[int(generate_exp)]
    model.to('cuda')

    obj_save_path = os.path.join(work_dir, str(generate_id))
    os.makedirs(obj_save_path, exist_ok=True)
    obj_path = os.path.join(obj_save_path, str(generate_exp) + '.obj')
    summary.extract_mesh(test_config, model, input_id, input_exp, obj_path, 'cuda', test_config.BATCH_SPLIT)
    print('finish generate {0} face: id={1} exp={2}......\n'.format(args.generate_type, generate_id, generate_exp))
    print('result saved at {0}'.format(obj_path))

if __name__ == '__main__':
    test_imface()