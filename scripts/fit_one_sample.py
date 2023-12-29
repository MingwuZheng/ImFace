import torch, os, datetime, shutil, random, numpy as np, sys, yaml
root_dir = os.path.abspath(os.path.join(__file__, os.pardir, os.pardir))
sys.path.append(root_dir)
from glob import glob
import torch.utils.data
from torch.utils.tensorboard import SummaryWriter
import torch.optim.lr_scheduler
from easydict import EasyDict as edict
from utils import fileio, summary
from models import imface
import engine
from dataset.OneSampleDataset import NormalDataset

use_cpu = False  # For debug only


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
    # torch.autograd.set_detect_anomaly(True)
    config_path = os.path.join(root_dir, 'config/fit_one_sample.yaml')
    test_config, train_config, id2idx, exp2idx = load_test_config(config_path)

    print(test_config, '\n')
    seed = test_config.SEED
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    print('Loaded from \'{}\'.\n'.format(test_config.LOAD_PATH))

    if test_config.RESUME:
        timestr = str(test_config.RESUME).split('/')[-3]
        work_dir = os.path.join(test_config.LOAD_PATH, 'fit', timestr)
    else:
        timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
        work_dir = os.path.join(test_config.LOAD_PATH, 'fit', timestr)
        tensorboard_path = os.path.join(work_dir, 'tensorboard')
        os.makedirs(tensorboard_path, exist_ok=True)
        shutil.copy(config_path, work_dir)
        writer = SummaryWriter(tensorboard_path)
    print('Time stamp:{}.\n'.format(timestr))

    # ----------------------- Build Dataset -----------------------#
    test_dataset = NormalDataset(test_config.DATA_PATH, train_config.DATA_CONFIG.SAMPLE_NUM)
    test_id_num = 1
    test_exp_num = 1

    train_config.DATA_CONFIG.ID_NUM = len(id2idx)
    train_config.DATA_CONFIG.EXP_NUM = len(exp2idx)
    train_config.DATA_CONFIG.TEMPLATE_KPTS_ALL = torch.zeros((13008, 3))
    train_config.DATA_CONFIG.TEMPLATE_KPTS = torch.zeros((68, 3))
    test_config.LOSS.LANDMARK_DIM = train_config.DATA_CONFIG.LANDMARK_DIM

    test_dataloader = torch.utils.data.DataLoader(test_dataset,
                                                  batch_size=test_config.OPTIMIZATION.BATCH_SIZE,
                                                  num_workers=4,
                                                  pin_memory=True,
                                                  drop_last=False)


    # ----------------------- Build Model ----------------------- #
    model = imface.ImFaceSDF(train_config.MODEL, train_config.DATA_CONFIG)
    envs = torch.load(os.path.join(test_config.LOAD_PATH, 'pth', 'envs_last.pth'), map_location='cpu')
    model.load_state_dict(envs['model'], strict=True)

    # ----------------------- Get Mean Embedding ----------------------- #
    id_embeddings = model.embeddings.id_embeddings.weight.data
    id_mean = torch.mean(id_embeddings, dim=0)
    exp_embeddings = model.embeddings.exp_embeddings.weight.data
    exp_mean = torch.mean(exp_embeddings, dim=0)
    detail_embeddings = model.detail_embeddings.detail_embeddings.weight.data
    detail_mean = torch.mean(detail_embeddings, dim=0)


    model.embeddings = imface.Embeddings(test_id_num * test_exp_num,
                                         test_exp_num * test_id_num,
                                         train_config.DATA_CONFIG.ID_DIM,
                                         train_config.DATA_CONFIG.EXP_DIM,
                                         initial_std=test_config.OPTIMIZATION.INITIAL_STD,
                                         extend=False)
    model.detail_embeddings = imface.DetailEmbeddings(test_id_num * test_exp_num,
                                                      test_exp_num * test_id_num,
                                                      train_config.DATA_CONFIG.DETAIL_DIM,
                                                      extend=False)
    model.embeddings.id_embeddings.weight.data = id_mean.unsqueeze(0).repeat(test_id_num * test_exp_num, 1)
    model.embeddings.exp_embeddings.weight.data = exp_mean.unsqueeze(0).repeat(test_id_num * test_exp_num, 1)
    model.detail_embeddings.detail_embeddings.weight.data = detail_mean.unsqueeze(0).repeat(test_id_num * test_exp_num,
                                                                                            1)
    model.training_losses = test_config.LOSS

    if test_config.RESUME:
        print(f'Loaded fitting results from {test_config.RESUME}.')
        envs = torch.load(test_config.RESUME, map_location='cpu')
        model.load_state_dict(envs['model'])
        model.to('cuda')
    else:
        model.to('cuda')
        engine.fit(model, test_dataloader, test_config, work_dir, writer)

    cd, f = summary.extract_mesh_and_compute_error(test_config, model, 0, 0, os.path.join(work_dir, 'fit_demo.obj'), 'cuda', test_config.BATCH_SPLIT)
    print('Chamfer Distance:{:.4f}mm; F-Score:{:.2f}.'.format(cd, f))

if __name__ == '__main__':
    test_imface()