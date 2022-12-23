import torch, cv2, yaml, os, datetime, shutil, random, trimesh, pickle, sys

root_dir = os.path.abspath(os.path.join(__file__, os.pardir, os.pardir))
sys.path.append(root_dir)

from glob import glob
from tqdm import tqdm
from easydict import EasyDict as edict

import torch.utils.data
from torch.utils.tensorboard import SummaryWriter
import torch.optim.lr_scheduler
from torch.utils.data import DataLoader
from prefetch_generator import BackgroundGenerator
import argparse

from utils import fileio, common, summary
from utils.sample import SAMPLE_FUNCTIONS
from model import imface
from data_loader.OneSampleDataset import NormalDataset

random.seed(42)
use_cpu = False


class DataLoaderX(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())


def write_tensorboard(train_loss_epoch, writer, epoch):
    embedding_loss = {}
    eikonal_loss = {}
    keypoints_loss = {}
    total_loss = 0.
    for loss_name in train_loss_epoch.keys():
        single_loss = train_loss_epoch[loss_name]
        total_loss += single_loss
        loss_name_splits = loss_name.split('_')
        if len(loss_name_splits) == 1:
            writer.add_scalar(loss_name, single_loss, epoch)
        elif loss_name_splits[1] == 'embeddings':
            embedding_loss[loss_name] = single_loss
        elif loss_name_splits[0] == 'eikonal':
            eikonal_loss[loss_name] = single_loss
        elif loss_name_splits[0] == 'keypoints':
            keypoints_loss[loss_name] = single_loss
        else:
            writer.add_scalar(loss_name, single_loss, epoch)
    writer.add_scalars('embedding_losses', embedding_loss, epoch)
    writer.add_scalars('eikonal_losses', eikonal_loss, epoch)
    writer.add_scalars('keypoints_losses', keypoints_loss, epoch)
    writer.add_scalar('total_loss', total_loss, epoch)


def get_dataset(config):
    fit_data = NormalDataset(config.data_path, config.sample_num, SAMPLE_FUNCTIONS[config.sample_func], config.keypoint_type)
    fit_dataloader = DataLoaderX(fit_data, batch_size=config.batch_size, shuffle=True, num_workers=4,
                                 pin_memory=False)

    print('Fitting one samples.\n')
    return fit_dataloader, fit_data


def load_config(path):
    with open(path, 'r') as f:
        config = f.read()
    config = edict(yaml.load(config, Loader=yaml.FullLoader))
    train_config = glob(os.path.join(config.load_path, '*.yaml'))
    assert len(train_config) == 1
    train_config = train_config[0]
    with open(train_config, 'r') as f:
        train_config = f.read()
    train_config = edict(yaml.load(train_config, Loader=yaml.FullLoader))
    id2idx = fileio.read_dict(os.path.join(config.load_path, 'id2idx.json'))
    exp2idx = fileio.read_dict(os.path.join(config.load_path, 'exp2idx.json'))
    return config, train_config, id2idx, exp2idx


def clean_result_dirs(clean_path):
    for saves in os.listdir(clean_path):
        remove = False
        if ('envs_last.pth' not in os.listdir(os.path.join(clean_path, saves))) and (
                'envs.pth' not in os.listdir(os.path.join(clean_path, saves))):
            remove = True
        if remove:
            print('WARNING: REMOVING \'{}\'...\n'.format(os.path.join(clean_path, saves)))
            shutil.rmtree(os.path.join(clean_path, saves))


def main(args):
    config_path = os.path.join(root_dir, 'config', args.config)
    config, train_config, id2idx, exp2idx = load_config(config_path)

    print(config, '\n')
    if use_cpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        device = 'cpu'
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = config.gpus
        device = 'cuda'

    fit_dataloader, fit_data = get_dataset(config)

    template_kpts = fit_data.get_template_kpts()

    decoder = imface.ImFace(config.network_params, len(id2idx), len(exp2idx), 5,
                                template_kpts)

    reload_path = config.load_path
    envs = torch.load(os.path.join(reload_path, 'pth', 'envs_last.pth'))
    decoder.load_state_dict(envs['decoder'])
    decoder = torch.nn.DataParallel(decoder).to(device)
    print('Loaded from \'{}\'.\n'.format(reload_path))

    save_path = os.path.join('demo', 'fit')
    common.cond_mkdir(save_path)
    timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
    work_dir = os.path.join(save_path, timestr)
    clean_result_dirs(save_path)
    common.cond_mkdir(work_dir)
    tensorboard_path = os.path.join(work_dir, 'tensorboard')
    writer = SummaryWriter(tensorboard_path)

    id_embedding = torch.nn.Embedding(1, config.network_params.id_embedding_dim).to(
        device)
    torch.nn.init.normal_(id_embedding.weight, mean=0, std=0.01)  # std=1/math.sqrt(self.id_dim) or 0.01
    exp_embedding = torch.nn.Embedding(1, config.network_params.exp_embedding_dim).to(
        device)
    torch.nn.init.normal_(exp_embedding.weight, mean=0, std=0.01)  # std=1/math.sqrt(self.exp_dim) or 0.01
    optimizer = torch.optim.Adam(
        [
            {
                "params": exp_embedding.parameters(),
                "lr": config.lr_decoder,
            },
            {
                "params": id_embedding.parameters(),
                "lr": config.lr_decoder,
            }
        ]
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, config.lr_interval, config.lr_factor)

    with tqdm(total=config.epoch) as tqdm_bar:
        for epoch in range(1, config.epoch + 1):
            tqdm_bar.set_description('{} Epoch {:>4d}/{}'.format(timestr, epoch, config.epoch))
            decoder.train()
            train_loss_epoch = {}
            for key in config.network_params.training_losses.keys():
                train_loss_epoch[key] = 0.0
            iterations = 0.

            for data_dict, _ in fit_dataloader:
                iterations += 1.
                data_dict = common.dict_to(data_dict, device)
                data_dict['xyz'].requires_grad_(True)

                idx = torch.tensor([0]).to(device)

                losses = decoder(data_dict, exp_embedding(idx), id_embedding(idx))

                train_loss = 0.
                for loss_name, loss_val in losses.items():
                    single_loss = loss_val.mean()
                    train_loss_epoch[loss_name] += single_loss
                    train_loss += single_loss

                optimizer.zero_grad()
                train_loss.backward()
                optimizer.step()

                tqdm_bar.set_postfix(loss='{:.5f}'.format(train_loss.item()))

            tqdm_bar.update(1)
            if epoch >= config.lr_begin_decay:
                scheduler.step()

            for key in train_loss_epoch.keys():
                train_loss_epoch[key] /= iterations
            write_tensorboard(train_loss_epoch, writer, epoch)

        envs = {'exp_embedding': exp_embedding, 'id_embedding': id_embedding}
        save_path = os.path.join(work_dir, 'envs_last.pth')
        torch.save(envs, save_path)

        cd, f, errors = summary.fit_one_decoder(config, decoder.module, fit_data, exp_embedding, id_embedding,
                                            os.path.join(work_dir, 'fit_results'), device, 1)
        print('AvgCamferDist: {:.4f}mm AvgF-Score:{:.2f}'.format(cd, f))
        fileio.write_dict(os.path.join(work_dir, 'errors.json'), errors)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="config path.", default="fit_one_sample.yaml")
    args = parser.parse_args()
    main(args)
