import torch, cv2, os, datetime, shutil, random, trimesh, pickle, sys

root_dir = os.path.abspath(os.path.join(__file__, os.pardir, os.pardir))
sys.path.append(root_dir)

from tqdm import tqdm
from omegaconf import OmegaConf
import argparse
import torch.utils.data
from torch.utils.tensorboard import SummaryWriter
import torch.optim.lr_scheduler
from torch.utils.data import DataLoader
from prefetch_generator import BackgroundGenerator

from utils import fileio, common, summary
from utils.sample import SAMPLE_FUNCTIONS
from model import lif_net_inter
from data_loader.NormalDataset import NormalDataset

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
    train_ids = []
    with open(config.train_list) as f:
        for line in f:
            train_ids.append(line.strip())

    train_data = NormalDataset(config.data_path, train_ids, config.exp_types, config.sample_num,
                               SAMPLE_FUNCTIONS[config.sample_func], config.keypoint_type)

    trainDataLoader = DataLoaderX(train_data, batch_size=config.batch_size, shuffle=True, num_workers=0,
                                  pin_memory=True)

    test_obj_list = random.sample(range(len(train_data)), config.test_obj_num)
    print('Training {} samples with {} identities.\n'.format(len(train_data), len(train_ids)))
    return trainDataLoader, train_data, test_obj_list, len(train_ids)


def clean_result_dirs(config):
    for saves in os.listdir(config.save_path):
        remove = False
        pth_path = os.path.join(config.save_path, saves, 'pth')
        if not os.path.exists(pth_path):
            remove = True
        elif ('envs_last.pth' not in os.listdir(pth_path)) and ('envs.pth' not in os.listdir(pth_path)):
            remove = True
        if remove:
            print('WARNING: REMOVING \'{}\'...\n'.format(os.path.join(config.save_path, saves)))
            shutil.rmtree(os.path.join(config.save_path, saves))


def main(args):
    config_path = os.path.join(root_dir, 'config', args.config)
    config = OmegaConf.load(config_path)

    print(config, '\n')
    if use_cpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        device = 'cpu'
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = config.gpus
        device = 'cuda'
    
    if not os.path.exists(config.save_path):
        os.makedirs(config.save_path)
    # load dataset
    train_dataloader, train_data, test_obj_list, train_id_num = get_dataset(config)

    template_kpts = train_data.get_template_kpts()
    # init model
    decoder = lif_net_inter.LIF(config.network_params, train_id_num, len(config.exp_types), 5,
                             template_kpts)

    # load from checkpoint
    if config.reload:
        reload_path = config.load_path
        timestr = reload_path.split('/')[-1]
        envs = torch.load(os.path.join(reload_path, 'pth', 'envs_last.pth'))
        print('Loaded from \'{}\'.\n'.format(reload_path))
        optimizer, scheduler, batch_cnt = envs['optimizer'], envs['scheduler'], envs['batch_cnt']
        start_epoch = envs['epoch'] + 1
        decoder.load_state_dict(envs['decoder'])
        tensorboard_path = os.path.join(reload_path, 'tensorboard')
    else:
        batch_cnt, start_epoch = 0, 1
        optimizer = torch.optim.Adam(lr=float(config.lr_decoder), params=decoder.parameters())
        if config.lr_scheduler == 'StepLR':
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, config.lr_interval, config.lr_factor)
        elif config.lr_scheduler == 'OneCycleLR':
            scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=config.lr_decoder,
                                                            steps_per_epoch=len(train_dataloader),
                                                            epochs=config.epoch, pct_start=0.05)
        else:
            raise ValueError

        timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
        print('Time stamp:{}.\n'.format(timestr))
        tensorboard_path = os.path.join(config.save_path, timestr, 'tensorboard')
        if not os.path.exists(tensorboard_path):
            os.makedirs(tensorboard_path)
        shutil.copy(config_path, os.path.join(config.save_path, timestr))

    decoder = torch.nn.DataParallel(decoder).to(device)
    work_dir = os.path.join(config.save_path, timestr)
    fileio.write_dict(os.path.join(work_dir, 'id2idx.json'), train_data.id2idx)
    fileio.write_dict(os.path.join(work_dir, 'exp2idx.json'), train_data.exp2idx)

    # tensorboard
    writer = SummaryWriter(tensorboard_path)

    pth_path = os.path.join(work_dir, 'pth')
    common.cond_mkdir(pth_path)
    scaler = torch.cuda.amp.GradScaler(enabled=True)

    with tqdm(total=config.epoch) as tqdm_bar:
        for epoch in range(start_epoch, config.epoch + 1):
            tqdm_bar.set_description('{} Epoch {:>4d}/{}'.format(timestr, epoch, config.epoch))
            decoder.train()

            train_loss_epoch = {}
            for key in config.network_params.training_losses.keys():
                train_loss_epoch[key] = 0.0
            iterations = 0.

            # train
            for data_dict, _ in train_dataloader:
                batch_cnt += 1
                iterations += 1.

                # put data to device
                data_dict = common.dict_to(data_dict, device)
                data_dict['xyz'].requires_grad_(True)

                optimizer.zero_grad()
                with torch.cuda.amp.autocast(enabled=True):
                    losses = decoder(data_dict)
                    train_loss = 0.
                    for loss_name, loss_val in losses.items():
                        single_loss = loss_val.mean()
                        train_loss_epoch[loss_name] += single_loss
                        train_loss += single_loss

                scaler.scale(train_loss).backward()
                scaler.step(optimizer)
                scaler.update()

                # update loss
                tqdm_bar.set_postfix(loss='{:.5f}'.format(train_loss.item()))
                if config.lr_scheduler == 'OneCycleLR':
                    scheduler.step()

            tqdm_bar.update(1)
            if config.lr_scheduler == 'StepLR' and epoch >= config.lr_begin_decay:
                scheduler.step()

            for key in train_loss_epoch.keys():
                train_loss_epoch[key] /= iterations
            write_tensorboard(train_loss_epoch, writer, epoch)

            if epoch % config.log_frequency == 0:
                envs = {'decoder': decoder.module.state_dict(), 'optimizer': optimizer,
                        'batch_cnt': batch_cnt, 'epoch': epoch, 'scheduler': scheduler}
                save_path = os.path.join(pth_path, 'envs_last.pth')
                torch.save(envs, save_path)

                # evaluate
                test_sdf_decoder = decoder.module
                summary.test_decoder_on_trainset_2d(config, test_sdf_decoder, train_data, test_obj_list, device, writer,
                                                    epoch)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="config name.", default="train_lif_facescape.yaml")
    args = parser.parse_args()
    main(args)
