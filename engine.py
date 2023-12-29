import torch, os
from tqdm import tqdm
import utils.misc as utils
import torch.utils.data
import torch.optim.lr_scheduler
import bisect
import numpy as np

def write_tensorboard(train_losses, writer, epoch):
    embedding_loss = {}
    eikonal_loss = {}
    ldmk_gen_loss = {}
    ldmk_consist_loss = {}
    total_loss = 0.
    for loss_name in train_losses.keys():
        single_loss = train_losses[loss_name]
        total_loss += single_loss
        if 'EMBEDDING' in loss_name:
            embedding_loss[loss_name] = single_loss
        elif 'EIKONAL' in loss_name:
            eikonal_loss[loss_name] = single_loss
        elif 'LANDMARK_GENERATION' in loss_name:
            ldmk_gen_loss[loss_name] = single_loss
        elif 'LANDMARK_CONSISTENCY' in loss_name:
            ldmk_consist_loss[loss_name] = single_loss
        else:
            writer.add_scalar(loss_name, single_loss, epoch)
    writer.add_scalars('EMBEDDING_LOSSES', embedding_loss, epoch)
    writer.add_scalars('EIKONAL_LOSSES', eikonal_loss, epoch)
    writer.add_scalars('LANDMARK_GENERATION_LOSSES', ldmk_gen_loss, epoch)
    writer.add_scalars('LANDMARK_CONSISTENCY_LOSSES', ldmk_consist_loss, epoch)
    writer.add_scalar('TOTAL_LOSS', total_loss, epoch)
    return total_loss

def train_one_epoch(model, dataloader, optimizer, scaler, epoch, writer, amp, progress):
    model.train()
    train_losses = {}
    iterations = 0.
    for data_dict, _ in dataloader:
        iterations += 1
        data_dict = utils.dict_to(data_dict, utils.get_rank(), non_blocking=True)
        data_dict['XYZ'].requires_grad_(True)
        data_dict['progress'] = progress
        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=amp):
            losses = model(data_dict)
            train_loss = 0.
            for loss_name, loss_val in losses.items():
                loss = loss_val.mean()
                if loss_name in train_losses.keys():
                    train_losses[loss_name] += loss
                else:
                    train_losses[loss_name] = loss
                train_loss += loss
        if amp:
            scaler.scale(train_loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            train_loss.backward()
            optimizer.step()
    torch.cuda.synchronize()
    train_losses = utils.reduce_loss_dict(train_losses)

    if utils.is_main_process():
        for loss_name, loss_val in train_losses.items():
            train_losses[loss_name] /= iterations
        total_loss = write_tensorboard(train_losses, writer, epoch)
        writer.add_scalar('EMBEB_LR', optimizer.param_groups[0]["lr"], epoch)
        return total_loss
    return 0

def _get_lr_lambdas(param_group2, epochs, opt):
        """
        If lr_factor == -1, nothing is done,
        Args:
            progress:
        """
        lr_lambdas = []
        _phase_progress = {key: [values[0] for values in opt.get(key, [[0, 1.0], [1, 0.01],])] for key in
                                param_group2}
        _phase_lr_factor = {key: [values[1] for values in opt.get(key, [[0, 1.0], [1, 0.01],])] for key in
                                param_group2}
        for name in param_group2:
            def func(epoch, name=name):
                progress = epoch / float(epochs)
                if name == 'detail_embeddings' and progress <= 0.2:
                    return 0
                _phase = bisect.bisect_left(_phase_progress[name], progress)
                if _phase >= len(_phase_progress[name]):
                    return  _phase_lr_factor[name][-1]

                if _phase > 0:
                    # cosine anealing
                    v0 = _phase_lr_factor[name][_phase-1]
                    p1p0 = (_phase_progress[name][_phase] - _phase_progress[name][_phase-1])
                    pp0 = progress - _phase_progress[name][_phase-1]
                    v1v0 = (_phase_lr_factor[name][_phase] - _phase_lr_factor[name][_phase-1])
                    return v0 + (1-np.cos(np.pi*pp0/p1p0)) * v1v0 / 2
                    # return v0 + pp0/p1p0*v1v0

                return _phase_lr_factor[name][_phase]
            lr_lambdas.append(func)
        return lr_lambdas

def _train_phase(model, scheduler, _param_groups):
    """ Set requires_grad to True/False based on give phase """
    for name, lr in dict(zip(_param_groups, scheduler.get_lr())).items():
        if lr == 0:
            getattr(model, name).requires_grad_(False)
            # print(f'Froze parameters of {name}.')
        elif lr > 0:
            getattr(model, name).requires_grad_(True)
            # print(f'Unfroze parameters of {name}.')

def fit(model, fit_dataloader, fit_config, save_path, writer):
    opt_config = fit_config.OPTIMIZATION
    start_epoch = 1
    param_group2 = ['embeddings', 'detail_embeddings']
    optimizer = torch.optim.Adam(lr=float(opt_config.LR), params=[{'params': [p for p in getattr(model, name).parameters()]} for name in param_group2])
    for group in optimizer.param_groups:
        group.setdefault('initial_lr', opt_config.LR)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
                                                  _get_lr_lambdas(param_group2, opt_config.NUM_EPOCHS, opt_config),
                                                  last_epoch=0)

    pth_path = os.path.join(save_path, 'pth')
    os.makedirs(pth_path, exist_ok=True)
    epochs = opt_config.NUM_EPOCHS
    tqdm_bar = tqdm(total=epochs)

    for epoch in range(start_epoch, opt_config.NUM_EPOCHS + 1):

        tqdm_bar.set_description('{} Epoch {:>4d}/{}'.format(save_path.split('/')[-1], epoch, epochs))
        _train_phase(model, scheduler, param_group2)
        progress = (epoch) / float(opt_config.NUM_EPOCHS)
        total_loss = train_one_epoch(model, fit_dataloader, optimizer, None, epoch, writer, False, progress)

        tqdm_bar.set_postfix(loss='{:.5f}'.format(total_loss))
        tqdm_bar.update(1)
        scheduler.step()

        if epoch % opt_config.LOG_INTERVAL == 0:
            envs = {'model': model.state_dict(), 'optimizer': optimizer, 'epoch': epoch}
            utils.save_on_master(envs, os.path.join(pth_path, 'envs_last.pth'))
