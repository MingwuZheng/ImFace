import torch
import torch.nn.functional as F
from models import diff_opts
import bisect
import numpy as np

def _get_loss_weight(loss_config, name, progress):
    if name not in loss_config.keys():
        return 0
    weight = getattr(loss_config, name)
    _phase_progress = {}
    _phase_value = {}
    _phase_progress[name] = [values[0] for values in weight]
    _phase_value[name] = [values[1] for values in weight]

    _phase = bisect.bisect_left(_phase_progress[name], progress)
    if _phase >= len(_phase_progress[name]):
        return _phase_value[name][_phase-1]

    if _phase > 0:
        # cosine anealing
        v0 = _phase_value[name][_phase-1]
        p1p0 = (_phase_progress[name][_phase] - _phase_progress[name][_phase-1])
        pp0 = progress-_phase_progress[name][_phase-1]
        assert(pp0/p1p0 <= 1.0 and pp0/p1p0 >= 0.0)
        v1v0 = (_phase_value[name][_phase] - _phase_value[name][_phase-1])
        return v0 + (1-np.cos(np.pi*pp0/p1p0)) * v1v0 / 2

    return _phase_value[name][_phase]

def imface_sdf_loss(net_input, net_middle, net_out, losses, batch_num):
    landmark_dim = losses["LANDMARK_DIM"]
    kpts_5_idx = torch.LongTensor([36, 45, 30, 48, 54])
    coords = net_input['coords']

    pred_sdf = net_out['sdf']
    B, N = pred_sdf.shape
    total_losses = {}
    progress = net_input['progress']
    sdf_loss = _get_loss_weight(losses, 'SDF_LOSS_PROGRESS', progress)
    if sdf_loss > 0:
        gt_sdf = net_input['gt_sdf']
        sdf_constraint = pred_sdf[:, :batch_num] - gt_sdf
        total_losses['SDF_LOSS'] = torch.abs(sdf_constraint).mean() * sdf_loss

    normal_loss_2 = _get_loss_weight(losses, 'NORMAL_LOSS_PROGRESS', progress)
    eikonal_obs_loss = _get_loss_weight(losses, 'EIKONAL_LOSS_OBSERVATION_PROGRESS', progress)
    eikonal_can_loss = _get_loss_weight(losses, 'EIKONAL_LOSS_CANONICAL_PROGRESS', progress)
    if normal_loss_2 > 0:
        if 'NORMAL_LOSS' in losses.keys():
            gt_normals = net_input['gt_normals']
            gradient_obs = net_out['gradient_obs']
            normal_loss = 1 - F.cosine_similarity(gradient_obs[:, :batch_num, :], gt_normals, dim=-1)
            total_losses['NORMAL_LOSS'] = normal_loss.mean() * normal_loss_2
        if 'EIKONAL_LOSS_OBSERVATION' in losses.keys():
            gradient_obs = net_out['gradient_obs']
            eikonal_obs = torch.abs(gradient_obs.norm(dim=-1) - 1)
            total_losses['EIKONAL_LOSS_OBSERVATION'] = eikonal_obs.mean() * eikonal_obs_loss
        if 'EIKONAL_LOSS_CANONICAL' in losses.keys():
            gradient_can = net_out['gradient_can']
            eikonal_can = torch.abs(gradient_can.norm(dim=-1) - 1)
            total_losses['EIKONAL_LOSS_CANONICAL'] = eikonal_can.mean() * eikonal_can_loss

    id_embed_loss = _get_loss_weight(losses, 'ID_EMBEDDING_REGULARIZATION_PROGRESS', progress)
    if id_embed_loss > 0:
        id_embedding = net_input['id_code']
        id_emb_loss = torch.mean(id_embedding**2)
        total_losses['ID_EMBEDDING_REGULARIZATION'] = id_emb_loss * id_embed_loss

    exp_embed_loss = _get_loss_weight(losses, 'EXP_EMBEDDING_REGULARIZATION_PROGRESS', progress)
    if exp_embed_loss > 0:
        exp_embedding = net_input['exp_code']
        exp_emb_loss = torch.mean(exp_embedding**2)
        total_losses['EXP_EMBEDDING_REGULARIZATION'] = exp_emb_loss * exp_embed_loss

    sdf_cor_loss = _get_loss_weight(losses, 'SDF_CORRECTION_CONSTRAINT_PROGRESS', progress)
    if sdf_cor_loss > 0:
        sdf_correct = net_middle['correction']
        sdf_correct_constraint = torch.abs(sdf_correct)
        total_losses['SDF_CORRECTION_CONSTRAINT'] = sdf_correct_constraint.mean() * sdf_cor_loss

    exp_gen_loss = _get_loss_weight(losses, 'EXP_LANDMARK_GENERATION_LOSS_PROGRESS', progress)
    if exp_gen_loss > 0:
        key_pts_exp = net_input['key_pts_exp_68']
        if landmark_dim == 5:
            key_pts_exp = key_pts_exp[:, kpts_5_idx, :]
        pred_exp_landmarks = net_middle['pred_exp_landmarks']
        total_losses['EXP_LANDMARK_GENERATION_LOSS'] = F.l1_loss(pred_exp_landmarks,
                                                                 key_pts_exp) * exp_gen_loss
    id_gen_loss = _get_loss_weight(losses, 'ID_LANDMARK_GENERATION_LOSS_PROGRESS', progress)
    if id_gen_loss > 0:
        pred_id_landmarks = net_middle['pred_id_landmarks']
        key_pts_id = net_input['key_pts_id_68']
        if landmark_dim == 5:
            key_pts_id = key_pts_id[:, kpts_5_idx, :]
        total_losses['ID_LANDMARK_GENERATION_LOSS'] = F.l1_loss(pred_id_landmarks,
                                                                key_pts_id) * id_gen_loss

    exp_con_loss = _get_loss_weight(losses, 'EXP_LANDMARK_CONSISTENCY_LOSS_PROGRESS', progress)
    if exp_con_loss > 0:
        canonical_landmarks = net_middle['canonical_landmarks']
        key_pts_id = net_input['key_pts_id']
        total_losses['EXP_LANDMARK_CONSISTENCY_LOSS'] = F.l1_loss(canonical_landmarks,
                                                                  key_pts_id) * exp_con_loss

    id_con_loss = _get_loss_weight(losses, 'ID_LANDMARK_CONSISTENCY_LOSS_PROGRESS', progress)
    if id_con_loss > 0:
        template_landmarks = net_middle['template_landmarks']
        key_pts_temp = net_input['key_pts_temp']
        total_losses['ID_LANDMARK_CONSISTENCY_LOSS'] = F.l1_loss(template_landmarks,
                                                                 key_pts_temp) * id_con_loss

    dis_loss = _get_loss_weight(losses, 'DISTANGLE_CONSTRAINT_PROGRESS', progress)
    if dis_loss > 0:
        total_losses['DISTANGLE_CONSTRAINT'] = torch.mean(net_middle['distangle_error']**2) * dis_loss

    re_sdf = _get_loss_weight(losses, 'RESIDUAL_SDF_PROGRESS', progress)
    if re_sdf > 0:
        gt_sdf = net_input['gt_sdf']
        sdf_constraint = net_out['residual_sdf'][:, :batch_num] - gt_sdf
        total_losses['RESIDUAL_SDF_LOSS'] = torch.abs(sdf_constraint).mean() * re_sdf

    re_normal = _get_loss_weight(losses, 'RESIDUAL_NORMAL_LOSS_PROGRESS', progress)
    re_eikonal_obs = _get_loss_weight(losses, 'RESIDUAL_EIKONAL_LOSS_OBSERVATION_PROGRESS', progress)
    re_eikonal_can = _get_loss_weight(losses, 'RESIDUAL_EIKONAL_LOSS_CANONICAL_PROGRESS', progress)
    if re_normal > 0:
        gradient_obs, gradient_can = diff_opts.gradients(net_out['residual_sdf'], [net_out['residual_observe_coords'], net_out['residual_canonical_coords']])
        if 'RESIDUAL_NORMAL_LOSS' in losses.keys():
            gt_normals = net_input['gt_normals']
            normal_loss = 1 - F.cosine_similarity(gradient_obs[:, :batch_num, :], gt_normals, dim=-1)
            total_losses['RESIDUAL_NORMAL_LOSS'] = normal_loss.mean() * re_normal
        if 'RESIDUAL_EIKONAL_LOSS_OBSERVATION' in losses.keys():
            eikonal_obs = torch.abs(gradient_obs.norm(dim=-1) - 1)
            total_losses['RESIDUAL_EIKONAL_LOSS_OBSERVATION'] = eikonal_obs.mean() * re_eikonal_obs
        if 'RESIDUAL_EIKONAL_LOSS_CANONICAL' in losses.keys():
            eikonal_can = torch.abs(gradient_can.norm(dim=-1) - 1)
            total_losses['RESIDUAL_EIKONAL_LOSS_CANONICAL'] = eikonal_can.mean() * re_eikonal_can

    detail_embed_loss = _get_loss_weight(losses, 'DETAIL_EMBEDDING_REGULARIZATION_PROGRESS', progress)
    if detail_embed_loss > 0 and progress > 0.2:
        detail_embedding = net_input['detail_code']
        detail_emb_loss = torch.mean(detail_embedding ** 2)
        total_losses['DETAIL_EMBEDDING_REGULARIZATION'] = detail_emb_loss * detail_embed_loss

    return total_losses
