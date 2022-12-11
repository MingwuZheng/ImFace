import torch
import torch.nn.functional as F
from model import diff_opts


################################################
# Loss functions
def laplace_mse(model_output, gt):
    # compute laplacian on the model
    laplace = diff_opts.laplace(model_output['model_out'], model_output['model_in'])
    # compare them with the ground truth
    laplace_loss = torch.mean((laplace - gt['laplace']) ** 2)
    return {'laplace_loss': laplace_loss}


def latent_loss(model_output):
    return torch.mean(model_output['latent_vec'] ** 2)


def hypo_weight_loss(model_output):
    weight_sum = 0
    total_weights = 0

    for weight in model_output['hypo_params'].values():
        weight_sum += torch.sum(weight ** 2)
        total_weights += weight.numel()

    return weight_sum * (1 / total_weights)


def gradients_mse(model_output, gt):
    # compute gradients on the model
    gradients = diff_opts.gradient(model_output['model_out'], model_output['model_in'])
    # compare them with the ground-truth
    gradients_loss = torch.mean((gradients - gt['gradients']).pow(2).sum(-1))
    return {'gradients_loss': gradients_loss}


def lif_loss(net_input, net_middle, net_out, losses):
    kpts_5_idx = torch.LongTensor([36, 45, 30, 48, 54])
    gt_sdf = net_input['gt_sdf']
    gt_normals = net_input['gt_normals']
    id_embedding = net_input['id_code']
    exp_embedding = net_input['exp_code']

    # Deform-Nets
    coords = net_input['coords']  # First Input
    deformation_exp = net_middle['deformation_exp']  # First Output

    # Reference-Nets
    deformed_coords = net_middle['deformed_coords']  # Second Input
    deformation_ref = net_middle['deformation_ref']  # Second Output 1
    sdf_correct = net_middle['correction']  # Second Output 2

    # Template-Nets
    refered_coords = net_middle['refered_coords']  # Third Input
    sdf_template = net_middle['sdf_template']  # Third Output
    pred_sdf = net_out['sdf_out']  # Final Output

    total_losses = {}

    if 'sdf' in losses.keys():
        sdf_constraint = pred_sdf - gt_sdf
        total_losses['sdf'] = torch.abs(sdf_constraint).mean() * losses['sdf']

    if ('normal_constraint' in losses.keys()) or ('eikonal_constraint_drt' in losses.keys()):
        gradient_drt, gradient_rt = diff_opts.gradients(pred_sdf, [coords, deformed_coords])
        # gradient_drt, gradient_rt, gradient_t = diff_opts.gradients(pred_sdf, [coords, deformed_coords, refered_coords])
        if 'normal_constraint' in losses.keys():
            normal_constraint = 1 - F.cosine_similarity(gradient_drt, gt_normals, dim=-1)
            total_losses['normal_constraint'] = normal_constraint.mean() * losses['normal_constraint']
        if 'eikonal_constraint_drt' in losses.keys():
            eikonal_constraint = torch.abs(gradient_drt.norm(dim=-1) - 1)
            total_losses['eikonal_constraint_drt'] = eikonal_constraint.mean() * losses['eikonal_constraint_drt']
        if 'eikonal_constraint_rt' in losses.keys():
            eikonal_constraint_rt = torch.abs(gradient_rt.norm(dim=-1) - 1)
            total_losses['eikonal_constraint_rt'] = eikonal_constraint_rt.mean() * losses['eikonal_constraint_rt']

    if 'id_embeddings_constraint' in losses.keys():
        id_embeddings_constraint = torch.mean(id_embedding ** 2)
        total_losses['id_embeddings_constraint'] = id_embeddings_constraint * losses['id_embeddings_constraint']
    if 'exp_embeddings_constraint' in losses.keys():
        exp_embeddings_constraint = torch.mean(exp_embedding ** 2)
        total_losses['exp_embeddings_constraint'] = exp_embeddings_constraint * losses['exp_embeddings_constraint']

    if 'sdf_correction' in losses.keys():
        sdf_correct_constraint = torch.abs(sdf_correct)
        total_losses['sdf_correction'] = sdf_correct_constraint.mean() * losses['sdf_correction']

    # if 'keypoints_gen_template_constraint' in losses.keys():
    #     template_landmarks = net_middle['template_landmarks']
    #     refered_landmarks = net_input['all_key_pts'][:, kpts_5_idx, :]
    #     # template_landmarks_batch = template_landmarks.unsqueeze(0).repeat(refered_landmarks.size(0), 1, 1)
    #     total_losses['keypoints_gen_template_constraint'] = F.l1_loss(template_landmarks, refered_landmarks) * losses[
    #         'keypoints_gen_template_constraint']
    if 'keypoints_gen_full_constraint' in losses.keys():
        landmarks_gt = net_input['key_pts'][:, kpts_5_idx, :]
        landmarks_gen = net_middle['landmarks']
        total_losses['keypoints_gen_full_constraint'] = F.l1_loss(landmarks_gen, landmarks_gt) * losses[
            'keypoints_gen_full_constraint']
    if 'keypoints_gen_id_constraint' in losses.keys():
        landmarks_id_gen = net_middle['landmarks_id']
        landmarks_id_gt = net_input['avg_key_pts'][:, kpts_5_idx, :]
        total_losses['keypoints_gen_id_constraint'] = F.l1_loss(landmarks_id_gen, landmarks_id_gt) * losses[
            'keypoints_gen_id_constraint']

    if 'keypoints_deform_constraint' in losses.keys():
        landmarks_deformed = net_middle['deformed_landmarks']
        landmarks_avg_id = net_input['avg_key_pts']
        total_losses['keypoints_deform_constraint'] = F.l1_loss(landmarks_deformed, landmarks_avg_id) * losses[
            'keypoints_deform_constraint']
    if 'keypoints_refer_constraint' in losses.keys():
        landmarks_refered = net_middle['refered_landmarks']
        landmarks_avg_all = net_input['all_key_pts']
        total_losses['keypoints_refer_constraint'] = F.l1_loss(landmarks_refered, landmarks_avg_all) * losses[
            'keypoints_refer_constraint']

    if 'distangle_constraint' in losses.keys():
        total_losses['distangle_constraint'] = torch.mean(net_middle['distangle_error'] ** 2) * losses[
            'distangle_constraint']

    if 'energy_constraint' in losses.keys():
        jac, status = diff_opts.jacobian(refered_coords, coords, True)
        jac_inv = torch.linalg.inv(jac)  # torch.linalg.pinv()
        jac = torch.log(jac)
        jac_inv = torch.log(jac_inv)
        loss_jac = (torch.sum(jac.flatten(start_dim=-2) ** 2, dim=-1) + torch.sum(jac_inv.flatten(start_dim=-2) ** 2,
                                                                                  dim=-1)) / 2
        total_losses['energy_constraint'] = torch.mean(loss_jac)

    return total_losses


def nbs_loss(net_input, net_middle, net_out, losses):
    gt_sdf = net_input['gt_sdf']
    gt_normals = net_input['gt_normals']
    id_embedding = net_input['id_code']
    exp_embedding = net_input['exp_code']

    # Deform-Nets
    coords = net_input['coords']  # First Input

    # Reference-Nets
    deformed_coords = net_middle['deformed_coords']  # Second Input

    # Template-Nets
    refered_coords = net_middle['refered_coords']  # Third Input
    pred_sdf = net_out['sdf_out']  # Final Output

    total_losses = {}

    if 'sdf' in losses.keys():
        sdf_constraint = pred_sdf - gt_sdf
        total_losses['sdf'] = torch.abs(sdf_constraint).mean() * losses['sdf']

    if ('normal_constraint' in losses.keys()) or ('eikonal_constraint_drt' in losses.keys()):
        gradient_drt = diff_opts.gradient(pred_sdf, coords)
        if 'normal_constraint' in losses.keys():
            normal_constraint = 1 - F.cosine_similarity(gradient_drt, gt_normals, dim=-1)
            total_losses['normal_constraint'] = normal_constraint.mean() * losses['normal_constraint']
        if 'eikonal_constraint_drt' in losses.keys():
            eikonal_constraint = torch.abs(gradient_drt.norm(dim=-1) - 1)
            total_losses['eikonal_constraint_drt'] = eikonal_constraint.mean() * losses['eikonal_constraint_drt']

    if 'eikonal_constraint_rt' in losses.keys():
        gradient_rt = diff_opts.gradient(pred_sdf, deformed_coords)
        eikonal_constraint_rt = torch.abs(gradient_rt.norm(dim=-1) - 1)
        total_losses['eikonal_constraint_rt'] = eikonal_constraint_rt.mean() * losses['eikonal_constraint_rt']
    if 'eikonal_constraint_t' in losses.keys():
        gradient_t = diff_opts.gradient(pred_sdf, refered_coords)
        eikonal_constraint_t = torch.abs(gradient_t.norm(dim=-1) - 1)
        total_losses['eikonal_constraint_t'] = eikonal_constraint_t.mean() * losses['eikonal_constraint_t']

    if 'id_embeddings_constraint' in losses.keys():
        id_embeddings_constraint = torch.mean(id_embedding ** 2)
        total_losses['id_embeddings_constraint'] = id_embeddings_constraint * losses['id_embeddings_constraint']
    if 'exp_embeddings_constraint' in losses.keys():
        exp_embeddings_constraint = torch.mean(exp_embedding ** 2)
        total_losses['exp_embeddings_constraint'] = exp_embeddings_constraint * losses['exp_embeddings_constraint']

    if 'keypoints_gen_full_constraint' in losses.keys():
        landmarks_gt = net_input['key_pts']
        landmarks_gen = net_middle['landmarks']
        total_losses['keypoints_gen_full_constraint'] = F.l1_loss(landmarks_gen, landmarks_gt) * losses[
            'keypoints_gen_full_constraint']

    if 'keypoints_deform_constraint' in losses.keys():
        landmarks_deformed = net_middle['deformed_landmarks']
        landmarks_avg_id = net_input['avg_key_pts']
        total_losses['keypoints_deform_constraint'] = F.l1_loss(landmarks_deformed, landmarks_avg_id) * losses[
            'keypoints_deform_constraint']
    if 'keypoints_refer_constraint' in losses.keys():
        landmarks_refered = net_middle['refered_landmarks']
        landmarks_avg_all = net_input['all_key_pts']
        total_losses['keypoints_refer_constraint'] = F.l1_loss(landmarks_refered, landmarks_avg_all) * losses[
            'keypoints_refer_constraint']

    return total_losses
