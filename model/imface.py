import torch, math
from torch import nn

from model import modules, net_utils
from model.modules import HyperNetwork
from model import loss, rigid


class MiniNet(nn.Module):
    def __init__(self, code_dim, in_features, out_features, hidden_features=32, num_hidden_layers=3,
                 hyper_hidden_layers=1, hyper_hidden_features=32, model_type='sine', positional_encoding=True,
                 num_encoding_functions=6):
        super().__init__()
        self.positional_encoding = positional_encoding
        self.num_encoding_functions = num_encoding_functions
        self.mini_net = modules.SingleBVPNet(
            type=model_type,
            hidden_features=hidden_features,
            num_hidden_layers=num_hidden_layers,
            in_features=(in_features + 6 * self.num_encoding_functions) if self.positional_encoding else in_features,
            out_features=out_features
        )
        if code_dim is not None:
            self.mini_hypernet = HyperNetwork(
                hyper_in_features=code_dim,
                hyper_hidden_layers=hyper_hidden_layers,
                hyper_hidden_features=hyper_hidden_features,
                hypo_module=self.mini_net
            )

    def forward(self, x_relative, code=None):
        """
        :param x: (B,N,3*5)
        :param code: (B,N)
        :return: f(x-p)
        """
        # x_relative = x - keypoint.unsqueeze(1).repeat(1, x.size(1), 1)
        if self.positional_encoding:
            x_relative = net_utils.positional_encoding(x_relative, num_encoding_functions=self.num_encoding_functions)
        if code is None:
            return self.mini_net({'coords': x_relative})['model_out']
        else:
            return self.mini_net({'coords': x_relative}, params=self.mini_hypernet(code))['model_out']


class FusionNet(nn.Module):
    def __init__(self, fusion_number, condition_dim, weight_feature_dim=128):
        super().__init__()
        self.fusion_number = fusion_number
        self.weights = nn.Sequential(
            nn.Linear(condition_dim, weight_feature_dim),
            nn.LayerNorm(weight_feature_dim),
            nn.LeakyReLU(inplace=True),
            nn.Linear(weight_feature_dim, weight_feature_dim),
            nn.LayerNorm(weight_feature_dim),
            nn.LeakyReLU(inplace=True),
            nn.Linear(weight_feature_dim, fusion_number),
            nn.LayerNorm(fusion_number),
            nn.Softmax(dim=2)
        )

    def forward(self, condition, inputs):
        """
        :param condition: (B,N,3)
        :param inputs: (B,N,K,O)
        """
        B, N, _ = condition.size()
        weights = self.weights(condition)  # (B,N,K)
        return torch.sum(weights[..., None] * inputs, dim=2)  # (B,N,O)


class MiniNets(nn.Module):
    def __init__(self, embedding_dim, kpt_num, in_features, out_features, hidden_features=128, num_hidden_layers=3,
                 hyper_hidden_layers=1, hyper_hidden_features=128, model_type='sine', positional_encoding=True,
                 num_encoding_functions=6):
        super().__init__()
        self.kpt_num = kpt_num

        self.mini_nets = MiniNet(embedding_dim, in_features=in_features * self.kpt_num,
                                 out_features=out_features * self.kpt_num,
                                 hidden_features=hidden_features, num_hidden_layers=num_hidden_layers,
                                 hyper_hidden_layers=hyper_hidden_layers, hyper_hidden_features=hyper_hidden_features,
                                 model_type=model_type, positional_encoding=positional_encoding,
                                 num_encoding_functions=num_encoding_functions)

        self.fusion = FusionNet(kpt_num, in_features)

    def forward(self, xyz, keypoints, code=None):
        """
        :param keypoints: (B,K,3) or (K,3)
        :param xyz: (B,N,3) or (N,3)
        :param code: (B,D)
        """
        if len(xyz.size()) == 2:
            xyz = xyz.unsqueeze(0)
        B, N, _ = xyz.size()
        if len(keypoints.size()) == 2:
            keypoints = keypoints.unsqueeze(0)
            if len(xyz.size()) == 3:
                keypoints = keypoints.repeat(B, 1, 1)

        k_xyz = xyz.unsqueeze(-2) - keypoints.unsqueeze(1).expand(-1, N, -1, -1)  # (B, N, K, 3)
        k_xyz = k_xyz.flatten(start_dim=2)  # (B, N, K*3)
        outputs = self.mini_nets(k_xyz, code).view(B, N, keypoints.size(1), -1)
        output = self.fusion(xyz, outputs)
        return output


def warp(xyz, deformation, warp_type):
    """
    :param xyz: (B,N,3)
    :param deformation: (B,N,6)/(B,N,3)
    :param warp_type: 'translation' or 'se3'
    :return: warped xyz
    """
    if len(xyz.size()) == 2:
        xyz = xyz.unsqueeze(0)
    if len(deformation.size()) == 2:
        deformation = deformation.unsqueeze(0)

    B, N, _ = xyz.size()
    if warp_type == 'translation':
        return xyz + deformation
    elif warp_type == 'se3':
        w = deformation[:, :, :3]
        v = deformation[:, :, 3:6]
        theta = torch.norm(w, dim=-1)
        w = w / (theta[..., None] + 1e-8)
        v = v / (theta[..., None] + 1e-8)
        screw_axis = torch.cat([w, v], dim=-1)
        transform = rigid.exp_se3(screw_axis, theta)
        warped_points = rigid.from_homogenous(
            (transform @ (rigid.to_homogenous(xyz)[..., None])).squeeze(-1))
        return warped_points
    else:
        raise ValueError


# noinspection DuplicatedCode
class ImFace(nn.Module):
    def __init__(self, config, id_num, exp_num, kpt_num, template_kpts=None, initial_std=0.01):
        super().__init__()
        self.kpt_num = kpt_num
        self.id_dim = config.id_embedding_dim
        self.exp_dim = config.exp_embedding_dim
        self.id_embedding = nn.Embedding(id_num, self.id_dim)
        nn.init.normal_(self.id_embedding.weight, mean=0, std=initial_std)  # std=1/math.sqrt(self.id_dim) or 0.01
        self.exp_embedding = nn.Embedding(id_num * exp_num, self.exp_dim)
        nn.init.normal_(self.exp_embedding.weight, mean=0, std=initial_std)  # std=1/math.sqrt(self.exp_dim) or 0.01

        self.num_encoding_functions = config.num_encoding_functions
        self.warp_type = config.warp_type
        out_features = 3 if config.warp_type == 'translation' else 6
        self.deform_pe = config.deform_pe
        self.reference_pe = config.reference_pe
        self.template_pe = config.template_pe

        self.training_losses = config.training_losses

        # Sparse-Net
        self.landmarks = nn.Sequential(nn.Linear(self.id_dim + self.exp_dim, 256),
                                       nn.LeakyReLU(inplace=True),
                                       nn.Linear(256, 256),
                                       nn.LeakyReLU(inplace=True),
                                       nn.Linear(256, 3 * self.kpt_num))
        self.id_landmarks = nn.Sequential(nn.Linear(self.id_dim, 256),
                                          nn.LeakyReLU(inplace=True),
                                          nn.Linear(256, 256),
                                          nn.LeakyReLU(inplace=True),
                                          nn.Linear(256, 3 * self.kpt_num))

        if template_kpts is not None:
            self.template_landmarks = torch.nn.Parameter(torch.FloatTensor(template_kpts), requires_grad=False)
        else:
            self.template_landmarks = torch.nn.Parameter(torch.zeros((kpt_num, 3)), requires_grad=True)

        # Deform-Net
        self.deform_net = MiniNets(
            self.exp_dim, self.kpt_num, in_features=3, out_features=out_features, hidden_features=config.exp_hidden_dim,
            num_hidden_layers=config.exp_num_hidden_layers, hyper_hidden_layers=config.exp_hyper_hidden_layers,
            hyper_hidden_features=config.exp_hyper_hidden_features, model_type=config.model_type,
            positional_encoding=self.deform_pe, num_encoding_functions=self.num_encoding_functions)

        self.reference_net = MiniNets(
            self.id_dim, self.kpt_num, in_features=3, out_features=out_features + 1,
            hidden_features=config.id_hidden_dim,
            num_hidden_layers=config.id_num_hidden_layers, hyper_hidden_layers=config.id_hyper_hidden_layers,
            hyper_hidden_features=config.id_hyper_hidden_features, model_type=config.model_type,
            positional_encoding=self.reference_pe, num_encoding_functions=self.num_encoding_functions)

        # Template-Net
        self.template_net = MiniNets(
            None, self.kpt_num, in_features=3, out_features=1, hidden_features=config.id_hidden_dim,
            num_hidden_layers=config.id_num_hidden_layers, model_type=config.model_type,
            positional_encoding=self.template_pe, num_encoding_functions=self.num_encoding_functions)

    def get_landmarks(self, exp_embedding, id_embedding, int_idx=True):
        if int_idx:
            exp_embedding = self.exp_embedding(exp_embedding.long())
            id_embedding = self.id_embedding(id_embedding.long())

        landmarks_ori = self.landmarks(torch.cat((exp_embedding, id_embedding))).view(self.kpt_num, 3)
        landmarks_id = self.id_landmarks(id_embedding).view(self.kpt_num, 3)

        return landmarks_ori, landmarks_id, self.template_landmarks

    def get_template_coords(self, sample_subset, exp_embedding, id_embedding, int_idx=True):
        coords_dict = self.inference(sample_subset, exp_embedding, id_embedding, int_idx=int_idx, return_coords=True)
        refered_coords = coords_dict[-1]['refered_coords']
        return refered_coords

    def hook_weight(self, xyz, seg=0):
        if seg == 0:
            return self.deform_net.fusion.weights(xyz)  # NxK
        elif seg == 1:
            return self.reference_net.fusion.weights(xyz)  # NxK
        elif seg == 2:
            return self.template_net.fusion.weights(xyz)  # NxK
        else:
            print('Segment number must be 0, 1 or 2!')
            raise ValueError

    def inference_by_batch(self, coords, exp_embedding, id_embedding, device, points_per_inference=163840, int_idx=True,
                           return_fix=False, return_coords=False):
        batch_split = math.ceil(coords.shape[0] / points_per_inference)
        sdf_batch = []
        fix_batch = []
        coords_dict = {'deformed_coords': [], 'refered_coords': []}
        grid_tensor = torch.chunk(coords.to(torch.float32), batch_split)
        for i in range(batch_split):
            infer_split = self.inference(
                grid_tensor[i].to(device),
                exp_embedding,
                id_embedding,
                int_idx=int_idx,
                return_fix=return_fix,
                return_coords=return_coords
            )
            sdf_split = infer_split[0]
            if return_fix:
                fix_split = infer_split[1]
                fix_batch.append(fix_split.flatten().detach().cpu())
            if return_coords:
                cur_coords_dict = infer_split[-1]
                for key in cur_coords_dict:
                    coords_dict[key].append(cur_coords_dict[key])
            sdf_batch.append(sdf_split.flatten().detach().cpu())

        if return_fix:
            result = (torch.cat(sdf_batch, dim=0).numpy(), torch.cat(fix_batch, dim=0).numpy())
        else:
            result = (torch.cat(sdf_batch, dim=0).numpy(),)
        if return_coords:
            coords_dict = {key: torch.cat(coords_dict[key], dim=0) for key in coords_dict}
            result += (coords_dict,)
        return result

    def inference(self, coords_ori, exp_embedding, id_embedding, int_idx=True, return_fix=False, return_coords=False):
        if int_idx:
            if exp_embedding is not None:
                exp_embedding_ori = torch.clone(exp_embedding)
                exp_embedding = exp_embedding * self.id_embedding.num_embeddings + id_embedding
                exp_embedding = self.exp_embedding(exp_embedding.long())
            if id_embedding is not None:
                id_embedding = self.id_embedding(id_embedding.long())

        if exp_embedding is not None:
            assert id_embedding is not None
            landmarks_ori = self.landmarks(torch.cat((exp_embedding, id_embedding))).view(self.kpt_num, 3)
            deformation_exp = self.deform_net(coords_ori, landmarks_ori, exp_embedding)
            deformed_coords = warp(coords_ori, deformation_exp, self.warp_type)
            if int_idx and exp_embedding_ori == 0:
                deformed_coords = coords_ori
        else:
            deformed_coords = coords_ori

        if id_embedding is not None:
            landmarks_id = self.id_landmarks(id_embedding).view(self.kpt_num, 3)
            reference_output = self.reference_net(deformed_coords, landmarks_id, id_embedding)
            deformation_ref = reference_output[:, :, :-1]
            correction = reference_output[:, :, -1:]
            # refered_coords = deformed_coords + deformation_ref
            refered_coords = warp(deformed_coords, deformation_ref, self.warp_type)
        else:
            refered_coords = deformed_coords
            correction = 0

        sdf_template = self.template_net(refered_coords,
                                         self.template_landmarks[torch.LongTensor([36, 45, 30, 48, 54]), :])

        if return_fix:
            results = (sdf_template, correction)
        else:
            results = (sdf_template + correction,)
        if return_coords:
            results += ({'deformed_coords': deformed_coords, 'refered_coords': refered_coords},)
        return results

    def forward(self, input_dict, exp_code=None, id_code=None):
        coords_ori = input_dict['xyz']  # (B,N,3)
        exp_idx_ori = input_dict['exp'].long()  # (B,)
        id_idx = input_dict['id'].long()  # (B,)
        key_pts = input_dict['key_pts']  # (B,K,3)
        
        exp_idx = exp_idx_ori * self.id_embedding.num_embeddings + id_idx

        if exp_code is None:
            exp_code = self.exp_embedding(exp_idx)
        if id_code is None:
            id_code = self.id_embedding(id_idx)

        B, K, _ = key_pts.shape

        landmarks_ori = self.landmarks(torch.cat((exp_code, id_code), dim=1)).view(B, self.kpt_num, 3)
        deformation_exp = self.deform_net(coords_ori, landmarks_ori, exp_code)

        deformed_coords = warp(coords_ori, deformation_exp, self.warp_type)

        distangle_error = torch.zeros_like(deformation_exp)
        distangle_error[torch.where(exp_idx_ori == 0)] = deformation_exp[torch.where(exp_idx_ori == 0)]

        deformation_exp_landmarks = self.deform_net(key_pts, landmarks_ori, exp_code)
        deformed_landmarks = warp(key_pts, deformation_exp_landmarks, self.warp_type)

        landmarks_id = self.id_landmarks(id_code).view(B, self.kpt_num, 3)
        reference_output = self.reference_net(deformed_coords, landmarks_id, id_code)
        deformation_ref = reference_output[:, :, :-1]
        correction = reference_output[:, :, -1:]

        reference_output_landmarks = self.reference_net(deformed_landmarks, landmarks_id, id_code)
        deformation_ref_landmarks = reference_output_landmarks[:, :, :-1]

        refered_landmarks = warp(deformed_landmarks, deformation_ref_landmarks, self.warp_type)
        refered_coords = warp(deformed_coords, deformation_ref, self.warp_type)

        sdf_template = self.template_net(refered_coords,
                                         self.template_landmarks[torch.LongTensor([36, 45, 30, 48, 54]), :])
        sdf = sdf_template + correction

        net_input = {'coords': coords_ori, 'gt_sdf': input_dict['gt_sdf'], 'gt_normals': input_dict['grad'],
                     'id_code': id_code, 'exp_code': exp_code, 'key_pts': key_pts,
                     'avg_key_pts': input_dict['key_pts_nu'], 'all_key_pts': self.template_landmarks}
        net_middle = {'landmarks': landmarks_ori, 'landmarks_id': landmarks_id,
                      'deformed_landmarks': deformed_landmarks, 'distangle_error': distangle_error,
                      'deformation_exp': deformation_exp, 'deformed_coords': deformed_coords,
                      'deformation_ref': deformation_ref, 'refered_coords': refered_coords,
                      'refered_landmarks': refered_landmarks, 'template_landmarks': self.template_landmarks,
                      'sdf_template': sdf_template.squeeze(), 'correction': correction.squeeze()}
        net_out = {'sdf_out': sdf.squeeze()}

        return loss.lif_loss(net_input, net_middle, net_out, self.training_losses)