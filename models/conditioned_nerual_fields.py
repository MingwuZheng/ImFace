import torch
import torch.nn as nn
from utils import registry
from models.modules import SingleBVPNet, HyperNetwork, positional_encoding, kaiming_leaky_init, frequency_init, \
    first_layer_film_sine_init

CONDITIONED_NEURAL_FIELD = registry.Registry()
COORDINATE_MLP = registry.Registry()

class ConditionedNeuralField(nn.Module):
    def __init__(self, condition_dim, input_dim, out_dim):
        super().__init__()
        self.condition_dim = condition_dim
        self.input_dim = input_dim
        self.out_dim = out_dim

    @property
    def requires_grad(self) -> bool:
        return any([p.requires_grad for p in self.parameters()])

class FiLMLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.layer = nn.Linear(input_dim, hidden_dim)

    def forward(self, x, freq, phase_shift):
        x = self.layer(x)
        freq = freq.unsqueeze(1).expand_as(x)
        phase_shift = phase_shift.unsqueeze(1).expand_as(x)
        return torch.sin(freq * x + phase_shift)


class MappingNetwork(nn.Module):
    def __init__(self, z_dim, map_hidden_dim, map_output_dim, num_map_layers=3):
        super().__init__()
        network = [nn.Sequential(nn.Linear(z_dim, map_hidden_dim), nn.LeakyReLU(0.2, inplace=True))]
        for _ in range(num_map_layers - 1):
            network.append(nn.Sequential(nn.Linear(map_hidden_dim, map_hidden_dim), nn.LeakyReLU(0.2, inplace=True)))
        network.append(nn.Linear(map_hidden_dim, map_output_dim))
        self.network = nn.ModuleList(network)
        self.network.apply(kaiming_leaky_init)
        with torch.no_grad():
            self.network[-1].weight *= 0.25

    def forward(self, z):
        for layer in self.network:
            z = layer(z)
        frequencies_offsets = z
        frequencies = frequencies_offsets[..., :frequencies_offsets.shape[-1] // 2]
        phase_shifts = frequencies_offsets[..., frequencies_offsets.shape[-1] // 2:]
        return frequencies, phase_shifts


@COORDINATE_MLP.register('FILM_SIREN')
class FiLMSiren(nn.Module):
    def __init__(self,
                 code_dim,
                 in_features,
                 out_features,
                 hidden_features=32,
                 num_hidden_layers=3,
                 hyper_hidden_layers=3,
                 hyper_hidden_features=256,
                 **kargs):
        super().__init__()
        self.hidden_features = hidden_features
        net_list = [FiLMLayer(in_features, hidden_features)]
        for _ in range(num_hidden_layers - 1):
            net_list.append(FiLMLayer(hidden_features, hidden_features))
        self.network = nn.ModuleList(net_list)
        self.final_layer = nn.Linear(hidden_features, out_features)

        self.mapping_network = MappingNetwork(code_dim, hyper_hidden_features,
                                              (len(self.network) + 1) * hidden_features * 2,
                                              hyper_hidden_layers)

        self.network.apply(frequency_init(25))
        self.final_layer.apply(frequency_init(25))
        self.network[0].apply(first_layer_film_sine_init)

    def forward(self, input, code=None):
        if code is not None and len(code.size()) == 1:
            code = code.unsqueeze(0)
        if len(input.size()) == 2:
            input = input.unsqueeze(0)
        frequencies, phase_shifts = self.mapping_network(code)
        frequencies = frequencies * 15 + 30
        x = input
        for index, layer in enumerate(self.network):
            start = index * self.hidden_features
            end = (index + 1) * self.hidden_features
            x = layer(x, frequencies[..., start:end], phase_shifts[..., start:end])
        return self.final_layer(x)


@COORDINATE_MLP.register('HYPER_SIREN')
class HyperSiren(nn.Module):
    def __init__(self,
                 code_dim,
                 in_features,
                 out_features,
                 hidden_features=32,
                 num_hidden_layers=3,
                 hyper_hidden_layers=1,
                 hyper_hidden_features=32,
                 model_type='sine',
                 pe=True,
                 num_encoding_functions=6,
                 sine_omega=30,
                 hyper_type='relu'):
        super().__init__()
        self.pe = pe
        self.num_encoding_functions = num_encoding_functions
        in_feature_dim = (in_features + 6 * self.num_encoding_functions) if self.pe else in_features
        self.mini_net = SingleBVPNet(
            type=model_type,
            hidden_features=hidden_features,
            num_hidden_layers=num_hidden_layers,
            in_features=(in_features + 6 * self.num_encoding_functions) if self.pe else in_features,
            out_features=out_features,
            sine_omega=sine_omega
        )
        # self.mini_net = FCBlock(nonlinearity=model_type,
        #                         hidden_features=hidden_features,
        #                         num_hidden_layers=num_hidden_layers,
        #                         in_features=in_feature_dim,
        #                         out_features=out_features,
        #                         outermost_linear=True)
        if code_dim is not None:
            self.mini_hypernet = HyperNetwork(hyper_in_features=code_dim,
                                              hyper_hidden_layers=hyper_hidden_layers,
                                              hyper_hidden_features=hyper_hidden_features,
                                              hypo_module=self.mini_net,
                                              hyper_type=hyper_type)

    def forward(self, x, code=None):
        """
        :param x: (B,N,D)
        :param code: (B,N)
        """
        if self.pe:
            x = positional_encoding(x, num_encoding_functions=self.num_encoding_functions)
        if code is None:
            return self.mini_net({'coords': x})['model_out']
        else:
            return self.mini_net({'coords': x}, params=self.mini_hypernet(code))['model_out']


class FusionNet(nn.Module):
    def __init__(self, fusion_number, condition_dim, weight_feature_dim=128):
        super().__init__()
        self.fusion_number = fusion_number
        self.weights = nn.Sequential( \
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
        weights = self.weights(condition)  # (B,N,K)
        return torch.sum(weights[..., None] * inputs, dim=2), weights  # (B,N,O)


@CONDITIONED_NEURAL_FIELD.register('NEURAL_BLEND_FIELD_IMFACE++')
class NeuralBlendFieldCVPR(ConditionedNeuralField):
    def __init__(self, model_cfg, condition_dim):
        super().__init__(condition_dim, model_cfg.INPUT_DIM, model_cfg.OUT_DIM)
        assert model_cfg.INPUT_DIM == 3
        self.cfg = model_cfg
        self.kpt_num = model_cfg.KPT_NUM
        mini_nets = []
        self.local_dim = self.condition_dim // self.kpt_num if self.condition_dim is not None else self.condition_dim
        for _ in range(self.kpt_num):
            mini_net = COORDINATE_MLP[model_cfg.MLP](self.local_dim,
                                                     in_features=self.input_dim,
                                                     out_features=self.out_dim,
                                                     hidden_features=model_cfg.HIDDEN_FEATURES,
                                                     num_hidden_layers=model_cfg.NUM_HIDDEN_LAYERS,
                                                     hyper_hidden_layers=model_cfg.NUM_HYPER_HIDDEN_LAYERS,
                                                     hyper_hidden_features=model_cfg.HYPER_HIDDEN_FEATURES,
                                                     model_type=model_cfg.NONLINEARITY,
                                                     pe=model_cfg.PE,
                                                     num_encoding_functions=model_cfg.NUM_ENCODING_FUNCTIONS,
                                                     sine_omega=model_cfg.OMEGA)
            mini_nets.append(mini_net)
        self.mini_nets = nn.ModuleList(mini_nets)

        self.fusion = FusionNet(self.kpt_num, self.input_dim)

    def forward(self, batch_dict):
        """
        keypoints: (B,K,3) or (K,3)
        xyz: (B,N,3) or (N,3)
        code: (B,D)
        """
        keypoints = batch_dict['KEY_POINTS']
        xyz = batch_dict['XYZ']
        code = batch_dict['CODE']

        if len(xyz.size()) == 2:
            xyz = xyz.unsqueeze(0)
        B, N, _ = xyz.size()
        if len(keypoints.size()) == 2:  # make sure that keypoints is (B,K,3)
            keypoints = keypoints.unsqueeze(0)
            if len(xyz.size()) == 3:
                keypoints = keypoints.repeat(B, 1, 1)
        outputs = []
        for i, mini_net in enumerate(self.mini_nets):
            xyz_relative = xyz - keypoints[:, i].unsqueeze(1).expand_as(xyz)
            local_code = code
            if self.condition_dim is not None:
                local_code = code[:, self.local_dim * i: self.local_dim * (i + 1)]
            outputs.append(mini_net(xyz_relative, local_code)[:, :, None, :])  # (B,N,1,O)
        outputs = torch.cat(outputs, dim=2)  # (B,N,K,O)
        output, weights = self.fusion(xyz, outputs)
        return output, weights
