import torch, re
from torch import nn
from torchmeta.modules import (MetaModule, MetaSequential)
import numpy as np
from collections import OrderedDict
import math


def get_subdict(dictionary, key=None):
    if dictionary is None:
        return None
    if (key is None) or (key == ''):
        return dictionary
    key_re = re.compile(r'^{0}\.(.+)'.format(re.escape(key)))
    return OrderedDict((key_re.sub(r'\1', k), value) for (k, value)
                       in dictionary.items() if key_re.match(k) is not None)


class BatchLinear(nn.Linear, MetaModule):
    '''A linear meta-layer that can deal with batched weight matrices and biases, as for instance output by a
    hypernetwork.
    '''
    __doc__ = nn.Linear.__doc__

    def forward(self, input, params=None):

        if params is None:
            return nn.Linear.forward(self, input)

        else:

            bias = params.get('bias', None)
            weight = params['weight']
            output = input.matmul(weight.permute(*[i for i in range(len(weight.shape) - 2)], -1, -2))
            output += bias.unsqueeze(-2)
            return output


class Sine(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return torch.sin(30 * input)


class FCBlock(MetaModule):
    '''A fully connected neural network that also allows swapping out the weights when used with a hypernetwork.
    Can be used just as a normal neural network though, as well.
    '''

    def __init__(self, in_features, out_features, num_hidden_layers, hidden_features,
                 outermost_linear=False, nonlinearity='relu', weight_init=None):
        super().__init__()

        self.first_layer_init = None

        # Dictionary that maps nonlinearity name to the respective function, initialization, and, if applicable,
        # special first-layer initialization scheme
        nls_and_inits = {'sine': (Sine(), sine_init, first_layer_sine_init, last_layer_sine_init),
                         'relu': (nn.ReLU(inplace=True), init_weights_normal, None, None),
                         'sigmoid': (nn.Sigmoid(), init_weights_xavier, None, None),
                         'tanh': (nn.Tanh(), init_weights_xavier, None, None),
                         'selu': (nn.SELU(inplace=True), init_weights_selu, None, None),
                         'softplus': (nn.Softplus(), init_weights_normal, None, None),
                         'elu': (nn.ELU(inplace=True), init_weights_elu, None, None)}

        nl, nl_weight_init, first_layer_init, last_layer_init = nls_and_inits[nonlinearity]

        if weight_init is not None:  # Overwrite weight init if passed
            self.weight_init = weight_init
        else:
            self.weight_init = nl_weight_init

        self.net = []
        self.net.append(MetaSequential(
            BatchLinear(in_features, hidden_features), nl
        ))

        for i in range(num_hidden_layers):
            self.net.append(MetaSequential(
                BatchLinear(hidden_features, hidden_features), nl
            ))

        if outermost_linear:
            self.net.append(MetaSequential(BatchLinear(hidden_features, out_features)))
        else:
            self.net.append(MetaSequential(
                BatchLinear(hidden_features, out_features), nl
            ))

        self.net = MetaSequential(*self.net)
        if self.weight_init is not None:
            self.net.apply(self.weight_init)

        if first_layer_init is not None:  # Apply special initialization to first layer, if applicable.
            self.net[0].apply(first_layer_init)

        if last_layer_init is not None:
            self.net[-1].apply(last_layer_init)

    def forward(self, coords, params=None, **kwargs):
        if params is not None:
            params = get_subdict(params, 'net')

        output = self.net(coords, params=params)
        return output


class SingleBVPNet(MetaModule):
    '''A canonical representation network for a BVP.'''

    def __init__(self, out_features=1, type='sine', in_features=2, hidden_features=256, num_hidden_layers=3, **kwargs):
        super().__init__()
        self.net = FCBlock(in_features=in_features, out_features=out_features, num_hidden_layers=num_hidden_layers,
                           hidden_features=hidden_features, outermost_linear=True, nonlinearity=type)
        # print(self)

    def forward(self, model_input, params=None):
        # Enables us to compute gradients w.r.t. coordinates
        coords_org = model_input['coords'].requires_grad_(True)
        # coords = coords_org
        # various input processing methods for different applications
        output = self.net(coords_org, get_subdict(params, 'net'))
        return {'model_in': coords_org, 'model_out': output}


class PosEncodingNeRF(nn.Module):
    '''Module to add positional encoding as in NeRF [Mildenhall et al. 2020].'''

    def __init__(self, in_features, sidelength=None, fn_samples=None, use_nyquist=True):
        super().__init__()

        self.in_features = in_features

        if self.in_features == 3:
            self.num_frequencies = 10
        elif self.in_features == 2:
            assert sidelength is not None
            if isinstance(sidelength, int):
                sidelength = (sidelength, sidelength)
            self.num_frequencies = 4
            if use_nyquist:
                self.num_frequencies = self.get_num_frequencies_nyquist(min(sidelength[0], sidelength[1]))
        elif self.in_features == 1:
            assert fn_samples is not None
            self.num_frequencies = 4
            if use_nyquist:
                self.num_frequencies = self.get_num_frequencies_nyquist(fn_samples)

        self.out_dim = in_features + 2 * in_features * self.num_frequencies

    def get_num_frequencies_nyquist(self, samples):
        nyquist_rate = 1 / (2 * (2 * 1 / samples))
        return int(math.floor(math.log(nyquist_rate, 2)))

    def forward(self, coords):
        coords = coords.view(coords.shape[0], -1, self.in_features)

        coords_pos_enc = coords
        for i in range(self.num_frequencies):
            for j in range(self.in_features):
                c = coords[..., j]

                sin = torch.unsqueeze(torch.sin((2 ** i) * np.pi * c), -1)
                cos = torch.unsqueeze(torch.cos((2 ** i) * np.pi * c), -1)

                coords_pos_enc = torch.cat((coords_pos_enc, sin, cos), axis=-1)

        return coords_pos_enc.reshape(coords.shape[0], -1, self.out_dim)


class HyperNetwork(nn.Module):
    def __init__(self, hyper_in_features, hyper_hidden_layers, hyper_hidden_features, hypo_module):
        """
        Args:
            hyper_in_features: In features of hypernetwork
            hyper_hidden_layers: Number of hidden layers in hypernetwork
            hyper_hidden_features: Number of hidden units in hypernetwork
            hypo_module: MetaModule. The module whose parameters are predicted.
        """
        super().__init__()

        hypo_parameters = hypo_module.meta_named_parameters()

        self.names = []
        self.nets = nn.ModuleList()
        self.param_shapes = []
        for name, param in hypo_parameters:
            self.names.append(name)
            self.param_shapes.append(param.size())

            hn = FCBlock(in_features=hyper_in_features,
                         out_features=int(torch.prod(torch.tensor(param.size()))),
                         num_hidden_layers=hyper_hidden_layers, hidden_features=hyper_hidden_features,
                         outermost_linear=True, nonlinearity='relu')
            self.nets.append(hn)

            if 'weight' in name:
                self.nets[-1].net[-1].apply(lambda m: hyper_weight_init(m, param.size()[-1]))
            elif 'bias' in name:
                self.nets[-1].net[-1].apply(lambda m: hyper_bias_init(m))

    def forward(self, z):
        """
        Args:
            z: Embedding. Input to hypernetwork. Could be output of "Autodecoder"
        Returns:
            params: OrderedDict. Can be directly passed as the "params" parameter of a MetaModule.
        """
        params = OrderedDict()
        for name, net, param_shape in zip(self.names, self.nets, self.param_shapes):
            batch_param_shape = (-1,) + param_shape
            params[name] = net(z).reshape(batch_param_shape)
        # print(params)
        return params


################################################
# Initialization schemes
################################################

def hyper_weight_init(m, in_features_main_net):
    if hasattr(m, 'weight'):
        nn.init.kaiming_normal_(m.weight, a=0.0, nonlinearity='relu', mode='fan_in')
        m.weight.data = m.weight.data / 1.e2

    if hasattr(m, 'bias'):
        with torch.no_grad():
            m.bias.uniform_(-1 / in_features_main_net, 1 / in_features_main_net)


def hyper_bias_init(m):
    if hasattr(m, 'weight'):
        nn.init.kaiming_normal_(m.weight, a=0.0, nonlinearity='relu', mode='fan_in')
        m.weight.data = m.weight.data / 1.e2

    if hasattr(m, 'bias'):
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
        with torch.no_grad():
            m.bias.uniform_(-1 / fan_in, 1 / fan_in)


################################################
# Initialization methods
def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    # For PINNet, Raissi et al. 2019
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    # grab from upstream pytorch branch and paste here for now
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [l, u], then translate to
        # [2l-1, 2u-1].
        tensor.uniform_(2 * l - 1, 2 * u - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor


def init_weights_trunc_normal(m):
    # For PINNet, Raissi et al. 2019
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    if type(m) == BatchLinear or type(m) == nn.Linear:
        if hasattr(m, 'weight'):
            fan_in = m.weight.size(1)
            fan_out = m.weight.size(0)
            std = math.sqrt(2.0 / float(fan_in + fan_out))
            mean = 0.
            # initialize with the same behavior as tf.truncated_normal
            # "The generated values follow a normal distribution with specified mean and
            # standard deviation, except that values whose magnitude is more than 2
            # standard deviations from the mean are dropped and re-picked."
            _no_grad_trunc_normal_(m.weight, mean, std, -2 * std, 2 * std)


def init_weights_normal(m):
    if type(m) == BatchLinear or type(m) == nn.Linear:
        if hasattr(m, 'weight'):
            nn.init.kaiming_normal_(m.weight, a=0.0, nonlinearity='relu', mode='fan_in')


def init_weights_selu(m):
    if type(m) == BatchLinear or type(m) == nn.Linear:
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            nn.init.normal_(m.weight, std=1 / math.sqrt(num_input))


def init_weights_elu(m):
    if type(m) == BatchLinear or type(m) == nn.Linear:
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            nn.init.normal_(m.weight, std=math.sqrt(1.5505188080679277) / math.sqrt(num_input))


def init_weights_xavier(m):
    if type(m) == BatchLinear or type(m) == nn.Linear:
        if hasattr(m, 'weight'):
            nn.init.xavier_normal_(m.weight)


def sine_init(m):
    with torch.no_grad():
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            # See supplement Sec. 1.5 for discussion of factor 30
            m.weight.uniform_(-np.sqrt(6 / num_input) / 30, np.sqrt(6 / num_input) / 30)


def first_layer_sine_init(m):
    with torch.no_grad():
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of factor 30
            m.weight.uniform_(-1 / num_input, 1 / num_input)


def last_layer_sine_init(m):
    with torch.no_grad():
        if hasattr(m, 'weight'):
            nn.init.zeros_(m.weight)
            nn.init.zeros_(m.bias)
