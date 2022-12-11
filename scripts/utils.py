import numpy as np
import torch


class CosineLR:
    """
    Cosine learning rate schedule with warmup
    """
    def __init__(self, config, optimizer):
        self.config = config
        self.epoch = 0
        self.optimizer = optimizer

        self.T0 = int(self.config['total_epoch'] / 5)
        self.eta_max = self.config['optim_args']['eta_max']

    def step(self):
        self.epoch += 1
        if self.epoch <= self.T0:
            eta = 1e-4 + (self.epoch / self.T0) * self.eta_max
        else:
            eta = self.eta_max * np.cos((np.pi / 2) * (self.epoch - self.T0) / (self.config['total_epoch'] - self.T0)) + 1e-6
        for op_params in self.optimizer.param_groups:
            op_params['lr'] = eta


def load_weight(net, weight_dict) :
    state_dict = net.state_dict()
    for key in weight_dict.keys() :
        if key in state_dict and (weight_dict[key].size() == state_dict[key].size()) :
            value = weight_dict[key]
            if not isinstance(value, torch.Tensor) :
                value = value.data
            state_dict[key] = value
    net.load_state_dict(state_dict)


def gpu(data) :
    if isinstance(data, list) or isinstance(data, tuple) :
        data = [gpu(x) for x in data]
    elif isinstance(data, dict) :
        data = {key :gpu(_data) for key,_data in data.items()}
    elif isinstance(data, torch.Tensor) :
        # ref: https://discuss.pytorch.org/t/should-we-set-non-blocking-to-true/38234
        data = data.contiguous().cuda(non_blocking=True)
    return data


def from_numpy(data) :
    if isinstance(data, dict) :
        for key in data.keys() :
            data[key] = from_numpy(data[key])
    if isinstance(data, list) or isinstance(data, tuple) :
        data = [from_numpy(x) for x in data]
    if isinstance(data, np.ndarray) :
        data = torch.from_numpy(data)
    return data


def to_numpy(data) :
    if isinstance(data, dict) :
        for key in data.keys() :
            data[key] = to_numpy(data[key])
    if isinstance(data, list) or isinstance(data, tuple) :
        data = [to_numpy(x) for x in data]
    if torch.is_tensor(data) :
        data = data.numpy()
    return data


def to_long(data) :
    if isinstance(data, dict) :
        for key in data.keys() :
            data[key] = to_long(data[key])
    if isinstance(data, list) or isinstance(data, tuple) :
        data = [to_long(x) for x in data]
    if torch.is_tensor(data) and data.dtype == torch.int16 :
        data = data.long()
    return data


def ref_copy(data) :
    if isinstance(data, list) :
        return [ref_copy(x) for x in data]
    if isinstance(data, dict) :
        d = dict()
        for key in data :
            d[key] = ref_copy(data[key])
        return d
    return data


def to_int16(data) :
    if isinstance(data, dict) :
        for key in data.keys() :
            data[key] = to_int16(data[key])
    if isinstance(data, list) or isinstance(data, tuple) :
        data = [to_int16(x) for x in data]
    if isinstance(data, np.ndarray) and data.dtype == np.int64 :
        data = data.astype(np.int16)
    return data

