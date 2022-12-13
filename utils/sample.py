import torch, random
from utils import registry

SAMPLE_FUNCTIONS = registry.Registry()


@SAMPLE_FUNCTIONS.register('ram_sample')
def ram_sample(pos_tensor, neg_tensor, sample_num):
    if sample_num is None:
        return torch.cat([pos_tensor, neg_tensor], 0)
    half = int(sample_num / 2)
    pos_size = pos_tensor.shape[0]
    neg_size = neg_tensor.shape[0]

    pos_start_ind = random.randint(0, pos_size - half)
    sample_pos = pos_tensor[pos_start_ind: (pos_start_ind + half)]

    if neg_size <= half:
        random_neg = (torch.rand(half) * neg_tensor.shape[0]).long()
        sample_neg = torch.index_select(neg_tensor, 0, random_neg)
    else:
        neg_start_ind = random.randint(0, neg_size - half)
        sample_neg = neg_tensor[neg_start_ind: (neg_start_ind + half)]

    samples = torch.cat([sample_pos, sample_neg], 0)

    return samples


@SAMPLE_FUNCTIONS.register('random_sample')
def random_sample(pos_tensor, neg_tensor, sample_num):
    if sample_num is None:
        return torch.cat([pos_tensor, neg_tensor], 0)
    # split the sample into half
    half = int(sample_num / 2)

    random_pos = (torch.rand(half) * pos_tensor.shape[0]).long()
    random_neg = (torch.rand(half) * neg_tensor.shape[0]).long()

    sample_pos = torch.index_select(pos_tensor, 0, random_pos)
    sample_neg = torch.index_select(neg_tensor, 0, random_neg)

    samples = torch.cat([sample_pos, sample_neg], 0)

    return samples


@SAMPLE_FUNCTIONS.register('random_sample_full')
def random_sample_full(tensor, sample_num):
    if sample_num is None:
        return tensor
    random_idx = (torch.rand(sample_num) * tensor.shape[0]).long()
    return torch.index_select(tensor, 0, random_idx)


@SAMPLE_FUNCTIONS.register('random_sample_full_ram')
def random_sample_full_ram(tensor, sample_num):
    if sample_num is None:
        return tensor
    size = tensor.shape[0]
    start_ind = random.randint(0, size - sample_num)
    samples = tensor[start_ind: (start_ind + sample_num)]
    return samples
