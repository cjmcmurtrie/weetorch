import torch
import torch.nn as nn
import torch.nn.functional as fnn
import numpy as np
from collections import Counter


def index_tensor_count(byte_tensor):
    '''
    Count the integer values in a Pytorch ByteTensor.
    :param byte_tensor:
    :return:
    '''
    flat = byte_tensor.view(-1)
    return Counter(flat.cpu().numpy())


class WeightedEmbedding(nn.Module):
    '''
    Embedding layer that maintains state of index frequency and weighs gradient updates accordingly.
    Less frequent indices consequently have larger updates than more frequent ones.
    '''
    # todo: handle cuda.

    def __init__(self, n, embedding_size, weight_mode='frequency'):
        super(WeightedEmbedding, self).__init__()
        self.n = n
        self.embedding_size = embedding_size
        self.grad_weight_mode = weight_mode
        self.weight = nn.Parameter(
            torch.rand(n, embedding_size) * 1e-3,
            requires_grad=True
        )
        self.grad_freq_mask = torch.ones(n, embedding_size)
        self.grad_mask = torch.ones(n, embedding_size)
        self.weight.register_hook(lambda grad: self._mult_grad_mask(grad))

    def _mult_grad_mask(self, grad_in):
        return grad_in * self.grad_mask

    def forward(self, input_tensor):
        '''
        :param input_tensor: tensor of indices.
        :return: tensor of
        '''
        self._update_index_freqs(input_tensor)
        if input_tensor.dim() == 1:
            input_tensor = input_tensor.unsqueeze(0)
        return torch.stack([
            self.weight.index_select(0, i) for i in input_tensor
        ])

    def _update_index_freqs(self, input_tensor):
        '''
        Function to update weight table index frequencies. This is used in the hook
        function to modify gradients. Each update gradient is multiplied as
            grad *= 1 / index_frequency.
        With all frequencies initialized to 1. Thus the first update for an index embedding
        is grad, the second is grad / 2, the third is grad / 3, and so on.
        :param input_tensor:
        :return:
        '''
        batch_counts = index_tensor_count(input_tensor)
        for k, v in batch_counts.items():
            self.grad_freq_mask[k] += v
        self.grad_mask = 1. / self.grad_freq_mask


if __name__ == '__main__':

    embedding = WeightedEmbedding(100, 50)
    input_inds = torch.LongTensor([
        [99, 0, 1, 87],
        [65, 9, 45, 99]
    ])
    targets = torch.rand(2, 4, 50)
    outp = embedding(input_inds)
    lossf = nn.CosineEmbeddingLoss()
    loss = lossf(outp[0], targets[0], torch.Tensor([1, 1, 1, 1]))
    loss.backward()
