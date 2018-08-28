import torch
import torch.nn as nn


class EuclideanLoss(torch.nn.Module):
    '''
    Loss function treats pairwise distance between vectors as an error function.
    Thus is is used to minimise Euclidean distance between inputs and targets.
    If input is a batch of 2d tensors, tensors are flattened as vectors first.
    '''
    # todo: cuda support.
    # todo: formalize normalization.

    def __init__(self):
        super(EuclideanLoss, self).__init__()
        self.euclid = nn.PairwiseDistance()
        self.eps = 1e-15

    def euclidean(self, inp, target):
        if inp.dim() > 2:
            inp = inp.contiguous().view(inp.size(0), inp.size(1) * inp.size(2))
            target = target.contiguous().view(target.size(0), target.size(1) * target.size(2))
        return self.euclid(inp, target) / \
            (torch.norm(inp, 2, 1) + torch.norm(target, 2, 1) + self.eps)

    def forward(self, inp, target):
        dist = self.euclidean(inp, target)
        loss = dist.sum() / dist.size(0)
        return loss
