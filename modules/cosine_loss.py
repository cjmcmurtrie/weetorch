import torch
import torch.nn as nn


class CosineLoss(torch.nn.Module):
    '''
    Loss calculated on the cosine distance between batches of vectors:
        loss = sum(1 - (a.b / (|a|*|b|))) / batch size
    If a, b are batches of 2d tensors, they are flattened to 2d arrays first.
    '''

    def __init__(self, eps=1e-15):
        super(CosineLoss, self).__init__()
        self.eps = eps

    def cosine_similarity(self, inp, target):
        if inp.dim() > 2:
            inp = inp.contiguous().view(
                inp.size(0),
                inp.size(1) * inp.size(2)
            )
            target = target.contiguous().view(
                inp.size(0),
                inp.size(1) * inp.size(2)
            )
        return inp.unsqueeze(1).bmm(target.unsqueeze(2)).squeeze() / \
            (torch.norm(inp, 2, 1) * torch.norm(target, 2, 1) + self.eps)

    def forward(self, inp, target):
        sim = self.cosine_similarity(inp, target)
        loss = (1.0 - sim).sum() / sim.size(0)
        return loss
