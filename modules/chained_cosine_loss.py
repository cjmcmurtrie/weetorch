import torch


class ChainedCosineLoss(torch.nn.Module):

    def __init__(self, eps=1e-15):
        super(ChainedCosineLoss, self).__init__()
        self.eps = eps

    def chain_function(self, tensor):
        '''
        Input is a batch of 2d tensors, N x embedding x length.
        Output takes window sized steps, flattens them.
        '''
        # todo: currently window size is not variable.
        return torch.cat([tensor[:, :-1, :], tensor[:, 1:, :]], 2)

    def reshape_function(self, tensor):
        '''
        Input is a batch of 2d tensors, N x embedding x length.
        Output takes window sized steps, flattens them.
        '''
        return tensor.contiguous().view(tensor.size(0) * tensor.size(1), -1)

    def cosine_similarity(self, inp, tar):
        mat1 = self.chain_function(inp)
        mat2 = self.chain_function(tar)
        mat1 = self.reshape_function(mat1)
        mat2 = self.reshape_function(mat2)
        dots = mat1.unsqueeze(1).bmm(mat2.unsqueeze(2)).squeeze()
        norms = (mat1.norm(2, 1) * mat2.norm(2, 1)).squeeze()
        return dots / (norms + self.eps)

    def forward(self, input_tensor, target_tensor):
        sims = self.cosine_similarity(input_tensor, target_tensor)
        sims = sims[sims.nonzero()]
        loss = (1.0 - sims).sum() / sims.size(0)
        return loss


if __name__ == '__main__':

    batch = 5
    embedding_size = 5
    length = 10
    loss_f = ChainedCosineLoss()
    input_tensor = torch.rand(batch, length, embedding_size)
    target_tensor = torch.rand(batch, length, embedding_size)
    print input_tensor[0, 0]
    print input_tensor[0, 1]
    loss = loss_f(input_tensor, target_tensor)
    print loss
