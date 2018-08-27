import torch
import torch.nn as nn
import torch.nn.functional as fnn


class Linear2D(nn.Module):
    '''
    Layer transforms a batch of 2d tensors of size input_size_1 x input_size_2 to a batch of
    2d tensors h1 x h2.

    This is produced by two linear transformations (without bias) with a transpose operation
    and nonlinearity in between.
    '''

    def __init__(self, input_size_1, input_size_2, h1, h2, activation=fnn.tanh):
        super(Linear2D, self).__init__()
        self.activation = activation
        self.input_size_1 = input_size_1
        self.input_size_2 = input_size_2
        self.h1 = h1
        self.h2 = h2
        self.weight_1 = nn.Parameter(
            torch.randn(input_size_2, h1)
        )
        self.weight_2 = nn.Parameter(
            torch.randn(input_size_1, h2)
        )

    def forward(self, input_tensor):
        '''
        Input tensor has size batch x input_size_1 x input_size_2
        :param input_tensor: batch x input_size_1 x input_size_2
        :return: batch x h1 x h2
        '''
        dim_1 = input_tensor.bmm(
            self.weight_1.repeat(input_tensor.size(0), 1, 1)
        )
        dim_2 = self.activation(dim_1).transpose(1, 2).bmm(
            self.weight_2.repeat(input_tensor.size(0), 1, 1)
        )
        return dim_2


if __name__ == '__main__':
    batch = 3
    in1 = 10; in2 = 20
    h1 = 5; h2 = 1
    input = torch.rand(batch, in1, in2)
    lin = Linear2D(in1, in2, h1, h2)
    print lin(input).size()
