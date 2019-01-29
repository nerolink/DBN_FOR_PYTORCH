import torch
import torch.nn.functional as F
from __future__ import print_function


class RBM(object):

    def __init__(self, input_size, output_size, name, params):
        super().__init__()
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        self.w = torch.randn(input_size, output_size).to(self.device)
        self.vb = torch.zeros(input_size).to(self.device)
        self.name = name
        self.input_size = input_size
        self.output_size = output_size
        self.params = params

    @staticmethod
    def probability_h_to_v(hidden, weights, v_bias):
        """
        :param hidden: 隐藏层 type=torch.tensor ,[output_size]
        :param weights: 权重 type=torch.tensor,[input_size * output_size]
        :param v_bias:  可见层偏置 type=torch.tensor ,[input_size]
        :return:  概率 type=torch.tensor [input_size]
        """
        return F.sigmoid(weights @ hidden + v_bias)

    @staticmethod
    def probability_v_to_h(visible, weights, h_bias):
        """
        :param visible: 可见层 type=torch.tensor,[input_size]
        :param weights: 权重 type=torch.tensor,[input_size*output_size]
        :param h_bias: 隐藏层偏置 type=torch.tensor，[output_size]
        :return: 概率 type=torch.tensor [output_size]
        """
        return F.sigmoid(weights.transpose(1, 0) @ visible + h_bias)

    @staticmethod
    def sample_from_probability(probability):
        """
        输入一个可见层或隐藏层概率为1向量，输出单元为0或1的向量，
        :param probability:    [0.1,0.2,0.4,0.3......]
        :return:                [0,0,1,1,.....]
        """
        return F.relu(torch.sign(torch.probability - torch.rand_like(probability)))

    @staticmethod
    def given_v_sample_h(visible, weights, h_bias):
        """
        给定可见层的值，通过概率求出隐含层的值
        :param visible:     可见层的值
        :param weights
        :param h_bias
        :return:            隐含层的值
        """
        return RBM.sample_from_probability(RBM.probability_v_to_h(visible, weights, h_bias))

    @staticmethod
    def given_h_sample_v(hidden, weights, v_bias):
        """
        给定隐含层的值，通过概率求出可见层的值
        :param hidden:      隐含层
        :param weights
        :param v_bias
        :return:            可见层
        """
        return RBM.sample_from_probability(RBM.probability_h_to_v(hidden, weights, v_bias))

    def train(self,data):


