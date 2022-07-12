import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import warnings

warnings.filterwarnings("ignore")

# class FocalLoss(nn.Module):
#
#     def __init__(self, focusing_param=2, balance_param=0.25):
#         super(FocalLoss, self).__init__()
#
#         self.focusing_param = focusing_param
#         self.balance_param = balance_param
#
#     def forward(self, output, target):
#
#         cross_entropy = F.cross_entropy(output, target)
#         cross_entropy_log = torch.log(cross_entropy)
#         logpt = - F.cross_entropy(output, target)
#         pt = torch.exp(logpt)
#
#         focal_loss = -((1 - pt) ** self.focusing_param) * logpt
#
#         balanced_focal_loss = self.balance_param * focal_loss
#
#         return balanced_focal_loss


class FocalLoss(nn.Module):

    def __init__(self,
                 gamma=0.,
                 alpha=None,
                 size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.size_average = size_average

        if isinstance(alpha, (float, int)):
            self.alpha = torch.Tensor([alpha, 1 - alpha])

        if isinstance(alpha, list):
            self.alpha = torch.Tensor(alpha)

    def forward(self, input, target):

        if input.dim() > 2:
            input = input.view(input.size(0), input.size(1), -1)
            input = input.transpose(1, 2)
            input = input.contiguous().view(-1, input.size(2))

        target = target.view(-1, 1)
        logpt = F.log_softmax(input)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type() != input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0, target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1 - pt) ** self.gamma * logpt

        if self.size_average:
            return loss.mean()

        return loss.sum()
