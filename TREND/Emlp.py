import torch
from torch import nn
from torch.nn import functional as F


class EMLP(nn.Module):
    def __init__(self, args):
        super(EMLP, self).__init__()
        self.vars = nn.ParameterList()
        # parameterlist作用是以列表形式保存参数，可以像列表一样被索引
        w1 = nn.Parameter(torch.ones(*[1, args.out_dim]))  # [1,d]
        torch.nn.init.kaiming_normal_(w1)  # 使用正态分布填充输入张量，为何恺明等人提出
        self.vars.append(w1)
        self.vars.append(nn.Parameter(torch.zeros(1)))
        # 此时参数列表里有两个参数，第一个为[1,d]，第二个为[1]

        # w2 = nn.Parameter(torch.ones(*[1, args.hid_dim]))
        # torch.nn.init.kaiming_normal_(w2)
        # self.vars.append(w2)
        # self.vars.append(nn.Parameter(torch.zeros(1)))

    def forward(self, x, vars=None):
        if vars == None:
            vars = self.vars
        x = F.linear(x, vars[0], vars[1])
        # 类似于nn.linear，即y=ax+b，因此上文中第一个参数是线路，第二个参数为单个值

        # x = torch.relu(x)
        # x = F.linear(x, vars[2], vars[3])
        # x = torch.relu(x)
        return x

    def parameters(self):
        return self.vars
