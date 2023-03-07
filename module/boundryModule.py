####################################################
# 这个模块用于处理边界
#
####################################################

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class BoundryModule(nn.Module):
    def __init__(self, in_channels=3):
        super(BoundryModule, self).__init__()

        self.in_channels = in_channels  #修改
        mid_channels = 256  #中间过渡
        num_directions = 8 #分多少个方向
        num_masks = 2 #蒙板

        self.Direction_branch = nn.Sequential(
            nn.Conv2d(in_channels,
                      mid_channels,
                      kernel_size=1,
                      stride=1,
                      padding=0,
                      bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(),
            nn.Conv2d(mid_channels,
                      num_directions,
                      kernel_size=1,
                      stride=1,
                      padding=0,
                      bias=False))
        self.Boundry_branch = nn.Sequential(
            nn.Conv2d(in_channels,
                      mid_channels,
                      kernel_size=1,
                      stride=1,
                      padding=0,
                      bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(),
            nn.Conv2d(mid_channels,
                      num_masks,
                      kernel_size=1,
                      stride=1,
                      padding=0,
                      bias=False))

    def forward(self, input):
        #这里的 输入（input）,是经过FaPN的上采样之后的
        _, _, h, w = input.size()

        mask_map = self.Boundry_branch(input)
        direction_map = self.Direction_branch(input)

        return mask_map, direction_map

def get_BoundryModule(in_channels=3):
    model = BoundryModule(in_channels)
    return model



if __name__ == '__main__':
    # model = get_BoundryModule(3)
    # x = torch.randn([1,3,512,512])
    # mask, direction = model(x)
    # mask = mask.max(dim=1)[1].unsqueeze(dim=1)
    x = torch.randn([1,2,4,4])
    x1 = x.max(dim=1)[1].unsqueeze(dim=1).float()
    print(x)
    print(x1)
    op = nn.Sigmoid()
    x = op(x)
    print(x)
    y = torch.randn([1,8,4,4])
    x2 = x.max(dim=1)[1].unsqueeze(dim=1).float()
    print(x2)
    #print(torch.mul(y,x))
    #print((direction*mask).shape)