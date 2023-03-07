import torch
import torch.nn as nn
import torch.nn.functional as F



class AlignModule(nn.Module):
    def __init__(self, inplane, outplane):
        super(AlignModule, self).__init__()
        self.down_h = nn.Conv2d(inplane, outplane, kernel_size=1, bias=False)
        self.down_l = nn.Conv2d(inplane, outplane, kernel_size=1, bias=False)
        self.flow_make = nn.Conv2d(outplane*2, 2, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        low_feature, h_feature= x     #low_feature 实际是分辨率大图  h_feature 是分辨率小图
        h_feature_orign = h_feature

        h, w = low_feature.size()[2:]
        size = (h, w)
        low_feature = self.down_l(low_feature)
        h_feature= self.down_h(h_feature)
        #print(low_feature.shape, h_feature.shape)
        h_feature = F.interpolate(h_feature,size=size, mode="bilinear", align_corners=False)
        flow = self.flow_make(torch.cat([h_feature, low_feature], 1))   #3x3卷积
        h_feature = self.flow_warp(h_feature_orign, flow, size=size)

        return h_feature

    def flow_warp(self, input, flow, size):
        out_h, out_w = size #大
        n, c, h, w = input.size()  #小

        norm = torch.tensor([[[[out_w, out_h]]]]).type_as(input).to(input.device)
        w = torch.linspace(-1.0, 1.0, out_h).view(-1, 1).repeat(1, out_w)
        h = torch.linspace(-1.0, 1.0, out_w).repeat(out_h, 1)
        grid = torch.cat((h.unsqueeze(2), w.unsqueeze(2)), 2)
        #print(norm,w.shape,h.shape,grid.shape)
        grid = grid.repeat(n, 1, 1, 1).type_as(input).to(input.device)

        grid = grid + flow.permute(0, 2, 3, 1) / norm
        print(grid.shape)
        print(input.shape)
        output = F.grid_sample(input, grid)

        return output

def get_fam(in_channels=256,out_channels=256):
    model = AlignModule(inplane=in_channels, outplane=out_channels)
    return model

if __name__=='__main__':
    low_feature = torch.randn([1,384,8,8])
    h_feature = torch.rand([1,384,4,4])
    x = [low_feature, h_feature]
    #torch.Size([1, 48, 128, 128]) torch.Size([1, 96, 64, 64]) torch.Size([1, 192, 32, 32]) torch.Size([1, 384, 16, 16])
    model = AlignModule(inplane=384, outplane=192)
    print(model(x).shape)
    total_params = sum(p.numel() for p in model.parameters())
    print('总参数个数：{}'.format(total_params))   #68046892


