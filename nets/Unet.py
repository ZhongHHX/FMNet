import torch.nn as nn
import torch
import numpy as np
#from module.SwinT import swin_s as ST
from module.SwinT import get_StageModule as StageM
#import module.SwinT as ST
import torch.nn.functional as F

class decoder(nn.Module):
    #这一部分：先降通道数，再进行batchnorm，再relu
    def __init__(self,in_ch,out_ch):
        super(decoder, self).__init__()
        self.conv1x1 = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
        self.batchnorm = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        _,c,h,w = x.size()

        x = self.relu(self.batchnorm(self.conv1x1(x)))
        output = F.interpolate(x, (h*2,w*2), mode='bilinear', align_corners=True)
        return output







class UNetSWT_P(nn.Module):
    def __init__(self, in_ch, out_ch,hidden_dim, layers,  heads, downscaling_factors=(4, 2, 2, 2), head_dim=32, window_size=7, relative_pos_embedding=True ,num_classes=3):
        super(UNetSWT_P, self).__init__()
        self.encode1 = StageM(in_channels=in_ch, hidden_dimension=hidden_dim, layers=layers[0],
                                  downscaling_factor=downscaling_factors[0], num_heads=heads[0], head_dim=head_dim,
                                  window_size=window_size, relative_pos_embedding=relative_pos_embedding)
        self.encode2 = StageM(in_channels=hidden_dim, hidden_dimension=hidden_dim * 2, layers=layers[1],
                                  downscaling_factor=downscaling_factors[1], num_heads=heads[1], head_dim=head_dim,
                                  window_size=window_size, relative_pos_embedding=relative_pos_embedding)
        self.encode3 = StageM(in_channels=hidden_dim * 2, hidden_dimension=hidden_dim * 4, layers=layers[2],
                                  downscaling_factor=downscaling_factors[2], num_heads=heads[2], head_dim=head_dim,
                                  window_size=window_size, relative_pos_embedding=relative_pos_embedding)
        self.encode4 = StageM(in_channels=hidden_dim * 4, hidden_dimension=hidden_dim * 8, layers=layers[3],
                                  downscaling_factor=downscaling_factors[3], num_heads=heads[3], head_dim=head_dim,
                                  window_size=window_size, relative_pos_embedding=relative_pos_embedding)

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(hidden_dim * 8 ),
            nn.Linear(hidden_dim * 8 , num_classes)
        )   #用于分类

        #用于分割  decoder : 先降通道再上采样
        self.de1 = decoder(768,384)

        self.de2 = decoder(768,192)

        self.de3 = decoder(192*2, 96)

        self.de4 = decoder(96*2, 48)

        #de5单纯为了上升到224
        self.de5 = decoder(48,out_ch)




    def forward(self, img):
        #下采样部分
        x1 = self.encode1(img)   #1 96 56 56

        x2 = self.encode2(x1)     #1 192 28 28

        x3 = self.encode3(x2)     #1 384 14 14

        x4 = self.encode4(x3)     #1, 768, 7, 7

        # x = x.mean(dim=[2, 3])
        # return self.mlp_head(x)

        up1 = self.de1(x4)
        x3 = torch.cat([x3, up1],dim=1)

        up2 = self.de2(x3)
        x2 = torch.cat([x2, up2], dim=1)

        up3 = self.de3(x2)
        x1 = torch.cat([x1, up3], dim=1)


        x = self.de4(x1)

        x = self.de5(x)
        #print(x.shape)
        return x
    # def forward(self, x):
    #     x = self.encode1(x)
    #     x = self.encode2(x)
    #     x = self.encode3(x)
    #     x = self.encode4(x)
    #     print(x.shape)
    #     x = x.mean(dim=[2, 3])
    #     x = self.mlp_head(x)
    #     return x




def swin_t(in_ch=3,num_classes=2, hidden_dim=96, layers=(2, 2, 6, 2), heads=(3, 6, 12, 24), **kwargs):
    return UNetSWT_P(in_ch=in_ch,out_ch=num_classes,hidden_dim=hidden_dim, layers=layers, heads=heads, **kwargs)

def swin_s(in_ch=3,num_classes=2, hidden_dim=96, layers=(2, 2, 18, 2), heads=(3, 6, 12, 24), **kwargs):
    return UNetSWT_P(in_ch=in_ch,out_ch=num_classes,hidden_dim=hidden_dim, layers=layers, heads=heads, **kwargs)

# def swin_b(in_ch=3,num_classes=2, hidden_dim=128, layers=(2, 2, 18, 2), heads=(4, 8, 16, 32), **kwargs):
#     return UNetSWT_P(in_ch=in_ch,out_ch=num_classes,hidden_dim=hidden_dim, layers=layers, heads=heads, **kwargs)
#
# def swin_l(in_ch=3,num_classes=2, hidden_dim=192, layers=(2, 2, 18, 2), heads=(6, 12, 24, 48), **kwargs):
#     return UNetSWT_P(in_ch=in_ch,out_ch=num_classes,hidden_dim=hidden_dim, layers=layers, heads=heads, **kwargs)

def get_net(mode='s',num_classes=14):
    if mode =='t':
        return swin_t(num_classes=num_classes)
    elif mode=='s':
        return swin_s(num_classes=num_classes)


if __name__=='__main__':
    x = torch.randn([1,3,448,448])
    # model = swin_t()    #参数28001695
    #model = swin_s()      #参数49312279
    model = get_net('s',num_classes=14)
    total_params = sum(p.numel() for p in model.parameters())
    print(total_params)
    print(x.shape)
    x = model(x)
    # print(x)
    x = torch.argmax(x,dim=1)
    # print(x)
    # x = x.data.cpu().numpy()
    # x = np.argmax(x, axis=1)
    #
    # print(x.shape)
    # x = torch.randn([1,5,5]).numpy()
    # print(x)
    # x = np.expand_dims(x, axis=1)
    # print(x)
    #
