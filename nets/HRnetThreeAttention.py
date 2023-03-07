#--------------------------------------------------------------------------------------------#
#   本实验： HRnet + three attention
#
#   backbone: HRnet_w48
#   1.并行
#   class attention + channels attention + Spatial attention
#   2.串行
#   class attention -> channels attention -> Spatial attention
#   该文件的目的是：
#               探究类别注意力机制 与 通道注意力机制和位置注意力的比
#--------------------------------------------------------------------------------------------#
from module.hrnet import get_HRnet
from module.OcrModule import get_ocr
from module.psa import get_spatial_channels
import torch
import torch.nn as nn
import torch.nn.functional as F

class HRnetThreeAttention(nn.Module):
    def __init__(self, config, Mode=1, **kwargs):
        super(HRnetThreeAttention, self).__init__()
        self.backbone = get_HRnet(config, **kwargs)  # 得到的是： N 720 H W
        self.ocr = get_ocr(down_channels=False)    # 得到256通道
        self.Mode = Mode
        #架构模式
        if self.Mode == 1: #并行
            self.down_channels = nn.Conv2d(720, 256, kernel_size=1, stride=1, padding=0, bias=True)
            self.spatial_channels = get_spatial_channels(inplanes=256, mode=1)
        elif self.Mode == 2: #穿行
            self.spatial_channels = get_spatial_channels(inplanes=256,mode=2)

        self.to_numclasses_channels = nn.Conv2d(256, 14, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x):
        input_size = x.shape[2:]
        x = self.backbone(x)
        outputs = self.ocr(x)  #class_attention_output

        if self.Mode ==1:#并行
            input = self.down_channels(x)
            spatial_add_channels_out = self.spatial_channels(input)
            outputs = outputs + spatial_add_channels_out
        elif self.Mode == 2:#串行
            outputs = self.spatial_channels(outputs)

        outputs = self.to_numclasses_channels(outputs)
        outputs = F.interpolate(outputs, size=input_size, mode='bilinear', align_corners=True)

        return outputs


def get_net(cfg, mode=1,**kwargs):
    '''
    Args:
        cfg: 网络架构层的详细信息
        mode: 1 表示并行  ， 2表示串行
        **kwargs:

    Returns:
    '''
    model = HRnetThreeAttention(cfg, mode=mode,**kwargs)
    return model


if __name__ == '__main__':
    import yaml
    #path = '../configs/seg_hrnet_ocr_w48_cls59_520x520_sgd_lr1e-3_wd1e-4_bs_16_epoch200.yaml'
    path = '../configs/config_hrnet_ocr.yml'
    with open(path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    model = get_net(cfg=config,mode=1)
    x =torch.randn([1,3,448,448])
    print(model(x).shape)
    print(sum(p.numel() for p in model.parameters())) #68366394