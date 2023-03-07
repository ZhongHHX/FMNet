###############################################
#  HRnet + fam +three attention(MAM: muti-attention)
#
#
#
###############################################




from module.hrnet import get_HRnet
from module.OcrModule import get_ocr
from module.fam import get_fam
from module.psa import get_spatial_channels
import torch
import torch.nn as nn
import torch.nn.functional as F

class HRnet_fam_MutiAttention(nn.Module):
    def __init__(self, config, Mode=1, **kwargs):
        super(HRnet_fam_MutiAttention, self).__init__()
        self.backbone = get_HRnet(config, upsample=False, **kwargs)
        self.fam1 = get_fam()  #
        self.fam2 = get_fam()
        self.fam3 = get_fam()
        #self.fam4 = get_fam()
        self.ocr = get_ocr(last_inp_channels=256, down_channels=False)

        self.Mode = Mode
        # 架构模式
        if self.Mode == 1:  # 并行
            #self.down_channels = nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0, bias=True)
            self.spatial_channels = get_spatial_channels(inplanes=256, mode=1)
        elif self.Mode == 2:  # 穿行
            self.spatial_channels = get_spatial_channels(inplanes=256, mode=2)

        self.to_numclasses_channels = nn.Conv2d(256, 14, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x):
        #input_size = x.shape[2:]
        x0, x1, x2, f = self.backbone(x)    #各自的分辨率128 64 32 16
        f = self.fam1([x2, f])
        f = x2 + f
        f = self.fam2([x1, f])
        f = x1 + f
        f = self.fam3([x0, f])
        f = x0 + f    #f: torch.Size([1, 256, 128, 128])

        outputs = self.ocr(f)

        if self.Mode ==1:#并行
            #input = self.down_channels(f)
            spatial_add_channels_out = self.spatial_channels(f)
            outputs = outputs + spatial_add_channels_out
        elif self.Mode == 2:#串行
            outputs = self.spatial_channels(outputs)

        outputs = self.to_numclasses_channels(outputs)
        output = F.interpolate(outputs, size=(512,512), mode='bilinear', align_corners=True)

        return output

def get_net(cfg, mode=1, **kwargs):
    model = HRnet_fam_MutiAttention(cfg, mode=mode, **kwargs)
    return model


def count_param(model):   #计算模型参数量
    param_count = 0
    for param in model.parameters():
        param_count += param.view(-1).size()[0]
    return param_count

if __name__ == '__main__':
    import yaml
    #path = '../configs/seg_hrnet_ocr_w48_cls59_520x520_sgd_lr1e-3_wd1e-4_bs_16_epoch200.yaml'
    path = '../configs/config_hrnet_ocr.yml'
    with open(path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    model = get_net(cfg=config)
    model = model.cuda()
    #print(model)
    x = torch.randn([1,3,512,512]).cuda()
    import time
    t = time.time()
    out = model(x)
    t1 = time.time()
    print(out.shape,t1-t)
    print(count_param(model))
