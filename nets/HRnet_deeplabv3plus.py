from module.deeplabv3.aspp import build_aspp
from module.deeplabv3.decoder import build_decoder
from module.deeplabv3.hrnet_deeplab import get_HRnet
from module.deeplabv3.resnet101 import ResNet101
import torch
import torch.nn as nn
import torch.nn.functional as F

class HRnet_deeplabv3plus(nn.Module):
    def __init__(self, config, **kwargs):
        super(HRnet_deeplabv3plus, self).__init__()
        #self.backbone = get_HRnet(config,**kwargs)
        self.backbone = ResNet101(output_stride=16, BatchNorm=nn.BatchNorm2d)
        self.aspp = build_aspp(backbone='hrnet_w48', output_stride=16, BatchNorm=nn.BatchNorm2d)
        #build_aspp(backbone, output_stride, BatchNorm):
        self.decoder = build_decoder(num_classes=14, backbone='hrnet_w48', BatchNorm=nn.BatchNorm2d)

    def forward(self, x):
        size = x.size()[2:]
        x, low_level_feat = self.backbone(x)
        print(x.shape, low_level_feat.shape)
        x = self.aspp(x)

        x = self.decoder(x, low_level_feat)
        x = F.interpolate(x, size=size, mode='bilinear', align_corners=True)
        return x


def get_net(cfg, **kwargs):
    model = HRnet_deeplabv3plus(cfg, **kwargs)
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
    print(count_param(model))
    #print(model)
    x = torch.randn([2,3,448,448]).cuda()
    import time
    t = time.time()
    out = model(x)
    t1 = time.time()
    print(out.shape,t1-t)  #时间比8.762192487716675 ： 15.435196876525879  －》 448 ： 512

    #
    # #------
    #total_params = sum(p.numel() for p in model.parameters())
    #print('总参数个数：{}'.format(total_params))   #68046892   68324092
    #                                                                     7.350347280502319
    # total_trainable_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # print('需训练参数个数：{}'.format(total_trainable_parameters))


#
#torch.Size([1, 14, 512, 512]) 3.2730817794799805
#68046892


#
#torch.Size([1, 14, 512, 512]) 16.475454330444336
#68324092
#
'''
设置 batch_size>1, 且 drop_last=True    解决 batchnorm问题

选择 hrnet_w48 : 参数量： 83388590   83M
选择 resnet101 : 参数量： 59342510   59M

'''
