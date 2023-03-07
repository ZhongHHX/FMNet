######################################################
#  HRnet + fam  特征对齐
#
#
######################################################



from module.hrnet import get_HRnet
from module.OcrModule import get_ocr
from module.fam import get_fam
import torch
import torch.nn as nn
import torch.nn.functional as F

class HRnet_fam_ocr(nn.Module):
    def __init__(self, config, **kwargs):
        super(HRnet_fam_ocr, self).__init__()
        self.backbone = get_HRnet(config, upsample=False, **kwargs)
        self.fam1 = get_fam()  #
        self.fam2 = get_fam()
        self.fam3 = get_fam()
        #self.fam4 = get_fam()
        self.ocr = get_ocr(last_inp_channels=256)

    def forward(self, x):
        #input_size = x.shape[2:]
        x0, x1, x2, f = self.backbone(x)    #各自的分辨率128 64 32 16
        f = self.fam1([x2, f])
        print('-',f.shape)
        f = x2 + f
        print('-', f.shape)
        f = self.fam2([x1, f])
        f = x1 + f
        print('-', f.shape)
        f = self.fam3([x0, f])
        f = x0 + f
        print('-', f.shape)
        out = self.ocr(f)
        output = F.interpolate(out, size=(512,512), mode='bilinear', align_corners=True)
        return output

def get_net(cfg, **kwargs):
    model = HRnet_fam_ocr(cfg, **kwargs)
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
    print(out.shape,t1-t)  #时间比8.762192487716675 ： 15.435196876525879  －》 448 ： 512
    print(count_param(model))
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


#67122268
#torch.Size([1, 14, 512, 512]) 16.475454330444336
#68324092
#