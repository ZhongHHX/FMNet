from module.hrnet import get_HRnet
from module.OcrModule import get_ocr
import torch
import torch.nn as nn
from module.FeatureAlignment import get_FeatureAlign_V2
import torch.nn.functional as F

class HRnet_ocr(nn.Module):
    def __init__(self, config, **kwargs):
        super(HRnet_ocr, self).__init__()
        self.backbone = get_HRnet(config, upsample=False,**kwargs)
        self.cls_head = nn.Conv2d(
            384, 14, kernel_size=1, stride=1, padding=0, bias=True)   #384是x0,x1,x2,x3特征对齐后的通道数，然后14是类别数

    def Feature_Align(self,input=None):  # list: input

        x0 = input[0]
        x1 = input[1]
        x2 = input[2]
        x3 = input[3]
        model_1 = get_FeatureAlign_V2(x2.size()[1], x3.size()[1]).cuda()
        output1 = model_1(x2, x3)
        model_2 = get_FeatureAlign_V2(x1.size()[1], output1.size()[1]).cuda()
        output1 = model_2(x1, output1)
        model_3 = get_FeatureAlign_V2(x0.size()[1], output1.size()[1]).cuda()
        output1 = model_3(x0, output1)
        return output1

    def forward(self, x):
        x0, x1, x2, x3 = self.backbone(x)
        c1, c2, c3, c4 = x0.size()[1], x1.size()[1], x2.size()[1], x3.size()[1]  #通道数
        #print(x0.shape, x1.shape, x2.shape, x3.shape)
        ocr0 = get_ocr(last_inp_channels=c1,
                       ocr_mid_channels=c1,
                       ocr_key_channels=c1,
                       down_channels=False).cuda()
        ocr1 = get_ocr(last_inp_channels=c2,
                       ocr_mid_channels=c2,
                       ocr_key_channels=c2,
                       down_channels=False).cuda()
        ocr2 = get_ocr(last_inp_channels=c3,
                       ocr_mid_channels=c3,
                       ocr_key_channels=c3,
                       down_channels=False).cuda()
        ocr3 = get_ocr(last_inp_channels=c4,
                       ocr_mid_channels=c4,
                       ocr_key_channels=c4,
                       down_channels=False).cuda()
        x = []

        x.append(ocr0(x0))
        x.append(ocr1(x1))
        x.append(ocr2(x2))
        x.append(ocr3(x3))

        feats = self.Feature_Align(x)
        output = self.cls_head(feats)
        output = F.interpolate(output, size=(512,512), mode='bilinear', align_corners=True)
        # input_size = x.shape[2:]
        # x = self.backbone(x)
        # out = self.ocr(x)
        # output = F.interpolate(out, size=input_size, mode='bilinear', align_corners=True)
        return output

def get_net(cfg, **kwargs):
    model = HRnet_ocr(cfg, **kwargs)
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
    #print('总参数个数：{}'.format(total_params))   #68046892   68324092  28001695
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