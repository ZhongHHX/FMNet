import torch
import torch.nn as nn
import torch.nn.functional as F


class ModuleHelper:

    @staticmethod
    def BNReLU(num_features, bn_type=None, **kwargs):
        return nn.Sequential(
            nn.BatchNorm2d(num_features, **kwargs),
            nn.ReLU()
        )

    @staticmethod
    def BatchNorm2d(*args, **kwargs):
        return nn.BatchNorm2d



class _ObjectAttentionBlock(nn.Module):
    '''
    The basic implementation for object context block
    Input:
        N X C X H X W
    Parameters:
        in_channels       : the dimension of the input feature map
        key_channels      : the dimension after the key/query transform
        scale             : choose the scale to downsample the input feature maps (save memory cost)
        bn_type           : specify the bn type
    Return:
        N X C X H X W
    '''
    def __init__(self,
                 in_channels,
                 key_channels,
                 scale=1,
                 bn_type=None):
        super(_ObjectAttentionBlock, self).__init__()
        self.scale = scale
        self.in_channels = in_channels
        self.key_channels = key_channels
        self.pool = nn.MaxPool2d(kernel_size=(scale, scale))
        self.f_pixel = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.key_channels,
                kernel_size=1, stride=1, padding=0, bias=False),
            ModuleHelper.BNReLU(self.key_channels, bn_type=bn_type),
            nn.Conv2d(in_channels=self.key_channels, out_channels=self.key_channels,
                kernel_size=1, stride=1, padding=0, bias=False),
            ModuleHelper.BNReLU(self.key_channels, bn_type=bn_type),
        )
        self.f_object = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.key_channels,
                kernel_size=1, stride=1, padding=0, bias=False),
            ModuleHelper.BNReLU(self.key_channels, bn_type=bn_type),
            nn.Conv2d(in_channels=self.key_channels, out_channels=self.key_channels,
                kernel_size=1, stride=1, padding=0, bias=False),
            ModuleHelper.BNReLU(self.key_channels, bn_type=bn_type),
        )
        self.f_down = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.key_channels,
                kernel_size=1, stride=1, padding=0, bias=False),
            ModuleHelper.BNReLU(self.key_channels, bn_type=bn_type),
        )
        self.f_up = nn.Sequential(
            nn.Conv2d(in_channels=self.key_channels, out_channels=self.in_channels,
                kernel_size=1, stride=1, padding=0, bias=False),
            ModuleHelper.BNReLU(self.in_channels, bn_type=bn_type),
        )

    def forward(self, x, proxy):
        batch_size, h, w = x.size(0), x.size(2), x.size(3)
        if self.scale > 1:
            x = self.pool(x)
        #print(x.shape,proxy.shape, self.in_channels)
        query = self.f_pixel(x).view(batch_size, self.key_channels, -1)

        query = query.permute(0, 2, 1)
        key = self.f_object(proxy).view(batch_size, self.key_channels, -1)

        value = self.f_down(proxy).view(batch_size, self.key_channels, -1)
        value = value.permute(0, 2, 1)

        sim_map = torch.matmul(query, key)

        sim_map = (self.key_channels**-.5) * sim_map
        sim_map = F.softmax(sim_map, dim=-1)

        # add bg context ...
        context = torch.matmul(sim_map, value)
        context = context.permute(0, 2, 1).contiguous()
        context = context.view(batch_size, self.key_channels, *x.size()[2:])
        context = self.f_up(context)
        if self.scale > 1:
            context = F.interpolate(input=context, size=(h, w), mode='bilinear', align_corners=True)
        #print(context.shape,'----')
        return context

class ObjectAttentionBlock2D(_ObjectAttentionBlock):
    def __init__(self,
                 in_channels,
                 key_channels,
                 scale=1,
                 bn_type=None):
        super(ObjectAttentionBlock2D, self).__init__(in_channels,
                                                     key_channels,
                                                     scale,
                                                     bn_type=bn_type)

class SpatialGather_Module(nn.Module):
    """
        Aggregate the context features according to the initial
        predicted probability distribution.
        Employ the soft-weighted method to aggregate the context.
    """
    def __init__(self, cls_num=0, scale=1):
        super(SpatialGather_Module, self).__init__()
        self.cls_num = cls_num
        self.scale = scale

    def forward(self, feats, probs):

        batch_size, c, h, w = probs.size(0), probs.size(1), probs.size(2), probs.size(3)
        probs = probs.view(batch_size, c, -1)
        feats = feats.view(batch_size, feats.size(1), -1)
        feats = feats.permute(0, 2, 1) # batch x hw x c
        probs = F.softmax(self.scale * probs, dim=2)# batch x k x hw
        ocr_context = torch.matmul(probs, feats)\
        .permute(0, 2, 1).unsqueeze(3)# batch x k x c

        return ocr_context


class SpatialOCR_Module(nn.Module):
    """
    Implementation of the OCR module:
    We aggregate the global object representation to update the representation for each pixel.
    """
    def __init__(self,
                 in_channels,
                 key_channels,
                 out_channels,
                 scale=1,
                 dropout=0.1,
                 bn_type=None):
        super(SpatialOCR_Module, self).__init__()
        self.object_context_block = ObjectAttentionBlock2D(in_channels,
                                                           key_channels,
                                                           scale,
                                                           bn_type)
        _in_channels = 2 * in_channels

        self.conv_bn_dropout = nn.Sequential(
            nn.Conv2d(_in_channels, out_channels, kernel_size=1, padding=0, bias=False),
            ModuleHelper.BNReLU(out_channels, bn_type=bn_type),
            nn.Dropout2d(dropout)
        )

    def forward(self, feats, proxy_feats):
        context = self.object_context_block(feats, proxy_feats)

        output = self.conv_bn_dropout(torch.cat([context, feats], 1))

        return output


class Object_RelationNet(nn.Module):
    #def __init__(self,last_inp_channels=24,relu_inplace=True,num_classes=2,ocr_mid_channels=256,ocr_key_channels=256):
    #down_channels : 为了进行三通道注意力融合，选择不进行通道维数降低的参数
    #down_channels = True 表示下降   False 表示不降
    def __init__(self,last_inp_channels=720,relu_inplace=True,num_classes=14,ocr_mid_channels=256,ocr_key_channels=256,down_channels=True):
        #ocr
        super(Object_RelationNet, self).__init__()
        self.down_channels_to_numclasses = down_channels
        self.aux_head = nn.Sequential(
            nn.Conv2d(last_inp_channels, last_inp_channels,
                      kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(last_inp_channels),
            nn.ReLU(inplace=relu_inplace),
            nn.Conv2d(last_inp_channels, num_classes,
                      kernel_size=1, stride=1, padding=0, bias=True)
        )
        self.conv3x3_ocr = nn.Sequential(
            nn.Conv2d(last_inp_channels, ocr_mid_channels,
                      kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(ocr_mid_channels),
            nn.ReLU(inplace=relu_inplace),
        )
        self.ocr_gather_head = SpatialGather_Module(num_classes)

        self.ocr_distri_head = SpatialOCR_Module(in_channels=ocr_mid_channels,
                                                 key_channels=ocr_key_channels,
                                                 out_channels=ocr_mid_channels,
                                                 scale=1,
                                                 dropout=0.05,
                                                 )
        self.cls_head = nn.Conv2d(
            ocr_mid_channels, num_classes, kernel_size=1, stride=1, padding=0, bias=True)

        self.aux_head = nn.Sequential(
            nn.Conv2d(last_inp_channels, last_inp_channels,
                      kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(last_inp_channels),
            nn.ReLU(inplace=relu_inplace),
            nn.Conv2d(last_inp_channels, num_classes,
                      kernel_size=1, stride=1, padding=0, bias=True)
        )

    def forward(self, feats):
        # ocr
        #conv(1x1)+BNRelu+conv(1x1xnumclasses)  相当于粗分割
        #out_aux_seg = []

        out_aux = self.aux_head(feats)
        # compute contrast feature
        #conv (3x3x512) same conv, ocr_mid_channels=512, 对应
        feats = self.conv3x3_ocr(feats)

        context = self.ocr_gather_head(feats, out_aux)
        #print('context:',context.shape)  #([1, 512, 10, 1])
        feats = self.ocr_distri_head(feats, context)
        #print('distri_head:',feats.shape)#([1, 512, 128, 128])

        if self.down_channels_to_numclasses == True:
            out = self.cls_head(feats)     #这里是直接将 ocr_mid_channels 数 通过1x1卷积降到num_classes
            # out_aux_seg.append(out_aux)
            # out_aux_seg.append(out)
            out_aux_seg = out_aux + out
            return out_aux_seg
        elif self.down_channels_to_numclasses == False:
            return feats




def get_caa(last_inp_channels=720, ocr_mid_channels=256, ocr_key_channels=256, down_channels=True):
    model = Object_RelationNet(last_inp_channels=last_inp_channels,
                               ocr_mid_channels=ocr_mid_channels,
                               ocr_key_channels=ocr_key_channels,
                               down_channels=down_channels)
    return model


# feats = torch.cat([x[0], x1, x2, x3], dim=1)
# out_aux_seg = []
# ocr
#out_aux = self.aux_head(feats)
# # compute contrast feature
#feats = self.conv3x3_ocr(feats)
#
# context = self.ocr_gather_head(feats, out_aux)
# feats = self.ocr_distri_head(feats, context)
#
# out = self.cls_head(feats)
#
# out_aux_seg.append(out_aux)
# out_aux_seg.append(out)
#
# return out_aux_seg

if __name__=='__main__':
    #torch.Size([1, 48, 128, 128]) torch.Size([1, 96, 64, 64]) torch.Size([1, 192, 32, 32]) torch.Size([1, 384, 16, 16])
    model = Object_RelationNet(last_inp_channels=256, down_channels=False)
                               #, ocr_mid_channels=192, ocr_key_channels=192, down_channels=False)

    #x = torch.randn([1,24,256,256])
    x = torch.randn([1, 256, 128, 128])
    out_seg = model(x)
    print(out_seg.shape)  #out_seg[0]是out_aux   out_seg[1]是out
    total_params = sum(p.numel() for p in model.parameters())
    print('总参数个数：{}'.format(total_params))   #2721772   68046892

