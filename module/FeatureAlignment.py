import torch
import torch.nn.functional as F
import torch.nn as nn


from module.Dcnv2 import DeformableConv2d as dcn_v2
import math



class Conv2d(torch.nn.Conv2d):
    """
    A wrapper around :class:`torch.nn.Conv2d` to support empty inputs and more features.
    """

    def __init__(self, *args, **kwargs):
        """
        Extra keyword arguments supported in addition to those in `torch.nn.Conv2d`:

        Args:
            norm (nn.Module, optional): a normalization layer
            activation (callable(Tensor) -> Tensor): a callable activation function

        It assumes that norm layer is used before activation.
        """
        norm = kwargs.pop("norm", None)
        activation = kwargs.pop("activation", None)
        super().__init__(*args, **kwargs)

        self.norm = norm
        self.activation = activation

    def forward(self, x):
        # torchscript does not support SyncBatchNorm yet
        # https://github.com/pytorch/pytorch/issues/40507
        # and we skip these codes in torchscript since:
        # 1. currently we only support torchscript in evaluation mode
        # 2. features needed by exporting module to torchscript are added in PyTorch 1.6 or
        # later version, `Conv2d` in these PyTorch versions has already supported empty inputs.
        if not torch.jit.is_scripting():
            if x.numel() == 0 and self.training:
                # https://github.com/pytorch/pytorch/issues/12013
                assert not isinstance(
                    self.norm, torch.nn.SyncBatchNorm
                ), "SyncBatchNorm does not support empty inputs!"

        x = F.conv2d(
            x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups
        )
        if self.norm is not None:
            x = self.norm(x)
        if self.activation is not None:
            x = self.activation(x)
        return x



class FeatureSelectionModule(nn.Module):
    def __init__(self, in_chan, out_chan, norm="GN"):
        super(FeatureSelectionModule, self).__init__()
        self.conv_atten = Conv2d(in_chan, in_chan, kernel_size=1, bias=False, norm=None)
        self.sigmoid = nn.Sigmoid()
        self.conv = Conv2d(in_chan, out_chan, kernel_size=1, bias=False, norm=None)
        # weight_init.c2_xavier_fill(self.conv_atten)
        # weight_init.c2_xavier_fill(self.conv)

    def forward(self, x):
        atten = self.sigmoid(self.conv_atten(F.avg_pool2d(x, x.size()[2:])))
        feat = torch.mul(x, atten)
        x = x + feat
        feat = self.conv(x)
        return feat


class FeatureAlign_V2(nn.Module):  # FaPN full version
    def __init__(self, in_nc=128, out_nc=128, norm=None):
        super(FeatureAlign_V2, self).__init__()
        self.lateral_conv = FeatureSelectionModule(in_nc, out_nc, norm="")
        self.offset = Conv2d(out_nc * 2, out_nc, kernel_size=1, stride=1, padding=0, bias=False, norm=norm)
        self.dcpack_L2 = dcn_v2(in_channels=out_nc, out_channels=out_nc)
        self.relu = nn.ReLU(inplace=True)
        #weight_init.c2_xavier_fill(self.offset)



    def forward(self, feat_l, feat_s, main_path=None):
        HW = feat_l.size()[2:]
        feat_s_channel = feat_s.size()[1]
        if feat_l.size()[2:] != feat_s.size()[2:]:
            feat_up = F.interpolate(feat_s, HW, mode='bilinear', align_corners=False)
        else:
            feat_up = feat_s
        # if feat_l.size()[1] != feat_s_channel:
        #     fe
        feat_arm = self.lateral_conv(feat_l)  # 0~1 * feats
        # offset = self.offset(torch.cat([feat_arm, feat_up * 2], dim=1))  # concat for offset by compute the dif
        # feat_align = self.relu(self.dcpack_L2([feat_up, offset], main_path))  # [feat, offset]
        offset = self.offset(torch.cat([feat_arm, feat_up * 2], dim=1))
        feat_align = self.dcpack_L2(offset)
        return feat_align + feat_arm

def get_FeatureAlign_V2(in_channels, out_channesl):
    model = FeatureAlign_V2(in_nc=in_channels, out_nc=out_channesl)
    return model

# torch.Size([1, 48, 128, 128]) torch.Size([1, 96, 64, 64]) torch.Size([1, 192, 32, 32]) torch.Size([1, 384, 16, 16])
if __name__ == '__main__':
    x = torch.randn([1,192,32,32]).float()
    y = torch.randn([1,384,16,16]).float()
    model = FeatureAlign_V2(in_nc=192, out_nc=384)
    out = model(x,y)
    print(out.shape)
