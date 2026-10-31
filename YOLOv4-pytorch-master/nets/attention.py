import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from collections import OrderedDict
import torch.nn.utils.parametrize as parametrize


# 卷积层的bias建议设置成False
def conv2d(filter_in, filter_out, kernel_size, groups=1, stride=1):
    pad = (kernel_size - 1) // 2 if kernel_size else 0
    return nn.Sequential(OrderedDict([
        ("conv", nn.Conv2d(filter_in, filter_out, kernel_size=kernel_size, stride=stride, padding=pad, groups=groups, bias=False)),
        ("bn", nn.BatchNorm2d(filter_out)),
        ("relu", nn.LeakyReLU(0.1)),
    ]))

class ClampParam(nn.Module):
    def forward(self, x):
        return x.clamp(0, 1)



class ConvSelfAttention(nn.Module):
    def __init__(self, in_channels, dropout=0.1, use_residual=True):
        super().__init__()
        self.use_residual = use_residual

        self.query_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False)
        self.key_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False)
        self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False)

        self.register_buffer('scale', torch.tensor(math.sqrt(in_channels)))
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Conv2d(in_channels, in_channels, kernel_size=1)


    def forward(self, x1, x2, x3, x_all):
        B, C, H, W = x1.shape

        # 投影
        q = self.query_conv(x3)  # (B, C, H, W)
        k = self.key_conv(x2)  # (B, C, H, W)
        v = self.value_conv(x1)  # (B, C, H, W)

        # 重塑为序列格式 (B, C, H*W)
        q = q.view(B, C, -1)  # (B, C, H*W)
        k = k.view(B, C, -1)  # (B, C, H*W)
        v = v.view(B, C, -1)  # (B, C, H*W)

        # 计算注意力分数 (B, H*W, H*W)
        attention_scores = torch.bmm(q.transpose(1, 2).contiguous(), k) / self.scale  # (B, H*W, H*W)
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        # 应用注意力 (B, C, H*W)
        out = torch.bmm(v, attention_weights.transpose(1, 2).contiguous())  # (B, C, H*W)

        # 重塑回原始形状
        out = out.view(B, C, H, W)
        out = self.out_proj(out)

        if self.use_residual:
            out = out + x_all

        return out

class Channel_Attention(nn.Module):
    def __init__(self, in_planes, ratio=8):
        super(Channel_Attention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # 利用1x1卷积代替全连接
        self.fc1   = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.LeakyReLU(0.1)
        self.fc2   = nn.Conv2d(in_planes // ratio, in_planes * 3, 1, bias=False)
        self.alpha = nn.Parameter(torch.tensor(0.5))
        parametrize.register_parametrization(self, 'alpha', ClampParam())
    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = self.alpha * avg_out + (1-self.alpha) * max_out
        return out


class SPPConv(nn.Module):
    def __init__(self, in_channels, conv_kernels=[3, 5, 7]):
        super(SPPConv, self).__init__()
        self.scale_conv = nn.ModuleList([
            conv2d(in_channels, in_channels, kernel_size)
            for kernel_size in conv_kernels
        ])
        self.ch_attention = Channel_Attention(in_channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.size()
        input = x
        multi_scale_feats = [conv(x) for conv in self.scale_conv]
        all_feature = multi_scale_feats[0] + multi_scale_feats[1] + multi_scale_feats[2]
        weight = self.ch_attention(all_feature)
        x3_weight, x5_weight, x7_weight = torch.split(weight, c, dim=1)
        x3_weight = self.sigmoid(x3_weight)
        x5_weight = self.sigmoid(x5_weight)
        x7_weight = self.sigmoid(x7_weight)
        # x3 = x3_weight * multi_scale_feats[0] + multi_scale_feats[0]
        # x5 = x5_weight * multi_scale_feats[1] + multi_scale_feats[1]
        # x7 = x7_weight * multi_scale_feats[2] + multi_scale_feats[2]
        x3 = x3_weight * multi_scale_feats[0] + input
        x5 = x5_weight * multi_scale_feats[1] + input
        x7 = x7_weight * multi_scale_feats[2] + input
        return x3, x5, x7, input
#
#
# # # Coord Attention
# import torch.nn as nn
# import torch
class h_sigmoid(nn.Module):
    def __init__(self):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=True)
    def forward(self, x):
        return self.relu(x + 3) / 6

class h_swish(nn.Module):
    def __init__(self):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid()
    def forward(self, x):
        return x * self.sigmoid(x)

class CoordAtt(nn.Module):
    def __init__(self, inp, oup, groups=32):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp//groups)
        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.conv2 = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv3 = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

        self.relu = h_swish()

    def forward(self, x):
        # identity = x
        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0,1,3,2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.relu(y)
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0,1,3,2)

        x_h = self.conv2(x_h).sigmoid()
        x_w = self.conv3(x_w).sigmoid()

        x_h = x_h.expand(-1, -1, h, w)
        x_w = x_w.expand(-1, -1, h, w)

        # y = identity * x_h * x_w
        weight = x_h * x_w
        return weight