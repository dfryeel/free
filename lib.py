# -*- coding: utf-8 -*-
import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable
# -*- coding: utf-8 -*-
import numpy as np
from einops import rearrange
class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads

        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.attn_drop = nn.Dropout(0.)

        self.attn1 = torch.nn.Parameter(torch.tensor([0.2]), requires_grad=True)
        self.attn2 = torch.nn.Parameter(torch.tensor([0.2]), requires_grad=True)
        self.attn3 = torch.nn.Parameter(torch.tensor([0.2]), requires_grad=True)
        self.attn4 = torch.nn.Parameter(torch.tensor([0.2]), requires_grad=True)

    def forward(self, x):
        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        _, _, C, _ = q.shape  # C=30，即通道数

        mask1 = torch.zeros(b, self.num_heads, C, C, device=x.device, requires_grad=False)
        mask2 = torch.zeros(b, self.num_heads, C, C, device=x.device, requires_grad=False)
        mask3 = torch.zeros(b, self.num_heads, C, C, device=x.device, requires_grad=False)
        mask4 = torch.zeros(b, self.num_heads, C, C, device=x.device, requires_grad=False)

        attn = (q @ k.transpose(-2, -1)) * self.temperature  # b 1 C C

        index = torch.topk(attn, k=int(C/2), dim=-1, largest=True)[1]
        mask1.scatter_(-1, index, 1.)
        attn1 = torch.where(mask1 > 0, attn, torch.full_like(attn, float('-inf')))

        # print(111, mask1.scatter_(-1, index, 1.))

        index = torch.topk(attn, k=int(C*2/3), dim=-1, largest=True)[1]
        mask2.scatter_(-1, index, 1.)
        attn2 = torch.where(mask2 > 0, attn, torch.full_like(attn, float('-inf')))

        index = torch.topk(attn, k=int(C*3/4), dim=-1, largest=True)[1]
        mask3.scatter_(-1, index, 1.)
        attn3 = torch.where(mask3 > 0, attn, torch.full_like(attn, float('-inf')))

        index = torch.topk(attn, k=int(C*4/5), dim=-1, largest=True)[1]
        mask4.scatter_(-1, index, 1.)
        attn4 = torch.where(mask4 > 0, attn, torch.full_like(attn, float('-inf')))

        attn1 = attn1.softmax(dim=-1)  # [1 6 30 30]
        attn2 = attn2.softmax(dim=-1)
        attn3 = attn3.softmax(dim=-1)
        attn4 = attn4.softmax(dim=-1)

        out1 = (attn1 @ v)
        out2 = (attn2 @ v)
        out3 = (attn3 @ v)
        out4 = (attn4 @ v)

        out = out1 * self.attn1 + out2 * self.attn2 + out3 * self.attn3 + out4 * self.attn4

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out
class LayerNorm2d(nn.Module):#二维层归一化（Layer Normalization）层
    """
    From https://github.com/facebookresearch/detectron2/blob/main/detectron2/layers/batch_norm.py
    Itself from https://github.com/facebookresearch/ConvNeXt/blob/d1fa8f6fef0a165b27399986cc2bdacc92777e40/models/convnext.py#L119
    """

    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x
class ConvMlp(nn.Module):#卷积多层感知机 特征提取和变换
    def __init__(
        self,
        in_features: int,
        out_features: int = None,
        mlp_times: float = 1,
        act_layer: nn.Module = nn.GELU,
        norm_layer: nn.Module = LayerNorm2d,
        bias: bool = False,
        drop: float = 0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = int(mlp_times * in_features)

        self.norm = norm_layer(in_features) if norm_layer else nn.Identity()
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1, bias=bias)
        self.dwconv = nn.Conv2d(hidden_features, hidden_features, 3, 1, 1, groups=hidden_features, bias=bias)
        self.act = act_layer()
        self.drop = nn.Dropout(drop)
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1, bias=bias)

    def forward(self, x):
        x = self.norm(x)
        x = self.fc1(x)
        x = self.dwconv(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        return x
class OctaveConv(nn.Module):#八分卷积层
    #多尺度特征处理——高频部分和低频部分，在这两部分上应用不同的卷积操作来减少计算量。
    def __init__(self, in_channels, out_channels, kernel_size, alpha=0.5, stride=1, padding=1, dilation=1,
                 groups=1, bias=False):
        super(OctaveConv, self).__init__()
        kernel_size = kernel_size[0]
        self.h2g_pool = nn.AvgPool2d(kernel_size=(2, 2), stride=2)
        self.upsample = torch.nn.Upsample(scale_factor=2, mode='nearest')
        self.stride = stride
        self.l2l = torch.nn.Conv2d(int(alpha * in_channels), int(alpha * out_channels),
                                   kernel_size, 1, padding, dilation, groups, bias)
        self.l2h = torch.nn.Conv2d(int(alpha * in_channels), out_channels - int(alpha * out_channels),
                                   kernel_size, 1, padding, dilation, groups, bias)
        self.h2l = torch.nn.Conv2d(in_channels - int(alpha * in_channels), int(alpha * out_channels),
                                   kernel_size, 1, padding, dilation, groups, bias)
        self.h2h = torch.nn.Conv2d(in_channels - int(alpha * in_channels),
                                   out_channels - int(alpha * out_channels),
                                   kernel_size, 1, padding, dilation, groups, bias)

    def forward(self, x):
        X_h, X_l = x

        if self.stride == 2:
            X_h, X_l = self.h2g_pool(X_h), self.h2g_pool(X_l)

        X_h2l = self.h2g_pool(X_h)

        X_h2h = self.h2h(X_h)
        X_l2h = self.l2h(X_l)

        X_l2l = self.l2l(X_l)
        X_h2l = self.h2l(X_h2l)

        # X_l2h = self.upsample(X_l2h)
        X_l2h = F.interpolate(X_l2h, (int(X_h2h.size()[2]),int(X_h2h.size()[3])), mode='bilinear')
        # print('X_l2h:{}'.format(X_l2h.shape))
        # print('X_h2h:{}'.format(X_h2h.shape))
        X_h = X_l2h + X_h2h
        X_l = X_h2l + X_l2l

        return X_h,X_l
class FA(nn.Module):#特征增强模块
    def __init__(self, in_channels,out_channels):
        super(FA, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, dilation=2, padding=2)
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=3, dilation=5, padding=5)
        self.F1 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        )
        self.F2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        )
        self.F3 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        )
        self.F4 = nn.Conv2d(out_channels*2, out_channels, kernel_size=3, padding=1)
        self.conv3=BasicConv2d(out_channels*2,out_channels,3,padding=1)
        
       
    def forward(self, feature_map):
        f3x3=self.conv1(feature_map)
        f5x5=self.conv2(feature_map)
        
        f1 = self.F1(f3x3)
        f2 =self.F2(f1 + f5x5)
        f3=self.F3(f2+f1)

        x=torch.cat((f2,f3),dim=1)
        f23=self.F4(x)
        f4=f3x3*f3
        f5=f5x5*f23
        feature=f4+f5
        x=torch.cat((f3x3,feature),dim=1)
        # print(x.size())
        feature=self.conv3(x)
        return feature
class getAlpha(nn.Module):#注意力机制模块，该模块通过结合全局平均池化和全局最大池化来生成一个注意力权重图（alpha）
    def __init__(self, in_channels):
        super(getAlpha, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_channels * 2, in_channels, kernel_size=1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_channels, 1, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)
class BasicConv2d(nn.Module):#基本的卷积模块
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, need_relu=True, bn=nn.BatchNorm2d):
        super(BasicConv2d, self).__init__()
#深度可分离卷积
        self.depthwise = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,dilation=dilation, padding=padding, bias=False)
        self.bn = bn(out_channels)
        self.relu = nn.LeakyReLU()
        self.need_relu = need_relu

    def forward(self, x):
        x = self.depthwise(x)
        x = self.bn(x)
        if self.need_relu:
            x = self.relu(x)
        return x
class HLML(nn.Module):#多尺度特征融合模块
    def __init__(self, channels):
        super(HLML, self).__init__()

        self.conv1 = nn.Conv2d(channels[0], channels[1], 3,padding=1)
        self.conv2=BasicConv2d(channels[1],channels[2],3,padding=1)
        self.conv3=BasicConv2d(channels[2],channels[3],3,padding=1)
        
        self.upx2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.upx4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.downx2 = nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=True)
        self.downx4 = nn.Upsample(scale_factor=0.25, mode='bilinear', align_corners=True)
    def forward(self, x1,x2,x3):
        x11=self.upx2(self.downx2(x1))
        x1=self.conv1(x1-x11)

        x22=self.upx2(self.downx2(x2))
        x2=x2-x22

        f=self.downx2(x1)*x2
        f1=self.conv2(f)

        x33=self.upx2(self.downx2(x3))
        x3=x3-x33

        f2=self.downx2(f1)*x3
        re1=self.conv3(f2)
        
        
        return f2,self.downx2(re1)



class BasicDeConv2d(nn.Module):#实现一个带有转置卷积（也称为反卷积）的层，通常用于上采样特征图
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, out_padding=0, need_relu=True,
                 bn=nn.BatchNorm2d):
        super(BasicDeConv2d, self).__init__()
        self.conv = nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                       stride=stride, padding=padding, dilation=dilation, output_padding=out_padding, bias=False)
        self.bn = bn(out_channels)
        self.relu = nn.LeakyReLU()
        self.need_relu = need_relu
        self.pointwise = nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False)


    def forward(self, x):
        x = self.conv(x)
        x=self.pointwise(x)
        x = self.bn(x)
        if self.need_relu:
            x = self.relu(x)
        return x
class TRA32(nn.Module):#注意力机制
    def __init__(self, channels):
        super(TRA32, self).__init__()
        self.conv_q = nn.Conv2d(channels, channels, kernel_size=1)
        self.conv_k = nn.Conv2d(channels, channels, kernel_size=1)
        self.conv_v = nn.Conv2d(channels, channels, kernel_size=1)
        self.softmax = nn.Softmax(-1)
        self.output_conv = nn.Conv2d(channels, channels, kernel_size=1)
    

    def forward(self, x1):  # 前面是较浅层特征，后面是修正后的较深层特征(64,22,22) & (64,22,22)
        # print("TRAM")
        # print(x1.size(),x2.size())
        batch_size, C, height, width = x1.shape
        # Compute queries, keys, values
        # print(x1.size())
        q = self.conv_q(x1).view(batch_size, -1, height * width) 
        k = self.conv_k(x1).view(batch_size, -1, height * width)
        v = self.conv_v(x1).view(batch_size, -1, height * width)
        # Compute attention
        x=torch.bmm(q.permute(0, 2, 1), k)
        # print(x.size())
        attn = self.softmax(x)
        # print(attn)
        res=torch.bmm(v,attn).view(batch_size, -1, height ,width)
        
        feature=self.output_conv(res)+res
        
        return feature
class SAM(nn.Module):#通道注意力机制，称为空间注意力模块
    def __init__(self, ch_in=32, reduction=16):#降维因子
        super(SAM, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 全局自适应池化
        self.fc = nn.Sequential(#计算通道注意力权重
            nn.Linear(ch_in, ch_in // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(ch_in // reduction, ch_in, bias=False),
            nn.Sigmoid()
        )
        self.fc_wight = nn.Sequential(#计算空间权重
            nn.Linear(ch_in, ch_in // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(ch_in // reduction, 1, bias=False),
            nn.Sigmoid()
        )
        self.upx2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
    def forward(self, x_l, x_h):
        b, c, _, _ = x_h.size()
        y_h = self.avg_pool(x_h).view(b, c)  # squeeze操作
        h_weight = self.fc_wight(y_h)#计算空间权重
        y_h = self.fc(y_h).view(b, c, 1, 1)  # FC获取通道注意力权重，是具有全局信息的
        x_fusion_h = x_h * y_h.expand_as(x_h)
        x_fusion_h = torch.mul(x_fusion_h, h_weight.view(b, 1, 1, 1))
        ##################----------------------------------
        b, c, _, _ = x_l.size()
        y_l = self.avg_pool(x_l).view(b, c)  # squeeze操作
        l_weight = self.fc_wight(y_l)#计算空间权重
        # print('l_weight',l_weight.shape,l_weight)
        y_l = self.fc(y_l).view(b, c, 1, 1)  # FC获取通道注意力权重，是具有全局信息的
        # print('***this is y_l shape', y_l.shape)
        x_fusion_l = x_l * y_l.expand_as(x_l)
        x_fusion_l = torch.mul(x_fusion_l, l_weight.view(b, 1, 1, 1))
        #################-------------------------------
        # print('x_fusion_h shape, x_fusion_l shape,h_weight shape',x_fusion_h.shape,x_fusion_l.shape,h_weight.shape)
        x_fusion = x_fusion_l + x_fusion_h
        # x_fusion = x_fusion_l + x_fusion_h
        return x_fusion  # 注意力作用每一个通道上
class SA(nn.Module):#实现一种注意力机制，具体来说是一种空间注意力
    def __init__(self, channels):
        super(SA, self).__init__()
        self.sa = nn.Sequential(
            nn.Conv2d(channels, channels // 4, 3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 4, 1, 3, padding=1, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        out = self.sa(x)
        y = x * out
        return y

class CA(nn.Module):#通道注意力
    def __init__(self, lf=True):
        super(CA, self).__init__()
        self.ap = nn.AdaptiveAvgPool2d(1) if lf else nn.AdaptiveMaxPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=3, padding=(3 - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.ap(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y.expand_as(x)


class CFM(nn.Module):#特征融合和上采样操作。这个模块通过上采样深层特征并将其与浅层特征相加，然后通过一个卷积层来融合特征。

    def __init__(self,in_channel,out_channel):

        super().__init__()
        self.conv2 = BasicConv2d(in_channel, in_channel, 3,padding=1)
        self.upx2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
    def forward(self, f1,f2):
        f=f1+self.upx2(f2)
        
        out=self.conv2(f)
        return  out