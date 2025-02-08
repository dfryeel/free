# -*- coding: utf-8 -*-
import torch
from torch import nn, autograd, optim, Tensor, cuda
from torch.nn import functional as F
from torch.autograd import Variable

from lib import *
out_channel=64
from agent import *
class Network(nn.Module):
    def __init__(self, config, encoder, feat):
        super(Network, self).__init__()

        self.encoder = encoder
        self.GFM1 = ConvMlp(feat[0], out_channel)
        self.GFM2 = ConvMlp(feat[1], out_channel)
        self.GFM3 = ConvMlp(feat[2], out_channel)
        self.GFM4 = ConvMlp(feat[3], out_channel)
      
        
        self.aspp=AgentAttention(out_channel,14*14)
        self.at1=Attention(out_channel,4,True)
        self.at2=Attention(out_channel,4,True)
        self.at3=Attention(out_channel,4,True)
        self.at4=Attention(out_channel,4,True)
        
        
        self.conv1=nn.Conv2d(out_channel,1,1)
        self.conv2=nn.Conv2d(out_channel,1,1)
        self.conv3=nn.Conv2d(out_channel,1,1)
        self.conv4=nn.Conv2d(out_channel,1,1)
        self.conve1=nn.Conv2d(out_channel,1,1)
        self.conve2=nn.Conv2d(out_channel,1,1)
        self.conve3=nn.Conv2d(out_channel,1,1)
        self.conve4=nn.Conv2d(out_channel,1,1)
        
        self.upx2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.upx6 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
        self.downx2 = nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=True)
        self.downx4 = nn.Upsample(scale_factor=0.25, mode='bilinear', align_corners=True)
        self.dePixelShuffle = torch.nn.PixelShuffle(2)
        self.dePixelShuffle4 = torch.nn.PixelShuffle(4)
    def forward(self, x):

        xs= self.encoder(x)
        x1=xs[0]
        x2=xs[1]
        x3=xs[2]
        x4=xs[3]
        
        x1 = self.GFM1(x1)
        x2 = self.GFM2(x2)
        x3 = self.GFM3(x3)
        x4 = self.GFM4(x4)
        
 
        # feature_f, feature_r,f_h,f_e
        # print(x4.size())
        o4=x4

        o4=self.at4(o4)
        o3=self.at3(x3+self.upx2(o4))
        o2=self.at2(x2+self.upx2(o3))
        o1=self.at1(x1+self.upx2(o2))

        o11=self.conv1(o1)
        o22=self.conv2(o2)
        o33=self.conv3(o3)
        o44=self.conv4(o4)

        # o3 = nn.functional.interpolate(o3, size=o2.size()[2:], mode='bilinear', align_corners=True)
        
        return o11,o22,o33,o44


from Res2Net import res2net50_v1b_26w_4s
if __name__ == '__main__':
    x = torch.rand(1, 3, 448, 448)
    config = {}
    schedule = {}
    resnet = res2net50_v1b_26w_4s(pretrained=False)
    # encoder = EfficientNet.from_pretrained('efficientnet-b7')

    # fl = [48, 80, 160, 640]
    config = {}
    schedule = {}
    # resnet =ResNet().initialize()
    
    encoder = resnet
    fl = [256, 512, 1024, 2048]
    # x1= torch.rand(1, 3, 384, 384)
    # x1,x2,x3,x4=encoder(x)
    # print(x1.size(),x2.size(),x3.size(),x4.size())
    net = Network(config,encoder,fl)
    o11,o22,o33,e11=net(x)
    print(o11.size())
