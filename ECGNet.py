import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from geoseg.models.resnet import resnet34
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import timm


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, norm_layer=nn.BatchNorm2d, bias=False):
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride, padding=((stride - 1) + dilation * (kernel_size - 1)) // 2),
            norm_layer(out_channels),
            nn.ReLU6()
        )


class ConvBN(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, norm_layer=nn.BatchNorm2d, bias=False):
        super(ConvBN, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride, padding=((stride - 1) + dilation * (kernel_size - 1)) // 2),
            norm_layer(out_channels)
        )


class Conv(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, bias=False):
        super(Conv, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride, padding=((stride - 1) + dilation * (kernel_size - 1)) // 2)
        )


class SeparableConvBNReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1,
                 norm_layer=nn.BatchNorm2d):
        super(SeparableConvBNReLU, self).__init__(
            nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride, dilation=dilation,
                      padding=((stride - 1) + dilation * (kernel_size - 1)) // 2,
                      groups=in_channels, bias=False),
            norm_layer(out_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.ReLU6()
        )


class SeparableConvBN(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1,
                 norm_layer=nn.BatchNorm2d):
        super(SeparableConvBN, self).__init__(
            nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride, dilation=dilation,
                      padding=((stride - 1) + dilation * (kernel_size - 1)) // 2,
                      groups=in_channels, bias=False),
            norm_layer(out_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        )


class SeparableConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1):
        super(SeparableConv, self).__init__(
            nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride, dilation=dilation,
                      padding=((stride - 1) + dilation * (kernel_size - 1)) // 2,
                      groups=in_channels, bias=False),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        )


class Muti_Large_filed(nn.Module):
     def __init__(self, in_channel):
         super().__init__()
         self.pri = nn.Conv2d(in_channel, in_channel, 1, 1, 0, bias=True)
         self.k11 = nn.Conv2d(in_channel//4, in_channel//4, 11, stride=1, padding=5,bias=False)
         self.k7 = nn.Conv2d(in_channel//4, in_channel//4, 7, stride=1, padding=3,bias=False)
         self.k5 = nn.Conv2d(in_channel//4, in_channel//4, 5, stride=1, padding=2,bias=False)
         self.k3 = nn.Conv2d(in_channel//4, in_channel//4, 3, stride=1, padding=1, bias=False)
         self.k1 = nn.Conv2d(in_channel, in_channel, 1, 1, 0, bias=True)
     def forward(self, x):
         input = x
         x = self.pri(x)
         x_block = torch.chunk(x,4,1)
         x11 = self.k11(x_block[0])
         x7 = self.k7(x_block[1]+ x11)
         x5 = self.k5(x_block[2]+ x7)
         x3 = self.k3(x_block[3]+ x5)
         
         x =  self.k1(torch.cat([x11, x7, x5, x3],dim=1)) + input
         return x



class pool_change(nn.Module):
     def __init__(self, in_channel):
         super().__init__()
         self.pri_ver = nn.Conv2d(in_channel, in_channel, 1, 1, 0, bias=True)
         self.k11_1 = nn.Conv2d(in_channel//4, in_channel//4, (11, 1), stride=1, padding=(5,0),bias=False)
         self.k7_1  = nn.Conv2d(in_channel//4, in_channel//4, (7, 1), stride=1, padding=(3,0),bias=False)
         self.k5_1  = nn.Conv2d(in_channel//4, in_channel//4, (5, 1), stride=1, padding=(2,0),bias=False)
         self.k3_1  = nn.Conv2d(in_channel//4, in_channel//4, (3, 1), stride=1, padding=(1,0), bias=False)
         
         self.pri_hor = nn.Conv2d(in_channel, in_channel, 1, 1, 0, bias=True)
         self.k1_11 = nn.Conv2d(in_channel//4, in_channel//4, (1, 11), stride=1, padding=(0,5),bias=False)
         self.k1_7  = nn.Conv2d(in_channel//4, in_channel//4, (1, 7), stride=1, padding=(0,3),bias=False)
         self.k1_5  = nn.Conv2d(in_channel//4, in_channel//4, (1, 5), stride=1, padding=(0,2),bias=False)
         self.k1_3  = nn.Conv2d(in_channel//4, in_channel//4, (1, 3), stride=1, padding=(0,1), bias=False)
         
         #self.last = nn.Conv2d(in_channel, in_channel, 1, 1, 0, bias=True)
         
     def forward(self, x):
         b, c, h, w = x.shape
         x_ver_pri = self.pri_ver(x)
         x_hor_pri = self.pri_hor(x)
         
         x_ver_block = torch.chunk(x_ver_pri,4,1)
         k11_1 = self.k11_1(x_ver_block[0])
         k7_1 = self.k7_1(x_ver_block[1])
         k5_1 = self.k5_1(x_ver_block[2])
         k3_1 = self.k3_1(x_ver_block[3])
         x_ver =  torch.cat([k11_1, k7_1, k5_1, k3_1],dim=1)
         
         x_hor_block = torch.chunk(x_hor_pri,4,1)
         k1_11 = self.k1_11(x_hor_block[0])
         k1_7 = self.k1_7(x_hor_block[1])
         k1_5 = self.k1_5(x_hor_block[2])
         k1_3 = self.k1_3(x_hor_block[3])
         x_hor =  torch.cat([k1_11, k1_7, k1_5, k1_3],dim=1)
         
         out = torch.cat([x_ver, x_hor.permute(0,1,3,2)], dim=3)
         return out



class class_pool_change(nn.Module):
     def __init__(self, in_channel):
         super().__init__()
         self.pri_ver = nn.Conv2d(in_channel, in_channel, 1, 1, 0, bias=True)
         self.k11_1 = nn.Conv2d(in_channel//8, 1, (11, 1), stride=1, padding=(5,0),bias=False)
         self.k7_1  = nn.Conv2d(in_channel//8, 1, (7, 1), stride=1, padding=(3,0),bias=False)
         self.k5_1  = nn.Conv2d(in_channel//8, 1, (5, 1), stride=1, padding=(2,0),bias=False)
         self.k3_1  = nn.Conv2d(in_channel//8, 1, (3, 1), stride=1, padding=(1,0), bias=False)
         self.k1_ver  = nn.Conv2d(in_channel//2, 2, 3, stride=1, padding=1, bias=False)
         
         self.pri_hor = nn.Conv2d(in_channel, in_channel, 1, 1, 0, bias=True)
         self.k1_11 = nn.Conv2d(in_channel//8, 1, (1, 11), stride=1, padding=(0,5),bias=False)
         self.k1_7  = nn.Conv2d(in_channel//8, 1, (1, 7), stride=1, padding=(0,3),bias=False)
         self.k1_5  = nn.Conv2d(in_channel//8, 1, (1, 5), stride=1, padding=(0,2),bias=False)
         self.k1_3  = nn.Conv2d(in_channel//8, 1, (1, 3), stride=1, padding=(0,1), bias=False)
         self.k1_hor  = nn.Conv2d(in_channel//2, 2, 3, stride=1, padding=1, bias=False)
         
         self.last = nn.Conv2d(12, 6, 1, 1, 0, bias=True)
         
     def forward(self, x):
         b, c, h, w = x.shape
         x_ver_pri = self.pri_ver(x)
         x_hor_pri = self.pri_hor(x)
         
         x_ver_block = torch.chunk(x_ver_pri,8,1)
         k11_1 = self.k11_1(x_ver_block[0])
         k7_1 = self.k7_1(x_ver_block[1])
         k5_1 = self.k5_1(x_ver_block[2])
         k3_1 = self.k3_1(x_ver_block[3])
         k1_ver = self.k1_ver(torch.cat([x_ver_block[4], x_ver_block[5], x_ver_block[6], x_ver_block[7]],dim=1))
         x_ver =  torch.cat([k11_1, k7_1, k5_1, k3_1, k1_ver],dim=1)
         
         x_hor_block = torch.chunk(x_hor_pri,8,1)
         k1_11 = self.k1_11(x_hor_block[0])
         k1_7 = self.k1_7(x_hor_block[1])
         k1_5 = self.k1_5(x_hor_block[2])
         k1_3 = self.k1_3(x_hor_block[3])
         k1_hor = self.k1_hor(torch.cat([x_hor_block[4], x_hor_block[5], x_hor_block[6], x_hor_block[7]],dim=1))
         x_hor =  torch.cat([k1_11, k1_7, k1_5, k1_3, k1_hor],dim=1)
         
         
         
         out1 = torch.cat([x_ver, x_hor.permute(0,1,3,2)], dim=3)
         out2 = self.last(torch.cat([x_ver, x_hor.permute(0,1,3,2)], dim=1))
         
         return out1, out2




class Prior_Edge(nn.Module):
    def __init__(self, st1_inc, st2_inc, st3_inc, st4_inc, decode_channels, edge_channels=64):
        super().__init__()

        self.pri_conv_1 = ConvBN(st1_inc, decode_channels, kernel_size=1)
        self.pri_conv_2 = ConvBN(st2_inc, decode_channels, kernel_size=1)
        self.pri_conv_3 = ConvBN(st3_inc, decode_channels, kernel_size=1)
        self.pri_conv_4 = ConvBN(st4_inc, decode_channels, kernel_size=1)


        self.edge_res1 = ConvBNReLU(st1_inc, st1_inc, kernel_size=3)
        self.edge_res2 = ConvBNReLU(st2_inc, st2_inc, kernel_size=3)
        self.edge_res3 = ConvBNReLU(st3_inc, st3_inc, kernel_size=3)

        self.edge3 = Conv(st3_inc, 1, kernel_size=1)
        self.edge2 = Conv(st2_inc, 1, kernel_size=1)
        self.edge1 = Conv(st1_inc, 1, kernel_size=1)

        self.edge1_pri_1 = Conv(st2_inc * 2, st2_inc, kernel_size=3)
        self.edge3_up_pri = Conv(st3_inc, st2_inc, kernel_size=3)
        self.edge1_pri_2 = Conv(st2_inc, st1_inc, kernel_size=3)
        self.edge2_up_pri = Conv(st1_inc * 2, st1_inc, kernel_size=3)
        self.sigmoid = nn.Sigmoid()

        self.edge1_down = Conv(st1_inc, st2_inc, kernel_size=3,stride=2)
        self.edge1_down_short_cut = Conv(st1_inc, st2_inc, kernel_size=1, stride=2)
        self.edge1_down_edge_2 = Conv(st2_inc * 2, st2_inc, kernel_size=1, stride=1)
        self.edge2_down = Conv(st2_inc, st3_inc, kernel_size=3, stride=2)
        self.edge2_down_short_cut = Conv(st2_inc, st3_inc, kernel_size=1, stride=2)
        self.edge2_down_edge_3 = Conv(st3_inc * 2, st3_inc, kernel_size=1, stride=1)

        self.edge_2_fus = Conv(st2_inc * 3, st2_inc, kernel_size=3, stride=1)

    def forward(self, res1, res2, res3, res4):

        res4_pri = self.pri_conv_4(res4)
        res3_pri = self.pri_conv_3(res3)
        res2_pri = self.pri_conv_2(res2)
        res1_pri = self.pri_conv_1(res1)

        edge_res1 = self.edge_res1(res1)
        edge_res2 = self.edge_res2(res2)
        edge_res3 = self.edge_res3(res3)

        edge3_up = F.interpolate(edge_res3, scale_factor=2, mode='bilinear', align_corners=False)
        edge3_up_edge_2 = self.edge1_pri_1(torch.cat([self.edge3_up_pri(edge3_up),edge_res2],dim=1))
        edge3_up_edge_2_clone = edge3_up_edge_2.clone()
        edge3_up_edge_2_up = F.interpolate(edge3_up_edge_2, scale_factor=2, mode='bilinear', align_corners=False)
        edge3_up_edge_2_up_edge_1 = self.edge1_pri_2(torch.cat([self.edge1_pri_2 (edge3_up_edge_2_up), edge_res1],dim=1))

        edge1_down = (self.edge1_down(edge_res1) + self.edge1_down_short_cut(edge_res1))
        edge1_down_edge_2 = self.edge1_down_edge_2(torch.cat([edge1_down, edge_res2],dim=1))
        edge1_down_edge_2_clone = edge1_down_edge_2.clone()
        edge1_down_edge_2_down = (self.edge2_down(edge1_down_edge_2) + self.edge2_down_short_cut(edge1_down_edge_2))
        edge2_down_edge_3 = self.edge2_down_edge_3(torch.cat([edge1_down_edge_2_down, edge_res3],dim=1))

        edge1_down_edge_3_up_edge_2 = self.edge_2_fus(torch.cat([edge3_up_edge_2_clone, edge1_down_edge_2_clone, edge_res2],dim=1))


        edge3 = self.sigmoid(self.edge3(edge2_down_edge_3))
        edge2 = self.sigmoid(self.edge2(edge1_down_edge_3_up_edge_2))
        edge1 = self.sigmoid(self.edge1(edge3_up_edge_2_up_edge_1))

        return res1_pri, res2_pri, res3_pri, res4_pri, edge1, edge2, edge3



class AAM(nn.Module):
    def __init__(self, decode_channels=128, eps=1e-8):
        super(AAM, self).__init__()

        self.eps = eps
        self.weights = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)

        # self.align = AlignModule(decode_channels)
        # self.block_last = Block_4(output_dim=decode_channels)
        #self.softmax = nn.Softmax(dim=-1)
        self.softmax2 = nn.Softmax(dim=-1)
        self.softmax3 = nn.Softmax(dim=-1)

        self.q = nn.Conv2d(decode_channels, decode_channels, 1, 1, 0)
        self.q_change = pool_change(decode_channels)
        
        #self.kv = nn.Conv2d(decode_channels, 6, 1, 1, 0)
        self.kv = class_pool_change(decode_channels)
        
        self.refine = nn.Conv2d(decode_channels * 2, decode_channels, 1, 1, 0)
        self.last = ConvBNReLU(decode_channels, decode_channels, kernel_size=3)
        self.short_cut = nn.Conv2d(decode_channels*2, decode_channels, 1, 1, 0)
        self.key_channels = 6
        
        self.Muti_Large_filed = Muti_Large_filed(decode_channels)
    def forward(self, high_fea, x, edge):
        b, c, h, w = high_fea.shape

        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        weights = nn.ReLU()(self.weights)
        fuse_weights = weights / (torch.sum(weights, dim=0) + self.eps)
        x = fuse_weights[0] * high_fea + fuse_weights[1] * x
        x_clone = x.clone()
        #Edge
        edge_fea = x * edge
        
        #Category(SCAA)
        bound = x
        q_pri = self.q(bound)
        q = self.q_change(q_pri)
        #class_attn_bound = self.kv(bound)
        #class_attn_bound_clone = class_attn_bound.clone()
        #class_attn_softmax = self.softmax(class_attn_bound)
        #class_attn_softmax_clone = class_attn_softmax.clone()
        k, class_attn_bound = self.kv(bound)
        k = self.softmax3(k)
        attn = self.softmax2((q.permute(0,3,1,2).view(b, h+w, c, h) @ k.permute(0, 3, 2, 1).view(b, h+w, w, 6)).view(b, h+w, c, 6))
        out_class_attn = (attn @ k.permute(0,3,1,2).view(b, h+w, 6, w)).permute(0,2,1,3).view(b, c, h+w, w)
        class_info = torch.chunk(out_class_attn, 2, 2)
        out_class_attn = self.refine(torch.cat([class_info[0], class_info[1]],dim=1))
        
        
        #x = out_class_attn + x_clone
        x = out_class_attn + x_clone + edge_fea
        x = self.Muti_Large_filed(x)
        x = self.last(x)
        return x, class_attn_bound


# class FeatureRefinementHead(nn.Module):
#     def __init__(self, in_channels=64, decode_channels=64):
#         super().__init__()
#         self.pre_conv = Conv(in_channels, decode_channels, kernel_size=1)
#
#         self.weights = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
#         self.eps = 1e-8
#         self.post_conv = ConvBNReLU(decode_channels, decode_channels, kernel_size=3)
#
#         self.pa = nn.Sequential(nn.Conv2d(decode_channels, decode_channels, kernel_size=3, padding=1, groups=decode_channels),
#                                 nn.Sigmoid())
#         self.ca = nn.Sequential(nn.AdaptiveAvgPool2d(1),
#                                 Conv(decode_channels, decode_channels//16, kernel_size=1),
#                                 nn.ReLU6(),
#                                 Conv(decode_channels//16, decode_channels, kernel_size=1),
#                                 nn.Sigmoid())
#
#         self.shortcut = ConvBN(decode_channels, decode_channels, kernel_size=1)
#         self.proj = SeparableConvBN(decode_channels, decode_channels, kernel_size=3)
#         self.act = nn.ReLU6()
#
#     def forward(self, x, res):
#         x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
#         weights = nn.ReLU()(self.weights)
#         fuse_weights = weights / (torch.sum(weights, dim=0) + self.eps)
#         x = fuse_weights[0] * self.pre_conv(res) + fuse_weights[1] * x
#         x = self.post_conv(x)
#         shortcut = self.shortcut(x)
#         pa = self.pa(x) * x
#         ca = self.ca(x) * x
#         x = pa + ca
#         x = self.proj(x) + shortcut
#         x = self.act(x)
#
#         return x




class Decoder(nn.Module):
    def __init__(self,
                 encoder_channels=(64, 128, 256, 512),
                 decode_channels=64,
                 dropout=0.1,
                 num_classes=6):
        super(Decoder, self).__init__()

        self.edge_get = Prior_Edge(encoder_channels[-4], encoder_channels[-3], encoder_channels[-2],
                                  encoder_channels[-1], decode_channels)

        self.p3 = AAM(decode_channels)

        self.p2 = AAM(decode_channels)

        self.p1 = AAM(decode_channels)

        self.segmentation_head = nn.Sequential(ConvBNReLU(decode_channels, decode_channels),
                                               nn.Dropout2d(p=dropout, inplace=True),
                                               Conv(decode_channels, num_classes, kernel_size=1))
        self.init_weight()

        self.last = ConvBNReLU(decode_channels, decode_channels, kernel_size=3)
        
        self.Muti_Large_filed = Muti_Large_filed(decode_channels)

    def forward(self, res1, res2, res3, res4, h, w):

        res1, res2, res3, res4, edge1, edge2, edge3 = self.edge_get(res1, res2, res3, res4)

        # x = self.b4(res4)
        x = self.Muti_Large_filed(res4)
        x = self.last(x)
        x, rg3 = self.p3(res3, x, edge3)
        x, rg2 = self.p2(res2, x, edge2)
        x, rg1 = self.p1(res1, x, edge1)
        x = self.segmentation_head(x)
        x = F.interpolate(x, size=(h, w), mode='bilinear', align_corners=False)

        return x, edge1, edge2, edge3, rg1, rg2, rg3

    def init_weight(self):
        for m in self.children():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)



class ECGNet(nn.Module):

    def __init__(self,
                 decode_channels=64,
                 dropout=0.1,
                 embed_dim=128,
                 num_classes=6,
                 backbone_name='swsl_resnet18'
                 ):
        super().__init__()

        self.backbone = timm.create_model(backbone_name, features_only=True, output_stride=32,
                                          out_indices=(1, 2, 3, 4), pretrained=True)
        encoder_channels = self.backbone.feature_info.channels()

        self.decoder = Decoder(encoder_channels, decode_channels, dropout, num_classes)

    def forward(self, x):
        h, w = x.size()[-2:]

        res1, res2, res3, res4 = self.backbone(x)

        x, edge1, edge2, edge3, rg1, rg2, rg3 = self.decoder(res1, res2, res3, res4, h, w)

        #return x, edge1, edge2, edge3, rg1, rg2, rg3
        return x
