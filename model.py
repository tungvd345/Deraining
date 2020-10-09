import torch
import torch.nn as nn
from torch.nn import init
import torchvision.transforms as transforms
import torchvision.models as models
import functools
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
from math import log10

class Deraining(nn.Module):
    def __init__(self,args):
        super(Deraining, self).__init__()
        self.args = args
        self.upsample = F.interpolate
        self.upx2 = nn.Upsample(scale_factor=2, mode = 'bilinear', align_corners=True)
        # self.extractor = feature_extractor()

        self.up_feature = up_feature(in_channels=16*16*3)
        # self.conv1 = nn.Conv2d(in_channels=128*3, out_channels=128, kernel_size=1)
        # self.afim = AFIM(in_channels=128, out_channels=128)
        # self.ats_model = ATS_model(args, in_channels=3)
        self.ats_model = SCA_UNet(in_channel=3, out_channel=3)
        self.operation_layer = operation_layer(in_channels=3)

        self.relu = nn.LeakyReLU(0.2, True)

        # self.channel_att = channel_attention(in_channels=128, out_channels=15)
        self.channel_att = channel_attention(in_channels=9)
        self.rcan = RCAN(args)

    def forward(self, x, kpts):
        b, c, height, width = x.size()
        # x = self.upsample1(x)

        # features = self.extractor(x)
        # features_clean = self.afim(features)
        # features_add = self.afim(features)
        # features_mul = self.afim(features)

        upsample1 = nn.Upsample((height, width), mode='bilinear', align_corners=True)
        # up_feat_func = up_feature(in_channels=128*3, up_size=(height, width))
        # up_feat_func.cuda()
        # features_add = up_feat_func(kpts)
        features_add = self.up_feature(kpts)
        features_add = self.upsample(features_add, size=(height, width), mode='bilinear', align_corners=True)

        features_mul = self.up_feature(kpts)
        features_mul = self.upsample(features_mul, size=(height, width), mode='bilinear', align_corners=True)

        # atm, trans, streak = self.ats_model(x)
        # clean = (x - (1-trans) * atm) / (trans + 0.0001) - streak
        clean = self.ats_model(x)

        add_residual = self.operation_layer(features_add)
        add_layer = x + add_residual

        mul_residual = self.operation_layer(features_mul)
        mul_layer = x * (mul_residual+1e-8)

        concatenates = torch.cat((clean, add_layer, mul_layer), dim=1)
        # concatenates = torch.cat((clean, mul_layer), dim=1)

        # w0, w1, w2, w3, w4 = self.channel_att(concatenates)
        # out_comb = w0 * clean + w1 * add_layer + w2 * mul_layer + w3 * add_layer + w4 * mul_layer
        w0, w1, w2 = self.channel_att(concatenates)
        out_comb = w0 * clean + w1 * add_layer + w2 * mul_layer
        # w1, w2 = self.channel_att(concatenates)
        # out_comb = w1 * clean + w2 * mul_layer

        out_SR = self.rcan(out_comb)
        # out_combine = self.upx2(out_comb)
        out_combine = out_comb

        return out_SR, out_combine, clean, add_layer, mul_layer, add_residual, mul_residual
        # return out_SR, out_combine, clean, clean, clean

class ATS_model(nn.Module):
    def __init__(self, args, in_channels):
        super(ATS_model, self).__init__()
        self.conv1 = nn.Conv2d(in_channels = in_channels, out_channels = 64, kernel_size = 3, padding = 1)
        # self.batch_norm = nn.BatchNorm2d(64)
        self.relu1 = nn.LeakyReLU(0.2, True)
        self.conv2 = nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size = 3, padding = 1)
        # self.pooling = nn.AvgPool2d(kernel_size = (3,3))
        # self.fc = nn.Linear(in_features = in_channels * (args.patch_size//6) * (args.patch_size//4), out_features = 3) # (patch*3//2) //3 = patch // 2
        # self.sigmoid = nn.Sigmoid()
        self.predict_S = predict_S(in_channel=3, out_channel=3)
        self.predict_A = predict_A(128)
        self.predict_T = predict_T(in_channel=3, out_channel=3)
        # self.conv = nn.Conv2d(in_channels, out_channels=128, kernel_size=3, padding=1)

    def forward(self,x):
        # T = self.predict_T(x)
        S = self.predict_S(x)
        T = self.predict_T(x)

        x = self.relu1(self.conv1(x))
        x = self.relu1(self.conv2(x))
        # conv_T = self.conv2(self.relu1(self.batch_norm(self.conv1(x))))
        # T = self.sigmoid(conv_T)
        # T = self.predict_A(x)


        # pooling = self.pooling(x)
        # b, c, h, w = pooling.size()
        # pooling = pooling.view(b,-1)
        # A = self.sigmoid(self.fc(pooling))
        # A = A.view(b,3,1,1)
        A = self.predict_A(x)

        # conv_S = self.conv2(self.relu1(self.batch_norm(self.conv1(x))))
        # S = self.sigmoid(conv_S)


        #clean = (img_in - (1 - T) * A) / (T + 0.0001) - S
        return A, T, S

class predict_S(nn.Module):
    def __init__(self, in_channel, out_channel=3):
        super(predict_S, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, 32, kernel_size=3, padding=1)
        self.dense_block1 = dense_block(in_channel=32, up_channel=32)
        # self.dense_block = dense_block(in_channel=in_channel, out_channel=in_channel)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=1)
        self.dense_block2 = dense_block(in_channel=64, up_channel=64)
        self.relu = nn.ReLU()
        sequence = [nn.Conv2d(64, 64 // 2, kernel_size=1),
                      nn.ReLU(True),
                      nn.Conv2d(64 // 2, out_channel, kernel_size=1),
                      nn.Dropout2d()
                    ]
        self.down_conv = nn.Sequential(*sequence)
        self.reset_params()

    def forward(self, x):
        # dense_block = self.dense_block(x)
        x = self.relu(self.conv1(x))
        dense_block1 = self.dense_block1(x)
        dense_block2 = self.relu(self.conv2(dense_block1))
        dense_block2 = self.dense_block2(dense_block2)
        streak = self.down_conv(dense_block2)
        return streak

    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Conv2d):
            init.xavier_normal_(m.weight)
            # init.constant(m.bias, 0)

    def reset_params(self):
        for i, m in enumerate(self.modules()):
            self.weight_init(m)

class predict_T(nn.Module):
    def __init__(self, in_channel, out_channel=3):
        super(predict_T, self).__init__()
        self.trans_unet = TransUNet(in_channel, out_channel)
        self.reset_params()

    def forward(self, x):
        trans = self.trans_unet(x)
        return trans

    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Conv2d):
            init.xavier_normal_(m.weight)
            # init.constant(m.bias, 0)

    def reset_params(self):
        for i, m in enumerate(self.modules()):
            self.weight_init(m)

class predict_A(nn.Module):
    def __init__(self, in_channel):
        super(predict_A, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, in_channel//4, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channel//4, in_channel//4, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channel//4, in_channel//16, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(in_channel//16, in_channel//16, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(in_channel//16, in_channel//64, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pooling1 = nn.AdaptiveAvgPool2d((128, 128))
        self.pooling2 = nn.AdaptiveAvgPool2d((64, 64))
        self.pooling3 = nn.AdaptiveAvgPool2d((32, 32))
        self.pooling4 = nn.AdaptiveAvgPool2d((16,16))
        self.pooling5 = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(in_channel//64, 1)
        self.reset_params()

    def forward(self, x):
        b, c, h, w = x.size()
        atm1 = self.pooling1(self.relu(self.conv1(x)))
        atm2 = self.pooling2(self.relu(self.conv2(atm1)))
        atm3 = self.pooling3(self.relu(self.conv3(atm2)))
        atm4 = self.pooling4(self.relu(self.conv4(atm3)))
        atm5 = self.pooling5(self.relu(self.conv5(atm4)))
        atm5 = atm5.view(b, -1)
        atm = self.fc(atm5)
        atm = atm.view(b, 1, 1, 1)
        return atm

    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Conv2d):
            init.xavier_normal_(m.weight)
            # init.constant(m.bias, 0)

    def reset_params(self):
        for i, m in enumerate(self.modules()):
            self.weight_init(m)

##################################################################################
# dense_block use pretrained dense-net
##################################################################################
# class dense_block(nn.Module):
#     def __init__(self, in_channel, out_channel):
#         super(dense_block, self).__init__()
#         model_dense_net = models.densenet121(pretrained=True)
#         model_dense_net = list(model_dense_net.children())[:]
#         self.dense_block = model_dense_net[0].denseblock1
#         self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=64, kernel_size=7, padding=3)
#         self.relu = nn.ReLU(True)
#         # sequence = []
#         sequence = [nn.Conv2d(256, 224, kernel_size = 1),
#                         nn.ReLU(True),
#                         nn.Conv2d(224, 192, kernel_size = 1),
#                         nn.ReLU(True),
#                         nn.Conv2d(192, 160, kernel_size = 1),
#                         nn.ReLU(True),
#                         nn.Conv2d(160, 128, kernel_size = 1),
#                         nn.ReLU(True),
#                         nn.Conv2d(128, 96, kernel_size = 1),
#                         nn.ReLU(True),
#                         nn.Conv2d(96, 64, kernel_size = 1),
#                         nn.ReLU(True),
#                         nn.Conv2d(64, 3, kernel_size = 1),
#                         nn.Dropout2d()]
#         self.down_conv = nn.Sequential(*sequence)
#
#     def forward(self, x):
#         x = self.relu(self.conv1(x))
#         dense_block = self.relu(self.dense_block(x))
#         out = self.down_conv(dense_block)
#
#         return out
##################################################################################

##################################################################################
# dense_block don't use pretrained
##################################################################################
class dense_block(nn.Module):
    def __init__(self, in_channel, up_channel=32, num_dense_layer=4):
        super(dense_block, self).__init__()
        in_chan = in_channel
        sequence_1 = []
        for i in range(num_dense_layer):
            sequence_1.append(dense_layer(in_chan, up_channel))
            in_chan += up_channel

        self.dense_block = nn.Sequential(*sequence_1)
        sequence_2 = [nn.Conv2d(in_chan, in_chan//2, kernel_size=1),
                    nn.ReLU(True),
                    nn.Conv2d(in_chan//2, in_channel, kernel_size = 1),
                    nn.Dropout2d()
                    ]
        self.down_conv = nn.Sequential(*sequence_2)

    def forward(self, x):
        dense_block = self.dense_block(x)
        out = self.down_conv(dense_block)
        out = out + x
        return out

class dense_layer(nn.Module):
    def __init__(self, in_channel, up_channel):
        super(dense_layer, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channel, out_channels=up_channel, kernel_size=3, padding=1)
        self.relu = nn.ReLU(True)

    def forward(self, x):
        out =self.relu(self.conv(x))
        out = torch.cat((x, out), 1)
        return out
##################################################################################

##################################################################################
# Defines the Unet-transmission
##################################################################################
class TransUNet(nn.Module):
    def __init__(self, in_channel, n_classes):
        super(TransUNet, self).__init__()
        self.conv1x1 = nn.Conv2d(in_channel, in_channel, kernel_size=1, stride=1, padding=0)
        # self.inc = inconv(in_channel, 64)
        self.inconv = nn.Sequential(
            nn.Conv2d(in_channel, 32, 3, padding=1),
            # nn.InstanceNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, padding=1),
            # nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.down1 = down(64, 128)
        self.down2 = down(128, 256)
        self.down3 = down(256, 512)
        self.down4 = down(512, 512)

        self.up1 = up(1024, 256)
        self.up2 = up(512, 128)
        self.up3 = up(256, 64)
        self.up4 = up(128, 32)
        self.outconv = nn.Conv2d(32, n_classes, kernel_size=1)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.conv1x1(x)
        x1 = self.inconv(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        # decoder
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.tanh(self.outconv(x))
        return x

class double_conv(nn.Module):
    '''(conv => BN => ReLU) * 2'''
    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            # nn.InstanceNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            # nn.InstanceNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(down, self).__init__()
        self.mpconv = nn.Sequential(
            # nn.MaxPool2d(2),
            # double_conv(in_ch, out_ch)
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(True),
            nn.Conv2d(out_ch, out_ch, kernel_size=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(True)
        )

    def forward(self, x):
        x = self.mpconv(x)
        return x

class up(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=False):
        super(up, self).__init__()

        #  would be a nice idea if the upsampling could be learned too,
        #  but my machine do not have enough memory to handle all those weights
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch//2, in_ch//2, 2, stride=2)

        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffX = x2.size()[2] - x1.size()[2]
        diffY = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, (0, diffY, 0, diffX))
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x

##################################################################################
# Defines the SCA-clean - base on UNet
##################################################################################
class SCA_UNet(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(SCA_UNet, self).__init__()
        self.conv1x1 = nn.Conv2d(in_channel, in_channel, kernel_size=1, stride=1, padding=0)
        # self.inc = inconv(in_channel, 64)
        self.inconv = nn.Sequential(
            nn.Conv2d(in_channel, 32, 3, padding=1),
            # nn.InstanceNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, padding=1),
            # nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.down1 = down_SCA(64, 128)
        self.down2 = down_SCA(128, 256)
        self.down3 = down_SCA(256, 512)
        self.down4 = down_SCA(512, 512)

        self.up1 = up_SCA(1024, 256)
        self.up2 = up_SCA(512, 128)
        self.up3 = up_SCA(256, 64)
        self.up4 = up_SCA(128, 32)
        self.outconv = nn.Conv2d(32, out_channel, kernel_size=1)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.conv1x1(x)
        x1 = self.inconv(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        # decoder
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = (self.outconv(x))
        return x

class down_SCA(nn.Module):
    def __init__(self, in_chan, out_chan, reduce=16):
        super(down_SCA, self).__init__()
        self.conv1 = nn.Conv2d(in_chan, out_chan, kernel_size=3, stride=2, padding=1)
        self.relu = nn.ReLU(True)
        self.conv2 = nn.Conv2d(out_chan, out_chan, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(out_chan, out_chan, kernel_size=3, padding=1)
        self.sigmoid = nn.Sigmoid()
        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_chan, out_chan//reduce, kernel_size=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(out_chan//reduce, out_chan, kernel_size=1, padding=0),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.relu(self.conv1(x))
        conv2 = self.relu(self.conv2(x))
        conv3_1 = self.conv3(conv2)
        conv3_2 = self.sigmoid(self.conv3(conv2))
        spatial = conv3_1 * conv3_2
        channel = self.ca(spatial)
        sca = channel * conv2
        out_layer = x + sca
        return out_layer



class up_SCA(nn.Module):
    def __init__(self, in_chan, out_chan, reduce=16, bilinear=True):
        super(up_SCA, self).__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_chan//2, in_chan//2, 2, stride=2)
        # self.conv = double_conv(in_ch, out_ch)

        self.conv1 = nn.Conv2d(in_chan, out_chan, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU(True)
        self.conv2 = nn.Conv2d(out_chan, out_chan, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(out_chan, out_chan, kernel_size=3, padding=1)
        self.sigmoid = nn.Sigmoid()
        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_chan, out_chan // reduce, kernel_size=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(out_chan // reduce, out_chan, kernel_size=1, padding=0),
            nn.Sigmoid()
        )

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffX = x2.size()[2] - x1.size()[2]
        diffY = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, (0, diffY, 0, diffX))
        x = torch.cat([x2, x1], dim=1)

        conv1 = self.relu(self.conv1(x))
        conv2 = self.relu(self.conv2(conv1))
        conv3_1 = self.conv3(conv2)
        conv3_2 = self.sigmoid(self.conv3(conv2))
        spatial = conv3_1 * conv3_2
        channel = self.ca(spatial)
        sca = channel * conv2
        out_layer = conv1 + sca

        # x = self.conv(x)
        return out_layer

# class outconv(nn.Module):
#     def __init__(self, in_ch, out_ch):
#         super(outconv, self).__init__()
#         self.conv = nn.Conv2d(in_ch, out_ch, 1)
#
#     def forward(self, x):
#         x = self.conv(x)
#         return x
##################################################################################

# class feature_extractor(nn.Module):
#     def __init__(self, out_channels = 128):
#         super(feature_extractor, self).__init__()
#         resnet18 = models.resnet18(pretrained = True)
#         num_ftrs = resnet18.fc.in_features
#         layer = list(resnet18.children())[:-2]
#         layer.append(nn.Conv2d(num_ftrs, out_channels, 1))
#         self.feature_extractor = nn.Sequential(*layer)
#         #print('feature extraction: \n',self.feature_extractor)
#
#     def forward(self,x):
#         feature = self.feature_extractor(x)
#         return feature

class feature_extractor(nn.Module):
    def __init__(self, out_channels = 128, n_block = 5):
        super(feature_extractor, self).__init__()
        # layer = [nn.Conv2d(3, 128, kernel_size = 1),
        #          nn.LeakyReLU(0.2, False)]
        layer = []
        self.conv = nn.Conv2d(3, 128, kernel_size=3, padding=1)
        for i in range(n_block - 1):
            layer += [ResidualBlockStraight(in_channels=128, out_channels=128, last=False)]
        layer += [ResidualBlockStraight(in_channels=128, out_channels=out_channels, last=True)]
        self.feature_extractor = nn.Sequential(*layer)
        # self.feature_extractor = nn.Conv2d(3, 128, kernel_size = 1)
        self.relu = nn.ReLU(False)
        self.reset_params()

    def forward(self,x):
        feature = self.conv(x)
        feature = self.relu(feature)
        feature = self.feature_extractor(feature)
        feature = self.relu(feature)
        # print('layer weigth grad', self.feature_extractor.weight.grad)
        return feature

    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Conv2d):
            init.xavier_normal_(m.weight)
            init.constant(m.bias, 0)

    def reset_params(self):
        for i, m in enumerate(self.modules()):
            self.weight_init(m)

class ResidualBlockStraight(nn.Module):
    def __init__(self, in_channels = 128, out_channels = 128, last=False):
        super(ResidualBlockStraight, self).__init__()
        assert (in_channels == out_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels // 4, kernel_size = 3, padding = 1)
        self.conv2 = nn.Conv2d(out_channels // 4, out_channels // 4, kernel_size = 3, padding = 1)
        self.batch_norm12 = nn.BatchNorm2d(out_channels // 4)
        self.conv3 = nn.Conv2d(out_channels // 4, out_channels, kernel_size = 3, padding = 1)
        # self.batch_norm3 = nn.BatchNorm2d(out_channels)
        self.relu = nn.LeakyReLU(0.2, inplace=False)
        self.tanh = nn.Tanh()
        self.last = last

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        # out = self.relu(self.batch_norm12(out))
        out = self.relu(out)
        out = self.conv2(out)
        # out = self.relu(self.batch_norm12(out))
        out = self.relu(out)
        out = self.conv3(out)
        out += residual
        if self.last:
            # out = self.tanh(out)
            out = self.relu(out)
        else:
            out = self.relu(out)
        return out

    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Conv2d):
            init.xavier_normal_(m.weight)
            init.constant(m.bias, 0.01)

    def reset_params(self):
        for i, m in enumerate(self.modules()):
            self.weight_init(m)

class operation_layer(nn.Module):
    def __init__(self, in_channels):
        super(operation_layer, self).__init__()
        self.conv1 = nn.Conv2d(in_channels = in_channels, out_channels = 64, kernel_size = 3, padding = 1)
        # self.batch_norm1 = nn.BatchNorm2d(64)
        self.batch_norm1 = nn.InstanceNorm2d(64)
        self.relu1 = nn.LeakyReLU(0.2, True)
        self.conv2 = nn.Conv2d(in_channels = 64, out_channels = 3, kernel_size = 3, padding = 1)
        # self.batch_norm2 = nn.BatchNorm2d(3)
        self.batch_norm2 = nn.InstanceNorm2d(3)
        self.relu2 = nn.ReLU(True)

    def forward(self, x):
        conv1 = self.relu1(self.batch_norm1(self.conv1(x)))
        R_layer = (self.batch_norm2(self.conv2(conv1)))
        return R_layer

# class AFIM is same in 'deep multi model fusion' paper
class AFIM(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(AFIM, self).__init__()
        # self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1)
        sequence1 = [
            nn.Conv2d(in_channels=out_channels, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64), nn.LeakyReLU(0.2, True),
            nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = 3, padding = 1),
            nn.BatchNorm2d(64), nn.LeakyReLU(0.2, True),
            nn.Conv2d(in_channels = 64, out_channels = out_channels, kernel_size = 3, padding=1),
            nn.Softmax2d()
        ]
        self.model1 = nn.Sequential(*sequence1)
        sequence2 = [
            nn.Conv2d(in_channels=out_channels, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64), nn.LeakyReLU(0.2, True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64), nn.LeakyReLU(0.2, True),
            nn.Conv2d(in_channels=64, out_channels=out_channels, kernel_size=1)
        ]
        self.model2 = nn.Sequential(*sequence2)

    def forward(self, x):
        # x = self.conv1(x)
        x  = x * self.model1(x)
        x  = x + self.model2(x)
        return x

class up_feature(nn.Module):
    def __init__(self, in_channels, out_channels=3):#, up_size = (200,300)):
        super(up_feature, self).__init__()
        sequence = [
            nn.Conv2d(in_channels=in_channels, out_channels=512, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, True),
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),   # 32x48
            nn.LeakyReLU(0.2, True),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),    # 64x96
            nn.LeakyReLU(0.2, True),
            nn.ConvTranspose2d(128, 32, kernel_size=4, stride=2, padding=1),    # 128x192
            nn.LeakyReLU(0.2, True),
            nn.ConvTranspose2d(32, 8, kernel_size=4, stride=2, padding=1),     # 256x384
            nn.LeakyReLU(0.2, True),
            # nn.Upsample(up_size, mode = 'bilinear', align_corners=True),
            nn.Conv2d(8, out_channels, kernel_size=1),
            nn.Dropout2d(0.5)
        ]
        self.sequence = nn.Sequential(*sequence)

    def forward(self, x):
        x = self.sequence(x)
        return x

class channel_attention(nn.Module):
    def __init__(self, in_channels = 15):
        super(channel_attention, self).__init__()
        sequence1 = [
            nn.Conv2d(in_channels=in_channels, out_channels=128, kernel_size=1),
            nn.BatchNorm2d(128), nn.LeakyReLU(0.2, True),
            nn.Conv2d(in_channels = 128, out_channels = 64, kernel_size =1),
            nn.BatchNorm2d(64), nn.LeakyReLU(0.2, True),
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=1),
            nn.BatchNorm2d(32), nn.LeakyReLU(0.2, True),
            nn.Conv2d(in_channels = 32, out_channels = in_channels, kernel_size = 1),
            # nn.Softmax2d()
        ]
        self.model1 = nn.Sequential(*sequence1)
        sequence2 = [
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels//4, kernel_size=1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(in_channels=in_channels//4, out_channels=in_channels, kernel_size=1),  # padding = 1),
            nn.Sigmoid()
        ]
        self.model2 = nn.Sequential(*sequence2)
    def forward(self, x):
        x  = self.model1(x)
        y = self.model2(x)
        out = x * y
        out0 = out[:,0:3,:,:]
        out1 = out[:,3:6,:,:]
        out2 = out[:,6:9,:,:]

        return out0, out1, out2

class SCA_block(nn.Module):
    def __init__(self, in_chan, out_chan, reduce=16):
        super(SCA_block, self).__init__()
        self.conv1 = nn.Conv2d(in_chan, out_chan, kernel_size=3, padding=1)
        self.relu = nn.ReLU(True)
        self.conv2 = nn.Conv2d(out_chan, out_chan, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(out_chan, out_chan, kernel_size=3, padding=1)
        self.sigmoid = nn.Sigmoid()
        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_chan, out_chan//reduce, kernel_size=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(out_chan//reduce, out_chan, kernel_size=1, padding=0),
            nn.Sigmoid()
        )

    def forward(self, x):
        conv1 = self.relu(self.conv1(x))
        conv2 = self.relu(self.conv2(conv1))
        conv3_1 = self.conv3(conv2)
        conv3_2 = self.sigmoid(self.conv3(conv2))
        spatial = conv3_1 * conv3_2
        channel = self.ca(spatial)
        sca = channel * conv2
        out_layer = x + sca
        return out_layer

class RCAN(nn.Module):
    def __init__(self, args):
        super(RCAN, self).__init__()
        nChannel = args.nchannel
        scale = args.scale
        self.args = args

        # Define Network
        # ===========================================
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(nChannel, 64, kernel_size=7, padding=3)
        # self.RG1 = residual_group(64, 64)
        # self.RG2 = residual_group(64, 64)
        # # self.RG3 = residual_group(64, 64)
        self.SCAB1 = SCA_block(64, 64)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 32, kernel_size=5, padding=2)
        self.conv4 = nn.Conv2d(32, 3, kernel_size=3, padding=1)
        # self.reset_params()
        # ===========================================

    def forward(self, x):
        # Make a Network path
        # ===========================================
        x = self.relu(self.conv1(x))

        sca1 = self.SCAB1(x)
        sca2 = self.SCAB1(sca1)
        sca3 = self.SCAB1(sca2)
        sca3 = sca3 + sca2
        sca4 = self.SCAB1(sca3)
        sca4 = sca4 + sca1
        sca5 = self.SCAB1(sca4)
        sca5 = sca5 + x

        x = self.relu(self.conv3(sca5))
        # x = self.pixel_shuffle(x)

        x = self.conv4(x)
        # ===========================================
        return x

    # @staticmethod
    # def weight_init(m):
    #     if isinstance(m, nn.Conv2d):
    #         init.xavier_normal_(m.weight)
    #         # init.constant(m.bias, 0)
    #
    # def reset_params(self):
    #     for i, m in enumerate(self.modules()):
    #         self.weight_init(m)


class residual_group(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(residual_group, self).__init__()
        self.rca_block1 = RCAB(in_channels, 64)
        self.rca_block2 = RCAB(64, out_channels)

    def forward(self, x):
        rcab1 = self.rca_block1(x)
        rcab2 = self.rca_block2(rcab1)
        return x + rcab2


class RCAB(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(RCAB, self).__init__()
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.ca_block = CA_block(64, out_channels)
        # self.reset_params()

    def forward(self, x):
        conv1 = self.conv1(x)
        conv1 = self.relu(conv1)
        conv2 = self.conv2(conv1)
        ca = self.ca_block(conv2)
        return x + ca

    # @staticmethod
    # def weight_init(m):
    #     if isinstance(m, nn.Conv2d):
    #         init.xavier_normal_(m.weight)
    #         # init.constant(m.bias, 0)
    #
    # def reset_params(self):
    #     for i, m in enumerate(self.modules()):
    #         self.weight_init(m)

class CA_block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(CA_block, self).__init__()
        # global average pooling
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_down_up = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, out_channels, kernel_size=1, padding=0),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_down_up(y)
        return x * y
