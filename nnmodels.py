# -*- coding: utf-8 -*-
"""
Created on Sat Apr 24 11:31:13 2021

@author: Yingying Wang

modified from the work of "Alfarraj M, AlRegib G. Semisupervised sequence modeling for elastic impedance inversion[J]. Interpretation, 2019, 7(3): SE237-SE249."
"""

import torch
from torch.nn.functional import conv1d
from torch import nn, optim
# from bruges.reflection import reflection
import numpy as np

class inverse_model(nn.Module):
    def __init__(self, in_channels,out_channels,nonlinearity,lr,lr_decray):
        super(inverse_model, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.activation =  nn.ReLU() if nonlinearity=="relu" else nn.Tanh()
        self.lr = lr
        self.lr_decray = lr_decray
        self.cnn1 = nn.Sequential(nn.Conv2d(in_channels=self.in_channels,
                                           out_channels=16,
                                           kernel_size=[5,5],
                                           padding=(2,0),
                                           dilation=(1,1)),
                                  # 此处的num_groups为批归一化处理时，每组的通道数
                                  nn.GroupNorm(num_groups=8,
                                               num_channels=16))

        self.cnn2 = nn.Sequential(nn.Conv2d(in_channels=self.in_channels,
                                           out_channels=16,
                                           kernel_size=[5,5],
                                           padding=(6,0),
                                           dilation=(3,1)),
                                  nn.GroupNorm(num_groups=8,
                                               num_channels=16))

        self.cnn3 = nn.Sequential(nn.Conv2d(in_channels=self.in_channels,
                                           out_channels=16,
                                           kernel_size=[5,5],
                                           padding=(12,0),
                                           dilation=(6,1)),
                                  nn.GroupNorm(num_groups=8,
                                               num_channels=16))

        self.cnn = nn.Sequential(self.activation,
                                 nn.Conv1d(in_channels=48,
                                           out_channels=32,
                                           kernel_size=3,
                                           padding=1),
                                 nn.GroupNorm(num_groups=8,
                                              num_channels=32),
                                 self.activation,

                                 nn.Conv1d(in_channels=32,
                                           out_channels=32,
                                           kernel_size=3,
                                           padding=1),
                                 nn.GroupNorm(num_groups=8,
                                              num_channels=32),
                                 self.activation,
                                 
                                 # 此处的卷积核大小为1，起到的作用是只整合channel方向上的信息
                                 nn.Conv1d(in_channels=32,
                                           out_channels=32,
                                           kernel_size=1),
                                 nn.GroupNorm(num_groups=8,
                                              num_channels=32),
                                 self.activation)

        self.gru = nn.GRU(input_size=self.in_channels,
                          hidden_size=16,
                          num_layers=3,
                          batch_first=True,
                          bidirectional=True)

        self.up = nn.Sequential(nn.ConvTranspose1d(in_channels=32,
                                                   out_channels=16,
                                                   stride=3,
                                                   kernel_size=5,
                                                   padding=1),
                                nn.GroupNorm(num_groups=8,
                                             num_channels=16),
                                self.activation,

                                nn.ConvTranspose1d(in_channels=16,
                                                   out_channels=16,
                                                   stride=2,
                                                   kernel_size=4,
                                                   padding=1),
                                nn.GroupNorm(num_groups=8,
                                             num_channels=16),
                                self.activation)

        self.up_realdata_ratio10 = nn.Sequential(nn.ConvTranspose1d(in_channels=32,
                                                   out_channels=16,
                                                   stride=5,
                                                   kernel_size=5,
                                                   padding=0),
                                nn.GroupNorm(num_groups=8,
                                             num_channels=16),
                                self.activation,

                                nn.ConvTranspose1d(in_channels=16,
                                                   out_channels=16,
                                                   stride=2,
                                                   kernel_size=4,
                                                   padding=1),
                                nn.GroupNorm(num_groups=8,
                                             num_channels=16),
                                self.activation)

        self.up_realdata_ratio4 = nn.Sequential(nn.ConvTranspose1d(in_channels=32,
                                                   out_channels=16,
                                                   stride=2,
                                                   kernel_size=4,
                                                   padding=1),
                                nn.GroupNorm(num_groups=8,
                                             num_channels=16),
                                self.activation,

                                nn.ConvTranspose1d(in_channels=16,
                                                   out_channels=16,
                                                   stride=2,
                                                   kernel_size=4,
                                                   padding=1),
                                nn.GroupNorm(num_groups=8,
                                             num_channels=16),
                                self.activation)

        self.up_realdata_ratio1 = nn.Sequential(nn.ConvTranspose1d(in_channels=32,
                                                   out_channels=16,
                                                   stride=1,
                                                   kernel_size=3,
                                                   padding=1),
                                nn.GroupNorm(num_groups=8,
                                             num_channels=16),
                                self.activation,

                                nn.ConvTranspose1d(in_channels=16,
                                                   out_channels=16,
                                                   stride=1,
                                                   kernel_size=3,
                                                   padding=1),
                                nn.GroupNorm(num_groups=8,
                                             num_channels=16),
                                self.activation)

        self.gru_out = nn.GRU(input_size=16,
                              hidden_size=16,
                              num_layers=1,
                              batch_first=True,
                              bidirectional=True)
        self.out = nn.Linear(in_features=32, out_features=self.out_channels)

        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.ConvTranspose1d):
                nn.init.xavier_uniform_(m.weight.data)
                m.bias.data.zero_()
            elif isinstance(m, nn.GroupNorm):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

        self.optimizer = optim.Adam(self.parameters(), self.lr, weight_decay=self.lr_decray)
        print("lr = {}".format(self.lr))
        
    def forward(self, x, x_2D):
        # print("x_2D.shape = {}".format(x_2D.size()))
        # print("x.shape = {}".format(x.size()))
        cnn_out1 = self.cnn1(x_2D)
        cnn_out2 = self.cnn2(x_2D)
        cnn_out3 = self.cnn3(x_2D)
        # print("cnn_out1.shape = {}".format(cnn_out1.size()))
        # print("cnn_out2.shape = {}".format(cnn_out2.size()))
        # print("cnn_out3.shape = {}".format(cnn_out3.size()))
        cnn_out = self.cnn(torch.squeeze(torch.cat((cnn_out1,cnn_out2,cnn_out3),dim=1),3))
        # print("cnn_out.shape = {}".format(cnn_out.size()))
        
        tmp_x = x.transpose(-1, -2)
        # print("tmp_x.shape = {}".format(tmp_x.size()))        
        rnn_out, _ = self.gru(tmp_x)
        rnn_out = rnn_out.transpose(-1, -2)
        # print("rnn_out.shape = {}".format(rnn_out.size()))
        
        x = rnn_out + cnn_out
        # print("x.shape = {}".format(x.size())) 
        # NOTE: It should be changed for different dataset
        # x = self.up(x) # for Marmousci data
        # x = self.up_realdata_ratio10(x) # for field data
        # x = self.up_realdata_ratio4(x) # for field data
        x = self.up_realdata_ratio1(x) # for field data
        # print("x_up.shape = {}".format(x.size()))        

        tmp_x = x.transpose(-1, -2)
        x, _ = self.gru_out(tmp_x)
        # print("x.shape = {}".format(x.size()))        

        x = self.out(x)
        x = x.transpose(-1,-2)
        # print("x.shape = {}".format(x.size()))        
        
        return x


import torch.nn as nn
import torch
import torch.nn as nn
import torch.nn.functional as F


class unetConv2(nn.Module):
    def __init__(self, in_size, out_size, is_batchnorm):
        super(unetConv2, self).__init__()
        # Kernel size: 3*3, Stride: 1, Padding: 1
        if is_batchnorm:
            self.conv1 = nn.Sequential(nn.Conv2d(in_size, out_size, 3, 1, 1),
                                       nn.BatchNorm2d(out_size),
                                       nn.ReLU(inplace=True), )
            self.conv2 = nn.Sequential(nn.Conv2d(out_size, out_size, 3, 1, 1),
                                       nn.BatchNorm2d(out_size),
                                       nn.ReLU(inplace=True), )
        else:
            self.conv1 = nn.Sequential(nn.Conv2d(in_size, out_size, 3, 1, 1),
                                       nn.ReLU(inplace=True), )
            self.conv2 = nn.Sequential(nn.Conv2d(out_size, out_size, 3, 1, 1),
                                       nn.ReLU(inplace=True), )

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        outputs = self.conv2(outputs)
        return outputs


class unetDown(nn.Module):
    def __init__(self, in_size, out_size, is_batchnorm):
        super(unetDown, self).__init__()
        self.conv = unetConv2(in_size, out_size, is_batchnorm)
        self.down = nn.MaxPool2d(2, 2, ceil_mode=True)

    def forward(self, inputs):
        outputs = self.conv(inputs)
        outputs = self.down(outputs)
        return outputs


class unetUp(nn.Module):
    def __init__(self, in_size, out_size, is_deconv):
        super(unetUp, self).__init__()
        self.conv = unetConv2(in_size, out_size, True)
        # Transposed convolution
        if is_deconv:
            self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=2, stride=2)
        else:
            self.up = nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, inputs1, inputs2):
        outputs2 = self.up(inputs2)
        offset1 = (outputs2.size()[2] - inputs1.size()[2])
        offset2 = (outputs2.size()[3] - inputs1.size()[3])
        padding = [offset2 // 2, (offset2 + 1) // 2, offset1 // 2, (offset1 + 1) // 2]
        # Skip and concatenate
        outputs1 = F.pad(inputs1, padding)
        return self.conv(torch.cat([outputs1, outputs2], 1))


class UnetModel(nn.Module):
    def __init__(self, n_classes, in_channels, is_deconv, is_batchnorm):
        super(UnetModel, self).__init__()
        self.is_deconv = is_deconv
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm
        self.n_classes = n_classes

        # filters = [64, 128, 256, 512, 1024]
        filters = [  16, 32,64,128,256]
        self.down1 = unetDown(self.in_channels, filters[0], self.is_batchnorm)
        self.down2 = unetDown(filters[0], filters[1], self.is_batchnorm)
        self.dropout1 = nn.Dropout(p=0.1)
        self.down3 = unetDown(filters[1], filters[2], self.is_batchnorm)
        self.dropout2 = nn.Dropout(p=0.1)
        self.down4 = unetDown(filters[2], filters[3], self.is_batchnorm)
        self.center = unetConv2(filters[3], filters[4], self.is_batchnorm)
        self.up4 = unetUp(filters[4], filters[3], self.is_deconv)
        self.up3 = unetUp(filters[3], filters[2], self.is_deconv)
        self.up2 = unetUp(filters[2], filters[1], self.is_deconv)
        self.up1 = unetUp(filters[1], filters[0], self.is_deconv)
        self.final = nn.Conv2d(filters[0], self.n_classes, 1)

    def forward(self, inputs, label_dsp_dim):
        down1 = self.down1(inputs)

        down2 = self.down2(down1)
        # down3  = self.down3(down2)
        # down4  = self.down4(down3)
        dropout1 = self.dropout1(down2)
        down3 = self.down3(dropout1)

        dropout2 = self.dropout1(down3)
        down4 = self.down4(dropout2)

        center = self.center(down4)
        up4 = self.up4(down4, center)
        up3 = self.up3(down3, up4)
        up2 = self.up2(down2, up3)
        up1 = self.up1(down1, up2)
        # print(up1.size(),'up1111')
        up1 = up1[:, :, 1:1 + label_dsp_dim[0], 1:1 + label_dsp_dim[1]].contiguous()
        # up1 = up1[:, :, 1:1 + label_dsp_dim[0], 0:1 + label_dsp_dim[1]].contiguous()     ##if applied in overthrust model, use it; else not.
        return self.final(up1)

    # Initialization of Parameters
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.ConvTranspose2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()


###########################



class cnnConv2(nn.Module):
    def __init__(self, in_size, out_size, is_batchnorm):
        super(cnnConv2, self).__init__()
        # Kernel size: 3*3, Stride: 1, Padding: 1
        if is_batchnorm:
            self.conv1 = nn.Sequential(nn.Conv2d(in_size, out_size, 3, 1, 1),
                                       nn.BatchNorm2d(out_size),
                                       nn.ReLU(inplace=True), )
            self.conv2 = nn.Sequential(nn.Conv2d(out_size, out_size, 3, 1, 1),
                                       nn.BatchNorm2d(out_size),
                                       nn.ReLU(inplace=True), )
        else:
            self.conv1 = nn.Sequential(nn.Conv2d(in_size, out_size, 3, 1, 1),
                                       nn.ReLU(inplace=True), )
            self.conv2 = nn.Sequential(nn.Conv2d(out_size, out_size, 3, 1, 1),
                                       nn.ReLU(inplace=True), )

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        outputs = self.conv2(outputs)
        return outputs


class cnnDown(nn.Module):
    def __init__(self, in_size, out_size, is_batchnorm):
        super(cnnDown, self).__init__()
        self.conv = cnnConv2(in_size, out_size, is_batchnorm)
        self.down = nn.MaxPool2d(2, 2, ceil_mode=True)

    def forward(self, inputs):
        outputs = self.conv(inputs)
        outputs = self.down(outputs)
        return outputs


class cnnUp(nn.Module):
    def __init__(self, in_size, out_size, is_deconv):
        super(cnnUp, self).__init__()
        # self.conv = unetConv2(out_size, out_size, True)
        # Transposed convolution
        # if is_deconv:
        #     self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=2, stride=2)
        # else:
        #     self.up = nn.UpsamplingBilinear2d(scale_factor=2)

        self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=2, stride=2)
        self.conv = cnnConv2(out_size, out_size, True)
    def forward(self, inputs):
        # outputs2 = self.up(inputs2)
        # offset1 = (outputs2.size()[2] - inputs1.size()[2])
        # offset2 = (outputs2.size()[3] - inputs1.size()[3])
        # padding = [offset2 // 2, (offset2 + 1) // 2, offset1 // 2, (offset1 + 1) // 2]
        # # Skip and concatenate
        # outputs1 = F.pad(inputs1, padding)
        # return self.conv(torch.cat([outputs1, outputs2], 1))
        outputs=self.up(inputs)
        outputs = self.conv(outputs)
        return outputs


class cnnModel(nn.Module):
    def __init__(self, n_classes, in_channels, is_deconv, is_batchnorm):
        super(cnnModel, self).__init__()
        self.is_deconv = is_deconv
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm
        self.n_classes = n_classes

        filters = [64, 128, 256, 512, 1024]
        # filters = [8, 16, 32, 64, 128]
        self.down1 = cnnDown(self.in_channels, filters[0], self.is_batchnorm)
        self.down2 = cnnDown(filters[0], filters[1], self.is_batchnorm)
        self.dropout1 = nn.Dropout(p=0.1)
        self.down3 = cnnDown(filters[1], filters[2], self.is_batchnorm)
        self.dropout2 = nn.Dropout(p=0.1)
        self.down4 = cnnDown(filters[2], filters[3], self.is_batchnorm)
        self.center = unetConv2(filters[3], filters[4], self.is_batchnorm)
        self.up4 = cnnUp(filters[4], filters[3], self.is_deconv)
        self.up3 = cnnUp(filters[3], filters[2], self.is_deconv)
        self.up2 = cnnUp(filters[2], filters[1], self.is_deconv)
        self.up1 = cnnUp(filters[1], filters[0], self.is_deconv)
        self.final = nn.Conv2d(filters[0], self.n_classes, 1)
        self.flatten=torch.flatten
        self.out = nn.Linear
        self.relu=nn.Tanh()

    def forward(self, inputs, label_dsp_dim):
        down1 = self.down1(inputs)

        down2 = self.down2(down1)
        # down3  = self.down3(down2)
        # down4  = self.down4(down3)
        dropout1 = self.dropout1(down2)
        down3 = self.down3(dropout1)

        dropout2 = self.dropout1(down3)
        down4 = self.down4(dropout2)

        center = self.center(down4)
        up4 = self.up4(center)
        up3 = self.up3(up4)
        up2 = self.up2(up3)
        up1 = self.up1(up2)
        # print(up1.size())
        final1=self.final(up1)
        final1_f=self.flatten(final1,1,3)
        # print(final1_f.size(),'final1_f')
        out1=self.out(in_features=final1_f.size()[1], out_features= label_dsp_dim[1])(final1_f)
        out1 = self.relu(out1)
        out1 = self.out(in_features=label_dsp_dim[1], out_features=label_dsp_dim[1])(out1)
        out1=self.relu(out1)
        print(out1.size(),'1111111111111')
        # final2=final1[:,:,:,1:]
        # print(out1.size(),'out1')
        out1 = out1[:, :, 1:1 + label_dsp_dim[0], 1:1 + label_dsp_dim[1]].contiguous()   ##if applied in marmousi model, use it; else not.
        # up1 = up1[:, :, 1:1 + label_dsp_dim[0], 1:1 + label_dsp_dim[1]].contiguous()

        return out1

    # Initialization of Parameters
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.ConvTranspose2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()



class inverse_model_cnn(nn.Module):
    def __init__(self, in_channels, out_channels, nonlinearity, lr, lr_decray):
        super(inverse_model_cnn, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.activation = nn.Tanh()
        self.lr = lr
        self.lr_decray = lr_decray
        self.cnn1 = nn.Sequential(nn.Conv2d(in_channels=self.in_channels,
                                            out_channels=32,
                                            kernel_size=[3, 11],
                                            padding=(0, 5)),
                                  # 此处的num_groups为批归一化处理时，每组的通道数
                                  nn.GroupNorm(num_groups=8,
                                               num_channels=32),self.activation)   #(32,9,175)

        self.cnn2 = nn.Sequential(nn.Conv2d(in_channels=32,
                                            out_channels=32,
                                            kernel_size=[3, 11],
                                            padding=(2, 5),
                                            stride=(2, 1),
                                            ),
                                  nn.GroupNorm(num_groups=8,
                                               num_channels=32),self.activation)   #(32,5,175)

        self.cnn3  = nn.Sequential(nn.Conv2d(in_channels=32,
                                            out_channels=64,
                                            kernel_size=[3, 11],
                                            padding=(1, 5),
                                            ),
                                  nn.GroupNorm(num_groups=8,
                                               num_channels=64),self.activation)  #(64,5,175)


        self.cnn4 = nn.Sequential(nn.Conv2d(in_channels=64,
                                            out_channels=64,
                                            kernel_size=[3, 11],
                                            padding=(1, 5),
                                            stride=(2, 1),
                                            ),
                                  nn.GroupNorm(num_groups=8,
                                               num_channels=64),self.activation)  #(64,3,175)

        self.cnn5  = nn.Sequential(nn.Conv2d(in_channels=64,
                                            out_channels=128,
                                            kernel_size=[3, 11],
                                            padding=(1, 5),
                                            ),
                                  nn.GroupNorm(num_groups=8,
                                               num_channels=128),self.activation)   #(128,3,175)


        self.cnn6 = nn.Sequential(nn.Conv2d(in_channels=128,
                                            out_channels=128,
                                            kernel_size=[3, 11],
                                            padding=(1, 5),
                                            ),
                                  nn.GroupNorm(num_groups=8,
                                               num_channels=128),self.activation)

        self.cnn7 = nn.Sequential(nn.Conv2d(in_channels=128,
                                            out_channels=64,
                                            kernel_size=[3, 11],
                                            padding=(1, 5),
                                            ),
                                  nn.GroupNorm(num_groups=8,
                                               num_channels=64),self.activation)

        self.cnn8 = nn.Sequential(nn.Conv2d(in_channels=64,
                                            out_channels=64,
                                            kernel_size=[3, 11],
                                            padding=(1, 5),
                                            ),
                                  nn.GroupNorm(num_groups=8,
                                               num_channels=64),self.activation)

        self.cnn9 = nn.Sequential(nn.Conv2d(in_channels=64,
                                            out_channels=32,
                                            kernel_size=[3, 11],
                                            padding=(1, 5),
                                            ),
                                  nn.GroupNorm(num_groups=8,
                                               num_channels=32),self.activation)

        self.cnn10 = nn.Sequential(nn.Conv2d(in_channels=32,
                                            out_channels=32,
                                            kernel_size=[3, 11],
                                            padding=(1, 5),
                                            ),
                                  nn.GroupNorm(num_groups=8,
                                               num_channels=32),self.activation)

        self.cnn11 = nn.Sequential(nn.Conv2d(in_channels=32,
                                            out_channels=16,
                                            kernel_size=[3, 11],
                                            padding=(1, 5),
                                            ),
                                  nn.GroupNorm(num_groups=8,
                                               num_channels=16),self.activation)

        self.cnn12 = nn.Sequential(nn.Conv2d(in_channels=16,
                                            out_channels=16,
                                            kernel_size=[3, 11],
                                            padding=(1, 5),
                                            ),
                                  nn.GroupNorm(num_groups=8,
                                               num_channels=16),self.activation)
        self.cnn13 = nn.Sequential(nn.Conv2d(in_channels=16,
                                            out_channels=8,
                                            kernel_size=[3, 11],
                                            padding=(1, 5),
                                            ),
                                  nn.GroupNorm(num_groups=8,
                                               num_channels=8),self.activation)
        self.cnn14 = nn.Sequential(nn.Conv2d(in_channels=8,
                                            out_channels=8,
                                            kernel_size=[3, 11],
                                            padding=(1, 5),
                                            ),
                                  nn.GroupNorm(num_groups=8,
                                               num_channels=8),self.activation)
        self.cnn15 = nn.Sequential(nn.Conv2d(in_channels=8,
                                            out_channels=4,
                                            kernel_size=[3, 11],
                                            padding=(1, 5),
                                            ),
                                  nn.GroupNorm(num_groups=4,
                                               num_channels=4),self.activation)
        self.cnn16 = nn.Sequential(nn.Conv2d(in_channels=4,
                                            out_channels=4,
                                            kernel_size=[3, 11],
                                            padding=(1, 5),
                                            ),
                                  nn.GroupNorm(num_groups=4,
                                               num_channels=4),self.activation)
        self.cnn17 = nn.Sequential(nn.Conv2d(in_channels=4,
                                            out_channels=2,
                                            kernel_size=[3, 11],
                                            padding=(1, 5),
                                            ),
                                  nn.GroupNorm(num_groups=2,
                                               num_channels=2),self.activation)
        self.cnn18 = nn.Sequential(nn.Conv2d(in_channels=2,
                                            out_channels=2,
                                            kernel_size=[3, 11],
                                            padding=(1, 5),
                                            ),
                                  nn.GroupNorm(num_groups=2,
                                               num_channels=2),self.activation)


        self.cnn19 = nn.Sequential(nn.Conv2d(in_channels=2,
                                            out_channels=1,
                                            kernel_size=[3, 11],
                                            padding=(0, 5),
                                            ),
                                  nn.GroupNorm(num_groups=1,
                                               num_channels=1),self.activation)  #(1,1,175)

        # self.out = nn.Linear(in_features=32, out_features=self.out_channels)

        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.ConvTranspose1d):
                nn.init.xavier_uniform_(m.weight.data)
                m.bias.data.zero_()
            elif isinstance(m, nn.GroupNorm):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

        self.optimizer = optim.Adam(self.parameters(), self.lr, weight_decay=self.lr_decray)
        print("lr = {}".format(self.lr))

    def forward(self,  x):
        # print("x_2D.shape = {}".format(x_2D.size()))
        cnn_out1 = self.cnn1(x)
        # print(cnn_out1.size(),'cnn_out1')
        cnn_out2 = self.cnn2(cnn_out1)
        # print(cnn_out2.size(),'cnn_out2')
        cnn_out3 = self.cnn3(cnn_out2)
        # print(cnn_out3.size(),'cnn_out3')
        cnn_out4 = self.cnn4(cnn_out3)
        # print(cnn_out4.size(), 'cnn_out4')
        cnn_out5 = self.cnn5(cnn_out4)
        # print(cnn_out5.size(), 'cnn_out5')
        cnn_out6 = self.cnn6(cnn_out5)
        cnn_out7 = self.cnn7(cnn_out6)
        cnn_out8 = self.cnn8(cnn_out7)
        cnn_out9 = self.cnn9(cnn_out8)
        cnn_out10 = self.cnn10(cnn_out9)
        cnn_out11 = self.cnn11(cnn_out10)
        cnn_out12 = self.cnn12(cnn_out11)
        cnn_out13 = self.cnn13(cnn_out12)
        cnn_out14 = self.cnn14(cnn_out13)
        cnn_out15 = self.cnn15(cnn_out14)
        cnn_out16 = self.cnn16(cnn_out15)
        cnn_out17 = self.cnn17(cnn_out16)
        cnn_out18 = self.cnn18(cnn_out17)
        cnn_out19 = self.cnn19(cnn_out18)
        # print(cnn_out19.size())


        return cnn_out19
"""dropout"""
class UnetModelDropout(nn.Module):
    def __init__(self, n_classes, in_channels, is_deconv, is_batchnorm):
        super(UnetModelDropout, self).__init__()
        self.is_deconv = is_deconv
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm
        self.n_classes = n_classes

        # filters = [64, 128, 256, 512, 1024]
        filters = [ 4, 8, 16, 32,64]
        self.down1 = unetDown(self.in_channels, filters[0], self.is_batchnorm)
        self.down2 = unetDown(filters[0], filters[1], self.is_batchnorm)
        self.dropout1 = nn.Dropout(p=0.1)
        self.down3 = unetDown(filters[1], filters[2], self.is_batchnorm)
        self.dropout2 = nn.Dropout(p=0.1)
        self.dropout3 = nn.Dropout(p=0.1)
        self.dropout4 = nn.Dropout(p=0.1)
        self.dropout5 = nn.Dropout(p=0.1)
        self.dropout6 = nn.Dropout(p=0.1)
        self.dropout7 = nn.Dropout(p=0.1)
        self.down4 = unetDown(filters[2], filters[3], self.is_batchnorm)
        self.center = unetConv2(filters[3], filters[4], self.is_batchnorm)
        self.up4 = unetUp(filters[4], filters[3], self.is_deconv)
        self.up3 = unetUp(filters[3], filters[2], self.is_deconv)
        self.up2 = unetUp(filters[2], filters[1], self.is_deconv)
        self.up1 = unetUp(filters[1], filters[0], self.is_deconv)
        self.final = nn.Conv2d(filters[0], self.n_classes, 1)

    def forward(self, inputs, label_dsp_dim):
        down1 = self.down1(inputs)
        dropout1 = self.dropout1(down1)

        down2 = self.down2(dropout1)
        dropout2 = self.dropout1(down2)
        down3 = self.down3(dropout2)

        dropout3 = self.dropout3(down3)
        down4 = self.down4(dropout3)

        center = self.center(down4)
        up4 = self.up4(down4, center)
        up3 = self.up3(down3, up4)
        up2 = self.up2(down2, up3)
        up1 = self.up1(down1, up2)
        # print(up1.size(),'up1111')
        # up1 = up1[:, :, 1:1 + label_dsp_dim[0], 1:1 + label_dsp_dim[1]].contiguous()
        up1 = up1[:, :, 1:1 + label_dsp_dim[0], 0:1 + label_dsp_dim[1]].contiguous()     ##if applied in overthrust model, use it; else not.
        return self.final(up1)

    # Initialization of Parameters
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.ConvTranspose2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()


                    
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(100, 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(True),
            # nn.Dropout(p=0.2),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d( 8, 32, 4, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            # nn.Dropout(p=0.2),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(32, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            # nn.Dropout(p=0.2),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d( 64, 32, 4, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            # nn.Dropout(p=0.2),

            nn.ConvTranspose2d(32, 16, 4, 2, 1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.Dropout(p=0.1),

            nn.ConvTranspose2d(16, 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(4),
            nn.ReLU(True),
            nn.Dropout(p=0.1),


            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d( 4, 1, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input,NX,NZ,bound):
        return self.main(input)[0,0,:NX-bound,:NZ]
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.ConvTranspose2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()


class Seis_UnetModelDropout(nn.Module):
    def __init__(self, n_classes, in_channels, NX,is_deconv, is_batchnorm):
        super(Seis_UnetModelDropout, self).__init__()
        self.is_deconv = is_deconv
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm
        self.n_classes = n_classes
        self.nx=NX

        # filters = [64, 128, 256, 512, 1024]
        self.conv1d_1=nn.Conv1d(in_channels=self.in_channels,
                                           out_channels=32,
                                           kernel_size=21,padding=10)

        self.conv1d_2=nn.Conv1d(in_channels=32,
                                           out_channels=64,
                                           kernel_size=21,padding=10)
        self.conv1d_3=nn.Conv1d(in_channels=64,
                                           out_channels=128,
                                           kernel_size=21,padding=10)
        self.conv1d_4=nn.Conv1d(in_channels=128,
                                           out_channels=64,
                                           kernel_size=21,padding=10)
        self.conv1d_5=nn.Conv1d(in_channels=64,
                                           out_channels=32,
                                           kernel_size=21,padding=10)
        self.conv1d_6=nn.Conv1d(in_channels=32,
                                           out_channels=8,
                                           kernel_size=21,padding=10)
        filters = [ 4, 8, 16, 32,64]
        self.down1 = unetDown(80, filters[0], self.is_batchnorm)
        self.down2 = unetDown(filters[0], filters[1], self.is_batchnorm)
        self.dropout1 = nn.Dropout(p=0.1)
        self.down3 = unetDown(filters[1], filters[2], self.is_batchnorm)
        self.dropout2 = nn.Dropout(p=0.1)
        self.dropout3 = nn.Dropout(p=0.1)
        self.dropout4 = nn.Dropout(p=0.1)
        self.dropout5 = nn.Dropout(p=0.1)
        self.dropout6 = nn.Dropout(p=0.1)
        self.dropout7 = nn.Dropout(p=0.1)
        self.down4 = unetDown(filters[2], filters[3], self.is_batchnorm)
        self.center = unetConv2(filters[3], filters[4], self.is_batchnorm)
        self.up4 = unetUp(filters[4], filters[3], self.is_deconv)
        self.up3 = unetUp(filters[3], filters[2], self.is_deconv)
        self.up2 = unetUp(filters[2], filters[1], self.is_deconv)
        self.up1 = unetUp(filters[1], filters[0], self.is_deconv)
        self.final = nn.Conv2d(filters[0], self.n_classes, 1)

    def forward(self, inputs, label_dsp_dim):
        # print(inputs.shape)
        conv1=self.conv1d_1(inputs)
        conv2 = self.conv1d_2(conv1)
        # dropout1_2 = self.dropout1_2(conv2)
        conv3 = self.conv1d_3(conv2)
        conv4 = self.conv1d_4(conv3)

        conv5 = self.conv1d_5(conv4)
        conv6 = self.conv1d_6(conv5)

        # print(conv3.shape)
        conv6=conv6.reshape([-1,8*10,1,self.nx])    #最后一层的conv1d output_channel=8  10为NX的倍数  可调
        down1 = self.down1(conv6)

        dropout1 = self.dropout1(down1)

        down2 = self.down2(dropout1)
        dropout2 = self.dropout1(down2)
        down3 = self.down3(dropout2)

        dropout3 = self.dropout3(down3)
        down4 = self.down4(dropout3)

        center = self.center(down4)
        up4 = self.up4(down4, center)
        up3 = self.up3(down3, up4)
        up2 = self.up2(down2, up3)
        up1 = self.up1(down1, up2)
        # print(up1.size(),'up1111')
        up1 = up1[:, :, 1:1 + label_dsp_dim[0], 1:1 + label_dsp_dim[1]].contiguous()
        # up1 = up1[:, :, 1:1 + label_dsp_dim[0], 0:1 + label_dsp_dim[1]].contiguous()     ##if applied in overthrust model, use it; else not.
        return self.final(up1)

    # Initialization of Parameters
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.ConvTranspose2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()