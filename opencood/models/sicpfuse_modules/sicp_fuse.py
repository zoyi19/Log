# -*- coding: utf-8 -*-
# Author: Deyuan Qu <deyuanqu@my.unt.edu>
# License: TDG-Attribution-NonCommercial-NoDistrib

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import random
import pickle
import yaml
import numpy as np

# from opencood.models.sub_modules.torch_transformation_utils import \
#     warp_affine_simple：主要功能是根据一个仿射变换矩阵 M，将输入特征图 src 变换到目标空间的尺寸 dsize，
# 并返回变换后的结果。对特征图进行仿射变换，将发送方特征对齐到接收方的坐标系，特征的坐标对齐，是重叠区域计算的前提。
#但我们的坐标已经统一，是否就不需要这个了呢？
from opencood.models.common_modules.torch_transformation_utils import warp_affine_simple


class SpatialFusion(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SpatialFusion, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=1),
            nn.ReLU()
            )
        self.conv2 = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=1),
            nn.Sigmoid()
            )
        self.compChannels1 = nn.Sequential(
            nn.Conv2d(in_channels, 1, kernel_size=1, stride=1,padding=0),
            nn.BatchNorm2d(num_features=1),
            nn.ReLU()
            )
        self.compChannels2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1,padding=0),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU()
            )

    def generate_overlap_selector(self, selector):
        #是一个专门用来生成重叠区域的选择器的函数；基于转换后的坐标系计算像素点是否属于重叠区域
        #统计不同通道中的信息，生成一个可以标识重叠区域的选择器。
        overlap_sel = torch.mean(selector, 1).unsqueeze(0).cuda()
        return overlap_sel

    def generate_nonoverlap_selector(self, overlap_sel):
        #生成一个重叠区域的互补选择器
        non_overlap_sel = torch.tensor(np.where(overlap_sel.cpu() > 0, 0, 1)).cuda()
        return non_overlap_sel

    def forward(self, x, record_len, pairwise_t_matrix):
        # split x to rec feature and sed feature
        rec_feature = x[0,:,:,:].unsqueeze(0)
        sed_feature = x[1,:,:,:].unsqueeze(0)

        # transfer sed to rec's space
        t_matrix = pairwise_t_matrix[0][:2, :2, :, :]
        t_sed_feature = warp_affine_simple(sed_feature, t_matrix[0, 1, :, :].unsqueeze(0), (x.shape[2], x.shape[3]))

        # generate overlap selector and non-overlap selector
        #基于发送方特征的投影结果，生成重叠和非重叠区域的选择器。
        selector = torch.ones_like(sed_feature)# # 初始化选择器为全1，表示所有区域均被选中。
        '''
        下面这行代码不知道是否需要删掉,这个是进行坐标仿射变换到ego坐标系,但我们用的应该都是统一的坐标系？
        '''
        selector = warp_affine_simple(selector, t_matrix[0, 1, :, :].unsqueeze(0), (x.shape[2], x.shape[3])) #将 selector 投影到接收方的坐标系中。
        overlap_sel = self.generate_overlap_selector(selector) # overlap area selector   计算重叠区域选择器，函数定义就在上面几十行
        non_overlap_sel = self.generate_nonoverlap_selector(overlap_sel) # non-overlap area selector  计算非重叠区域选择器

        '''权重计算更迭,SICP我们大概需要主要用到的,替代where2com的权重。'''
        # generate the weight map
        cat_feature = torch.cat((rec_feature, t_sed_feature), dim=1)
        comp_feature = self.compChannels1(cat_feature)
        f1 = self.conv1(comp_feature)       
        f2 = self.conv2(f1)     
        weight_map = comp_feature + f2

        # normalize the weight map to [0,1]
        normalize_weight_map = (weight_map - torch.min(weight_map)) / (torch.max(weight_map) - torch.min(weight_map))

        '''根据重叠和非重叠区域，对接收方和发送方的特征进行加权融合 '''
        # apply normalized weight map to rec_feature and t_sed_feature 
        weight_to_rec = rec_feature * (normalize_weight_map * overlap_sel + non_overlap_sel)
        weight_to_t_sed = t_sed_feature * (1 - normalize_weight_map)

        x = torch.cat((weight_to_rec, weight_to_t_sed), dim=1)
        x = self.compChannels2(x) 

        return x
    #输出特征整合了接收方和发送方的语义信息，同时保留了重叠区域和非重叠区域的特性