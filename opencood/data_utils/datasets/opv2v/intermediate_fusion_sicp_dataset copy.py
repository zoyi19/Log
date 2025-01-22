# -*- coding: utf-8 -*-
# Author: Runsheng Xu <rxx3386@ucla.edu>
# Modified by: Deyuan Qu <deyuanqu@my.unt.edu>
# License: TDG-Attribution-NonCommercial-NoDistrib

# Jinlong Wang - 2024-09-04 4DRadar+lidar
"""
Dataset class for intermediate fusion
"""
import random
import math
import warnings
import torch
import numpy as np
import opencood.data_utils.datasets
import opencood.data_utils.post_processor as post_processor

from collections import OrderedDict
from itertools import islice
from opencood.utils import box_utils
from opencood.data_utils.datasets.opv2v import basedataset
from opencood.data_utils.pre_processor import build_preprocessor
from opencood.utils.pcd_utils import \
    mask_points_by_range, mask_ego_points, shuffle_points, \
    downsample_lidar_minimum
from opencood.utils.transformation_utils import x1_to_x2

from opencood.pcdet_utils.roiaware_pool3d.roiaware_pool3d_utils \
            import points_in_boxes_cpu

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import opencood.visualization.simple_plot3d.canvas_3d as canvas_3d
import opencood.visualization.simple_plot3d.canvas_bev as canvas_bev

class IntermediateFusionSicpDataset(basedataset.BaseDataset):
    """
    This class is for intermediate fusion where each vehicle transmit the
    deep features to ego.
    """
    def __init__(self, params, visualize, train=True):
        super(IntermediateFusionSicpDataset, self). \
            __init__(params, visualize, train)
        # if project first, cav's lidar will first be projected to
        # the ego's coordinate frame. otherwise, the feature will be
        # projected instead.        
        self.train = train
        self.proj_first = True        
        if 'proj_first' in params['fusion']['args'] and \
            not params['fusion']['args']['proj_first']:
            self.proj_first = False

        # whether there is a time delay between the time that cav project
        # lidar to ego and the ego receive the delivered feature
        self.cur_ego_pose_flag = True if 'cur_ego_pose_flag' not in \
            params['fusion']['args'] else \
            params['fusion']['args']['cur_ego_pose_flag']

        self.pre_processor = build_preprocessor(params['preprocess'],
                                                train)
        self.post_processor = post_processor.build_postprocessor(
            params['postprocess'],
            dataset='opv2v', 
            train=train)

        self.use_comm_range_check = True   #通信范围检查用于筛选出与 Ego 有效通信的车辆。     
        if 'use_comm_range_check' in params['fusion']['args'] and \
            not params['fusion']['args']['use_comm_range_check']:
            self.use_comm_range_check = False


        
    def __getitem__(self, idx):
        #数据加载
        # base_data_dict = self.retrieve_base_data(idx,
        #                                          cur_ego_pose_flag=self.cur_ego_pose_flag)
        base_data_dict, _, _ = self.retrieve_base_data(idx, cur_ego_pose_flag=self.cur_ego_pose_flag)
        #加载索引的数据，包含场景中各车辆的点云、位置信息。
        processed_data_dict = OrderedDict()
        processed_data_dict['ego'] = {}

        ego_id = -1
        ego_lidar_pose = []

        # first find the ego vehicle's lidar pose   
        for cav_id, cav_content in base_data_dict.items(): #找寻到ego，找到后退出循环
            if cav_content['ego']:
                ego_id = cav_id
                ego_lidar_pose = cav_content['params']['lidar_pose']   #确定当前场景下Ego车辆的Lidar位姿
                break        
        assert cav_id == list(base_data_dict.keys())[
            0], "The first element in the OrderedDict must be ego"
        assert ego_id != -1 #验证ego的存在性和顺序
        assert len(ego_lidar_pose) > 0
       
        pairwise_t_matrix = \
            self.get_pairwise_transformation(base_data_dict,
                                             self.max_cav) # 计算所有车辆与Ego之间的变换矩阵，用于将点云投影到统一坐标系
        ''' 上面的代码是将不同agent投影到统一的ego坐标系中,我们已经使用了统一的坐标系,是否这些就不需要了'''
        processed_features = []
        object_stack = []
        object_id_stack = []

        object_stack_ego = []
        object_id_stack_ego = []

        # prior knowledge for time delay correction and indicating data type
        # (V2V vs V2i)
        velocity = []
        time_delay = []
        infra = []
        spatial_correction_matrix = []

        if self.visualize:
            projected_lidar_stack = []

        # First-Come-First-Serve
        for cav_id, selected_cav_base in islice(base_data_dict.items(), 2):         
            # for ego car only
            if cav_id == ego_id:
                selected_cav_processed = self.get_item_single_car(
                    selected_cav_base,
                    ego_lidar_pose)
                object_stack_ego = selected_cav_processed['object_bbx_center'] #根据当前车辆和自车Lidar，输出ego车辆的边界框信息和物体id
                object_id_stack_ego = selected_cav_processed['object_ids']
            '''检测通信范围'''
            # check if the cav is within the communication range with ego
            distance = \
                math.sqrt((selected_cav_base['params']['lidar_pose'][0] -
                           ego_lidar_pose[0]) ** 2 + (
                                  selected_cav_base['params'][
                                      'lidar_pose'][1] - ego_lidar_pose[
                                      1]) ** 2)
            
            if self.train:  #通信距离检查，仅处理通信范围内的车辆。   
                if self.use_comm_range_check:   
                    if distance > opencood.data_utils.datasets.COM_RANGE:
                        continue  
            else:
                if distance > opencood.data_utils.datasets.COM_RANGE:
                    continue  
            # Project the lidar and bbx to ego space first, and then do clipping.
            selected_cav_processed = self.get_item_single_car(
                selected_cav_base,
                ego_lidar_pose)

            object_stack.append(selected_cav_processed['object_bbx_center'])#当前车辆检查到的目标边界框
            object_id_stack += selected_cav_processed['object_ids'] #物体id
            processed_features.append(
                selected_cav_processed['processed_features'])#将当前车辆处理后的特征数据追加到 processed_features 中

            velocity.append(selected_cav_processed['velocity'])
            time_delay.append(float(selected_cav_base['time_delay']))
            # this is only useful when proj_first = True, and communication
            # delay is considered. Right now only V2X-ViT utilizes the
            # spatial_correction. There is a time delay when the cavs project
            # their lidar to ego and when the ego receives the feature, and
            # this variable is used to correct such pose difference (ego_t-1 to
            # ego_t)
            '''我们没有延迟，这里是不是可以直接删掉'''
            spatial_correction_matrix.append(
                selected_cav_base['params']['spatial_correction_matrix'])
            infra.append(1 if int(cav_id) < 0 else 0)
            
            if self.visualize:
                projected_lidar_stack.append(
                    selected_cav_processed['projected_lidar']) #将处理后的投影点云保存到 projected_lidar_stack

            # if self.visualize:
            #     if cav_id == ego_id:
            #         projected_lidar_stack.append(
            #             selected_cav_processed['projected_lidar'])

        # exclude all repetitive objects 除重
        unique_indices = \
            [object_id_stack.index(x) for x in set(object_id_stack)]
        object_stack = np.vstack(object_stack) #将所有协作车辆检测到的边界框信息按行堆叠为一个二维数组
        object_stack = object_stack[unique_indices] #按唯一索引提取去重后的边界框

        # make sure bounding boxes across all frames have the same number 确保目标边界框的数量固定
        object_bbx_center = \
            np.zeros((self.params['postprocess']['max_num'], 7))
        mask = np.zeros(self.params['postprocess']['max_num']) #创建一个零向量 mask，用于记录有效的边界框。
        object_bbx_center[:object_stack.shape[0], :] = object_stack #有效信息填充
        mask[:object_stack.shape[0]] = 1
        #Ego 车辆的边界框标准化
        object_bbx_center_ego = \
            np.zeros((self.params['postprocess']['max_num'], 7))
        mask_ego = np.zeros(self.params['postprocess']['max_num'])
        object_bbx_center_ego[:object_stack_ego.shape[0], :] = object_stack_ego
        mask_ego[:object_stack_ego.shape[0]] = 1

        # merge preprocessed features from different cavs into the same dict 融合所有agent的特征
        cav_num = len(processed_features) #计算有效CAV数量
        merged_feature_dict = self.merge_features_to_dict(processed_features) #将不同车辆的特征按照键值对形式合并，输出一个字典

        # generate the anchor boxes 为整个场景和 Ego 车辆分别生成目标检测所需的锚框
        anchor_box = self.post_processor.generate_anchor_box()

        anchor_box_ego = self.post_processor.generate_anchor_box()

        # generate targets label 生成与锚框匹配的目标检测标签
        label_dict = \
            self.post_processor.generate_label(
                gt_box_center=object_bbx_center,
                anchors=anchor_box,
                mask=mask)

        # generate targets label for ego car only
        label_dict_ego = \
            self.post_processor.generate_label(
                gt_box_center=object_bbx_center_ego,
                anchors=anchor_box_ego,
                mask=mask_ego)

        # pad dv, dt, infra to max_cav
        #补全速度、时间延迟】基础设施判定（infra_1:ja）
        velocity = velocity + (self.max_cav - len(velocity)) * [0.]
        time_delay = time_delay + (self.max_cav - len(time_delay)) * [0.]
        infra = infra + (self.max_cav - len(infra)) * [0.]
        spatial_correction_matrix = np.stack(spatial_correction_matrix)#将每辆车的修正矩阵（4x4 齐次变换矩阵）堆叠为一个三维数组。
        velocity = velocity + (self.max_cav - len(velocity)) * [0.]
        time_delay = time_delay + (self.max_cav - len(time_delay)) * [0.]
        infra = infra + (self.max_cav - len(infra)) * [0.]
        spatial_correction_matrix = np.stack(spatial_correction_matrix)
        padding_eye = np.tile(np.eye(4)[None],(self.max_cav - len(
                                               spatial_correction_matrix),1,1))#创建若干个单位矩阵（4x4），用于填充不足的车辆数。
        spatial_correction_matrix = np.concatenate([spatial_correction_matrix,
                                                   padding_eye], axis=0)#将原始修正矩阵和单位矩阵拼接在一起，确保形状为 (max_cav, 4, 4)。

        processed_data_dict['ego'].update( #将处理后的所有数据存储到 processed_data_dict['ego']，供后续模块使用。
            {'object_bbx_center': object_bbx_center,
             'object_bbx_mask': mask,
             'object_ids': [object_id_stack[i] for i in unique_indices],
             'anchor_box': anchor_box,
             'processed_lidar': merged_feature_dict,
             'label_dict': label_dict,
             'label_dict_ego': label_dict_ego,
             'cav_num': cav_num,
             'velocity': velocity,
             'time_delay': time_delay,
             'infra': infra,
             'spatial_correction_matrix': spatial_correction_matrix,
             'pairwise_t_matrix': pairwise_t_matrix})

        if self.visualize: #将所有车辆的投影点云堆叠到一起，并存储到 processed_data_dict['ego']['origin_lidar'] 中。
            processed_data_dict['ego'].update({'origin_lidar':
                np.vstack(
                    projected_lidar_stack)})
        return processed_data_dict


    def get_item_single_car(self, selected_cav_base, ego_pose):
        """
        Project the lidar and bbx to ego space first, and then do clipping.

        Parameters
        ----------
        selected_cav_base : dict
            The dictionary contains a single CAV's raw information.
        ego_pose : list
            The ego vehicle lidar pose under world coordinate.

        Returns
        -------
        selected_cav_processed : dict
            The dictionary contains the cav's processed information.
        针对单独的一个cav:
        将其点云数据投影到 Ego 车辆的坐标系。
        提取目标物体信息（边界框和 ID).
        对点云数据进行滤波和体素化预处理。
        返回包含处理结果的字典.
        """
        selected_cav_processed = {}

        # calculate the transformation matrix
        transformation_matrix = \
            selected_cav_base['params']['transformation_matrix']

        # retrieve objects under ego coordinates
        object_bbx_center, object_bbx_mask, object_ids = \
            self.post_processor.generate_object_center([selected_cav_base],
                                                       ego_pose)

        # filter lidar
        lidar_np = selected_cav_base['lidar_np']
        lidar_np = shuffle_points(lidar_np)
        # remove points that hit itself
        lidar_np = mask_ego_points(lidar_np) #过滤掉点云数据中属于车辆自身的部分（通常是 LiDAR 探测到的车体表面点）。
        # project the lidar to ego space
        if self.proj_first:
            lidar_np[:, :3] = \
                box_utils.project_points_by_matrix_torch(lidar_np[:, :3],
                                                         transformation_matrix)

        lidar_np = mask_points_by_range(lidar_np,
                                        self.params['preprocess'][
                                            'cav_lidar_range'])#裁剪超过ROI区域的点云
        processed_lidar = self.pre_processor.preprocess(lidar_np)

        # velocity，速度归一化，便于特征融合时使用
        velocity = selected_cav_base['params']['ego_speed']
        # normalize veloccity by average speed 30 km/h
        velocity = velocity / 30
        
        selected_cav_processed.update(
            {'object_bbx_center': object_bbx_center[object_bbx_mask == 1],
             'object_ids': object_ids,
             'projected_lidar': lidar_np,
             'processed_features': processed_lidar,
             'velocity': velocity})

        return selected_cav_processed

    @staticmethod
    def merge_features_to_dict(processed_feature_list):
        """
        Merge the preprocessed features from different cavs to the same
        dictionary.
        将不同CAV的处理后的特征合并到一个字典,以便后续处理
        Parameters
        ----------
        processed_feature_list : list
            A list of dictionary containing all processed features from
            different cavs.

        Returns
        -------
        merged_feature_dict: dict
            key: feature names, value: list of features.
        """

        merged_feature_dict = OrderedDict()
        
        for i in range(len(processed_feature_list)):
            for feature_name, feature in processed_feature_list[i].items():
                if feature_name not in merged_feature_dict:
                    merged_feature_dict[feature_name] = []
                if isinstance(feature, list):
                    merged_feature_dict[feature_name] += feature
                else:
                    merged_feature_dict[feature_name].append(feature)

        return merged_feature_dict

    def collate_batch_train(self, batch):
        '''功能：
        初始化存储容器。
        遍历批次中的样本，提取并存储其数据。
        将数据转换为 PyTorch 张量。
        合并点云特征并处理标签。
        生成先验编码和变换矩阵。
        构造最终的输出字典。
       

        '''
        # Intermediate fusion is different the other two
        output_dict = {'ego': {}}

        object_bbx_center = []
        object_bbx_mask = []
        object_ids = []
        processed_lidar_list = []
        # used to record different scenario
        record_len = []
        label_dict_list = []
        label_dict_ego_list = []

        # used for PriorEncoding for models
        velocity = []
        time_delay = []
        infra = []

        # pairwise transformation matrix
        pairwise_t_matrix_list = []

        # used for correcting the spatial transformation between delayed timestamp
        # and current timestamp
        spatial_correction_matrix_list = []

        if self.visualize:
            origin_lidar = []

        for i in range(len(batch)):
            ego_dict = batch[i]['ego']
            object_bbx_center.append(ego_dict['object_bbx_center'])
            object_bbx_mask.append(ego_dict['object_bbx_mask'])
            object_ids.append(ego_dict['object_ids'])

            processed_lidar_list.append(ego_dict['processed_lidar'])
            record_len.append(ego_dict['cav_num'])
            label_dict_list.append(ego_dict['label_dict'])
            pairwise_t_matrix_list.append(ego_dict['pairwise_t_matrix'])
            label_dict_ego_list.append(ego_dict['label_dict_ego'])

            velocity.append(ego_dict['velocity'])
            time_delay.append(ego_dict['time_delay'])
            infra.append(ego_dict['infra'])
            spatial_correction_matrix_list.append(
                ego_dict['spatial_correction_matrix'])

            if self.visualize:
                origin_lidar.append(ego_dict['origin_lidar'])
        # convert to numpy, (B, max_num, 7)  
        '''张量转换,形式如上'''
        object_bbx_center = torch.from_numpy(np.array(object_bbx_center))
        object_bbx_mask = torch.from_numpy(np.array(object_bbx_mask))

        # example: {'voxel_features':[np.array([1,2,3]]),
        # np.array([3,5,6]), ...]}
        merged_feature_dict = self.merge_features_to_dict(processed_lidar_list)#合并点云信息
        processed_lidar_torch_dict = \
            self.pre_processor.collate_batch(merged_feature_dict) # 对合并的特征进行批量化处理，生成统一格式
        # [2, 3, 4, ..., M], M <= max_cav
        record_len = torch.from_numpy(np.array(record_len, dtype=int))
        label_torch_dict = \
            self.post_processor.collate_batch(label_dict_list) #使用后处理器 collate_batch 对普通和自车的标签进行整理，生成模型输入需要的格式。

        label_torch_dict_ego = \
            self.post_processor.collate_batch(label_dict_ego_list)

        # (B, max_cav)
        velocity = torch.from_numpy(np.array(velocity))
        time_delay = torch.from_numpy(np.array(time_delay))
        infra = torch.from_numpy(np.array(infra))
        spatial_correction_matrix_list = \
            torch.from_numpy(np.array(spatial_correction_matrix_list))
        # (B, max_cav, 3)
        prior_encoding = \
            torch.stack([velocity, time_delay, infra], dim=-1).float()
        # (B, max_cav)
        pairwise_t_matrix = torch.from_numpy(np.array(pairwise_t_matrix_list))

        # object id is only used during inference, where batch size is 1.
        # so here we only get the first element.
        output_dict['ego'].update({'object_bbx_center': object_bbx_center,
                                   'object_bbx_mask': object_bbx_mask,
                                   'processed_lidar': processed_lidar_torch_dict,
                                   'record_len': record_len,
                                   'label_dict': label_torch_dict,
                                   'label_dict_ego': label_torch_dict_ego,
                                   'object_ids': object_ids[0],
                                   'prior_encoding': prior_encoding,
                                   'spatial_correction_matrix': spatial_correction_matrix_list,
                                   'pairwise_t_matrix': pairwise_t_matrix})

        if self.visualize:
            origin_lidar = \
                np.array(downsample_lidar_minimum(pcd_np_list=origin_lidar))
            origin_lidar = torch.from_numpy(origin_lidar)
            output_dict['ego'].update({'origin_lidar': origin_lidar})
       
        return output_dict

    def collate_batch_test(self, batch):
        assert len(batch) <= 1, "Batch size 1 is required during testing!"
        output_dict = self.collate_batch_train(batch)

        # check if anchor box in the batch
        if batch[0]['ego']['anchor_box'] is not None:
            output_dict['ego'].update({'anchor_box':
                torch.from_numpy(np.array(
                    batch[0]['ego'][
                        'anchor_box']))})

        # save the transformation matrix (4, 4) to ego vehicle
        transformation_matrix_torch = \
            torch.from_numpy(np.identity(4)).float()
        output_dict['ego'].update({'transformation_matrix':
                                       transformation_matrix_torch})

        return output_dict

    def post_process(self, data_dict, output_dict):
        """
        Process the outputs of the model to 2D/3D bounding box.

        Parameters
        ----------
        data_dict : dict
            The dictionary containing the origin input data of model.

        output_dict :dict
            The dictionary containing the output of the model.

        Returns
        -------
        pred_box_tensor : torch.Tensor
            The tensor of prediction bounding box after NMS.
        gt_box_tensor : torch.Tensor
            The tensor of gt bounding box.
        """
        pred_box_tensor, pred_score = \
            self.post_processor.post_process(data_dict, output_dict)
        gt_box_tensor = self.post_processor.generate_gt_bbx(data_dict)

        return pred_box_tensor, pred_score, gt_box_tensor

    def get_pairwise_transformation(self, base_data_dict, max_cav):
        """
        Get pair-wise transformation matrix accross different agents.

        Parameters
        ----------
        base_data_dict : dict
            Key : cav id, item: transformation matrix to ego, lidar points.

        max_cav : int
            The maximum number of cav, default 5

        Return
        ------
        pairwise_t_matrix : np.array
            The pairwise transformation matrix across each cav.
            shape: (L, L, 4, 4)
        """
        pairwise_t_matrix = np.zeros((max_cav, max_cav, 4, 4))

        if self.proj_first:
            # if lidar projected to ego first, then the pairwise matrix
            # becomes identity
            pairwise_t_matrix[:, :] = np.identity(4)
        else:
            warnings.warn("Projection later is not supported in "
                          "the current version. Using it will throw"
                          "an error.")
            t_list = []

            # save all transformation matrix in a list in order first.
            for cav_id, cav_content in base_data_dict.items():
                t_list.append(cav_content['params']['transformation_matrix'])

            for i in range(len(t_list)):
                for j in range(len(t_list)):
                    # identity matrix to self
                    if i == j:
                        t_matrix = np.eye(4)
                        pairwise_t_matrix[i, j] = t_matrix
                        continue
                    # i->j: TiPi=TjPj, Tj^(-1)TiPi = Pj
                    t_matrix = np.dot(np.linalg.inv(t_list[j]), t_list[i])
                    pairwise_t_matrix[i, j] = t_matrix

        return pairwise_t_matrix