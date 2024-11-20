"""
Dataset class for early fusion

4DRadar 

SCOPE 
"""
import math
from collections import OrderedDict

import numpy as np
import torch

import opencood
import opencood.data_utils.post_processor as post_processor
from opencood.utils import box_utils
from opencood.data_utils.datasets.opv2v import basedataset
from opencood.data_utils.pre_processor import build_preprocessor
from opencood.utils.pcd_utils import \
    mask_points_by_range, mask_ego_points, shuffle_points, \
    downsample_lidar_minimum

from opencood.pcdet_utils.roiaware_pool3d.roiaware_pool3d_utils \
            import points_in_boxes_cpu

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import opencood.visualization.simple_plot3d.canvas_3d as canvas_3d
import opencood.visualization.simple_plot3d.canvas_bev as canvas_bev

class IntermediateFusionDataset(basedataset.BaseDataset):
    def __init__(self, params, visualize, train=True):
        super(IntermediateFusionDataset, self). \
            __init__(params, visualize, train)
        self.cur_ego_pose_flag = params['fusion']['args']['cur_ego_pose_flag']
        self.frame = params['train_params']['frame']
        self.pre_processor = build_preprocessor(params['preprocess'],
                                                train)
        self.post_processor = post_processor.build_postprocessor(
            params['postprocess'],
            dataset='opv2v', 
            train=train)

        self.proj_first = False
        if 'proj_first' in params['fusion']['args']:
            self.proj_first = params['fusion']['args']['proj_first']
        print('proj_first: ', self.proj_first)

        self.use_Radar = True
        self.use_Lidar = True
        self.training = train

        self.fov = True # OpenCOOD is True
        if 'fov' in params:
            self.fov = params['fov']

        if 'use_modality' in params['model']['args']:
            if params['model']['args']['use_modality'] == 'processed_radar':
                self.use_Lidar = False
            if params['model']['args']['use_modality'] == 'processed_lidar':
                self.use_Radar = False          


    def __getitem__(self, idx):
        select_num = self.frame
        select_dict,scenario_index,index_list,timestamp_index = self.retrieve_multi_data(idx,select_num,cur_ego_pose_flag=self.cur_ego_pose_flag)
        if timestamp_index < select_num:
            idx += select_num 
        try:
            assert idx == list(select_dict.keys())[
                0], "The first element in the multi frame must be current index"
        except AssertionError as aeeor:
            print("assert error dataset",list(select_dict.keys()),idx,timestamp_index)
        processed_data_list = []
        ego_id = -1
        ego_lidar_pose = []
        ego_id_list = []
        for index,base_data_dict in select_dict.items():
            processed_data_dict = OrderedDict()
            processed_data_dict['ego'] = {}

            if index == idx:
                # first find the ego vehicle's lidar pose
                for cav_id, cav_content in base_data_dict.items():
                    if cav_content['ego']:
                        ego_id = cav_id
                        ego_params = cav_content['params']
                        ego_lidar_pose = cav_content['params']['lidar_pose']
                        break
                assert cav_id == list(base_data_dict.keys())[
                    0], "The first element in the OrderedDict must be ego"
            assert ego_id != -1
            assert len(ego_lidar_pose) > 0
            ego_id_list.append(ego_id)
            # this is used for v2vnet and disconet
            pairwise_t_matrix = \
                self.get_pairwise_transformation(base_data_dict,
                                                self.params['train_params'][
                                                    'max_cav'])

            # processed_features = []
            object_stack = []
            object_id_stack = []

            object_id_stack_exten  = []
            processed_lidar_features = []
            processed_radar_features = []

            # prior knowledge for time delay correction and indicating data type
            # (V2V vs V2i)
            velocity = []
            time_delay = []
            infra = []
            spatial_correction_matrix = []

            if self.visualize:
                projected_lidar_stack = []

            projected_lidar_stack = []
            projected_radar_stack = []
            sum1 = 0
            sum2 = 0

            # loop over all CAVs to process information
            for cav_id, selected_cav_base in base_data_dict.items():
                # check if the cav is within the communication range with ego
                distance = \
                    math.sqrt((selected_cav_base['params']['lidar_pose'][0] -
                            ego_lidar_pose[0]) ** 2 + (
                                    selected_cav_base['params'][
                                        'lidar_pose'][1] - ego_lidar_pose[
                                        1]) ** 2)
                if distance > opencood.data_utils.datasets.COM_RANGE and index == idx:
                    continue

                # selected_cav_processed, void_lidar = self.get_item_single_car(
                #     selected_cav_base,
                #     ego_lidar_pose)

                selected_cav_processed, void_lidar = self.get_item_single_car(
                    selected_cav_base,
                    ego_lidar_pose,
                    ego_params)
                
                if void_lidar:
                    continue

                object_stack.append(selected_cav_processed['object_bbx_center'])
                # object_id_stack += selected_cav_processed['object_ids']
                object_id_stack.append(selected_cav_processed['object_ids'])
                # processed_features.append(
                #     selected_cav_processed['processed_features'])
                sum1+=len(selected_cav_processed['object_ids'])
                sum2+=len(selected_cav_processed['object_bbx_center'])
                object_id_stack_exten += selected_cav_processed['object_ids']
                processed_lidar_features.append(
                        selected_cav_processed['processed_lidar_features'])
                processed_radar_features.append(
                        selected_cav_processed['processed_radar_features'])

                velocity.append(selected_cav_processed['velocity'])
                time_delay.append(float(selected_cav_base['time_delay']))
                spatial_correction_matrix.append(
                    selected_cav_base['params']['spatial_correction_matrix'])
                infra.append(1 if int(cav_id) < 0 else 0)

                # if self.visualize:
                #     projected_lidar_stack.append(
                #         selected_cav_processed['projected_lidar'])
                projected_lidar_stack.append(
                        selected_cav_processed['projected_lidar'])
                projected_radar_stack.append(
                        selected_cav_processed['projected_radar'])

            # Filter empty boxes
            object_stack_filtered = []
            object_id_stack_filtered = []
            empty_mask_num = 0
            cached_path = None
            if self.use_Lidar and self.use_Radar:
                cached_path = '/mnt/16THDD/hx/BM2CP/opencood/cached_data/LR/'
            elif self.use_Lidar:
                cached_path = '/mnt/16THDD/hx/BM2CP/opencood/cached_data/L/'
            elif self.use_Radar:
                cached_path = '/mnt/16THDD/hx/BM2CP/opencood/cached_data/R/'
            
            if self.fov:
                for boxes, object_id, points, r_points in zip(object_stack, object_id_stack, projected_lidar_stack, projected_radar_stack):
                    if self.use_Lidar and self.use_Radar:
                        # #use cached points in boxes to save time 
                        # if cached_path is not None and os.path.exists(cached_path):
                        #     cached_file = os.path.join(cached_path, base_data_dict[ego_id]['params']['scenario'] + base_data_dict[ego_id]['params']['timestamp'] + '.npy')
                        # if not os.path.exists(cached_path):
                        point_indices = points_in_boxes_cpu(points[:, :3], boxes[:,
                                                                    [0, 1, 2, 5, 4,
                                                                        3, 6]])
                        r_point_indices = points_in_boxes_cpu(r_points[:, :3], boxes[:,
                                                                        [0, 1, 2, 5, 4,
                                                                            3, 6]])
                                                        
                        cur_mask = point_indices.sum(axis=1) > 0
                        cur_r_mask = r_point_indices.sum(axis=1) > 0
                        if  cur_mask.sum() > 0 or  cur_r_mask.sum() > 0:
                            all_mask = cur_mask | cur_r_mask
                            empty_mask_num += (~all_mask).sum()
                            object_stack_filtered.append(boxes[all_mask])                     
                            object_id_stack_filtered.extend(np.array(object_id)[all_mask])
                    elif self.use_Lidar:
                        point_indices = points_in_boxes_cpu(points[:, :3], boxes[:,
                                                                    [0, 1, 2, 5, 4,
                                                                        3, 6]])
                        cur_mask = point_indices.sum(axis=1) > 0
                        if  cur_mask.sum() > 0:
                            empty_mask_num += (~cur_mask).sum()
                            object_stack_filtered.append(boxes[cur_mask])
                            object_id_stack_filtered.extend(np.array(object_id)[cur_mask])
                    elif self.use_Radar:
                        point_indices = points_in_boxes_cpu(r_points[:, :3], boxes[:,
                                                                    [0, 1, 2, 5, 4,
                                                                        3, 6]])
                        cur_mask = point_indices.sum(axis=1) > 0
                        # print('cur_mask', cur_mask)
                        # print("object_id", object_id)
                        if  cur_mask.sum() > 0 :
                            empty_mask_num += (~cur_mask).sum()
                            object_stack_filtered.append(boxes[cur_mask])
                            object_id_stack_filtered.extend(np.array(object_id)[cur_mask])

                object_id_stack = object_id_stack_filtered
                object_stack = object_stack_filtered
            else:
                object_id_stack = object_id_stack_exten

            # exclude all repetitive objects
            unique_indices = \
                [object_id_stack.index(x) for x in set(object_id_stack)]


            if self.training and len(object_stack) == 0:
                new_index = np.random.randint(self.__len__())
                #base_data_dict[ego_id]['params']['scenario'] + base_data_dict[ego_id]['params']['timestamp'] + '.npy')
                return self.__getitem__(new_index)
            elif len(object_stack) == 0:
                object_stack = [np.zeros((1,7))]

            object_stack = np.vstack(object_stack)
            object_stack = object_stack[unique_indices]
            
            # make sure bounding boxes across all frames have the same number
            object_bbx_center = \
                np.zeros((self.params['postprocess']['max_num'], 7))
            mask = np.zeros(self.params['postprocess']['max_num'])
            object_bbx_center[:object_stack.shape[0], :] = object_stack
            mask[:object_stack.shape[0]] = 1

            # merge preprocessed features from different cavs into the same dict
            # cav_num = len(processed_features)
            # merged_feature_dict = self.merge_features_to_dict(processed_features)
            cav_num = len(processed_lidar_features)

            merged_lidar_feature_dict = self.merge_features_to_dict(processed_lidar_features)
            merged_radar_feature_dict = self.merge_features_to_dict(processed_radar_features)

            # generate the anchor boxes
            anchor_box = self.post_processor.generate_anchor_box()

            # generate targets label
            label_dict = \
                self.post_processor.generate_label(
                    gt_box_center=object_bbx_center,
                    anchors=anchor_box,
                    mask=mask)

            # pad dv, dt, infra to max_cav
            velocity = velocity + (self.max_cav - len(velocity)) * [0.]
            time_delay = time_delay + (self.max_cav - len(time_delay)) * [0.]
            infra = infra + (self.max_cav - len(infra)) * [0.]
            spatial_correction_matrix = np.stack(spatial_correction_matrix)
            padding_eye = np.tile(np.eye(4)[None],(self.max_cav - len(
                                                spatial_correction_matrix),1,1))
            spatial_correction_matrix = np.concatenate([spatial_correction_matrix, padding_eye], axis=0)

            processed_data_dict['ego'].update(
                {'object_bbx_center': object_bbx_center,
                'object_bbx_mask': mask,
                'object_ids': [object_id_stack[i] for i in unique_indices],
                'anchor_box': anchor_box,
                # 'processed_lidar': merged_feature_dict,
                'projected_radar': projected_radar_stack,
                'projected_lidar': projected_lidar_stack,
                'processed_lidar': merged_lidar_feature_dict,
                'processed_radar': merged_radar_feature_dict,
                'label_dict': label_dict,
                'cav_num': cav_num,
                'velocity': velocity,
                'time_delay': time_delay,
                'infra': infra,
                'spatial_correction_matrix': spatial_correction_matrix,
                'pairwise_t_matrix': pairwise_t_matrix})

            if self.visualize:
                processed_data_dict['ego'].update({'origin_lidar':
                    np.vstack(
                        projected_lidar_stack)})
            processed_data_list.append(processed_data_dict)
        try:
            assert len(set(ego_id_list)) == 1, "The ego id must be same"
        except AssertionError as aeeor:
            print("assert error ego id",ego_id_list)
        return processed_data_list

    def mask_ego_fov_flag(self, selected_cav_base, lidar, ego_params):
        """
        Args:
            lidar: lidar point clouds in ego lidar pose 
            ego_params : epo params
        Returns:
            mask of fov lidar point clouds <<in ego coords>>
        """
        xyz = lidar[:,:3]
        
        xyz_hom = np.concatenate(
            [xyz, np.ones((xyz.shape[0], 1), dtype=np.float32)], axis=1)
        
        # print(ego_params['camera0'])
        intrinsic = np.array(ego_params['camera0']['intrinsic'])
        # img_shape = selected_cav_base['camera0'].shape
        # 如果selected_cav_base['camera0']是图像对象，使用size获取图像尺寸
        img_size = selected_cav_base['camera0'].size  # 返回 (宽度, 高度)
        img_shape = (img_size[1], img_size[0])  # 转换为 (高度, 宽度)
        ext_matrix = ego_params['c2e_transformation_matrix']
        ext_matrix = np.linalg.inv(ext_matrix)[:3,:4]
        img_pts = (intrinsic @ ext_matrix @ xyz_hom.T).T
        depth = img_pts[:, 2]
        uv = img_pts[:, :2] / depth[:, None]

        val_flag_1 = np.logical_and(uv[:, 0] >= 0, uv[:, 0] < img_shape[1])
        val_flag_2 = np.logical_and(uv[:, 1] >= 0, uv[:, 1] < img_shape[0])
        val_flag_merge = np.logical_and(val_flag_1, val_flag_2)
        pts_valid_flag = np.logical_and(depth > 0, val_flag_merge)
        return lidar[pts_valid_flag]

    @staticmethod
    def get_pairwise_transformation(base_data_dict, max_cav):
        """
        Get pair-wise transformation matrix across different agents.
        This is only used for v2vnet and disconet. Currently we set
        this as identity matrix as the pointcloud is projected to
        ego vehicle first.

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
        # default are identity matrix
        pairwise_t_matrix[:, :] = np.identity(4)

        return pairwise_t_matrix

    def get_item_single_car(self, selected_cav_base, ego_pose,ego_params, cav_id=None):
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
        lidar_np = mask_ego_points(lidar_np)
        # project the lidar to ego space
        if self.proj_first:
            lidar_np[:, :3] = box_utils.project_points_by_matrix_torch(lidar_np[:, :3], transformation_matrix)

        if self.fov:
            lidar_np = self.mask_ego_fov_flag(selected_cav_base, lidar_np, ego_params)

        lidar_np = mask_points_by_range(lidar_np,
                                        self.params['preprocess'][
                                            'cav_lidar_range'])
        # Check if filtered LiDAR points are not void
        void_lidar = True if lidar_np.shape[0] < 1 else False

        processed_lidar = self.pre_processor.preprocess(lidar_np)

        # add radar
        radar_np = selected_cav_base['radar_np']
        radar_np = shuffle_points(radar_np)
        # remove points that hit itself
        radar_np = mask_ego_points(radar_np)
        # project the radar to ego space
        radar_np[:, :3] = \
            box_utils.project_points_by_matrix_torch(radar_np[:, :3],
                                                     transformation_matrix)
        radar_np = mask_points_by_range(radar_np,
                                        self.params['preprocess'][
                                            'cav_lidar_range'])
        if self.fov:
            radar_np = self.mask_ego_fov_flag(selected_cav_base, radar_np, ego_params)
        # to keep same num of cav modality data
        if len(radar_np)==0 and len(lidar_np)!=0:
            radar_np = lidar_np[:3,:]
        if len(lidar_np)==0 and len(radar_np)!=0:
            lidar_np = radar_np[:3,:]
        processed_radar = self.pre_processor.preprocess(radar_np)
        processed_lidar = self.pre_processor.preprocess(lidar_np)
        import random
        while processed_lidar['voxel_features'].shape[0]!=0 and processed_radar['voxel_features'].shape[0]==0:
            radar_np = lidar_np[random.choices(range(0,len(lidar_np)),k=3),:]
            processed_radar = self.pre_processor.preprocess(radar_np)
        while processed_lidar['voxel_features'].shape[0]==0 and processed_radar['voxel_features'].shape[0]!=0:
            lidar_np = radar_np[random.choices(range(0,len(radar_np)),k=3),:]
            processed_lidar = self.pre_processor.preprocess(lidar_np)     

        # velocity
        velocity = selected_cav_base['params']['ego_speed']
        # normalize veloccity by average speed 30 km/h
        velocity = velocity / 30

        selected_cav_processed.update(
            {'object_bbx_center': object_bbx_center[object_bbx_mask == 1],
             'object_ids': object_ids,
             'projected_lidar': lidar_np,
             'projected_radar': radar_np,
            #  'processed_features': processed_lidar,
             'processed_lidar_features': processed_lidar,
             'processed_radar_features': processed_radar,
             'velocity': velocity})

        return selected_cav_processed, void_lidar

    @staticmethod
    def merge_features_to_dict(processed_feature_list):
        """
        Merge the preprocessed features from different cavs to the same
        dictionary.

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
        # Intermediate fusion is different the other two
        output_dict_list = []
        for j in range(len(batch[0])):
            output_dict = {'ego': {}}

            object_bbx_center = []
            object_bbx_mask = []
            object_ids = []
            processed_lidar_list = []
            processed_radar_list = []
            # used to record different scenario
            record_len = []
            label_dict_list = []

            # used for PriorEncoding
            velocity = []
            time_delay = []
            infra = []

            projected_lidar = []
            projected_radar = []
            # pairwise transformation matrix
            pairwise_t_matrix_list = []

            # used for correcting the spatial transformation between delayed timestamp
            # and current timestamp
            spatial_correction_matrix_list = []

            if self.visualize:
                origin_lidar = []

            for i in range(len(batch)):
                ego_dict = batch[i][j]['ego']
                object_bbx_center.append(ego_dict['object_bbx_center'])
                object_bbx_mask.append(ego_dict['object_bbx_mask'])
                object_ids.append(ego_dict['object_ids'])

                processed_lidar_list.append(ego_dict['processed_lidar'])
                processed_radar_list.append(ego_dict['processed_radar'])
                record_len.append(ego_dict['cav_num'])
                label_dict_list.append(ego_dict['label_dict'])

                velocity.append(ego_dict['velocity'])
                time_delay.append(ego_dict['time_delay'])
                infra.append(ego_dict['infra'])
                spatial_correction_matrix_list.append(
                    ego_dict['spatial_correction_matrix'])
                
                projected_lidar.append(
                    ego_dict['projected_lidar'])
                projected_radar.append(
                    ego_dict['projected_radar'])
                
                pairwise_t_matrix_list.append(ego_dict['pairwise_t_matrix'])

                if self.visualize:
                    origin_lidar.append(ego_dict['origin_lidar'])
            # convert to numpy, (B, max_num, 7)
            object_bbx_center = torch.from_numpy(np.array(object_bbx_center))
            object_bbx_mask = torch.from_numpy(np.array(object_bbx_mask))

            # example: {'voxel_features':[np.array([1,2,3]]),
            # np.array([3,5,6]), ...]}
            merged_feature_dict = self.merge_features_to_dict(processed_lidar_list)
            processed_lidar_torch_dict = \
                self.pre_processor.collate_batch(merged_feature_dict)

            merged_radar_feature_dict = self.merge_features_to_dict(processed_radar_list)        
            processed_radar_torch_dict = \
                self.pre_processor.collate_batch(merged_radar_feature_dict)            
            
            # [2, 3, 4, ..., M]
            record_len = torch.from_numpy(np.array(record_len, dtype=int))
            label_torch_dict = \
                self.post_processor.collate_batch(label_dict_list)

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
                                    'projected_lidar' : projected_lidar,
                                    'projected_radar' : projected_radar,
                                    'processed_lidar': processed_lidar_torch_dict,
                                    'processed_radar': processed_radar_torch_dict,
                                    'record_len': record_len,
                                    'label_dict': label_torch_dict,
                                    'object_ids': object_ids[0],
                                    'prior_encoding': prior_encoding,
                                    'spatial_correction_matrix': spatial_correction_matrix_list,
                                    'pairwise_t_matrix': pairwise_t_matrix})

            if self.visualize:
                origin_lidar = \
                    np.array(downsample_lidar_minimum(pcd_np_list=origin_lidar))
                origin_lidar = torch.from_numpy(origin_lidar)
                output_dict['ego'].update({'origin_lidar': origin_lidar})
            output_dict_list.append(output_dict)

        return output_dict_list
    
    def collate_batch_test(self, batch):
        assert len(batch) <= 1, "Batch size 1 is required during testing!"
        output_dict_list = self.collate_batch_train(batch)

        # check if anchor box in the batch
        for i in range(len(batch[0])):
            if batch[0][i]['ego']['anchor_box'] is not None:
                output_dict_list[i]['ego'].update({'anchor_box':
                    torch.from_numpy(np.array(
                        batch[0][i]['ego'][
                            'anchor_box']))})

            # save the transformation matrix (4, 4) to ego vehicle
            transformation_matrix_torch = \
                torch.from_numpy(np.identity(4)).float()
            output_dict_list[i]['ego'].update({'transformation_matrix':
                                        transformation_matrix_torch})

        return output_dict_list

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