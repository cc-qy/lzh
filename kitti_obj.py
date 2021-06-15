import sys, os
import os.path as osp
import numpy as np
import copy
import numpy.ma as ma
from PIL import Image

import torch.utils.data as data

__all__ = ['KITTI_obj']

T01 = np.matrix([[1.0, 0,   0,   0],
                [0 ,   -1.0 ,  0,0],
                [0 ,   0 ,  -1.0  , 0],
                [0 ,  0   ,0  , 1.0]])
class KITTI_obj(data.Dataset):
    """
    Args:
        train (bool): If True, creates dataset from training set, otherwise creates from test set.
        transform (callable):
        gen_func (callable):
        args:
    """

    def __init__(self,
                 train,
                 transform,
                 num_points,
                 data_root,
                 remove_ground = True):
        self.root = osp.join(data_root, 'training')
        self.root_2 = osp.join(data_root, 'training')
        #assert train is False
        self.train = train
        self.transform = transform
        self.num_points = num_points
        self.remove_ground = remove_ground

        self.samples = self.make_dataset()
        if len(self.samples) == 0:
            raise (RuntimeError("Found 0 files in subfolders of: " + self.root + "\n"))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        pc1_loaded, pc2_loaded ,sf_loaded, sf_2_loaded= self.pc_loader(self.samples[index])
        pc1_transformed, pc2_transformed, sf_transformed = self.transform([pc1_loaded, pc2_loaded,sf_loaded, sf_2_loaded])
        if pc1_transformed is None:
            print('path {} get pc1 is None'.format(self.samples[index]), flush=True)
            index = np.random.choice(range(self.__len__()))
            return self.__getitem__(index)

        pc1_norm = pc1_transformed
        pc2_norm = pc2_transformed
        return pc1_transformed, pc2_transformed, pc1_norm, pc2_norm, sf_transformed, self.samples[index]

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Number of points per point cloud: {}\n'.format(self.num_points)
        fmt_str += '    is removing ground: {}\n'.format(self.remove_ground)
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))

        return fmt_str

    def make_dataset(self):
        do_mapping = False#True
        root =  osp.realpath(osp.expanduser(self.root+'/data_loader'))
#/home/cc/data_del/training/velodyne/
        all_paths = sorted(os.walk(root))
        # print (all_paths[0])
        useful_paths = [item[0] for item in all_paths if len(item[1]) == 0]
        # print (root)
        # print ((useful_paths[0]))
        last_path = []
        for i in range(0, 21):
            if self.train == True:
                seq_path = root + '/' + str(i).zfill(4) + '.txt'
                f = open(seq_path, 'r')
                lines = f.readlines()
                for j in range(1,len(lines)-2):
                    if int(lines[j]) != 6:
                        for k in range(j*10-4, j*10+6):
                            if k!=(j*10):
                                path = str(int(lines[j])) + self.root_2 + '/velodyne/' + str(i).zfill(4) + '/' + str(k).zfill(6)
                                last_path.append(path)
                    if int(lines[j]) < 4:
                        for k in range(j*10-4, j*10+6):
                            if k!=(j*10):
                                path = str(5) + self.root_2 + '/velodyne/' + str(i).zfill(4) + '/' + str(k).zfill(6)
                                last_path.append(path)
                for j in range(0,1):
                    if int(lines[j]) != 6:
                        for k in range(j*10, j*10+6):
                            if k!=(j*10):
                                path = str(int(lines[j])) + self.root_2 + '/velodyne/' + str(i).zfill(4) + '/' + str(k).zfill(6)
                                last_path.append(path)
                    if int(lines[j]) < 4:
                        for k in range(j*10, j*10+6):
                            if k!=(j*10):
                                path = str(5) + self.root_2 + '/velodyne/' + str(i).zfill(4) + '/' + str(k).zfill(6)
                                last_path.append(path)
                for j in range(len(lines)-2,len(lines)-1):
                    if int(lines[j]) != 6:
                        for k in range(j*10-4, j*10):
                            if k!=(j*10):
                                path = str(int(lines[j])) + self.root_2 + '/velodyne/' + str(i).zfill(4) + '/' + str(k).zfill(6)
                                last_path.append(path)
                    if int(lines[j]) < 4:
                        for k in range(j*10-4, j*10):
                            if k!=(j*10):
                                path = str(5) + self.root_2 + '/velodyne/' + str(i).zfill(4) + '/' + str(k).zfill(6)
                                last_path.append(path)
            else:
                seq_path = root + '/' + str(i).zfill(4) + '.txt'
                f = open(seq_path, 'r')
                lines = f.readlines()
                num = 0
                for j in range(0,len(lines)-1):
                    if int(lines[j]) < 4:
                        path = str(int(lines[j])) + self.root_2 + '/velodyne/' + str(i).zfill(4) + '/' + str(j*10).zfill(6)
                        last_path.append(path)
                        # path = str(int(lines[j])) + self.root_2 + '/velodyne/' + str(i).zfill(4) + '/' + str(j*10+5).zfill(6)
                        # last_path.append(path)
                        num += 1
                print ("seq",i," num: ", num )
                    
        # # xxx
        # print (last_path[1000:1100])
        # xxx
        # try:
        #     assert (len(useful_paths) == 200)
        # except AssertionError:
        #     print('assert (len(useful_paths) == 200) failed!', len(useful_paths))

        # if do_mapping:
        #     mapping_path = osp.join(osp.dirname(__file__), 'KITTI_mapping.txt')
        #     print('mapping_path', mapping_path)

        #     with open(mapping_path) as fd:
        #         lines = fd.readlines()
        #         lines = [line.strip() for line in lines]
        #     useful_paths = [path for path in useful_paths if lines[int(osp.split(path)[-1])] != '']

        res_paths = last_path#useful_paths
        # print ("last_path", len(last_path))
        # print ("res_paths", len(res_paths))

        return res_paths

    def min_points(self, pc1_tran, pc2_tran, sf_tran, sf_tran2):
        min_point = min(pc1_tran.shape[0], pc2_tran.shape[0])
        pc1_tran = pc1_tran[0:min_point]
        pc2_tran = pc2_tran[0:min_point]
        sf_tran = sf_tran[0:min_point]
        sf_tran2 = sf_tran2[0:min_point]

        return pc1_tran, pc2_tran, sf_tran, sf_tran2

    def pc_loader(self, path):
        """
        Args:
            path:
        Returns:
            pc1: ndarray (N, 3) np.float32
            pc2: ndarray (N, 3) np.float32
        """
        flag = int (path.split('/')[0])
        path = path[1:]
        num = int (path.split('/')[-1])
        seq = int (path.split('/')[-2])

        # path1 = path[:-6] + str(num).zfill(6) + '.bin'
        # path2 = path[:-6] + str(num+1).zfill(6) + '.bin'
        path1 = path[:-6] + str(num).zfill(6) + 'pcl.npz'
        path2 = path[:-6] + str(num+1).zfill(6) + 'pcl.npz'
        path3 = path[:-6] + str(num).zfill(6) + 'gt.npz'

        # pc1 = np.fromfile(path1, dtype=np.float32, count=-1).reshape([-1,4])[:,0:3]
        # pc2 = np.fromfile(path2, dtype=np.float32, count=-1).reshape([-1,4])[:,0:3]
        # data = np.load(path3)
        pc1_tran = np.load(path1)['arr_0'].astype(np.float32)[:,0:3]
        pc2_tran = np.load(path2)['arr_0'].astype(np.float32)[:,0:3]
        #data = np.load(path3)
        sf_tran = np.load(path3)['arr_0'].astype(np.float32)
        # print (pc1_tran.shape)
        # xxx

        # pc1_tran, pc2_tran, sf_tran = self.tran_velo_to_cam2(pc1, pc2, sf, path)
        sf_tran2 = copy.deepcopy(sf_tran)
        pc1_tran, pc2_tran, sf_tran, sf_tran2 = self.min_points(pc1_tran, pc2_tran, sf_tran, sf_tran2)

        self.remove_ground = True
        if self.remove_ground:
            is_ground = np.logical_and(pc1_tran[:,1] > 1.4, pc1_tran[:,1] > 1.4)
            not_ground = np.logical_not(is_ground)
            pc1_tran = pc1_tran[not_ground]
            sf_tran = sf_tran[not_ground]
            sf_tran2 = sf_tran2[not_ground]

            is_ground = np.logical_and(pc2_tran[:,1] > 1.4, pc2_tran[:,1] > 1.4)
            not_ground = np.logical_not(is_ground)
            pc2_tran = pc2_tran[not_ground]

        pc1_tran, pc2_tran, sf_tran, sf_tran2 = self.min_points(pc1_tran, pc2_tran, sf_tran, sf_tran2)

        flag = int(flag)
        if self.train == True:
            # if flag < 4 :
            #     if np.random.rand()>-10.7:
            #         pc1_tran = pc1_tran
            #         pc2_tran = pc1_tran #+ sf_tran[:,0:3]
            #         sf_tran = sf_tran
            #         # print("zengqiang")
            if (flag == 5) or (flag == 4) or (flag == 7):
                pc1_tran = pc1_tran
                pc2_tran = pc1_tran + sf_tran[:,0:3]
                sf_tran = sf_tran
        # img_1 = self.view_point(pc1_tran)
        # img_2 = self.view_point(pc2_tran)
        # img_1_warp_2 = self.view_point(pc1_tran + sf_tran[:,0:3])
        # img_1_rigid_2 = self.view_point(pc2_tran + sf_tran[:,0:3])

        # h, w = img_1.shape
        # im_write = np.zeros_like(img_1)[:,0:10]+255

        # im_write = np.expand_dims(im_write,axis = 2)
        # im_write_zero = np.zeros_like(im_write)
        # im_write = np.dstack([im_write, im_write, im_write]) 

        # img_1 = np.expand_dims(img_1,axis = 2)
        # im_write_zero = np.zeros_like(img_1)
        # img_1 = np.dstack([im_write_zero, img_1, im_write_zero]) 

        # img_2 = np.expand_dims(img_2,axis = 2)
        # im_write_zero = np.zeros_like(img_2)
        # img_2 = np.dstack([img_2, im_write_zero, im_write_zero]) 

        # img_1_warp_2 = np.expand_dims(img_1_warp_2,axis = 2)
        # im_write_zero = np.zeros_like(img_1_warp_2)
        # img_1_warp_2 = np.dstack([im_write_zero, img_1_warp_2, im_write_zero]) 

        # img_1_rigid_2 = np.expand_dims(img_1_rigid_2,axis = 2)
        # im_write_zero = np.zeros_like(img_1_rigid_2)
        # img_1_rigid_2 = np.dstack([img_1_rigid_2, img_1_rigid_2, img_1_rigid_2]) 

        # image_real_all = np.hstack([im_write,img_1, im_write, img_1+img_2, im_write, img_2+img_1_warp_2, im_write, img_2+img_1_rigid_2,im_write])

        # image_real_all_show = Image.fromarray(255-image_real_all)
        # mm = np.random.randint(1000)
        # image_real_all_show.save('./view/' + str(mm).zfill(4) + '_' + str(mm).zfill(6) + 'view_all.jpg')
        # xxx

        return pc1_tran, pc2_tran, sf_tran, sf_tran2


    def tran_velo_to_cam2(self, pc1, pc2, sf, path):
        # pc1_tran = 
        num = int (path.split('/')[-1])
        seq = int (path.split('/')[-2])
        R_rect, Tr_imu_velo, Tr_velo_cam, Tr_cam0_cam2, P2 = self.read_calib(seq)
        obj_3D_box = self.read_obj(seq)
        pose_imu_gt_0,pose_imu_gt_1,pose_imu_gt_2,pose_imu_gt_3 = self.read_abs_pose_in_imu(seq)

        pose_imu_gt_j_to_0 = self.read_one_pose_in_imu(num,pose_imu_gt_0,pose_imu_gt_1,pose_imu_gt_2,pose_imu_gt_3)
        pose_imu_gt_jt1_to_0 = self.read_one_pose_in_imu(num+1,pose_imu_gt_0,pose_imu_gt_1,pose_imu_gt_2,pose_imu_gt_3)

        pose_imu_gt_jt1_to_0_inv = np.linalg.inv(copy.deepcopy(pose_imu_gt_jt1_to_0))
        pose_imu_gt_jt_to_t1 = np.matmul(pose_imu_gt_jt1_to_0_inv, pose_imu_gt_j_to_0)

        T01_inv = np.linalg.inv(copy.deepcopy(T01))
        pose_imu_gt_jt_to_t1 = np.matmul(np.matmul(T01, pose_imu_gt_jt_to_t1), T01_inv)

        Tr_imu_velo_inv = np.linalg.inv(copy.deepcopy(Tr_imu_velo))
        pose_lidar_gt_jt_to_t1 = np.matmul(np.matmul(Tr_imu_velo, pose_imu_gt_jt_to_t1), Tr_imu_velo_inv)

        Tr_velo_cam2 = np.matmul(Tr_cam0_cam2, Tr_velo_cam)
        Tr_velo_cam2_inv = np.linalg.inv(copy.deepcopy(Tr_velo_cam2))
        pose_cam2_gt_jt_to_t1 = np.matmul(np.matmul(Tr_velo_cam2, pose_lidar_gt_jt_to_t1), Tr_velo_cam2_inv)

        one = np.expand_dims(np.ones_like(pc1[:,0]), 1)
        one_2 = np.expand_dims(np.ones_like(pc2[:,0]), 1)

        pc1 = np.hstack((pc1[:, 0:3], one))
        pc2 = np.hstack((pc2[:, 0:3], one_2))

        pc1_cam2 = np.swapaxes(np.matmul(Tr_velo_cam2, np.swapaxes(pc1, 1, 0)), 0, 1)
        pc2_cam2 = np.swapaxes(np.matmul(Tr_velo_cam2, np.swapaxes(pc2, 1, 0)), 0, 1)

        pc1_cam2_image = np.swapaxes(np.matmul(P2, np.matmul(R_rect, np.matmul(Tr_velo_cam, np.swapaxes(pc1, 1, 0)))), 0, 1)
        pc1_cam2_image = pc1_cam2_image / np.repeat(np.expand_dims(pc1_cam2_image[:,2], axis=1),3,axis=1)

        mask_w = ma.masked_inside(pc1_cam2_image[:,0],-0,1300)
        mask_h = ma.masked_inside(pc1_cam2_image[:,1],-0,370)
        mask_depth = ma.masked_inside(pc1_cam2[:,2],-0,35)
        mask = mask_w.mask * mask_h.mask * mask_depth.mask

        pc2_cam2_image = np.swapaxes(np.matmul(P2, np.matmul(R_rect, np.matmul(Tr_velo_cam, np.swapaxes(pc2, 1, 0)))), 0, 1)
        pc2_cam2_image = pc2_cam2_image / np.repeat(np.expand_dims(pc2_cam2_image[:,2], axis=1),3,axis=1)

        mask_w = ma.masked_inside(pc2_cam2_image[:,0],-0,1300)
        mask_h = ma.masked_inside(pc2_cam2_image[:,1],-0,370)
        mask_depth = ma.masked_inside(pc2_cam2[:,2],-0,35)
        mask_2 = mask_w.mask * mask_h.mask * mask_depth.mask

        # return pc1_cam2[:,0:3].astype(np.float32), pc2_cam2[:,0:3].astype(np.float32), sf.astype(np.float32)
        return pc1_cam2[:,0:3][mask].astype(np.float32), pc2_cam2[:,0:3][mask_2].astype(np.float32), sf[mask].astype(np.float32)#pc1_tran, pc2_tran, sf_tran, sf_tran2



    def read_obj_data(self,obj_i, obj_3D_box):
        obi_xyz = np.ones([1,3])
        obi_hwl = np.ones([1,3])
        obi_rotation_y = np.ones([1,1])
        obi_alpha = np.ones([1,1])
       
        obi_xyz = obj_3D_box[obj_i].split()[13:16]
        obi_hwl = obj_3D_box[obj_i].split()[10:13]
        obi_rotation_y = obj_3D_box[obj_i].split()[16]
        obi_alpha = obj_3D_box[obj_i].split()[5]

        return obi_xyz, obi_hwl, obi_rotation_y, obi_alpha

    def read_pcl(self,i, j, Tr_velo_cam2):
        x = pointcloud[:, 0]  # x position of point
        y = pointcloud[:, 1]  # y position of point
        z = pointcloud[:, 2]  # z position of point

        # pose = np.ones([4,4])
        one = np.expand_dims(np.ones_like(z), 1)

        Nor_points = np.hstack((pointcloud[:, 0:3], one))
        Nor_points_cam2 = np.swapaxes(np.matmul(Tr_velo_cam2, np.swapaxes(Nor_points, 1, 0)), 0, 1)

        return Nor_points, Nor_points_cam2   

    def read_calib(self, i):
        calib_path = self.root_2 + '/calib/' + str(i).zfill(4) + '.txt'
        f  = open(calib_path,'r')
        lines = f.readlines()
        P0 = np.ones([4,4])
        P1 = np.ones([4,4])
        P2 = np.ones([3,4])
        P3 = np.ones([4,4])
        R_rect = np.ones([4,4])
        Tr_velo_cam = np.ones([4,4])
        Tr_imu_velo = np.ones([4,4])
        Tr_cam0_cam2 = np.ones([4,4])
        # R_rect = np.ones([3,3])
        lens = len(lines)
        Tr_velo_cam[0] = lines[5].split()[1:5] 
        Tr_velo_cam[1] = lines[5].split()[5:9] 
        Tr_velo_cam[2] = lines[5].split()[9:13] 
        Tr_velo_cam[3] = [0,0,0,1]
        Tr_imu_velo[0] = lines[6].split()[1:5] 
        Tr_imu_velo[1] = lines[6].split()[5:9] 
        Tr_imu_velo[2] = lines[6].split()[9:13] 
        Tr_imu_velo[3] = [0,0,0,1]
        R_rect[0,0:3] = lines[4].split()[1:4] 
        R_rect[1,0:3] = lines[4].split()[4:7] 
        R_rect[2,0:3] = lines[4].split()[7:10] 
        R_rect[3] = 0.0
        R_rect[:,3] = 0.0
        R_rect[3,3] = 1.0

        Tr_cam0_cam2[0:3,0:3] = R_rect[:3,:3]
        Tr_cam0_cam2[3] = [0,0,0,1]
        Tr_cam0_cam2[1,3] = lines[2].split()[4]
        Tr_cam0_cam2[2,3] = lines[2].split()[1]
        Tr_cam0_cam2[0,3] = -Tr_cam0_cam2[1,3]/Tr_cam0_cam2[2,3]
        Tr_cam0_cam2[1:3,3] = 0
        P2[0] = lines[2].split()[1:5] 
        P2[1] = lines[2].split()[5:9] 
        P2[2] = lines[2].split()[9:13] 

        return R_rect, Tr_imu_velo, Tr_velo_cam, Tr_cam0_cam2, P2

    def read_obj(self,i):
        obj_3D_box_path = self.root_2 + '/label_02/' + str(i).zfill(4) + '.txt'
        f  = open(obj_3D_box_path,'r')
        obj_3D_box = f.readlines()

        return obj_3D_box

    def read_abs_pose_in_imu(self,i):
        pose_imu_gt_path = self.root_2 + '/pose_gt/' + str(i).zfill(4) + '.txt'
        f  = open(pose_imu_gt_path,'r')
        lines = f.readlines()
        pose_imu_gt_0 = lines[0].split(',')
        pose_imu_gt_1 = lines[1].split(',')
        pose_imu_gt_2 = lines[2].split(',')
        pose_imu_gt_3 = lines[3].split(',')

        return pose_imu_gt_0,pose_imu_gt_1,pose_imu_gt_2,pose_imu_gt_3

    def read_one_pose_in_imu(self,j,pose_imu_gt_0,pose_imu_gt_1,pose_imu_gt_2,pose_imu_gt_3):
        pose_imu_gt_j_to_0 = np.zeros([4,4])
        # pose_imu_gt_jt1_to_0 = np.zeros([4,4])
        pose_imu_gt_j_to_0[0] = pose_imu_gt_0[(j * 4):(j * 4 + 4)]
        pose_imu_gt_j_to_0[1] = pose_imu_gt_1[(j * 4):(j * 4 + 4)]
        pose_imu_gt_j_to_0[2] = pose_imu_gt_2[(j * 4):(j * 4 + 4)]
        pose_imu_gt_j_to_0[3] = pose_imu_gt_3[(j * 4):(j * 4 + 4)]
        # print (pose_imu_gt_j_to_0)
        return pose_imu_gt_j_to_0

    def cal_mask_obj(self,Nor_points_cam2, obi_xyz, obi_hwl, obi_rotation_y):
        Nor_points_obj = copy.deepcopy(Nor_points_cam2) 
        Nor_points_obj[:,0] = Nor_points_obj[:,0] - float(obi_xyz[0])
        Nor_points_obj[:,1] = Nor_points_obj[:,1] - float(obi_xyz[1])
        Nor_points_obj[:,2] = Nor_points_obj[:,2] - float(obi_xyz[2])
        mask_y = ma.masked_inside(Nor_points_obj[:,1],-float(obi_hwl[0]),0.0)
        # print (mask_y.mask)
        obi_rotation_y_final = 0.0
        if float(obi_rotation_y) < 0.0:
            obi_rotation_y_final = 2 * 3.1415926 + float(obi_rotation_y)
        if float(obi_rotation_y) > 0.0:
            obi_rotation_y_final = float(obi_rotation_y)
        if float(obi_rotation_y) ==0.0:
            print ("error obi_rotation_y",i," ",j)
        sin_rotation_y = np.sin(obi_rotation_y_final)
        cos_rotation_y = np.cos(obi_rotation_y_final)
        z_new = Nor_points_obj[:,0] * cos_rotation_y - Nor_points_obj[:,2] * sin_rotation_y
        x_new = Nor_points_obj[:,0] * sin_rotation_y + Nor_points_obj[:,2] * cos_rotation_y
        # print (obi_rotation_y_final,cos_rotation_y)
        mask_x = ma.masked_inside(x_new,-0.5*float(obi_hwl[1])-0.15,0.5*float(obi_hwl[1])+0.15)
        mask_z = ma.masked_inside(z_new,-0.5*float(obi_hwl[2])-0.15,0.5*float(obi_hwl[2])+0.15)
        mask_obj = mask_y.mask * mask_x.mask * mask_z.mask

        return mask_obj

    def cal_obj_pose(self,obi_xyz, obi_rotation_y,obj_xyz, obj_rotation_y):
        obj_t_t_to_t1 = [float(obj_xyz[0]) - float(obi_xyz[0]), float(obj_xyz[1]) - float(obi_xyz[1]), float(obj_xyz[2]) - float(obi_xyz[2])]
        # print ("obj_t_t_to_t1: ", obj_t_t_to_t1)
        obj_R_t_to_t1 = 0.0
        if (float(obi_rotation_y)<0.0)&(float(obj_rotation_y)<0.0):
            obj_R_t_to_t1 = -float(obj_rotation_y) - (-float(obi_rotation_y))
        if (float(obi_rotation_y)<0.0)&(float(obj_rotation_y)>0.0):
            obj_R_t_to_t1 = (3.1415926+float(obi_rotation_y)) + (float(obj_rotation_y))
        if (float(obi_rotation_y)>0.0)&(float(obj_rotation_y)>0.0):
            obj_R_t_to_t1 = float(obj_rotation_y) - float(obi_rotation_y)
        if (float(obi_rotation_y)>0.0)&(float(obj_rotation_y)<0.0):
            obj_R_t_to_t1 = 3.1415926-float(obi_rotation_y) - (float(obj_rotation_y))
        obj_T_t_to_t1_matrix =  np.matrix([ [np.cos(obj_R_t_to_t1), 0,   np.sin(obj_R_t_to_t1),   float(obj_xyz[0]) - float(obi_xyz[0])],
                                            [0 ,   1,  0,float(obj_xyz[1]) - float(obi_xyz[1])],
                                            [-np.sin(obj_R_t_to_t1) ,   0 ,  np.cos(obj_R_t_to_t1)  , float(obj_xyz[2]) - float(obi_xyz[2])],
                                            [0 ,  0   ,0  , 1.0]])
        return obj_R_t_to_t1, obj_T_t_to_t1_matrix

    def cal_mask_type(self,obj_i, obj_3D_box):
        mask_type = -1
        if (obj_3D_box[obj_i].split()[2]) == 'car':
            mask_type = 0
        if (obj_3D_box[obj_i].split()[2]) == 'Van':
            mask_type = 1
        if (obj_3D_box[obj_i].split()[2]) == 'Truck':
            mask_type = 2
        if (obj_3D_box[obj_i].split()[2]) == 'Pedestrian':
            mask_type = 3
        if (obj_3D_box[obj_i].split()[2]) == 'Person_sitting':
            mask_type = 4
        if (obj_3D_box[obj_i].split()[2]) == 'Cyclist':
            mask_type = 5
        if (obj_3D_box[obj_i].split()[2]) == 'Tram':
            mask_type = 6
        if (obj_3D_box[obj_i].split()[2]) == 'Misc':
            mask_type = 7
        if (obj_3D_box[obj_i].split()[2]) == 'DontCare':
            mask_type = 8

        return mask_type

    def save_gt(self,i ,j ,scene_flow, scene_flow_gt):
        sceneflow_path = './velodyne/' + str(i).zfill(4) + '/' + str(j).zfill(6) + 'gt'
        scene_flow = np.hstack([scene_flow, scene_flow_gt])
        np.savez(sceneflow_path,scene_flow)

        return True

    def view_point(self,points):
        real_points = copy.deepcopy(points)
        real_points[:,0] = +points[:,2]
        real_points[:,1] = -points[:,0]
        real_points[:,2] = -points[:,1]
        img = self.point_cloud_2_birdseye(real_points)

        return img

        # ==============================================================================
    #                                                                   SCALE_TO_255
    # ==============================================================================
    def scale_to_255(self, a, min, max, dtype=np.uint8):
        """ Scales an array of values from specified min, max range to 0-255
            Optionally specify the data type of the output (default is uint8)
        """
        return (((a - min) / float(max - min)) * 255).astype(dtype)
     
    # ==============================================================================
    #                                                         POINT_CLOUD_2_BIRDSEYE
    # ==============================================================================
    def point_cloud_2_birdseye(self,
                               points,
                               res=0.02,
                               side_range=(-20., 20.),  # left-most to right-most
                               fwd_range = (-10., 35.), # back-most to forward-most
                               height_range=(-3., 5.),  # bottom-most to upper-most
                               ):
        """ Creates an 2D birds eye view representation of the point cloud data.
     
        Args:
            points:     (numpy array)
                        N rows of points data
                        Each point should be specified by at least 3 elements x,y,z
            res:        (float)
                        Desired resolution in metres to use. Each output pixel will
                        represent an square region res x res in size.
            side_range: (tuple of two floats)
                        (-left, right) in metres
                        left and right limits of rectangle to look at.
            fwd_range:  (tuple of two floats)
                        (-behind, front) in metres
                        back and front limits of rectangle to look at.
            height_range: (tuple of two floats)
                        (min, max) heights (in metres) relative to the origin.
                        All height values will be clipped to this min and max value,
                        such that anything below min will be truncated to min, and
                        the same for values above max.
        Returns:
            2D numpy array representing an image of the birds eye view.
        """
        # EXTRACT THE POINTS FOR EACH AXIS
        x_points = points[:, 0]
        y_points = points[:, 1]
        z_points = points[:, 2]
        # FILTER - To return only indices of points within desired cube
        # Three filters for: Front-to-back, side-to-side, and height ranges
        # Note left side is positive y axis in LIDAR coordinates
        f_filt = np.logical_and((x_points > fwd_range[0]), (x_points < fwd_range[1]))
        s_filt = np.logical_and((y_points > -side_range[1]), (y_points < -side_range[0]))
        filter = np.logical_and(f_filt, s_filt)
        indices = np.argwhere(filter).flatten()
        # KEEPERS
        x_points = x_points[indices]
        y_points = y_points[indices]
        z_points = z_points[indices]
        # CONVERT TO PIXEL POSITION VALUES - Based on resolution
        x_img = (-y_points / res).astype(np.int32)  # x axis is -y in LIDAR
        y_img = (-x_points / res).astype(np.int32)  # y axis is -x in LIDAR
          # - Camera:   x: right,   y: down,  z: forward
      # - Velodyne: x: forward, y: left,  z: up
      # - GPS/IMU:  x: forward, y: left,  z: up
        # SHIFT PIXELS TO HAVE MINIMUM BE (0,0)
        # floor & ceil used to prevent anything being rounded to below 0 after shift
        x_img -= int(np.floor(side_range[0] / res))
        y_img += int(np.ceil(fwd_range[1] / res))
        # CLIP HEIGHT VALUES - to between min and max heights
        pixel_values = np.clip(a=z_points,
                               a_min=height_range[0],
                               a_max=height_range[1])
        # RESCALE THE HEIGHT VALUES - to be between the range 0-255
        pixel_values = self.scale_to_255(pixel_values,
                                    min=height_range[0],
                                    max=height_range[1])
        # INITIALIZE EMPTY ARRAY - of the dimensions we want
        x_max = 1 + int((side_range[1] - side_range[0]) / res)
        y_max = 1 + int((fwd_range[1] - fwd_range[0]) / res)
        im = np.zeros([y_max, x_max], dtype=np.uint8)
        # FILL PIXEL VALUES IN IMAGE ARRAY
        im[y_img, x_img] = pixel_values
     
        return im
    # def pc_loader(self, path):
    #     """
    #     Args:
    #         path:
    #     Returns:
    #         pc1: ndarray (N, 3) np.float32
    #         pc2: ndarray (N, 3) np.float32
    #     """
    #     pc1 = np.load(osp.join(path, 'pc1.npy'))  #.astype(np.float32)
    #     pc2 = np.load(osp.join(path, 'pc2.npy'))  #.astype(np.float32)

    #     if self.remove_ground:
    #         is_ground = np.logical_and(pc1[:,1] < -1.4, pc2[:,1] < -1.4)
    #         not_ground = np.logical_not(is_ground)

    #         pc1 = pc1[not_ground]
    #         pc2 = pc2[not_ground]

    #     return pc1, pc2
