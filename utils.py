import numpy as np
import os
import os.path as osp
import numpy as np
import copy
import numpy.ma as ma
from PIL import Image
# import numpy as np
from sklearn.metrics import confusion_matrix as CM

T01 = np.matrix([[1.0, 0,   0,   0],
                [0 ,   -1.0 ,  0,0],
                [0 ,   0 ,  -1.0  , 0],
                [0 ,  0   ,0  , 1.0]])
root_2 = '/home/cc/data_del/training/'
def cal_rigid_scene_flow(pc1, path):
    num = int (path.split('/')[-1])
    seq = int (path.split('/')[-2])
    R_rect, Tr_imu_velo, Tr_velo_cam, Tr_cam0_cam2, P2 = read_calib(seq)
    obj_3D_box = read_obj(seq)
    pose_imu_gt_0,pose_imu_gt_1,pose_imu_gt_2,pose_imu_gt_3 = read_abs_pose_in_imu(seq)

    pose_imu_gt_j_to_0 = read_one_pose_in_imu(num,pose_imu_gt_0,pose_imu_gt_1,pose_imu_gt_2,pose_imu_gt_3)
    pose_imu_gt_jt1_to_0 = read_one_pose_in_imu(num+1,pose_imu_gt_0,pose_imu_gt_1,pose_imu_gt_2,pose_imu_gt_3)

    pose_imu_gt_jt1_to_0_inv = np.linalg.inv(copy.deepcopy(pose_imu_gt_jt1_to_0))
    pose_imu_gt_jt_to_t1 = np.matmul(pose_imu_gt_jt1_to_0_inv, pose_imu_gt_j_to_0)

    T01_inv = np.linalg.inv(copy.deepcopy(T01))
    pose_imu_gt_jt_to_t1 = np.matmul(np.matmul(T01, pose_imu_gt_jt_to_t1), T01_inv)

    Tr_imu_velo_inv = np.linalg.inv(copy.deepcopy(Tr_imu_velo))
    pose_lidar_gt_jt_to_t1 = np.matmul(np.matmul(Tr_imu_velo, pose_imu_gt_jt_to_t1), Tr_imu_velo_inv)

    Tr_velo_cam2 = np.matmul(Tr_cam0_cam2, Tr_velo_cam)
    Tr_velo_cam2_inv = np.linalg.inv(copy.deepcopy(Tr_velo_cam2))
    pose_cam2_gt_jt_to_t1 = np.matmul(np.matmul(Tr_velo_cam2, pose_lidar_gt_jt_to_t1), Tr_velo_cam2_inv)

    # print (pc1.shape)
    # print (pose_cam2_gt_jt_to_t1.shape)
    one = np.expand_dims(np.ones_like(pc1[:,0]), 1)
    # one_2 = np.expand_dims(np.ones_like(pc2[:,0]), 1)

    pc1 = np.hstack((pc1[:, 0:3], one))

    rigid_scene_flow = np.swapaxes(np.matmul(pose_cam2_gt_jt_to_t1, np.swapaxes(pc1, 1, 0)), 0, 1) - pc1

    return rigid_scene_flow

def tran_velo_to_cam2( pc1, pc2, sf, path):
        # pc1_tran = 
    num = int (path.split('/')[-1])
    seq = int (path.split('/')[-2])
    R_rect, Tr_imu_velo, Tr_velo_cam, Tr_cam0_cam2, P2 = read_calib(seq)
    obj_3D_box = read_obj(seq)
    pose_imu_gt_0,pose_imu_gt_1,pose_imu_gt_2,pose_imu_gt_3 = read_abs_pose_in_imu(seq)

    pose_imu_gt_j_to_0 = read_one_pose_in_imu(num,pose_imu_gt_0,pose_imu_gt_1,pose_imu_gt_2,pose_imu_gt_3)
    pose_imu_gt_jt1_to_0 = read_one_pose_in_imu(num+1,pose_imu_gt_0,pose_imu_gt_1,pose_imu_gt_2,pose_imu_gt_3)

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



def read_obj_data(obj_i, obj_3D_box):
    obi_xyz = np.ones([1,3])
    obi_hwl = np.ones([1,3])
    obi_rotation_y = np.ones([1,1])
    obi_alpha = np.ones([1,1])
   
    obi_xyz = obj_3D_box[obj_i].split()[13:16]
    obi_hwl = obj_3D_box[obj_i].split()[10:13]
    obi_rotation_y = obj_3D_box[obj_i].split()[16]
    obi_alpha = obj_3D_box[obj_i].split()[5]

    return obi_xyz, obi_hwl, obi_rotation_y, obi_alpha

def read_pcl(i, j, Tr_velo_cam2):
    x = pointcloud[:, 0]  # x position of point
    y = pointcloud[:, 1]  # y position of point
    z = pointcloud[:, 2]  # z position of point

    # pose = np.ones([4,4])
    one = np.expand_dims(np.ones_like(z), 1)

    Nor_points = np.hstack((pointcloud[:, 0:3], one))
    Nor_points_cam2 = np.swapaxes(np.matmul(Tr_velo_cam2, np.swapaxes(Nor_points, 1, 0)), 0, 1)

    return Nor_points, Nor_points_cam2   

def read_calib( i):
    calib_path = root_2 + '/calib/' + str(i).zfill(4) + '.txt'
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

def read_obj(i):
    obj_3D_box_path = root_2 + '/label_02/' + str(i).zfill(4) + '.txt'
    f  = open(obj_3D_box_path,'r')
    obj_3D_box = f.readlines()

    return obj_3D_box

def read_abs_pose_in_imu(i):
    pose_imu_gt_path = root_2 + '/pose_gt/' + str(i).zfill(4) + '.txt'
    f  = open(pose_imu_gt_path,'r')
    lines = f.readlines()
    pose_imu_gt_0 = lines[0].split(',')
    pose_imu_gt_1 = lines[1].split(',')
    pose_imu_gt_2 = lines[2].split(',')
    pose_imu_gt_3 = lines[3].split(',')

    return pose_imu_gt_0,pose_imu_gt_1,pose_imu_gt_2,pose_imu_gt_3

def read_one_pose_in_imu(j,pose_imu_gt_0,pose_imu_gt_1,pose_imu_gt_2,pose_imu_gt_3):
    pose_imu_gt_j_to_0 = np.zeros([4,4])
    # pose_imu_gt_jt1_to_0 = np.zeros([4,4])
    pose_imu_gt_j_to_0[0] = pose_imu_gt_0[(j * 4):(j * 4 + 4)]
    pose_imu_gt_j_to_0[1] = pose_imu_gt_1[(j * 4):(j * 4 + 4)]
    pose_imu_gt_j_to_0[2] = pose_imu_gt_2[(j * 4):(j * 4 + 4)]
    pose_imu_gt_j_to_0[3] = pose_imu_gt_3[(j * 4):(j * 4 + 4)]
    # print (pose_imu_gt_j_to_0)
    return pose_imu_gt_j_to_0

def cal_mask_obj(Nor_points_cam2, obi_xyz, obi_hwl, obi_rotation_y):
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

def cal_obj_pose(obi_xyz, obi_rotation_y,obj_xyz, obj_rotation_y):
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

def cal_mask_type(obj_i, obj_3D_box):
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

def save_gt(i ,j ,scene_flow, scene_flow_gt):
    sceneflow_path = './velodyne/' + str(i).zfill(4) + '/' + str(j).zfill(6) + 'gt'
    scene_flow = np.hstack([scene_flow, scene_flow_gt])
    np.savez(sceneflow_path,scene_flow)

    return True

def view_point(points):
    real_points = copy.deepcopy(points)
    real_points[:,0] = +points[:,2]
    real_points[:,1] = -points[:,0]
    real_points[:,2] = -points[:,1]
    img = point_cloud_2_birdseye(real_points)

    return img

    # ==============================================================================
#                                                                   SCALE_TO_255
# ==============================================================================
def scale_to_255( a, min, max, dtype=np.uint8):
    """ Scales an array of values from specified min, max range to 0-255
        Optionally specify the data type of the output (default is uint8)
    """
    return (((a - min) / float(max - min)) * 255).astype(dtype)
 
# ==============================================================================
#                                                         POINT_CLOUD_2_BIRDSEYE
# ==============================================================================
def point_cloud_2_birdseye(
                           points,
                           res=0.02,
                           side_range=(-15., 15.),  # left-most to right-most
                           fwd_range = (-0., 35.), # back-most to forward-most
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
    # f_filt = np.logical_and((x_points > fwd_range[0]), (x_points < fwd_range[1]))
    # s_filt = np.logical_and((y_points > -side_range[1]), (y_points < -side_range[0]))
    # filter = np.logical_and(f_filt, s_filt)

    f_filt = ma.masked_inside(x_points, fwd_range[0], fwd_range[1])
    s_filt = ma.masked_inside(y_points, -side_range[1], -side_range[0])
    filter0 = f_filt.mask * s_filt.mask 
    # indices = np.argwhere(filter0).flatten()
    # KEEPERS
    x_points = x_points[filter0]
    y_points = y_points[filter0]
    z_points = z_points[filter0]
    # CONVERT TO PIXEL POSITION VALUES - Based on resolution
    x_img = (-y_points / res).astype(np.int32)  # x axis is -y in LIDAR
    y_img = (-x_points / res).astype(np.int32)  # y axis is -x in LIDAR
      # - Camera:   x: right,   y: down,  z: forward
  # - Velodyne: x: forward, y: left,  z: up
  # - GPS/IMU:  x: forward, y: left,  z: up
    # SHIFT PIXELS TO HAVE MINIMUM BE (0,0)
    # floor & ceil used to prevent anything being rounded to below 0 after shift
    # print (int(np.floor(side_range[0] / res)))
    x_img -= int(np.floor(side_range[0] / res))
    # x_img -= int(np.floor(side_range[0] / res))
    y_img += int(np.ceil(fwd_range[1] / res))
    # CLIP HEIGHT VALUES - to between min and max heights
    pixel_values = np.clip(a=z_points,
                           a_min=height_range[0],
                           a_max=height_range[1])
    # RESCALE THE HEIGHT VALUES - to be between the range 0-255
    pixel_values = scale_to_255(pixel_values,
                                min=height_range[0],
                                max=height_range[1])
    # INITIALIZE EMPTY ARRAY - of the dimensions we want
    x_max = 1 + int((side_range[1] - side_range[0]) / res)
    y_max = 1 + int((fwd_range[1] - fwd_range[0]) / res)
    im = np.zeros([y_max, x_max], dtype=np.uint8)
    # print (x_max,y_max,y_img.max(),x_img.max())

    # FILL PIXEL VALUES IN IMAGE ARRAY
    im[y_img, x_img] = pixel_values
    # xxx
 
    return im



class Evaluator(object):
    def __init__(self, class_names, labels=None):
        self.class_names = tuple(class_names)
        # self.class_names = class_names#tuple('static','move')
        self.num_classes = len(class_names)
        self.labels = np.arange(self.num_classes) if labels is None else np.array(labels)
        assert self.labels.shape[0] == self.num_classes
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes))

    def update(self, pred_label, gt_label):
        """Update per instance

        Args:
            pred_label (np.ndarray): (num_points)
            gt_label (np.ndarray): (num_points,)

        """
        # convert ignore_label to num_classes
        # refer to sklearn.metrics.confusion_matrix
        gt_label[gt_label == -100] = self.num_classes
        confusion_matrix = CM(gt_label.flatten(),
                              pred_label.flatten(),
                              labels=self.labels)
        self.confusion_matrix += confusion_matrix

    def batch_update(self, pred_labels, gt_labels):
        assert len(pred_labels) == len(gt_labels)
        for pred_label, gt_label in zip(pred_labels, gt_labels):
            self.update(pred_label, gt_label)

    @property
    def overall_acc(self):
        return np.sum(np.diag(self.confusion_matrix)) / np.sum(self.confusion_matrix)

    @property
    def overall_iou(self):
        class_iou = np.array(self.class_iou.copy())
        class_iou[np.isnan(class_iou)] = 0
        return np.mean(class_iou)

    @property
    def class_seg_acc(self):
        return [self.confusion_matrix[i, i] / np.sum(self.confusion_matrix[i])
                for i in range(self.num_classes)]

    @property
    def class_iou(self):
        iou_list = []
        for i in range(self.num_classes):
            tp = self.confusion_matrix[i, i]
            p = self.confusion_matrix[:, i].sum()
            g = self.confusion_matrix[i, :].sum()
            union = p + g - tp
            if union == 0:
                iou = float('nan')
            else:
                iou = tp / union
            iou_list.append(iou)
        return iou_list

    def print_table(self):
        from tabulate import tabulate
        header = ['Class', 'Accuracy', 'IOU', 'Total']
        seg_acc_per_class = self.class_seg_acc
        iou_per_class = self.class_iou
        table = []
        for ind, class_name in enumerate(self.class_names):
            table.append([class_name,
                          seg_acc_per_class[ind] * 100,
                          iou_per_class[ind] * 100,
                          int(self.confusion_matrix[ind].sum()),
                          ])
        # print (tabulate(table, headers=header, tablefmt='psql', floatfmt='.2f'))
        return tabulate(table, headers=header, tablefmt='psql', floatfmt='.2f')

    def save_table(self, filename):
        from tabulate import tabulate
        header = ('overall acc', 'overall iou') + self.class_names
        table = [[self.overall_acc, self.overall_iou] + self.class_iou]
        with open(filename, 'w') as f:
            # In order to unify format, remove all the alignments.
            f.write(tabulate(table, headers=header, tablefmt='tsv', floatfmt='.5f',
                             numalign=None, stralign=None))


##########################################r,t  transform
def cal_one_pose_matrix(pose):
    def _euler_to_mat(rot_euler):
        x = rot_euler[:, 0]
        y = rot_euler[:, 1]
        z = rot_euler[:, 2]

        zeros = torch.zeros_like(x).unsqueeze(1)  #(B,1)
        ones = torch.ones_like(x).unsqueeze(1)

        cosz = torch.cos(z).unsqueeze(1)
        sinz = torch.sin(z).unsqueeze(1)
        rotz_1 = torch.cat([cosz, -sinz, zeros], 1).unsqueeze(1) #(B,1,3)
        rotz_2 = torch.cat([sinz, cosz, zeros], 1).unsqueeze(1)
        rotz_3 = torch.cat([zeros, zeros, ones], 1).unsqueeze(1)
        zmat = torch.cat((rotz_1, rotz_2, rotz_3), 1)   #(b,3,3)

        cosy = torch.cos(y).unsqueeze(1)
        siny = torch.sin(y).unsqueeze(1)
        roty_1 = torch.cat([cosy, zeros, siny], 1).unsqueeze(1)
        roty_2 = torch.cat([zeros, ones, zeros], 1).unsqueeze(1)
        roty_3 = torch.cat([-siny, zeros, cosy], 1).unsqueeze(1)
        ymat = torch.cat((roty_1, roty_2, roty_3), 1)

        cosx = torch.cos(x).unsqueeze(1)
        sinx = torch.sin(x).unsqueeze(1)
        rotx_1 = torch.cat([ones, zeros, zeros], 1).unsqueeze(1)
        rotx_2 = torch.cat([zeros, cosx, -sinx], 1).unsqueeze(1)
        rotx_3 = torch.cat([zeros, sinx, cosx], 1).unsqueeze(1)
        xmat = torch.cat((rotx_1, rotx_2, rotx_3), 1)
        # rotMat = torch.matmul(zmat, torch.matmul(ymat, xmat))
        rotMat = torch.matmul(torch.matmul(xmat, ymat), zmat)

        return rotMat

    B,_ = pose.shape
    rotMat = _euler_to_mat(pose[:,0:3])
    # print("rotMat:  ", rotMat.shape)
    translation = (pose[:, 3:]).unsqueeze(2)
    filler = torch.tensor([[[0.0, 0.0, 0.0, 1.0]]])
    filler = filler.repeat(B,1,1).cuda()
    # print("filler:  ", filler.shape)
    pose_matrix = torch.cat((torch.cat((rotMat, translation), 2), filler), 1)
    return pose_matrix

def quat2mat(q):
    ''' Calculate rotation matrix corresponding to quaternion
    https://afni.nimh.nih.gov/pub/dist/src/pkundu/meica.libs/nibabel/quaternions.py
    Parameters
    ----------
    q : 4 element array-like

    Returns
    -------
    M : (3,3) array
      Rotation matrix corresponding to input quaternion *q*

    Notes
    -----
    Rotation matrix applies to column vectors, and is applied to the
    left of coordinate vectors.  The algorithm here allows non-unit
    quaternions.

    References
    ----------
    Algorithm from
    http://en.wikipedia.org/wiki/Rotation_matrix#Quaternion

    Examples
    --------
    >>> import numpy as np
    >>> M = quat2mat([1, 0, 0, 0]) # Identity quaternion
    >>> np.allclose(M, np.eye(3))
    True
    >>> M = quat2mat([0, 1, 0, 0]) # 180 degree rotn around axis 0
    >>> np.allclose(M, np.diag([1, -1, -1]))
    True
    '''
    w, x, y, z = q
    Nq = w * w + x * x + y * y + z * z
    if Nq < 1e-8:
        return np.eye(3)
    s = 2.0 / Nq
    X = x * s
    Y = y * s
    Z = z * s
    wX = w * X
    wY = w * Y
    wZ = w * Z
    xX = x * X
    xY = x * Y
    xZ = x * Z
    yY = y * Y
    yZ = y * Z
    zZ = z * Z
    return np.array([[1.0 - (yY + zZ), xY - wZ, xZ + wY],
                     [xY + wZ, 1.0 - (xX + zZ), yZ - wX],
                     [xZ - wY, yZ + wX, 1.0 - (xX + yY)]])


def mat2euler(M, cy_thresh=None, seq='zyx'):
    '''
    Taken From: http://afni.nimh.nih.gov/pub/dist/src/pkundu/meica.libs/nibabel/eulerangles.py
    Discover Euler angle vector from 3x3 matrix
    Uses the conventions above.
    Parameters
    ----------
    M : array-like, shape (3,3)
    cy_thresh : None or scalar, optional
     threshold below which to give up on straightforward arctan for
     estimating x rotation.  If None (default), estimate from
     precision of input.
    Returns
    -------
    z : scalar
    y : scalar
    x : scalar
     Rotations in radians around z, y, x axes, respectively
    Notes
    -----
    If there was no numerical error, the routine could be derived using
    Sympy expression for z then y then x rotation matrix, which is::
    [                       cos(y)*cos(z),                       -cos(y)*sin(z),         sin(y)],
    [cos(x)*sin(z) + cos(z)*sin(x)*sin(y), cos(x)*cos(z) - sin(x)*sin(y)*sin(z), -cos(y)*sin(x)],
    [sin(x)*sin(z) - cos(x)*cos(z)*sin(y), cos(z)*sin(x) + cos(x)*sin(y)*sin(z),  cos(x)*cos(y)]
    with the obvious derivations for z, y, and x
     z = atan2(-r12, r11)
     y = asin(r13)
     x = atan2(-r23, r33)
    for x,y,z order
    y = asin(-r31)
    x = atan2(r32, r33)
    z = atan2(r21, r11)
    Problems arise when cos(y) is close to zero, because both of::
     z = atan2(cos(y)*sin(z), cos(y)*cos(z))
     x = atan2(cos(y)*sin(x), cos(x)*cos(y))
    will be close to atan2(0, 0), and highly unstable.
    The ``cy`` fix for numerical instability below is from: *Graphics
    Gems IV*, Paul Heckbert (editor), Academic Press, 1994, ISBN:
    0123361559.  Specifically it comes from EulerAngles.c by Ken
    Shoemake, and deals with the case where cos(y) is close to zero:
    See: http://www.graphicsgems.org/
    The code appears to be licensed (from the website) as "can be used
    without restrictions".
    '''
    M = np.asarray(M)
    if cy_thresh is None:
        try:
            cy_thresh = np.finfo(M.dtype).eps * 4
        except ValueError:
            cy_thresh = _FLOAT_EPS_4
    r11, r12, r13, r21, r22, r23, r31, r32, r33 = M.flat
    # cy: sqrt((cos(y)*cos(z))**2 + (cos(x)*cos(y))**2)
    cy = math.sqrt(r33 * r33 + r23 * r23)
    if seq == 'zyx':
        if cy > cy_thresh:  # cos(y) not close to zero, standard form
            z = math.atan2(-r12, r11)  # atan2(cos(y)*sin(z), cos(y)*cos(z))
            y = math.atan2(r13, cy)  # atan2(sin(y), cy)
            x = math.atan2(-r23, r33)  # atan2(cos(y)*sin(x), cos(x)*cos(y))
        else:  # cos(y) (close to) zero, so x -> 0.0 (see above)
            # so r21 -> sin(z), r22 -> cos(z) and
            z = math.atan2(r21, r22)
            y = math.atan2(r13, cy)  # atan2(sin(y), cy)
            x = 0.0
    elif seq == 'xyz':
        if cy > cy_thresh:
            y = math.atan2(-r31, cy)
            x = math.atan2(r32, r33)
            z = math.atan2(r21, r11)
        else:
            z = 0.0
            if r31 < 0:
                y = np.pi / 2
                x = atan2(r12, r13)
            else:
                y = -np.pi / 2
    else:
        raise Exception('Sequence not recognized')
    return z, y, x


import functools


# def euler2mat(z=0, y=0, x=0, isRadian=True):
#     ''' Return matrix for rotations around z, y and x axes
#     Uses the z, then y, then x convention above
#     Parameters
#     ----------
#     z : scalar
#          Rotation angle in radians around z-axis (performed first)
#     y : scalar
#          Rotation angle in radians around y-axis
#     x : scalar
#          Rotation angle in radians around x-axis (performed last)
#     Returns
#     -------
#     M : array shape (3,3)
#          Rotation matrix giving same rotation as for given angles
#     Examples
#     --------
#     >>> zrot = 1.3 # radians
#     >>> yrot = -0.1
#     >>> xrot = 0.2
#     >>> M = euler2mat(zrot, yrot, xrot)
#     >>> M.shape == (3, 3)
#     True
#     The output rotation matrix is equal to the composition of the
#     individual rotations
#     >>> M1 = euler2mat(zrot)
#     >>> M2 = euler2mat(0, yrot)
#     >>> M3 = euler2mat(0, 0, xrot)
#     >>> composed_M = np.dot(M3, np.dot(M2, M1))
#     >>> np.allclose(M, composed_M)
#     True
#     You can specify rotations by named arguments
#     >>> np.all(M3 == euler2mat(x=xrot))
#     True
#     When applying M to a vector, the vector should column vector to the
#     right of M.  If the right hand side is a 2D array rather than a
#     vector, then each column of the 2D array represents a vector.
#     >>> vec = np.array([1, 0, 0]).reshape((3,1))
#     >>> v2 = np.dot(M, vec)
#     >>> vecs = np.array([[1, 0, 0],[0, 1, 0]]).T # giving 3x2 array
#     >>> vecs2 = np.dot(M, vecs)
#     Rotations are counter-clockwise.
#     >>> zred = np.dot(euler2mat(z=np.pi/2), np.eye(3))
#     >>> np.allclose(zred, [[0, -1, 0],[1, 0, 0], [0, 0, 1]])
#     True
#     >>> yred = np.dot(euler2mat(y=np.pi/2), np.eye(3))
#     >>> np.allclose(yred, [[0, 0, 1],[0, 1, 0], [-1, 0, 0]])
#     True
#     >>> xred = np.dot(euler2mat(x=np.pi/2), np.eye(3))
#     >>> np.allclose(xred, [[1, 0, 0],[0, 0, -1], [0, 1, 0]])
#     True
#     Notes
#     -----
#     The direction of rotation is given by the right-hand rule (orient
#     the thumb of the right hand along the axis around which the rotation
#     occurs, with the end of the thumb at the positive end of the axis;
#     curl your fingers; the direction your fingers curl is the direction
#     of rotation).  Therefore, the rotations are counterclockwise if
#     looking along the axis of rotation from positive to negative.
#     '''

#     if not isRadian:
#         z = ((np.pi) / 180.) * z
#         y = ((np.pi) / 180.) * y
#         x = ((np.pi) / 180.) * x
#     assert z >= (-np.pi) and z < np.pi, 'Inapprorpriate z: %f' % z
#     assert y >= (-np.pi) and y < np.pi, 'Inapprorpriate y: %f' % y
#     assert x >= (-np.pi) and x < np.pi, 'Inapprorpriate x: %f' % x

#     Ms = []
#     if z:
#         cosz = math.cos(z)
#         sinz = math.sin(z)
#         Ms.append(np.array([[cosz, -sinz, 0], [sinz, cosz, 0], [0, 0, 1]]))
#     if y:
#         cosy = math.cos(y)
#         siny = math.sin(y)
#         Ms.append(np.array([[cosy, 0, siny], [0, 1, 0], [-siny, 0, cosy]]))
#     if x:
#         cosx = math.cos(x)
#         sinx = math.sin(x)
#         Ms.append(np.array([[1, 0, 0], [0, cosx, -sinx], [0, sinx, cosx]]))
#     if Ms:
#         return functools.reduce(np.dot, Ms[::-1])
#     return np.eye(3)
def euler2mat(pose, isRadian=True):
    ''' Return matrix for rotations around z, y and x axes
    Uses the z, then y, then x convention above
    Parameters
    ----------
    z : scalar
         Rotation angle in radians around z-axis (performed first)
    y : scalar
         Rotation angle in radians around y-axis
    x : scalar
         Rotation angle in radians around x-axis (performed last)
    Returns
    -------
    M : array shape (3,3)
         Rotation matrix giving same rotation as for given angles
    Examples
    --------
    >>> zrot = 1.3 # radians
    >>> yrot = -0.1
    >>> xrot = 0.2
    >>> M = euler2mat(zrot, yrot, xrot)
    >>> M.shape == (3, 3)
    True
    The output rotation matrix is equal to the composition of the
    individual rotations
    >>> M1 = euler2mat(zrot)
    >>> M2 = euler2mat(0, yrot)
    >>> M3 = euler2mat(0, 0, xrot)
    >>> composed_M = np.dot(M3, np.dot(M2, M1))
    >>> np.allclose(M, composed_M)
    True
    You can specify rotations by named arguments
    >>> np.all(M3 == euler2mat(x=xrot))
    True
    When applying M to a vector, the vector should column vector to the
    right of M.  If the right hand side is a 2D array rather than a
    vector, then each column of the 2D array represents a vector.
    >>> vec = np.array([1, 0, 0]).reshape((3,1))
    >>> v2 = np.dot(M, vec)
    >>> vecs = np.array([[1, 0, 0],[0, 1, 0]]).T # giving 3x2 array
    >>> vecs2 = np.dot(M, vecs)
    Rotations are counter-clockwise.
    >>> zred = np.dot(euler2mat(z=np.pi/2), np.eye(3))
    >>> np.allclose(zred, [[0, -1, 0],[1, 0, 0], [0, 0, 1]])
    True
    >>> yred = np.dot(euler2mat(y=np.pi/2), np.eye(3))
    >>> np.allclose(yred, [[0, 0, 1],[0, 1, 0], [-1, 0, 0]])
    True
    >>> xred = np.dot(euler2mat(x=np.pi/2), np.eye(3))
    >>> np.allclose(xred, [[1, 0, 0],[0, 0, -1], [0, 1, 0]])
    True
    Notes
    -----
    The direction of rotation is given by the right-hand rule (orient
    the thumb of the right hand along the axis around which the rotation
    occurs, with the end of the thumb at the positive end of the axis;
    curl your fingers; the direction your fingers curl is the direction
    of rotation).  Therefore, the rotations are counterclockwise if
    looking along the axis of rotation from positive to negative.
    '''
    ###  n*3    n*3*3
    # if not isRadian:
    #     z = ((np.pi) / 180.) * pose[:,0]
    #     y = ((np.pi) / 180.) * pose[:,1]
    #     x = ((np.pi) / 180.) * pose[:,2]
    # assert (z >= (-np.pi)).all() and (z < np.pi).all(), 'Inapprorpriate z: %f' % z
    # assert (y >= (-np.pi)).all() and (y < np.pi).all(), 'Inapprorpriate y: %f' % y
    # assert (x >= (-np.pi)).all() and (x < np.pi).all(), 'Inapprorpriate x: %f' % x

    def _euler_to_mat(z, y, x):
        # x = rot_euler[:, 0]
        # y = rot_euler[:, 1]
        # z = rot_euler[:, 2]

        zeros = torch.zeros_like(x)  #(B,1)
        # print (z.shape, zeros.shape)
        ones = torch.ones_like(x)

        cosz = torch.cos(z)
        sinz = torch.sin(z)
        rotz_1 = torch.cat([cosz, -sinz, zeros], 1).unsqueeze(1) #(B,1,3)
        rotz_2 = torch.cat([sinz, cosz, zeros], 1).unsqueeze(1)
        rotz_3 = torch.cat([zeros, zeros, ones], 1).unsqueeze(1)
        zmat = torch.cat((rotz_1, rotz_2, rotz_3), 1)   #(b,3,3)
        # print (zmat.shape, zeros.shape)

        cosy = torch.cos(y)
        # print ("cos:",cosy[0])
        siny = torch.sin(y)
        roty_1 = torch.cat([cosy, zeros, siny], 1).unsqueeze(1)
        roty_2 = torch.cat([zeros, ones, zeros], 1).unsqueeze(1)
        roty_3 = torch.cat([-siny, zeros, cosy], 1).unsqueeze(1)
        ymat = torch.cat((roty_1, roty_2, roty_3), 1)

        cosx = torch.cos(x)
        sinx = torch.sin(x)
        rotx_1 = torch.cat([ones, zeros, zeros], 1).unsqueeze(1)
        rotx_2 = torch.cat([zeros, cosx, -sinx], 1).unsqueeze(1)
        rotx_3 = torch.cat([zeros, sinx, cosx], 1).unsqueeze(1)
        xmat = torch.cat((rotx_1, rotx_2, rotx_3), 1)
        # rotMat = torch.matmul(zmat, torch.matmul(ymat, xmat))
        rotMat = torch.matmul(torch.matmul(xmat, ymat), zmat)

        return rotMat
    z = np.zeros_like(pose[:,0])
    y = np.zeros_like(pose[:,1])
    x = np.zeros_like(pose[:,2])
    if not isRadian:
        z = ((np.pi) / 180.) * pose[:,0]
        y = ((np.pi) / 180.) * pose[:,1]
        x = ((np.pi) / 180.) * pose[:,2]
    assert (z >= (-np.pi)).all() and (z < np.pi).all(), 'Inapprorpriate z: %f' % z
    assert (y >= (-np.pi)).all() and (y < np.pi).all(), 'Inapprorpriate y: %f' % y
    assert (x >= (-np.pi)).all() and (x < np.pi).all(), 'Inapprorpriate x: %f' % x
    B,_ = pose.shape
    pose = torch.from_numpy(pose.astype(np.float64))
    rotMat = _euler_to_mat(pose[:,0:1], pose[:,1:2], pose[:,2:3])
    # return rotMat
    # print("rotMat:  ", rotMat.shape)
    # xxx
    translation = (pose[:, 3:]).unsqueeze(2)
    filler = torch.tensor([[[0.0, 0.0, 0.0, 1.0]]])
    filler = filler.repeat(B,1,1).double()
    # print("filler:  ", filler.shape)
    pose_matrix = torch.cat((torch.cat((rotMat, translation), 2), filler), 1)
    # print (rotMat.shape, pose_matrix.shape)
    return rotMat, pose_matrix


def euler2quat(z=0, y=0, x=0, isRadian=True):
    ''' Return quaternion corresponding to these Euler angles
    Uses the z, then y, then x convention above
    Parameters
    ----------
    z : scalar
         Rotation angle in radians around z-axis (performed first)
    y : scalar
         Rotation angle in radians around y-axis
    x : scalar
         Rotation angle in radians around x-axis (performed last)
    Returns
    -------
    quat : array shape (4,)
         Quaternion in w, x, y z (real, then vector) format
    Notes
    -----
    We can derive this formula in Sympy using:
    1. Formula giving quaternion corresponding to rotation of theta radians
         about arbitrary axis:
         http://mathworld.wolfram.com/EulerParameters.html
    2. Generated formulae from 1.) for quaternions corresponding to
         theta radians rotations about ``x, y, z`` axes
    3. Apply quaternion multiplication formula -
         http://en.wikipedia.org/wiki/Quaternions#Hamilton_product - to
         formulae from 2.) to give formula for combined rotations.
    '''

    if not isRadian:
        z = ((np.pi) / 180.) * z
        y = ((np.pi) / 180.) * y
        x = ((np.pi) / 180.) * x
    z = z / 2.0
    y = y / 2.0
    x = x / 2.0
    cz = math.cos(z)
    sz = math.sin(z)
    cy = math.cos(y)
    sy = math.sin(y)
    cx = math.cos(x)
    sx = math.sin(x)
    return np.array([
        cx * cy * cz - sx * sy * sz, cx * sy * sz + cy * cz * sx,
        cx * cz * sy - sx * cy * sz, cx * cy * sz + sx * cz * sy
    ])


def pose_vec_to_mat(vec):
    tx = vec[0]
    ty = vec[1]
    tz = vec[2]
    trans = np.array([tx, ty, tz]).reshape((3, 1))
    rot = euler2mat(vec[5], vec[4], vec[3])
    Tmat = np.concatenate((rot, trans), axis=1)
    hfiller = np.array([0, 0, 0, 1]).reshape((1, 4))
    Tmat = np.concatenate((Tmat, hfiller), axis=0)
    return Tmat

def rot2quat(R):
    rz, ry, rx = mat2euler(R)
    qw, qx, qy, qz = euler2quat(rz, ry, rx)
    return qw, qx, qy, qz
