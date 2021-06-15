"""
Borrowed from HPLFlowNet
Date: May 2020

@inproceedings{HPLFlowNet,
  title={HPLFlowNet: Hierarchical Permutohedral Lattice FlowNet for
Scene Flow Estimation on Large-scale Point Clouds},
  author={Gu, Xiuye and Wang, Yijie and Wu, Chongruo and Lee, Yong Jae and Wang, Panqu},
  booktitle={Computer Vision and Pattern Recognition (CVPR), 2019 IEEE International Conference on},
  year={2019}
}
"""

import os, sys
import os.path as osp
from collections import defaultdict
import numbers
import math
import numpy as np
import traceback
import time

import torch

import numba
from numba import njit

from . import functional as F

def euler2mat(pose, isRadian=True):
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
    M = torch.from_numpy(M)
    # r11, r12, r13, r21, r22, r23, r31, r32, r33 = M.flat
    r11 = M[:,0,0]
    r12 = M[:,0,1]
    r13 = M[:,0,2]
    r21 = M[:,1,0]
    r22 = M[:,1,1]
    r23 = M[:,1,2]
    r31 = M[:,2,0]
    r32 = M[:,2,1]
    r33 = M[:,2,2]

    # cy: sqrt((cos(y)*cos(z))**2 + (cos(x)*cos(y))**2)

    cy = torch.sqrt((r33 * r33 + r23 * r23))
    # print (cy.shape)
    # xxx
    if seq == 'zyx':
        # if cy > cy_thresh: 
        if 1:  # cos(y) not close to zero, standard form
            z = torch.atan2(-r12, r11)  # atan2(cos(y)*sin(z), cos(y)*cos(z))
            y = torch.atan2(r13, cy)  # atan2(sin(y), cy)
            x = torch.atan2(-r23, r33)  # atan2(cos(y)*sin(x), cos(x)*cos(y))
        else:  # cos(y) (close to) zero, so x -> 0.0 (see above)
            # so r21 -> sin(z), r22 -> cos(z) and
            z = torch.atan2(r21, r22)
            y = torch.atan2(r13, cy)  # atan2(sin(y), cy)
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
# ---------- BASIC operations ----------
class Compose(object):
    """Composes several transforms together.

    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.

    Example:
        >>> transforms.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, data):
        for t in self.transforms:
            data = t(data)
        return data

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string


class ToTensor(object):
    def __call__(self, pic):
        if not isinstance(pic, np.ndarray):
            return pic
        else:
            return F.to_tensor(pic)

    def __repr__(self):
        return self.__class__.__name__ + '()'


# ---------- Build permutalhedral lattice ----------
@njit(numba.int64(numba.int64[:], numba.int64, numba.int64[:], numba.int64[:], ))
def key2int(key, dim, key_maxs, key_mins):
    """
    :param key: np.array
    :param dim: int
    :param key_maxs: np.array
    :param key_mins: np.array
    :return:
    """
    tmp_key = key - key_mins
    scales = key_maxs - key_mins + 1
    res = 0
    for idx in range(dim):
        res += tmp_key[idx]
        res *= scales[idx + 1]
    res += tmp_key[dim]
    return res


@njit(numba.int64[:](numba.int64, numba.int64, numba.int64[:], numba.int64[:], ))
def int2key(int_key, dim, key_maxs, key_mins):
    key = np.empty((dim + 1,), dtype=np.int64)
    scales = key_maxs - key_mins + 1
    for idx in range(dim, 0, -1):
        key[idx] = int_key % scales[idx]
        int_key -= key[idx]
        int_key //= scales[idx]
    key[0] = int_key

    key += key_mins
    return key


@njit
def advance_in_dimension(d1, increment, adv_dim, key):
    key_cp = key.copy()

    key_cp -= increment
    key_cp[adv_dim] += increment * d1
    return key_cp


class Traverse:
    def __init__(self, neighborhood_size, d):
        self.neighborhood_size = neighborhood_size
        self.d = d

    def go(self, start_key, hash_table_list):
        walking_keys = np.empty((self.d + 1, self.d + 1), dtype=np.long)
        self.walk_cuboid(start_key, 0, False, walking_keys, hash_table_list)

    def walk_cuboid(self, start_key, d, has_zero, walking_keys, hash_table_list):
        if d <= self.d:
            walking_keys[d] = start_key.copy()

            range_end = self.neighborhood_size + 1 if (has_zero or (d < self.d)) else 1
            for i in range(range_end):
                self.walk_cuboid(walking_keys[d], d + 1, has_zero or (i == 0), walking_keys, hash_table_list)
                walking_keys[d] = advance_in_dimension(self.d + 1, 1, d, walking_keys[d])
        else:
            hash_table_list.append(start_key.copy())


# ---------- MAIN operations ----------
class ProcessData(object):
    def __init__(self, data_process_args, num_points, allow_less_points):
        self.DEPTH_THRESHOLD = data_process_args['DEPTH_THRESHOLD']
        self.no_corr = data_process_args['NO_CORR']
        self.num_points = num_points
        self.allow_less_points = allow_less_points

    def __call__(self, data):
        pc1, pc2, sf, sf_2 = data
        if pc1 is None:
            return None, None, None,

  
        pc2 = pc1 + sf[:,0:3]

        if self.DEPTH_THRESHOLD > 0:
            near_mask = np.logical_and(pc1[:, 2] < 100000*self.DEPTH_THRESHOLD, pc2[:, 2] < 10000*self.DEPTH_THRESHOLD)
        else:
            near_mask = np.ones(pc1.shape[0], dtype=np.bool)
        indices = np.where(near_mask)[0]
        if len(indices) == 0:
            print('indices = np.where(mask)[0], len(indices) == 0')
            return None, None, None

        if self.num_points > 0:
            try:
                sampled_indices1 = np.random.choice(indices, size=self.num_points, replace=False, p=None)
                if self.no_corr:
                    sampled_indices2 = np.random.choice(indices, size=self.num_points, replace=False, p=None)
                else:
                    sampled_indices2 = sampled_indices1
            except ValueError:
                '''
                if not self.allow_less_points:
                    print('Cannot sample {} points'.format(self.num_points))
                    return None, None, None
                else:
                    sampled_indices1 = indices
                    sampled_indices2 = indices
                '''
                if not self.allow_less_points:
                    #replicate some points
                    sampled_indices1 = np.random.choice(indices, size=self.num_points, replace=True, p=None)
                    if self.no_corr:
                        sampled_indices2 = np.random.choice(indices, size=self.num_points, replace=True, p=None)
                    else:
                        sampled_indices2 = sampled_indices1
                else:
                    sampled_indices1 = indices
                    sampled_indices2 = indices
        else:
            sampled_indices1 = indices
            sampled_indices2 = indices

        pc1 = pc1[sampled_indices1]
        sf = sf[sampled_indices1]
        pc2 = pc2[sampled_indices2]

        return pc1, pc2, sf#[:,0:3]

    def __repr__(self):
        format_string = self.__class__.__name__ + '\n(data_process_args: \n'
        format_string += '\tDEPTH_THRESHOLD: {}\n'.format(self.DEPTH_THRESHOLD)
        format_string += '\tNO_CORR: {}\n'.format(self.no_corr)
        format_string += '\tallow_less_points: {}\n'.format(self.allow_less_points)
        format_string += '\tnum_points: {}\n'.format(self.num_points)
        format_string += ')'
        return format_string


class Augmentation(object):
    def __init__(self, aug_together_args, aug_pc2_args, data_process_args, num_points, allow_less_points=False):
        self.together_args = aug_together_args
        self.pc2_args = aug_pc2_args
        self.DEPTH_THRESHOLD = data_process_args['DEPTH_THRESHOLD']
        self.no_corr = data_process_args['NO_CORR']
        self.num_points = num_points
        self.allow_less_points = allow_less_points

    def __call__(self, data):
        pc1, pc2, sf, sf_2= data
        if pc1 is None:
            return None, None, None
        pc1_init = pc1
        pc2_init = pc2
        sf_init = sf
        # pose_eur = sf[:,7:13]#[mask.A[:,0]]
        # ori_mat_multi, pose_mat_multi = euler2mat((pose_eur))
        # one = np.expand_dims(np.ones_like(pc1[:,0]), 1)
        # Nor_points = np.hstack((pc1[:, 0:3], one))
        # Trans_Nor_points_copy = torch.matmul((pose_mat_multi), torch.from_numpy(np.expand_dims(np.float64(Nor_points), axis = -1))).numpy()
        # scene_flow_2 = Trans_Nor_points_copy.squeeze(-1)-Nor_points
        # error = sf[:,0:3] - scene_flow_2[:,0:3]
        # print ("sum:", np.sum(error))
        # xxx

        # together, order: scale, rotation, shift, jitter
        # scale
        scale = np.diag(np.random.uniform(self.together_args['scale_low'],
                                          self.together_args['scale_high'],
                                          3).astype(np.float32))
        # rotation
        angle = np.random.uniform(-self.together_args['degree_range'],
                                  self.together_args['degree_range'])
        cosval = np.cos(angle)
        sinval = np.sin(angle)
        rot_matrix = np.array([[cosval, 0, sinval],
                               [0, 1, 0],
                               [-sinval, 0, cosval]], dtype=np.float64)
        matrix = scale.dot(rot_matrix.T)
        # print ("matrix:",matrix)
        # matrix = rot_matrix.T.dot(scale)
        # print (matrix)
        # # print (scale)
        # xxx

        # shift
        shifts = np.random.uniform(-self.together_args['shift_range'],
                                   self.together_args['shift_range'],
                                   (1, 3)).astype(np.float32)

        # jitter
        jitter = np.clip(self.together_args['jitter_sigma'] * np.random.randn(pc1.shape[0]*2, 3),
                         -self.together_args['jitter_clip'],
                         self.together_args['jitter_clip']).astype(np.float32)
        bias = shifts + jitter

        pc1_ori = pc1[:,0:3]
        pc2_pri = pc1_ori + sf[:,:3]
        pc1_tran = pc1_ori.dot(matrix) + bias[0:pc1_ori.shape[0]]
        pc2_tran = pc2_pri.dot(matrix) + bias[0:pc2_pri.shape[0]]
        sf[:,:3] = pc2_tran - pc1_tran

        pc1[:, :3] = pc1[:, :3].dot(matrix) + bias[0:pc1.shape[0]]
        pc2[:, :3] = pc2[:, :3].dot(matrix) + bias[0:pc2.shape[0]]

        pose_eur = sf[:,7:13]#[mask.A[:,0]]
        ori_mat_multi, pose_mat_multi = euler2mat((pose_eur))

        # B,_ = pose.shape
        # pose = torch.from_numpy(pose.astype(np.float64))
        # rotMat = _euler_to_mat(pose[:,0:1], pose[:,1:2], pose[:,2:3])
        # return rotMat
        # print("rotMat:  ", rotMat.shape)
        # xxx
        rotMat = torch.from_numpy(np.expand_dims(rot_matrix, axis = 0).repeat(sf.shape[0], axis=0)).double()
        translation = torch.from_numpy(np.expand_dims(bias[0:pc1_ori.shape[0]], axis = -1)).double()
        filler = torch.tensor([[[0.0, 0.0, 0.0, 1.0]]])
        filler = filler.repeat(sf.shape[0],1,1).double()
        pose_matrix = torch.cat((torch.cat((rotMat, translation), 2), filler), 1)

        rotMat_inv = torch.from_numpy(np.expand_dims(rot_matrix, axis = 0).repeat(sf.shape[0], axis=0)).double().permute(0,2,1)
        translation_inv = torch.matmul(-rotMat_inv,translation)  
        pose_matrix_inv = torch.cat((torch.cat((rotMat_inv, translation_inv), 2), filler), 1)


        rotMat_s = torch.from_numpy(np.expand_dims(scale, axis = 0).repeat(sf.shape[0], axis=0)).double()
        translation_s = torch.from_numpy(np.expand_dims(bias[0:pc1_ori.shape[0]]*0.0,axis = -1)).double()
        # filler = torch.tensor([[[0.0, 0.0, 0.0, 1.0]]])
        # filler = filler.repeat(sf.shape[0],1,1).double()
        pose_matrix_s = torch.cat((torch.cat((rotMat_s, translation_s), 2), filler), 1)
        scale[0,0] = 1/scale[0,0]
        scale[1,1] = 1/scale[1,1]
        scale[2,2] = 1/scale[2,2]
        rotMat_s_inv = torch.from_numpy(np.expand_dims(scale, axis = 0).repeat(sf.shape[0], axis=0)).double()
        translation_s_inv = translation_s#torch.matmul(-rotMat_inv,translation)  
        pose_matrix_s_inv = torch.cat((torch.cat((rotMat_s_inv, translation_s_inv), 2), filler), 1)

        # I = torch.matmul(pose_matrix_s,pose_matrix_s_inv)
        # print (I[0])
        # xxx
        T_new = torch.matmul(torch.matmul(pose_matrix_s, pose_mat_multi), pose_matrix_s_inv)
        T_new = torch.matmul(torch.matmul(pose_matrix, T_new), pose_matrix_inv)
        one = np.expand_dims(np.ones_like(pc1_init[:,0]), 1)
        # Nor_points = np.hstack((pc1_init[:, 0:3], one))

        # pc1_init_tran = torch.matmul(T_new, torch.from_numpy(np.expand_dims(np.float64(Nor_points), axis = -1)))
        # sf_init = pc1_init_tran[:,0:3,0] - pc1_init[:,0:3]
        # error = sf[:,0:3] - sf_init.numpy()
        # print (pc1_init_tran.shape,sf_init.shape)
        # print ("sum:", np.sum(error))

        # xxx

        # pc2, order: rotation, shift, jitter
        # rotation
        angle2 = np.random.uniform(-self.pc2_args['degree_range'],
                                   self.pc2_args['degree_range'])
        cosval2 = np.cos(angle2)
        sinval2 = np.sin(angle2)
        matrix2 = np.array([[cosval2, 0, sinval2],
                            [0, 1, 0],
                            [-sinval2, 0, cosval2]], dtype=pc1.dtype)
        # shift
        shifts2 = np.random.uniform(-self.pc2_args['shift_range'],
                                    self.pc2_args['shift_range'],
                                    (pc1_ori.shape[0], 3)).astype(np.float32)

        pc1_ori = pc1[:,:3]
        pc2_ori = pc1[:,:3] + sf[:,:3]
        if np.random.rand()>10.7:
            pc2_tran = pc2_ori.dot(matrix2.T)
            sf[:,:3] = pc2_tran - pc1_ori
            pc2[:, :3] = pc2[:, :3].dot(matrix2.T) + shifts2*0.3
        else:
            pc2_tran = pc2_ori.dot(matrix2.T) + shifts2
            sf[:,:3] = pc2_tran - pc1_ori
            pc2[:, :3] = pc2[:, :3].dot(matrix2.T) + shifts2

            rotMat_2 = torch.from_numpy(np.expand_dims(matrix2, axis = 0).repeat(sf.shape[0], axis=0)).double()
            translation_2 = torch.from_numpy(np.expand_dims(shifts2[0:pc1_ori.shape[0]], axis = -1)).double()
            pose_matrix_2 = torch.cat((torch.cat((rotMat_2, translation_2), 2), filler), 1)
            T_new_2 = torch.matmul(pose_matrix_2, T_new)
            # pc1_init_tran = torch.matmul(T_new_2, torch.from_numpy(np.expand_dims(np.float64(Nor_points), axis = -1)))
            # sf_init = pc1_init_tran[:,0:3,0] - pc1_init[:,0:3]
            # error = sf[:,0:3] - sf_init.numpy()
            # print (pc1_init_tran.shape,sf_init.shape)
            # print ("sum:", np.sum(error))

            T_new_2_euler_z,T_new_2_euler_y,T_new_2_euler_x= mat2euler(T_new_2[:,0:3,0:3])
            t = T_new_2[:,0:3,3]
            TT = torch.cat((T_new_2_euler_z.unsqueeze(1), T_new_2_euler_y.unsqueeze(1),T_new_2_euler_x.unsqueeze(1),t), 1)
            sf[:,7:13] = TT
            # ori_mat_multi000, pose_mat_multi000 = euler2mat(TT.numpy())
            # # pc1_init_tran = torch.matmul(pose_mat_multi000, torch.from_numpy(np.expand_dims(np.float64(Nor_points), axis = -1)))
            # # sf_init = pc1_init_tran[:,0:3,0] - pc1_init[:,0:3]
            # # error = sf[:,0:3] - sf_init.numpy()
            # one = np.expand_dims(np.ones_like(pc1_init[:,0]), 1)
            # Nor_points = np.hstack((pc1[:, 0:3], one))
            # # print (Nor_points.shape)
            # pc1_init_tran = torch.matmul(T_new_2, torch.from_numpy(np.expand_dims(np.float64(Nor_points), axis = -1)))
            # sf_init = pc1_init_tran[:,0:3,0] - pc1[:,0:3]
            # error = sf[:,0:3] - sf_init.numpy()
            # print (pc1_init_tran.shape,sf_init.shape)
            # print ("sum:", np.sum(np.abs(error)))
            # print (T_new_2_euler_z.shape)
        # xxx
        # pose_eur = sf[:,7:13]#[mask.A[:,0]]
        # ori_mat_multi, pose_mat_multi = euler2mat((pose_eur))
        # Trans_Nor_points_copy = torch.matmul((pose_mat_multi), torch.from_numpy(np.expand_dims(np.float64(pc1), axis = -1))).numpy()
        # scene_flow_2 = Trans_Nor_points_copy.squeeze(-1)-pc1
        # error = scene_flow[:,0:3] - scene_flow_2[:,0:3]



        if not self.no_corr:
            jitter2 = np.clip(self.pc2_args['jitter_sigma'] * np.random.randn(pc1.shape[0], 3),
                              -self.pc2_args['jitter_clip'],
                              self.pc2_args['jitter_clip']).astype(np.float32)
            pc2[:, :3] += jitter2

        if self.DEPTH_THRESHOLD > 0:
            near_mask = np.logical_and(pc1[:, 2] < 10000*self.DEPTH_THRESHOLD, pc2[:, 2] < 10000*self.DEPTH_THRESHOLD)
        else:
            near_mask = np.ones(pc1.shape[0], dtype=np.bool)

        indices = np.where(near_mask)[0]
        if len(indices) == 0:
            print('indices = np.where(mask)[0], len(indices) == 0')
            return None, None, None

        if self.num_points > 0:
            try:
                sampled_indices1 = np.random.choice(indices, size=self.num_points, replace=False, p=None)
                if self.no_corr:
                    sampled_indices2 = np.random.choice(indices, size=self.num_points, replace=False, p=None)
                else:
                    sampled_indices2 = sampled_indices1
            except ValueError:
                '''
                if not self.allow_less_points:
                    print('Cannot sample {} points'.format(self.num_points))
                    return None, None, None
                else:
                    sampled_indices1 = indices
                    sampled_indices2 = indices
                '''
                if not self.allow_less_points:
                    #replicate some points
                    sampled_indices1 = np.random.choice(indices, size=self.num_points, replace=True, p=None)
                    if self.no_corr:
                        sampled_indices2 = np.random.choice(indices, size=self.num_points, replace=True, p=None)
                    else:
                        sampled_indices2 = sampled_indices1
                else:
                    sampled_indices1 = indices
                    sampled_indices2 = indices
        else:
            sampled_indices1 = indices
            sampled_indices2 = indices

        pc1 = pc1[sampled_indices1]
        pc2 = pc2[sampled_indices2]
        sf = sf[sampled_indices1]

        return pc1, pc2, sf#[:,0:3]

    def __repr__(self):
        format_string = self.__class__.__name__ + '\n(together_args: \n'
        for key in sorted(self.together_args.keys()):
            format_string += '\t{:10s} {}\n'.format(key, self.together_args[key])
        format_string += '\npc2_args: \n'
        for key in sorted(self.pc2_args.keys()):
            format_string += '\t{:10s} {}\n'.format(key, self.pc2_args[key])
        format_string += '\ndata_process_args: \n'
        format_string += '\tDEPTH_THRESHOLD: {}\n'.format(self.DEPTH_THRESHOLD)
        format_string += '\tNO_CORR: {}\n'.format(self.no_corr)
        format_string += '\tallow_less_points: {}\n'.format(self.allow_less_points)
        format_string += '\tnum_points: {}\n'.format(self.num_points)
        format_string += ')'
        return format_string
