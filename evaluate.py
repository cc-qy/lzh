"""
Evaluation
Author: Wenxuan Wu
Date: May 2020
"""

import argparse
import sys 
import os 

import torch, numpy as np, glob, math, torch.utils.data, scipy.ndimage, multiprocessing as mp
import torch.nn.functional as F
import time
import torch.nn as nn
import pickle 
import datetime
import logging

from tqdm import tqdm 
from models import PointConvSceneFlowPWC8192selfglobalPointConv as PointConvSceneFlow
from models import multiScaleLoss
from pathlib import Path
from collections import defaultdict

import transforms
import datasets
import cmd_args 
from main_utils import *
from utils import geometry
from evaluation_utils import evaluate_2d, evaluate_3d
from utils import utils
import numpy as np
import copy
import numpy.ma as ma
from PIL import Image

evaluate_seq_num = [0,
             0,26,1,5,3,
             0,11,31,6,11,
             0,32,6 ,7,4,
             0,0,0,13,62,
             68]
def main():

    #import ipdb; ipdb.set_trace()
    if 'NUMBA_DISABLE_JIT' in os.environ:
        del os.environ['NUMBA_DISABLE_JIT']

    global args 
    args = cmd_args.parse_args_from_yaml(sys.argv[1])

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu if args.multi_gpu is None else '0,1'

    '''CREATE DIR'''
    experiment_dir = Path('./Evaluate_experiment/')
    experiment_dir.mkdir(exist_ok=True)
    file_dir = Path(str(experiment_dir) + '/%sFlyingthings3d-'%args.model_name + str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')))
    file_dir.mkdir(exist_ok=True)
    checkpoints_dir = file_dir.joinpath('checkpoints/')
    checkpoints_dir.mkdir(exist_ok=True)
    log_dir = file_dir.joinpath('logs/')
    log_dir.mkdir(exist_ok=True)
    os.system('cp %s %s' % ('models.py', log_dir))
    os.system('cp %s %s' % ('pointconv_util.py', log_dir))
    os.system('cp %s %s' % ('evaluate.py', log_dir))
    os.system('cp %s %s' % ('config_evaluate.yaml', log_dir))

    '''LOG'''
    logger = logging.getLogger(args.model_name)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler(str(log_dir) + 'train_%s_sceneflow.txt'%args.model_name)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.info('----------------------------------------TRAINING----------------------------------')
    logger.info('PARAMETER ...')
    logger.info(args)

    blue = lambda x: '\033[94m' + x + '\033[0m'
    model = PointConvSceneFlow()

    val_dataset = datasets.__dict__[args.dataset](
        train=False,
        transform=transforms.ProcessData(args.data_process,
                                         args.num_points,
                                         args.allow_less_points),
        num_points=args.num_points,
        data_root = args.data_root
    )
    logger.info('val_dataset: ' + str(val_dataset))
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        worker_init_fn=lambda x: np.random.seed((torch.initial_seed()) % (2 ** 32))
    )

    #load pretrained model
    pretrain = args.ckpt_dir + args.pretrain
    model.load_state_dict(torch.load(pretrain))
    print('load model %s'%pretrain)
    logger.info('load model %s'%pretrain)

    model.cuda()

    epe3ds = AverageMeter()
    acc3d_stricts = AverageMeter()
    acc3d_relaxs = AverageMeter()
    outliers = AverageMeter()
    evaluator_3d = utils.Evaluator(['static','move']) 
    # 2D
    epe2ds = AverageMeter()
    acc2ds = AverageMeter()

    total_loss = 0
    total_seen = 0
    total_epe = 0
    total_move_num = 0
    metrics = defaultdict(lambda:list())
    for i, data in tqdm(enumerate(val_loader, 0), total=len(val_loader), smoothing=0.9):
        pos1, pos2, norm1, norm2, flow, path = data  

        #move to cuda 
        pos1 = pos1.cuda()
        pos2 = pos2.cuda() 
        norm1 = norm1.cuda()
        norm2 = norm2.cuda()
        flow = flow.cuda() 

        model = model.eval()
        with torch.no_grad(): 
            pred_flows, fps_pc1_idxs, _, _, _ = model(pos1, pos2, norm1, norm2)

            loss = multiScaleLoss(pred_flows, flow[:,:,0:3], fps_pc1_idxs)

            full_flow = pred_flows[0].permute(0, 2, 1)
            epe3d = torch.norm(full_flow - flow[:,:,0:3], dim = 2).mean()

        total_loss += loss.cpu().data * args.batch_size
        total_epe += epe3d.cpu().data * args.batch_size
        total_seen += args.batch_size

        pc1_np = pos1.cpu().numpy()
        pc2_np = pos2.cpu().numpy() 
        sf_np = flow.cpu().numpy()
        pred_sf = full_flow.cpu().numpy()

        EPE3D, acc3d_strict, acc3d_relax, outlier = evaluate_3d(pred_sf, sf_np[:,:,0:3])

        epe3ds.update(EPE3D)
        acc3d_stricts.update(acc3d_strict)
        acc3d_relaxs.update(acc3d_relax)
        outliers.update(outlier)

        # 2D evaluation metrics
        # print (path)
        flow_pred, flow_gt = geometry.get_batch_2d_flow(pc1_np,
                                                        pc1_np+sf_np[:,:,0:3],
                                                        pc1_np+pred_sf,
                                                        path)
        EPE2D, acc2d = evaluate_2d(flow_pred, flow_gt)

        epe2ds.update(EPE2D)
        acc2ds.update(acc2d)

        # evaluator_3d = Evaluator(tuple('static','move')) 
        # evaluator_3d.update(pred_label_3d, curr_seg_label)
        # print (evaluator_3d.overall_iou)
        # xxx

        ####view
        pos1 = pos1[0].detach().cpu().numpy()
        pos2 = pos2[0].detach().cpu().numpy()

        rigid_scene_flow = utils.cal_rigid_scene_flow(pos1, path[0])
        # print (rigid_scene_flow.shape)
        # xxx
        # flow_pred = flow_pred.cpu().numpy()
        # flow_gt = flow_gt.detach().cpu().numpy()
        # print (pos1.shape)
        # print (pos2.shape)
        # print (flow_pred.shape)
        # print (flow_gt.shape)
        # print (sf_np[0,0])
        # xxx
        gt_moving_mask = (sf_np[0,:,5] == 1)
        # print (sum(gt_moving_mask), path)
        moving_mask =  np.linalg.norm(pred_sf[0,:,0:3] - rigid_scene_flow[:,0:3], axis=-1)>0.1
        static_mask = np.logical_not(moving_mask)
        ###
        if sum(gt_moving_mask)>-1:
            total_move_num += 1
            evaluator_3d.update(moving_mask, gt_moving_mask)
        # evaluator_3d.update(moving_mask, gt_moving_mask)
        # print (100 * evaluator_3d.overall_iou, 100.0 * evaluator.overall_acc)
        # xxx
        ######
        if sum(gt_moving_mask)>-1:
            img_1 = utils.view_point(pos1)
            img_2 = utils.view_point(pos2)
            img_1_warp_2 = utils.view_point(pos1 + pred_sf[0,:,0:3])
            # img_1_rigid_2 = utils.view_point(pos1 + sf_np[0,:,0:3])
            img_1_rigid_2 = utils.view_point(pos1 + rigid_scene_flow[:,0:3])

            img_1_move = utils.view_point(pos1[moving_mask])
            img_1_move_gt = utils.view_point(pos1[gt_moving_mask])

            h, w = img_1.shape
            im_write = np.zeros_like(img_1)[:,0:10]+255

            im_write = np.expand_dims(im_write,axis = 2)
            im_write_zero = np.zeros_like(im_write)
            im_write = np.dstack([im_write, im_write, im_write]) 

            img_1 = np.expand_dims(img_1,axis = 2)
            im_write_zero = np.zeros_like(img_1)
            img_1 = np.dstack([im_write_zero, img_1, im_write_zero]) 

            img_2 = np.expand_dims(img_2,axis = 2)
            im_write_zero = np.zeros_like(img_2)
            img_2 = np.dstack([img_2, im_write_zero, im_write_zero]) 

            img_1_warp_2 = np.expand_dims(img_1_warp_2,axis = 2)
            im_write_zero = np.zeros_like(img_1_warp_2)
            img_1_warp_2 = np.dstack([im_write_zero, img_1_warp_2, im_write_zero]) 

            img_1_rigid_2 = np.expand_dims(img_1_rigid_2,axis = 2)
            im_write_zero = np.zeros_like(img_1_rigid_2)
            img_1_rigid_2 = np.dstack([img_1_rigid_2, img_1_rigid_2, img_1_rigid_2]) 

            img_1_move = np.expand_dims(img_1_move,axis = 2)
            im_write_zero = np.zeros_like(img_1_move)
            img_1_move = np.dstack([im_write_zero, img_1_move, im_write_zero]) 

            img_1_move_gt = np.expand_dims(img_1_move_gt,axis = 2)
            im_write_zero = np.zeros_like(img_1_move_gt)
            img_1_move_gt = np.dstack([img_1_move_gt, im_write_zero, im_write_zero]) 

            image_real_all = np.hstack([im_write,img_1_move, im_write, img_1_move_gt, im_write, img_2+img_1_warp_2, im_write, img_2+img_1_rigid_2,im_write])
            # image_real_all = np.hstack([im_write,img_1, im_write, img_2, im_write, img_2+img_1_warp_2, im_write, img_2+img_1_rigid_2,im_write])

            image_real_all_show = Image.fromarray(255-image_real_all)
            num = int (path[0].split('/')[-1])
            seq = int (path[0].split('/')[-2])
            # EPE3D = EPE3D.detach().cpu().numpy()
            # print (EPE3D*100)
            EPE3D = round(EPE3D*100, 2)
            image_real_all_show.save('./view/' + str(seq).zfill(4) + '/' + str(num).zfill(6) + '_'+str(EPE3D)+'view_all.jpg')

    mean_loss = total_loss / total_seen
    mean_epe = total_epe / total_seen
    str_out = '%s mean loss: %f mean epe: %f'%(blue('Evaluate'), mean_loss, mean_epe)
    print(str_out)
    logger.info(str_out)
    print ( "overall_iou:", 100 * evaluator_3d.overall_iou)
    print ( "overall_acc:",100 * evaluator_3d.overall_acc)
    print (evaluator_3d.print_table(), total_move_num)
    evaluator_3d.save_table('./table.txt')
    # print (100 * evaluator_3d.overall_iou, 100.0 * evaluator.overall_acc)

    res_str = (' * EPE3D {epe3d_.avg:.4f}\t'
               'ACC3DS {acc3d_s.avg:.4f}\t'
               'ACC3DR {acc3d_r.avg:.4f}\t'
               'Outliers3D {outlier_.avg:.4f}\t'
               'EPE2D {epe2d_.avg:.4f}\t'
               'ACC2D {acc2d_.avg:.4f}'
               .format(
                       epe3d_=epe3ds,
                       acc3d_s=acc3d_stricts,
                       acc3d_r=acc3d_relaxs,
                       outlier_=outliers,
                       epe2d_=epe2ds,
                       acc2d_=acc2ds
                       ))

    print(res_str)
    logger.info(res_str)


if __name__ == '__main__':
    main()




