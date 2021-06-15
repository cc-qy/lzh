"""
Train on FlyingThings3D
Author: Wenxuan Wu
Date: May 2020
"""

import argparse
import sys 
import os 

import torch, numpy as np, glob, math, torch.utils.data, scipy.ndimage, multiprocessing as mp
import torch.nn.functional as F
import time
import pickle 
import datetime
import logging

from tqdm import tqdm 
from models import PointConvSceneFlowPWC8192selfglobalPointConv as PointConvSceneFlow
from models import multiScaleLoss, multiScaleLoss_pose
from pathlib import Path
from collections import defaultdict

import transforms
import datasets
import cmd_args 
from utils import utils
from main_utils import *

def main():

    if 'NUMBA_DISABLE_JIT' in os.environ:
        del os.environ['NUMBA_DISABLE_JIT']

    global args 
    args = cmd_args.parse_args_from_yaml(sys.argv[1])

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu if args.multi_gpu is None else '1'

    '''CREATE DIR'''
    experiment_dir = Path('../model-pose/')
    experiment_dir.mkdir(exist_ok=True)
    file_dir = Path(str(experiment_dir) + '/PointConv%sKITTI-'%args.model_name + str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')))
    file_dir.mkdir(exist_ok=True)
    checkpoints_dir = file_dir.joinpath('checkpoints/')
    checkpoints_dir.mkdir(exist_ok=True)
    log_dir = file_dir.joinpath('logs/')
    log_dir.mkdir(exist_ok=True)
    os.system('cp %s %s' % ('models.py', log_dir))
    os.system('cp %s %s' % ('pointconv_util.py', log_dir))
    os.system('cp %s %s' % ('train.py', log_dir))
    os.system('cp %s %s' % ('config_train.yaml', log_dir))

    '''LOG'''
    logger = logging.getLogger(args.model_name)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler(str(log_dir) + '/train_%s_sceneflow.txt'%args.model_name)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.info('----------------------------------------TRAINING----------------------------------')
    logger.info('PARAMETER ...')
    logger.info(args)

    blue = lambda x: '\033[94m' + x + '\033[0m'
    model = PointConvSceneFlow()

    train_dataset = datasets.__dict__[args.dataset](
        train=True,
        transform=transforms.Augmentation(args.aug_together,
                                            args.aug_pc2,
                                            args.data_process,
                                            args.num_points),
        num_points=args.num_points,
        data_root = args.data_root#,
       # full=args.full
    )
    logger.info('train_dataset: ' + str(train_dataset))
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        worker_init_fn=lambda x: np.random.seed((torch.initial_seed()) % (2 ** 32))
    )

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

    # '''GPU selection and multi-GPU'''
    # if args.multi_gpu is not None:
    #     device_ids = [int(x) for x in args.multi_gpu.split(',')]
    #     torch.backends.cudnn.benchmark = True 
    #     model.cuda(device_ids[0])
    #     model = torch.nn.DataParallel(model, device_ids = device_ids)
    # else:
    #     model.cuda()

    if args.pretrain is not None:
        model.load_state_dict(torch.load(args.pretrain))
        print('load model %s'%args.pretrain)
        logger.info('load model %s'%args.pretrain)
    else:
        print('Training from scratch')
        logger.info('Training from scratch')

        '''GPU selection and multi-GPU'''
    if args.multi_gpu is not None:
        device_ids = [int(x) for x in args.multi_gpu.split(',')]
        torch.backends.cudnn.benchmark = True 
        model.cuda(device_ids[0])
        model = torch.nn.DataParallel(model, device_ids = device_ids)
    else:
        model.cuda()

    pretrain = args.pretrain 
    init_epoch = int(pretrain[-14:-11]) if args.pretrain is not None else 0 

    if args.optimizer == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9)
    elif args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.999), 
                                     eps=1e-08, weight_decay=args.weight_decay)
                
    optimizer.param_groups[0]['initial_lr'] = args.learning_rate 
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=80, gamma=0.5, last_epoch = init_epoch - 1)
    LEARNING_RATE_CLIP = 1e-5 

    history = defaultdict(lambda: list())
    best_epe = 1000.0
    for epoch in range(init_epoch, args.epochs):
        lr = max(optimizer.param_groups[0]['lr'], LEARNING_RATE_CLIP)
        print('Learning rate:%f'%lr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        total_loss = 0
        total_seen = 0
        optimizer.zero_grad()
        for i, data in tqdm(enumerate(train_loader, 0), total=len(train_loader), smoothing=0.9):
            pos1, pos2, norm1, norm2, flow, _ = data  
            #move to cuda 
            pos1 = pos1.cuda()
            pos2 = pos2.cuda() 
            norm1 = norm1.cuda()
            norm2 = norm2.cuda()
            flow = flow.cuda() 

            model = model.train() 
            pred_flows, pred_poses_r,pred_poses_t,fps_pc1_idxs, _, _, _ = model(pos1, pos2, norm1, norm2)

            loss = multiScaleLoss(pred_flows, flow[:,:,0:3], fps_pc1_idxs)
            loss_pose_r = multiScaleLoss_pose(pred_poses_r, flow[:,:,7:10], fps_pc1_idxs)
            loss_pose_t = multiScaleLoss_pose(pred_poses_t, flow[:,:,10:13], fps_pc1_idxs)
            
            # print ("loss:",(loss.cpu().detach().numpy()).round(2),"loss_pose_r:",
            #     (loss_pose_r.cpu().detach().numpy()).round(2),"loss_pose_r:",(loss_pose_t.cpu().detach().numpy()).round(2)
            #     ,"loss_real_rt:",(1 * (20*loss_pose_r + loss_pose_t).cpu().detach().numpy()).round(2))

            epe3d = torch.norm(pred_flows[0].permute(0, 2, 1) - flow[:,:,0:3], dim = 2).mean()
            # print (loss_pose_r[0].shape)
            # ori_mat_multi, pose_mat_multi = utils.euler2mat((pose_eur))
            B,_,N = pred_poses_r[0].shape
            pose_eur = torch.cat((pred_poses_r[0],pred_poses_t[0]),1).permute(0, 2, 1)
            pose_eur = pose_eur.reshape(B*N,6)
            # pose_eur = flow[:,:,7:13].reshape(8192,6)
            # flow[:,:,7:13]
            # print (flow[:,:,7:13].shape)

            ori_mat_multi, pose_mat_multi = utils.euler2mat((pose_eur.cpu().detach().numpy()))
            # print (pose_eur.shape)flow[:,:,0:3]
            pos1 = pos1.reshape(B*N,3).cpu().detach().numpy()
            one = np.expand_dims(np.ones_like(pos1[:,0]), 1)
            Nor_points = np.hstack((pos1[:, 0:3], one))
            # xxx
            pc1_init_tran = torch.matmul(pose_mat_multi, torch.from_numpy(np.expand_dims(np.float64(Nor_points), axis = -1)))
            flow_p = pc1_init_tran[:,:,0] - Nor_points
            flow_p = flow_p[:,0:3].reshape(B,N,3)
            epe3d_p = torch.norm(flow_p.cuda() - flow[:,:,0:3], dim = 2).mean()
            # print (flow_p.shape)
            # xxx
            # epe3d = torch.norm(flow_p - flow[:,:,0:3], dim = 2).mean()
            # print (pc1_init_tran.shape)
            # xxx
            # sf_init = pc1_init_tran[:,0:3,0] - pc1_init[:,0:3]
            # # # error = sf[:,0:3] - sf_init.numpy()
            # ori_mat_multi, pose_mat_multi = euler2mat((pose_eur))
            # epe_r = torch.norm(loss_pose_r[0].permute(0, 2, 1) - flow[:,:,0:3], dim = 2).mean()
            # epe_t = torch.norm(loss_pose_t[0].permute(0, 2, 1) - flow[:,:,0:3], dim = 2).mean()
            if i%100 == 0:
                print ("loss:",(loss.cpu().detach().numpy()).round(2),"loss_pose_r:",
                    (loss_pose_r.cpu().detach().numpy()).round(2),"loss_pose_r:",(loss_pose_t.cpu().detach().numpy()).round(2)
                    ,"loss_real_rt:",(1 * (20*loss_pose_r + loss_pose_t).cpu().detach().numpy()).round(2))
                print ("error:",(epe3d.cpu().detach().numpy()).round(2),"error_p:",(epe3d_p.cpu().detach().numpy()).round(2))

            history['loss'].append(loss.cpu().data.numpy())
            (loss + 1 * (20*loss_pose_r + loss_pose_t)).backward()
            optimizer.step() 
            optimizer.zero_grad()

            total_loss += (loss.cpu().data ) * args.batch_size
            total_seen += args.batch_size

        scheduler.step()

        train_loss = total_loss / total_seen
        str_out = 'EPOCH %d %s mean loss: %f'%(epoch, blue('train'), train_loss)
        print(str_out)
        logger.info(str_out)

        eval_epe3d, eval_epe3d_p, eval_loss = eval_sceneflow(model.eval(), val_loader)
        str_out = 'EPOCH %d %s mean epe3d: %f mean epe3d_p: %f mean eval loss: %f'%(epoch, blue('eval'), eval_epe3d, eval_epe3d_p, eval_loss)
        print(str_out)
        logger.info(str_out)

        if eval_epe3d < best_epe:
            best_epe = eval_epe3d
            if args.multi_gpu is not None:
                torch.save(model.module.state_dict(), '%s/%s_%.3d_%.4f.pth'%(checkpoints_dir, args.model_name, epoch, best_epe))
            else:
                torch.save(model.state_dict(), '%s/%s_%.3d_%.4f.pth'%(checkpoints_dir, args.model_name, epoch, best_epe))
            logger.info('Save model ...')
            print('Save model ...')
        print('Best epe loss is: %.5f'%(best_epe))
        logger.info('Best epe loss is: %.5f'%(best_epe))


def eval_sceneflow(model, loader):

    metrics = defaultdict(lambda:list())
    for batch_id, data in tqdm(enumerate(loader), total=len(loader), smoothing=0.9):
        pos1, pos2, norm1, norm2, flow, _ = data  
        
        #move to cuda 
        pos1 = pos1.cuda()
        pos2 = pos2.cuda() 
        norm1 = norm1.cuda()
        norm2 = norm2.cuda()
        flow = flow.cuda() 

        with torch.no_grad():
            pred_flows, pred_poses_r,pred_poses_t,fps_pc1_idxs, _, _, _ = model(pos1, pos2, norm1, norm2)

            eval_loss = multiScaleLoss(pred_flows, flow[:,:,0:3], fps_pc1_idxs)

            epe3d = torch.norm(pred_flows[0].permute(0, 2, 1) - flow[:,:,0:3], dim = 2).mean()

            B,_,N = pred_poses_r[0].shape
            pose_eur = torch.cat((pred_poses_r[0],pred_poses_t[0]),1).permute(0, 2, 1)
            pose_eur = pose_eur.reshape(B*N,6)

            ori_mat_multi, pose_mat_multi = utils.euler2mat((pose_eur.cpu().detach().numpy()))
            pos1 = pos1.reshape(B*N,3).cpu().detach().numpy()
            one = np.expand_dims(np.ones_like(pos1[:,0]), 1)
            Nor_points = np.hstack((pos1[:, 0:3], one))
            pc1_init_tran = torch.matmul(pose_mat_multi, torch.from_numpy(np.expand_dims(np.float64(Nor_points), axis = -1)))
            flow_p = pc1_init_tran[:,:,0] - Nor_points
            flow_p = flow_p[:,0:3].reshape(B,N,3)
            epe3d_p = torch.norm(flow_p.cuda() - flow[:,:,0:3], dim = 2).mean()

        metrics['epe3d_loss'].append(epe3d.cpu().data.numpy())
        metrics['epe3d_p_loss'].append(epe3d_p.cpu().data.numpy())
        metrics['eval_loss'].append(eval_loss.cpu().data.numpy())

    mean_epe3d = np.mean(metrics['epe3d_loss'])
    mean_epe3d_p = np.mean(metrics['epe3d_p_loss'])
    mean_eval = np.mean(metrics['eval_loss'])

    return mean_epe3d, mean_epe3d_p, mean_eval

if __name__ == '__main__':
    main()




