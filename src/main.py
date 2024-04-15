#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.7

import os
import copy
from time import time
import numpy as np
from tqdm import tqdm

import torch
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt

from options import args_parser
from utils import CustomDataset,get_dataset, DSSGD, exp_details,fedavg
from model import LSTM
from update import LocalUpdate, test_inference




if __name__ == '__main__':
    all_time_start = time()

    ## define paths
    path_project = os.path.abspath('..')
    logger = SummaryWriter('../logs')

    args = args_parser()
    exp_details(args)

    device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    print("Training on:", device)
    
    
    train_X, train_y, test_X, test_y, user_groups = get_dataset(args)
    
    ## BUILD MODEL
    if args.model == 'lstm':
        global_model = LSTM(args=args)
        
    ## Set the model to train and send it to device.
    global_model.to(device)
    global_model.train()
    print(global_model)
    
    ## copy weights
    global_weight = global_model.state_dict()
    
    ## Training
    epoch_train_losses =  []
    epoch_val_losses =  []
    epoch_test_losses = []
    
    ## 创建自定义数据集实例
    train_dataset = CustomDataset(train_X, train_y)
    test_dataset = CustomDataset(test_X, test_y)
    agg_time = 0.0
    total_params = []
    for epoch in tqdm(range(args.epochs)):
        local_weights, local_train_losses, val_losses = [], [], []
        print(f'\n | Global Training Round : {epoch+1} |\n')
        
        global_model.train()
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        
        ## 本地梯度上传
        for idx in idxs_users:
            ## 用户更新本地模型
            uer_model = copy.deepcopy(global_model)
            local_train = LocalUpdate(args=args, dataset=train_dataset,
                                      idxs=user_groups[idx], logger=logger)
            w, local_train_loss = local_train.update_weights(
                model=uer_model, global_round=epoch)
            ## 记录当前用户组本地误差
            local_train_losses.append(copy.deepcopy(local_train_loss))
            local_weights.append(w)
            total_param = sum(p.numel() for p in uer_model.parameters())
            # pytorch float32 = 4B
            bandwidth = (total_param * 4) >> 10 
            # print(f"Total parameters:{total_param}")
            # print('bandwidth : {:.4f} KB'.format(bandwidth))

            ## 更新全局模型
            # agg_time_start = time()
            # global_weight = DSSGD(w)
            # global_model.load_state_dict(global_weight)
            # agg_time += (time() - agg_time_start)
            ## 验证全局模型
            # global_model.eval()
            # val_loss = local_train.inference(model=global_model)
            # val_losses.append(val_loss)
        global_weights = fedavg(local_weights)
        global_model.load_state_dict(global_weights)
        ## 验证全局模型
        global_model.eval()
        val_loss = local_train.inference(model=global_model)
        # val_losses.append(val_loss)
        local_train_loss_avg = sum(local_train_losses) / len(local_train_losses)
        # val_loss_avg = sum(val_losses) / len(val_losses)
        epoch_train_losses.append(local_train_loss_avg)
        # epoch_val_losses.append(val_loss_avg)
        epoch_val_losses.append(val_loss)
        print(f' \nAvg Training Stats after {epoch+1} global rounds:')
        print(f'Training Loss : {np.mean(np.array(epoch_train_losses))}')
        print(f'Validation Loss : {val_loss}')
        epoch_test_loss = test_inference(args, global_model, test_dataset)
        print(f' \n Results after {epoch+1} global rounds of training:')
        print("|---- Test Loss: {:.4f}".format(epoch_test_loss))
        epoch_test_losses.append(epoch_test_loss)

    
    ## summarize history for loss
    plt.plot(epoch_train_losses)
    plt.plot(epoch_val_losses)
    plt.plot(epoch_test_losses)
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val', 'test'], loc='upper right')
    plt.show()
    all_time_end = time()
    print('\n Total Run Time: {0:0.4f}'.format(all_time_end-all_time_start))
    print('\n Aggregation Time: {0:0.4f}'.format(agg_time/args.epochs))
    
    # Saving the objects train_loss and train_accuracy:
    acc_file = './res/acc_{}_{}_{}_C[{}]_E[{}]_B[{}].txt'.\
        format(args.dataset, args.model, args.epochs, args.frac,
               args.local_ep, args.local_bs)
    val_file = './res/val_{}_{}_{}_C[{}]_E[{}]_B[{}].txt'.\
        format(args.dataset, args.model, args.epochs, args.frac,
               args.local_ep, args.local_bs)
    loss_file = './res/loss_{}_{}_{}_C[{}]_E[{}]_B[{}].txt'.\
        format(args.dataset, args.model, args.epochs, args.frac,
               args.local_ep, args.local_bs)

    with open(acc_file, 'w') as f:
        for i in epoch_train_losses:
            f.write(str(i)+'\n')
    with open(val_file, 'w') as f:
        for i in epoch_val_losses:
            f.write(str(i)+'\n')   
    with open(loss_file, 'w') as f:
        for i in epoch_test_losses:
            f.write(str(i)+'\n') 