#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.7

import copy
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

import torch
from torch.utils.data import Dataset
from sampling import epower_iid


# 自定义数据集类
class CustomDataset(Dataset):
    def __init__(self, data, label):
        self.data = data
        self.label = label

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        x = torch.FloatTensor(self.data[index, :])
        y = torch.FloatTensor([self.label[index]])
        return x, y


def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
	n_vars = 1 if type(data) is list else data.shape[1]
	dff = pd.DataFrame(data)
	cols, names = list(), list()
	# input sequence (t-n, ... t-1)
	for i in range(n_in, 0, -1):
		cols.append(dff.shift(i))
		names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
	# forecast sequence (t, t+1, ... t+n)
	for i in range(0, n_out):
		cols.append(dff.shift(-i))
		if i == 0:
			names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
		else:
			names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
	# put it all together
	agg = pd.concat(cols, axis=1)
	agg.columns = names
	# drop rows with NaN values
	if dropnan:
		agg.dropna(inplace=True)
	return agg

def get_dataset(args):
    if args.dataset == 'epower':
        data_dir = './data/household_power_consumption.txt'
        df = pd.read_csv(data_dir, sep=';', 
                        parse_dates={'dt' : ['Date', 'Time']}, infer_datetime_format=True, 
                        low_memory=False, na_values=['nan','?'], index_col='dt')
        droping_list_all=[]
        for j in range(0,7):
            if not df.iloc[:, j].notnull().all():
                droping_list_all.append(j)        
        for j in range(0,7):        
            df.iloc[:,j]=df.iloc[:,j].fillna(df.iloc[:,j].mean())
        ## resampling of data over hour
        df_resample = df.resample('h').mean() 
        values = df_resample.values 
        ## full data without resampling
        #values = df.values
        ## normalize features
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled = scaler.fit_transform(values)
        ## frame as supervised learning
        reframed = series_to_supervised(scaled, 1, 1)
        ## drop columns we don't want to predict
        reframed.drop(reframed.columns[[8,9,10,11,12,13]], axis=1, inplace=True)
        # split into train and test sets
        values = reframed.values
        n_train_time = 365*24
        train = values[:n_train_time, :]
        test = values[n_train_time:, :]
        train_x, train_y = train[:, :-1], train[:, -1]
        test_x, test_y = test[:, :-1], test[:, -1]
        train_x = train_x.reshape((train_x.shape[0], 1, train_x.shape[1]))
        test_x = test_x.reshape((test_x.shape[0], 1, test_x.shape[1]))
        user_groups = epower_iid(train_x, args.num_users)
    return train_x, train_y, test_x, test_y, user_groups

def fedavg(w):
    """
    Returns the average of the weights.
    """
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        w_avg[key] = torch.div(w_avg[key], len(w))
    return w_avg

def DSSGD(w, w_glob, theta_upload=0.1):
    # theta_upload
    w_row = torch.cat([v.flatten() for _, v in w.items()])
    w_glob_row = torch.cat([v.flatten() for _, v in w_glob.items()])
    delta_grad = torch.abs(w_glob_row - w_row)
    # upload theta_upload% weights to global model for each user
    _, indexes = torch.topk(delta_grad, int(len(delta_grad) * theta_upload))
    # update global weights
    w_glob_row[indexes] = w_row[indexes]
    uploaded_w = copy.deepcopy(w_glob)
    start = 0
    for k, v in uploaded_w.items():
        uploaded_w[k] = w_glob_row[start:start + v.nelement()].view(v.size())
        start += v.nelement()
    return uploaded_w

def exp_details(args):
    print('\nExperimental details:')
    print(f'    Model     : {args.model}')
    print(f'    Optimizer : {args.optimizer}')
    print(f'    Learning  : {args.lr}')
    print(f'    Global Rounds   : {args.epochs}\n')

    print('    Federated parameters:')
    if args.iid:
        print('    IID')
    else:
        print('    Non-IID')
    print(f'    Fraction of users  : {args.frac}')
    print(f'    Local Batch size   : {args.local_bs}')
    print(f'    Local Epochs       : {args.local_ep}\n')
    return