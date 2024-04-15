#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

class DatasetSplit(Dataset):
    """An abstract Dataset class wrapped around Pytorch Dataset class.
    """

    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return torch.tensor(image), torch.tensor(label)

class LocalUpdate(object):
    def __init__(self, args, dataset, idxs, logger):
        self.args = args
        self.logger = logger
        self.trainloader, self.validloader = self.train_val_test(
            dataset, list(idxs))
        self.device = "cuda:0" if args.gpu == 0 else 'cpu'
        # Default criterion set to NLL loss function
        self.criterion = nn.MSELoss().to(self.device)

    def train_val_test(self, dataset, idxs):
        """
        Returns train, validation and test dataloaders for a given dataset
        and user indexes.
        """
        # split indexes for train, validation, and test (80, 10, 10)
        idxs_train = idxs[:int(0.8*len(idxs))]
        idxs_val = idxs[int(0.8*len(idxs)):]

        trainloader = DataLoader(DatasetSplit(dataset, idxs_train),
                                 batch_size=self.args.local_bs, shuffle=True)
        validloader = DataLoader(DatasetSplit(dataset, idxs_val),
                                 batch_size=max(int(len(idxs_val)/10),1), shuffle=False)
        return trainloader, validloader

    def update_weights(self, model, global_round):
        # Set mode to train model
        model.train()
        epoch_loss = []

        # Set optimizer for the local updates
        if self.args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=self.args.lr,
                                        momentum=0.5)
        elif self.args.optimizer == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr,
                                         weight_decay=1e-4)
        elif self.args.optimizer == 'adagd':
            optimizer = torch.optim.Adagrad(model.parameters(), lr=self.args.lr, 
                                            lr_decay=1e-4)

        for _ in range(self.args.local_ep):
            batch_loss = []
            batch_data_parameters_grad_dict = {}  # 用于存储批数据计算得到的参数梯度
            for batch_idx, (datas, labels) in enumerate(self.trainloader):
                per_data_parameters_grad_dict = {}  # 用于存储每个样本计算得到的参数梯度
                datas, labels = datas.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                probs = model(datas)
                loss = self.criterion(probs, labels)
                loss.backward()
                optimizer.step()
                if self.args.verbose and (batch_idx % 10 == 0):
                    print('| Global Round : {} | Local Epoch : {} | [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        global_round, iter, batch_idx * len(datas),
                        len(self.trainloader.dataset),
                        100. * batch_idx / len(self.trainloader), loss.item()))
                self.logger.add_scalar('loss', loss.item())

                batch_loss.append(loss.item())
            #     # 计算每个样本计算得到的参数梯度的范数
            #     model_parameter_grad_norm = 0.0
            #     with torch.no_grad():
            #         for name, param in model.named_parameters():
            #             model_parameter_grad_norm += (torch.norm(param.grad) ** 2).item()
            #             per_data_parameters_grad_dict[name] = param.grad.clone().detach()
            #         model_parameter_grad_norm = np.sqrt(model_parameter_grad_norm)
                
            #         for name in per_data_parameters_grad_dict:
            #             per_data_parameters_grad_dict[name] /= max(1, model_parameter_grad_norm / 4.0)  # 梯度裁剪
            #             if name not in batch_data_parameters_grad_dict:
            #                 batch_data_parameters_grad_dict[name] = per_data_parameters_grad_dict[name]
            #             else:
            #                 batch_data_parameters_grad_dict[name] += per_data_parameters_grad_dict[name]
            #         for param in model.parameters():
            #             param.grad.zero_()  # 梯度清零

            # for name in batch_data_parameters_grad_dict:  # 为批数据计算得到的参数梯度加噪，并求平均
            #     batch_data_parameters_grad_dict[name] += torch.randn(batch_data_parameters_grad_dict[name].shape).to(self.device) * 4.0 * 2.0  # 梯度加噪
            #     batch_data_parameters_grad_dict[name] /= len(datas)
            
            # with torch.no_grad():  # 使用加噪后梯度进行SGD优化
            #     for name, param in model.named_parameters():
            #         param -= self.args.lr * batch_data_parameters_grad_dict[name]

            epoch_loss.append(sum(batch_loss)/len(batch_loss))

        return model.state_dict(), sum(epoch_loss) / len(epoch_loss)

    def inference(self, model):
        """ Returns the inference accuracy and loss.
        """
        model.eval()
        losses = []

        for batch_idx, (images, labels) in enumerate(self.validloader):
            images, labels = images.to(self.device), labels.to(self.device)

            # Inference
            outputs = model(images)
            batch_loss = self.criterion(outputs, labels)
            losses.append(batch_loss.item())

        return sum(losses) / len(losses) 


def test_inference(args, model, test_dataset):
    """ Returns the test accuracy and loss.
    """

    model.eval()
    test_losses = []
    device = "cuda:0" if args.gpu==0 else 'cpu'
    criterion = nn.MSELoss()
    testloader = DataLoader(test_dataset, batch_size=512,
                            shuffle=False)
    for (datas, labels) in testloader:
        datas, labels = datas.to(device), labels.to(device)

        # Inference
        outputs = model(datas)
        epoch_loss = criterion(outputs, labels)
        test_losses.append(epoch_loss.item())

    return sum(test_losses) / len(test_losses)
