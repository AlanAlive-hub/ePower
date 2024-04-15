#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.7


import numpy as np



def epower_iid(dataset, num_users):
    """
    Sample I.I.D. client data
    :param dataset:
    :param num_users:
    :return: dict of col index
    """
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items,
                                             replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users

