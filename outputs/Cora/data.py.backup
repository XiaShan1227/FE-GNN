#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: XiaShan
@Contact: 153765931@qq.com
@Time: 2024/6/22 17:03
"""
from torch_geometric.datasets import Planetoid


def load_dataset(args):
    """
    Name      nodes   features   edges   classes   train   val   test
    Cora      2708    1433       10556   7         140     500   1000
    CiteSeer  3327    3703       9104    6         120     500   1000
    PubMed    19717   500        88648   3         60      500   1000
    """
    dataset = Planetoid(root='./dataset/Planetoid', name=args.dataset)
    data = dataset[0]

    return data


if __name__ == '__main__':
    Cora = Planetoid(root='./dataset/Planetoid', name='Cora')
    print(Cora[0])
    print(Cora.train_mask.sum().item(), Cora.val_mask.sum().item(), Cora.test_mask.sum().item()) # 训练、验证及测试节点数

    CiteSeer = Planetoid(root='./dataset/Planetoid', name='CiteSeer')
    print(CiteSeer[0])
    print(CiteSeer.train_mask.sum().item(), CiteSeer.val_mask.sum().item(), CiteSeer.test_mask.sum().item())

    PubMed = Planetoid(root='./dataset/Planetoid', name='PubMed')
    print(PubMed[0])
    print(PubMed.train_mask.sum().item(), PubMed.val_mask.sum().item(), PubMed.test_mask.sum().item())
