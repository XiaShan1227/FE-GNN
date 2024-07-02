#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: XiaShan
@Contact: 153765931@qq.com
@Time: 2024/6/22 18:05
"""
import argparse
from texttable import Texttable


def parameter_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, help='Dataset to Use: [Cora, CiteSeer, PubMed]', required=True)
    parser.add_argument('--exp_name', type=str, default='Exp', help='Name of the experiment')
    parser.add_argument('--gpu_index', type=int, default=0, help='Index of GPU(set <0 to use CPU)')

    # 训练
    parser.add_argument('--optimizer', type=str, default='Adam', choices=['Adam', 'SGD'], help='Choose optimizer: Adam or SGD')
    parser.add_argument('--init_lr', type=float, default=0.01, help='Learning rate initialization')
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum of SGD')
    parser.add_argument('--weight_decay', type=float, default=0.0001, help='Weight decay of L2 penalty')
    parser.add_argument('--epochs', type=int, default=1000, help='Epochs of training')
    parser.add_argument('--runs', type=int, default=10, help='Runs to train')
    parser.add_argument('--patience', type=int, default=200, help='Patience for early stop')

    # 模型
    parser.add_argument('--nx', type=int, default=-1, help='Rank of singular value decomposition of node feature matrix, defaulting to -1: Use the node feature dimension')
    parser.add_argument('--nlx', type=int, default=-1, help='Rank of singular value decomposition of feature matrix, defaulting to -1: Use the node feature dimension')
    parser.add_argument('--nl', type=int, default=50, help='Rank of singular value decomposition of graph structure matrix, defaulting to 0: Do Not Use')
    parser.add_argument('--k', type=int, default=2)
    parser.add_argument("--operator", type=str, default='gpr', choices=['gcn', 'gpr', 'cheb', 'ours'])
    parser.add_argument('--nhid', type=int, default=64, help='Hidden dimension of feature transformation')
    parser.add_argument('--share_lx', action='store_true', default=False, help='Share the same W for different hops of lx') # 终端中存在输入即为True，如 python main.py --share_lx

    args = parser.parse_args()  # 解析命令行参数

    return args


class IOStream():
    """训练日志文件"""
    def __init__(self, path):
        self.file = open(path, 'a') # 附加模式：用于在文件末尾添加内容，如果文件不存在则创建新文件

    def cprint(self, text):
        # print(text)
        self.file.write(text + '\n')
        self.file.flush() # 确保将写入的内容刷新到文件中，以防止数据在缓冲中滞留

    def close(self):
        self.file.close()


def table_printer(args):
    """绘制参数表格"""
    args = vars(args) # 转成字典类型
    keys = sorted(args.keys()) # 按照字母顺序进行排序
    table = Texttable()
    table.set_cols_dtype(['t', 't']) # 列的类型都为文本(str)
    rows = [["Parameter", "Value"]] # 设置表头
    for k in keys:
        rows.append([k.replace("_", " ").capitalize(), str(args[k])]) # 下划线替换成空格，首字母大写
    table.add_rows(rows)
    return table.draw()