#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: XiaShan
@Contact: 153765931@qq.com
@Time: 2024/6/30 22:12
"""
from tqdm import tqdm
import os, time, torch, random
import torch.optim as optim
import torch.nn.functional as F

from data import load_dataset
from model import FE_GNN
from parameter import parameter_parser, IOStream, table_printer


def one_run(args, IO, seed, run, bar):
    random.seed(seed)  # 设置Python随机种子
    torch.manual_seed(seed)  # 设置PyTorch随机种子

    data = load_dataset(args)

    # 使用GPU or CPU
    device = torch.device('cpu' if args.gpu_index < 0 else 'cuda:{}'.format(args.gpu_index))
    if args.gpu_index < 0:
        IO.cprint('Using CPU')
    else:
        IO.cprint('Using GPU: {}'.format(args.gpu_index))
        torch.cuda.manual_seed(seed)  # 设置PyTorch GPU随机种子

    # 加载模型及参数量统计
    model = FE_GNN(args=args, ninput=data.x.shape[1], nclass=data.y.max()+1).to(device)
    IO.cprint(str(model))
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    IO.cprint('Model Parameter: {}'.format(total_params))

    # 优化器选择
    if args.optimizer == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=args.init_lr, weight_decay=args.weight_decay)
        IO.cprint('Using Adam')
    elif args.optimizer == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=args.init_lr, momentum=args.momentum, weight_decay=args.weight_decay)
        IO.cprint('Using SGD')

    start_time = time.time()

    best_epoch = 0 # 记录每次运行结果最好的epoch
    best_val_acc = 0
    best_test_acc = 0
    best_val_loss = 99999
    val_loss_history = []

    patience = 0

    for epoch in range(args.epochs):
        bar.set_description('Run:{:2d}, Epoch:{:4d}'.format(run, epoch))

        model.train()  # 训练模式
        data = data.to(device)

        optimizer.zero_grad()  # 梯度清0
        train_output = model(data)  # 前向传播

        train_loss = F.nll_loss(train_output[data.train_mask], data.y[data.train_mask])  # 计算损失
        train_loss.backward()  # 反向传播
        optimizer.step()  # 参数更新

        train_pred = train_output.argmax(dim=1)
        train_correct = (train_pred == data.y)
        train_acc = train_correct[data.train_mask].sum() / data.train_mask.sum()

        model.eval() # 评估模式
        eval_output = model(data)
        eval_pred = eval_output.argmax(dim=1)
        eval_correct = (eval_pred == data.y)

        val_loss = F.nll_loss(eval_output[data.val_mask], data.y[data.val_mask])
        val_acc = eval_correct[data.val_mask].sum() / data.val_mask.sum()
        test_loss = F.nll_loss(eval_output[data.test_mask], data.y[data.test_mask])
        test_acc = eval_correct[data.test_mask].sum() / data.test_mask.sum()

        print("\033[F\033[K", end='') # 清除上一行输出
        tqdm.write("Epoch {}: train_loss: {:.4f}, val_loss: {:.4f}, test_loss: {:.4f}, train_acc: {:.4f}, val_acc: {:.4f}, test_acc: {:.4f}".format(
                    epoch, train_loss.item(), val_loss.item(), test_loss.item(), train_acc.item(), val_acc.item(), test_acc.item()))

        IO.cprint('Epoch {}: train_loss={:.4f}, val_loss={:.4f}, test_loss={:.4f}, train_acc={:.4f}, val_acc={:.4f}, test_acc={:.4f}'.format(
                   epoch, train_loss.item(), val_loss.item(), test_loss.item(), train_acc.item(), val_acc.item(), test_acc.item()))

        val_loss_history.append(val_loss.item())

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            best_test_acc = test_acc
            best_epoch = epoch
            patience = 0
        else:
            patience +=1

        if patience > args.patience:
            runtime = time.time() - start_time
            epoch_time = runtime / (epoch + 1)
            IO.cprint('Average running time of epoch: {:.4f} seconds'.format(epoch_time))
            IO.cprint('Current epoch %d for run %d: val_loss: %.4f, val_acc: %.4f, test_acc %.4f'
                       % (epoch, run, val_loss, val_acc, test_acc))
            IO.cprint('Best epoch %d for run %d: val_loss: %.4f, val_acc: %.4f, test_acc %.4f'
                       % (best_epoch, run, best_val_loss, best_val_acc, best_test_acc))
            break

    return best_test_acc.item(), best_val_acc.item()


def exp_init():
    """实验初始化"""
    if not os.path.exists('outputs'):
        os.mkdir('outputs')
    if not os.path.exists('outputs/' + args.exp_name):
        os.mkdir('outputs/' + args.exp_name)

    # 跟踪执行脚本，windows下使用copy命令，且使用双引号
    os.system(f"copy main.py outputs\\{args.exp_name}\\main.py.backup")
    os.system(f"copy data.py outputs\\{args.exp_name}\\data.py.backup")
    os.system(f"copy model.py outputs\\{args.exp_name}\\model.py.backup")
    os.system(f"copy parameter.py outputs\\{args.exp_name}\\parameter.py.backup")
    # os.system('cp main.py outputs' + '/' + args.exp_name + '/' + 'main.py.backup')
    # os.system('cp data.py outputs' + '/' + args.exp_name + '/' + 'data.py.backup')
    # os.system('cp model.py outputs' + '/' + args.exp_name + '/' + 'model.py.backup')
    # os.system('cp parameter.py outputs' + '/' + args.exp_name + '/' + 'parameter.py.backup')


if __name__ == "__main__":
    args = parameter_parser()
    torch.set_num_threads(4)  # 设置线程数
    exp_init()

    IO = IOStream('outputs/' + args.exp_name + '/run.log')
    IO.cprint(str(table_printer(args)))  # 参数可视化

    for i in range(1):
        seeds = list(range(args.runs)) # 生成随机种子
        progress_bar = tqdm(range(args.runs))

        test_accs = []
        val_accs = []

        for idx in progress_bar:
            IO.cprint('')
            IO.cprint('Run:{:2d}'.format(idx))
            test_acc, val_acc = one_run(args, IO, seed=seeds[idx], run=idx, bar=progress_bar)
            test_accs.append(test_acc)
            val_accs.append(val_acc)

        test_acc_mean = torch.Tensor(test_accs).mean().item()
        val_acc_mean = torch.Tensor(val_accs).mean().item()

        IO.cprint('Average Accuracy for {:s}: Test_Acc:{:.4f}, Val_Acc:{:.4f}'.format(args.dataset, test_acc_mean, val_acc_mean))