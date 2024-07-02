【ICML-2023 FE-GNN】[Feature Expansion for Graph Neural Networks](https://proceedings.mlr.press/v202/sun23p/sun23p.pdf)
<img src="https://github.com/XiaShan1227/FE-GNN/assets/67092235/df136c26-7353-44ac-99b0-f1dbaa9ec936" alt="Image" width="650" height="500">

### 1.Environment
```bash
torch            1.13.1+cu116
torch_geometric  2.5.0
```

### 2.Run
```python
python main.py --dataset=Cora --exp_name=Cora
```

**注意：本实验和原论文数据集划分不一致，本实验每个类别选取20个样本做训练，所有类别随机选取500、1000个样本做验证、测试。**

Code Framework Reference: [FE-GNN](https://github.com/sajqavril/Feature-Extension-Graph-Neural-Networks)
