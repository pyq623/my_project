import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class HAR70PlusDataset(Dataset):
    """HAR70+ Dataset Loader"""
    def __init__(self, root_dir, split='train', transform=None):
        self.root_dir = os.path.join(root_dir, 'har70plus')
        self.split = split
        self.transform = transform
        
        # Load data and labels
        self.X = np.load(os.path.join(self.root_dir, f'X_{split}.npy'))
        self.y = np.load(os.path.join(self.root_dir, f'y_{split}.npy'))
        
        # Load label dictionary
        with open(os.path.join(self.root_dir, 'har70plus.json'), 'r') as f:
            self.label_dict = json.load(f)['label_dictionary']
        
        # Convert to PyTorch tensors
        self.X = torch.from_numpy(self.X).float()  # [N, 500, 6]
        self.y = torch.from_numpy(self.y).long()

        # Add classes attribute from label_dict
        self.classes = list(self.label_dict.values())  # This is the key addition
        
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        x = self.X[idx]  # [500, 6]
        y = self.y[idx]
        
        if self.transform:
            x = self.transform(x)
            
        return x.permute(1, 0), y  # [6, 500]

class GenericDataset(Dataset):
    """通用数据集加载器"""
    def __init__(self, root_dir, dataset_name, split='train', transform=None):
        """
        初始化通用数据集加载器

        参数:
            root_dir (str): 数据集根目录
            dataset_name (str): 数据集名称（如 'har70plus', 'motionsense', 'whar', 'USCHAD', 'UTD-MHAD', 'WISDM'）
            split (str): 数据集划分（'train' 或 'test'）
            transform (callable, optional): 数据变换
        """
        self.root_dir = os.path.join(root_dir, dataset_name)
        self.split = split
        self.transform = transform
        
        # 加载数据和标签
        self.X = np.load(os.path.join(self.root_dir, f'X_{split}.npy'))
        self.y = np.load(os.path.join(self.root_dir, f'y_{split}.npy'))
        
        # 加载标签字典（如果存在）
        label_dict_path = os.path.join(self.root_dir, f'{dataset_name}.json')
        if os.path.exists(label_dict_path):
            with open(label_dict_path, 'r') as f:
                self.label_dict = json.load(f).get('label_dictionary', {})
            self.classes = list(self.label_dict.values())
        else:
            # 如果没有 JSON 文件，则用数字类
            self.label_dict = None
            self.classes = list(range(int(self.y.max().item() + 1)))
        
        # 转换为 PyTorch 张量
        self.X = torch.from_numpy(self.X).float()
        self.y = torch.from_numpy(self.y).long()
        
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        x = self.X[idx]
        y = self.y[idx]
        
        if self.transform:
            x = self.transform(x)
        
        # 转换为 [C, T] 格式
        return x.permute(1, 0), y  # 将 [T, C] 转换为 [C, T]


def get_multitask_dataloaders(root_dir, batch_size=1, datasets=None):
    """创建多任务数据加载器"""
    transform = transforms.Compose([
        transforms.Lambda(lambda x: x + torch.randn_like(x) * 0.01),  # 添加轻微噪声
    ])
    # data/DSADS
    if datasets is None:
        datasets = ['DSADS', 'har70plus', 'Harth', 'Mhealth', 'MMAct1', 'MotionSense', 'Opp_g', 'PAMAP', 'realworld', 'Shoaib', 'TNDA-HAR', 'UCIHAR', 'USCHAD', 'ut-complex', 'UTD-MHAD', 'w-HAR', 'Wharf', 'WISDM']

    # 创建数据加载器
    dataloaders = {}
    for dataset_name in datasets:
        train_dataset = GenericDataset(root_dir, dataset_name, split='train', transform=transform)
        test_dataset = GenericDataset(root_dir, dataset_name, split='test', transform=transform)
        
        dataloaders[dataset_name] = {
            'train': DataLoader(train_dataset, batch_size=batch_size, shuffle=True),
            'test': DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        }

    # # 创建数据集
    # datasets = {
    #     'har70plus': HAR70PlusDataset(root_dir, transform=transform),
    #     'motionsense': MotionSenseDataset(root_dir, transform=transform),
    #     'whar': WHARDataset(root_dir, transform=transform)
    # }
    
    # # 创建数据加载器
    # dataloaders = {
    #     name: {
    #         'train': DataLoader(ds, batch_size=batch_size, shuffle=True),
    #         'test': DataLoader(
    #             type(ds)(root_dir, split='test', transform=transform),  # Create same type of dataset
    #             batch_size=batch_size, 
    #             shuffle=False
    #         )
    #     }
    #     for name, ds in datasets.items()
    # }
    
    return dataloaders


# if __name__ == "__main__":
#     # 获取数据加载器
#     dataloaders = get_multitask_dataloaders('/root/tinyml/data')

#     # 查看数据集信息
#     print(f"HAR70+ 训练集样本数: {len(dataloaders['har70plus']['train'].dataset)}")
#     print(f"MotionSense 训练集样本数: {len(dataloaders['motionsense']['train'].dataset)}")
#     print(f"w-HAR 训练集样本数: {len(dataloaders['whar']['train'].dataset)}")

#     # 示例数据检查
#     sample, label = next(iter(dataloaders['har70plus']['train']))
#     print(f"样本形状: {sample.shape}")  # 应为 [batch, 6, 500]
#     print(f"标签: {label}")