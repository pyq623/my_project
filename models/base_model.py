# /root/tinyml/models/base_model.py
import torch
import torch.nn as nn
# from .conv_blocks import DWSepConvBlock, MBConvBlock, get_activation
from .conv_blocks import (
    DWSepConvBlock, 
    MBConvBlock,
    DpConvBlock,
    SeSepConvBlock,
    SeDpConvBlock,
    get_activation
)
import numpy as np

# 设置随机数种子
SEED = 42  # 你可以选择任何整数作为种子
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

class TinyMLModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.stages = nn.ModuleList()
        self._build_model()
        # 添加 output_dim 属性
        self.output_dim = self._get_output_dim()
    
    def _get_output_dim(self):
        """计算模型的输出维度"""
        # 取最后一个stage的通道数
        return self.config["stages"][-1]["channels"]
        
    def _build_model(self):
        # 从配置中获取输入通道数，默认为6（兼容旧配置）
        in_channels = self.config.get("input_channels", 6)
        
        # 构建每个stage
        for stage_config in self.config["stages"]:
            out_channels = stage_config["channels"]
            blocks = []
            
            # 构建每个block
            for block_config in stage_config["blocks"]:
                # 公共参数
                common_args = {
                    'in_channels': in_channels,
                    'out_channels': out_channels,
                    'kernel_size': block_config["kernel_size"],
                    'stride': block_config.get("stride", 1),
                    'activation': block_config["activation"]
                }

                if block_config["type"] == "DWSepConv":
                    block = DWSepConvBlock(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=block_config["kernel_size"],
                        stride=block_config.get("stride", 1),
                        has_se=block_config.get("has_se", False),
                        se_ratio=block_config.get("se_ratios", 0),
                        activation=block_config["activation"],
                        skip_connection=block_config.get("skip_connection", True)
                    )
                    print(f"has_se: {block_config.get('has_se', False)}, se_ratio: {block_config.get('se_ratios', 0)} skip_connection: {block_config.get('skip_connection', True)}")
                elif block_config["type"] == "MBConv":
                    block = MBConvBlock(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=block_config["kernel_size"],
                        expansion=block_config["expansion"],
                        stride=block_config.get("stride", 1),
                        has_se=block_config.get("has_se", False),
                        se_ratio=block_config.get("se_ratios", 0),
                        activation=block_config["activation"],
                        skip_connection=block_config.get("skip_connection", True)
                    )
                    print(f"has_se: {block_config.get('has_se', False)}, se_ratio: {block_config.get('se_ratios', 0)} skip_connection: {block_config.get('skip_connection', True)}")
                elif block_config["type"] == "DpConv":
                    block = DpConvBlock(**common_args)
                elif block_config["type"] == "SeSepConv":
                    block = SeSepConvBlock(
                        **common_args,
                        has_se=True,  # 强制启用SE
                        se_ratio=block_config.get("se_ratio", 0.25),
                        se_activation=block_config.get("se_activation", "Sigmoid")
                    )
                elif block_config["type"] == "SeDpConv":
                    block = SeDpConvBlock(
                        **common_args,
                        has_se=True,  # 强制启用SE
                        se_ratio=block_config.get("se_ratio", 0.25),
                        se_activation=block_config.get("se_activation", "Sigmoid")
                    )
                else:
                    raise ValueError(f"Unknown block type: {block_config['type']}")
                
                blocks.append(block)
                in_channels = out_channels  # 更新输入通道数
            
            # 将多个block组合成一个stage
            stage = nn.Sequential(*blocks)
            self.stages.append(stage)
        
        # 分类头（1D全局平均池化 + 全连接）
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(in_channels, self.config.get("num_classes", 10))  # 默认为10类
        
    def forward(self, x):
        for stage in self.stages:
            x = stage(x)
        
        # 之前的代码
        # x = self.avgpool(x)  # (B, C, 1)
        # x = x.squeeze(-1)    # (B, C)

        # 自适应全局平均池化
        # 自适应全局平均池化
        x = nn.functional.adaptive_avg_pool1d(x, 1)  # [B, C, 1]
        x = x.squeeze(-1)  # 将最后一维的 1 去掉，变成 [B, C]

        # 调试打印
        # print(f"Before classifier - features shape: {x.shape}")
        return x        
