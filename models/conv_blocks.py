import torch
import torch.nn as nn
import numpy as np

# 设置随机数种子
SEED = 42  # 你可以选择任何整数作为种子
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

# def get_activation(activation_name):
#     """获取激活函数层"""
#     activations = {
#         'ReLU': nn.ReLU,
#         'ReLU6': nn.ReLU6,
#         'LeakyReLU': nn.LeakyReLU,
#         'Swish': nn.SiLU,  # PyTorch中的SiLU就是Swish
#         'Sigmoid': nn.Sigmoid,
#         'HardSigmoid': nn.Hardsigmoid,
#     }
#     return activations.get(activation_name, nn.ReLU)(inplace=True)
def get_activation(activation_name):
    """获取激活函数层"""
    activations = {
        'ReLU': lambda: nn.ReLU(inplace=True),
        'ReLU6': lambda: nn.ReLU6(),  # ReLU6 不支持 inplace 参数
        'LeakyReLU': lambda: nn.LeakyReLU(negative_slope=0.01, inplace=True),
        'Swish': lambda: nn.SiLU(),  # SiLU 不支持 inplace 参数
        'Sigmoid': lambda: nn.Sigmoid(),
        'HardSigmoid': lambda: nn.Hardsigmoid(),
    }
    return activations.get(activation_name, lambda: nn.ReLU(inplace=True))()


# ------------SE模块实现------------
class SEBlock(nn.Module):
    """通用的SE模块（支持可选的激活函数类型）"""
    def __init__(self, channel, se_ratio, se_activation='Sigmoid'):
        super().__init__()
        reduced_ch = int(channel * se_ratio)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(channel, reduced_ch, 1),
            nn.ReLU(inplace=True),  # 中间固定使用ReLU
            nn.Conv1d(reduced_ch, channel, 1),
            get_activation(se_activation)
        )
    
    def forward(self, x):
        return x * self.se(x)

# ------------卷积块实现------------
class DWSepConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, 
                 stride=1, has_se=False, se_ratio=0.25, activation='ReLU6', skip_connection=True):
        
        super().__init__()

        # Depthwise 1D卷积
        self.dw_conv = nn.Conv1d(
            in_channels, in_channels, kernel_size,
            stride=stride, padding=kernel_size//2, groups=in_channels
        )
        self.bn1 = nn.BatchNorm1d(in_channels)

        self.activation = get_activation(activation)

        # SE模块（1D全局平均池化）
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),  # 1D池化
            nn.Conv1d(in_channels, int(in_channels * se_ratio), 1),
            self.activation,
            nn.Conv1d(int(in_channels * se_ratio), in_channels, 1),
            nn.Sigmoid()
        ) if se_ratio > 0 else None
        
        # Pointwise 1D卷积
        self.pw_conv = nn.Conv1d(in_channels, out_channels, 1)
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        # 跳跃连接
        self.skip_connection = skip_connection and (stride == 1) and (in_channels == out_channels)
        
    def forward(self, x):
        out = self.dw_conv(x)
        out = self.bn1(out)
        out = self.activation(out)
        
        if self.se is not None:
            se = self.se(out)
            out = out * se
            
        out = self.pw_conv(out)
        out = self.bn2(out)
        return out

class MBConvBlock(nn.Module):
    # 类似实现MBConv块
    def __init__(self, in_channels, out_channels, kernel_size,
                 expansion=4, stride=1, has_se=False, se_ratio=0.25, 
                 activation='ReLU6', skip_connection=True):
        super().__init__()
        self.skip_connection = skip_connection and (stride == 1) and (in_channels == out_channels)
        hidden_dim = int(in_channels * expansion)
        
        # Expansion phase (1x1 Conv1D)
        self.expand = nn.Sequential(
            nn.Conv1d(in_channels, hidden_dim, 1),
            nn.BatchNorm1d(hidden_dim),
            get_activation(activation)
        ) if expansion != 1 else nn.Identity()
        
        # Depthwise 1D卷积
        self.dw_conv = nn.Conv1d(
            hidden_dim, hidden_dim, kernel_size,
            stride=stride, padding=kernel_size//2, groups=hidden_dim
        )
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.activation = get_activation(activation)
        
        # SE模块（1D全局平均池化）
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(hidden_dim, int(hidden_dim * se_ratio), 1),
            self.activation,
            nn.Conv1d(int(hidden_dim * se_ratio), hidden_dim, 1),
            nn.Sigmoid()
        ) if se_ratio > 0 else None
        
        # Pointwise 1D卷积
        self.pw_conv = nn.Conv1d(hidden_dim, out_channels, 1)
        self.bn2 = nn.BatchNorm1d(out_channels)
        
    def forward(self, x):
        identity = x
        
        out = self.expand(x)
        out = self.dw_conv(out)
        out = self.bn1(out)
        out = self.activation(out)
        
        if self.se is not None:
            se = self.se(out)
            out = out * se
            
        out = self.pw_conv(out)
        out = self.bn2(out)
        
        if self.skip_connection:
            out += identity
            
        return out



# ------------------- 新增模块实现 -------------------
class DpConvBlock(nn.Module):
    """纯Depthwise + Pointwise卷积（无SE模块）"""
    def __init__(self, in_channels, out_channels, kernel_size, has_se=False,
                 stride=1, activation='ReLU6'):
        super().__init__()
        # Depthwise卷积
        self.dw_conv = nn.Conv1d(
            in_channels, in_channels, kernel_size,
            stride=stride, padding=kernel_size//2, groups=in_channels
        )
        self.bn1 = nn.BatchNorm1d(in_channels)
        self.activation = get_activation(activation)
        
        # Pointwise卷积
        self.pw_conv = nn.Conv1d(in_channels, out_channels, 1)
        self.bn2 = nn.BatchNorm1d(out_channels)
        
    def forward(self, x):
        out = self.dw_conv(x)
        out = self.bn1(out)
        out = self.activation(out)
        out = self.pw_conv(out)
        out = self.bn2(out)
        return out

class SeSepConvBlock(nn.Module):
    """带SE模块的Depthwise Separable卷积"""
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, has_se=True, se_ratio=0.25, activation='ReLU6', 
                 se_activation='Sigmoid'):
        super().__init__()
        # Depthwise卷积
        self.dw_conv = nn.Conv1d(
            in_channels, in_channels, kernel_size,
            stride=stride, padding=kernel_size//2, groups=in_channels
        )
        self.bn1 = nn.BatchNorm1d(in_channels)
        self.activation = get_activation(activation)
        
        # SE模块
        self.se = SEBlock(in_channels, se_ratio, se_activation) if se_ratio > 0 else None
        
        # Pointwise卷积
        self.pw_conv = nn.Conv1d(in_channels, out_channels, 1)
        self.bn2 = nn.BatchNorm1d(out_channels)
        
    def forward(self, x):
        out = self.dw_conv(x)
        out = self.bn1(out)
        out = self.activation(out)
        
        if self.se is not None:
            out = self.se(out)
            
        out = self.pw_conv(out)
        out = self.bn2(out)
        return out

class SeDpConvBlock(nn.Module):
    """带SE模块的纯Depthwise卷积（无Pointwise）"""
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, has_se=True, se_ratio=0.25, activation='ReLU6',
                 se_activation='Sigmoid'):
        super().__init__()
        # Depthwise卷积（输入输出通道相同）
        self.dw_conv = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=kernel_size//2, groups=in_channels
        )
        self.bn = nn.BatchNorm1d(out_channels)
        self.activation = get_activation(activation)
        
        # SE模块（作用于输出通道）
        self.se = SEBlock(out_channels, se_ratio, se_activation) if se_ratio > 0 else None
        
    def forward(self, x):
        out = self.dw_conv(x)
        out = self.bn(out)
        out = self.activation(out)
        
        if self.se is not None:
            out = self.se(out)
            
        return out
