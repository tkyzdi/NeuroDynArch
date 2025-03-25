import torch
import torch.nn as nn




class QuantileLoss(nn.Module):
    """分位数回归损失函数"""
    
    def __init__(self, quantiles=[0.1, 0.5, 0.9]):
        """
        初始化分位数回归损失函数
        
        参数:
            quantiles: 分位数列表，默认为[0.1, 0.5, 0.9]
        """
        super(QuantileLoss, self).__init__()
        self.quantiles = quantiles
        
    def forward(self, pred, target):
        """
        计算分位数回归损失
        
        参数:
            pred: 预测值 [batch_size, 1]
            target: 目标值 [batch_size, 1]
            
        返回:
            loss: 分位数回归损失
        """
        losses = []
        for q in self.quantiles:
            errors = target - pred
            q_loss = torch.max(q * errors, (q-1) * errors)
            losses.append(torch.mean(q_loss))
        
        return sum(losses) / len(losses)
        
class WeightedQuantileLoss(nn.Module):
    """带权重的分位数回归损失函数，特别适用于股票收益率这类不平衡数据"""
    
    def __init__(self, quantiles=[0.1, 0.5, 0.9], threshold=0.03, weight_factor=3.0):
        """
        初始化带权重的分位数回归损失函数
        
        参数:
            quantiles: 分位数列表，默认为[0.1, 0.5, 0.9]
            threshold: 高收益率样本的阈值，默认为0.03
            weight_factor: 高收益率样本的权重因子，默认为3.0
        """
        super(WeightedQuantileLoss, self).__init__()
        self.quantiles = quantiles
        self.threshold = threshold
        self.weight_factor = weight_factor
        
    def forward(self, pred, target):
        """
        计算带权重的分位数回归损失
        
        参数:
            pred: 预测值 [batch_size, 1]
            target: 目标值 [batch_size, 1]
            
        返回:
            loss: 带权重的分位数回归损失
        """
        # 计算样本权重
        weights = torch.ones_like(target)
        weights[torch.abs(target) > self.threshold] = self.weight_factor
        
        losses = []
        for q in self.quantiles:
            errors = target - pred
            q_loss = torch.max(q * errors, (q-1) * errors)
            # 应用权重
            weighted_q_loss = weights * q_loss
            losses.append(torch.mean(weighted_q_loss))
        
        return sum(losses) / len(losses)


    

