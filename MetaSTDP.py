import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple


class MetaSTDP(nn.Module):
    """元学习STDP
    
    利用元学习动态调整STDP参数，适应不同的输入模式
    """
    def __init__(
        self, 
        hyper_dim: int = 128,
        min_tau_plus: float = 10.0,
        max_tau_plus: float = 40.0,
        min_tau_minus: float = 20.0,
        max_tau_minus: float = 60.0,
        min_A_plus: float = 0.001,
        max_A_plus: float = 0.01,
        min_A_minus: float = 0.001,
        max_A_minus: float = 0.01
    ):
        super().__init__()
        
        # 元学习网络生成STDP参数
        self.meta_net = nn.Sequential(
            nn.Linear(3, hyper_dim),  # 输入: [pre_act, post_act, time_diff]
            nn.LayerNorm(hyper_dim),
            nn.ReLU(),
            nn.Linear(hyper_dim, 4),  # 输出: [A_plus, A_minus, tau_plus, tau_minus]
            nn.Sigmoid()
        )
        
        # 参数范围
        self.register_buffer('min_tau_plus', torch.tensor(min_tau_plus))
        self.register_buffer('max_tau_plus', torch.tensor(max_tau_plus))
        self.register_buffer('min_tau_minus', torch.tensor(min_tau_minus))
        self.register_buffer('max_tau_minus', torch.tensor(max_tau_minus))
        self.register_buffer('min_A_plus', torch.tensor(min_A_plus))
        self.register_buffer('max_A_plus', torch.tensor(max_A_plus))
        self.register_buffer('min_A_minus', torch.tensor(min_A_minus))
        self.register_buffer('max_A_minus', torch.tensor(max_A_minus))
        
        # 跟踪变量
        self.register_buffer('pre_trace', torch.tensor(0.0))
        self.register_buffer('post_trace', torch.tensor(0.0))
        self.register_buffer('post_trace_y', torch.tensor(0.0))
        
        # 当前参数
        self.register_buffer('tau_plus', torch.tensor(20.0))
        self.register_buffer('tau_minus', torch.tensor(40.0))
        self.register_buffer('A_plus', torch.tensor(0.005))
        self.register_buffer('A_minus', torch.tensor(0.005))
        self.register_buffer('A_3', torch.tensor(0.005))
        
        # 上下文记忆
        self.register_buffer('context_buffer', torch.zeros(10, 3))  # 改用张量存储上下文
        self.register_buffer('context_buffer_ptr', torch.tensor(0))  # 环形缓冲区指针
        self.register_buffer('context_buffer_full', torch.tensor(False))  # 缓冲区是否已满
        
        # 参数更新计数器（事件驱动更新）
        self.register_buffer('update_counter', torch.tensor(0))
        self.update_interval = 10  # 每10个事件更新一次参数
    
    def update_params(self, pre_spike, post_spike, dt):
        """更新STDP参数（事件驱动）"""
        device = self.tau_plus.device
        
        # 获取当前上下文
        pre_act = float(self.pre_trace.item())
        post_act = float(self.post_trace.item())
        time_diff = dt
        
        # 更新上下文缓冲区（环形缓冲区实现）
        idx = self.context_buffer_ptr.item()
        self.context_buffer[idx, 0] = pre_act
        self.context_buffer[idx, 1] = post_act 
        self.context_buffer[idx, 2] = time_diff
        
        # 更新指针
        self.context_buffer_ptr[...] = (self.context_buffer_ptr.item() + 1) % self.context_buffer.size(0)
        if not self.context_buffer_full and self.context_buffer_ptr.item() == 0:
            self.context_buffer_full = torch.tensor(True, device=self.context_buffer_full.device)
            
        # 事件驱动更新：仅在累积足够事件时更新
        self.update_counter += 1
        if self.update_counter >= self.update_interval:
            self.update_counter.zero_()
            
            # 计算有效上下文数量
            num_contexts = self.context_buffer.size(0) if self.context_buffer_full else self.context_buffer_ptr
            
            # 如果有足够的上下文，则更新参数
            if num_contexts > 0:
                # 计算平均上下文
                if self.context_buffer_full:
                    context = self.context_buffer.mean(dim=0)
                else:
                    context = self.context_buffer[:num_contexts].mean(dim=0)
                
                # 元网络生成参数
                with torch.no_grad():
                    normalized_params = self.meta_net(context)
                    
                    # 映射到合适的参数范围
                    self.A_plus = self.min_A_plus + normalized_params[0] * (self.max_A_plus - self.min_A_plus)
                    self.A_minus = self.min_A_minus + normalized_params[1] * (self.max_A_minus - self.min_A_minus)
                    self.tau_plus = self.min_tau_plus + normalized_params[2] * (self.max_tau_plus - self.min_tau_plus)
                    self.tau_minus = self.min_tau_minus + normalized_params[3] * (self.max_tau_minus - self.min_tau_minus)
                    
                    # 动态更新A_3，让其与A_plus成比例但有波动
                    self.A_3 = self.A_plus * (0.8 + torch.rand(1, device=device) * 0.4)
            
    def update_traces(self, pre_spike: bool, post_spike: bool, dt: float = 1.0):
        """更新突触前后跟踪变量"""
        # 仅在有脉冲事件时更新参数
        if pre_spike or post_spike:
            self.update_params(pre_spike, post_spike, dt)
        
        # 指数衰减
        decay_plus = torch.exp(-dt / self.tau_plus)
        decay_minus = torch.exp(-dt / self.tau_minus)
        
        # 使用tau_plus的2倍作为三元组时间常数
        decay_y = torch.exp(-dt / (self.tau_plus * 2.0))
        
        # 更新跟踪变量
        self.pre_trace = self.pre_trace * decay_plus + (1.0 if pre_spike else 0.0)
        self.post_trace = self.post_trace * decay_minus + (1.0 if post_spike else 0.0)
        self.post_trace_y = self.post_trace_y * decay_y + (1.0 if post_spike else 0.0)
    
    def compute_weight_change(self, pre_spike: bool, post_spike: bool) -> torch.Tensor:
        """计算突触权重变化"""
        dw = torch.tensor(0.0, device=self.tau_plus.device)
        
        # 突触后脉冲导致的LTD (t_post < t_pre)
        if post_spike:
            dw -= self.A_minus * self.pre_trace
        
        # 突触前脉冲导致的LTP (t_pre < t_post)
        if pre_spike:
            # 标准STDP项
            dw += self.A_plus * self.post_trace
            
            # 三元组增强项 (考虑突触后神经元历史活动)
            dw += self.A_3 * self.post_trace * self.post_trace_y
            
        return dw


class BCMMetaplasticity(nn.Module):
    """BCM元可塑性调控机制
    
    实现Bienenstock-Cooper-Munro(BCM)理论的扩展版本，
    动态调整LTP/LTD阈值，实现对输入二阶相关性的不变学习
    """
    def __init__(
        self, 
        sliding_threshold_tau: float = 1000.0,  # 滑动阈值时间常数
        target_activity: float = 0.01,          # 目标平均活动率
        init_threshold: float = 0.5             # 初始可塑性阈值
    ):
        super().__init__()
        self.register_buffer('sliding_threshold_tau', torch.tensor(sliding_threshold_tau))
        self.register_buffer('target_activity', torch.tensor(target_activity))
        self.register_buffer('threshold', torch.tensor(init_threshold))
        
        # 追踪历史活动率
        self.register_buffer('avg_activity', torch.tensor(0.0))
        self.register_buffer('avg_activity_sq', torch.tensor(0.0))
    
    def update(self, current_activity: torch.Tensor, dt: float = 1.0):
        """更新BCM可塑性参数
        
        Args:
            current_activity: 当前神经元活动率
            dt: 时间步长
        """
        # 计算指数衰减因子
        decay = torch.exp(-dt / self.sliding_threshold_tau)
        
        # 更新平均活动率和平方活动率
        self.avg_activity = decay * self.avg_activity + (1 - decay) * current_activity
        self.avg_activity_sq = decay * self.avg_activity_sq + (1 - decay) * (current_activity ** 2)
        
        # 根据BCM理论调整可塑性阈值，与平均活动率的平方成正比
        target_ratio = self.avg_activity / self.target_activity
        self.threshold = self.avg_activity_sq / (self.target_activity * torch.clamp(target_ratio, 0.1, 10.0))
    
    def compute_plasticity_modulation(self, activity: torch.Tensor) -> torch.Tensor:
        """计算可塑性调制因子
        
        Args:
            activity: 当前活动率
            
        Returns:
            modulation: 可塑性调制因子，正值促进LTP，负值促进LTD
        """
        # BCM可塑性函数：activity * (activity - threshold)
        return activity * (activity - self.threshold)


class EphapticCoupling(nn.Module):
    """突触外耦合效应(Ephaptic Coupling)
    
    模拟离子凝胶纳米纤维的全局电势耦合，实现无物理连接的模块间信息协同
    """
    def __init__(
        self, 
        num_modules: int,
        feature_dim: int,
        coupling_strength: float = 0.1,
        spatial_decay: float = 2.0,
        kernel_size: int = 3
    ):
        super().__init__()
        self.num_modules = num_modules
        self.feature_dim = feature_dim
        
        # 耦合强度参数
        self.register_buffer('coupling_strength', torch.tensor(coupling_strength))
        self.register_buffer('spatial_decay', torch.tensor(spatial_decay))
        
        # 电场传播卷积核 (模拟电场扩散)
        self.field_propagation = nn.Conv1d(
            in_channels=1,
            out_channels=feature_dim,
            kernel_size=kernel_size,
            padding=kernel_size//2,
            bias=False
        )
        
        # 初始化为高斯分布权重
        with torch.no_grad():
            # 生成高斯核
            kernel = torch.exp(-torch.arange(-(kernel_size//2), kernel_size//2 + 1) ** 2 / (2 * spatial_decay ** 2))
            kernel = kernel / kernel.sum()  # 归一化
            # 扩展到所有输出特征维度
            kernel = kernel.repeat(feature_dim, 1, 1)
            self.field_propagation.weight.data = kernel
            
        # 模块间距离矩阵 (用于计算电场衰减)
        distances = torch.zeros(num_modules, num_modules)
        for i in range(num_modules):
            for j in range(num_modules):
                distances[i, j] = abs(i - j)
        self.register_buffer('distance_matrix', distances)
    
    def forward(self, module_activities: List[torch.Tensor]) -> List[torch.Tensor]:
        """计算模块间的突触外耦合效应
        
        Args:
            module_activities: 各模块的活动向量列表
            
        Returns:
            coupled_activities: 考虑耦合效应后的活动向量列表
        """
        batch_size = module_activities[0].shape[0]
        device = module_activities[0].device
        
        # 堆叠所有模块活动，形状: [batch_size, num_modules, feature_dim]
        stacked_activities = torch.stack(module_activities, dim=1)
        
        # 计算各模块产生的电场
        # 重塑为卷积输入格式 - 将特征维度作为batch维度的一部分，让通道维度为1
        # [batch_size * num_modules, 1, feature_dim] 而不是 [batch_size * num_modules, feature_dim, 1]
        reshaped = stacked_activities.view(batch_size * self.num_modules, 1, self.feature_dim)
        
        # 应用电场传播 - 输出形状 [batch_size * num_modules, feature_dim, feature_dim]
        propagated = self.field_propagation(reshaped)
        
        # 计算每个输出特征的平均影响
        propagated = propagated.mean(dim=2)  # [batch_size * num_modules, feature_dim]
        
        # 还原形状 [batch_size, num_modules, feature_dim]
        electric_fields = propagated.view(batch_size, self.num_modules, self.feature_dim)
        
        # 计算距离衰减系数 e^(-d/λ)
        decay_coeffs = torch.exp(-self.distance_matrix / self.spatial_decay)
        
        # 应用耦合效应: 原活动 + 耦合强度 * 距离加权电场和
        coupled_activities = []
        
        for i in range(self.num_modules):
            # 获取其他模块对当前模块的影响
            weighted_fields = torch.zeros_like(stacked_activities[:, 0])
            
            for j in range(self.num_modules):
                if i != j:  # 排除自身
                    # 距离衰减的场强影响
                    influence = decay_coeffs[i, j] * electric_fields[:, j]
                    weighted_fields += influence
            
            # 原活动 + 耦合影响
            coupled = module_activities[i] + self.coupling_strength * weighted_fields
            coupled_activities.append(coupled)
        
        return coupled_activities


class DynamicResourceAllocation(nn.Module):
    """动态资源分配机制
    
    实现基于短时程可塑性模型的资源耗尽与恢复机制，动态关闭无关参数路径
    """
    def __init__(
        self, 
        num_resources: int,
        recovery_rate: float = 0.01,   # 资源恢复速率
        depletion_rate: float = 0.1,   # 资源消耗速率
        baseline_resources: float = 0.6  # 基础资源水平
    ):
        super().__init__()
        self.num_resources = num_resources
        
        # 资源参数
        self.register_buffer('recovery_rate', torch.tensor(recovery_rate))
        self.register_buffer('depletion_rate', torch.tensor(depletion_rate))
        
        # 初始化资源水平为基础值
        self.register_buffer(
            'resource_levels', 
            torch.ones(num_resources) * baseline_resources
        )
        
        # 资源分配策略网络
        self.allocation_policy = nn.Sequential(
            nn.Linear(num_resources * 2, 128),
            nn.ReLU(),
            nn.Linear(128, num_resources),
            nn.Sigmoid()
        )
    
    def update_resources(self, activity_levels: torch.Tensor):
        """更新资源水平
        
        Args:
            activity_levels: 各资源单元的活动水平 [0,1]范围
        """
        # 资源消耗: 与活动水平成正比
        depletion = self.depletion_rate * activity_levels
        
        # 资源恢复: 与剩余可恢复空间成正比
        recovery = self.recovery_rate * (1.0 - self.resource_levels)
        
        # 更新资源水平
        self.resource_levels = torch.clamp(
            self.resource_levels + recovery - depletion,
            min=0.0, 
            max=1.0
        )
    
    def allocate(self, importance_scores: torch.Tensor) -> torch.Tensor:
        """基于重要性和资源水平分配计算资源
        
        Args:
            importance_scores: 各计算单元的重要性得分 [0,1]范围
            
        Returns:
            allocation_mask: 资源分配掩码，1表示分配资源，0表示休眠
        """
        # 将重要性和资源水平连接作为分配策略的输入
        policy_input = torch.cat([importance_scores, self.resource_levels])
        
        # 计算资源分配概率
        allocation_probs = self.allocation_policy(policy_input)
        
        # 基于资源水平约束分配概率
        constrained_probs = allocation_probs * self.resource_levels
        
        # 生成二值分配掩码 (硬阈值决策)
        allocation_mask = (constrained_probs > 0.5).float()
        
        # 更新资源消耗
        self.update_resources(allocation_mask)
        
        return allocation_mask