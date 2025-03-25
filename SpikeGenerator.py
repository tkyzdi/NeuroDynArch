import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple
from MetaSTDP import MetaSTDP,BCMMetaplasticity
from NanocolumnUnit import NanocolumnUnit,DynamicResourceAllocation

class FlashMoERouter(nn.Module):
    """闪速混合专家路由器
    
    实现动态专家容量分配的高效路由机制
    """
    def __init__(self, input_dim, num_experts):
        super().__init__()
        self.gate = nn.Linear(input_dim, num_experts, bias=False)
        
        # 动态专家容量分配
        self.capacity_net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        # 学习温度参数
        self.temperature = nn.Parameter(torch.ones(1) * 0.5)
        
        # 全局状态跟踪
        self.register_buffer('expert_usage', torch.zeros(num_experts))
        self.register_buffer('usage_decay', torch.tensor(0.99))
        
        # Top-K稀疏化参数
        self.k = 2  # 默认选择前2个专家
    
    def update_usage(self, routing_weights):
        """更新专家使用统计"""
        # 对批次维度求和，获取每个专家被使用的频率
        batch_usage = routing_weights.sum(dim=0)
        
        # 指数移动平均更新
        self.expert_usage = self.expert_usage * self.usage_decay + batch_usage * (1 - self.usage_decay)
    
    def forward(self, x):
        """前向计算路由权重"""
        # 计算基础路由得分
        scores = self.gate(x)  # [batch_size, num_experts]
        
        # 计算动态容量系数
        capacity = self.capacity_net(x)  # [batch_size, 1]
        
        # 负载均衡：基于历史使用情况进行惩罚
        load_balancing = self.expert_usage / (self.expert_usage.sum() + 1e-6)
        balanced_scores = scores - 0.1 * load_balancing.unsqueeze(0)
        
        # 应用容量系数和温度参数
        gated_scores = balanced_scores * capacity * torch.clamp(self.temperature, min=0.1)
        
        # 使用Top-K稀疏化代替Gumbel-Softmax，提高效率
        batch_size, num_experts = gated_scores.shape
        
        # 获取Top-K值和索引
        topk_values, topk_indices = torch.topk(gated_scores, k=self.k, dim=1)
        
        # 创建稀疏路由权重矩阵
        routing_weights = torch.zeros_like(gated_scores)
        
        # 使用scatter将topk值填充到对应位置
        routing_weights.scatter_(1, topk_indices, topk_values)
        
        # 归一化路由权重
        routing_weights = routing_weights / (routing_weights.sum(dim=1, keepdim=True) + 1e-6)
        
        # 更新使用统计
        self.update_usage(routing_weights.detach())
        
        return routing_weights
class MSLIFNeuron(nn.Module):
    """多尺度自适应漏积分放电神经元
    
    同时捕获快速和慢速时间尺度的神经元动态
    """
    def __init__(
        self, 
        threshold: float = 1.0,
        reset_mode: str = "subtract",
        tau_range: Tuple[float, float] = (5.0, 50.0),  # 时间常数范围(ms)
        num_scales: int = 3,                           # 时间尺度数量
        refractory_period: int = 0
    ):
        super().__init__()
        self.register_buffer('threshold', torch.tensor(threshold))
        self.reset_mode = reset_mode
        self.register_buffer('refractory_period', torch.tensor(refractory_period))
        
        # 生成多个时间尺度常数
        taus = torch.linspace(tau_range[0], tau_range[1], num_scales)
        self.register_buffer('taus', taus)
        
        # 时间尺度权重（可学习）
        self.scale_weights = nn.Parameter(torch.ones(num_scales) / num_scales)
        
        # 膜电位和不应期计数器
        self.register_buffer('membrane_potentials', None)  # 多尺度膜电位
        self.register_buffer('refractory_count', None)
        
        # 自适应阈值
        self.register_buffer('adaptive_threshold', None)
        self.register_buffer('threshold_decay', torch.tensor(0.95))
    
    def reset_state(self, batch_size: int, num_neurons: int, device=None):
        """重置神经元状态"""
        device = device or self.threshold.device
        self.membrane_potentials = torch.zeros(batch_size, num_neurons, len(self.taus), device=device)
        self.refractory_count = torch.zeros(batch_size, num_neurons, device=device)
        self.adaptive_threshold = torch.ones(batch_size, num_neurons, device=device) * self.threshold
    
    def forward(self, inputs: torch.Tensor, dt: float = 1.0, verbose: bool = False) -> torch.Tensor:
        """多尺度神经元脉冲生成"""
        # 保存原始形状以便输出相同形状
        original_shape = inputs.shape
        batch_size = original_shape[0]
        
        # 展平输入张量为二维 [batch_size, -1]
        if len(original_shape) > 2:
            inputs_flat = inputs.view(batch_size, -1)
        else:
            inputs_flat = inputs
            
        # 获取展平后的神经元数量
        num_neurons = inputs_flat.shape[1]
        device = inputs.device
        
        # 首次调用时初始化状态
        if self.membrane_potentials is None:
            self.reset_state(batch_size, num_neurons, device)
        # 当输入形状变化时重新初始化状态
        elif self.membrane_potentials.shape[1] != num_neurons:
            if verbose:
                print(f"输入形状变化，重置状态")
            self.reset_state(batch_size, num_neurons, device)
        # 当批次大小变化时重新初始化状态
        elif self.membrane_potentials.shape[0] != batch_size:
            if verbose:
                print(f"批次大小变化，重置状态")
            self.reset_state(batch_size, num_neurons, device)
        
        # 更新多尺度膜电位 (仅对不在不应期的神经元)
        active_mask = (self.refractory_count <= 0).float().unsqueeze(-1)
        
        # 计算每个时间尺度的衰减
        decays = torch.exp(-dt / self.taus)
        
        # 膜电位衰减 + 输入电流贡献 (对每个时间尺度)
        self.membrane_potentials = (
            self.membrane_potentials * decays.view(1, 1, -1) + 
            active_mask * inputs_flat.unsqueeze(-1)
        )
        
        # 加权融合多尺度膜电位
        weights = F.softmax(self.scale_weights, dim=0)
        weighted_potentials = torch.sum(self.membrane_potentials * weights, dim=2)
        
        # 生成脉冲（与自适应阈值比较）
        spikes_flat = (weighted_potentials >= self.adaptive_threshold).float()
        
        # 更新自适应阈值
        self.adaptive_threshold = self.adaptive_threshold * self.threshold_decay + spikes_flat * 0.1
        
        # 优化：重置膜电位，使用向量化操作代替循环
        spike_reset = spikes_flat.unsqueeze(-1)  # 扩展维度以匹配膜电位 [batch, neurons, 1]
        
        if self.reset_mode == "subtract":
            # 从所有时间尺度的膜电位中减去阈值
            self.membrane_potentials = self.membrane_potentials - (spike_reset * self.threshold)
        elif self.reset_mode == "zero":
            # 将放电神经元的膜电位置零
            self.membrane_potentials = self.membrane_potentials * (1.0 - spike_reset)
        
        # 设置不应期
        self.refractory_count = torch.maximum(
            self.refractory_count - 1,
            spikes_flat * self.refractory_period
        )
        
        # 将脉冲重塑为原始输入形状
        if len(original_shape) > 2:
            spikes = spikes_flat.view(original_shape)
        else:
            spikes = spikes_flat
            
        return spikes


class DynamicPlasticityModule(nn.Module):
    """动态可塑性模块
    
    整合STDP学习、BCM元可塑性和事件驱动计算
    """
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_nanocolumns: int = 8,
        stdp_params: dict = None,
        bcm_params: dict = None
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_nanocolumns = num_nanocolumns
        
        # 创建纳米柱单元
        self.nanocolumns = nn.ModuleList([
            NanocolumnUnit(input_dim, output_dim) 
            for _ in range(num_nanocolumns)
        ])
        
        # 创建STDP学习器
        if stdp_params is None:
            stdp_params = {
                "hyper_dim": 128,
                "min_tau_plus": 10.0,
                "max_tau_plus": 40.0,
                "min_tau_minus": 20.0,
                "max_tau_minus": 60.0,
                "min_A_plus": 0.001,
                "max_A_plus": 0.01,
                "min_A_minus": 0.001,
                "max_A_minus": 0.01
            }
        self.stdp_learners = nn.ModuleList([
            MetaSTDP(**stdp_params) 
            for _ in range(num_nanocolumns)
        ])
        
        # 创建BCM元可塑性调节器
        bcm_params = bcm_params or {}
        self.bcm_regulator = BCMMetaplasticity(**bcm_params)
        
        # 脉冲生成器
        self.input_spike_gen = MSLIFNeuron(threshold=0.8)
        self.output_spike_gen = MSLIFNeuron(threshold=0.8)
        
        # 动态资源分配
        self.resource_manager = DynamicResourceAllocation(num_nanocolumns)
        
        # 纳米柱路由器
        self.nanocolumn_router = FlashMoERouter(input_dim, num_nanocolumns)
        
        # 缓存上一次的对齐度得分
        self.register_buffer('cached_alignment_scores', None)
        self.alignment_cache_valid = False
    
    def reset_state(self, batch_size: int):
        """重置所有动态状态
        
        Args:
            batch_size: 批次大小
        """
        self.input_spike_gen.reset_state(batch_size, self.input_dim)
        self.output_spike_gen.reset_state(batch_size, self.output_dim)
        self.alignment_cache_valid = False
    
    def apply_stdp_learning(self, nanocolumn_idx, nanocolumn, output, input_spikes, output_spikes, batch_size, dt):
        """应用STDP学习（向量化批处理）
        
        Args:
            nanocolumn_idx: 纳米柱索引
            nanocolumn: 纳米柱实例
            output: 纳米柱输出
            input_spikes: 输入脉冲 [batch, time, ...]
            output_spikes: 输出脉冲 [batch, ...]
            batch_size: 批次大小
            dt: 时间步长
        """
        # 1. 获取脉冲事件指示器（布尔张量）
        has_pre_spike = (input_spikes.sum(dim=(1, 2)) > 0)  # [batch]
        has_post_spike = (output_spikes.sum(dim=1) > 0)     # [batch]
        
        # 2. 获取STDP学习器
        stdp_learner = self.stdp_learners[nanocolumn_idx]
        
        # 3. 计算平均活动
        activity = output.abs().mean()
        
        # 4. 更新BCM调节器
        self.bcm_regulator.update(activity, dt)
        bcm_mod = self.bcm_regulator.compute_plasticity_modulation(activity)
        
        # 5. 向量化处理所有批次样本
        for b in range(batch_size):
            # 只有在有脉冲事件时才更新
            if has_pre_spike[b] or has_post_spike[b]:
                # 更新STDP跟踪变量
                stdp_learner.update_traces(
                    pre_spike=has_pre_spike[b].item(),
                    post_spike=has_post_spike[b].item(),
                    dt=dt
                )
                
                # 计算权重变化
                dw = stdp_learner.compute_weight_change(
                    pre_spike=has_pre_spike[b].item(),
                    post_spike=has_post_spike[b].item()
                )
                
                # 更新纳米柱权重
                with torch.no_grad():
                    nanocolumn.synaptic_weights.data += dw * bcm_mod
    
    def forward(
        self, 
        x: torch.Tensor, 
        prev_activity: torch.Tensor = None,
        learning_enabled: bool = True,
        dt: float = 1.0
    ) -> Dict[str, torch.Tensor]:
        """动态可塑性模块前向传播
        
        Args:
            x: 输入张量
            prev_activity: 前一时间步的活动
            learning_enabled: 是否启用学习
            dt: 时间步长(ms)
            
        Returns:
            outputs: 输出字典
        """
        batch_size = x.shape[0]
        device = x.device
        
        # 生成输入脉冲
        input_spikes = self.input_spike_gen(x, dt)
        
        # 根据输入计算纳米柱重要性
        importance_scores = self.nanocolumn_router(x).mean(dim=0)
        
        # 动态资源分配
        allocation_mask = self.resource_manager.allocate(importance_scores)
        
        # 激活的纳米柱处理输入
        nanocolumn_outputs = []
        alignment_scores = []
        
        # 如果缓存无效，重新计算所有对齐度
        if not self.alignment_cache_valid or self.cached_alignment_scores is None:
            compute_all_alignments = True
        else:
            compute_all_alignments = False
            
        for i, nanocolumn in enumerate(self.nanocolumns):
            if allocation_mask[i] > 0:
                # 对激活的纳米柱进行前向传播
                output, alignment = nanocolumn(x)
                nanocolumn_outputs.append(output)
                alignment_scores.append(alignment)
                
                # 应用STDP学习
                if learning_enabled:
                    # 生成输出脉冲
                    output_spikes = self.output_spike_gen(output, dt)
                    
                    # 使用向量化的STDP学习
                    self.apply_stdp_learning(
                        nanocolumn_idx=i,
                        nanocolumn=nanocolumn,
                        output=output,
                        input_spikes=input_spikes,
                        output_spikes=output_spikes,
                        batch_size=batch_size,
                        dt=dt
                    )
            else:
                # 未激活的纳米柱
                if compute_all_alignments:
                    # 仅计算对齐度，不执行完整前向传播
                    with torch.no_grad():
                        _, alignment = nanocolumn(x)
                    nanocolumn_outputs.append(torch.zeros(batch_size, self.output_dim, device=device))
                    alignment_scores.append(alignment)
                else:
                    # 使用缓存的对齐度
                    nanocolumn_outputs.append(torch.zeros(batch_size, self.output_dim, device=device))
                    alignment_scores.append(self.cached_alignment_scores[i])
        
        # 更新对齐度缓存
        if compute_all_alignments:
            self.cached_alignment_scores = torch.tensor(alignment_scores, device=device)
            self.alignment_cache_valid = True
        
        # 加权融合激活的纳米柱输出
        stacked_alignments = torch.stack(alignment_scores)
        weights = F.softmax(stacked_alignments, dim=0)
        
        stacked_outputs = torch.stack(nanocolumn_outputs)
        final_output = torch.sum(stacked_outputs * weights.unsqueeze(-1).unsqueeze(-1), dim=0)
        
        return {
            'output': final_output,
            'nanocolumn_outputs': nanocolumn_outputs,
            'alignment_scores': alignment_scores,
            'allocation_mask': allocation_mask,
            'input_spikes': input_spikes
        }