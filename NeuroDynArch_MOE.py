import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import copy
import time
from collections import defaultdict, deque
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
import math
from typing import List, Dict, Tuple, Optional, Union, Any

from NeuroDynArch import NeuroDynArch, EnergyConstrainedLoss, dynamic_inference, DynamicPlasticityModule
from loss import  QuantileLoss, WeightedQuantileLoss
from MetaSTDP import EphapticCoupling, DynamicResourceAllocation
from SpikeGenerator import MSLIFNeuron


    


class NeuroDynExpert(nn.Module):
    """基于NeuroDynArch的专家模块
    
    将生物启发的神经动态架构用作MoE中的专家单元，具有适应性学习能力
    """
    def __init__(
        self, 
        input_dim,
        output_dim=1,
        expert_id=None,
        shared_dim=128,
        num_modules=2,
        hidden_layer_configs=None,
        learning_rate=1e-4,
        max_train_epochs=50,
        loss_type='mse',
        quantiles=[0.1, 0.5, 0.9],
        quantile_threshold=0.03,
        quantile_weight_factor=3.0,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.expert_id = expert_id
        self.shared_dim = shared_dim
        self.num_modules = num_modules
        self.learning_rate = learning_rate
        self.max_train_epochs = max_train_epochs
        self.loss_type = loss_type
        self.quantiles = quantiles
        self.quantile_threshold = quantile_threshold
        self.quantile_weight_factor = quantile_weight_factor
        self.device = device
        
        # 初始化批次记忆
        self.batch_memory = []  # 存储专家训练过的数据批次
        
        # 初始化神经动态架构模型
        self.model = NeuroDynArch(
            input_dim=input_dim,
            output_dim=output_dim,
            shared_dim=shared_dim,
            num_modules=num_modules
        ).to(device)  # 创建后立即移动到设备上
        
        # 初始化脉冲生成器
        self.spike_generator = MSLIFNeuron(
            threshold=0.5,
            reset_mode="subtract",
            tau_range=(5.0, 20.0),  # 时间常数范围，替换 membrane_time_constant
            num_scales=3,           # 时间尺度数量
            refractory_period=2
        ).to(device)
        
        # 初始化损失函数
        if loss_type == 'mse':
            self.criterion = nn.MSELoss()
        elif loss_type == 'huber':
            self.criterion = nn.HuberLoss(delta=1.0)
        elif loss_type == 'quantile':
            self.criterion = QuantileLoss(quantiles=quantiles)
        elif loss_type == 'weighted_quantile':
            self.criterion = WeightedQuantileLoss(
                quantiles=quantiles,
                threshold=quantile_threshold,
                weight_factor=quantile_weight_factor
            )
        else:
            self.criterion = nn.MSELoss()  # 默认使用MSE损失
            
        # 初始化优化器
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
        # 初始化训练历史
        self.train_losses = []
        self.val_losses = []
        self.training_time = 0.0
    
    def clean_batch_memory(self):
        """清理批次记忆中的无效值"""
        if not hasattr(self, 'batch_memory'):
            self.batch_memory = []
            return
            
        cleaned_memory = []
        for memory in self.batch_memory:
            if 'mean' not in memory or 'std' not in memory:
                continue
                
            # 检查并清理NaN值
            if isinstance(memory['mean'], np.ndarray) and np.isnan(memory['mean']).any():
                memory['mean'] = np.nan_to_num(memory['mean'], nan=0.0)
                
            if isinstance(memory['std'], np.ndarray) and np.isnan(memory['std']).any():
                memory['std'] = np.nan_to_num(memory['std'], nan=0.0)
                
            cleaned_memory.append(memory)
            
        self.batch_memory = cleaned_memory
    
    def forward(self, features, return_activations=False, verbose=False):
        """专家前向推理
        
        Args:
            features: 输入特征张量，可能是多维张量（如图像）
            return_activations: 是否返回中间活动状态
            verbose: 是否打印详细日志
            
        Returns:
            输出预测值和可选的活动状态
        """
        # 检查输入形状并根据需要进行重塑
        original_shape = features.shape
        batch_size = original_shape[0]
        
        # 如果是多维输入（如图像），需要将其展平
        if len(original_shape) > 2:
            # 保存原始形状以便调试
            if not hasattr(self, 'last_input_shape') or self.last_input_shape != original_shape:
                self.last_input_shape = original_shape
                # 仅在形状变化时打印，避免过多输出
                if verbose:
                    print(f"输入形状变化：从 {getattr(self, 'last_input_shape', 'None')} 到 {original_shape}")
            
            # 展平为二维张量 [batch_size, -1]
            features_flat = features.view(batch_size, -1)
        else:
            features_flat = features
        
        # 生成输入脉冲序列
        if not hasattr(self, 'spike_state_initialized') or not self.spike_state_initialized:
            self.spike_generator.reset_state(batch_size, self.input_dim, features_flat.device)
            self.spike_state_initialized = True
            
        # 将输入转换为脉冲
        input_spikes = self.spike_generator(features_flat, dt=1.0, verbose=verbose)
        
        # 通过主模型
        outputs = self.model(features_flat, learning_enabled=False, verbose=verbose)
        
        # 更新活动历史
        if return_activations:
            return outputs['output'], {
                'core_features': outputs['core_features'],
                'mix_ratio': outputs['mix_ratio'],
                'input_spikes': input_spikes,
                'plasticity_outputs': outputs['plasticity_outputs'],
                'coupled_activities': outputs['coupled_activities']
            }
        
        return outputs['output']
    
    def compute_uncertainty(self, features):
        """计算预测的不确定性
        
        使用动态推理过程和脉冲活动估计预测的可信度
        
        Args:
            features: 输入特征张量
            
        Returns:
            uncertainty: 不确定性评分
        """
        # 获取脉冲活动
        batch_size = features.shape[0]
        if not hasattr(self, 'spike_state_initialized') or not self.spike_state_initialized:
            self.spike_generator.reset_state(batch_size, self.input_dim, features.device)
            self.spike_state_initialized = True
            
        # 生成输入脉冲
        input_spikes = self.spike_generator(features, dt=1.0)
        spike_activity = input_spikes.float().mean()
        
        # 使用动态推理获取多步结果
        outputs = dynamic_inference(
            self.model,
            features, 
            max_time_steps=5,
            confidence_threshold=0.99
        )
        
        # 使用神经生物学启发的不确定性指标
        energy_usage = outputs['energy_usage']  # 能量消耗
        time_steps = outputs['time_steps_used']  # 所需时间步
        
        # 获取纳米柱激活情况
        module_activations = []
        for module_out in outputs['plasticity_outputs']:
            if 'allocation_mask' in module_out:
                activation_ratio = module_out['allocation_mask'].float().mean()
                module_activations.append(activation_ratio)
            
        # 计算平均模块激活率
        mean_activation = sum(module_activations) / len(module_activations) if module_activations else 0.5
        
        # 计算神经元激活混乱度 (基于熵的概念)
        if 'coupled_activities' in outputs and outputs['coupled_activities']:
            activities = torch.stack(outputs['coupled_activities'])
            activity_probs = F.softmax(activities.abs().mean(dim=-1), dim=0)
            entropy = -torch.sum(activity_probs * torch.log(activity_probs + 1e-10))
            normalized_entropy = entropy / math.log(len(activities))  # 归一化到[0,1]
        else:
            normalized_entropy = 0.5
        
        # 基于生物启发的不确定性计算：
        # 1. 高能量消耗、长处理时间表示高复杂度/不确定性
        # 2. 低纳米柱激活率表示信息处理受限
        # 3. 高信息熵表示激活模式混乱
        uncertainty = (energy_usage * time_steps / 5.0) * (1.0 - mean_activation) * (0.5 + 0.5 * normalized_entropy)
        
        return uncertainty
    
    def train_expert(self, batch_features, batch_labels, num_epochs=None, verbose=False):
        """训练专家模型
        
        实现两阶段训练策略：
        1. 静态预训练阶段：训练静态基座，冻结动态模块
        2. 动态微调阶段：冻结静态基座，训练动态可塑性机制
        
        Args:
            batch_features: 训练特征批次
            batch_labels: 训练标签批次
            num_epochs: 训练轮数，若为None则使用默认值
            verbose: 是否显示详细训练日志
        
        Returns:
            train_stats: 训练统计信息字典
        """
        if num_epochs is None:
            num_epochs = self.max_train_epochs
            
        # 记录训练开始时间
        start_time = time.time()
        
        # 不再分割数据，直接使用全部批次数据进行训练
        # 转换为PyTorch张量并移动到设备
        if isinstance(batch_features, torch.Tensor):
            X_train = batch_features.clone().detach().to(self.device)
        else:
            X_train = torch.tensor(batch_features, dtype=torch.float32).to(self.device)
            
        if isinstance(batch_labels, torch.Tensor):
            y_train = batch_labels.clone().detach().to(self.device)
        else:
            y_train = torch.tensor(batch_labels, dtype=torch.float32).to(self.device)
        
        # 计算并验证批次统计信息（使用更安全的方法）
        batch_mean = X_train.mean(dim=0).clone().detach().cpu().numpy()
        # 使用unbiased=False来避免自由度不足问题，并添加一个小值避免零标准差
        batch_std = X_train.std(dim=0, unbiased=False).clone().detach().cpu().numpy() + 1e-8
        
        # 检查并处理NaN值
        if np.isnan(batch_mean).any() or np.isnan(batch_std).any():
            print(f"警告: 专家{self.expert_id}的批次统计信息包含NaN，将被替换为0")
            batch_mean = np.nan_to_num(batch_mean, nan=0.0)
            batch_std = np.nan_to_num(batch_std, nan=0.0)
        
        # 保存训练数据特征摘要
        self.batch_memory.append({
            'mean': batch_mean,
            'std': batch_std,
            'size': len(X_train)
        })
        
        # 阶段1: 静态预训练 (60% 的训练轮数)
        static_epochs = int(num_epochs * 0.6)
        
        # 冻结动态模块，训练静态基座
        self.model.freeze_static_core = False
        for module in self.model.plasticity_modules:
            for param in module.parameters():
                param.requires_grad = False
        
        if verbose:
            print(f"阶段1: 静态预训练 ({static_epochs} 轮)")
            
        for epoch in range(static_epochs):
            # 训练模式
            self.model.train()
            
            # 前向传播
            outputs = self.model(X_train, learning_enabled=False)
            predictions = outputs['output']
            
            # 计算损失
            if self.loss_type in ['quantile', 'weighted_quantile']:
                loss = self.criterion(predictions, y_train)
            else:
                loss = self.criterion(predictions.squeeze(), y_train.squeeze())
            
            # 反向传播和优化
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # 不再需要验证步骤，直接记录训练损失
            self.train_losses.append(loss.item())
            
            if verbose and (epoch + 1) % 5 == 0:
                print(f"Epoch {epoch+1}/{num_epochs}, "
                      f"Train Loss: {loss.item():.4f}")
        
        # 阶段2: 动态微调 (剩余训练轮数)
        dynamic_epochs = num_epochs - static_epochs
        
        # 冻结静态基座，训练动态模块
        self.model.freeze_static_core = True
        for module in self.model.plasticity_modules:
            for param in module.parameters():
                param.requires_grad = True
        
        if verbose:
            print(f"阶段2: 动态微调 ({dynamic_epochs} 轮)")
            
        for epoch in range(dynamic_epochs):
            # 训练模式
            self.model.train()
            
            # 前向传播 (启用学习)
            outputs = self.model(X_train, learning_enabled=True)
            predictions = outputs['output']
            
            # 使用能量约束损失
            energy_loss_fn = EnergyConstrainedLoss(
                task_loss_fn=self.criterion,
                energy_weight=0.05,
                sparsity_target=0.1
            )
            
            # 计算损失
            if self.loss_type in ['quantile', 'weighted_quantile']:
                total_loss, loss_components = energy_loss_fn(outputs, y_train)
            else:
                # 需要调整标签形状以匹配能量约束损失的期望
                reshaped_y = y_train.view(-1, 1) if y_train.dim() == 1 else y_train
                total_loss, loss_components = energy_loss_fn(outputs, reshaped_y)
            
            # 反向传播和优化
            self.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # 不再需要验证步骤，直接记录训练损失
            self.train_losses.append(loss_components['task_loss'].item())
            
            if verbose and (epoch + 1) % 5 == 0:
                print(f"Epoch {static_epochs+epoch+1}/{num_epochs}, "
                      f"Train Loss: {loss_components['task_loss'].item():.4f}, "
                      f"Energy Loss: {loss_components['energy_loss'].item():.4f}")
        
        # 训练后重置动态状态
        self.model.reset_dynamic_state()
        
        # 计算训练时间
        self.training_time = time.time() - start_time
        
        return {
            'train_losses': self.train_losses,
            'training_time': self.training_time,
            'expert_id': self.expert_id
        }

class NeuroDynMoE(nn.Module):
    """神经动态混合专家系统
    
    实现动态专家生成、高效路由和生物启发机制的混合专家模型
    """
    def __init__(
        self,
        input_size,
        hidden_size,
        output_size,
        expert_memory_size=1000,
        expert_hidden_size=128,
        expert_type='neurodyn',
        create_first_expert=True,
        expert_args={},
        meta_expert_args={},
        routing_hidden_size=64,
        device='cpu',
        memory_device=None,
        uncertainty_threshold=0.3,
        similarity_threshold=0.6,
        memory_update_rate=0.05,
        expert_lr=0.001,
        max_experts=50,
        router_init_temp=1.0,
        memory_update_strategy='moving_average',
        uncertainty_mode="default",
        enable_biological_mechanisms=True,
        modalities=None,
        expert_generation_mode='dynamic',
        dynamic_check_interval=100,
    ):
        super().__init__()
        
        # 主要配置参数
        self.input_size = input_size
        self.hidden_size = hidden_size  # 统一使用这个作为特征维度
        self.output_size = output_size
        self.expert_hidden_size = expert_hidden_size 
        self.expert_type = expert_type
        self.max_experts = max_experts
        self.expert_generation_mode = expert_generation_mode
        self.dynamic_check_interval = dynamic_check_interval
        
        # 验证特征维度一致性
        assert hidden_size % 2 == 0, "hidden_size必须是偶数以支持特征分割"
        self.feature_dim = hidden_size  # 明确定义特征维度
        
        # 设备配置
        self.device = device
        self.memory_device = memory_device if memory_device is not None else device
        
        # 路由配置
        self.router_init_temp = router_init_temp
        self.routing_hidden_size = routing_hidden_size
        
        # 专家生成和维护参数
        self.similarity_threshold = similarity_threshold
        self.memory_update_rate = memory_update_rate
        self.memory_update_strategy = memory_update_strategy
        self.expert_memory_size = expert_memory_size
        self.uncertainty_threshold = uncertainty_threshold
        self.uncertainty_mode = uncertainty_mode
        self.expert_lr = expert_lr
        
        # 生物启发机制开关
        self.enable_biological_mechanisms = enable_biological_mechanisms
        
        # 多模态配置
        self.modalities = modalities
        
        # 输入投影层 - 将原始输入转换为特征向量
        self.input_projection = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU()
        ).to(device)
        
        # 输出映射层 - 将特征向量映射到输出尺寸
        self.output_mapping = nn.Linear(hidden_size, output_size).to(device)
        
        # 初始化专家列表，在路由器初始化之前
        self.experts = nn.ModuleList()
        
        # 初始化专家记忆库
        self.expert_memories = {}
        
        # 专家路由器 - 现在可以安全地使用self.experts
        self.router = nn.Sequential(
            nn.Linear(hidden_size, routing_hidden_size),
            nn.LayerNorm(routing_hidden_size),
            nn.GELU(),
            nn.Linear(routing_hidden_size, 1 if len(self.experts) == 0 else len(self.experts), bias=False)  # 输出维度匹配专家数量
        ).to(device)
        
        # 路由温度参数
        self.register_parameter(
            'router_temperature', 
            nn.Parameter(torch.ones(1, device=device) * router_init_temp)
        )
        
        # 路由熵统计 (用于判断是否需要新专家)
        self.register_buffer('routing_entropy_history', torch.zeros(100, device=device))
        self.register_buffer('routing_entropy_ptr', torch.tensor(0, device=device))
        
        # 资源利用率跟踪
        self.register_buffer('resource_utilization', torch.zeros(1, device=device))
        
        # 不确定性趋势跟踪
        self.register_buffer('uncertainty_history', torch.zeros(100, device=device))
        self.register_buffer('uncertainty_ptr', torch.tensor(0, device=device))
        
        # 批次计数器
        self.register_buffer('batch_counter', torch.tensor(0, device=device))
        
        # 初始化专家活动历史记录
        self.history_length = 5
        # 使用register_buffer确保状态持久化，且形状统一
        self.register_buffer(
            'activity_history', 
            torch.zeros(1, self.history_length, 2, self.feature_dim, device=device)
        )
        self.register_buffer('activity_ptr', torch.tensor(0, device=device))
        
        # 创建突触外耦合层 - 初始化时确保特征维度一致
        self.ephaptic_coupling = EphapticCoupling(
            num_modules=2,  # 两个专家组之间的耦合
            feature_dim=self.feature_dim,  # 使用统一的特征维度
            coupling_strength=0.1,
            spatial_decay=2.0
        ).to(device)
        
        # 生物特性：维度统一的专家活动跟踪器
        self.expert_activity_tracker = torch.zeros(
            max_experts, 10, device=device
        )  # [num_experts, history_length]
        
        # 创建首个专家(如果需要)
        if create_first_expert:
            self._create_first_expert()
            
        # 设置合适的默认dropout和初始化
        self._initialize_params()
    
    def _initialize_params(self):
        """初始化模型参数
        
        设置权重初始化、dropout等超参数
        """
        # 权重初始化
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Xavier初始化线性层
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                # 初始化LayerNorm
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
                
        # 初始化路由器温度参数
        self.router_temp = nn.Parameter(torch.ones(1, device=self.device) * self.router_init_temp)
        
        # 初始化元专家 (用于新专家初始化)
        self.meta_expert = nn.Sequential(
            nn.Linear(self.feature_dim, self.expert_hidden_size),
            nn.GELU(),
            nn.Linear(self.expert_hidden_size, self.output_size)
        ).to(self.device)
        
        # 初始化路由网络
        self.routing_network = nn.Sequential(
            nn.Linear(self.feature_dim, self.routing_hidden_size),
            nn.LayerNorm(self.routing_hidden_size),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.routing_hidden_size, 1, bias=False)
        ).to(self.device)
        
        # 初始化专家计数器
        self.expert_counts = [0] * (self.max_experts if self.max_experts > 0 else 1)
        
        # 初始化专家映射列表
        self.expert_memories = [{} for _ in range(self.max_experts if self.max_experts > 0 else 1)]
        
        # 初始化专家性能字典
        self.expert_performance = {}
        
        # 能量统计
        self.register_buffer('energy_history', torch.zeros(100, device=self.device))
        self.register_buffer('energy_ptr', torch.tensor(0, device=self.device))
        
        # 记录专家使用频率
        self.register_buffer('expert_usage_counts', torch.zeros(self.max_experts, device=self.device))
    
    def _create_first_expert(self):
        """创建第一个专家"""
        if self.expert_type == 'neurodyn':
            expert = NeuroDynExpert(
                input_dim=self.feature_dim,  # 使用统一特征维度
                output_dim=self.output_size,
                expert_id=0,
                shared_dim=self.expert_hidden_size,
                device=self.device
            )
        else:
            # 其他专家类型的实现...
            expert = nn.Sequential(
                nn.Linear(self.feature_dim, self.expert_hidden_size),
                nn.ReLU(),
                nn.Linear(self.expert_hidden_size, self.output_size)
            )
        
        self.experts.append(expert.to(self.device))
        
        # 初始化此专家的记忆
        self.expert_memories[0] = {
            'centroid': torch.zeros(self.feature_dim, device=self.memory_device),
            'count': 0,
            'batch_assignments': [],
            'samples': torch.zeros(
                self.expert_memory_size, self.feature_dim, 
                device=self.memory_device
            ),
            'labels': torch.zeros(
                self.expert_memory_size, self.output_size, 
                device=self.memory_device
            ) if self.output_size > 1 else torch.zeros(
                self.expert_memory_size, 1, 
                device=self.memory_device
            ),
            'ptr': 0,
            'is_full': False
        }
    
    def reset_activity_history(self, batch_size):
        """重置活动历史，使用统一的特征维度"""
        # 确保活动历史使用正确的形状和维度
        self.activity_history = torch.zeros(
            batch_size, self.history_length, 2, self.feature_dim, 
            device=self.device
        )
        self.activity_ptr = torch.tensor(0, device=self.device)
        
        print(f"重置活动历史：新形状 {self.activity_history.shape}")
    
    def forward(self, x, return_weights=True, return_expert_preds=True, learning_enabled=False, verbose=False):
        """NeuroDynMoE前向传播
        
        Args:
            x: 输入张量 [batch_size, input_size]
            return_weights: 是否返回路由权重
            return_expert_preds: 是否返回各专家预测
            learning_enabled: 是否启用学习
            verbose: 是否打印详细信息
            
        Returns:
            outputs: 包含预测和内部状态的字典
        """
        batch_size = x.shape[0]
        
        # 如果活动历史形状不匹配当前批次大小，重置活动历史
        if self.activity_history.shape[0] != batch_size:
            if verbose:
                print(f"重置活动历史：新形状 {batch_size, self.history_length, 2, self.feature_dim}")
            self.reset_activity_history(batch_size)
        
        # 如果特征维度不一致，重新初始化耦合层
        if hasattr(self, 'ephaptic_coupling') and self.ephaptic_coupling.feature_dim != self.feature_dim:
            if verbose:
                print(f"重新创建耦合层：原始特征维度 {self.ephaptic_coupling.feature_dim}，实际特征维度 {self.feature_dim}")
            self.ephaptic_coupling = EphapticCoupling(
                num_modules=2,
                feature_dim=self.feature_dim,
                coupling_strength=0.1,
                spatial_decay=2.0
            ).to(self.device)
        
        # 检查专家数量，确保至少有一个专家
        if len(self.experts) == 0:
            self._create_first_expert()
        
        # 增加批次计数
        self.batch_counter += 1
        
        # 提取特征
        features = self.extract_features(x)
        
        # 分组专家 (便于后续分组处理)
        num_experts = len(self.experts)
        expert_groups = {
            'group1': list(range(0, num_experts, 2)),   # 偶数索引专家
            'group2': list(range(1, num_experts, 2))    # 奇数索引专家
        }
        
        # 计算每个专家的路由得分
        num_experts = len(self.experts)
        if num_experts == 0:
            self._create_first_expert()
            num_experts = 1

        # 检查路由器输出维度是否匹配专家数量
        if self.router[-1].out_features != num_experts:
            if verbose:
                print(f"调整路由器输出维度: {self.router[-1].out_features} -> {num_experts}")
            # 重新创建匹配当前专家数量的路由器
            new_router = nn.Sequential(
                nn.Linear(self.feature_dim, self.routing_hidden_size),
                nn.LayerNorm(self.routing_hidden_size),
                nn.GELU(),
                nn.Linear(self.routing_hidden_size, num_experts, bias=False)
            ).to(self.device)
            
            # 复制前面层的权重
            with torch.no_grad():
                new_router[0].weight.copy_(self.router[0].weight)
                new_router[0].bias.copy_(self.router[0].bias)
                new_router[1].weight.copy_(self.router[1].weight)
                new_router[1].bias.copy_(self.router[1].bias)
                
                # 复制输出层权重 (如果可能)
                if self.router[-1].out_features < num_experts:
                    # 扩展权重矩阵
                    new_router[-1].weight[:self.router[-1].out_features].copy_(self.router[-1].weight)
                elif self.router[-1].out_features > num_experts:
                    # 截断权重矩阵
                    new_router[-1].weight.copy_(self.router[-1].weight[:num_experts])
            
            # 更新路由器
            self.router = new_router
        
        # 获取路由得分
        router_logits = self.router(features)  # [batch_size, num_experts]
        
        # 确保router_logits维度正确
        if router_logits.dim() == 3:
            router_logits = router_logits.squeeze(-1)  # 处理 [batch_size, num_experts, 1] 情况
        
        # 使用温度参数调整路由得分并应用softmax
        # 确保使用router_temp而不是router_temperature
        if hasattr(self, 'router_temp'):
            temp = self.router_temp
        else:
            temp = self.router_temperature if hasattr(self, 'router_temperature') else torch.tensor([1.0], device=self.device)
            
        routing_weights = F.softmax(
            router_logits / torch.clamp(temp, min=0.1),
            dim=-1
        )  # [batch_size, num_experts]
        
        # 确保routing_weights是二维张量 [batch_size, num_experts]
        if routing_weights.dim() != 2 or routing_weights.shape[1] != num_experts:
            if verbose:
                print(f"路由权重维度不匹配: {routing_weights.shape}，重塑为 [{batch_size}, {num_experts}]")
            # 重新塑形或填充权重
            if routing_weights.dim() == 1:
                routing_weights = routing_weights.unsqueeze(1)  # [batch_size, 1]
            
            if routing_weights.shape[1] < num_experts:
                # 扩展权重
                padding = torch.zeros(batch_size, num_experts - routing_weights.shape[1], device=self.device)
                routing_weights = torch.cat([routing_weights, padding], dim=1)
                routing_weights = routing_weights / routing_weights.sum(dim=1, keepdim=True)
            elif routing_weights.shape[1] > num_experts:
                # 截断权重
                routing_weights = routing_weights[:, :num_experts]
                routing_weights = routing_weights / routing_weights.sum(dim=1, keepdim=True)
        
        # 存储专家预测和不确定性
        expert_outputs = []
        expert_uncertainties = []
        
        # 处理每个专家组
        group_features = []
        
        for group_name, expert_indices in expert_groups.items():
            if not expert_indices:  # 忽略空组
                continue
            
            # 获取此组的专家输出和不确定性
            group_outputs = []
            group_uncertainties = []
            
            for idx in expert_indices:
                # 越界检查
                if idx >= len(self.experts):
                    continue
                
                # 获取专家
                expert = self.experts[idx]
                
                try:
                    # 处理神经动态专家
                    if self.expert_type == 'neurodyn':
                        # 确保输入特征维度正确
                        if verbose and features.shape[1] != self.feature_dim:
                            print(f"警告：特征维度不匹配 - 输入特征维度: {features.shape[1]}, 期望维度: {self.feature_dim}")
                        
                        # 运行专家前向传播
                        expert_result = expert(features)
                        # 确保expert_result是字典类型
                        if isinstance(expert_result, dict):
                            expert_output = expert_result['output']
                        else:
                            # 如果直接返回的是张量而不是字典
                            expert_output = expert_result
                            
                        uncertainty = expert.compute_uncertainty(features)  # 计算不确定性
                    else:
                        # 处理标准专家
                        expert_output = expert(features)
                        uncertainty = torch.ones(batch_size, device=self.device) * 0.5  # 默认不确定性
                    
                    # 收集输出和不确定性
                    group_outputs.append(expert_output)
                    group_uncertainties.append(uncertainty)
                    
                except Exception as e:
                    print(f"专家 {idx} 前向计算错误: {e}")
                    # 出错时使用零张量作为备选
                    zero_output = torch.zeros(batch_size, self.output_size, device=self.device)
                    group_outputs.append(zero_output)
                    group_uncertainties.append(torch.ones(batch_size, device=self.device))
            
            if group_outputs:
                # 计算组合特征，并应用加权平均
                if len(group_outputs) > 0:
                    # 确保所有输出维度一致
                    group_outputs_fixed = []
                    for output in group_outputs:
                        # 确保输出是 [batch_size, output_size] 形状
                        if output.dim() == 1:
                            output = output.unsqueeze(1)
                        # 确保输出尺寸正确
                        if output.shape[1] != self.output_size:
                            if verbose:
                                print(f"调整输出维度: {output.shape} -> [{batch_size}, {self.output_size}]")
                            if output.shape[1] < self.output_size:
                                # 扩展维度
                                padding = torch.zeros(batch_size, self.output_size - output.shape[1], device=self.device)
                                output = torch.cat([output, padding], dim=1)
                            else:
                                # 截断维度
                                output = output[:, :self.output_size]
                        group_outputs_fixed.append(output)
                    
                    # 堆叠固定后的输出
                    group_stack = torch.stack(group_outputs_fixed, dim=1)  # [batch, num_experts, output_size]
                    
                    # 提取组平均特征 (用于突触外耦合)
                    group_avg = group_stack.mean(dim=1)  # [batch, output_size]
                    # 确保特征维度与模型期望一致
                    if group_avg.shape[1] != self.feature_dim:
                        # 创建填充特征向量
                        padded_feature = torch.zeros(batch_size, self.feature_dim, device=self.device)
                        # 复制可复制的部分
                        min_dim = min(group_avg.shape[1], self.feature_dim)
                        padded_feature[:, :min_dim] = group_avg[:, :min_dim]
                        group_features.append(padded_feature)
                    else:
                        group_features.append(group_avg)
                else:
                    # 如果组内没有专家，使用零向量
                    dummy_features = torch.zeros(batch_size, self.feature_dim, device=self.device)
                    group_features.append(dummy_features)
            
            # 收集所有专家输出
            expert_outputs.extend(group_outputs)
            expert_uncertainties.extend(group_uncertainties)
        
        # 应用突触外耦合(如果启用)
        if self.enable_biological_mechanisms and len(group_features) > 1:
            try:
                # 确保每个组特征向量维度一致
                for i, gf in enumerate(group_features):
                    if gf.shape[1] != self.feature_dim:
                        # 调整特征维度与self.feature_dim一致
                        if verbose:
                            print(f"调整组 {i} 特征维度: {gf.shape[1]} -> {self.feature_dim}")
                        if gf.shape[1] < self.feature_dim:
                            # 扩展维度
                            pad_size = self.feature_dim - gf.shape[1]
                            group_features[i] = torch.cat([
                                gf, 
                                torch.zeros(batch_size, pad_size, device=self.device)
                            ], dim=1)
                        else:
                            # 截断维度
                            group_features[i] = gf[:, :self.feature_dim]
                
                # 应用突触外耦合
                coupled_features = self.ephaptic_coupling(group_features)
                
                # 更新活动历史
                self.update_expert_activity_history(coupled_features)
            except Exception as e:
                print(f"突触外耦合错误: {e}")
                coupled_features = group_features  # 出错时不应用耦合
        else:
            coupled_features = group_features
    
        # 确保返回的输出维度匹配目标
        # 最终输出预测
        if expert_outputs:
            # 确保所有输出维度一致
            expert_outputs_fixed = []
            for output in expert_outputs:
                # 确保输出是 [batch_size, output_size] 形状
                if output.dim() == 1:
                    output = output.unsqueeze(1)
                # 确保输出尺寸正确
                if output.shape[1] != self.output_size:
                    if verbose:
                        print(f"调整输出维度: {output.shape} -> [{batch_size}, {self.output_size}]")
                    if output.shape[1] < self.output_size:
                        # 扩展维度
                        padding = torch.zeros(batch_size, self.output_size - output.shape[1], device=self.device)
                        output = torch.cat([output, padding], dim=1)
                    else:
                        # 截断维度
                        output = output[:, :self.output_size]
                expert_outputs_fixed.append(output)
                
            # 堆叠所有专家输出
            stacked_outputs = torch.stack(expert_outputs_fixed, dim=1)  # [batch, num_experts, output_size]
            
            if verbose:
                print(f"预测形状: {stacked_outputs[:, 0, :].shape}, 标签形状: [{batch_size}, {self.output_size}]")
                print(f"权重形状: {routing_weights.shape}, 专家数量: {num_experts}")
            
            # 确保路由权重维度正确
            if routing_weights.shape[1] != stacked_outputs.shape[1]:
                if verbose:
                    print(f"调整路由权重维度以匹配专家输出: {routing_weights.shape[1]} -> {stacked_outputs.shape[1]}")
                
                # 处理维度不匹配
                if routing_weights.shape[1] < stacked_outputs.shape[1]:
                    # 扩展路由权重
                    padding = torch.zeros(
                        batch_size, 
                        stacked_outputs.shape[1] - routing_weights.shape[1], 
                        device=self.device
                    )
                    routing_weights = torch.cat([routing_weights, padding], dim=1)
                    # 重新归一化
                    routing_weights = routing_weights / (routing_weights.sum(dim=1, keepdim=True) + 1e-10)
                else:
                    # 截断路由权重
                    routing_weights = routing_weights[:, :stacked_outputs.shape[1]]
                    # 重新归一化
                    routing_weights = routing_weights / (routing_weights.sum(dim=1, keepdim=True) + 1e-10)
            
            # 计算加权平均 - 确保维度正确
            # 扩展为三维张量以便于乘法 [batch, num_experts, 1]
            expanded_weights = routing_weights.unsqueeze(-1)
            
            # 计算加权平均
            mixed_output = (stacked_outputs * expanded_weights).sum(dim=1)  # [batch, output_size]
            
            # 确保输出维度正确
            if mixed_output.shape[1] != self.output_size:
                if verbose:
                    print(f"调整最终输出维度: {mixed_output.shape} -> [{batch_size}, {self.output_size}]")
                if mixed_output.shape[1] < self.output_size:
                    # 扩展维度
                    padding = torch.zeros(batch_size, self.output_size - mixed_output.shape[1], device=self.device)
                    mixed_output = torch.cat([mixed_output, padding], dim=1)
                else:
                    # 截断维度
                    mixed_output = mixed_output[:, :self.output_size]
            
            final_output = mixed_output
        else:
            # 如果没有专家输出，使用零向量
            final_output = torch.zeros(batch_size, self.output_size, device=self.device)
        
        # 准备返回值
        outputs = {
            'output': final_output,
            'routing_weights': routing_weights if return_weights else None,
            'expert_outputs': expert_outputs_fixed if return_expert_preds else None,
            'expert_uncertainties': expert_uncertainties,
            'batch_counter': self.batch_counter
        }
        
        return outputs
    
    def extract_features(self, x):
        """提取输入特征
        
        Args:
            x: 输入数据
            
        Returns:
            router_features: 路由特征
        """
        x = x.to(self.device)
        
        # 使用输入投影层获取特征
        with torch.no_grad():
            features = self.input_projection(x)
        
        return features
    
    def add_expert(self, batch_features, batch_labels, batch_id=None, router_weights=None, uncertainties=None, verbose=False):
        """添加新专家
        
        根据专家生成模式创建新专家并添加到模型中
        
        Args:
            batch_features: 批次特征
            batch_labels: 批次标签
            batch_id: 批次ID
            router_weights: 路由权重
            uncertainties: 不确定性值
            verbose: 是否显示详细日志
            
        Returns:
            expert_id: 新专家ID
        """
        # 根据专家生成模式选择不同的创建策略
        if self.expert_generation_mode == 'dynamic':
            # 纯动态模式 - 仅基于系统状态触发
            if router_weights is not None:
                expert_id = self.dynamic_expert_creation(
                    batch_features, 
                    batch_labels, 
                    router_weights=router_weights,
                    uncertainties=uncertainties,
                    batch_id=batch_id,
                    verbose=verbose
                )
                return expert_id
            else:
                # 缺少动态创建所需参数时，回退到标准创建
                if verbose:
                    print("动态创建缺少路由权重，回退到标准创建")
            
        # 默认为fixed模式 - 直接创建新专家
        # 创建新专家
        expert_id = batch_id if batch_id is not None else len(self.experts)
        
        # 初始化新专家
        new_expert = NeuroDynExpert(
            input_dim=self.input_size,
            output_dim=self.output_size,
            expert_id=expert_id,
            shared_dim=self.expert_hidden_size,
            num_modules=2,
            learning_rate=self.expert_lr,
            device=self.device
        )
        
        # 训练新专家
        if verbose:
            print(f"训练新专家 {expert_id} (标准创建)")
        
        train_stats = new_expert.train_expert(batch_features, batch_labels, verbose=verbose)
        
        # 集成新专家到模型
        self._integrate_expert(new_expert, verbose=verbose)
        
        # 记录专家创建
        self.expert_counts.append(0)
        self.expert_memories.append([])
        
        # 记录专家性能
        self.expert_performance[expert_id] = {
            'train_loss': train_stats.get('loss', 0.0) if train_stats else 0.0,
            'creation_batch': batch_id,
            'trained_samples': len(batch_features),
            'creation_method': 'standard'
        }
        
        return expert_id
    
    def dynamic_expert_creation(self, batch_features, batch_labels, router_weights=None, uncertainties=None, batch_id=None, verbose=False):
        """动态创建新专家
        
        基于路由决策和不确定性趋势动态创建新专家
        
        Args:
            batch_features: 批次特征
            batch_labels: 批次标签
            router_weights: 路由器权重
            uncertainties: 不确定性估计
            batch_id: 批次ID
            verbose: 是否打印详细日志
    
        Returns:
            expert_id: 新创建的专家ID
        """
        # 当前专家数量
        current_num_experts = len(self.experts)
        
        if verbose:
            print(f"尝试创建第 {current_num_experts + 1} 个专家")
    
        # 提取特征
        features = self.extract_features(batch_features)
    
        # 创建新专家
        if verbose:
            print("创建新专家...")
        
        # 创建适应性专家，设置更长的训练周期
        new_expert_id = self._create_adaptive_expert(
            batch_features=batch_features,
            batch_labels=batch_labels,
            batch_id=batch_id,
            train_epochs=30,  # 增加训练轮数，避免训练不充分
            verbose=verbose
        )
    
        if new_expert_id is None:
            if verbose:
                print("专家创建失败")
            return None
        
        # 获取新专家
        new_expert = self.experts[new_expert_id]
    
        # 更新专家活动记录
        if self.activity_history is not None:
            # 检查并扩展活动历史以匹配当前专家数量
            if verbose:
                print(f"更新专家活动历史: 原专家数 {self.activity_history.shape[1]}，新专家数 {len(self.experts)}")
        
            if self.activity_history.shape[1] < len(self.experts):
                # 扩展活动历史张量
                feature_dim = self.activity_history.shape[-1]
                if feature_dim != features.shape[-1]:
                    if verbose:
                        print(f"特征维度不匹配: 活动历史 {feature_dim}, 当前特征 {features.shape[-1]}")
                    # 修复特征维度不匹配问题
                    self.reset_activity_history(batch_size=batch_features.shape[0])
                else:
                    # 扩展活动历史张量以包含新专家
                    device = self.activity_history.device
                    batch_size, _, steps, _ = self.activity_history.shape
                    
                    # 创建扩展张量
                    extended_history = torch.zeros(
                        batch_size, 
                        len(self.experts), 
                        steps, 
                        feature_dim, 
                        device=device
                    )
                    
                    # 复制原有活动
                    extended_history[:, :self.activity_history.shape[1]] = self.activity_history
                    self.activity_history = extended_history
                    
                    if verbose:
                        print(f"扩展活动历史形状: {self.activity_history.shape}")
    
        if verbose:
            print(f"新专家创建成功，ID: {new_expert_id}, 当前专家数: {len(self.experts)}")
    
        return new_expert_id
    
    def _create_adaptive_expert(self, batch_features, batch_labels, batch_id=None, train_epochs=25, verbose=False):
        """创建自适应专家
        
        基于当前批次数据创建新专家，并进行初始训练
        
        Args:
            batch_features: 批次特征
            batch_labels: 批次标签
            batch_id: 批次ID
            train_epochs: 训练轮数
            verbose: 是否打印详细日志
            
        Returns:
            expert_id: 新专家ID
        """
        if verbose:
            print(f"创建自适应专家 (训练轮数: {train_epochs})")
        
        # 为新专家创建唯一ID
        expert_id = len(self.experts)
        
        # 创建新专家
        expert_args = {}
        
        # 根据专家类型创建不同的专家
        if self.expert_type == 'mlp':
            # 创建MLP专家
            if verbose:
                print(f"创建MLP专家 (ID: {expert_id})")
                
            # 定义MLP专家类
            class MLPExpert(nn.Module):
                def __init__(
                    self, 
                    input_dim,
                    output_dim=1,
                    expert_id=None,
                    hidden_dims=[128, 64],
                    dropout=0.1,
                    learning_rate=1e-4,
                    device='cuda' if torch.cuda.is_available() else 'cpu'
                ):
                    super().__init__()
                    
                    self.input_dim = input_dim
                    self.output_dim = output_dim
                    self.expert_id = expert_id
                    self.device = device
                    
                    # 创建MLP网络
                    layers = []
                    prev_dim = input_dim
                    for hidden_dim in hidden_dims:
                        layers.append(nn.Linear(prev_dim, hidden_dim))
                        layers.append(nn.LayerNorm(hidden_dim))
                        layers.append(nn.ReLU())
                        layers.append(nn.Dropout(dropout))
                        prev_dim = hidden_dim
                    
                    # 输出层
                    layers.append(nn.Linear(prev_dim, output_dim))
                    
                    # 完整网络
                    self.model = nn.Sequential(*layers).to(device)
                    
                    # 批次记忆 - 保持与NeuroDynExpert接口兼容
                    self.batch_memory = []
                
                def forward(self, features, return_activations=False, verbose=False):
                    # 检查输入形状并重塑
                    batch_size = features.shape[0]
                    if len(features.shape) > 2:
                        features_flat = features.view(batch_size, -1)
                    else:
                        features_flat = features
                    
                    # 前向传播
                    output = self.model(features_flat)
                    
                    # 为保持与NeuroDynExpert兼容，返回带输出键的字典
                    return {'output': output}
                
                def compute_uncertainty(self, features):
                    # 简单返回常数不确定性
                    return torch.ones(features.shape[0], device=self.device) * 0.5
                
                def clean_batch_memory(self):
                    self.batch_memory = []
                
            # 创建MLP专家实例
            expert = MLPExpert(
                input_dim=self.input_size,
                output_dim=self.output_size,
                expert_id=expert_id,
                hidden_dims=[self.expert_hidden_size, self.expert_hidden_size // 2],
                device=self.device
            ).to(self.device)
            
        else:
            # 默认使用神经动态专家
            if verbose:
                print(f"创建NeuroDynExpert (ID: {expert_id})")
                
            expert = NeuroDynExpert(
                input_dim=self.input_size,
                output_dim=self.output_size,
                expert_id=expert_id,
                shared_dim=self.expert_hidden_size,
                device=self.device,
                **expert_args
            )

        # 初始训练专家
        self.train_expert_unified(
            expert=expert,
            batch_features=batch_features,
            batch_labels=batch_labels,
            train_epochs=train_epochs,
            verbose=verbose
        )
        
        # 扩展记忆库
        if hasattr(self, 'expert_memory'):
            # 为新专家初始化记忆
            self.expert_memory.append({
                'feature_mean': torch.zeros(self.input_size, device=self.device),
                'feature_std': torch.ones(self.input_size, device=self.device),
                'counter': 0,
                'last_batch_id': batch_id
            })
            
        if verbose:
            print(f"创建专家成功 (总数: {len(self.experts)})")
        
        return expert_id
    
    def train_expert_unified(self, expert, batch_features, batch_labels, train_epochs=25, verbose=False):
        """统一训练专家
        
        使用单一批次数据对专家进行训练
        
        Args:
            expert: 要训练的专家
            batch_features: 批次特征
            batch_labels: 批次标签
            train_epochs: 训练轮数
            verbose: 是否打印详细日志
            
        Returns:
            train_stats: 训练统计信息
        """
        # 确保批次维度正确
        if batch_features.shape[0] == 0 or batch_labels.shape[0] == 0:
            if verbose:
                print("警告: 空批次，跳过训练")
            return None
        
        # 检查一致性
        if batch_features.shape[0] != batch_labels.shape[0]:
            if verbose:
                print(f"警告: 批次不一致，特征: {batch_features.shape[0]}，标签: {batch_labels.shape[0]}")
            min_size = min(batch_features.shape[0], batch_labels.shape[0])
            batch_features = batch_features[:min_size]
            batch_labels = batch_labels[:min_size]
        
        # 确保标签维度正确
        if len(batch_labels.shape) == 1:
            batch_labels = batch_labels.unsqueeze(1)
        
        # 保存原始形状
        original_shape = batch_features.shape
        
        # 如果特征是高维的（如图像），则展平
        if len(original_shape) > 2:
            batch_size = original_shape[0]
            batch_features = batch_features.view(batch_size, -1)
        
        # 设置专家为训练模式
        expert.train()
        
        # 创建优化器
        optimizer = torch.optim.Adam(expert.parameters(), lr=0.001)
        
        # 创建损失函数
        criterion = nn.MSELoss()
        
        # 记录训练损失
        losses = []
        best_loss = float('inf')
        patience = 5  # 设置适当的早停耐心值
        patience_counter = 0
        best_state = None
        
        if verbose:
            print(f"开始训练专家，批次大小: {batch_features.shape[0]}，轮数: {train_epochs}")
            print(f"输入形状: {batch_features.shape}，标签形状: {batch_labels.shape}")
        
        # 训练循环
        for epoch in range(train_epochs):
            # 前向传播
            optimizer.zero_grad()
            outputs = expert(batch_features)
            
            # 处理不同类型专家的输出格式
            if isinstance(outputs, dict) and 'output' in outputs:
                # 对于NeuroDynExpert和MLPExpert
                outputs = outputs['output']
            
            # 确保输出和标签形状匹配
            if outputs.shape != batch_labels.shape:
                if outputs.ndim == 1:
                    outputs = outputs.unsqueeze(1)
                elif batch_labels.ndim == 1:
                    batch_labels = batch_labels.unsqueeze(1)
            
            # 计算损失
            loss = criterion(outputs, batch_labels)
            
            # 反向传播
            loss.backward()
            
            # 梯度裁剪，防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(expert.parameters(), max_norm=1.0)
            
            # 更新参数
            optimizer.step()
            
            # 记录损失
            current_loss = loss.item()
            losses.append(current_loss)
            
            # 检查是否有改进
            if current_loss < best_loss:
                best_loss = current_loss
                patience_counter = 0
                best_state = {k: v.clone() for k, v in expert.state_dict().items()}
            else:
                patience_counter += 1
            
            # 每5个epoch打印一次
            if verbose and (epoch+1) % 5 == 0:
                print(f"专家训练 - Epoch {epoch+1}/{train_epochs}, 损失: {current_loss:.6f}, 最佳: {best_loss:.6f}")
            
            # 早停检查 - 但确保至少训练10个epoch
            if patience_counter >= patience and epoch >= 10:
                if verbose:
                    print(f"早停触发: {patience}个epoch没有改善，当前epoch: {epoch+1}")
                break
        
        # 恢复最佳状态
        if best_state is not None:
            expert.load_state_dict(best_state)
            if verbose:
                print(f"恢复最佳模型状态，最终损失: {best_loss:.6f}")
        
        # 设置专家为评估模式
        expert.eval()
        
        # 添加到专家列表
        if expert not in self.experts:
            self.experts.append(expert)
            if verbose:
                print(f"专家已添加到模型 (当前专家总数: {len(self.experts)})")
        
        # 返回训练统计
        if verbose:
            print(f"专家训练完成，最终损失: {losses[-1]:.6f}, 最佳损失: {best_loss:.6f}")
            
        return {
            'losses': losses,
            'best_loss': best_loss,
            'total_epochs': len(losses)
        }
    
    def save_model(self, path):
        """保存模型到指定路径
        
        Args:
            path: 保存路径
        """
        state_dict = {
            'router': self.router.state_dict(),
            'meta_expert': self.meta_expert.state_dict(),
            'routing_network': self.routing_network.state_dict(),
            'num_experts': len(self.experts),
            'input_size': self.input_size,
            'output_size': self.output_size,
            'expert_hidden_size': self.expert_hidden_size,
            'expert_lr': self.expert_lr,
            'expert_type': self.expert_type,
            'uncertainty_threshold': self.uncertainty_threshold,
            'similarity_threshold': self.similarity_threshold,
            'memory_update_rate': self.memory_update_rate,
            'router_init_temp': self.router_temp.item(),
            'memory_update_strategy': self.memory_update_strategy,
            'uncertainty_mode': self.uncertainty_mode,
            'enable_biological_mechanisms': self.enable_biological_mechanisms
        }
        
        # 保存每个专家
        for i, expert in enumerate(self.experts):
            state_dict[f'expert_{i}'] = expert.state_dict()
            state_dict[f'expert_{i}_id'] = expert.expert_id
            state_dict[f'expert_{i}_batch_memory'] = expert.batch_memory
        
        torch.save(state_dict, path)
        print(f"模型已保存到 {path}")
    
    def load_model(self, path, map_location=None):
        """从指定路径加载模型
        
        Args:
            path: 加载路径
            map_location: 设备映射
        """
        if map_location is None:
            map_location = self.device
            
        state_dict = torch.load(path, map_location=map_location)
        
        # 重建模型基础参数
        self.input_size = state_dict['input_size']
        self.output_size = state_dict['output_size']
        self.expert_hidden_size = state_dict['expert_hidden_size']
        self.expert_lr = state_dict['expert_lr']
        self.expert_type = state_dict['expert_type']
        self.uncertainty_threshold = state_dict['uncertainty_threshold']
        self.similarity_threshold = state_dict['similarity_threshold']
        self.memory_update_rate = state_dict['memory_update_rate']
        self.router_temp = nn.Parameter(torch.tensor([state_dict['router_init_temp']]))
        self.memory_update_strategy = state_dict['memory_update_strategy']
        self.uncertainty_mode = state_dict['uncertainty_mode']
        self.enable_biological_mechanisms = state_dict['enable_biological_mechanisms']
        
        # 加载路由器和路由网络
        self.router.load_state_dict(state_dict['router'])
        self.routing_network.load_state_dict(state_dict['routing_network'])
        self.meta_expert.load_state_dict(state_dict['meta_expert'])
        
        # 重建专家列表
        num_experts = state_dict['num_experts']
        self.experts = nn.ModuleList([])
        
        for i in range(num_experts):
            expert_id = state_dict[f'expert_{i}_id']
            
            # 创建专家
            expert = NeuroDynExpert(
                input_dim=self.input_size,
                output_dim=self.output_size,
                expert_id=expert_id,
                shared_dim=self.expert_hidden_size,
                learning_rate=self.expert_lr,
                loss_type=self.expert_type,
                quantiles=[0.1, 0.5, 0.9],
                quantile_threshold=0.03,
                quantile_weight_factor=3.0,
                device=self.device
            )
            
            # 加载专家参数
            expert.load_state_dict(state_dict[f'expert_{i}'])
            
            # 恢复批次记忆
            if f'expert_{i}_batch_memory' in state_dict:
                expert.batch_memory = state_dict[f'expert_{i}_batch_memory']
                # 清理批次记忆中的任何无效值
                expert.clean_batch_memory()
                
            # 添加到专家列表
            self.experts.append(expert)
        
        # 确保整个模型在正确的设备上
        self.to(self.device)
        
        print(f"模型已从 {path} 加载，包含 {num_experts} 个专家")

    def _integrate_expert(self, expert, verbose=False):
        """集成专家到模型中
        
        添加专家到专家列表并更新路由器
        
        Args:
            expert: 新专家实例
            verbose: 是否显示详细日志
        """
        # 添加到专家列表
        self.experts.append(expert)
        
        # 获取最新的专家数量
        num_experts = len(self.experts)
        
        if verbose:
            print(f"专家集成: 当前专家数量 {num_experts}")
        
        # 更新路由器以适应新专家数量
        # 创建新的路由层，保留原始权重
        if hasattr(self, 'router') and isinstance(self.router, nn.Module):
            old_router = self.router
            
            # 创建新的路由器，添加对应新专家的输出单元
            new_router = nn.Sequential(
                nn.Linear(self.feature_dim, self.routing_hidden_size),
                nn.LayerNorm(self.routing_hidden_size),
                nn.GELU(),
                nn.Linear(self.routing_hidden_size, num_experts, bias=False)
            ).to(self.device)
            
            # 复制原始权重
            with torch.no_grad():
                # 复制第一个线性层
                new_router[0].weight.copy_(old_router[0].weight)
                new_router[0].bias.copy_(old_router[0].bias)
                
                # 复制LayerNorm层
                new_router[1].weight.copy_(old_router[1].weight)
                new_router[1].bias.copy_(old_router[1].bias)
                
                # 复制第二个线性层 - 有选择地复制现有专家的权重
                if num_experts > 1:  # 确保不是首个专家
                    old_out_layer = old_router[-1]
                    new_out_layer = new_router[-1]
                    
                    # 复制现有权重（对应旧专家的部分）
                    new_out_layer.weight[:num_experts-1].copy_(old_out_layer.weight)
                    
                    # 对新专家的权重进行随机初始化
                    nn.init.xavier_uniform_(new_out_layer.weight[num_experts-1:])
            
            # 更新路由器
            self.router = new_router
        
        # 确保其他相关数据结构也被更新
        # 例如，确保expert_usage_counts维度匹配专家数量
        if hasattr(self, 'expert_usage_counts'):
            old_counts = self.expert_usage_counts
            self.register_buffer('expert_usage_counts', 
                              torch.zeros(max(num_experts, self.max_experts), 
                                        device=self.device))
            
            # 复制旧计数
            if num_experts > 1:
                self.expert_usage_counts[:num_experts-1].copy_(old_counts[:num_experts-1])

class NeuroDynMoETrainer:
    """NeuroDynMoE模型训练器
    
    管理模型训练过程，包括专家生成、批次训练和评估
    """
    def __init__(
        self,
        model,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        max_experts=None,
        batch_interval=1,
        verbose=True
    ):
        self.model = model
        self.device = device
        self.max_experts = max_experts or model.max_experts
        self.batch_interval = batch_interval
        self.verbose = verbose
        
        # 训练统计信息
        self.batch_stats = {
            'losses': [],
            'expert_counts': [],
            'active_experts': [],
            'routing_weights': [],
            'batch_similarity': [],
            'new_expert_triggers': []
        }
        
        # 创建张量缓冲区用于状态跟踪
        self.register_buffers()
    
    def register_buffers(self):
        """注册持久性缓冲区用于状态跟踪"""
        # 使用模型设备确保一致性
        device = self.model.device
        
        # 记录每个专家的性能数据
        self.expert_performance = {}
        
        # 跟踪训练批次数量
        if not hasattr(self.model, 'batch_counter'):
            self.model.register_buffer('batch_counter', torch.tensor(0, device=device))
        
        # 用于专家生成决策的统计缓冲区
        if not hasattr(self.model, 'routing_entropy_history'):
            self.model.register_buffer('routing_entropy_history', torch.zeros(100, device=device))
            self.model.register_buffer('routing_entropy_ptr', torch.tensor(0, device=device))
        
        if not hasattr(self.model, 'uncertainty_history'):
            self.model.register_buffer('uncertainty_history', torch.zeros(100, device=device))
            self.model.register_buffer('uncertainty_ptr', torch.tensor(0, device=device))
    
    def train_on_batch(self, features, labels, batch_id=None, loss_type='mse', 
                       quantiles=[0.1, 0.5, 0.9], quantile_threshold=0.03, 
                       quantile_weight_factor=3.0):
        """训练单个批次
        
        Args:
            features: 输入特征 [batch_size, input_size]
            labels: 目标标签 [batch_size, output_size] 或 [batch_size]
            batch_id: 批次ID，如果为None则使用内部计数器
            loss_type: 损失函数类型，'mse'、'huber'、'quantile'或'weighted_quantile'
            quantiles: 分位数列表，用于quantile损失
            quantile_threshold: 分位数权重阈值
            quantile_weight_factor: 分位数权重因子
            
        Returns:
            stats: 训练统计信息
        """
        # 确保数据在正确的设备上
        features = features.to(self.device)
        labels = labels.to(self.device)
        
        # 使用模型的batch_counter作为批次ID
        if batch_id is None:
            batch_id = self.model.batch_counter.item()
        
        batch_size = features.shape[0]
        
        # 确保标签维度正确 - 统一始终使用两维形式 [batch_size, output_dim]
        if labels.ndim == 1:
            labels = labels.view(-1, 1)  # 将[batch_size]变为[batch_size, 1]
        
        # 重塑输入以适应模型
        if len(features.shape) > 2:
            flattened_features = features.view(batch_size, -1)
        else:
            flattened_features = features
        
        # 模型前向传播
        outputs = self.model(flattened_features, return_weights=True, return_expert_preds=True, learning_enabled=True)
        predictions = outputs['output']  # 改为使用'output'替代'predictions'
        weights = outputs['routing_weights']  # 使用'routing_weights'代替'weights'
        expert_outputs = outputs['expert_outputs']  # 使用'expert_outputs'代替'expert_preds'
        expert_uncertainties = outputs['expert_uncertainties']
        
        # 打印模型输出形状，用于调试
        if self.verbose and batch_id % 100 == 0:
            print(f"预测形状: {predictions.shape}, 标签形状: {labels.shape}")
            print(f"权重形状: {weights.shape}, 专家数量: {len(self.model.experts)}")
        
        # 确保预测和标签维度匹配
        if predictions.shape != labels.shape:
            if self.verbose and batch_id % 100 == 0:
                print(f"调整维度匹配! 预测: {predictions.shape}, 标签: {labels.shape}")
            
            if predictions.ndim == 1:
                predictions = predictions.view(-1, 1)
            elif predictions.shape[1] != labels.shape[1]:
                # 处理不同的输出维度
                if predictions.shape[1] > labels.shape[1]:
                    # 截断多余的维度
                    predictions = predictions[:, :labels.shape[1]]
                else:
                    # 扩展不足的维度
                    padding = torch.zeros(batch_size, labels.shape[1] - predictions.shape[1], device=self.device)
                    predictions = torch.cat([predictions, padding], dim=1)
        
        # 选择合适的损失函数
        if loss_type == 'mse':
            loss = F.mse_loss(predictions, labels)
        elif loss_type == 'huber':
            loss = F.huber_loss(predictions, labels, delta=1.0)
        elif loss_type == 'quantile':
            from loss import QuantileLoss
            criterion = QuantileLoss(quantiles=quantiles)
            loss = criterion(predictions, labels)
        elif loss_type == 'weighted_quantile':
            from loss import WeightedQuantileLoss
            criterion = WeightedQuantileLoss(
                quantiles=quantiles,
                threshold=quantile_threshold,
                weight_factor=quantile_weight_factor
            )
            loss = criterion(predictions, labels)
        else:
            # 默认使用MSE
            loss = F.mse_loss(predictions, labels)
        
        # 创建根据不确定性加权的损失
        if expert_uncertainties and len(expert_uncertainties) > 0:
            try:
                # 计算根据不确定性加权的损失，不确定性越高权重越低
                # 确保不确定性张量维度一致
                valid_uncertainties = []
                for unc in expert_uncertainties:
                    if unc.dim() == 1 and unc.shape[0] == batch_size:
                        valid_uncertainties.append(unc)
                    elif unc.dim() > 1 and unc.shape[0] == batch_size:
                        # 如果是多维的，取平均值
                        valid_uncertainties.append(unc.mean(dim=1))
                
                if valid_uncertainties:
                    stacked_uncertainties = torch.stack(valid_uncertainties, dim=1)
                    uncertainty_weights = 1.0 / (stacked_uncertainties + 1e-6)
                    normalized_uncertainty_weights = uncertainty_weights / uncertainty_weights.sum(dim=1, keepdim=True)
                    
                    # 创建一个不确定性感知的精细化损失
                    dynamic_loss = 0.0
                    valid_experts = 0
                    
                    if expert_outputs:
                        for i, expert_output in enumerate(expert_outputs):
                            if i < normalized_uncertainty_weights.shape[1]:
                                # 确保专家输出和标签维度匹配
                                if expert_output.shape != labels.shape:
                                    if expert_output.dim() == 1:
                                        expert_output = expert_output.view(-1, 1)
                                    elif expert_output.shape[1] != labels.shape[1]:
                                        # 处理输出维度不匹配
                                        if expert_output.shape[1] > labels.shape[1]:
                                            # 截断多余的维度
                                            expert_output = expert_output[:, :labels.shape[1]]
                                        else:
                                            # 扩展不足的维度
                                            padding = torch.zeros(
                                                batch_size, 
                                                labels.shape[1] - expert_output.shape[1], 
                                                device=self.device
                                            )
                                            expert_output = torch.cat([expert_output, padding], dim=1)
                                
                                # 专家损失乘以不确定性权重 - 使用更稳定的MSE损失计算
                                expert_loss = F.mse_loss(expert_output, labels, reduction='none')
                                
                                # 应用权重时加入数值稳定性处理
                                expert_weights = normalized_uncertainty_weights[:, i].unsqueeze(-1)
                                # 移除可能的NaN权重
                                expert_weights = torch.nan_to_num(expert_weights, nan=0.0, posinf=0.0, neginf=0.0)
                                
                                weighted_loss = (expert_loss * expert_weights).mean()
                                
                                # 仅当损失值合理时才加入总损失
                                if torch.isfinite(weighted_loss) and not torch.isnan(weighted_loss):
                                    dynamic_loss += weighted_loss
                                    valid_experts += 1
                    
                    # 仅当有有效专家时才使用动态损失
                    if valid_experts > 0:
                        # 使用更保守的权重混合总损失，减少波动
                        total_loss = 0.9 * loss + 0.1 * (dynamic_loss / valid_experts)
                    else:
                        total_loss = loss
                else:
                    total_loss = loss
            except Exception as e:
                # 发生任何错误时退回到基础损失
                if self.verbose:
                    print(f"动态损失计算错误: {e}，使用基础损失")
                total_loss = loss
        else:
            total_loss = loss
        
        # 专家生成逻辑
        create_new_expert = False
        trigger_reason = None
        
        # 固定间隔创建专家
        if (self.model.expert_generation_mode == 'fixed' and 
            batch_id > 0 and 
            batch_id % self.batch_interval == 0 and 
            len(self.model.experts) < self.max_experts):
            create_new_expert = True
            trigger_reason = "fixed_interval"
        
        # 动态触发创建专家
        elif (self.model.expert_generation_mode == 'dynamic' and 
              batch_id > 200 and  # 初始稳定期，确保有更多的数据积累
              len(self.model.experts) < self.max_experts):
            
            # 基于当前批次更新指标统计
            routing_entropy = -(weights * torch.log(weights + 1e-10)).sum(dim=1).mean().item()
            
            # 更新路由熵历史
            entropy_idx = int(self.model.routing_entropy_ptr.item())
            self.model.routing_entropy_history[entropy_idx] = routing_entropy
            # 使用in-place操作更新指针，避免直接用整数赋值
            self.model.routing_entropy_ptr[...] = (entropy_idx + 1) % self.model.routing_entropy_history.size(0)
            
            # 计算近期平均路由熵 - 使用更长的窗口
            window_size = min(100, entropy_idx + 1 if self.model.routing_entropy_ptr > 0 else self.model.routing_entropy_history.size(0))
            recent_entropy_mean = self.model.routing_entropy_history[-window_size:].mean().item()
            
            # 计算不确定性趋势
            if expert_uncertainties and len(expert_uncertainties) > 0:
                # 确保不确定性值是一维的
                valid_uncertainties = []
                for unc in expert_uncertainties:
                    if unc.dim() == 1:
                        valid_uncertainties.append(unc)
                    else:
                        # 如果是多维的，取平均值
                        valid_uncertainties.append(unc.mean())
                
                if valid_uncertainties:
                    avg_uncertainty = torch.stack(valid_uncertainties).mean().item()
                    # 更新不确定性历史值
                    uncertainty_idx = int(self.model.uncertainty_ptr.item())
                    self.model.uncertainty_history[uncertainty_idx] = avg_uncertainty
                    # 使用in-place操作更新指针，避免直接用整数赋值
                    self.model.uncertainty_ptr[...] = (uncertainty_idx + 1) % self.model.uncertainty_history.size(0)
            
            # 计算近期不确定性趋势 - 使用更长的窗口
            window_size = min(100, self.model.uncertainty_ptr.item() if self.model.uncertainty_ptr > 0 else self.model.uncertainty_history.size(0))
            if window_size > 0:
                history_slice = self.model.uncertainty_history[-window_size:]
                valid_values = history_slice[torch.isfinite(history_slice)]
                recent_uncertainty = valid_values.mean().item() if len(valid_values) > 0 else 0.0
            else:
                recent_uncertainty = 0.0
            
            # 专家资源利用率
            expert_usage = weights.max(dim=0)[0].mean().item()
            resource_utilization = expert_usage * len(self.model.experts) / self.max_experts
            
            # 动态触发专家创建条件 - 调整阈值使其更难触发
            entropy_threshold = 0.7  # 提高熵阈值
            uncertainty_threshold = 0.6  # 提高不确定性阈值
            resource_threshold = 0.85  # 提高资源阈值
            
            # 增加批次间隔检查 - 避免频繁创建专家
            if batch_id % 50 == 0:  # 每50个批次才检查一次动态触发条件
                if routing_entropy > entropy_threshold and expert_usage > 0.85:
                    create_new_expert = True
                    trigger_reason = "high_entropy"
                elif recent_uncertainty > uncertainty_threshold and expert_usage > 0.8:
                    create_new_expert = True
                    trigger_reason = "high_uncertainty"
                elif resource_utilization > resource_threshold:
                    create_new_expert = True
                    trigger_reason = "high_utilization"
        
        # 创建新专家
        if create_new_expert:
            if self.verbose:
                print(f"\n批次 {batch_id}: 创建新专家 (当前: {len(self.model.experts)}/{self.max_experts})")
            
            # 准备不确定性值
            uncertainty_tensor = None
            if expert_uncertainties and len(expert_uncertainties) > 0:
                try:
                    # 确保不确定性值是一维的
                    valid_uncertainties = []
                    for unc in expert_uncertainties:
                        if unc.dim() == 1 and unc.shape[0] == batch_size:
                            valid_uncertainties.append(unc)
                        elif unc.dim() > 1 and unc.shape[0] == batch_size:
                            # 如果是多维的，取平均值
                            valid_uncertainties.append(unc.mean(dim=1))
                    
                    if valid_uncertainties:
                        uncertainty_tensor = torch.stack(valid_uncertainties, dim=0).mean(dim=0)
                except Exception as e:
                    if self.verbose:
                        print(f"处理不确定性值错误: {e}")
            
            # 调用模型的动态专家创建方法
            expert_id = self.model.dynamic_expert_creation(
                batch_features=flattened_features,  # 使用已经重塑的特征
                batch_labels=labels,
                router_weights=weights,
                uncertainties=uncertainty_tensor,
                batch_id=batch_id,
                verbose=self.verbose
            )
            
            # 记录专家创建信息
            if expert_id is not None and trigger_reason:  # 确保expert_id非空且trigger_reason已定义
                self.batch_stats['new_expert_triggers'].append({
                    'batch_id': batch_id,
                    'expert_id': expert_id,
                    'reason': trigger_reason,
                    'expert_count': len(self.model.experts)
                })
        
        # 收集批次统计
        stats = {
            'loss': total_loss.item(),
            'raw_loss': loss.item(),  # 添加原始损失，方便对比
            'batch_id': batch_id,
            'num_experts': len(self.model.experts),
            'active_experts': torch.sum(weights > 0.05, dim=1).float().mean().item(),
            'routing_weights': weights.detach().cpu().numpy(),
        }
        
        # 更新统计历史
        self.batch_stats['losses'].append(stats['loss'])
        self.batch_stats['expert_counts'].append(stats['num_experts'])
        self.batch_stats['active_experts'].append(stats['active_experts'])
        
        # 增加批次计数
        self.model.batch_counter += 1
        
        return stats
    
    def train_on_dataloader(self, dataloader, max_batches=None, loss_type='mse', 
                           quantiles=[0.1, 0.5, 0.9], quantile_threshold=0.03, 
                           quantile_weight_factor=3.0):
        """在数据加载器上训练
        
        Args:
            dataloader: PyTorch数据加载器
            max_batches: 最大批次数，None表示处理整个数据集
            loss_type: 损失函数类型
            quantiles: 分位数列表
            quantile_threshold: 分位数权重阈值
            quantile_weight_factor: 分位数权重因子
            
        Returns:
            stats: 训练统计信息
        """
        all_stats = []
        
        # 设置进度条
        pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc="批次训练")
        
        # 打印损失函数信息
        if self.verbose:
            print(f"使用损失函数: {loss_type}")
            if loss_type in ['quantile', 'weighted_quantile']:
                print(f"分位数: {quantiles}")
            if loss_type == 'weighted_quantile':
                print(f"阈值: {quantile_threshold}, 权重因子: {quantile_weight_factor}")
        
        for batch_idx, (features, labels) in pbar:
            if max_batches is not None and batch_idx >= max_batches:
                break
            
            # 训练批次 - 传递损失函数参数
            batch_stats = self.train_on_batch(
                features, 
                labels, 
                batch_id=batch_idx,
                loss_type=loss_type,
                quantiles=quantiles,
                quantile_threshold=quantile_threshold, 
                quantile_weight_factor=quantile_weight_factor
            )
            all_stats.append(batch_stats)
            
            # 更新进度条
            pbar.set_postfix({
                'loss': f"{batch_stats['loss']:.4f}",
                'experts': f"{batch_stats['num_experts']}/{self.max_experts}"
            })
        
        # 汇总统计
        summary_stats = {
            'avg_loss': np.mean([s['loss'] for s in all_stats]),
            'final_experts': len(self.model.experts),
            'training_batches': len(all_stats)
        }
        
        return summary_stats

# 添加在NeuroDynMoETrainer类定义后面

def train_neurodyn_moe(
    model,
    train_loader,
    max_experts=50,
    verbose=True,
    max_batches=None,
    batch_interval=100,
    enable_coupling=True,
    expert_generation_mode='dynamic',
    dynamic_check_interval=100,
    loss_type='mse',
    quantiles=[0.1, 0.5, 0.9],
    quantile_threshold=0.03,
    quantile_weight_factor=3.0
):
    """训练神经动态混合专家模型
    
    Args:
        model: NeuroDynMoE模型实例
        train_loader: 训练数据加载器
        max_experts: 最大专家数量
        verbose: 是否显示详细日志
        max_batches: 最大训练批次数
        batch_interval: 固定间隔创建专家的批次间隔
        enable_coupling: 是否启用专家间耦合
        expert_generation_mode: 专家生成模式，'fixed'（固定间隔）或'dynamic'（动态触发）
        dynamic_check_interval: 动态专家创建检查间隔
        loss_type: 损失函数类型，'mse', 'huber', 'quantile', 'weighted_quantile'
        quantiles: 分位数列表，用于分位数损失函数
        quantile_threshold: 分位数权重阈值
        quantile_weight_factor: 分位数权重因子
        
    Returns:
        trainer: NeuroDynMoETrainer实例
    """
    # 确保模型设置正确
    model.expert_generation_mode = expert_generation_mode
    model.enable_biological_mechanisms = True
    
    # 创建训练器
    trainer = NeuroDynMoETrainer(
        model=model,
        device=next(model.parameters()).device,
        max_experts=max_experts,
        batch_interval=batch_interval,
        verbose=verbose
    )
    
    # 打印训练配置
    if verbose:
        print(f"开始训练NeuroDynMoE模型，配置:")
        print(f"- 最大专家数: {max_experts}")
        print(f"- 专家生成模式: {expert_generation_mode}")
        print(f"- 批次间隔: {batch_interval}")
        print(f"- 动态检查间隔: {dynamic_check_interval}")
        print(f"- 专家间耦合: {'启用' if enable_coupling else '禁用'}")
        print(f"- 损失函数类型: {loss_type}")
        if loss_type in ['quantile', 'weighted_quantile']:
            print(f"- 分位数: {quantiles}")
            if loss_type == 'weighted_quantile':
                print(f"- 分位数权重阈值: {quantile_threshold}")
                print(f"- 分位数权重因子: {quantile_weight_factor}")
    
    # 训练模型
    summary_stats = trainer.train_on_dataloader(
        dataloader=train_loader,
        max_batches=max_batches,
        loss_type=loss_type,
        quantiles=quantiles,
        quantile_threshold=quantile_threshold,
        quantile_weight_factor=quantile_weight_factor
    )
    
    if verbose:
        print(f"\n训练完成！结果概要:")
        print(f"- 平均损失: {summary_stats['avg_loss']:.6f}")
        print(f"- 最终专家数: {summary_stats['final_experts']}/{max_experts}")
        print(f"- 训练批次: {summary_stats['training_batches']}")
    
    return trainer

def evaluate_neurodyn_moe(
    model,
    val_loader,
    loss_type='mse',
    quantiles=[0.1, 0.5, 0.9],
    quantile_threshold=0.03,
    quantile_weight_factor=3.0,
    report_energy_metrics=False  # 默认关闭能量指标
):
    """评估神经动态混合专家模型
    
    Args:
        model: NeuroDynMoE模型实例
        val_loader: 验证数据加载器
        loss_type: 损失函数类型
        quantiles: 分位数列表
        quantile_threshold: 分位数权重阈值
        quantile_weight_factor: 分位数权重因子
        report_energy_metrics: 是否报告能量相关指标
        
    Returns:
        metrics: 评估指标字典，只包含基础指标
    """
    device = next(model.parameters()).device
    
    # 只初始化基础评估指标
    metrics = {
        'mse': 0.0,
        'mae': 0.0,
        'rmse': 0.0,
        'num_samples': 0
    }
    
    # 根据损失函数类型选择评估损失函数
    if loss_type == 'mse':
        criterion = nn.MSELoss(reduction='none')
    elif loss_type == 'huber':
        criterion = nn.HuberLoss(delta=1.0, reduction='none')
    elif loss_type == 'quantile':
        criterion = QuantileLoss(quantiles=quantiles)
    elif loss_type == 'weighted_quantile':
        criterion = WeightedQuantileLoss(
            quantiles=quantiles,
            threshold=quantile_threshold,
            weight_factor=quantile_weight_factor
        )
    else:
        criterion = nn.MSELoss(reduction='none')
    
    # 设置模型为评估模式
    model.eval()
    
    # 处理所有批次
    all_outputs = []
    all_labels = []
    
    with torch.no_grad():
        for batch_idx, (features, labels) in enumerate(val_loader):
            # 将数据移动到设备
            features = features.to(device)
            
            # 确保标签维度正确 - 统一为 [batch_size, output_dim]
            if labels.ndim == 1:
                labels = labels.unsqueeze(1)
            labels = labels.to(device)
            
            # 重塑输入以适应模型
            batch_size = features.shape[0]
            if len(features.shape) > 2:
                flattened_features = features.view(batch_size, -1)
            else:
                flattened_features = features
            
            # 前向传播 - 不需要返回路由权重和专家预测
            outputs = model(flattened_features, return_weights=False, return_expert_preds=False)
            
            # 提取结果
            preds = outputs['output']
            
            # 确保预测和标签具有相同的形状
            if preds.shape != labels.shape:
                if preds.ndim == 1:
                    preds = preds.unsqueeze(1)
                elif labels.ndim == 1:
                    labels = labels.unsqueeze(1)
            
            # 计算损失
            batch_losses = criterion(preds, labels)
            
            # 如果损失是一个张量(如MSE/Huber的reduction='none')，计算平均值
            if isinstance(batch_losses, torch.Tensor) and batch_losses.ndim > 0:
                batch_loss = batch_losses.mean().item()
            else:
                batch_loss = batch_losses.item()
            
            # 更新指标 - 只计算基础指标
            metrics['mse'] += F.mse_loss(preds, labels).item() * batch_size
            metrics['mae'] += F.l1_loss(preds, labels).item() * batch_size
            metrics['num_samples'] += batch_size
            
            # 收集数据用于后处理
            all_outputs.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    
    # 计算平均指标
    metrics['mse'] /= metrics['num_samples']
    metrics['mae'] /= metrics['num_samples']
    metrics['rmse'] = np.sqrt(metrics['mse'])
    
    return metrics

    def update_expert_activity_history(self, coupled_features):
        """更新专家活动历史
        
        记录最近的专家活动以支持后续分析和决策
        
        Args:
            coupled_features: 通过突触外耦合的特征列表
        """
        # 获取当前批次大小
        if not coupled_features or len(coupled_features) == 0:
            return
            
        batch_size = coupled_features[0].shape[0]
        
        # 如果活动历史尺寸不匹配，重新初始化
        if self.activity_history.shape[0] != batch_size:
            self.reset_activity_history(batch_size)
        
        # 获取当前指针位置
        ptr = int(self.activity_ptr.item())
        
        # 更新活动历史 - 仅更新两个组的历史，以保持简单
        for group_idx, group_features in enumerate(coupled_features[:2]):
            if group_idx < 2:  # 确保只使用前两个组
                # 计算均值校正后的活动
                self.activity_history[:, ptr, group_idx] = group_features
        
        # 更新指针 - 循环缓冲区
        ptr = self.activity_ptr.item()
        self.activity_ptr[...] = (ptr + 1) % self.history_length


