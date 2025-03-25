import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Dict, List, Tuple, Optional, Union
from MetaSTDP import MetaSTDP, BCMMetaplasticity, DynamicResourceAllocation

class NanocolumnUnit(nn.Module):
    """跨突触纳米柱单元
    
    模拟突触前RIM蛋白簇与突触后PSD-95蛋白簇的空间匹配与对齐，
    实现突触级别的精细动态路由机制
    """
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        alignment_gain: float = 1.0,
        alignment_threshold: float = 0.3
    ):
        super().__init__()
        # 突触前RIM蛋白表征
        self.pre_synaptic_rim = nn.Parameter(torch.randn(input_dim))
        # 突触后PSD-95蛋白表征
        self.post_synaptic_psd95 = nn.Parameter(torch.randn(output_dim))
        # 纳米柱内AMPA受体基础权重
        self.synaptic_weights = nn.Parameter(torch.randn(output_dim, input_dim) * 0.01)
        
        # 自适应增益因子（受全局调控信号影响）
        self.register_buffer('alignment_gain', torch.tensor(alignment_gain))
        # 对齐度激活阈值
        self.register_buffer('alignment_threshold', torch.tensor(alignment_threshold))
        
        # 活动历史记录 (用于STDP)
        self.register_buffer('activity_history', torch.zeros(10))
        self.register_buffer('active', torch.tensor(False))
        
    def compute_alignment(self, presynaptic, postsynaptic):
        """
        计算前突触信号和后突触信号的对齐度，模拟受体亲和力机制
        
        Args:
            presynaptic (torch.Tensor): 前突触信号
            postsynaptic (torch.Tensor): 后突触信号
            
        Returns:
            float: 对齐分数 (0-1)
        """
        # 确保输入是一维向量
        presynaptic = presynaptic.view(-1)
        postsynaptic = postsynaptic.view(-1)
        
        # 如果维度不同，截取到相同长度
        min_length = min(presynaptic.size(0), postsynaptic.size(0))
        presynaptic = presynaptic[:min_length]
        postsynaptic = postsynaptic[:min_length]
        
        # 归一化向量，计算余弦相似度
        pre_norm = F.normalize(presynaptic, p=2, dim=0)
        post_norm = F.normalize(postsynaptic, p=2, dim=0)
        
        # 余弦相似度计算
        similarity = torch.sum(pre_norm * post_norm)
        
        # 应用增益和Sigmoid转换
        alignment_score = torch.sigmoid(similarity * self.alignment_gain)
        
        return alignment_score
    
    def forward(
        self, 
        input_signal: torch.Tensor, 
        calcium_signal: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """前向传播，计算动态调制的突触输出
        
        Args:
            input_signal: 输入信号张量
            calcium_signal: 细胞内钙信号，用于调节可塑性
            
        Returns:
            output: 纳米柱输出信号
            alignment_score: 当前纳米柱对齐度
        """
        # 计算当前纳米柱对齐度
        alignment_score = self.compute_alignment(self.pre_synaptic_rim, self.post_synaptic_psd95)
        
        # 如果对齐度低于阈值，则抑制此纳米柱的活动
        if alignment_score < self.alignment_threshold:
            self.active = torch.tensor(False, device=input_signal.device)
            # 返回零输出和对齐度
            return torch.zeros_like(input_signal), alignment_score
        
        # 纳米柱被激活
        self.active = torch.tensor(True, device=input_signal.device)
        
        # 应用动态权重调制：基础权重 * 对齐度得分
        effective_weights = self.synaptic_weights * alignment_score
        
        # 计算纳米柱输出
        output = F.linear(input_signal, effective_weights)
        
        # 更新活动历史（用于STDP学习）
        self.activity_history = torch.cat([self.activity_history[1:], output.mean().unsqueeze(0)])
        
        return output, alignment_score


class FunctionalColumn(nn.Module):
    """功能柱 - 由多个纳米柱组成的计算单元
    
    模拟大脑中的功能区域，多个纳米柱协同工作形成一个功能柱
    """
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        nanocolumn_config: Dict,
        num_layers: int = 3,
        columns_per_layer: int = 5,
        **kwargs
    ):
        """初始化功能柱
        
        Args:
            input_dim: 输入维度
            output_dim: 输出维度
            nanocolumn_config: 纳米柱配置参数
            num_layers: 功能柱中的层数
            columns_per_layer: 每层的纳米柱数量
        """
        super().__init__()
        self.num_layers = num_layers
        self.columns_per_layer = columns_per_layer
        
        # 创建多层纳米柱
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            # 第一层使用输入维度，中间层使用输出维度
            layer_input_dim = input_dim if i == 0 else output_dim
            
            # 创建当前层的纳米柱
            layer = nn.ModuleList([
                NanocolumnUnit(
                    input_dim=layer_input_dim,
                    output_dim=output_dim,
                    **nanocolumn_config
                )
                for _ in range(columns_per_layer)
            ])
            
            self.layers.append(layer)
        
        # 创建跨层连接 (模拟Neuroligin-Neurexin优先连接)
        self.cross_layer_connections = nn.ParameterDict()
        for i in range(num_layers - 2):
            for j in range(i + 2, num_layers):
                conn_name = f'layer_{i}_to_{j}'
                self.cross_layer_connections[conn_name] = nn.Parameter(
                    torch.randn(columns_per_layer, columns_per_layer) * 0.01
                )
        
    def apply_cross_layer_connection(self, effective_conn, src_output):
        """
        应用跨层连接，处理批次维度不匹配问题
        
        Args:
            effective_conn: 有效连接矩阵，形状为 [columns_per_layer, columns_per_layer]
            src_output: 源层输出，可能是批次数据，形状为 [batch_size, hidden_dim]
            
        Returns:
            connection_output: 处理后的连接输出，与src_output形状匹配
        """
        # 检查批次维度
        is_batched = len(src_output.shape) > 1
        
        # 获取矩阵和向量尺寸
        matrix_size = effective_conn.shape[0]  # 假设是方阵，例如256x256
        
        if is_batched:
            batch_size = src_output.shape[0]
            feature_dim = src_output.shape[1]  # 例如384
            
            # 处理尺寸不匹配情况
            if matrix_size != feature_dim:
                # 创建与src_output形状相同的输出张量
                connection_output = torch.zeros_like(src_output)
                
                # 确定可处理的最小维度
                min_dim = min(matrix_size, feature_dim)
                
                # 对每个样本单独处理
                for b in range(batch_size):
                    sample = src_output[b]  # 提取单个样本
                    
                    # 在最小共同维度上执行矩阵乘法
                    if matrix_size < feature_dim:
                        # 如果矩阵小于向量，只处理向量的前matrix_size个元素
                        sample_result = torch.matmul(effective_conn, sample[:matrix_size])
                        # 只更新结果的前matrix_size个元素
                        connection_output[b, :matrix_size] = sample_result
                    else:
                        # 如果矩阵大于向量，只使用矩阵的前feature_dim行和列
                        effective_conn_subset = effective_conn[:feature_dim, :feature_dim]
                        sample_result = torch.matmul(effective_conn_subset, sample)
                        connection_output[b] = sample_result
                
                return connection_output
            else:
                # 尺寸匹配，正常处理
                connection_outputs = []
                
                for b in range(batch_size):
                    # 对每个样本单独处理
                    sample = src_output[b]  # 提取单个样本
                    
                    # 对单样本应用连接
                    sample_result = torch.matmul(effective_conn, sample)
                    connection_outputs.append(sample_result)
                
                # 堆叠所有结果，但要先检查列表是否为空
                if connection_outputs:
                    connection_output = torch.stack(connection_outputs)
                    return connection_output
                else:
                    # 如果列表为空，创建一个与src_output形状匹配的零张量
                    return torch.zeros_like(src_output)
        else:
            # 非批次数据处理尺寸不匹配情况
            feature_dim = src_output.shape[0]  # 向量长度
            
            if matrix_size != feature_dim:
                # 确定可处理的最小维度
                min_dim = min(matrix_size, feature_dim)
                
                if matrix_size < feature_dim:
                    # 如果矩阵小于向量，只处理向量的前matrix_size个元素
                    result = torch.matmul(effective_conn, src_output[:matrix_size])
                    # 创建一个与src_output形状相同的输出张量
                    connection_output = torch.zeros_like(src_output)
                    # 只更新结果的前matrix_size个元素
                    connection_output[:matrix_size] = result
                    return connection_output
                else:
                    # 如果矩阵大于向量，只使用矩阵的前feature_dim行和列
                    effective_conn_subset = effective_conn[:feature_dim, :feature_dim]
                    return torch.matmul(effective_conn_subset, src_output)
            else:
                # 尺寸匹配，直接执行矩阵乘法
                return torch.matmul(effective_conn, src_output)
            
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """功能柱前向传播
        
        Args:
            x: 输入张量
            
        Returns:
            outputs: 包含各层输出和通道活跃度的字典
        """
        batch_size = x.shape[0]
        layer_outputs = []
        nanocolumn_activities = []
        skip_connections = []
        
        current_input = x
        
        # 逐层处理
        for i, layer in enumerate(self.layers):
            layer_nanocolumn_outputs = []
            layer_alignment_scores = []
            
            # 每个纳米柱并行处理输入
            for nano_idx, nanocolumn in enumerate(layer):
                nano_output, alignment = nanocolumn(current_input)
                layer_nanocolumn_outputs.append(nano_output)
                layer_alignment_scores.append(alignment)
            
            # 将所有纳米柱输出加权融合
            if layer_alignment_scores:
                alignments = torch.stack(layer_alignment_scores)
                alignment_weights = F.softmax(alignments, dim=0)
                
                # 应用软注意力融合纳米柱输出
                if layer_nanocolumn_outputs:
                    stacked_outputs = torch.stack(layer_nanocolumn_outputs)
                    
                    # 动态扩展维度以匹配stacked_outputs
                    alignment_weights_expanded = alignment_weights
                    for dim_idx in range(1, stacked_outputs.dim()):
                        alignment_weights_expanded = alignment_weights_expanded.unsqueeze(-1)
                    
                    # 使用经过扩展的权重进行加权融合
                    layer_output = torch.sum(stacked_outputs * alignment_weights_expanded, dim=0)
                else:
                    # 如果没有纳米柱输出，创建零输出
                    layer_output = torch.zeros_like(current_input)
            else:
                # 如果没有对齐分数，使用输入作为输出
                layer_output = current_input
            
            # 保存当前层输出用于跳跃连接
            layer_outputs.append(layer_output)
            # 确保layer_alignment_scores非空再调用torch.stack
            if layer_alignment_scores:
                nanocolumn_activities.append(torch.stack(layer_alignment_scores))
            else:
                nanocolumn_activities.append(torch.zeros(self.columns_per_layer, device=x.device))
            
            # 添加跨层连接（如果不是最后一层）
            if i < len(self.layers) - 1:
                # 获取每个纳米柱的活跃状态
                column_activities = torch.tensor([
                    col.active for col in layer
                ], device=x.device)
                skip_connections.append((i, layer_output, column_activities))
            
            current_input = layer_output
        
        # 处理跨层连接（模拟Neuroligin-Neurexin优先通路）
        for src_idx, src_output, src_activities in skip_connections:
            for tgt_idx in range(src_idx + 2, len(self.layers)):
                if tgt_idx >= len(layer_outputs):
                    continue
                    
                # 获取跨层连接强度
                cross_conn = self.cross_layer_connections.get(f'layer_{src_idx}_to_{tgt_idx}')
                if cross_conn is not None:
                    # 仅考虑活跃纳米柱的跨层连接
                    mask = src_activities.float().unsqueeze(1)
                    effective_conn = cross_conn * mask
                    
                    # 应用跨层连接，处理好批次维度
                    connection_output = self.apply_cross_layer_connection(effective_conn, src_output)
                    
                    # 将源层输出添加到目标层
                    layer_outputs[tgt_idx] = layer_outputs[tgt_idx] + connection_output
        
        return {
            'final_output': layer_outputs[-1],
            'layer_outputs': layer_outputs,
            'nanocolumn_activities': nanocolumn_activities
        }


class StaticAnatomicalCore(nn.Module):
    """静态解剖基座
    
    模拟神经回路的固定解剖结构，包含多个功能柱模块
    """
    def __init__(
        self, 
        input_dim: int,
        shared_dim: int = 512,
        column_types: List[str] = ["vision", "language", "motor"],
        hidden_layer_sizes: Dict[str, List[int]] = None
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.shared_dim = shared_dim
        self.column_types = column_types
        
        # 设置每种功能柱的隐藏层结构
        if hidden_layer_sizes is None:
            hidden_layer_sizes = {
                "vision": [256, 512, 384],
                "language": [256, 512, 384],
                "motor": [128, 256, 192],
                "default": [256, 512, 384]
            }
        
        # 输入编码器 - 将输入映射到共享特征空间
        self.input_encoder = nn.Sequential(
            nn.Flatten(),  # 添加展平层，确保输入被正确展平为二维张量
            nn.Linear(input_dim, shared_dim),
            nn.LayerNorm(shared_dim),
            nn.GELU()
        )
        
        # 创建不同类型的功能柱
        self.functional_columns = nn.ModuleDict()
        for col_type in column_types:
            hidden_dims = hidden_layer_sizes.get(
                col_type, 
                hidden_layer_sizes["default"]
            )
            self.functional_columns[col_type] = FunctionalColumn(
                input_dim=shared_dim,
                output_dim=hidden_dims[-1],
                nanocolumn_config={
                    "alignment_gain": 1.0,
                    "alignment_threshold": 0.3
                },
                num_layers=len(hidden_dims),
                columns_per_layer=hidden_dims[0]
            )
            
        # 柱间整合层 - 将不同功能柱的输出整合
        total_dim = sum([hidden_layer_sizes.get(ct, hidden_layer_sizes["default"])[-1] 
                         for ct in column_types])
        self.integration_layer = nn.Sequential(
            nn.Linear(total_dim, shared_dim),
            nn.LayerNorm(shared_dim),
            nn.GELU()
        )
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """静态基座前向传播
        
        Args:
            x: 输入张量
            
        Returns:
            outputs: 包含基座输出和各功能柱结果的字典
        """
        # 共享输入编码
        shared_features = self.input_encoder(x)
        
        # 各功能柱并行处理
        column_outputs = {}
        column_activities = {}
        
        for col_type, column in self.functional_columns.items():
            column_result = column(shared_features)
            column_outputs[col_type] = column_result['final_output']
            column_activities[col_type] = column_result['nanocolumn_activities']
        
        # 将所有柱的输出连接并整合
        concatenated = torch.cat(list(column_outputs.values()), dim=1)
        integrated_output = self.integration_layer(concatenated)
        
        return {
            'output': integrated_output,
            'column_outputs': column_outputs,
            'column_activities': column_activities,
            'shared_features': shared_features
        }


class DynamicPlasticityModule(nn.Module):
    """动态可塑性模块
    
    实现突触可塑性和动态资源分配的神经元学习模块，
    整合STDP和BCM元可塑性机制动态调整突触权重
    """
    def __init__(
        self, 
        input_dim: int,
        output_dim: int,
        num_nanocolumns: int = 8,
        stdp_params: Dict[str, float] = None,
        bcm_params: Dict[str, float] = None
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_nanocolumns = num_nanocolumns
        
        # 默认STDP参数
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
        
        # 默认BCM参数
        if bcm_params is None:
            bcm_params = {
                "sliding_threshold_tau": 1000.0,
                "target_activity": 0.01,
                "init_threshold": 0.5
            }
            
        # 输入投影层
        self.input_projection = nn.Linear(input_dim, input_dim)
        
        # 自适应可塑性功能柱
        self.nanocolumns = nn.ModuleList([
            NanocolumnUnit(
                input_dim=input_dim, 
                output_dim=output_dim // num_nanocolumns
            ) for _ in range(num_nanocolumns)
        ])
        
        # 突触可塑性机制
        self.stdp_mechanisms = nn.ModuleList([
            MetaSTDP(
                hyper_dim=stdp_params.get("hyper_dim", 128),
                min_tau_plus=stdp_params.get("min_tau_plus", 10.0),
                max_tau_plus=stdp_params.get("max_tau_plus", 40.0),
                min_tau_minus=stdp_params.get("min_tau_minus", 20.0),
                max_tau_minus=stdp_params.get("max_tau_minus", 60.0),
                min_A_plus=stdp_params.get("min_A_plus", 0.001),
                max_A_plus=stdp_params.get("max_A_plus", 0.01),
                min_A_minus=stdp_params.get("min_A_minus", 0.001),
                max_A_minus=stdp_params.get("max_A_minus", 0.01)
            ) for _ in range(num_nanocolumns)
        ])
        
        # 元可塑性机制
        self.bcm_mechanisms = nn.ModuleList([
            BCMMetaplasticity(
                sliding_threshold_tau=bcm_params["sliding_threshold_tau"],
                target_activity=bcm_params["target_activity"],
                init_threshold=bcm_params.get("init_threshold", 0.5)
            ) for _ in range(num_nanocolumns)
        ])
        
        # 动态资源分配
        self.resource_allocator = DynamicResourceAllocation(
            num_resources=num_nanocolumns,
            recovery_rate=0.01,
            depletion_rate=0.1
        )
        
        # 输出整合层
        self.output_integration = nn.Linear(output_dim, output_dim)
        
        # 记录前一活动状态
        self.register_buffer('prev_activity', None)
        self.register_buffer('allocation_mask', torch.ones(num_nanocolumns))
    
    def reset_state(self, batch_size: int = 1):
        """重置动态状态
        
        Args:
            batch_size: 批次大小
        """
        device = next(self.parameters()).device
        self.prev_activity = torch.zeros(
            batch_size, self.output_dim, device=device
        )
        self.allocation_mask = torch.ones(
            self.num_nanocolumns, device=device
        )
    
    def forward(
        self, 
        x: torch.Tensor, 
        prev_activity: Optional[torch.Tensor] = None,
        learning_enabled: bool = True,
        dt: float = 1.0
    ) -> Dict[str, torch.Tensor]:
        """前向传播
        
        Args:
            x: 输入特征
            prev_activity: 前一时间步的活动状态
            learning_enabled: 是否启用学习
            dt: 时间步长
            
        Returns:
            outputs: 包含输出和各种状态的字典
        """
        batch_size = x.shape[0]
        device = x.device
        
        # 初始化状态
        if self.prev_activity is None or self.prev_activity.shape[0] != batch_size:
            self.reset_state(batch_size)
        
        # 如果提供了外部活动状态，使用它
        if prev_activity is not None:
            self.prev_activity = prev_activity
        
        # 输入投影
        projected_input = self.input_projection(x)
        
        # 计算每个纳米柱的当前重要性
        importance_scores = torch.zeros(self.num_nanocolumns, device=device)
        
        for i, column in enumerate(self.nanocolumns):
            # 计算对齐度作为重要性指标
            with torch.no_grad():
                importance_scores[i] = column.compute_alignment(column.pre_synaptic_rim, column.post_synaptic_psd95)
        
        # 动态资源分配
        if learning_enabled:
            self.allocation_mask = self.resource_allocator.allocate(importance_scores)
        
        # 收集各纳米柱输出
        nanocolumn_outputs = []
        alignments = []
        
        for i, column in enumerate(self.nanocolumns):
            # 如果该柱被资源分配器选中
            if self.allocation_mask[i] > 0.5:
                # 纳米柱处理
                nano_output, alignment = column(projected_input)
                
                # 如果启用学习，应用可塑性机制
                if learning_enabled:
                    # 计算当前和前一活动模拟放电事件
                    current_activity = nano_output.abs().mean()
                    prev_activity = self.prev_activity[:, i*nano_output.shape[1]:(i+1)*nano_output.shape[1]].abs().mean()
                    
                    pre_spike = current_activity > 0.1
                    post_spike = prev_activity > 0.1
                    
                    # 更新STDP跟踪变量
                    self.stdp_mechanisms[i].update_traces(pre_spike, post_spike, dt)
                    
                    # 计算权重变化
                    if pre_spike or post_spike:
                        weight_change = self.stdp_mechanisms[i].compute_weight_change(pre_spike, post_spike)
                        
                        # 修改纳米柱突触权重
                        with torch.no_grad():
                            # 应用BCM元可塑性调制
                            bcm_modulation = self.bcm_mechanisms[i].compute_plasticity_modulation(current_activity)
                            self.bcm_mechanisms[i].update(current_activity, dt)
                            
                            # 根据BCM调制系数调整权重变化
                            modulated_change = weight_change * bcm_modulation
                            
                            # 应用到突触权重
                            column.synaptic_weights.data += modulated_change * 0.01
                
                nanocolumn_outputs.append(nano_output)
                alignments.append(alignment)
            else:
                # 创建零输出
                nano_output_shape = (batch_size, self.output_dim // self.num_nanocolumns)
                nanocolumn_outputs.append(torch.zeros(nano_output_shape, device=device))
                alignments.append(torch.tensor(0.0, device=device))
        
        # 融合所有纳米柱输出
        if len(nanocolumn_outputs) > 0:
            # 如果是分段输出，连接它们
            if all(out.shape[1] == self.output_dim // self.num_nanocolumns for out in nanocolumn_outputs):
                combined_output = torch.cat(nanocolumn_outputs, dim=1)
            else:
                # 否则，对齐维度后求和
                resized_outputs = []
                for out in nanocolumn_outputs:
                    if out.shape[1] != self.output_dim:
                        # 调整输出大小以匹配目标维度
                        if out.shape[1] > self.output_dim:
                            resized = out[:, :self.output_dim]
                        else:
                            resized = F.pad(out, (0, self.output_dim - out.shape[1]))
                        resized_outputs.append(resized)
                    else:
                        resized_outputs.append(out)
                combined_output = torch.stack(resized_outputs).sum(dim=0) if resized_outputs else torch.zeros(batch_size, self.output_dim, device=device)
        else:
            combined_output = torch.zeros(batch_size, self.output_dim, device=device)
        
        # 输出整合
        integrated_output = self.output_integration(combined_output)
        
        # 更新前一活动状态
        self.prev_activity = combined_output.detach().clone()
        
        return {
            'output': integrated_output,
            'raw_output': combined_output,
            'alignments': torch.stack(alignments) if alignments else torch.zeros(self.num_nanocolumns, device=device),
            'allocation_mask': self.allocation_mask
        }