import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
from NanocolumnUnit import StaticAnatomicalCore,DynamicPlasticityModule
from MetaSTDP import EphapticCoupling
class NeuroDynArch(nn.Module):
    """神经动态架构
    
    整合静态解剖基座、动态可塑性机制和全局耦合系统，
    实现生物启发的高效动态神经网络
    """
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        shared_dim: int = 512,
        column_types: List[str] = ["vision", "language", "reasoning"],
        hidden_layer_configs: Dict[str, List[int]] = None,
        num_modules: int = 4,
        freeze_static_core: bool = False
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.column_types = column_types
        self.num_modules = num_modules
        self.freeze_static_core = freeze_static_core
        
        # 默认隐藏层配置
        if hidden_layer_configs is None:
            hidden_layer_configs = {
                "vision": [256, 512, 384],
                "language": [256, 512, 384],
                "reasoning": [256, 384, 256],
                "default": [256, 384, 256]
            }
        
        # 1. 静态解剖基座
        self.static_core = StaticAnatomicalCore(
            input_dim=input_dim,
            shared_dim=shared_dim,
            column_types=column_types,
            hidden_layer_sizes=hidden_layer_configs
        )
        
        if freeze_static_core:
            for param in self.static_core.parameters():
                param.requires_grad = False
        
        # 获取静态基座输出维度
        core_output_dim = shared_dim
        
        # 2. 动态可塑性层
        # 为每个功能模块创建单独的动态可塑性模块
        self.plasticity_modules = nn.ModuleList([
            DynamicPlasticityModule(
                input_dim=core_output_dim,
                output_dim=core_output_dim // 2,
                num_nanocolumns=8,
                stdp_params={
                    "hyper_dim": 128,
                    "min_tau_plus": 10.0,
                    "max_tau_plus": 40.0,
                    "min_tau_minus": 20.0,
                    "max_tau_minus": 60.0,
                    "min_A_plus": 0.001,
                    "max_A_plus": 0.01,
                    "min_A_minus": 0.001,
                    "max_A_minus": 0.01
                },
                bcm_params={"sliding_threshold_tau": 1000.0, "target_activity": 0.01}
            ) for _ in range(num_modules)
        ])
        
        # 3. 突触外耦合系统
        self.ephaptic_coupling = EphapticCoupling(
            num_modules=num_modules,
            feature_dim=core_output_dim // 2,
            coupling_strength=0.1,
            spatial_decay=2.0
        )
        
        # 4. 整合输出层
        plasticity_output_dim = (core_output_dim // 2) * num_modules
        self.output_layer = nn.Sequential(
            nn.Linear(plasticity_output_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Linear(256, output_dim)
        )
        
        # 5. 自适应混合门控 (决定静态和动态路径的混合比例)
        self.adaptive_gate = nn.Sequential(
            nn.Linear(core_output_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
        # 静态捷径通路
        self.static_shortcut = nn.Linear(core_output_dim, output_dim)
        
        # 轨迹记忆 (存储历史活动状态)
        self.register_buffer('activity_history', None)
        self.history_length = 5  # 保留的历史状态数量
    
    def reset_dynamic_state(self, batch_size: int = 1):
        """重置所有动态状态
        
        Args:
            batch_size: 批次大小
        """
        for module in self.plasticity_modules:
            module.reset_state(batch_size)
        
        # 重置活动历史
        device = next(self.parameters()).device
        core_output_dim = self.static_core.shared_dim
        self.activity_history = torch.zeros(
            batch_size, self.history_length, core_output_dim, device=device
        )
    
    def update_activity_history(self, current_activity: torch.Tensor):
        """更新活动历史轨迹
        
        Args:
            current_activity: 当前活动状态
        """
        if self.activity_history is None:
            batch_size = current_activity.shape[0]
            self.reset_dynamic_state(batch_size)
        
        # 移除最早的历史，添加当前活动
        self.activity_history = torch.cat([
            self.activity_history[:, 1:], 
            current_activity.unsqueeze(1)
        ], dim=1)
    
    def forward(
        self, 
        x: torch.Tensor, 
        learning_enabled: bool = True,
        dt: float = 1.0,
        verbose: bool = False
    ) -> Dict[str, torch.Tensor]:
        """神经动态架构前向传播
        
        Args:
            x: 输入张量
            learning_enabled: 是否启用学习
            dt: 时间步长(ms)
            verbose: 是否打印详细日志
            
        Returns:
            outputs: 包含最终输出和中间状态的字典
        """
        batch_size = x.shape[0]
        
        # 首次调用时初始化状态
        if self.activity_history is None:
            self.reset_dynamic_state(batch_size)
        # 当批次大小变化时重新初始化状态
        elif self.activity_history.shape[0] != batch_size:
            if verbose:
                print(f"批次大小变化: activity_history={self.activity_history.shape[0]}, 当前输入={batch_size}，重置状态")
            self.reset_dynamic_state(batch_size)
        
        # 1. 静态解剖基座处理
        core_outputs = self.static_core(x)
        core_features = core_outputs['output']
        
        # 2. 动态可塑性处理
        plasticity_outputs = []
        module_activities = []
        
        for i, module in enumerate(self.plasticity_modules):
            # 将静态特征与历史活动结合
            if i % 2 == 0:  # 一半模块使用当前特征
                module_input = core_features
            else:  # 另一半模块整合历史信息
                # 计算注意力权重
                attn_weights = torch.bmm(
                    core_features.unsqueeze(1),
                    self.activity_history.transpose(1, 2)
                ).squeeze(1)
                attn_weights = F.softmax(attn_weights, dim=1)
                
                # 加权融合历史活动
                history_context = torch.bmm(
                    attn_weights.unsqueeze(1),
                    self.activity_history
                ).squeeze(1)
                
                # 混合当前特征和历史信息
                module_input = core_features + 0.2 * history_context
            
            # 动态可塑性模块处理
            module_output = module(
                module_input, 
                prev_activity=self.activity_history[:, -1],
                learning_enabled=learning_enabled,
                dt=dt
            )
            
            plasticity_outputs.append(module_output)
            module_activities.append(module_output['output'])
        
        # 3. 突触外耦合处理
        coupled_activities = self.ephaptic_coupling(module_activities)
        
        # 4. 整合所有动态模块输出
        # 对活动向量使用detach()断开梯度连接，防止计算图重复访问导致的backward错误
        detached_activities = [activity.detach() for activity in coupled_activities]
        combined_dynamic = torch.cat(detached_activities, dim=1)
        dynamic_output = self.output_layer(combined_dynamic)
        
        # 5. 静态捷径通路
        static_output = self.static_shortcut(core_features)
        
        # 6. 自适应门控混合
        mix_ratio = self.adaptive_gate(core_features)
        final_output = mix_ratio * dynamic_output + (1 - mix_ratio) * static_output
        
        # 更新活动历史
        self.update_activity_history(core_features)
        
        return {
            'output': final_output,
            'static_output': static_output,
            'dynamic_output': dynamic_output,
            'mix_ratio': mix_ratio,
            'core_features': core_features,
            'plasticity_outputs': plasticity_outputs,
            'coupled_activities': coupled_activities
        }


class EnergyConstrainedLoss(nn.Module):
    """能量约束损失函数
    
    结合任务性能和能量消耗的多目标损失函数
    """
    def __init__(
        self,
        task_loss_fn: nn.Module = nn.MSELoss(),
        energy_weight: float = 0.01,
        sparsity_target: float = 0.1,
        enable_caching: bool = True  # 新增参数：启用缓存以减少计算
    ):
        super().__init__()
        self.task_loss_fn = task_loss_fn
        self.energy_weight = energy_weight
        self.sparsity_target = sparsity_target
        self.enable_caching = enable_caching
        
        # 缓存上一次计算的能量损失（按批次ID索引）
        self.energy_loss_cache = {}
        self.cache_max_size = 100  # 最大缓存条目数
        self.cache_hits = 0
        self.cache_misses = 0
    
    def forward(
        self, 
        outputs: Dict[str, torch.Tensor],
        targets: torch.Tensor,
        batch_id: Optional[int] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """计算多目标损失
        
        Args:
            outputs: 模型输出字典
            targets: 目标标签
            batch_id: 批次ID，用于能量损失缓存，None表示不使用缓存
            
        Returns:
            total_loss: 总损失
            loss_components: 各损失分量字典
        """
        # 确保输出和目标维度匹配
        predictions = outputs['output']
        if predictions.shape != targets.shape:
            # 调整维度
            if predictions.ndim == 1 and targets.ndim == 2:
                predictions = predictions.unsqueeze(1)
            elif predictions.ndim == 2 and targets.ndim == 1:
                targets = targets.unsqueeze(1)
        
        # 1. 任务损失
        task_loss = self.task_loss_fn(predictions, targets)
        
        # 2. 能量消耗损失
        # 首先检查缓存
        use_cache = self.enable_caching and batch_id is not None
        if use_cache and batch_id in self.energy_loss_cache:
            energy_loss = self.energy_loss_cache[batch_id]
            self.cache_hits += 1
        else:
            self.cache_misses += 1
            energy_loss = 0.0
            allocation_counts = 0
            
            # 统计动态模块活跃度
            if 'plasticity_outputs' in outputs:
                for plasticity_output in outputs['plasticity_outputs']:
                    if 'allocation_mask' in plasticity_output:
                        allocation_mask = plasticity_output['allocation_mask']
                        activity_ratio = allocation_mask.float().mean()
                        
                        # 稀疏性惩罚: (activity - target)^2
                        sparsity_loss = (activity_ratio - self.sparsity_target) ** 2
                        energy_loss += sparsity_loss
                        allocation_counts += 1
            
            if allocation_counts > 0:
                energy_loss /= allocation_counts
            
            # 更新缓存
            if use_cache:
                # 如果缓存过大，移除最早的条目
                if len(self.energy_loss_cache) >= self.cache_max_size:
                    oldest_key = min(self.energy_loss_cache.keys())
                    del self.energy_loss_cache[oldest_key]
                
                # 存储当前能量损失
                self.energy_loss_cache[batch_id] = energy_loss
        
        # 3. 总损失
        total_loss = task_loss + self.energy_weight * energy_loss
        
        return total_loss, {
            'task_loss': task_loss,
            'energy_loss': energy_loss if isinstance(energy_loss, torch.Tensor) else torch.tensor(energy_loss, device=task_loss.device),
            'total_loss': total_loss
        }


def train_neurodynarch(
    model: NeuroDynArch,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    num_epochs: int = 10,
    learning_rate: float = 1e-4,
    device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    checkpoint_dir: str = 'checkpoints',
    stage: str = 'pretrain'
):
    """训练神经动态架构模型
    
    Args:
        model: NeuroDynArch模型
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        num_epochs: 训练轮数
        learning_rate: 学习率
        device: 计算设备
        checkpoint_dir: 检查点保存目录
        stage: 训练阶段 ('pretrain'或'finetune')
    """
    import os
    from torch.optim import AdamW
    from torch.optim.lr_scheduler import CosineAnnealingLR
    from tqdm import tqdm
    
    # 确保检查点目录存在
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # 移动模型到指定设备
    model.to(device)
    
    # 根据训练阶段配置模型
    if stage == 'pretrain':
        # 预训练阶段: 冻结动态模块，训练静态基座
        model.freeze_static_core = False
        for module in model.plasticity_modules:
            for param in module.parameters():
                param.requires_grad = False
        
        # 禁用学习
        learning_enabled = False
    else:
        # 微调阶段: 解冻动态模块，固定静态基座
        model.freeze_static_core = True
        for module in model.plasticity_modules:
            for param in module.parameters():
                param.requires_grad = True
        
        # 启用学习
        learning_enabled = True
    
    # 优化器
    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=learning_rate,
        weight_decay=0.01
    )
    
    # 学习率调度器
    scheduler = CosineAnnealingLR(
        optimizer, 
        T_max=num_epochs,
        eta_min=learning_rate / 10
    )
    
    # 损失函数
    criterion = EnergyConstrainedLoss(
        task_loss_fn=nn.MSELoss(),
        energy_weight=0.01 if stage == 'pretrain' else 0.05,
        sparsity_target=0.2 if stage == 'pretrain' else 0.1
    )
    
    # 训练循环
    best_val_acc = 0.0
    
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        train_acc = 0.0
        train_samples = 0
        
        # 进度条
        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]')
        
        for inputs, targets in train_pbar:
            # 移动数据到设备
            inputs = inputs.to(device)
            targets = targets.to(device)
            batch_size = inputs.shape[0]
            
            # 清零梯度
            optimizer.zero_grad()
            
            # 前向传播
            outputs = model(inputs, learning_enabled=learning_enabled)
            
            # 计算损失
            loss, loss_components = criterion(outputs, targets)
            
            # 反向传播
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # 参数更新
            optimizer.step()
            
            # 计算准确率
            preds = outputs['output'].argmax(dim=1)
            acc = (preds == targets).float().mean().item()
            
            # 更新统计
            train_loss += loss.item() * batch_size
            train_acc += acc * batch_size
            train_samples += batch_size
            
            # 更新进度条
            train_pbar.set_postfix({
                'loss': loss_components['total_loss'].item(),
                'task_loss': loss_components['task_loss'].item(),
                'energy': loss_components['energy_loss'].item(),
                'acc': acc
            })
        
        # 计算训练集平均损失和准确率
        train_loss /= train_samples
        train_acc /= train_samples
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_acc = 0.0
        val_samples = 0
        
        # 不计算梯度
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Val]')
            
            for inputs, targets in val_pbar:
                # 移动数据到设备
                inputs = inputs.to(device)
                targets = targets.to(device)
                batch_size = inputs.shape[0]
                
                # 前向传播
                outputs = model(inputs, learning_enabled=False)
                
                # 计算损失
                loss, loss_components = criterion(outputs, targets)
                
                # 计算准确率
                preds = outputs['output'].argmax(dim=1)
                acc = (preds == targets).float().mean().item()
                
                # 更新统计
                val_loss += loss.item() * batch_size
                val_acc += acc * batch_size
                val_samples += batch_size
                
                # 更新进度条
                val_pbar.set_postfix({
                    'loss': loss.item(),
                    'acc': acc
                })
        
        # 计算验证集平均损失和准确率
        val_loss /= val_samples
        val_acc /= val_samples
        
        # 更新学习率
        scheduler.step()
        
        # 输出当前轮结果
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}')
        print(f'  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
        
        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            checkpoint_path = os.path.join(
                checkpoint_dir, 
                f'neurodynarch_{stage}_best.pth'
            )
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss
            }, checkpoint_path)
            print(f'  Saved best model checkpoint to {checkpoint_path}')
        
        # 保存阶段性检查点
        if (epoch + 1) % 5 == 0:
            checkpoint_path = os.path.join(
                checkpoint_dir, 
                f'neurodynarch_{stage}_epoch{epoch+1}.pth'
            )
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss
            }, checkpoint_path)

    return model


def dynamic_inference(
    model: NeuroDynArch,
    inputs: torch.Tensor,
    max_time_steps: int = 10,
    confidence_threshold: float = 0.95,
    device: torch.device = None,
    min_time_steps: int = 3  # 添加最小时间步参数
):
    """动态推理过程
    
    根据输入复杂度和置信度动态调整推理时间和计算量
    
    Args:
        model: NeuroDynArch模型
        inputs: 输入张量
        max_time_steps: 最大推理时间步数
        confidence_threshold: 提前停止的置信度阈值
        device: 计算设备
        min_time_steps: 最小执行步数，即使达到置信度阈值也至少执行这么多步
        
    Returns:
        outputs: 最终输出和中间状态
    """
    if device is not None:
        model = model.to(device)
        inputs = inputs.to(device)
    
    # 确保模型在评估模式
    model.eval()
    
    # 重置模型动态状态
    batch_size = inputs.shape[0]
    model.reset_dynamic_state(batch_size)
    
    # 存储中间结果
    all_outputs = []
    all_confidences = []
    
    # 逐步推理
    with torch.no_grad():
        for t in range(max_time_steps):
            # 单步推理
            outputs = model(inputs, learning_enabled=False, dt=1.0)
            all_outputs.append(outputs)
            
            # 计算输出置信度 (使用softmax后的最大概率值)
            if model.output_dim > 1:  # 分类任务
                probs = F.softmax(outputs['output'], dim=1)
                confidences, _ = probs.max(dim=1)
            else:  # 回归任务，使用输出值的稳定性作为置信度
                if t > 0:
                    # 计算与前一步输出的变化程度，变化小表示高置信度
                    prev_output = all_outputs[t-1]['output']
                    delta = torch.abs(outputs['output'] - prev_output)
                    confidences = 1.0 - torch.clamp(delta / (torch.abs(prev_output) + 1e-6), 0, 1)
                else:
                    # 第一步没有前一步比较，设置中等置信度
                    confidences = torch.ones_like(outputs['output']).squeeze() * 0.5
            
            mean_confidence = confidences.mean().item()
            all_confidences.append(mean_confidence)
            
            # 检查是否达到置信度阈值，同时确保至少执行min_time_steps步
            if t >= min_time_steps - 1 and mean_confidence >= confidence_threshold:
                if t < max_time_steps - 1:  # 只有在提前终止时才打印
                    print(f"Early stopping at step {t+1}/{max_time_steps} "
                        f"(confidence: {mean_confidence:.4f}, threshold: {confidence_threshold:.4f})")
                break
    
    # 选择最终输出 (最后一个时间步)
    final_output = all_outputs[-1]
    
    # 添加推理统计信息
    final_output['time_steps_used'] = len(all_outputs)
    final_output['confidences'] = all_confidences
    
    # 计算能耗估计 (基于激活的纳米柱数量)
    energy_usage = 0.0
    for t, output in enumerate(all_outputs):
        active_count = 0
        total_count = 0
        
        for plasticity_output in output['plasticity_outputs']:
            allocation_mask = plasticity_output['allocation_mask']
            active_count += allocation_mask.sum().item()
            total_count += allocation_mask.numel()
        
        energy_ratio = active_count / total_count if total_count > 0 else 0
        energy_usage += energy_ratio
    
    final_output['energy_usage'] = energy_usage / len(all_outputs)
    
    return final_output