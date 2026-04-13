---
date: 2026-04-04
question_id: Q008
topics: ["rollout-manager", "rollout", "data-generation", "evaluation"]
related_files:
  - slime/ray/rollout.py
---

# Question

介绍 `RolloutManager` 的初始化函数以外的其他函数都在干什么

# Answer

`RolloutManager` 除了 `__init__` 外有 20+ 个方法，可分为以下几类：

## 1. 核心业务流程

### 1.1 generate() - 生成训练数据 ⭐核心

**文件**: `slime/ray/rollout.py:479-492`

```python
def generate(self, rollout_id):
    """执行一次 rollout，生成训练数据"""
    start_time = time.time()
    self.rollout_id = rollout_id
    
    # 1. 恢复健康监控
    self.health_monitoring_resume()
    
    # 2. 获取 rollout 数据（调用 SGLang 生成）
    data, metrics = self._get_rollout_data(rollout_id=rollout_id)
    
    # 3. 保存调试数据（可选）
    self._save_debug_rollout_data(data, rollout_id=rollout_id, evaluation=False)
    
    # 4. 记录 rollout 指标
    _log_rollout_data(rollout_id, self.args, data, metrics, time.time() - start_time)
    
    # 5. 转换为训练数据格式
    data = self._convert_samples_to_train_data(data)
    
    # 6. 按 DP size 分割数据，返回给训练节点
    return self._split_train_data_by_dp(data, self.train_parallel_config["dp_size"])
```

**调用链**:
```
train.py -> rollout_manager.generate.remote(rollout_id)
    -> _get_rollout_data()         # 调用 SGLang 生成文本
    -> _convert_samples_to_train_data()  # 转换为训练格式
    -> _split_train_data_by_dp()   # 分割给各 DP rank
```

---

### 1.2 eval() - 评估模型

**文件**: `slime/ray/rollout.py:494-503`

```python
def eval(self, rollout_id):
    """在评估数据集上生成并记录指标"""
    if self.args.debug_train_only:
        return  # 只训练不评估模式
    
    self.health_monitoring_resume()
    
    # 调用评估 rollout 函数（与训练不同，evaluation=True）
    result = call_rollout_fn(
        self.eval_generate_rollout,  # 可能是不同的函数
        self.args, 
        rollout_id, 
        self.data_source, 
        evaluation=True
    )
    
    # 保存调试数据
    self._save_debug_rollout_data(result.data, rollout_id=rollout_id, evaluation=True)
    
    # 记录评估指标
    _log_eval_rollout_data(rollout_id, self.args, result.data, result.metrics)
```

**与 generate() 的区别**:
| 区别 | generate() | eval() |
|------|-----------|--------|
| 目的 | 生成训练数据 | 评估模型性能 |
| 返回 | 训练数据（给 Actor） | 只记录指标 |
| rollout 函数 | `generate_rollout` | `eval_generate_rollout` |
| 数据处理 | 完整转换流程 | 简单记录 |

---

## 2. 内存管理（Offload/Onload）

### 2.1 offload() - 释放 GPU 内存

**文件**: `slime/ray/rollout.py:511-514`

```python
def offload(self):
    """将 SGLang 引擎从 GPU offload 到 CPU"""
    self.health_monitoring_pause()  # 暂停健康检查
    for srv in self.servers.values():
        srv.offload()  # 调用每个 server 的 offload
```

**何时调用**:
- Colocate 模式下，训练前释放 GPU 给 Megatron
- `train.py:80` `ray.get(rollout_manager.offload.remote())`

**内部实现**: `ServerGroup.offload()`
```python
def offload(self):
    # 对每个引擎调用 release_memory_occupation
    return [engine.release_memory_occupation.remote() 
            for engine in self.engines]
```

---

### 2.2 onload() / onload_weights() / onload_kv()

**文件**: `slime/ray/rollout.py:516-526`

```python
def onload(self, tags=None):
    """将 SGLang 引擎从 CPU 加载回 GPU"""
    for srv in self.servers.values():
        srv.onload(tags)

def onload_weights(self):
    """只加载权重（用于 colocate 模式）"""
    for srv in self.servers.values():
        srv.onload_weights()  # tags=[GPU_MEMORY_TYPE_WEIGHTS]

def onload_kv(self):
    """只加载 KV cache 和 CUDA graphs"""
    for srv in self.servers.values():
        srv.onload_kv()  # tags=[KV_CACHE, CUDA_GRAPH]
```

**调用时机** (train.py):
```python
# 训练后，准备生成前
if args.offload_rollout:
    ray.get(rollout_manager.onload_weights.remote())  # 加载权重
    ray.get(rollout_manager.onload_kv.remote())       # 加载 KV cache
```

**为什么分开？**
- 权重需要从 Megatron 同步（通过 `update_weights`）
- KV cache 和 CUDA graphs 可以直接从 CPU 恢复

---

## 3. 故障恢复

### 3.1 recover_updatable_engines()

**文件**: `slime/ray/rollout.py:528-549`

```python
def recover_updatable_engines(self):
    """恢复故障的 rollout 引擎
    
    被 actor.update_weights() 调用，在权重更新前恢复故障引擎
    """
    self.health_monitoring_pause()
    
    srv = self._get_updatable_server()
    
    if self.rollout_id == -1 or srv is None:
        # 初始状态，无需恢复
        return engines, lock, num_new, gpu_counts, gpu_offsets
    
    # 调用 RolloutServer.recover() 重新创建引擎
    srv.recover()
    
    return (
        srv.engines,           # 恢复后的引擎列表
        self.rollout_engine_lock,
        srv.num_new_engines,   # 新创建的引擎数
        srv.engine_gpu_counts,
        srv.engine_gpu_offsets,
    )
```

**调用链**:
```
actor.update_weights()
    -> rollout_manager.recover_updatable_engines.remote()
        -> RolloutServer.recover()
            -> group.start_engines()  # 重新创建故障引擎
```

---

### 3.2 clear_updatable_num_new_engines()

**文件**: `slime/ray/rollout.py:551-555`

```python
def clear_updatable_num_new_engines(self):
    """清除 num_new_engines 计数
    
    在权重更新完成后调用，标记新引擎已同步
    """
    srv = self._get_updatable_server()
    if srv:
        srv.num_new_engines = 0
```

---

## 4. 数据处理和转换

### 4.1 _get_rollout_data() - 获取生成数据 ⭐核心

**文件**: `slime/ray/rollout.py:568-608`

```python
def _get_rollout_data(self, rollout_id):
    """获取 rollout 生成的原始数据"""
    
    # 调试模式：从文件加载预存数据
    if self.args.load_debug_rollout_data:
        data = torch.load(...)['samples']
        metrics = None
    else:
        # 正常模式：调用 SGLang 生成
        result = call_rollout_fn(
            self.generate_rollout,  # 如 sglang_rollout.generate_rollout
            self.args,
            rollout_id,
            self.data_source,
            evaluation=False
        )
        metrics = result.metrics
        data = result.samples
        
        # 展平嵌套列表
        while isinstance(data[0], list):
            data = list(itertools.chain.from_iterable(data))
        
        # 调整样本数量以匹配 global_batch_size
        if not self.args.disable_rollout_trim_samples:
            global_batch_size = self.args.global_batch_size
            if len(data) % global_batch_size != 0:
                trim_len = (len(data) // global_batch_size) * global_batch_size
                data = data[:trim_len]
    
    return data, metrics
```

---

### 4.2 _convert_samples_to_train_data() - 转换为训练格式 ⭐核心

**文件**: `slime/ray/rollout.py:683-746`

```python
def _convert_samples_to_train_data(self, samples):
    """将 Sample 对象转换为训练数据字典"""
    
    # 1. 后处理奖励（如 GRPO 的归一化）
    raw_rewards, rewards = self._post_process_rewards(samples)
    
    # 2. 构建训练数据字典
    train_data = {
        'tokens': [sample.tokens for sample in samples],
        'response_lengths': [sample.response_length for sample in samples],
        'rewards': rewards,
        'raw_reward': raw_rewards,
        'truncated': [1 if sample.status == Sample.Status.TRUNCATED else 0 ...],
        'sample_indices': [sample.index for sample in samples],
    }
    
    # 3. 处理 loss mask
    loss_masks = []
    for sample in samples:
        if sample.loss_mask is None:
            sample.loss_mask = [1] * sample.response_length
        if sample.remove_sample:
            sample.loss_mask = [0] * sample.response_length
        loss_masks.append(sample.loss_mask)
    train_data['loss_masks'] = loss_masks
    
    # 4. 添加可选字段
    if samples[0].rollout_log_probs is not None:
        train_data['rollout_log_probs'] = [...]
    
    if samples[0].rollout_routed_experts is not None:
        train_data['rollout_routed_experts'] = [...]
    
    if samples[0].teacher_log_probs is not None:
        train_data['teacher_log_probs'] = [...]
    
    return train_data
```

**输出格式**:
```python
{
    'tokens': [[1, 2, 3, ...], [...], ...],  # token IDs
    'response_lengths': [100, 150, ...],      # 生成长度
    'rewards': [0.5, 1.0, ...],               # 奖励值
    'loss_masks': [[1, 1, 0, ...], ...],      # loss mask
    'rollout_log_probs': [[-2.3, ...], ...],  # 生成时的 log prob
    # ... 其他字段
}
```

---

### 4.3 _split_train_data_by_dp() - 按 DP 分割

**文件**: `slime/ray/rollout.py:751-802`

```python
def _split_train_data_by_dp(self, data, dp_size):
    """将训练数据按 DP size 分割，返回每个 rank 的数据"""
    
    total_lengths = [len(t) for t in data['tokens']]
    
    # 选择分区策略
    if self.args.balance_data:
        # 使用 karmarkar_karp 算法平衡序列长度
        partitions = get_seqlen_balanced_partitions(total_lengths, dp_size)
    else:
        # 简单轮询
        partitions = [range(i, len(total_lengths), dp_size) for i in range(dp_size)]
    
    rollout_data_refs = []
    for i in range(dp_size):
        rollout_data = {'partition': partitions[i]}
        
        # 为每个 DP rank 提取对应的数据
        for key in ['tokens', 'response_lengths', 'rewards', ...]:
            if key in data:
                rollout_data[key] = [data[key][j] for j in partitions[i]]
        
        # 放入 Ray 对象存储，返回引用
        rollout_data_refs.append(Box(ray.put(rollout_data)))
    
    return rollout_data_refs
```

**示例** (dp_size=4):
```
原始数据: [S0, S1, S2, S3, S4, S5, S6, S7]  (8 samples)

分区后:
  DP rank 0: [S0, S4]
  DP rank 1: [S1, S5]
  DP rank 2: [S2, S6]
  DP rank 3: [S3, S7]
```

---

### 4.4 _post_process_rewards()

**文件**: `slime/ray/rollout.py:656-681`

```python
def _post_process_rewards(self, samples):
    """奖励后处理（如 GRPO/GSpO 的归一化）"""
    
    raw_rewards = [sample.get_reward_value(self.args) for sample in samples]
    
    if self.args.advantage_estimator in ['grpo', 'gspo', ...]:
        # Group normalization
        rewards = torch.tensor(raw_rewards, dtype=torch.float)
        rewards = rewards.reshape(-1, self.args.n_samples_per_prompt)
        
        mean = rewards.mean(dim=-1, keepdim=True)
        rewards = rewards - mean  # 减去组均值
        
        if self.args.grpo_std_normalization:
            std = rewards.std(dim=-1, keepdim=True)
            rewards = rewards / (std + 1e-6)  # 除以组标准差
        
        return raw_rewards, rewards.flatten().tolist()
    
    return raw_rewards, raw_rewards
```

---

## 5. 数据持久化

### 5.1 save() / load()

**文件**: `slime/ray/rollout.py:505-509`

```python
def save(self, rollout_id):
    """保存数据源状态（用于断点续训）"""
    self.data_source.save(rollout_id)

def load(self, rollout_id=None):
    """加载数据源状态"""
    self.data_source.load(rollout_id)
```

**调用时机** (train.py):
```python
# 保存检查点时
save(rollout_id)
    -> actor_model.save_model(rollout_id)
    -> rollout_manager.save.remote(rollout_id)  # 保存数据状态
```

---

### 5.2 _save_debug_rollout_data()

**文件**: `slime/ray/rollout.py:637-654`

```python
def _save_debug_rollout_data(self, data, rollout_id, evaluation):
    """保存 rollout 数据用于调试"""
    if self.args.save_debug_rollout_data:
        path = self.args.save_debug_rollout_data.format(
            rollout_id=('eval_' if evaluation else '') + str(rollout_id)
        )
        torch.save({'rollout_id': rollout_id, 'samples': [...]}, path)
```

---

## 6. 工具方法

### 6.1 get_num_rollout_per_epoch()

**文件**: `slime/ray/rollout.py:475-477`

```python
def get_num_rollout_per_epoch(self):
    """计算每 epoch 需要多少次 rollout"""
    assert self.args.rollout_global_dataset
    return len(self.data_source) // self.args.rollout_batch_size
```

**示例**:
```
数据集大小: 10000
rollout_batch_size: 32
num_rollout_per_epoch = 10000 // 32 = 312
```

---

### 6.2 dispose()

**文件**: `slime/ray/rollout.py:433-436`

```python
def dispose(self):
    """清理资源（训练结束时调用）"""
    for monitor in self._health_monitors:
        monitor.stop()  # 停止健康监控
    logging_utils.finish_tracking(self.args)
```

---

### 6.3 health_monitoring_pause() / resume()

**文件**: `slime/ray/rollout.py:557-563`

```python
def health_monitoring_pause(self):
    """暂停健康检查（offload 前）"""
    for monitor in self._health_monitors:
        monitor.pause()

def health_monitoring_resume(self):
    """恢复健康检查（onload 后）"""
    for monitor in self._health_monitors:
        monitor.resume()
```

---

### 6.4 check_weights()

**文件**: `slime/ray/rollout.py:565-566`

```python
def check_weights(self, action: str):
    """检查/快照引擎权重（用于调试）"""
    return ray.get([
        engine.check_weights.remote(action=action) 
        for engine in self.rollout_engines
    ])
```

**用途**:
- `action="snapshot"`: 保存权重快照
- `action="reset_tensors"`: 重置张量
- `action="compare"`: 比较权重差异

---

## 7. 辅助属性

### 7.1 server / _get_updatable_server()

**文件**: `slime/ray/rollout.py:438-454`

```python
@property
def server(self) -> RolloutServer | None:
    """默认 server（第一个模型）"""
    return next(iter(self.servers.values())) if self.servers else None

def _get_updatable_server(self) -> RolloutServer | None:
    """获取可更新权重的 server"""
    for srv in self.servers.values():
        if srv.update_weights:
            return srv
    return None
```

---

### 7.2 rollout_engines

**文件**: `slime/ray/rollout.py:456-459`

```python
@property
def rollout_engines(self):
    """获取所有 node-0 的引擎（用于权重更新）"""
    return [e for srv in self.servers.values() for e in srv.engines]
```

---

### 7.3 get_updatable_engines_and_lock()

**文件**: `slime/ray/rollout.py:461-473`

```python
def get_updatable_engines_and_lock(self):
    """获取可更新权重的引擎列表和锁"""
    srv = self._get_updatable_server()
    return (
        srv.engines if srv else [],
        self.rollout_engine_lock,  # 分布式锁
        srv.num_new_engines if srv else 0,
        srv.engine_gpu_counts,
        srv.engine_gpu_offsets,
    )
```

被 `actor.update_weights()` 调用用于获取引擎列表。

---

## 8. 内部计算方法

### 8.1 _compute_dynamic_global_batch_size()

**文件**: `slime/ray/rollout.py:610-635`

```python
def _compute_dynamic_global_batch_size(self, num_samples: int) -> int:
    """计算动态 global_batch_size，确保只进行一次训练步骤"""
    dp_size = self.train_parallel_config['dp_size']
    
    # 向下取整到 dp_size 的倍数
    dynamic_gbs = (num_samples // dp_size) * dp_size
    
    if dynamic_gbs == 0:
        dynamic_gbs = dp_size
    
    wasted = num_samples - dynamic_gbs  # 被丢弃的样本数
    
    return dynamic_gbs
```

**目的**: 当 `use_dynamic_global_batch_size=True` 时，调整 GBS 确保 `num_steps_per_rollout=1`。

---

## 方法分类总结

| 类别 | 方法 | 作用 |
|------|------|------|
| **核心流程** | `generate()` | 生成训练数据 |
| | `eval()` | 评估模型 |
| **内存管理** | `offload()` / `onload()` | GPU/CPU 切换 |
| | `onload_weights()` / `onload_kv()` | 分步加载 |
| **故障恢复** | `recover_updatable_engines()` | 恢复故障引擎 |
| | `clear_updatable_num_new_engines()` | 清除新引擎标记 |
| **数据处理** | `_get_rollout_data()` | 获取生成数据 |
| | `_convert_samples_to_train_data()` | 转换训练格式 |
| | `_split_train_data_by_dp()` | 按 DP 分割 |
| | `_post_process_rewards()` | 奖励后处理 |
| **持久化** | `save()` / `load()` | 数据状态保存/加载 |
| | `_save_debug_rollout_data()` | 调试数据保存 |
| **工具** | `get_num_rollout_per_epoch()` | 计算迭代次数 |
| | `dispose()` | 清理资源 |
| | `check_weights()` | 权重检查 |
| | `health_monitoring_pause/resume()` | 健康监控控制 |

## Follow-up Questions

- [ ] `call_rollout_fn` 是如何调用自定义 rollout 函数的？
- [ ] `Sample` 类的完整结构是什么？
- [ ] `DataIterator` 的工作原理？
- [ ] 动态 batch size 的具体实现？
