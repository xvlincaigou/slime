---
date: 2026-04-12
question_id: Q014
topics: ["training", "megatron", "actor", "distributed-training", "weight-sync"]
related_files:
  - slime/backends/megatron_utils/actor.py
---

# Question

讲解一下 `slime/backends/megatron_utils/actor.py` 中的 `MegatronTrainRayActor` 类。

# Answer

## 一句话总结

`MegatronTrainRayActor` 是 Megatron 训练的核心 Actor 类，封装了完整的模型初始化、训练循环、权重管理和同步到 SGLang 的逻辑。

## 1. 初始化（`init`）

**文件**: `slime/backends/megatron_utils/actor.py:48-162`

```python
def init(self, args, role, with_ref=False, with_opd_teacher=False):
    # 1. 初始化分布式环境
    monkey_patch_torch_dist()
    init(args)
    
    # 2. 加载 HF config 和 tokenizer
    self.hf_config = AutoConfig.from_pretrained(args.hf_checkpoint)
    self.tokenizer = AutoTokenizer.from_pretrained(args.hf_checkpoint)
    
    # 3. 初始化模型和优化器
    (self.model, self.optimizer, self.opt_param_scheduler, loaded_rollout_id) = \
        initialize_model_and_optimizer(args, role)
    
    # 4. 设置权重备份器（用于切换 actor/ref/old_actor/teacher）
    self.weights_backuper = TensorBackuper.create(...)
    self.weights_backuper.backup("actor")
    
    # 5. 加载参考模型（用于 KL 散度）
    if with_ref:
        self.load_other_checkpoint("ref", args.ref_load)
    
    # 6. 创建权重更新器（同步到 SGLang）
    update_weight_cls = UpdateWeightFromTensor if args.colocate else UpdateWeightFromDistributed
    self.weight_updater = update_weight_cls(...)
    
    return loaded_rollout_id + 1  # 下一个 rollout_id
```

**关键组件**：
- `TensorBackuper` - 管理多个模型版本（actor/ref/old_actor/teacher）
- `UpdateWeightFromTensor/Distributed` - 将权重同步到 SGLang

## 2. 训练逻辑（`train` → `train_actor` / `train_critic`）

### 2.1 主入口

**文件**: `slime/backends/megatron_utils/actor.py:363-376`

```python
def train(self, rollout_id, rollout_data_ref):
    if self.args.offload_train:
        self.wake_up()  # 从 CPU 恢复
    
    rollout_data = self._get_rollout_data(rollout_data_ref)
    
    if self.role == "critic":
        return self.train_critic(rollout_id, rollout_data)
    else:
        return self.train_actor(rollout_id, rollout_data)
```

### 2.2 Actor 训练（PPO/GRPO）

**文件**: `slime/backends/megatron_utils/actor.py:406-513`

```python
def train_actor(self, rollout_id, rollout_data):
    # 1. 创建数据迭代器
    data_iterator, num_microbatches = get_iterator(...)
    
    # 2. 计算参考模型的 log_prob（用于 KL 散度）
    if "ref" in self.weights_backuper.backup_tags:
        self._switch_model("ref")
        rollout_data.update(self.compute_log_prob(..., store_prefix="ref_"))
    
    # 3. 计算旧策略的 log_prob
    self._switch_model("old_actor" if self.args.keep_old_actor else "actor")
    rollout_data.update(self.compute_log_prob(...))
    
    # 4. 同步 Critic 数据并计算 advantage
    if self.args.use_critic:
        sync_actor_critic_data(...)
    compute_advantages_and_returns(...)
    
    # 5. 执行训练
    train(rollout_id, self.model, self.optimizer, ...)
    
    # 6. 备份新权重
    self.weights_backuper.backup("actor")
```

### 2.3 Critic 训练

**文件**: `slime/backends/megatron_utils/actor.py:378-404`

```python
def train_critic(self, rollout_id, rollout_data):
    data_iterator, num_microbatches = get_data_iterator(...)
    
    # 1. 计算 value
    rollout_data.update(forward_only(get_values, ...))
    
    # 2. 同步 Actor 数据
    if rollout_id >= args.num_critic_only_steps:
        sync_actor_critic_data(...)
    
    # 3. 计算 advantage 和 return
    compute_advantages_and_returns(...)
    
    # 4. 训练
    self.args.loss_type = "value_loss"
    train(...)
```

## 3. 权重同步到 SGLang（`update_weights`）

**文件**: `slime/backends/megatron_utils/actor.py:542-595`

```python
def update_weights(self):
    # 1. 故障恢复（如果有引擎挂了）
    if self.args.use_fault_tolerance:
        ray.get(self.rollout_manager.recover_updatable_engines.remote())
    
    # 2. 获取 SGLang 引擎句柄
    rollout_engines, lock, num_new_engines, gpu_counts, gpu_offsets = ray.get(
        self.rollout_manager.get_updatable_engines_and_lock.remote()
    )
    
    # 3. 如果有新引擎，重新建立连接
    if num_new_engines > 0:
        self.weight_updater.connect_rollout_engines(...)
    
    # 4. 执行权重更新（NCCL/RDMA 传输）
    self.weight_updater.update_weights()
    
    # 5. 更新 old_actor（如果需要）
    if self.args.keep_old_actor:
        self.weights_backuper.copy(src_tag="rollout_actor", dst_tag="old_actor")
```

**两种更新方式**：
- `UpdateWeightFromTensor`（colocate 模式）：直接从内存拷贝
- `UpdateWeightFromDistributed`（分离模式）：通过 NCCL/RDMA 传输

## 4. Colocate 模式内存管理

**文件**: `slime/backends/megatron_utils/actor.py:165-185`

```python
def sleep(self):
    """Offload GPU 内存到 CPU"""
    assert self.args.offload_train
    clear_memory(clear_host_memory=True)
    destroy_process_groups()  # 销毁进程组
    torch_memory_saver.pause()  # 暂停内存占用

def wake_up(self):
    """从 CPU 恢复 GPU 内存"""
    assert self.args.offload_train
    torch_memory_saver.resume()  # 恢复内存
    reload_process_groups()  # 重建进程组
```

## 5. 模型切换（`_switch_model`）

**文件**: `slime/backends/megatron_utils/actor.py:262-266`

```python
def _switch_model(self, target_tag):
    """在 actor/ref/old_actor/teacher 之间切换"""
    self.weights_backuper.restore(target_tag)
    self._active_model_tag = target_tag
```

支持的模型标签：
- `"actor"` - 当前策略
- `"ref"` - 参考模型（用于 KL 散度）
- `"old_actor"` - 旧策略（用于 PPO clip）
- `"teacher"` - 教师模型（用于蒸馏）
- `"rollout_actor"` - 用于多步 rollout

## 6. Actor-Critic 连接

**文件**: `slime/backends/megatron_utils/actor.py:627-648`

```python
def connect_actor_critic(self, actor_handle=None, master_address=None, master_port=None):
    """建立 Actor 和 Critic 之间的进程组（用于 PPO）"""
    group_name = "actor_critic"
    world_size = 2
    self._actor_critic_groups = init_process_group(
        backend="nccl",
        init_method=f"tcp://{master_address}:{master_port}",
        world_size=world_size,
        rank=0 if self.role == "actor" else 1,
        group_name=group_name,
    )
```

## 流程图

```
┌─────────────────────────────────────────────────────────────┐
│                    MegatronTrainRayActor                    │
├─────────────────────────────────────────────────────────────┤
│  init()                                                     │
│    ├── initialize_model_and_optimizer()  ← 加载模型/优化器   │
│    ├── TensorBackuper.create()           ← 多版本权重管理    │
│    ├── load_other_checkpoint("ref")      ← 加载参考模型      │
│    └── UpdateWeight...()                 ← 创建权重更新器    │
├─────────────────────────────────────────────────────────────┤
│  train()                                                    │
│    ├── _get_rollout_data()               ← 获取训练数据      │
│    ├── train_actor() / train_critic()    ← 执行训练          │
│    │   ├── compute_log_prob()            ← 计算 log prob     │
│   │   ├── sync_actor_critic_data()       ← 同步数据          │
│   │   ├── compute_advantages_and_returns() ← 计算 advantage  │
│   │   └── train()                        ← Megatron 训练     │
│   └── weights_backuper.backup()          ← 备份新权重        │
├─────────────────────────────────────────────────────────────┤
│  update_weights()                                           │
│    ├── recover_updatable_engines()       ← 故障恢复          │
│    └── weight_updater.update_weights()   ← 同步到 SGLang     │
├─────────────────────────────────────────────────────────────┤
│  sleep() / wake_up()                     ← colocate 内存管理 │
└─────────────────────────────────────────────────────────────┘
```

## Key Points

- `MegatronTrainRayActor` 是 Megatron 训练的完整实现
- 通过 `TensorBackuper` 管理多个模型版本（actor/ref/old_actor/teacher）
- `update_weights()` 将权重同步到 SGLang（NCCL/RDMA）
- `sleep/wake_up` 实现 colocate 模式的 GPU/CPU 切换
- Actor 和 Critic 通过 `connect_actor_critic` 建立 NCCL 通信

## Code References

- `slime/backends/megatron_utils/actor.py:46` - 类定义
- `slime/backends/megatron_utils/actor.py:48` - `init` 方法
- `slime/backends/megatron_utils/actor.py:363` - `train` 方法
- `slime/backends/megatron_utils/actor.py:542` - `update_weights` 方法

## Follow-up Questions

- [ ] `TensorBackuper` 是如何实现权重备份和切换的？
- [ ] `UpdateWeightFromTensor` 和 `UpdateWeightFromDistributed` 的具体区别？
- [ ] `compute_advantages_and_returns` 的具体实现？
- [ ] `torch_memory_saver` 的工作原理？
