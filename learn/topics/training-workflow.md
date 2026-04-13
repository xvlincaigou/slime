# Topic: Training Workflow

## Overview

Slime 的训练流程采用 **Ray 分布式架构**，将训练（Megatron）和推理（SGLang）分离，通过权重同步机制协作。

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                         train.py                             │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│                  Placement Groups                            │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐       │
│  │   Actor GPUs │  │  Critic GPUs │  │ Rollout GPUs │       │
│  │  (Megatron)  │  │  (Megatron)  │  │  (SGLang)    │       │
│  └──────────────┘  └──────────────┘  └──────────────┘       │
└─────────────────────────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│                     Training Loop                            │
│                                                              │
│   1. rollout_manager.generate()  →  生成训练数据             │
│   2. actor_model.async_train()   →  训练 Actor               │
│   3. critic_model.async_train()  →  训练 Critic (PPO)        │
│   4. actor_model.update_weights() → 同步权重到 SGLang        │
└─────────────────────────────────────────────────────────────┘
```

## Key Components

### 1. RayTrainGroup (`slime/ray/actor_group.py`)

管理一组 Ray actors 进行训练：
- **初始化**: 在 placement group 中分配 GPU
- **async_init**: 初始化模型、优化器、检查点
- **async_train**: 异步训练
- **update_weights**: 广播权重并同步到 SGLang

### 2. RolloutManager (`slime/ray/rollout.py`)

管理推理和 rollout 生成：
- **generate**: 生成训练数据
- **eval**: 评估模型
- **start_rollout_servers**: 启动 SGLang 服务器
- **_convert_samples_to_train_data**: 转换样本格式

### 3. 自定义 Rollout 函数

通过参数指定自定义函数：
```bash
--custom-generate-function-path generate_with_retool.generate
--custom-rm-path generate_with_retool.reward_func
```

## Data Flow

```
Prompt Dataset
      ↓
RolloutManager.generate()
      ↓
Custom Generate Function (e.g., retool.generate)
      ↓
SGLang Engine (generate + tool calls)
      ↓
Custom Reward Function (reward calculation)
      ↓
Convert to Train Data (tokens, rewards, loss_masks)
      ↓
Split by DP Size
      ↓
RayTrainGroup.async_train()
      ↓
Megatron Training (forward + backward + update)
      ↓
Update Weights to SGLang
```

## Memory Management

### Colocate Mode
当 `--colocate` 开启时，训练和推理共用 GPU：

```python
# 训练阶段
offload_rollout()      # SGLang offload 到 CPU
onload_train()         # Megatron onload 到 GPU
train()

# Rollout 阶段
offload_train()        # Megatron offload 到 CPU
onload_rollout()       # SGLang onload 到 GPU
generate()
```

### 非 Colocate Mode
训练 GPU 和推理 GPU 分离，无需 offload。

## Training Loop Pseudocode

```python
for rollout_id in range(num_rollout):
    # 1. Generate rollout data
    data = rollout_manager.generate(rollout_id)
    
    # 2. Train critic (if using PPO)
    if use_critic:
        critic_train_handle = critic_model.async_train(rollout_id, data)
    
    # 3. Train actor
    actor_model.async_train(rollout_id, data)
    
    # 4. Wait for training complete
    if use_critic:
        ray.get(critic_train_handle)
    
    # 5. Save checkpoint
    if should_save(rollout_id):
        save(rollout_id)
    
    # 6. Update weights to SGLang
    actor_model.update_weights()
    
    # 7. Evaluation
    if should_eval(rollout_id):
        rollout_manager.eval(rollout_id)
```

## Related Questions

- [Q001: retool 训练完整工作流程](../questions/Q001-training-workflow-retool.md)

## Code References

- `train.py` - 主入口
- `slime/ray/placement_group.py` - Placement group 创建
- `slime/ray/rollout.py` - RolloutManager
- `slime/ray/actor_group.py` - RayTrainGroup
- `examples/retool/generate_with_retool.py` - 自定义 rollout 示例
