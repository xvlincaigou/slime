# Topic: Multi-Node Training

## Overview

Slime 使用 Ray 集群来管理多机器分布式训练。Ray 负责跨节点的资源调度和任务分发，Slime 在此基础上构建训练（Megatron）和推理（SGLang）的分布式架构。

## Ray Cluster Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           Ray Cluster                                    │
│                                                                          │
│  ┌──────────────┐     ┌──────────────┐     ┌──────────────┐              │
│  │   Node 1     │     │   Node 2     │     │   Node N     │              │
│  │  (Head)      │◄────┤   (Worker)   │◄────┤   (Worker)   │              │
│  │              │     │              │     │              │              │
│  │ GPU [0-7]    │     │ GPU [0-7]    │     │ GPU [0-7]    │              │
│  └──────────────┘     └──────────────┘     └──────────────┘              │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                     Placement Group (跨节点资源)                          │
│                                                                          │
│  Bundle [0-7]      Bundle [8-15]     Bundle [16-23]    ...               │
│  (Node 1)          (Node 2)          (Node 3)                            │
└─────────────────────────────────────────────────────────────────────────┘
```

## Cluster Setup

### Head Node

```bash
export MASTER_ADDR=<head_ip>

ray start --head \
    --node-ip-address ${MASTER_ADDR} \
    --num-gpus 8 \
    --dashboard-host=0.0.0.0 \
    --dashboard-port=8265
```

### Worker Nodes

```bash
ray start --address=<HEAD_IP>:6379 --num-gpus 8
```

## Multi-Node Resource Allocation

### 1. GPU Discovery

Ray 自动发现所有节点的 GPU：

```python
# Slime 不需要手动指定节点，Ray 会自动管理
# 只需要指定总资源需求
--actor-num-nodes 4      # 4 个节点用于训练
--actor-num-gpus-per-node 8  # 每节点 8 GPU
```

### 2. Placement Group 跨节点

```python
bundles = [{"GPU": 1, "CPU": 1} for _ in range(32)]  # 32 GPU 跨 4 节点
pg = placement_group(bundles, strategy="PACK")

# Ray 自动分布 bundles:
# Node 1: bundles 0-7
# Node 2: bundles 8-15
# Node 3: bundles 16-23
# Node 4: bundles 24-31
```

### 3. GPU 排序（跨节点一致性）

```python
def sort_key(x):
    index, node_ip, gpu_id = x
    node_ip_parts = list(map(int, node_ip.split(".")))
    return (node_ip_parts, gpu_id)

# 按 IP 排序，再按 GPU ID 排序
# 保证每次运行资源分配一致
```

## Configuration Examples

### 4-Node Colocate Training

```bash
# 4 节点 x 8 GPU = 32 GPU
--actor-num-nodes 4
--actor-num-gpus-per-node 8
--colocate

# 资源分配:
# - Actor: 32 GPUs (4 节点)
# - Rollout: 32 GPUs (共用)
```

### Multi-Node with Separate Rollout

```bash
# 3 节点：2 训练 + 1 推理
--actor-num-nodes 2
--actor-num-gpus-per-node 8
--rollout-num-gpus 8

# 资源分配:
# - Actor: 16 GPUs (Node 1-2)
# - Rollout: 8 GPUs (Node 3)
```

### PPO with Multi-Node Critic

```bash
# 4 节点：2 actor + 1 critic + 1 rollout
--actor-num-nodes 2
--actor-num-gpus-per-node 8
--critic-num-nodes 1
--critic-num-gpus-per-node 8
--rollout-num-gpus 8

# 资源分配:
# - Actor: 16 GPUs (Node 1-2)
# - Critic: 8 GPUs (Node 3)
# - Rollout: 8 GPUs (Node 4)
```

## Communication

### Training Internal (Megatron)

```python
# Tensor Parallel (TP): 同一节点内
# Data Parallel (DP): 跨节点梯度同步
# Pipeline Parallel (PP): 跨阶段

tp_size = mpu.get_tensor_model_parallel_world_size()
dp_size = mpu.get_data_parallel_world_size()
pp_size = mpu.get_pipeline_model_parallel_world_size()
```

### Cross-Component (Ray)

```python
# Actor -> RolloutManager (跨节点 Ray 调用)
rollout_data = ray.get(rollout_manager.generate.remote())

# Weight Update (跨节点)
actor_model.update_weights()
```

## Related Questions

- [Q003: 多机器资源分配详解](../questions/Q003-multi-node-resource-allocation.md)

## Code References

- `slime/ray/placement_group.py` - 跨节点资源分配
- `slime/ray/actor_group.py` - 分布式训练组
- `slime/backends/megatron_utils/actor.py` - Megatron 分布式初始化
