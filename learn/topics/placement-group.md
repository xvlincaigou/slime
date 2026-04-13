# Topic: Placement Group

## Overview

Placement Group 是 Slime 中用于管理 GPU 资源分配的核心机制，基于 Ray 的 Placement Group API 实现。它负责将物理 GPU 分配给训练（Actor/Critic）和推理（Rollout）组件。

## Key Concepts

### 1. Placement Group

Ray 的资源分配原语，保证一组资源被原子性地分配：

```python
bundles = [{"GPU": 1, "CPU": 1} for _ in range(num_gpus)]
pg = placement_group(bundles, strategy="PACK")
```

- **Bundle**: 资源分配的最小单位
- **Strategy**: `PACK`（紧凑）、`SPREAD`（分散）、`STRICT_PACK`（严格紧凑）

### 2. GPU 排序

为了保证多机环境下资源分配的一致性，Slime 会对 GPU 进行排序：

```python
# 按 (IP, GPU_ID) 排序
def sort_key(x):
    index, node_identifier, gpu_id = x
    node_ip_parts = list(map(int, ip_address.split(".")))
    return (node_ip_parts, gpu_id)
```

排序后生成 `reordered_bundle_indices`，建立逻辑索引到物理 bundle 的映射。

### 3. 资源分段

根据运行模式将 GPU 划分为不同用途：

```
┌─────────────────────────────────────────────────────────────┐
│                    Placement Group                           │
│  ┌────────────────┬────────────────┬────────────────────┐   │
│  │     Actor      │    Critic      │      Rollout       │   │
│  │   (Megatron)   │   (Megatron)   │     (SGLang)       │   │
│  └────────────────┴────────────────┴────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

## Run Modes

| 模式 | GPU 计算 | 特点 |
|------|----------|------|
| `debug_train_only` | Actor + Critic | 不启动 SGLang，用预存数据训练 |
| `debug_rollout_only` | Rollout | 不训练，只生成 rollout 数据 |
| `colocate` | Actor (+ Critic) | 训练+推理共用 GPU |
| 分离模式 | Actor + Critic + Rollout | 训练 GPU 和推理 GPU 独立 |

## Allocation Flow

```python
def create_placement_groups(args):
    # 1. 计算 GPU 数量
    num_gpus = calculate_num_gpus(args)
    
    # 2. 创建 Placement Group
    pg = placement_group(bundles, strategy="PACK")
    
    # 3. 获取并排序 GPU
    indices, gpu_ids = get_and_sort_gpus(pg)
    
    # 4. 分段划分
    return {
        "actor": (pg, actor_indices, actor_gpu_ids),
        "critic": (pg, critic_indices, critic_gpu_ids),
        "rollout": (pg, rollout_indices, rollout_gpu_ids),
    }
```

## Usage in Components

### RayTrainGroup

```python
actor = TrainRayActor.options(
    scheduling_strategy=PlacementGroupSchedulingStrategy(
        placement_group=pg,
        placement_group_bundle_index=reordered_bundle_indices[rank],
    ),
).remote(...)
```

### RolloutManager

```python
group = ServerGroup(
    pg=pg,
    gpu_offset=gpu_offset,  # 在 PG 中的起始偏移
    ...
)
```

## Related Questions

- [Q002: create_placement_groups 详解](../questions/Q002-placement-groups-allocation.md)

## Code References

- `slime/ray/placement_group.py` - 核心实现
- `slime/ray/actor_group.py` - 训练组使用 PG
- `slime/ray/rollout.py` - Rollout 使用 PG
