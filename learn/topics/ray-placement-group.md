# Topic: Ray Placement Group

## Overview

Ray Placement Group 是 Ray 中用于**原子性资源分配**的核心机制。它允许用户预留一组资源，确保这些资源要么全部被分配，要么一个都不分配。

## Core Concepts

### Bundle

Bundle 是 Placement Group 中的**资源分配单元**：

```python
bundle = {"GPU": 1, "CPU": 1}  # 1 GPU + 1 CPU
bundles = [bundle] * 8          # 8 个这样的资源包
```

### Placement Group

PG 是资源的**逻辑分组**：

```python
from ray.util.placement_group import placement_group

pg = placement_group(
    bundles=bundles,
    strategy="PACK"  # 紧凑放置策略
)
```

### Scheduling Strategy

| 策略 | 说明 | 适用场景 |
|------|------|----------|
| `PACK` | 紧凑放置，优先填满一个节点 | 减少跨节点通信 |
| `SPREAD` | 分散放置，尽可能分散到不同节点 | 提高容错性 |
| `STRICT_PACK` | 所有 bundle 必须在同一个节点 | 需要共享内存 |
| `STRICT_SPREAD` | 每个 bundle 必须在不同节点 | 最大容错 |

## Workflow

```
┌──────────────────────────────────────────────────────────────┐
│ 1. Define Bundles                                             │
│    bundles = [{"GPU":1}, {"GPU":1}, ...]                     │
└──────────────────────────────┬───────────────────────────────┘
                               │
┌──────────────────────────────▼───────────────────────────────┐
│ 2. Create Placement Group                                     │
│    pg = placement_group(bundles, strategy="PACK")            │
└──────────────────────────────┬───────────────────────────────┘
                               │
┌──────────────────────────────▼───────────────────────────────┐
│ 3. Wait for Ready                                             │
│    ray.get(pg.ready())  # 阻塞等待资源分配完成                 │
└──────────────────────────────┬───────────────────────────────┘
                               │
┌──────────────────────────────▼───────────────────────────────┐
│ 4. Schedule Actors                                            │
│    Actor.options(                                             │
│        scheduling_strategy=PlacementGroupSchedulingStrategy( │
│            placement_group=pg,                                │
│            placement_group_bundle_index=i                    │
│        )                                                      │
│    ).remote()                                                 │
└──────────────────────────────────────────────────────────────┘
```

## Key Ray APIs

### Creating PG

```python
from ray.util.placement_group import placement_group

pg = placement_group(
    bundles=[{"GPU": 1, "CPU": 1}] * 4,
    strategy="PACK"
)
```

### Waiting for Ready

```python
ray.get(pg.ready())  # 阻塞直到资源分配完成
```

### Scheduling Actor on PG

```python
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

@ray.remote(num_gpus=1)
class MyActor:
    pass

actor = MyActor.options(
    scheduling_strategy=PlacementGroupSchedulingStrategy(
        placement_group=pg,
        placement_group_bundle_index=0  # 使用第 0 个 bundle
    )
).remote()
```

### Getting GPU Info

```python
@ray.remote(num_gpus=1)
class InfoActor:
    def get_info(self):
        ip = ray.util.get_node_ip_address()
        gpu_id = ray.get_gpu_ids()[0]
        return ip, gpu_id
```

### Killing Actor

```python
ray.kill(actor)  # 强制终止 Actor，释放资源
```

## Use Cases

### Slime 中的使用

1. **Actor Training Group**: 在 PG 上分配 Megatron 训练 actors
2. **Rollout Engines**: 在 PG 上分配 SGLang 推理引擎
3. **Resource Mapping**: 建立逻辑 rank 到物理 GPU 的映射

## Related Questions

- [Q004: Ray Placement Group 深度解析](../questions/Q004-ray-placement-group-deep-dive.md)

## Code References

- `slime/ray/placement_group.py` - `_create_placement_group` 实现
- [Ray Placement Group 官方文档](https://docs.ray.io/en/latest/ray-core/scheduling/placement-group.html)
