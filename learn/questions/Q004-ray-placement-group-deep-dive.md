---
date: 2026-04-04
question_id: Q004
topics: ["ray", "placement-group", "ray-core", "resource-scheduling"]
related_files:
  - slime/ray/placement_group.py
---

# Question

用 `_create_placement_group` 这个函数的例子来介绍一下 ray 的 `placement_group` 是怎么工作的，以及这个函数里面涉及到的 ray 的一些函数都是什么意思。

# Answer

## 1. Ray Placement Group 核心概念

### 1.1 什么是 Placement Group？

**Placement Group** 是 Ray 中用于**原子性资源分配**的机制。它允许你预留一组资源，确保这些资源要么全部被分配，要么一个都不分配。

```
┌─────────────────────────────────────────────────────────────────┐
│                     Placement Group                              │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  Bundle 0     Bundle 1     Bundle 2     Bundle 3         │   │
│  │  {GPU:1,      {GPU:1,      {GPU:1,      {GPU:1,          │   │
│  │   CPU:1}      CPU:1}      CPU:1}      CPU:1}             │   │
│  │     │            │            │            │             │   │
│  │     └────────────┴────────────┴────────────┘             │   │
│  │                    Placement Group                        │   │
│  │                  (原子性分配单元)                          │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                  │
│  特性：                                                           │
│  1. 原子性：所有 bundle 要么全部分配，要么都不分配                    │
│  2. 预留：资源会一直被保留，直到 placement group 被删除             │
│  3. 调度策略：PACK/SPREAD/STRICT_PACK/STRICT_SPREAD              │
└─────────────────────────────────────────────────────────────────┘
```

---

## 2. `_create_placement_group` 函数详解

### 2.1 完整代码

```python
def _create_placement_group(num_gpus):
    """Create a placement group with the specified number of GPUs."""
    # Step 1: 创建资源包
    bundles = [{"GPU": 1, "CPU": 1} for _ in range(num_gpus)]
    
    # Step 2: 创建 Placement Group
    pg = placement_group(bundles, strategy="PACK")
    num_bundles = len(bundles)
    
    # Step 3: 等待 PG 就绪
    ray.get(pg.ready())
    
    # Step 4: 在每个 bundle 上启动 InfoActor 获取 GPU 信息
    info_actors = []
    for i in range(num_bundles):
        info_actors.append(
            InfoActor.options(
                scheduling_strategy=PlacementGroupSchedulingStrategy(
                    placement_group=pg,
                    placement_group_bundle_index=i,
                )
            ).remote()
        )
    
    # Step 5: 收集所有 GPU 的 IP 和 ID
    gpu_ids = ray.get([actor.get_ip_and_gpu_id.remote() for actor in info_actors])
    
    # Step 6: 清理 actors
    for actor in info_actors:
        ray.kill(actor)
    
    # Step 7: 排序并生成映射
    bundle_infos = [(i, gpu_ids[i][0], gpu_ids[i][1]) for i in range(num_bundles)]
    sorted_bundle_infos = sorted(bundle_infos, key=sort_key)
    pg_reordered_bundle_indices = [info[0] for info in sorted_bundle_infos]
    pg_reordered_gpu_ids = [gpu_ids[info[0]][1] for info in sorted_bundle_infos]
    
    return pg, pg_reordered_bundle_indices, pg_reordered_gpu_ids
```

---

## 3. 逐行解释 Ray API

### 3.1 `placement_group(bundles, strategy)`

**文件**: `from ray.util.placement_group import placement_group`

```python
bundles = [{"GPU": 1, "CPU": 1} for _ in range(num_gpus)]
pg = placement_group(bundles, strategy="PACK")
```

**作用**：创建一个 Placement Group。

**参数**：
- `bundles`: 资源包列表，每个 bundle 是一个资源字典
  - `[{"GPU": 1, "CPU": 1}, {"GPU": 1, "CPU": 1}, ...]`
  - 表示需要 N 个 bundle，每个包含 1 GPU + 1 CPU
- `strategy`: 调度策略
  - `"PACK"`: **紧凑放置**（默认）- 优先填满一个节点，再分配到下一个
  - `"SPREAD"`: **分散放置** - 尽可能分散到不同节点
  - `"STRICT_PACK"`: 严格紧凑 - 所有 bundle 必须在同一个节点
  - `"STRICT_SPREAD"`: 严格分散 - 每个 bundle 必须在不同节点

**示例**（8 个 bundle，2 节点，每个节点 4 GPU）：
```
PACK 策略：
Node 1: [B0, B1, B2, B3]
Node 2: [B4, B5, B6, B7]

SPREAD 策略：
Node 1: [B0, B2, B4, B6]
Node 2: [B1, B3, B5, B7]
```

**返回值**：`PlacementGroup` 对象，用于后续的资源调度。

---

### 3.2 `ray.get(pg.ready())`

```python
ray.get(pg.ready())
```

**作用**：**阻塞等待**，直到 Placement Group 的资源全部被分配完成。

**解释**：
- `pg.ready()` 返回一个 `ObjectRef`，表示 PG 的就绪状态
- `ray.get()` 阻塞直到该 ObjectRef 完成
- 这确保了后续代码在资源确实可用后才执行

**为什么需要？**
```python
# 错误示例：不等待就使用
pg = placement_group(bundles)
# 此时资源可能还没分配好，调度可能失败

# 正确示例：等待就绪
pg = placement_group(bundles)
ray.get(pg.ready())  # 确保资源已分配
# 现在可以安全使用
```

---

### 3.3 `@ray.remote(num_gpus=1)` - Actor 装饰器

**文件**: `slime/ray/placement_group.py:14-17`

```python
@ray.remote(num_gpus=1)
class InfoActor:
    def get_ip_and_gpu_id(self):
        return ray.util.get_node_ip_address(), ray.get_gpu_ids()[0]
```

**作用**：定义一个 Ray Actor，并指定资源需求。

**参数**：
- `num_gpus=1`: 这个 Actor 需要 1 个 GPU
- `num_cpus=1`: 这个 Actor 需要 1 个 CPU（可以省略，默认 1）

**解释**：
- `@ray.remote` 将 Python 类转换为 Ray Actor 类
- 当调用 `.remote()` 时，Ray 会创建 Actor 实例并分配资源
- 这里的资源需求（1 GPU）必须与 bundle 中的资源匹配

---

### 3.4 `Actor.options(scheduling_strategy=...)`

```python
InfoActor.options(
    scheduling_strategy=PlacementGroupSchedulingStrategy(
        placement_group=pg,
        placement_group_bundle_index=i,
    )
).remote()
```

**作用**：指定 Actor 的**调度策略**，让它运行在特定的 Placement Group 的特定 bundle 上。

**参数**：
- `placement_group=pg`: 指定使用哪个 Placement Group
- `placement_group_bundle_index=i`: 指定使用 PG 中的第 i 个 bundle

**可视化**：
```
Placement Group:
┌─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┐
│ B0  │ B1  │ B2  │ B3  │ B4  │ B5  │ B6  │ B7  │
│{GPU}│{GPU}│{GPU}│{GPU}│{GPU}│{GPU}│{GPU}│{GPU}│
└──┬──┴──┬──┴──┬──┴──┬──┴──┬──┴──┬──┴──┬──┴──┬──┘
   │     │     │     │     │     │     │     │
   ▼     ▼     ▼     ▼     ▼     ▼     ▼     ▼
 InfoActor0 ... (每个运行在对应的 bundle 上)
```

**为什么需要？**
- 确保每个 Actor 获得确定的资源
- 可以在 Actor 内部获取实际的 GPU ID
- 建立逻辑索引到物理 GPU 的映射

---

### 3.5 `ray.get_gpu_ids()`

```python
def get_ip_and_gpu_id(self):
    return ray.util.get_node_ip_address(), ray.get_gpu_ids()[0]
```

**作用**：获取当前 Actor 被分配的 GPU ID 列表。

**返回值**：`List[int]`，例如 `[0]`、`[2]`、`[0, 1, 2, 3]`（多 GPU）

**解释**：
- Ray 会给每个 Actor 分配一个或多个 GPU
- `ray.get_gpu_ids()` 返回分配给该 Actor 的物理 GPU 编号
- 注意：这是**物理 GPU ID**，不是 Ray 的 bundle 索引

**示例**：
```
机器有 8 个 GPU: [0, 1, 2, 3, 4, 5, 6, 7]

Actor 运行在 bundle 0 (Node 1, GPU 2 上):
ray.get_gpu_ids() -> [2]

Actor 运行在 bundle 1 (Node 1, GPU 5 上):
ray.get_gpu_ids() -> [5]

Actor 运行在 bundle 4 (Node 2, GPU 0 上):
ray.get_gpu_ids() -> [0]
```

---

### 3.6 `ray.util.get_node_ip_address()`

```python
ray.util.get_node_ip_address()
```

**作用**：获取当前 Actor 所在节点的 IP 地址。

**返回值**：字符串，例如 `"192.168.1.10"`

**用途**：
- 确定 Actor 运行在哪个物理节点上
- 多机环境下用于节点识别和排序

---

### 3.7 `actor.get_ip_and_gpu_id.remote()`

```python
gpu_ids = ray.get([actor.get_ip_and_gpu_id.remote() for actor in info_actors])
```

**作用**：**异步调用** Actor 的方法，并获取结果。

**解释**：
- `.remote()`：异步调用，立即返回 `ObjectRef`，不等待执行完成
- `ray.get([...])`：批量获取多个 ObjectRef 的结果
- 这是一个**非阻塞调用**的集合，然后**阻塞等待**所有结果

**执行流程**：
```
1. 为每个 Actor 调用 .remote() -> 立即返回 ObjectRef
   
   Actor0.get_ip_and_gpu_id.remote() -> ObjectRef0
   Actor1.get_ip_and_gpu_id.remote() -> ObjectRef1
   ...

2. 批量获取结果（并行等待）
   
   ray.get([ObjectRef0, ObjectRef1, ...]) 
   -> [("192.168.1.10", 0), ("192.168.1.10", 1), ...]
```

---

### 3.8 `ray.kill(actor)`

```python
for actor in info_actors:
    ray.kill(actor)
```

**作用**：强制终止 Actor 并释放其占用的资源。

**解释**：
- InfoActor 只是用来获取 GPU 信息的临时 Actor
- 获取完信息后就可以销毁了
- 释放资源，避免占用 Placement Group 的容量

**注意**：
- `ray.kill()` 是强制终止，不会执行 Actor 的 `__del__`
- 用于清理不再需要的临时 Actor

---

## 4. 完整工作流程图解

### 4.1 步骤 1-3：创建并等待 PG

```python
bundles = [{"GPU": 1, "CPU": 1} for _ in range(4)]
pg = placement_group(bundles, strategy="PACK")
ray.get(pg.ready())
```

```
┌─────────────────────────────────────────────────────────────┐
│                     Ray Cluster                              │
│                                                              │
│  Node 1 (4 GPUs)              Node 2 (4 GPUs)               │
│  ┌─────────────────┐         ┌─────────────────┐            │
│  │ GPU 0 │ GPU 1   │         │ GPU 0 │ GPU 1   │            │
│  │ GPU 2 │ GPU 3   │         │ GPU 2 │ GPU 3   │            │
│  └─────────────────┘         └─────────────────┘            │
│                                                              │
└─────────────────────────────────────────────────────────────┘
                              │
                              │ placement_group(bundles, "PACK")
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    Placement Group                           │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  B0     B1     B2     B3                            │    │
│  │ {GPU} {GPU} {GPU} {GPU}                             │    │
│  │  │      │      │      │                             │    │
│  │  ▼      ▼      ▼      ▼                             │    │
│  │ GPU0   GPU1   GPU2   GPU3  (Node 1, PACK 策略填满)   │    │
│  └─────────────────────────────────────────────────────┘    │
│                                                              │
│  Status: READY (ray.get(pg.ready()) 完成)                   │
└─────────────────────────────────────────────────────────────┘
```

### 4.2 步骤 4-6：启动 InfoActor 获取 GPU 信息

```python
for i in range(4):
    info_actors.append(
        InfoActor.options(
            scheduling_strategy=PlacementGroupSchedulingStrategy(
                placement_group=pg,
                placement_group_bundle_index=i,
            )
        ).remote()
    )

gpu_ids = ray.get([actor.get_ip_and_gpu_id.remote() for actor in info_actors])
```

```
Placement Group:
┌─────────────────────────────────────────────────────┐
│  B0         B1         B2         B3                │
│  │          │          │          │                 │
│  ▼          ▼          ▼          ▼                 │
│ InfoActor0 InfoActor1 InfoActor2 InfoActor3         │
│  │          │          │          │                 │
│  └──────────┴──────────┴──────────┘                 │
│              ray.get([...])                          │
│                 │                                    │
│                 ▼                                    │
│  [("192.168.1.10", 0), ("192.168.1.10", 1),         │
│   ("192.168.1.10", 2), ("192.168.1.10", 3)]         │
└─────────────────────────────────────────────────────┘
```

### 4.3 步骤 7：排序生成映射

```python
bundle_infos = [(0, "192.168.1.10", 0), (1, "192.168.1.10", 1), 
                (2, "192.168.1.10", 2), (3, "192.168.1.10", 3)]
sorted_bundle_infos = sorted(bundle_infos, key=sort_key)
# 结果: 已经是排序好的

pg_reordered_bundle_indices = [0, 1, 2, 3]  # bundle 索引
pg_reordered_gpu_ids = [0, 1, 2, 3]          # 物理 GPU ID
```

**返回结果**：
- `pg`: Placement Group 对象，用于后续调度
- `pg_reordered_bundle_indices`: 逻辑索引 -> bundle 索引的映射
- `pg_reordered_gpu_ids`: 逻辑索引 -> 物理 GPU ID 的映射

---

## 5. 映射关系总结

```
逻辑索引 (rank)        Bundle 索引          物理 GPU ID
     │                      │                    │
     0 ──────────────────► B0 ────────────────► GPU 0 (Node 1)
     1 ──────────────────► B1 ────────────────► GPU 1 (Node 1)
     2 ──────────────────► B2 ────────────────► GPU 2 (Node 1)
     3 ──────────────────► B3 ────────────────► GPU 3 (Node 1)

为什么需要这些映射？
- 逻辑索引：代码中使用的 rank (0, 1, 2, 3)
- Bundle 索引：Ray Placement Group 中的位置
- 物理 GPU ID：CUDA 看到的 GPU 编号 (nvidia-smi 中的编号)
```

---

## Key Points

1. **Placement Group 是资源预留机制**：原子性分配一组资源
2. **Bundle 是资源分配单元**：每个 bundle 包含一组资源（GPU/CPU）
3. **PACK 策略紧凑放置**：优先填满一个节点，适合减少跨节点通信
4. **Actor + SchedulingStrategy**：将 Actor 绑定到特定的 bundle
5. **ray.get_gpu_ids()**：获取物理 GPU ID，建立逻辑到物理的映射
6. **ray.get(pg.ready())**：阻塞等待，确保资源就绪后才继续

## Follow-up Questions

- [ ] Ray Placement Group 的预留资源是如何实现的？
- [ ] 如果资源不足，`pg.ready()` 会永远等待吗？
- [ ] Placement Group 的调度策略对性能的影响？
- [ ] Ray 如何处理 Placement Group 的故障恢复？
