---
date: 2026-04-04
question_id: Q005
topics: ["rollout-manager", "sglang", "ray", "inference"]
related_files:
  - slime/ray/placement_group.py
  - slime/ray/rollout.py
---

# Question

回到最早的 `train` 函数。这一步 `rollout_manager, num_rollout_per_epoch = create_rollout_manager(args, pgs["rollout"])` 是在做什么？

# Answer

## 1. 函数作用概述

`create_rollout_manager` 是**初始化推理引擎管理器**的核心函数，它负责：

1. 创建 `RolloutManager` Ray Actor（协调推理的"指挥官"）
2. 启动 SGLang 推理引擎（实际生成文本的"工人"）
3. 计算训练迭代次数（从 epoch 转换为 rollout）
4. 初始化权重检查和内存管理

```
┌─────────────────────────────────────────────────────────────────┐
│                 create_rollout_manager                          │
│                                                                 │
│  1. 创建 RolloutManager Actor (Ray)                             │
│     └── options(num_cpus=1, num_gpus=0).remote(args, pg)        │
│                                                                 │
│  2. 计算 num_rollout_per_epoch (如果 num_rollout 未指定)         │
│     └── 数据集大小 // rollout_batch_size                        │
│                                                                 │
│  3. 可选：权重检查快照                                            │
│                                                                 │
│  4. 可选：offload rollout 引擎到 CPU                             │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 2. 代码逐行解析

**文件**: `slime/ray/placement_group.py:183-203`

```python
def create_rollout_manager(args, pg):
    # 第 1 步：创建 RolloutManager Ray Actor
    rollout_manager = RolloutManager.options(
        num_cpus=1,
        num_gpus=0,  # Manager 本身不需要 GPU
    ).remote(args, pg)
```

**关键点**:
- `RolloutManager` 是一个 Ray Actor，它本身不占用 GPU（`num_gpus=0`）
- 它通过 `pg`（Placement Group）来调度 SGLang 引擎的创建
- Manager 是"指挥官"，实际生成任务由 SGLang 引擎执行

---

```python
    # 第 2 步：计算每轮的 rollout 数量
    num_rollout_per_epoch = None
    if args.num_rollout is None:
        # 如果用户没有指定 num_rollout，从 num_epoch 计算
        num_rollout_per_epoch = ray.get(rollout_manager.get_num_rollout_per_epoch.remote())
        args.num_rollout = num_rollout_per_epoch * args.num_epoch
        assert args.num_rollout > 0
```

**计算逻辑**:
```python
# RolloutManager.get_num_rollout_per_epoch:
def get_num_rollout_per_epoch(self):
    assert self.args.rollout_global_dataset
    return len(self.data_source) // self.args.rollout_batch_size
```

**示例**:
```
数据集大小: 10000
rollout_batch_size: 32
num_epoch: 3

num_rollout_per_epoch = 10000 // 32 = 312
num_rollout = 312 * 3 = 936
```

---

```python
    # 第 3 步：权重检查（可选）
    if args.check_weight_update_equal:
        ray.get(rollout_manager.check_weights.remote(action="snapshot"))
        ray.get(rollout_manager.check_weights.remote(action="reset_tensors"))
```

**作用**: 在训练开始前保存权重的快照，用于后续验证权重更新是否正确。

---

```python
    # 第 4 步：offload rollout 引擎（可选）
    if args.offload_rollout:
        ray.get(rollout_manager.offload.remote())
```

**作用**: 如果开启了 colocate 模式，在训练前先将 SGLang 引擎 offload 到 CPU，释放 GPU 给训练使用。

---

## 3. RolloutManager 初始化详解

**文件**: `slime/ray/rollout.py:350-393`

```python
@ray.remote
class RolloutManager:
    """The class to run rollout and convert rollout data to training data."""

    def __init__(self, args, pg):
        configure_logger()

        self.pg = pg  # Placement Group，用于调度 SGLang 引擎
        self.args = args

        # 加载数据源（用于获取 prompts）
        data_source_cls = load_function(self.args.data_source_path)
        self.data_source = data_source_cls(args)

        # 加载 rollout 生成函数（默认是 sglang_rollout.generate_rollout）
        self.generate_rollout = load_function(self.args.rollout_function_path)
        self.eval_generate_rollout = load_function(self.args.eval_function_path)
```

**核心组件初始化**:
1. **Data Source**: 提供训练用的 prompts
2. **Rollout Function**: 定义如何生成文本（可自定义，如 retool 的多轮对话）
3. **Reward Post Process**: 奖励后处理函数（如 GRPO 的归一化）

---

```python
        # 启动 SGLang 推理服务器（最关键的一步）
        if self.args.debug_train_only:
            self.servers: dict[str, RolloutServer] = {}
        else:
            init_http_client(args)
            self.servers = start_rollout_servers(args, pg)
```

**关键分支**:
- `debug_train_only=True`: 不启动 SGLang，用预存数据训练
- 正常情况: 启动 SGLang 推理服务器

---

```python
        # 初始化健康监控（可选）
        self._health_monitors = []
        if not self.args.debug_train_only and self.args.use_fault_tolerance:
            for srv in self.servers.values():
                for group in srv.server_groups:
                    monitor = RolloutHealthMonitor(group, args)
                    monitor.start()
                    self._health_monitors.append(monitor)
```

**故障恢复**: 如果开启了 `use_fault_tolerance`，会启动健康监控，自动检测和恢复故障的 SGLang 引擎。

---

## 4. start_rollout_servers 详解

**文件**: `slime/ray/rollout.py:986-1118`

这是真正创建 SGLang 推理引擎的地方：

```python
def start_rollout_servers(args, pg) -> dict[str, RolloutServer]:
    """Start rollout servers: one per model, each with its own router."""

    # 解析 SGLang 配置（支持多模型）
    config = _resolve_sglang_config(args)

    for model_idx, model_cfg in enumerate(config.models):
        # 1. 启动路由（sglang_router）
        router_ip, router_port = _start_router(args, has_pd_disaggregation=has_pd)

        # 2. 创建 Server Groups（按 worker_type）
        for group_cfg in model_cfg.server_groups:
            group = _make_group(group_cfg, router_ip, router_port)

            # 3. 启动 SGLang 引擎
            handles, port_cursors = group.start_engines(port_cursors)
            all_init_handles.extend(handles)
            server_groups.append(group)

        # 4. 等待所有引擎初始化完成
        if all_init_handles:
            ray.get(all_init_handles)
```

### 4.1 创建过程图解

```
┌─────────────────────────────────────────────────────────────────┐
│                    start_rollout_servers                        │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ 1. _start_router()                                       │   │
│  │    └── 启动 sglang_router 进程                            │   │
│  │    └── 监听端口（如 30000）                               │   │
│  └─────────────────────────────────────────────────────────┘   │
│                              │                                  │
│                              ▼                                  │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ 2. _make_group()                                         │   │
│  │    └── 创建 ServerGroup                                  │   │
│  │    └── 确定 GPU 分配（在 Placement Group 中的位置）         │   │
│  └─────────────────────────────────────────────────────────┘   │
│                              │                                  │
│                              ▼                                  │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ 3. group.start_engines()                                 │   │
│  │    └── 创建多个 SGLangEngine Ray Actors                   │   │
│  │    └── 每个 Actor 占用 PG 中的一个 bundle                  │   │
│  │    └── 初始化 SGLang 推理引擎                              │   │
│  └─────────────────────────────────────────────────────────┘   │
│                              │                                  │
│                              ▼                                  │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ 4. ray.get(all_init_handles)                             │   │
│  │    └── 等待所有引擎初始化完成                              │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 4.2 SGLangEngine Actor 创建

**文件**: `slime/ray/rollout.py:127-142`

```python
RolloutRayActor = ray.remote(SGLangEngine)

rollout_engine = RolloutRayActor.options(
    num_cpus=num_cpus,
    num_gpus=num_gpus,  # 通常 0.2，因为 SGLang 使用多进程
    scheduling_strategy=PlacementGroupSchedulingStrategy(
        placement_group=pg,
        placement_group_bundle_index=reordered_bundle_indices[gpu_index],
    ),
    runtime_env={"env_vars": {...}},
).remote(
    self.args,
    rank=global_rank,
    worker_type=self.worker_type,
    base_gpu_id=base_gpu_id,
    ...
)
```

**关键点**:
- 每个 SGLang Engine 是一个 Ray Actor
- 占用 Placement Group 中的一个 bundle
- `num_gpus=0.2` 表示每个引擎占用 0.2 个 GPU（SGLang 内部使用多进程管理）

---

## 5. RolloutManager 的核心职责

创建完成后，`RolloutManager` 在训练循环中承担以下职责：

### 5.1 generate() - 生成训练数据

**文件**: `slime/ray/rollout.py:479-492`

```python
def generate(self, rollout_id):
    # 1. 调用自定义 rollout 函数生成数据
    data, metrics = self._get_rollout_data(rollout_id=rollout_id)

    # 2. 转换为训练数据格式
    data = self._convert_samples_to_train_data(data)

    # 3. 按 DP 大小分割数据
    return self._split_train_data_by_dp(data, self.train_parallel_config["dp_size"])
```

### 5.2 eval() - 评估模型

**文件**: `slime/ray/rollout.py:494-503`

```python
def eval(self, rollout_id):
    result = call_rollout_fn(self.eval_generate_rollout, self.args, rollout_id, self.data_source, evaluation=True)
    # 记录评估指标...
```

### 5.3 权重更新相关

```python
def offload(self):      # 将 SGLang 权重 offload 到 CPU
def onload_weights(self):  # 将权重加载回 GPU
def recover_updatable_engines(self):  # 故障恢复
```

---

## 6. 完整流程图

```
┌──────────────────────────────────────────────────────────────────────┐
│                           train.py                                    │
│                                                                       │
│  1. create_placement_groups()                                         │
│     └── 创建 Placement Group（预留 GPU 资源）                          │
│                                                                       │
│  2. create_rollout_manager(args, pgs["rollout"])  ◄── 我们在这里       │
│     ├── 创建 RolloutManager Actor                                     │
│     ├── 启动 SGLang 推理引擎（使用 PG 中的 GPU）                       │
│     └── 计算 num_rollout_per_epoch                                    │
│                                                                       │
│  3. create_training_models()                                          │
│     └── 创建 Actor/Critic 训练模型                                     │
│                                                                       │
│  4. for rollout_id in range(num_rollout):                             │
│     ├── rollout_manager.generate()  ──► 调用 SGLang 生成数据            │
│     ├── actor_model.async_train()   ──► 训练 Actor                     │
│     └── actor_model.update_weights() ──► 同步权重到 SGLang             │
│                                                                       │
└──────────────────────────────────────────────────────────────────────┘
```

---

## Key Points

1. **RolloutManager 是"指挥官"**: 本身不占 GPU，负责协调 SGLang 引擎
2. **SGLang 引擎是"工人"**: 实际执行推理生成，占用 PG 中的 GPU
3. **懒加载 Data Source**: 在 `__init__` 中初始化，但不加载全部数据
4. **支持多模型**: 可以配置多个模型，每个有自己的 router 和 engines
5. **故障恢复**: 可选的健康监控，自动恢复故障引擎

## Follow-up Questions

- [ ] SGLang 引擎的具体初始化流程？
- [ ] Router 如何负载均衡请求到多个引擎？
- [ ] 权重如何从 Megatron 同步到 SGLang？
- [ ] `_get_rollout_data` 的具体实现？
- [ ] 故障恢复机制是如何工作的？
