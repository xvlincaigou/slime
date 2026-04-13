# Topic: RolloutManager

## Overview

RolloutManager 是 Slime 中负责**推理数据生成**的核心组件。它管理 SGLang 推理引擎，协调数据生成、评估和权重同步。

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     RolloutManager                               │
│                    (Ray Actor, 无 GPU)                           │
│                                                                  │
│  ┌──────────────────┐  ┌──────────────────┐                     │
│  │   DataSource     │  │  RolloutFunction │                     │
│  │   (prompts)      │  │  (generate)      │                     │
│  └──────────────────┘  └──────────────────┘                     │
│           │                     │                               │
│           └──────────┬──────────┘                               │
│                      ▼                                           │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │               SGLang RolloutServers                      │    │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐              │    │
│  │  │ Router   │──│ Engine 0 │  │ Engine 1 │  ...         │    │
│  │  │ (30000)  │  │ (GPU 0)  │  │ (GPU 1)  │              │    │
│  │  └──────────┘  └──────────┘  └──────────┘              │    │
│  └─────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
```

## Key Components

### 1. RolloutManager (Ray Actor)

**创建**: `create_rollout_manager()`
```python
rollout_manager = RolloutManager.options(
    num_cpus=1,
    num_gpus=0,
).remote(args, pg)
```

**特点**:
- 本身不占用 GPU
- 通过 `pg` 调度 SGLang 引擎
- 协调数据生成和评估

### 2. Data Source

提供训练用的 prompts:
```python
data_source_cls = load_function(args.data_source_path)
self.data_source = data_source_cls(args)
```

### 3. Rollout Function

定义如何生成文本:
```python
self.generate_rollout = load_function(args.rollout_function_path)
# 默认: slime.rollout.sglang_rollout.generate_rollout
```

### 4. RolloutServers

SGLang 推理引擎组:
```python
self.servers = start_rollout_servers(args, pg)
# 包含 router 和多个 engines
```

## Lifecycle

### 初始化

```python
# 1. 创建 RolloutManager
rollout_manager = RolloutManager.options(...).remote(args, pg)

# 2. 在 __init__ 中启动 SGLang 服务器
if not args.debug_train_only:
    self.servers = start_rollout_servers(args, pg)
```

### 训练循环

```python
for rollout_id in range(num_rollout):
    # 生成数据
    rollout_data = ray.get(rollout_manager.generate.remote(rollout_id))
    
    # 训练...
    
    # 评估
    if should_eval(rollout_id):
        ray.get(rollout_manager.eval.remote(rollout_id))
```

### 清理

```python
# 停止健康监控
rollout_manager.dispose.remote()
```

## Key Methods

| Method | Purpose |
|--------|---------|
| `generate(rollout_id)` | 生成训练数据 |
| `eval(rollout_id)` | 评估模型 |
| `offload()` | Offload SGLang 到 CPU |
| `onload_weights()` | 加载权重到 GPU |
| `recover_updatable_engines()` | 故障恢复 |
| `get_num_rollout_per_epoch()` | 计算每轮 rollout 数 |

## Configuration

### 关键参数

```bash
# Rollout 配置
--rollout-num-gpus 8              # SGLang 使用的 GPU 数
--rollout-num-gpus-per-engine 2   # 每个引擎的 GPU 数（TP size）
--rollout-batch-size 32           # 每轮 prompts 数

# 自定义函数
--rollout-function-path slime.rollout.sglang_rollout.generate_rollout
--custom-generate-function-path generate_with_retool.generate

# 故障恢复
--use-fault-tolerance
--rollout-health-check-interval 30
```

## Related Questions

- [Q005: create_rollout_manager 详解](../questions/Q005-create-rollout-manager.md)

## Code References

- `slime/ray/placement_group.py:183` - `create_rollout_manager()`
- `slime/ray/rollout.py:350` - `RolloutManager` class
- `slime/ray/rollout.py:986` - `start_rollout_servers()`
