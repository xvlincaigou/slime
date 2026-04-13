---
date: 2026-04-04
question_id: Q006
topics: ["sglang", "rollout-server", "ray", "inference", "pd-disaggregation"]
related_files:
  - slime/ray/rollout.py
---

# Question

详细解释一下 `start_rollout_servers` 函数

# Answer

## 1. 函数概述

`start_rollout_servers` 是 Slime 中**启动 SGLang 推理服务器**的核心函数。它负责：

1. **解析配置**：从 args 或 YAML 文件读取 SGLang 配置
2. **启动 Router**：为每个模型启动一个 sglang_router（负载均衡器）
3. **创建 Server Groups**：按 worker_type 分组管理引擎
4. **启动 Engines**：在 Ray 集群中创建 SGLang 推理引擎
5. **支持高级特性**：PD 分离、EPD、多模型部署

---

## 2. 核心数据结构

### 2.1 架构层次

```
┌─────────────────────────────────────────────────────────────────┐
│                     RolloutServer                                │
│  (一个模型对应一个 RolloutServer，包含一个 router)                 │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  Router (sglang_router)                                  │   │
│  │  - IP: 192.168.1.10:30000                                │   │
│  │  - 负责负载均衡请求到各个 engine                          │   │
│  └─────────────────────────────────────────────────────────┘   │
│                              │                                   │
│  ┌───────────────────────────┼─────────────────────────────┐   │
│  │                           ▼                             │   │
│  │  ┌────────────┐  ┌────────────┐  ┌────────────┐        │   │
│  │  │ServerGroup │  │ServerGroup │  │ServerGroup │        │   │
│  │  │ (prefill)  │  │  (decode)  │  │ (regular)  │        │   │
│  │  └──────┬─────┘  └──────┬─────┘  └──────┬─────┘        │   │
│  │         │               │               │              │   │
│  │  ┌──────▼─────┐  ┌──────▼─────┐  ┌──────▼─────┐       │   │
│  │  │ Engines    │  │ Engines    │  │ Engines    │       │   │
│  │  │ [GPU 0-1]  │  │ [GPU 2-3]  │  │ [GPU 4-7]  │       │   │
│  │  └────────────┘  └────────────┘  └────────────┘       │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 ServerGroup

**文件**: `slime/ray/rollout.py:38-209`

```python
@dataclasses.dataclass
class ServerGroup:
    """一组同构的 SGLang 引擎，共享相同的配置"""
    
    args: Any
    pg: Any  # Placement Group
    all_engines: list  # 所有引擎（包括多节点）
    num_gpus_per_engine: int  # 每个引擎的 GPU 数（TP size）
    worker_type: str  # "regular", "prefill", "decode", "encoder", "placeholder"
    rank_offset: int  # 在全局中的起始 rank
    gpu_offset: int  # 在 Placement Group 中的起始偏移
    needs_offload: bool  # 是否需要 offload（colocate 模式）
    router_ip: str
    router_port: int
```

### 2.3 RolloutServer

**文件**: `slime/ray/rollout.py:211-348`

```python
@dataclasses.dataclass
class RolloutServer:
    """一个模型部署，包含多个 Server Groups"""
    
    server_groups: list[ServerGroup]
    router_ip: str
    router_port: int
    model_name: str
    update_weights: bool  # 是否接收来自训练的权重更新
```

---

## 3. 函数流程详解

### 3.1 步骤 1：解析 SGLang 配置

**文件**: `slime/ray/rollout.py:999 + 1121-1142`

```python
config = _resolve_sglang_config(args)
```

```python
def _resolve_sglang_config(args) -> SglangConfig:
    # 方式 1：从 YAML 文件加载
    if args.sglang_config is not None:
        return SglangConfig.from_yaml(args.sglang_config)
    
    # 方式 2：从 prefill_num_servers 推断（PD 分离）
    if args.prefill_num_servers is not None:
        return SglangConfig.from_prefill_num_servers(args)
    
    # 方式 3：默认配置（单模型，regular 模式）
    return SglangConfig(
        models=[
            ModelConfig(
                name="default",
                server_groups=[
                    ServerGroupConfig(worker_type="regular", num_gpus=args.rollout_num_gpus)
                ]
            )
        ]
    )
```

**配置示例**（YAML）：
```yaml
models:
  - name: "qwen3-4b"
    update_weights: true
    server_groups:
      - worker_type: "regular"
        num_gpus: 8
        num_gpus_per_engine: 2  # TP=2
```

---

### 3.2 步骤 2：为每个模型启动 Router

**文件**: `slime/ray/rollout.py:1009-1018 + 908-959`

```python
for model_idx, model_cfg in enumerate(config.models):
    has_pd = model_cfg.has_pd_disaggregation
    router_ip, router_port = _start_router(args, has_pd_disaggregation=has_pd, force_new=(model_idx > 0))
```

```python
def _start_router(args, has_pd_disaggregation=False, force_new=False):
    """启动 sglang_router 进程"""
    
    # 分配端口
    router_ip = get_host_info()[1]
    router_port = find_available_port(random.randint(3000, 4000))
    
    # 创建 RouterArgs
    router_args = RouterArgs.from_cli_args(args)
    router_args.host = router_ip
    router_args.port = router_port
    router_args.prometheus_port = find_available_port(random.randint(4000, 5000))
    
    # PD 分离特殊配置
    if has_pd_disaggregation:
        router_args.pd_disaggregation = True
        router_args.disable_circuit_breaker = True
    
    # 启动 router 进程
    process = multiprocessing.Process(target=run_router, args=(router_args,))
    process.daemon = True
    process.start()
    
    time.sleep(3)  # 等待启动
    return router_ip, router_port
```

**Router 作用**：
- 接收生成请求（HTTP）
- 负载均衡到后端 SGLang 引擎
- 支持 PD 分离路由（prefill → decode）

---

### 3.3 步骤 3：创建 Server Groups

**文件**: `slime/ray/rollout.py:1025-1061`

```python
def _make_group(group_cfg, router_ip, router_port, overrides_extra=None):
    """创建一个 ServerGroup"""
    
    # 计算引擎数量
    gpus_per_engine = group_cfg.num_gpus_per_engine
    num_gpu_per_engine_local = min(gpus_per_engine, args.num_gpus_per_node)
    num_engines = group_cfg.num_gpus // num_gpu_per_engine_local
    
    # 计算是否需要 offload（colocate 模式下）
    group_abs_start = rollout_pg_offset + gpu_offset
    needs_offload = args.offload_rollout and group_abs_start < megatron_num_gpus
    
    # 创建 ServerGroup
    group = ServerGroup(
        args=args,
        pg=pg,
        all_engines=[None] * num_engines,
        num_gpus_per_engine=gpus_per_engine,
        num_new_engines=0,
        worker_type=group_cfg.worker_type,
        rank_offset=engine_offset,
        gpu_offset=gpu_offset,
        sglang_overrides=overrides,
        needs_offload=needs_offload,
        router_ip=router_ip,
        router_port=router_port,
    )
    
    # 更新偏移
    engine_offset += num_engines
    gpu_offset += group_cfg.num_gpus
    return group
```

---

### 3.4 步骤 4：启动 SGLang 引擎

**文件**: `slime/ray/rollout.py:70-176 (ServerGroup.start_engines)`

```python
def start_engines(self, port_cursors=None):
    """创建 Ray actors 并启动引擎"""
    
    # 创建 Ray Actor 类
    RolloutRayActor = ray.remote(SGLangEngine)
    
    rollout_engines = []
    for i in range(len(self.all_engines)):
        global_rank = self.rank_offset + i
        
        # 计算 GPU 索引
        gpu_index = self.gpu_offset + i * num_gpu_per_engine
        base_gpu_id = int(reordered_gpu_ids[gpu_index])
        
        # 创建调度策略（绑定到特定 bundle）
        scheduling_strategy = PlacementGroupSchedulingStrategy(
            placement_group=pg,
            placement_group_bundle_index=reordered_bundle_indices[gpu_index],
        )
        
        # 创建 SGLang Engine Actor
        rollout_engine = RolloutRayActor.options(
            num_cpus=0.2,
            num_gpus=0.2,  # SGLang 内部管理 GPU
            scheduling_strategy=scheduling_strategy,
            runtime_env={"env_vars": {...}},  # SGLang 环境变量
        ).remote(
            self.args,
            rank=global_rank,
            worker_type=self.worker_type,
            base_gpu_id=base_gpu_id,
            sglang_overrides=self.sglang_overrides,
        )
        
        rollout_engines.append((global_rank, rollout_engine))
        self.all_engines[i] = rollout_engine
```

**关键点**：
- `num_gpus=0.2`：Ray 只分配 0.2 GPU，实际 GPU 使用由 SGLang 内部管理
- `PlacementGroupSchedulingStrategy`：确保 Actor 绑定到正确的 bundle
- `base_gpu_id`：告诉 SGLang 使用哪个物理 GPU

---

### 3.5 步骤 5：分配端口并初始化

**文件**: `slime/ray/rollout.py:151-176`

```python
# 分配地址和端口
addr_and_ports, port_cursors = _allocate_rollout_engine_addr_and_ports_normal(...)
# 返回：
# {
#   0: {"host": "192.168.1.10", "port": 15000, "nccl_port": 15001, "dist_init_addr": "192.168.1.10:15002"},
#   1: {...},
#   ...
# }

# 异步初始化所有引擎
init_handles = [
    engine.init.remote(
        **(addr_and_ports[rank]),
        router_ip=self.router_ip,
        router_port=self.router_port,
    )
    for rank, engine in rollout_engines
]

return init_handles, port_cursors
```

**分配的端口**：
- `port`：SGLang 服务端口（接收生成请求）
- `nccl_port`：NCCL 通信端口（TP 同步）
- `dist_init_addr`：分布式初始化地址（多节点）

---

### 3.6 步骤 6：等待引擎就绪

**文件**: `slime/ray/rollout.py:1097-1105`

```python
all_init_handles = []
for group_cfg in model_cfg.server_groups:
    group = _make_group(group_cfg, router_ip, router_port)
    handles, port_cursors = group.start_engines(port_cursors)
    all_init_handles.extend(handles)
    server_groups.append(group)

# 等待所有引擎初始化完成
if all_init_handles:
    ray.get(all_init_handles)
```

**`ray.get(all_init_handles)`**：阻塞等待所有 SGLang 引擎初始化完成（加载模型、启动服务）。

---

## 4. 特殊模式

### 4.1 PD 分离（Prefill-Decode Disaggregation）

```
┌─────────────────────────────────────────────────────────────┐
│                     PD 分离架构                              │
│                                                              │
│  ┌──────────────┐         ┌──────────────┐                 │
│  │   Prefill    │ ──────► │    Decode    │                 │
│  │   Engines    │  RDMA   │   Engines    │                 │
│  │  (计算密集)   │  传输   │  (内存密集)   │                 │
│  │   TP=2       │ KV Cache │   TP=4       │                 │
│  └──────────────┘         └──────────────┘                 │
│         ▲                        ▲                          │
│         └────────┐    ┌──────────┘                          │
│                  ▼    ▼                                     │
│            ┌──────────────┐                                 │
│            │    Router    │                                 │
│            │  (PD 路由)    │                                 │
│            └──────────────┘                                 │
└─────────────────────────────────────────────────────────────┘
```

**配置**：
```yaml
models:
  - name: "qwen3-4b"
    server_groups:
      - worker_type: "prefill"
        num_gpus: 4
        num_gpus_per_engine: 2
      - worker_type: "decode"
        num_gpus: 8
        num_gpus_per_engine: 4
```

---

### 4.2 EPD（Encoder-Prefill-Decode）

**文件**: `slime/ray/rollout.py:1063-1094`

```python
if has_epd:
    # Phase 1: 先启动 encoder groups
    encoder_urls = []
    for group_cfg in model_cfg.server_groups:
        if group_cfg.worker_type == "encoder":
            group = _make_group(...)
            handles, _ = group.start_engines(port_cursors)
            ray.get(handles)
            urls = ray.get([e.get_url.remote() for e in group.engines])
            encoder_urls.extend(urls)
    
    # Phase 2: 启动 prefill，注入 encoder URLs
    for group_cfg in model_cfg.server_groups:
        if group_cfg.worker_type == "prefill":
            overrides_extra = {
                "language_only": True,
                "encoder_urls": encoder_urls,
            }
            group = _make_group(..., overrides_extra=overrides_extra)
```

**EPD 流程**：
1. Encoder 处理图像/多模态输入
2. Prefill 处理文本前缀
3. Decode 生成后续文本

---

### 4.3 多模型部署

```python
servers = {}
for model_idx, model_cfg in enumerate(config.models):
    # 每个模型有自己的 router
    router_ip, router_port = _start_router(..., force_new=(model_idx > 0))
    
    # 创建该模型的 Server Groups
    server_groups = [...]
    
    # 创建 RolloutServer
    servers[model_cfg.name] = RolloutServer(
        server_groups=server_groups,
        router_ip=router_ip,
        router_port=router_port,
        model_name=model_cfg.name,
        update_weights=model_cfg.update_weights,
    )
```

**使用场景**：
- 主模型（Actor）+ 参考模型（Ref）+ 奖励模型（RM）
- 不同模型可以有不同的配置（TP size、GPU 数）

---

## 5. 完整流程图

```
┌──────────────────────────────────────────────────────────────────────┐
│                       start_rollout_servers                          │
│                                                                      │
│  1. _resolve_sglang_config(args)                                     │
│     └── 解析配置（YAML / args / 默认）                                │
│                                                                      │
│  2. for each model in config.models:                                 │
│     │                                                                │
│     ├── 2.1 _start_router()                                          │
│     │      └── 启动 sglang_router 进程                               │
│     │      └── 监听端口（如 30000）                                   │
│     │                                                                │
│     ├── 2.2 for each group in model.server_groups:                   │
│     │      │                                                         │
│     │      ├── _make_group()                                         │
│     │      │   └── 创建 ServerGroup                                  │
│     │      │   └── 确定 GPU 分配                                     │
│     │      │                                                         │
│     │      └── group.start_engines()                                 │
│     │          ├── 创建 SGLangEngine Ray Actors                      │
│     │          ├── 分配端口（server/nccl/dist）                       │
│     │          └── engine.init.remote()                              │
│     │                                                                │
│     ├── 2.3 ray.get(all_init_handles)                                │
│     │      └── 等待所有引擎初始化完成                                │
│     │                                                                │
│     └── 2.4 servers[model.name] = RolloutServer(...)                 │
│                                                                      │
│  3. return servers                                                   │
└──────────────────────────────────────────────────────────────────────┘
```

---

## Key Points

1. **分层架构**：RolloutServer → ServerGroup → Engines
2. **多模型支持**：每个模型独立的 router 和 engine groups
3. **灵活配置**：通过 YAML 或 args 配置 worker_type、TP size
4. **PD 分离**：支持 prefill/decode 分离部署，优化吞吐
5. **EPD 支持**：多模态场景下的 encoder-prefill-decode 分离
6. **端口管理**：自动分配不冲突的端口（server/nccl/dist）

## Follow-up Questions

- [ ] SGLangEngine 的具体初始化流程？
- [ ] Router 的负载均衡策略？
- [ ] PD 分离中的 KV Cache 传输机制？
- [ ] 多模型场景下的权重更新如何区分？
- [ ] Engine 故障恢复的详细流程？
