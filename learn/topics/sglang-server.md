# Topic: SGLang Server

## Overview

Slime 使用 SGLang 作为推理引擎，`start_rollout_servers` 负责启动和管理这些引擎。支持灵活的部署模式，包括普通模式、PD 分离、EPD 和多模型部署。

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     RolloutServer                                │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  Router (sglang_router)                                  │   │
│  │  - HTTP 入口                                              │   │
│  │  - 负载均衡                                               │   │
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

## Components

### ServerGroup

一组同构的 SGLang 引擎：

```python
@dataclass
class ServerGroup:
    all_engines: list  # SGLangEngine Ray Actors
    num_gpus_per_engine: int  # TP size
    worker_type: str  # "regular" | "prefill" | "decode" | "encoder"
    needs_offload: bool  # 是否支持 colocate offload
```

### RolloutServer

一个模型的完整部署：

```python
@dataclass
class RolloutServer:
    server_groups: list[ServerGroup]
    router_ip: str
    router_port: int
    update_weights: bool  # 是否接收训练权重
```

## Deployment Modes

### 1. Regular Mode

```yaml
models:
  - name: "default"
    server_groups:
      - worker_type: "regular"
        num_gpus: 8
        num_gpus_per_engine: 2
```

### 2. PD Disaggregation

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

### 3. EPD (Encoder-Prefill-Decode)

```yaml
models:
  - name: "vlm"
    server_groups:
      - worker_type: "encoder"   # 处理图像
        num_gpus: 2
      - worker_type: "prefill"   # 处理文本前缀
        num_gpus: 4
      - worker_type: "decode"    # 生成文本
        num_gpus: 8
```

## Startup Flow

```
1. _resolve_sglang_config()
   └── 解析模型配置

2. _start_router()
   └── 启动 sglang_router
   └── 监听 HTTP 端口

3. for each ServerGroup:
   ├── _make_group()
   │   └── 创建 ServerGroup
   │   └── 确定 GPU 分配
   │
   └── group.start_engines()
       ├── 创建 SGLangEngine Ray Actors
       ├── 分配端口 (server/nccl/dist)
       └── engine.init.remote()

4. ray.get(all_init_handles)
   └── 等待所有引擎就绪
```

## Related Questions

- [Q006: start_rollout_servers 深度解析](../questions/Q006-start-rollout-servers-deep-dive.md)

## Code References

- `slime/ray/rollout.py:986` - `start_rollout_servers()`
- `slime/ray/rollout.py:38` - `ServerGroup` class
- `slime/ray/rollout.py:211` - `RolloutServer` class
