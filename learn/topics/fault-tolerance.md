# Topic: Fault Tolerance

## Overview

Slime 提供故障容错机制，通过 `RolloutHealthMonitor` 自动检测和恢复故障的 SGLang 推理引擎，确保训练流程的稳定性。

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    RolloutHealthMonitor                          │
│                    (独立监控线程)                                 │
│                                                                  │
│  ┌──────────────────┐         ┌──────────────────┐             │
│  │  Health Check    │────────►│  Kill Engine     │             │
│  │  (interval)      │  fail   │  (mark None)     │             │
│  └──────────────────┘         └────────┬─────────┘             │
│                                         │                       │
│                                         ▼                       │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  RolloutServer.recover()                                │   │
│  │  └── group.start_engines()  # 重新创建引擎               │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

## RolloutHealthMonitor

### Purpose

- 定期检查 SGLang 引擎健康状态
- 自动 kill 故障引擎
- 配合训练流程恢复引擎

### Lifecycle

```python
# 1. 启动监控（初始化）
monitor = RolloutHealthMonitor(server_group, args)
monitor.start()  # 启动线程，初始 paused 状态

# 2. 恢复检查（engine onload 后）
monitor.resume()

# 3. 暂停检查（engine offload 前）
monitor.pause()

# 4. 停止监控（训练结束）
monitor.stop()
```

### Configuration

```bash
--use-fault-tolerance                    # 启用故障容错
--rollout-health-check-interval 30.0     # 检查间隔（秒）
--rollout-health-check-timeout 30.0      # 超时时间（秒）
--rollout-health-check-first-wait 0.0    # 首次等待（秒）
```

## Recovery Flow

```
1. HealthMonitor 检测到引擎故障
   └── health_generate() timeout
   
2. kill 故障引擎
   └── _kill_engine() → engine = None
   
3. 下次 update_weights 时恢复
   └── recover_updatable_engines()
       └── RolloutServer.recover()
           └── start_engines() 重新创建
```

## Related Questions

- [Q007: RolloutHealthMonitor 详解](../questions/Q007-rollout-health-monitor.md)

## Code References

- `slime/utils/health_monitor.py` - `RolloutHealthMonitor` class
- `slime/ray/rollout.py:528` - `recover_updatable_engines()`
