---
date: 2026-04-04
question_id: Q007
topics: ["fault-tolerance", "health-monitor", "sglang", "ray"]
related_files:
  - slime/utils/health_monitor.py
  - slime/ray/rollout.py
---

# Question

介绍一下 `RolloutHealthMonitor` 这个类

# Answer

## 1. 类概述

`RolloutHealthMonitor` 是 Slime 的**故障容错机制**核心组件，用于监控 SGLang 推理引擎的健康状态，自动检测和恢复故障引擎。

```
┌─────────────────────────────────────────────────────────────────┐
│                    RolloutHealthMonitor                          │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  职责                                                    │   │
│  │  1. 定期检查 SGLang 引擎健康状态                          │   │
│  │  2. 检测无响应或崩溃的引擎                                │   │
│  │  3. 自动 kill 故障引擎，标记为 None                       │   │
│  │  4. 支持 pause/resume（配合 offload/onload）              │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  生命周期                                                │   │
│  │  start()  →  启动监控线程（初始 paused 状态）              │   │
│  │  resume() →  恢复检查（engine onload 后调用）             │   │
│  │  pause()  →  暂停检查（engine offload 前调用）            │   │
│  │  stop()   →  停止监控线程（训练结束时调用）               │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

---

## 2. 初始化

**文件**: `slime/utils/health_monitor.py:23-33`

```python
class RolloutHealthMonitor:
    def __init__(self, server_group, args):
        self._server_group = server_group  # 要监控的 ServerGroup
        
        # 线程控制
        self._thread = None          # 监控线程
        self._stop_event = None      # 停止信号
        self._pause_event = None     # 暂停信号（set=暂停，clear=运行）
        
        # 配置参数
        self._check_interval = args.rollout_health_check_interval   # 检查间隔（默认 30s）
        self._check_timeout = args.rollout_health_check_timeout     # 超时时间（默认 30s）
        self._check_first_wait = args.rollout_health_check_first_wait  # 首次等待（默认 0s）
        
        # 状态
        self._need_first_wait = True     # 每次 resume 后需要首次等待
        self._is_checking_enabled = False  # 是否启用检查
```

**参数说明**：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `rollout_health_check_interval` | 30.0 | 健康检查间隔（秒）|
| `rollout_health_check_timeout` | 30.0 | 健康检查超时（秒）|
| `rollout_health_check_first_wait` | 0.0 | resume 后首次等待（给大模型准备时间）|

---

## 3. 生命周期方法

### 3.1 start() - 启动监控

**文件**: `slime/utils/health_monitor.py:35-59`

```python
def start(self) -> bool:
    """启动健康监控线程（在 RolloutManager 初始化时调用）"""
    
    # 如果没有引擎，不启动监控
    if not self._server_group.all_engines:
        return False
    
    # 创建事件和线程
    self._stop_event = threading.Event()
    self._pause_event = threading.Event()
    self._pause_event.set()  # 初始状态：暂停（等待 resume）
    
    self._thread = threading.Thread(
        target=self._health_monitor_loop,
        name="RolloutHealthMonitor",
        daemon=True,  # 守护线程，主进程退出时自动结束
    )
    self._thread.start()
    return True
```

**调用时机**：
```python
# RolloutManager.__init__
if self.args.use_fault_tolerance:
    for group in srv.server_groups:
        monitor = RolloutHealthMonitor(group, args)
        monitor.start()  # 启动监控（但处于 paused 状态）
```

---

### 3.2 resume() - 恢复检查

**文件**: `slime/utils/health_monitor.py:92-99`

```python
def resume(self) -> None:
    """恢复健康检查（在 engine onload 后调用）"""
    self._need_first_wait = True  # 需要首次等待
    self._pause_event.clear()     # 清除暂停标志，开始检查
    self._is_checking_enabled = True
```

**调用时机**：
```python
# RolloutManager.generate()
def generate(self, rollout_id):
    self.health_monitoring_resume()  # 生成前恢复监控
    ...
```

---

### 3.3 pause() - 暂停检查

**文件**: `slime/utils/health_monitor.py:84-90`

```python
def pause(self) -> None:
    """暂停健康检查（在 engine offload 前调用）"""
    self._pause_event.set()  # 设置暂停标志
    self._is_checking_enabled = False
```

**调用时机**：
```python
# RolloutManager.offload()
def offload(self):
    self.health_monitoring_pause()  # offload 前暂停监控
    ...
```

**为什么要 pause/resume？**
- 当 SGLang 引擎被 offload 到 CPU 时，无法接受健康检查请求
- 如果这时候检查，会误报故障
- 所以在 offload 前 pause，onload 后 resume

---

### 3.4 stop() - 停止监控

**文件**: `slime/utils/health_monitor.py:61-82`

```python
def stop(self) -> None:
    """停止监控线程（训练结束时调用）"""
    self._stop_event.set()
    self._pause_event.clear()  # 确保线程能退出等待
    
    # 等待线程结束（带超时）
    timeout = self._check_timeout + self._check_interval + 5
    self._thread.join(timeout=timeout)
    
    if self._thread.is_alive():
        logger.warning("健康监控线程未在超时内终止")
```

**调用时机**：
```python
# RolloutManager.dispose()
def dispose(self):
    for monitor in self._health_monitors:
        monitor.stop()  # 训练结束时停止所有监控
```

---

## 4. 健康检查循环

### 4.1 主循环

**文件**: `slime/utils/health_monitor.py:105-135`

```python
def _health_monitor_loop(self) -> None:
    while not self._stop_event.is_set():
        # 1. 等待暂停状态解除
        while self._pause_event.is_set() and not self._stop_event.is_set():
            self._stop_event.wait(timeout=0.5)  # 每 0.5s 检查一次
        
        if self._stop_event.is_set():
            break
        
        # 2. 首次等待（给大模型准备时间）
        if self._need_first_wait:
            if self._stop_event.wait(self._check_first_wait):
                break  # 等待期间收到 stop 信号
            if self._pause_event.is_set():
                continue  # 等待期间被 pause，下次 resume 重新等待
            self._need_first_wait = False
        
        # 3. 执行健康检查
        if not self._pause_event.is_set() and not self._stop_event.is_set():
            self._run_health_checks()
        
        # 4. 等待下一次检查间隔
        if self._stop_event.wait(self._check_interval):
            break
```

---

### 4.2 执行健康检查

**文件**: `slime/utils/health_monitor.py:137-143`

```python
def _run_health_checks(self) -> None:
    """遍历所有引擎执行健康检查"""
    for rollout_engine_id, engine in enumerate(self._server_group.engines):
        if self._stop_event.is_set() or self._pause_event.is_set():
            break
        self._check_engine_health(rollout_engine_id, engine)
```

---

### 4.3 检查单个引擎

**文件**: `slime/utils/health_monitor.py:145-158`

```python
def _check_engine_health(self, rollout_engine_id, engine) -> None:
    if engine is None:
        logger.info(f"跳过引擎 {rollout_engine_id}（已为 None）")
        return
    
    try:
        # 调用 SGLang 引擎的 health_generate 方法
        ray.get(engine.health_generate.remote(timeout=self._check_timeout))
        
    except Exception as e:
        # 超时或出错，判定为故障
        logger.error(f"引擎 {rollout_engine_id} 健康检查失败: {e}")
        self._kill_engine(rollout_engine_id=rollout_engine_id)
        
    else:
        logger.debug(f"引擎 {rollout_engine_id} 健康检查通过")
```

**health_generate**：
- SGLang 引擎提供的方法
- 内部执行一次简单的生成操作（如生成 1 个 token）
- 如果引擎卡住或崩溃，会超时或抛出异常

---

## 5. 故障处理

### 5.1 kill 故障引擎

**文件**: `slime/utils/health_monitor.py:160-177`

```python
def _kill_engine(self, rollout_engine_id: int):
    """kill 故障引擎，将其标记为 None"""
    
    # 计算该引擎占用的所有索引（考虑多节点）
    start_idx = rollout_engine_id * self._server_group.nodes_per_engine
    end_idx = (rollout_engine_id + 1) * self._server_group.nodes_per_engine
    
    for i in range(start_idx, end_idx):
        engine = self._server_group.all_engines[i]
        if engine:
            try:
                # 优雅关闭
                ray.get(engine.shutdown.remote())
                ray.kill(engine)
            except Exception as e:
                logger.warning(f"kill 引擎 {i} 失败: {e}")
        
        # 标记为 None（后续会触发恢复）
        self._server_group.all_engines[i] = None
```

---

### 5.2 故障恢复流程

```
1. HealthMonitor 检测到引擎故障
   └── _check_engine_health() 抛出异常
   
2. kill 故障引擎
   └── _kill_engine() 将引擎标记为 None
   
3. 下次训练步骤中恢复
   └── actor_model.update_weights()
       └── rollout_manager.recover_updatable_engines()
           └── RolloutServer.recover()
               └── group.start_engines() 重新创建引擎
```

**代码**：`slime/ray/rollout.py:528-549`

```python
def recover_updatable_engines(self):
    """恢复故障的 rollout 引擎"""
    srv = self._get_updatable_server()
    srv.recover()  # 调用 RolloutServer.recover()
    return (srv.engines, ..., srv.num_new_engines, ...)
```

---

## 6. 完整工作流程

```
训练开始时:
  RolloutManager.__init__()
    └── for each server_group:
        monitor = RolloutHealthMonitor(group, args)
        monitor.start()        # 启动监控线程（paused 状态）
        
生成数据前:
  rollout_manager.generate()
    └── self.health_monitoring_resume()
        └── monitor.resume()   # 恢复检查
    
生成过程中:
  监控线程循环:
    ├── 等待 interval（如 30s）
    ├── 对每个引擎调用 health_generate()
    ├── 如果超时/异常:
    │   └── _kill_engine()  # 标记为 None
    └── 继续循环
    
生成完成后 (colocate 模式):
  rollout_manager.offload()
    └── self.health_monitoring_pause()
        └── monitor.pause()    # 暂停检查
        
权重更新时 (如果引擎为 None):
  actor_model.update_weights()
    └── rollout_manager.recover_updatable_engines()
        └── 重新创建故障引擎
        
训练结束时:
  rollout_manager.dispose()
    └── monitor.stop()         # 停止监控线程
```

---

## Key Points

1. **独立线程**：使用 Python threading，不阻塞主训练流程
2. **pause/resume 机制**：配合 offload/onload，避免误报
3. **首次等待**：`first_wait` 给大模型（如 MoE）初始化时间
4. **标记为 None**：kill 后将引擎标记为 None，触发后续恢复
5. **多节点支持**：`_kill_engine` 会处理多节点引擎的所有分片

## Follow-up Questions

- [ ] `health_generate` 在 SGLang 内部是如何实现的？
- [ ] 故障恢复时如何保持权重一致性？
- [ ] 多模型场景下各模型的健康监控是独立的吗？
- [ ] 健康检查间隔和超时的最佳实践？
