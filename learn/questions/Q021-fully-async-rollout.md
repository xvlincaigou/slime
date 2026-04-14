---
date: 2026-04-14
question_id: Q021
topics: ["rollout", "fully-async", "async", "training"]
related_files:
  - examples/fully_async/fully_async_rollout.py
  - examples/fully_async/run-qwen3-4b-fully_async.sh
---

# Question

`examples/fully_async` 这种完全异步的 rollout 是怎么实现的？有什么优点？

# Answer

## 一句话总结

Fully Async 通过**全局持续运行的异步 Worker 线程**实现真正的 rollout 生成与训练完全解耦，训练进程只从队列中取已完成的 rollout，实现最大化的流水线并行。

## 架构对比

### 1. 同步训练 (`train.py`)

```
Rollout 0 → Train 0 → Rollout 1 → Train 1 → ...
[=======] [=======] [=======]   [=======]
串行执行，无重叠
```

### 2. 半异步训练 (`train_async.py`)

```
Rollout 0 →
          Train 0 | Rollout 1 →
                    Train 1 | Rollout 2 →
          ↑ 相邻 rollout/train 重叠
```

### 3. 完全异步 (`fully_async`)

```
Continuous Worker:  [R0][R1][R2][R3][R4][R5][R6]... (持续生成)
                              ↓
Training Process:   [Train 0][Train 1][Train 2]... (从队列取)
                              ↑ 完全解耦，永不等待
```

## 核心实现

### AsyncRolloutWorker 类

`fully_async_rollout.py:37-147`

```python
class AsyncRolloutWorker:
    """
    使用线程而非进程的简化异步 rollout worker
    支持持续运行，独立于 rollout 函数生命周期
    """

    def __init__(self, args, data_buffer, concurrency=10):
        self.data_buffer = data_buffer  # 直接引用 data_buffer
        self.output_queue = queue.Queue(maxsize=1000)  # 持续输出队列
        self.state = GenerateState(args)

    async def continuous_worker_loop(self):
        """持续工作循环 - 不断从 data_buffer 获取数据并处理"""
        active_tasks = set()
        max_concurrent_tasks = self.args.rollout_batch_size

        while self.running:
            # 清理已完成任务
            done_tasks = {task for task in active_tasks if task.done()}
            active_tasks -= done_tasks

            # 启动新任务直到达到并发上限
            while len(active_tasks) < max_concurrent_tasks:
                samples = self.data_buffer.get_samples(1)

                for group in samples:
                    # 创建新的异步任务
                    task = asyncio.create_task(
                        generate_and_rm_group(
                            self.args,
                            group,
                            sampling_params=self.state.sampling_params.copy(),
                            evaluation=False,
                        )
                    )
                    # 添加完成回调，结果放入 output_queue
                    task.add_done_callback(make_callback(group_id))
                    active_tasks.add(task)

            await asyncio.sleep(1)  # 避免忙等待
```

### 关键设计

**全局 Worker 单例模式** (`fully_async_rollout.py:17-35`)

```python
_global_worker = None
_worker_lock = threading.Lock()

def get_global_worker(args, data_buffer):
    """获取或创建全局 worker"""
    global _global_worker
    with _worker_lock:
        if _global_worker is None or not _global_worker.worker_thread.is_alive():
            _global_worker = AsyncRolloutWorker(args, data_buffer, concurrency=args.sglang_server_concurrency)
            _global_worker.start()
        return _global_worker
```

**训练进程只需从队列取结果** (`fully_async_rollout.py:149-250`)

```python
async def generate_rollout_async(args, rollout_id: int, data_buffer) -> list[list[Sample]]:
    # 获取全局 worker（已在后台持续运行）
    worker = get_global_worker(args, data_buffer)

    target_data_size = args.rollout_batch_size
    data = []

    # 主循环：从全局 worker 的输出队列收集结果
    while len(data) < target_data_size:
        # 收集已完成的结果
        completed = worker.get_completed_groups()

        for group_id, group in completed:
            # 处理完成的数据
            data.append(group)

        # 短暂休眠避免忙等待
        if not completed:
            await asyncio.sleep(0.01)

    return data
```

## 优点

| 特性 | train.py | train_async.py | fully_async |
|------|----------|----------------|-------------|
| **Rollout-Train 重叠** | ❌ 无 | ✅ 相邻重叠 | ✅ 完全解耦 |
| **GPU 利用率** | 低 | 中 | 高 |
| **等待时间** | 需等待每个 rollout | 等待第一个 rollout | 基本无需等待 |
| **实现复杂度** | 简单 | 中等 | 较高 |
| **适用场景** | 通用 | 资源充足 | 极致性能 |

### 具体优势

**1. 零等待训练**

训练进程调用 `generate_rollout_fully_async` 时，worker 已经预生成了大量 rollout，直接 dequeue 即可开始训练。

**2. 最大化 GPU 利用率**

- SGLang Server 持续满负荷运行
- 训练 GPU 不再等待数据，始终有数据可训练
- 避免了半异步模式中训练等数据的间隙

**3. 自适应吞吐量**

```python
# 自动调节：如果训练比生成快，队列会被耗尽，worker 自动加速生成
# 如果生成比训练快，队列积压，worker 自然减速
while len(data) < target_data_size:
    completed = worker.get_completed_groups()
    # ...
```

**4. 异常处理与重试**

```python
# 如果 group 中有被 abort 的 sample，返回到 buffer 重试
any_aborted = any([sample.status == Sample.Status.ABORTED for sample in group])
if any_aborted:
    data_buffer.add_samples([group])
    print(f"Returned aborted group {group_id} to data buffer")
```

## 使用方式

**启动脚本** (`run-qwen3-4b-fully_async.sh:41`)

```bash
ROLLOUT_ARGS=(
   --rollout-function-path fully_async_rollout.generate_rollout_fully_async
   # ... 其他参数
)

ray job submit --address="http://127.0.0.1:8265" \
   -- python3 train_async.py \
   # ...
```

**注意**：虽然脚本名是 `train_async.py`，但通过 `--rollout-function-path` 指定了完全异步的 rollout 函数。

## 限制与注意事项

1. **不支持 evaluation 模式**
   ```python
   if evaluation:
       raise ValueError("Evaluation mode not supported in simple async rollout")
   ```

2. **需要 global dataset**
   ```python
   assert args.rollout_global_dataset
   ```

3. **线程安全**：使用 `threading.Lock()` 保护全局 worker 访问

4. **资源占用**：持续运行的 worker 会持续占用 SGLang server 资源

## Key Points

1. **核心思想**：真正的完全解耦——有一个独立的后台进程持续生成 rollout，训练进程只是消费者
2. **实现机制**：`asyncio + threading.Queue` 实现异步任务管理和结果传递
3. **最大优势**：训练进程永不等待 rollout 生成，实现极致的流水线并行
4. **适用场景**：当 rollout 生成是瓶颈时（如生成长序列、使用复杂 reward model）效果最佳

## Code References

- `fully_async_rollout.py:37` - `AsyncRolloutWorker` 类定义
- `fully_async_rollout.py:52` - `continuous_worker_loop` 持续工作循环
- `fully_async_rollout.py:149` - `generate_rollout_async` 主函数
- `fully_async_rollout.py:253` - `generate_rollout_fully_async` 入口函数
- `run-qwen3-4b-fully_async.sh:41` - 使用方式示例

## Follow-up Questions

- [ ] 如何监控 fully_async 的队列深度和吞吐量？
- [ ] fully_async 对内存占用有什么影响？
- [ ] 在什么场景下 fully_async 不如半异步？
