---
date: 2026-04-04
question_id: Q001
topics: ["training-workflow", "retool", "ray", "rollout", "actor-training"]
related_files:
  - train.py
  - slime/ray/placement_group.py
  - slime/ray/rollout.py
  - slime/ray/actor_group.py
  - slime/utils/arguments.py
  - examples/retool/retool_qwen3_4b_rl.sh
  - examples/retool/generate_with_retool.py
---

# Question

从 `examples/retool` 为例子，讲解训练时整个工作流程。从入口开始一个函数一个函数的跳转来讲。

# Answer

## 1. 入口与启动

### 1.1 Shell 脚本入口
**文件**: `examples/retool/retool_qwen3_4b_rl.sh`

```bash
# 启动 Ray 集群
ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus 4

# 提交训练任务
ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 train.py \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node 4 \
   --colocate \
   ...
```

关键参数：
- `--colocate`: 训练和推理使用相同 GPU，需要 offload 切换
- `--custom-generate-function-path generate_with_retool.generate`: 自定义 rollout 生成函数
- `--custom-rm-path generate_with_retool.reward_func`: 自定义奖励函数

---

## 2. 主训练流程

### 2.1 train.py - 主入口
**文件**: `train.py:108-111`

```python
if __name__ == "__main__":
    args = parse_args()      # 解析参数
    train(args)              # 进入训练主函数
```

### 2.2 train() 函数
**文件**: `train.py:9-111`

这是训练的核心函数，按顺序执行：

```python
def train(args):
    # 1. 配置日志
    configure_logger()
    
    # 2. 创建 Ray Placement Groups（分配 GPU 资源）
    pgs = create_placement_groups(args)
    
    # 3. 初始化追踪（wandb/tensorboard）
    init_tracking(args)
    
    # 4. 创建 RolloutManager（包含 SGLang 推理引擎）
    rollout_manager, num_rollout_per_epoch = create_rollout_manager(args, pgs["rollout"])
    
    # 5. 获取 SGLang metrics 路由地址
    router_addr = ray.get(rollout_manager.get_metrics_router_addr.remote())
    update_tracking_open_metrics(args, router_addr)
    
    # 6. 创建 Actor 和 Critic 模型
    actor_model, critic_model = create_training_models(args, pgs, rollout_manager)
    
    # 7. 初始权重同步（Megatron → SGLang）
    actor_model.update_weights()
    
    # 8. 训练循环
    for rollout_id in range(args.start_rollout_id, args.num_rollout):
        # 8.1 执行 rollout 生成数据
        rollout_data_ref = ray.get(rollout_manager.generate.remote(rollout_id))
        
        # 8.2 训练 Critic（PPO 需要）
        if args.use_critic:
            critic_train_handle = critic_model.async_train(rollout_id, rollout_data_ref)
            
        # 8.3 训练 Actor
        if not args.critic_train_only:
            actor_model.async_train(rollout_id, rollout_data_ref)
            
        # 8.4 等待训练完成
        if args.use_critic:
            ray.get(critic_train_handle)
            
        # 8.5 保存检查点
        save(rollout_id)
        
        # 8.6 更新 SGLang 权重
        actor_model.update_weights()
```

---

## 3. 关键组件初始化

### 3.1 参数解析
**文件**: `slime/utils/arguments.py:1408-1451`

```python
def parse_args(add_custom_arguments=None):
    # 第一阶段：预解析模式
    pre = _pre_parse_mode()  # 解析 --train-backend, --debug-rollout-only 等
    
    # 第二阶段：解析 SGLang 参数
    sglang_ns = sglang_parse_args()
    
    # 第三阶段：解析 Megatron + Slime 参数
    args = megatron_parse_args(extra_args_provider=add_slime_arguments)
    
    # 参数验证
    slime_validate_args(args)
    megatron_validate_args(args)
    sglang_validate_args(args)
    
    return args
```

### 3.2 创建 Placement Groups
**文件**: `slime/ray/placement_group.py:79-119`

```python
def create_placement_groups(args):
    """为 actor 和 rollout 引擎创建 placement groups"""
    
    # 计算需要的 GPU 数量
    if args.colocate:
        # 训练+推理共用 GPU
        num_gpus = args.actor_num_nodes * args.actor_num_gpus_per_node
        rollout_offset = 0
    else:
        # 训练+推理分开
        num_gpus = (args.actor_num_nodes * args.actor_num_gpus_per_node + 
                   args.rollout_num_gpus)
    
    # 创建 placement group
    pg, bundle_indices, gpu_ids = _create_placement_group(num_gpus)
    
    return {
        "actor": (pg, actor_indices, actor_gpu_ids),
        "critic": (pg, critic_indices, critic_gpu_ids),  # 如果使用 PPO
        "rollout": (pg, rollout_indices, rollout_gpu_ids),
    }
```

### 3.3 创建 RolloutManager
**文件**: `slime/ray/placement_group.py:183-203` → `slime/ray/rollout.py:350-393`

```python
def create_rollout_manager(args, pg):
    # 创建 RolloutManager Actor（Ray remote）
    rollout_manager = RolloutManager.options(
        num_cpus=1, num_gpus=0,
    ).remote(args, pg)
    
    # 计算每轮 rollout 的数量
    num_rollout_per_epoch = ray.get(rollout_manager.get_num_rollout_per_epoch.remote())
    
    return rollout_manager, num_rollout_per_epoch
```

**RolloutManager.__init__**: `slime/ray/rollout.py:354-393`
```python
def __init__(self, args, pg):
    # 加载数据源
    data_source_cls = load_function(self.args.data_source_path)
    self.data_source = data_source_cls(args)
    
    # 加载自定义 rollout 函数（如 retool 的 generate）
    self.generate_rollout = load_function(self.args.rollout_function_path)
    self.eval_generate_rollout = load_function(self.args.eval_function_path)
    
    # 启动 SGLang 服务器
    self.servers = start_rollout_servers(args, pg)
```

### 3.4 启动 SGLang 服务器
**文件**: `slime/ray/rollout.py:986-1118`

```python
def start_rollout_servers(args, pg) -> dict[str, RolloutServer]:
    for model_cfg in config.models:
        # 启动路由
        router_ip, router_port = _start_router(args)
        
        # 创建 server groups
        for group_cfg in model_cfg.server_groups:
            group = _make_group(group_cfg, router_ip, router_port)
            
            # 启动引擎
            handles, port_cursors = group.start_engines(port_cursors)
            all_init_handles.extend(handles)
            server_groups.append(group)
        
        # 等待所有引擎初始化完成
        if all_init_handles:
            ray.get(all_init_handles)
```

### 3.5 创建训练模型
**文件**: `slime/ray/placement_group.py:133-180`

```python
def create_training_models(args, pgs, rollout_manager):
    # 创建 Actor 训练组
    actor_model = allocate_train_group(
        args=args,
        num_nodes=args.actor_num_nodes,
        num_gpus_per_node=args.actor_num_gpus_per_node,
        pg=pgs["actor"],
        role="actor",
    )
    
    # 创建 Critic 训练组（PPO 需要）
    if args.use_critic:
        critic_model = allocate_train_group(..., role="critic")
        critic_init_handle = critic_model.async_init(args, role="critic")
    
    # 初始化 Actor
    start_rollout_ids = ray.get(actor_model.async_init(args, role="actor"))
    
    # 连接 rollout_manager
    actor_model.set_rollout_manager(rollout_manager)
```

### 3.6 RayTrainGroup 初始化
**文件**: `slime/ray/actor_group.py:29-99`

```python
class RayTrainGroup:
    def __init__(self, args, num_nodes, num_gpus_per_node, pg, role="actor"):
        # 分配 GPU
        self._allocate_gpus_for_actor(pg, num_gpus_per_actor=0.4)
    
    def _allocate_gpus_for_actor(self, pg, num_gpus_per_actor):
        world_size = self._num_nodes * self._num_gpus_per_node
        
        # 根据 backend 选择 actor 实现
        from slime.backends.megatron_utils.actor import MegatronTrainRayActor
        actor_impl = MegatronTrainRayActor
        
        # 创建 Ray actors
        for rank in range(world_size):
            actor = TrainRayActor.options(
                num_cpus=num_gpus_per_actor,
                num_gpus=num_gpus_per_actor,
                scheduling_strategy=PlacementGroupSchedulingStrategy(...),
            ).remote(world_size, rank, master_addr, master_port)
            self._actor_handlers.append(actor)
```

---

## 4. 训练循环详解

### 4.1 Rollout 生成阶段
**文件**: `slime/ray/rollout.py:479-492`

```python
def generate(self, rollout_id):
    self.rollout_id = rollout_id
    
    # 调用自定义 rollout 函数（如 retool 的 generate）
    data = call_rollout_fn(
        self.generate_rollout, 
        self.args, 
        rollout_id, 
        self.data_source, 
        evaluation=False
    )
    
    # 转换为训练数据格式
    data = self._convert_samples_to_train_data(data.samples)
    
    # 按 DP 大小分割数据
    return self._split_train_data_by_dp(data, self.train_parallel_config["dp_size"])
```

### 4.2 Retool 的自定义 Rollout
**文件**: `examples/retool/generate_with_retool.py:215-350`

```python
async def generate(args, sample: Sample, sampling_params) -> Sample:
    """支持工具调用的自定义生成函数"""
    state = GenerateState(args)
    url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"
    
    # 格式化带工具的对话
    prompt = format_conversation_with_tools(prompt=sample.prompt, tools=tool_specs)
    prompt_tokens_ids = state.tokenizer(prompt)["input_ids"]
    
    # 多轮生成循环
    for turn in range(TOOL_CONFIGS["max_turns"]):
        # 发送请求到 SGLang
        payload = {"input_ids": current_token_ids, "sampling_params": sampling_params}
        output = await post(url, payload)
        
        # 执行工具调用
        next_obs, done = await execute_predictions(cur_response)
        
        if done:
            break
        
        # 添加 observation 到上下文
        response += next_obs
        response_token_ids += obs_tokens_ids
        loss_masks += [0] * len(obs_tokens_ids)  # observation 不计入 loss
    
    # 填充 Sample 对象
    sample.tokens = prompt_tokens_ids + response_token_ids
    sample.response_length = len(response_token_ids)
    sample.loss_mask = loss_masks
    return sample
```

### 4.3 训练阶段
**文件**: `slime/ray/actor_group.py:111-114`

```python
def async_train(self, rollout_id, rollout_data_ref):
    """在 Ray actors 上异步训练"""
    return [
        actor.train.remote(rollout_id, rollout_data_ref) 
        for actor in self._actor_handlers
    ]
```

实际的训练在 `MegatronTrainRayActor.train()` 中执行：
- 计算 log probs
- 计算优势值（使用 GRPO/PPO）
- 计算损失并反向传播
- 更新模型权重

### 4.4 权重同步
**文件**: `slime/ray/actor_group.py:119-121`

```python
def update_weights(self):
    """广播权重从 rank 0 到其他 rank，并更新到 SGLang"""
    return ray.get([
        actor.update_weights.remote() 
        for actor in self._actor_handlers
    ])
```

---

## 5. 关键数据流

```
+-------------+       +-----------------+       +-----------------+
|   Prompt    | ----> |  RolloutManager | ----> |  SGLang Engine  |
|   Data      |       |  .generate()    |       |  (generate)     |
+-------------+       +-----------------+       +-----------------+
                                                          |
                                                          v
+-------------+       +-----------------+       +-----------------+
|   Training  | <---- |  RayTrainGroup  | <---- |   Samples       |
|   (Actor)   |       |  .async_train() |       |   (tokens,      |
+-------------+       +-----------------+       |   rewards)      |
        |                                       +-----------------+
        v
+-------------+
|   SGLang    |
|   update    |
|   weights   |
+-------------+
```

---

## Key Points

1. **Ray 分布式架构**
   - 使用 Ray placement groups 管理 GPU 资源
   - 训练和推理可以共用 GPU（colocate）或分开
   - 通过 Ray actors 实现分布式训练和推理

2. **训练与推理分离**
   - 训练使用 Megatron（在 RayTrainGroup 中）
   - 推理使用 SGLang（在 RolloutManager 中）
   - 通过 `update_weights()` 同步权重

3. **可定制的 Rollout**
   - 通过 `--custom-generate-function-path` 指定自定义生成函数
   - 通过 `--custom-rm-path` 指定自定义奖励函数
   - Retool 示例展示了多轮工具调用的实现

4. **内存管理**
   - `--colocate` 模式下需要 offload 切换
   - `offload_train()`: 训练时 offload SGLang，onload Megatron
   - `offload_rollout()`: rollout 时 offload Megatron，onload SGLang

## Follow-up Questions

- [ ] MegatronTrainRayActor 的具体训练逻辑是怎样的？
- [ ] SGLang 权重更新的具体实现机制？
- [ ] 优势值计算（GRPO/PPO）的详细流程？
- [ ] Retool 的奖励函数是如何工作的？
- [ ] 如何处理多轮对话的 loss mask？
