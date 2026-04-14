---
date: 2026-04-14
question_id: Q023
topics: ["multi-agent", "checkpoint", "architecture", "design"]
related_files:
  - slime/ray/placement_group.py
  - examples/multi_agent/agent_system.py
  - examples/multi_agent/rollout_with_multi_agents.py
---

# Question

现在 `examples/multi_agent` 其实是训练的一个 model 的不同角色。如果在一个 multi-agent system 里面我希望不同的角色是从不同的 ckpt 开始训练，应该怎么设计？

# Answer

## 一句话总结
n
可以通过**多模型并行训练架构**实现，核心思路是：扩展 `create_training_models` 创建多个独立的 Actor 模型，每个模型有自己的 checkpoint 路径和 Placement Group。

## 当前设计分析

### 现有架构 (`examples/multi_agent`)

```
当前设计:
┌─────────────────┐
│  SGLang Server  │ ← 同一个模型权重
│  (Inference)    │
└────────┬────────┘
         │
    ┌────┴────┐
    ▼         ▼
┌───────┐ ┌───────┐ ┌───────┐
│Solver │ │Rewriter│ │Selector│ ← 同一模型的不同角色（prompt不同）
└───────┘ └───────┘ └───────┘
         │
    ┌────┴────┐
    ▼         ▼
┌─────────────────┐
│  Megatron Actor │ ← 训练时只有一个模型
│  (Training)     │
└─────────────────┘
```

**关键代码** (`rollout_with_multi_agents.py:16-33`):

```python
async def generate_with_multi_agents(args, sample: Sample, sampling_params, evaluation=False):
    # 所有角色共享同一个 args，使用同一个 SGLang server
    args.sampling_params = sampling_params
    custom_multi_agent_func = load_function(args.custom_multi_agent_function_path)
    samples = await custom_multi_agent_func(args, sample)  # 只是调用逻辑不同
    return samples
```

## 设计方案

### 方案 1：多独立模型架构（推荐）

每个角色有独立的训练和推理模型，完全分离。

```
设计架构:
┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
│ SGLang-Solver   │ │ SGLang-Rewriter │ │ SGLang-Selector │
│  (Model A)      │ │  (Model B)      │ │  (Model C)      │
└────────┬────────┘ └────────┬────────┘ └────────┬────────┘
         │                   │                   │
         └───────────────────┼───────────────────┘
                             │
                    ┌────────▼────────┐
                    │ Multi-Agent Flow│
                    │ (agent_system)  │
                    └────────┬────────┘
                             │
         ┌───────────────────┼───────────────────┐
         │                   │                   │
┌────────▼────────┐ ┌────────▼────────┐ ┌────────▼────────┐
│ Megatron-ActorA │ │ Megatron-ActorB │ │ Megatron-ActorC │
│  (load ckpt A)  │ │  (load ckpt B)  │ │  (load ckpt C)  │
└─────────────────┘ └─────────────────┘ └─────────────────┘
```

**实现步骤**:

1. **扩展参数配置** (`arguments.py`):

```python
parser.add_argument("--multi-agent-config", type=str, default=None,
    help="JSON配置文件，定义每个角色的模型配置")

# 示例配置 multi_agent_config.json:
{
    "agents": {
        "solver": {
            "actor_num_nodes": 1,
            "actor_num_gpus_per_node": 4,
            "load": "/path/to/solver_ckpt",
            "hf_checkpoint": "/path/to/solver_hf",
            "rollout_num_gpus": 2,
        },
        "rewriter": {
            "actor_num_nodes": 1,
            "actor_num_gpus_per_node": 4,
            "load": "/path/to/rewriter_ckpt",
            "hf_checkpoint": "/path/to/rewriter_hf",
            "rollout_num_gpus": 2,
        },
        "selector": {
            "actor_num_nodes": 1,
            "actor_num_gpus_per_node": 2,
            "load": "/path/to/selector_ckpt",
            "hf_checkpoint": "/path/to/selector_hf",
            "rollout_num_gpus": 1,
        }
    }
}
```

2. **创建多模型管理器**:

```python
# slime/ray/multi_agent_placement_group.py

class MultiAgentPlacementGroupManager:
    """管理多个角色的 Placement Group 和模型"""

    def __init__(self, args):
        self.agent_configs = self._load_agent_configs(args.multi_agent_config)
        self.pgs = {}
        self.models = {}
        self.rollout_managers = {}

    def create_all_placement_groups(self):
        """为每个角色创建独立的 PG"""
        for agent_name, config in self.agent_configs.items():
            pg = self._create_pg_for_agent(config)
            self.pgs[agent_name] = pg

    def create_all_models(self):
        """为每个角色创建独立的训练和推理模型"""
        for agent_name, config in self.agent_configs.items():
            # 为每个角色创建独立的 rollout manager
            rollout_mgr = create_rollout_manager_for_agent(config)
            self.rollout_managers[agent_name] = rollout_mgr

            # 为每个角色创建独立的 training model
            model = allocate_train_group(
                args=config,  # 使用角色特定的配置
                pg=self.pgs[agent_name],
                role=f"actor_{agent_name}",
            )
            self.models[agent_name] = model

            # 每个模型加载自己的 checkpoint
            model.async_init(config, role=f"actor_{agent_name}")
```

3. **修改 Agent System** 调用不同的 SGLang server:

```python
# examples/multi_agent/agent_system_v2.py

class MultiModelAgentSystem:
    def __init__(self, rollout_managers):
        # 每个角色使用自己的 rollout manager
        self.solver_rm = rollout_managers["solver"]
        self.rewriter_rm = rollout_managers["rewriter"]
        self.selector_rm = rollout_managers["selector"]

    async def solver_generate(self, prompt):
        # 调用 solver 专用的 SGLang server
        return await self._call_rm(self.solver_rm, prompt)

    async def rewriter_generate(self, prompt):
        # 调用 rewriter 专用的 SGLang server
        return await self._call_rm(self.rewriter_rm, prompt)

    async def selector_generate(self, prompt):
        # 调用 selector 专用的 SGLang server
        return await self._call_rm(self.selector_rm, prompt)
```

### 方案 2：共享 Backbone + 独立 Heads

共享大部分参数，每个角色只训练特定的 head 层。

```
设计架构:
┌─────────────────────────────────────────┐
│           Shared Backbone               │
│     (Pretrained Base Model)             │
│         Frozen / Low LR                 │
└──────────────┬──────────────────────────┘
               │
    ┌──────────┼──────────┐
    ▼          ▼          ▼
┌───────┐ ┌───────┐ ┌───────┐
│Solver │ │Rewriter│ │Selector│
│ Head  │ │ Head   │ │ Head   │
│(Train)│ │(Train) │ │(Train) │
└───────┘ └───────┘ └───────┘
```

**实现思路**:

```python
# slime/backends/megatron_utils/multi_head_model.py

class MultiHeadActorModel:
    """共享 backbone，多个独立 head"""

    def __init__(self, args, agent_heads_config):
        # 加载共享的 backbone
        self.backbone = load_backbone(args.shared_checkpoint)

        # 为每个角色创建独立的 head
        self.heads = {}
        for agent_name, head_config in agent_heads_config.items():
            self.heads[agent_name] = AgentHead(
                input_dim=head_config.input_dim,
                hidden_dim=head_config.hidden_dim,
                load_path=head_config.checkpoint  # 每个 head 有自己的 ckpt
            )

    def forward(self, input_ids, agent_name):
        # 共享 backbone 提取特征
        hidden_states = self.backbone(input_ids)

        # 使用对应角色的 head
        output = self.heads[agent_name](hidden_states)
        return output
```

**优点**:
- 节省显存（共享大部分参数）
- 训练效率高

**缺点**:
- 实现复杂，需要修改模型结构
- 角色间的干扰可能影响性能

### 方案 3：顺序训练 + 权重切换

只有一个模型，但在不同训练阶段加载不同角色的 checkpoint。

```python
# 训练流程
for epoch in range(num_epochs):
    # Phase 1: 训练 Solver
    load_checkpoint("solver_ckpt")
    train_solver()
    save_checkpoint("solver_ckpt_new")

    # Phase 2: 训练 Rewriter
    load_checkpoint("rewriter_ckpt")
    train_rewriter()
    save_checkpoint("rewriter_ckpt_new")

    # Phase 3: 训练 Selector
    load_checkpoint("selector_ckpt")
    train_selector()
    save_checkpoint("selector_ckpt_new")
```

**优点**:
- 实现简单
- 资源需求最低

**缺点**:
- 不能同时训练多个角色
- 每次切换都有加载开销

## 方案对比

| 方案 | 资源需求 | 实现复杂度 | 训练效率 | 适用场景 |
|------|---------|-----------|---------|---------|
| **方案 1** (多独立模型) | 高 | 中 | 高 | 角色差异大，资源充足 |
| **方案 2** (共享 Backbone) | 中 | 高 | 中 | 角色相似，需要参数共享 |
| **方案 3** (顺序训练) | 低 | 低 | 低 | 资源紧张，角色训练可分离 |

## 推荐实现：方案 1 的详细代码

### 1. 配置文件 (`multi_agent_config.yaml`)

```yaml
agents:
  solver:
    actor:
      num_nodes: 1
      num_gpus_per_node: 4
      load: "/checkpoints/solver"
      hf_checkpoint: "/models/solver_hf"
    rollout:
      num_gpus: 2
      num_gpus_per_engine: 1
      mem_fraction_static: 0.7

  rewriter:
    actor:
      num_nodes: 1
      num_gpus_per_node: 4
      load: "/checkpoints/rewriter"
      hf_checkpoint: "/models/rewriter_hf"
    rollout:
      num_gpus: 2
      num_gpus_per_engine: 1
      mem_fraction_static: 0.7

  selector:
    actor:
      num_nodes: 1
      num_gpus_per_node: 2
      load: "/checkpoints/selector"
      hf_checkpoint: "/models/selector_hf"
    rollout:
      num_gpus: 1
      num_gpus_per_engine: 1
      mem_fraction_static: 0.8

training:
  use_cross_agent_kl: true  # 是否计算角色间的 KL 散度
  agent_loss_weights:
    solver: 1.0
    rewriter: 0.8
    selector: 0.5
```

### 2. 核心实现 (`multi_agent_trainer.py`)

```python
class MultiAgentTrainer:
    """多角色独立模型训练器"""

    def __init__(self, config_path):
        self.config = load_config(config_path)
        self.agents = {}
        self._setup_agents()

    def _setup_agents(self):
        for agent_name, agent_config in self.config["agents"].items():
            # 为每个角色创建独立的 Placement Group
            pg = create_placement_group_for_agent(agent_config)

            # 创建独立的 Rollout Manager
            rollout_mgr = RolloutManager.remote(agent_config, pg["rollout"])

            # 创建独立的 Training Model
            actor = RayTrainGroup(
                args=agent_config["actor"],
                pg=pg["actor"],
                role=f"actor_{agent_name}",
            )

            # 初始化并加载各自的 checkpoint
            actor.async_init(
                agent_config["actor"],
                role=f"actor_{agent_name}",
                checkpoint_path=agent_config["actor"]["load"]
            )

            self.agents[agent_name] = {
                "rollout": rollout_mgr,
                "actor": actor,
                "config": agent_config,
            }

    def train_step(self, batch_data):
        """一步训练所有角色"""
        futures = {}

        for agent_name, agent in self.agents.items():
            # 获取该角色的 rollout 数据
            rollout_data = batch_data[agent_name]

            # 异步训练
            future = agent["actor"].async_train(rollout_data)
            futures[agent_name] = future

        # 等待所有角色训练完成
        results = ray.get(list(futures.values()))

        # 可选：角色间的知识蒸馏或 KL 约束
        if self.config["training"]["use_cross_agent_kl"]:
            self._apply_cross_agent_kl_constraint()

        return results
```

### 3. 多模型 Rollout (`multi_model_rollout.py`)

```python
async def generate_with_multi_model_agents(args, sample: Sample) -> dict[str, list[Sample]]:
    """为每个角色使用独立的模型生成"""

    results = {}

    # Solver 生成
    solver_samples = await generate_from_agent(
        agent_name="solver",
        rollout_mgr=args.agents["solver"]["rollout"],
        prompt=sample.prompt,
        sampling_params=args.solver_sampling_params,
    )
    results["solver"] = solver_samples

    # Rewriter 生成
    rewriter_samples = await generate_from_agent(
        agent_name="rewriter",
        rollout_mgr=args.agents["rewriter"]["rollout"],
        prompt=build_rewriter_prompt(solver_samples),
        sampling_params=args.rewriter_sampling_params,
    )
    results["rewriter"] = rewriter_samples

    # Selector 生成
    selector_samples = await generate_from_agent(
        agent_name="selector",
        rollout_mgr=args.agents["selector"]["rollout"],
        prompt=build_selector_prompt(rewriter_samples),
        sampling_params=args.selector_sampling_params,
    )
    results["selector"] = selector_samples

    return results
```

## 关键修改点总结

| 文件 | 修改内容 |
|------|---------|
| `slime/ray/placement_group.py` | 支持创建多个独立的 PG |
| `slime/ray/actor_group.py` | `RayTrainGroup` 支持按角色加载不同 checkpoint |
| `slime/ray/rollout.py` | `RolloutManager` 支持多实例 |
| `examples/multi_agent/` | 新的 agent system 实现，调用不同模型的 SGLang server |
| `train.py` | 支持多角色并行训练循环 |

## Key Points

1. **核心思路**：从"一个模型多个角色"扩展到"多个模型多个角色"，每个角色有自己的训练和推理资源

2. **资源分配**：需要为每个角色独立分配 GPU，通过不同的 Placement Group 隔离

3. **Checkpoint 管理**：每个角色维护自己的 checkpoint 路径，训练时独立保存/加载

4. **通信机制**：角色间通过 `agent_system` 协调，每个角色内部保持原有的 Actor-Critic 结构

5. **推荐方案**：方案 1（多独立模型）最灵活，适合角色差异大的场景；如果角色相似且资源紧张，考虑方案 3

## Follow-up Questions

- [ ] 如何在多角色间共享经验（cross-agent experience sharing）？
- [ ] 角色间是否需要 KL 散度约束来保持策略一致性？
- [ ] 如何设计多角色的联合评估指标？
