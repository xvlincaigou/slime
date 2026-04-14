# Slime 学习记录

这是 Slime 代码库的学习记录文件夹。每次问答都会被记录在这里，方便日后复习和查阅。

## 学习进度

### 已覆盖主题

- [x] 训练流程 (Q001) - 以 retool 为例的完整工作流程
- [x] Placement Group (Q002) - GPU 资源分配机制
- [x] 多节点训练 (Q003) - 多机器资源分配
- [x] Ray Placement Group (Q004) - Ray PG 深度解析
- [x] RolloutManager (Q005) - 推理引擎管理器
- [x] SGLang Server (Q006) - SGLang 推理服务器
- [x] 故障容错 (Q007) - RolloutHealthMonitor
- [x] RolloutManager 方法 (Q008) - 除初始化外的其他方法
- [x] Actor/Critic 创建 (Q009) - 训练模型初始化
- [x] 训练初始化阶段 (Q010) - 权重同步与 colocate 处理
- [x] 动态 Global Batch Size (Q011) - 训练步骤优化
- [x] Data Parallel 分割与 GRPO (Q012) - DP 分割策略
- [x] RayTrainGroup (Q013) - Megatron 训练 Actor 组管理
- [x] MegatronTrainRayActor (Q014) - Megatron 训练 Actor 实现
- [x] Actor 数据函数详解 (Q015) - _get_rollout_data, train_actor 等
- [x] get_data_iterator 详解 (Q016) - 数据迭代器与动态 batch size
- [x] train() 函数详解 (Q017) - Megatron Pipeline 训练循环
- [x] Dynamic Sampling 详解 (Q018) - 动态采样与过滤机制
- [x] train_async vs train.py (Q019) - 异步与同步训练对比
- [x] Data Packing 与 Loss 计算 (Q020) - per sample loss 机制
- [x] Fully Async Rollout (Q021) - 完全异步 rollout 实现
- [x] Off-Policy 处理 (Q022) - TIS 与重要性采样
- [x] Multi-Agent 多模型设计 (Q023) - 不同角色不同 checkpoint

### 问题统计

- 总问题数: 23
- 最后更新: 2026-04-14

### 问答索引

| 编号 | 问题 | 主题 |
|------|------|------|
| [Q001](questions/Q001-training-workflow-retool.md) | retool 训练完整工作流程 | training-workflow, retool, ray |
| [Q002](questions/Q002-placement-groups-allocation.md) | create_placement_groups 详解 | placement-group, gpu-scheduling |
| [Q003](questions/Q003-multi-node-resource-allocation.md) | 多机器资源分配 | multi-node, distributed-training |
| [Q004](questions/Q004-ray-placement-group-deep-dive.md) | Ray Placement Group 深度解析 | ray-core, resource-scheduling |
| [Q005](questions/Q005-create-rollout-manager.md) | create_rollout_manager 详解 | rollout-manager, sglang |
| [Q006](questions/Q006-start-rollout-servers-deep-dive.md) | start_rollout_servers 详解 | sglang, inference, pd-disaggregation |
| [Q007](questions/Q007-rollout-health-monitor.md) | RolloutHealthMonitor 详解 | fault-tolerance, health-monitor |
| [Q008](questions/Q008-rollout-manager-methods.md) | RolloutManager 其他方法详解 | rollout-manager, data-generation, evaluation |
| [Q009](questions/Q009-create-training-models.md) | create_training_models 详解 | training, actor, critic, megatron |
| [Q010](questions/Q010-train-init-phase.md) | 训练初始化阶段详解 | weight-sync, colocate, initialization |
| [Q011](questions/Q011-dynamic-global-batch-size.md) | 动态 Global Batch Size 详解 | batch-size, optimization, rollout |
| [Q012](questions/Q012-split-train-data-grpo.md) | Data Parallel 分割与 GRPO | data-parallel, grpo, distributed-training |
| [Q013](questions/Q013-raytraingroup.md) | RayTrainGroup 详解 | ray, megatron, actor, critic |
| [Q014](questions/Q014-megatron-train-ray-actor.md) | MegatronTrainRayActor 详解 | megatron, training, weight-sync |
| [Q015](questions/Q015-actor-data-functions-detail.md) | Actor 数据函数详解 | data-processing, routing-replay, train_actor |
| [Q016](questions/Q016-get-data-iterator.md) | get_data_iterator 详解 | data-loading, micro-batch, dynamic-batch-size |
| [Q017](questions/Q017-train-function-model-py.md) | train() 函数详解 | training, megatron, pipeline-parallelism |
| [Q018](questions/Q018-dynamic-sampling.md) | Dynamic Sampling 详解 | rollout, dynamic-sampling, filter |
| [Q019](questions/Q019-train-async-vs-sync.md) | train_async vs train.py | training, async, sync |
| [Q020](questions/Q020-data-packing-loss.md) | Data Packing 与 Loss 计算 | data-packing, loss, dynamic-batch-size |
| [Q021](questions/Q021-fully-async-rollout.md) | Fully Async Rollout 详解 | fully-async, rollout, async |
| [Q022](questions/Q022-off-policy-correction.md) | Off-Policy 问题处理 | off-policy, tis, importance-sampling |
| [Q023](questions/Q023-multi-agent-multi-checkpoint.md) | Multi-Agent 多模型设计 | multi-agent, checkpoint, architecture |

## 目录结构

```
learn/
├── README.md              # 本文件
├── questions/             # 问答记录
├── topics/                # 按主题整理的知识
└── summaries/             # 阶段性总结
```

## 如何使用

1. 向 Claude 提问关于代码库的问题
2. Claude 会回答并自动记录 Q&A 到 `questions/` 文件夹
3. 相关主题会被整理到 `topics/` 文件夹
4. 定期回顾和整理学习成果

## 快速链接

- [项目主 README](../README.md)
- [贡献指南](../CONTRIBUTING.md)
- [文档目录](../docs/)
