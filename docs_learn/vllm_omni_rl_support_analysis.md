# vLLM-Omni 强化学习(RL)支持分析

- **RFC Issue**: [#778](https://github.com/vllm-project/vllm-omni/issues/778)
- **创建日期**: 2026-01-14
- **最后更新**: 2026-02-16

## 1. 概述

本文档分析 vLLM-Omni 项目中与强化学习(RL)相关的设计、现有进展和未来规划。vLLM-Omni 旨在与 [verl](https://github.com/volcengine/verl)（火山引擎开源的生产级 RLHF 框架）集成，支持多模态模型的 RL 训练流程。

### 核心目标

- 支持 Ray-based vLLM rollout workers (`VLLMRolloutActor`) 用于并行推理
- 实现 Zero-copy 权重同步（从训练 worker 到推理引擎）
- 兼容 DataProto batch protocol 双向数据传输协议
- 支持多模态模型（如 Qwen2.5-Omni 等）的 RLHF

## 2. 设计架构

### 2.1 整体架构

```
┌─────────────────────────────────────────────────────────────────┐
│                       verl RL Framework                          │
│  ┌─────────────────────┐    ┌───────────────────────────────┐  │
│  │   Training Worker   │    │    VLLMRolloutActor (vLLM)    │  │
│  │  (FSDP/Megatron)    │───▶│      Rollout Generation       │  │
│  └─────────────────────┘    └───────────────────────────────┘  │
│            │                                   │                │
│            │    Zero-copy Weight Sync          │                │
│            │    DataProto Protocol             │                │
│            ▼                                   ▼                │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │              vLLM-Omni Rollout Engine                    │   │
│  │  ┌──────────────┐  ┌──────────────┐  ┌────────────────┐ │   │
│  │  │ AsyncOmniLLM │  │AsyncOmniDiff │  │  Custom       │ │   │
│  │  │   (Thinker)  │  │   (Diffusion)│  │  Pipeline     │ │   │
│  │  └──────────────┘  └──────────────┘  └────────────────┘ │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 关键组件

| 组件 | 说明 | 状态 |
|------|------|------|
| `AsyncOmniLLM` | 异步 LLM 推理引擎，继承自 vLLM 的 `AsyncLLM` | ✅ 已完成 |
| `AsyncOmniDiffusion` | 异步扩散模型推理引擎 | ✅ 已完成 |
| `DiffusionWorker` | 管理 GPU 基础设施的扩散模型工作器 | ✅ 已完成 |
| `WorkerWrapperBase` | 支持扩展的 Worker 包装基类 | ✅ 已完成 |
| `DiffusionLoRAManager` | LoRA adapter 管理器 | ✅ 已完成 |

## 3. 现有进展

### 3.1 模型权重管理（已完成 ✓）

**相关 Issue/PR**: [#316](https://github.com/vllm-project/vllm-omni/issues/316), [#376](https://github.com/vllm-project/vllm-omni/pull/376)

为支持 RL 训练中的权重动态更新，已实现：

- **`sleep(level)`** - 将模型权重卸载到 CPU，释放 GPU 内存供训练使用
  - Level 1: 仅卸载权重
  - Level 2: 同时保存缓冲区状态
- **`wake_up(tags)`** - 从睡眠模式唤醒，将权重重新加载到 GPU
- **`load_weights(weights)`** - 动态加载新权重

**代码位置**: `vllm_omni/diffusion/worker/diffusion_worker.py:204-271`

```python
def sleep(self, level: int = 1) -> bool:
    """Put the worker to sleep, offloading model weights."""
    from vllm.device_allocator.cumem import CuMemAllocator
    allocator = CuMemAllocator.get_instance()
    allocator.sleep(offload_tags=("weights",) if level == 1 else tuple())
    ...

def wake_up(self, tags: list[str] | None = None) -> bool:
    """Wake up the worker from sleep mode."""
    from vllm.device_allocator.cumem import CuMemAllocator
    allocator = CuMemAllocator.get_instance()
    allocator.wake_up(tags)
    ...
```

### 3.2 LoRA 适配器支持（已完成 ✓）

**相关 Issue/PR**: [#281](https://github.com/vllm-project/vllm-omni/issues/281), [#657](https://github.com/vllm-project/vllm-omni/pull/657), [#758](https://github.com/vllm-project/vllm-omni/pull/758)

实现扩散模型的 LoRA 支持：

- 运行时动态加载/卸载 LoRA adapter，无需重启服务
- Per-worker LRU 缓存，可配置 VRAM 预算
- PEFT 格式兼容
- 路径白名单安全机制
- 支持 SD3.5/SDXL 等模型

**代码位置**: `vllm_omni/diffusion/lora/manager.py`

**异步 API 示例**:
```python
async def add_lora(self, lora_request: LoRARequest, lora_scale: float = 1.0) -> bool
async def remove_lora(self, adapter_id: int) -> bool
async def list_loras(self) -> list[int]
async def pin_lora(self, lora_id: int) -> bool
```

**vLLM 集成步骤**:
1. 初始化 LoRA manager
2. Per-request 动态激活 LoRA
3. 推理时通过 vLLM 自研 LoRA 层进行计算

### 3.3 异步接口 / EngineClient（已完成 ✓）

**相关 Issue/PR**: [#342](https://github.com/vllm-project/vllm-omni/issues/342)

已提供兼容 verl 的异步接口：

#### AsyncOmniLLM
- 继承自 vLLM 的 `AsyncLLM`
- 支持 `from_vllm_config()` 类方法用于从配置初始化
- 支持 `reset_mm_cache()` 用于重置多模态缓存
- 专为多模态输入/输出优化的处理器

**代码位置**: `vllm_omni/entrypoints/async_omni_llm.py:32`<br>
**from_vllm_config**: `vllm_omni/entrypoints/async_omni_llm.py:193`

#### AsyncOmniDiffusion
- 扩散模型异步推理接口
- 不支持 `reset_mm_cache`（扩散模型尚未构建多模态缓存系统）
- 完整的 LoRA API 支持

**代码位置**: `vllm_omni/entrypoints/async_omni_diffusion.py:30`

### 3.4 WorkerWrapperBase 与自定义 Pipeline（已完成 ✓）

**相关 Issue/PR**: [#686](https://github.com/vllm-project/vllm-omni/issues/686), [#764](https://github.com/vllm-project/vllm-omni/pull/764)

关键实现：

- **`WorkerWrapperBase`** - 类似 vLLM 设计，支持动态继承扩展
- **自定义 Pipeline 支持** - 允许用户传入自定义 pipeline 类
  - 支持 RL 场景中的中间变量返回（prompt embeddings、cached latents）
  - 支持自定义 scheduler（如 SDE 版本的 Euler sampler）
- **`load_format=dummy`** - 无模型加载的 worker 初始化，便于测试
- **通过 `re_init_pipeline` 方法动态更换 pipeline**

**代码位置**: `vllm_omni/diffusion/worker/diffusion_worker.py:488-684`

**架构图**:
```
┌─────────────────────────────────────┐
│      CustomPipelineWorkerExtension  │ (用户扩展)
│         ┌──────────────────────────┐│
│         │ DiffusionWorker          ││
│         │ └────────────────────────┐│
│         │  │ DiffusionModelRunner  ││
│         │  │ └────────────────────┐││
│         │  │  │ Pipeline (Custom) │││
│         │  │  └───────────────────┘││
└─────────┴──┴───────────────────────┘
```

## 4. 当前限制

### 4.1 Ray Backend 不支持（关键障碍）

**状态**: ❌ 未支持<br>
**说明**: Diffusion 模型目前不支持 Ray 分布式执行器后端

```python
# verl 中使用 Ray 启动 vLLM 的方式
# vllm_omni 中 diffusion 部分尚不兼容
from verl.workers.rollout.vllm_rollout import vllm_async_server
```

### 4.2 mm_cache for Diffusion

**状态**: ❓ 待讨论<br>
**说明**: `AsyncOmniDiffusion` 尚未构建多模态缓存系统。但参与者 [ZJY0516](https://github.com/ZJY0516) 提出疑问：扩散模型是否需要 `mm_cache`？

### 4.3 Batching 支持

**状态**: ⚠️ 需优化<br>
**说明**:
- 扩散模型推理的 batching 支持在理论上比纯 diffusers 后端慢
- 需要进一步优化以支持 RL 训练中的高吞吐需求

### 4.4 from_vllm_config 兼容性

**状态**: ⚠️ 待完善<br>
**说明**: `AsyncOmniDiffusion` 需要支持类似 `from_vllm_config` 的接口以完全兼容 verl

## 5. Future Work

### 5.1 短期规划（进行中）

1. **Ray Backend 支持**
   - 为 `DiffusionWorker` 实现 Ray-based 分布式执行器
   - 目标：完整兼容 `verl/workers/rollout/vllm_rollout/vllm_async_server.py`

2. **接口统一**
   - 为 `AsyncOmniDiffusion` 添加 `from_vllm_config` 类方法
   - 确保与 `EngineClient` 接口兼容

3. **性能优化**
   - 扩散模型 batching 推理优化
   - 支持分布式 executor backend for diffusion

### 5.2 架构设计原则

根据维护者 [ZJY0516](https://github.com/ZJY0516) 的意见：

> "FYI, I don't want OmniDiffusion be coupled with vllm config"

- **解耦设计**: `OmniDiffusion` 应保持独立性，不与 vllm 配置过度耦合
- **可选功能**: `mm_cache` 是否引入扩散模型需进一步评估
- **模块化**: 保持各组件的独立性和可替换性

### 5.3 verl 集成路线图

```
Phase 1: Core Infrastructure (✅ Done)
├── Weight management (sleep/wake/load_weights)
├── LoRA adapter support
├── Async interfaces (AsyncOmniLLM, AsyncOmniDiffusion)
└── WorkerWrapperBase and custom pipeline

Phase 2: Distributed Execution (🔄 In Progress)
├── Ray backend support for diffusion
├── Distributed executor backend
└── Batching optimization

Phase 3: Full Integration (📅 Planned)
├── Complete verl compatibility
├── Performance tuning for RL workloads
└── Production hardening
```

## 6. 相关链接

### Issues
- [#778](https://github.com/vllm-project/vllm-omni/issues/778) - RL Support RFC (本文档基础)
- [#316](https://github.com/vllm-project/vllm-omni/issues/316) - 权重卸载/加载功能
- [#281](https://github.com/vllm-project/vllm-omni/issues/281) - LoRA 适配器支持
- [#686](https://github.com/vllm-project/vllm-omni/issues/686) - 自定义 Pipeline 支持

### Pull Requests
- [#376](https://github.com/vllm-project/vllm-omni/pull/376) - sleep/wake 和 load_weights 支持 (已合并)
- [#657](https://github.com/vllm-project/vllm-omni/pull/657) - LoRA 请求路径和 worker 缓存 (已合并)
- [#758](https://github.com/vllm-project/vllm-omni/pull/758) - LoRA Adapter 支持 (已合并)
- [#764](https://github.com/vllm-project/vllm-omni/pull/764) - WorkerWrapperBase 和 CustomPipeline (已合并)

### 外部参考
- [verl](https://github.com/volcengine/verl) - 火山引擎 RLHF 框架
- [mm_grpo](https://github.com/leibniz-csi/mm_grpo) - 多模态 GRPO 项目
- [FlowGRPO](https://github.com/yifan123/flow_grpo)
- [DiffusionNFT](https://github.com/NVlabs/DiffusionNFT)

## 7. 参与者

- **主要负责人**: [@SamitHuang](https://github.com/SamitHuang), [@zhtmike](https://github.com/zhtmike), [@knlnguyen1802](https://github.com/knlnguyen1802)
- **贡献者**: [@ZJY0516](https://github.com/ZJY0516), [@hsliuustc0106](https://github.com/hsliuustc0106), [@princepride](https://github.com/princepride), [@KevinZeng08](https://github.com/KevinZeng08)

## 8. 总结

vLLM-Omni 的 RL 支持已完成核心基础设施：
- ✅ 权重管理 (sleep/wake/load_weights)
- ✅ LoRA 适配器支持
- ✅ 异步接口 (AsyncOmniLLM, AsyncOmniDiffusion)
- ✅ 自定义 Pipeline 支持

**当前主要障碍**: Ray backend 尚不支持 diffusion 模型，这是与 verl 完全集成的关键。

完成 Ray 支持后，vLLM-Omni 将成为支持多模态模型 RLHF 的完整推理引擎。
