# JobPacer：代码结构、执行数据流与实验使用指南

本文描述 2026-09-30 整理后的代码，以当前文件和 CLI 为准。JobPacer 是本仓库中构造 workload、调用通信调度器、测量与验证实验的上层工具，不是完整训练框架。通用调度器在 `src/runtime_comm_scheduler/`，本目录提供实验所需的输入、绑定、rank harness、编排与分析。

当前边界：Gloo 保留历史线性 workload 和 schema-v1 DAG；GPU/NCCL 只接受 **schema-v2、`cuda-program` DAG**。线性 GPU 场景用链式 DAG 表达。

- `python -m examples.jobpacer.scripts.run_phase3`：GPU 实验流程，使用 `prepare / qualify / check / run / analyze / finalize` 子命令。
- `python -m examples.jobpacer.runtime.replay_launcher`：单次 Phase 3 replay，接受 workload/DAG、policy 和 backend 参数。

## 1. 仓库与文件架构

### 1.1 仓库层级

以下路径相对于仓库根目录。

| 路径 | 内容与边界 |
| --- | --- |
| `src/runtime_comm_scheduler/runtime/` | 新中心化通信 runtime：任务模型、coordinator、policy、控制通道、本地发射、完成与 handle |
| `src/runtime_comm_scheduler/dag/` | 通用 DAG 模型、合法性检查、静态顺序和节点推进，不在 coordinator 内调度计算 |
| `src/runtime_comm_scheduler/` 根层模块 | 旧 Plan/AdmissionScheduler 执行体系，继续用于历史 Gloo 与 GPU 旧 scheduler 基线 |
| `src/runtime_comm_scheduler/adapters/` | 外部框架适配相关代码；存在模块不代表已完成 Megatron 训练验收 |
| `examples/jobpacer/` | 本文描述的实验工具和执行绑定 |
| `benchmark/phase1.2/` | 历史 Gloo 实验输入及本地产物 |
| `benchmark/phase3/experiments/` | DAG/线性输入、诊断配置、GPU 七臂版本化候选与冻结输入 |
| `benchmark/phase3/results/` | 原始 replay、批次 ledger、源码快照、审计及统计，Git 忽略 |
| `tests/unit/` | 模型、适配、统计、CLI、调度协议及失败路径的单元检查 |
| `tests/integration/` | 多进程 Gloo 和显式启用的真实 NCCL 验证 |
| `docs/JobPacer/plan/` | 设计与实验合同 |
| `docs/JobPacer/process/` | 实施说明、准备过程与交接 |
| `docs/JobPacer/result/` | 已验证事实与结果报告，与原始 results 不同，应进入版本管理 |
| `README.md`、`AGENTS.md`、`pyproject.toml` | 项目入口、协作约束、包与测试配置 |

核心文件分工如下，包内 `__init__.py` 负责包声明或公共符号导出。

| 核心文件 | 内容 |
| --- | --- |
| `runtime/model.py` | group、collective、TaskSpec、TaskHint、LocalBinding 等规范与本地绑定 |
| `runtime/coordinator.py` | 中央状态、合法候选、全局容量、决定及成员完成处理 |
| `runtime/policy.py` | FIFO、LTF、StaticOrder、Lookahead 的选择逻辑 |
| `runtime/runtime.py` | rank 侧提交、回执、发射和完成生命周期 |
| `runtime/protocol.py`、`runtime/transport.py` | 控制消息及 TCP 传输、服务端/客户端交互 |
| `runtime/executor.py` | 真实 collective 发射、CUDA receipt 与物理完成探测支持 |
| `runtime/handle.py`、`runtime/telemetry.py` | 用户等待接口、状态及事件记录 |
| `dag/model.py`、`dag/runner.py` | 图校验、依赖/顺序/估计及计算通信节点推进 |
| 根层 `intent.py`、`plan.py` | 旧通信 intent、TaskKey 和静态 Plan |
| 根层 `scheduler.py`、`work.py`、`executor.py` | 旧准入 scheduler、ScheduledWork 与 PyTorch 执行封装 |
| 根层 `telemetry.py`、`validate.py` | 旧路径观测与验证工具 |

### 1.2 JobPacer 目录

```text
examples/jobpacer/
├── analysis/       # 结果读取、校验、统计、可视化
├── diagnostics/    # 可重复的机制、测量与恢复诊断
├── experiments/   # 输入及批次编排、实验资格、封存
│   └── phase3/     # 当前 GPU 实验流程
├── gloo/          # 保留的线性输入和 Gloo 计算/适配支持
├── gpu/           # 显式 CUDA DAG 资源、profile、设备支持
├── runtime/       # 共同执行基础、DAG 适配及新旧 rank harness
├── scripts/       # 用户入口；部分历史入口仍包含实现逻辑
└── paths.py       # 共享仓库/benchmark 路径解析
```

每个目录的 `__init__.py` 以及根目录 `__init__.py` 声明 Python 包；应通过下述具体模块理解业务逻辑。`paths.py` 当前仍在 JobPacer 根目录，负责仓库相对路径、正式输入识别和历史迁移映射。

### 1.3 `gloo/` 与 `gpu/`

| 文件 | 实际内容 |
| --- | --- |
| `gloo/workloads.py` | 旧 CollectiveComm/Job/Workload 模型、内置 workload、JSON 读取及确定性时长采样 |
| `gloo/workload_builder.py` | 将外部链式 DAG JSON 转换为旧线性 manifest；不用于生成 GPU schema-v2 |
| `gloo/comm_profile.py` | 把共享通信 profile 应用于旧 Workload，以及旧输入摘要 |
| `gloo/runtime_adapter.py` | 旧线性输入到新 TaskSpec/Hint 的映射、相关计算时长和 Gloo DAG host compute 支持 |
| `gpu/cuda_devices.py` | 可见 CUDA 设备、UUID 和 compute profile 软件环境信息 |
| `gpu/gpu_compute_profile.py` | schema-v2 计算 program 的签名、profile 读取和匹配校验 |
| `gpu/gpu_dag_resources.py` | 每 job 的 buffer 分配、初始化、warmup/reset、compute stream/event、节点 program、通信 buffer 绑定、CPU 参考值与终点数值验证 |

GPU 计算量来自输入的 program 和 repeats。profile 提供策略估计，不把秒数转换为运行时追赶时长的循环。Gloo 旧 producer/consumer 时间字段仍是 host 模拟工作量，不与 GPU 实际计算时长直接等同。

### 1.4 `runtime/`

| 文件 | 实际内容 |
| --- | --- |
| `runtime_adapter.py` | DagInput/ReplayExecutionConfig、schema-v1/v2 解析、buffer hazard 校验、submit_after/terminal 校验、profile 应用、静态顺序读取和绑定支持 |
| `runtime_worker.py` | Phase 3 rank harness：初始化 group、准备资源、构造 new/old/bare 引擎、推进 job、收集终点、drain 和结果校验 |
| `replay_launcher.py` | 单次 Phase 3 replay 的 CLI 与父进程实现：参数/输入校验、端口、rank 子进程、超时清理、结果合并 |
| `dag_comm_adapters.py` | GPU DAG 的旧 scheduler adapter、现役有序多在途 bare、冻结顺序合同、CUDA 完成探测与共享失败传播；bare 当前不另拆文件 |
| `comm_profile.py` | Gloo/NCCL 共用的通信签名、profile 数据模型、读取和环境匹配 |
| `plan_builder.py` | Phase 2 旧 Workload 的静态 Plan 构造与策略诊断；不属于新 coordinator 内部 |
| `replay_worker.py` | Phase 1/2 Gloo rank 执行：直接裸发或旧 AdmissionScheduler，完成观察、trace 和数值校验 |

`examples/jobpacer/runtime/` 是实验 harness，`src/runtime_comm_scheduler/runtime/` 才是通信协议核心，前者调用后者。

### 1.5 `experiments/`

| 文件 | 实际内容 |
| --- | --- |
| `runner_batch.py` | 历史 Phase 1.2 capacity、SRJF、polling 等配置驱动的批次运行与产物管理 |
| `gloo_phase3_batch.py` | 历史 Phase 3 多臂批次、命令构造、resume、shared/isolated 数据关联；目前仍含配对和机制统计 |
| `compact_suite.py` | CPU/Gloo compact suite 的预览、profile 校验、分阶段执行与机制证据检查 |
| `phase3/suite.py` | 六场景、seed、arm 与合同定义；workload 生成、profile 冻结、FIFO/LTF 顺序及配对块顺序表 |
| `phase3/prepare.py` | generate/freeze/order/preview 的可调用实现，含历史运行成本预算估计 |
| `phase3/batch.py` | 冻结输入和源码、创建批次、进程管理、attempt/ledger、整块恢复、有限启动重试及执行前资格检查 |
| `phase3/qualify.py` | bare/backend 正常路径与缺任务、绑定、launch、probe 等故障资格检查 |
| `phase3/checks.py` | 已有证据的状态汇总，以及 measurement/recovery/mechanism 诊断分派 |
| `phase3/gates.py` | 候选与策略证据检查、机制门槛、七类 readiness audit 的字段及 hash 绑定校验 |
| `phase3/finalize.py` | 关联通过的 pilot、校验正式准备证据、封存正式输入；不负责启动正式 replay |

### 1.6 `analysis/`

| 文件 | 实际内容 |
| --- | --- |
| `runtime_results.py` | 单次 Phase 3 replay 的预期任务、rank 覆盖、顺序与结果校验，以及 metrics/performance 汇总 |
| `phase3_results.py` | 当前 GPU 批次独立校验、完整配对块选择、逐 arm/seed 指标、bootstrap、策略机制证据及表格；当前还会生成 mechanism/rehearsal audit |
| `measurement.py` | 旧 trace 的占用区间等测量辅助函数 |
| `rebuild_experiment_batch.py` | 从已有 ledger/raw 重建历史批次统计，不重跑 collective |
| `visualize.py` | 旧 replay 的 trace 校验、scheduler 状态、时间线、capacity/策略比较图 |
| `visualize_phase3.py` | Phase 3 时间线、控制路径/开销、逐 job 配对统计及 suite 可视化 |

这里描述实际归属，而非声称职责拆分已经全部完成。旧脚本及 `gloo_phase3_batch.py` 仍有分析逻辑，`phase3_results.py` 仍包含部分放行审计生成。分析 CLI 不启动 replay，但会写统计、图表或审计产物。

### 1.7 `diagnostics/`

| 文件 | 实际内容 |
| --- | --- |
| `bare_nccl_mechanism.py` | 双卡单独 A/B、物理串行和多在途通信诊断，保存各 rank 与机制汇总 |
| `gpu_interference_profile.py` | 计算/通信单独及组合执行的干扰测量 |
| `gpu_measurement.py` | A/A 和 minimal/full 配对诊断的 plan/run/analyze；分析日志开销与测量一致性 |
| `gpu_recovery.py` | 端口冲突、进程清理、整块重试、中断恢复与 hash 检查的彩排 |
| `layered_fifo_pilot.py` | 默认/反转 job 顺序的小规模敏感性诊断，不按性能挑选默认顺序 |
| `interleaved_isolated.py` | 历史 Gloo shared/isolated 交错测量与单 job 分母检查 |

诊断可以调用实验执行模块和 runtime，不应导入 scripts 内部函数。诊断中的轻量统计属于该诊断的一部分；其结果不自动等于正式实验资格。

### 1.8 `scripts/`

| 文件 | 用户用途及当前实现边界 |
| --- | --- |
| `run_phase1.py` | 历史 Gloo bare 入口，转调 Phase 2 CLI 的 bare 模式 |
| `run_phase2.py` | 历史 Gloo replay；当前还包含父进程启动、校验和性能汇总 |
| `run_phase1_2_experiments.py` | 历史 Gloo 实验入口；当前还包含批次运行、排序和结果统计 |
| `run_phase3.py` | 当前 GPU 实验统一入口，解析子命令并调用 experiments/diagnostics/analysis |
| `run_comm_profile.py` | Gloo/NCCL 通信标定入口；当前包含 rank 采集和子进程实现 |
| `run_gpu_compute_profile.py` | schema-v2 GPU compute 标定入口；当前包含签名收集、真实计算测量和 profile 写出 |

## 2. 当前数据流

### 2.1 从实验输入到策略估计

```text
模板 + workload seed
  → 具体 DAG sample：buffer / program / comm binding / 依赖 / terminal
  → 通信 profile + compute profile
  → 同一实际 workload 的估计视图、remaining tail、冻结 FIFO/LTF 顺序
  → suite manifest + 输入/profile/order hash
  → block 顺序表 + batch manifest + 源码快照
```

schema-v2 顶层记录 groups、jobs、seed 等图信息；`execution` 记录 buffers、compute_programs、comm_bindings、submit_after、application_terminals 等执行信息。每个通信节点有稳定 task 身份、group 和规范 group_seq；不能由线程到达次序生成组内序号。

`compute_programs` 的 fill/matmul/sum_join 等操作读写命名 buffer，collective 明确绑定 buffer。图依赖决定允许何时推进，buffer 绑定决定实际数据在哪里。两者须共同校验，不能只靠节点连线推断数据流。

`submit_after` 表达等待指定通信已提交后才允许推进相关计算的关系，不等同于等待通信物理完成。application terminals 定义应用计时终点，不能省略为任意最后一条 host 调用。

workload seed、tensor 初始化 seed 与 arm 执行顺序随机化承担不同作用。固定输入工作量及重复测量身份，不固定所有节点绝对 ready 时刻。策略使用名义/profile 估计，不能读取未来实际扰动来排序。

### 2.2 单次 Phase 3 replay

1. **父进程校验。** `replay_launcher` 解析输入和 profile，拒绝 GPU 非 schema-v2 路径，核对通信引擎、静态顺序及 backend 条件，建立本次启动配置。
2. **启动 ranks。** 父进程为本次 rendezvous 分配端口，启动 rank worker 并收集日志；仅在符合条件的启动失败上有限重试，不能把已经执行任务的失败当作无损重启。
3. **初始化。** worker 初始化 PyTorch process group，按共同定义创建 job/group；GPU rank 使用继承可见设备中的逻辑 `cuda:rank`。新 runtime 建立独立控制通道。
4. **准备资源。** 创建 tensor/buffer、计算程序和通信绑定，执行 warmup 并恢复输入状态。初始化、warmup 与后续 compute/consumer 之间建立 stream/event 依赖。
5. **共同释放应用。** 准备阶段完成后开始应用测量。每个 job 的 DAG runner 根据节点依赖推进，GPU 当前保持每 job 一个 active compute 的模型。
6. **提交通信。** runner 构造 TaskSpec、TaskHint 和本地 LocalBinding，调用选定 adapter；获得 handle 后继续推进允许独立执行的节点，不在 submit 内等待全局 grant。
7. **推进与完成。** 计算 receipt 和通信 handle 分别报告完成，runner 解锁后继。GPU program 缺失是错误，不退回 sleep。
8. **终点与清理。** 记录 application terminals 的设备完成、每 job 与 rank 的应用终点；随后 finish/drain，检查 tensor、任务覆盖、顺序和失败状态，关闭 worker 与控制服务。
9. **父进程汇总。** 聚合各 rank JSON，独立校验并生成 validation、performance、metrics 和 config。进程退出码不是唯一正确性证据。

### 2.3 新 runtime 的通信闭环

```text
DAG ready
  → RankRuntime.submit(TaskSpec, LocalBinding, TaskHint)
  → OFFER → coordinator：成员就绪 + group_seq + 容量 → policy
  → GRANT（共同且不可撤销的决定）
  → rank 唯一 launch worker → PyTorch collective
  → SUBMITTED → 独立物理完成探测 → COMPLETED
  → coordinator 收齐所有成员完成 → 释放全局容量
```

控制消息走独立 TCP，不使用受调度的 job collective 作为 rendezvous。tensor、ProcessGroup、CUDA event 和 closure 留在 rank 本地，不发送给 coordinator。lookahead 路径还使用 DECLARE 等预测信息，其主动等待有固定预算，不属于当前七臂中的独立 arm。

新 runtime 当前全局 `max_inflight=1`；容量从 grant 到全部成员物理完成期间占用。应用何时消费 handle 不控制容量释放。每个 rank 的实际 launch 必须是共同决定序列的本地投影前缀。

NCCL `Work.wait()` 的 host 返回不能直接视为 GPU 物理完成。执行器建立 CUDA stream 依赖并通过事件观察设备完成；consumer 的 stream 依赖和 host 等待是不同接口。失败路径须停发、传播首个根因、唤醒等待者并在 deadline 内清理。

### 2.4 结果如何进入分析

```text
rank trace / tensor validation / collective launch / DAG events
  → 单次 replay JSON
  → batch raw + logs + runs ledger + attempt 状态
  → 独立校验 + 完整配对块筛选
  → 按 seed/repeat 配对的 makespan、mean/逐 job JCT、候选/顺序证据
  → analysis.json、CSV、audit、图表
```

应用 makespan、通信 drain 和总 wall time 分别报告；环境准备、进程启动、warmup 不应混成应用耗时。跨 rank 使用可比较的本地持续时间聚合，不直接相减不同 rank 未同步的绝对时间戳。

失败 attempt 保留。比较使用同一场景、seed、repeat 的完整块，不拼接不同失败重试块中的“最好结果”。完成数、语义通过、机制通过和性能差异应分别阅读；置信区间跨零或负收益不是实现失败。

## 3. 实验方式与模块

### 3.1 当前 GPU 实验

所有 arm 使用同一 schema-v2 输入、GPU DAG 资源、估计与应用终点。不同之处是通信准入和发射方式。

| Arm | 引擎 / policy | 顺序与容量 |
| --- | --- | --- |
| `bare-ordered` | `bare / bare` | 冻结默认顺序，单 dispatcher 有序提交，独立观察完成，允许多在途，不经过 JobPacer 中央准入 |
| `old-static-fifo` | `old / static_fifo` | 共同 FIFO Plan，旧 scheduler 本地单在途 |
| `old-static-ltf` | `old / static_ltf` | 共同 LTF Plan，旧 scheduler 本地单在途 |
| `new-static-fifo` | `new / static_fifo` | 新 runtime 严格静态队首，全局单在途 |
| `new-static-ltf` | `new / static_ltf` | 新 runtime 严格静态 LTF 队首，全局单在途 |
| `new-dynamic-fifo` | `new / fifo` | 从当前合法候选按首次 eligible 次序选，全局单在途 |
| `new-dynamic-ltf` | `new / ltf` | 从当前合法候选按 estimated_comm + remaining_tail 优先选，全局单在途 |

默认 FIFO 为 `topological-layer-job-round-robin-v1`，层级用于构造静态顺序，不额外变成运行时层间 barrier。StaticOrder 的队首未 eligible 时必须等待，不能跳过；动态策略才可选其他合法候选。

旧 scheduler 本地容量与新 runtime 全成员完成门控有差异，old/new 耗时差不能全部归为 policy 计算成本。bare 保持 NCCL 正确性顺序不等于中央性能调度；多在途也不自动证明 kernel 重叠或吞吐提升。当前 bare 有双 rank/backend 条件限制，须经过实际资格检查。

### 3.2 场景、随机样例与重复

`suite.py` 定义 L0 均衡链、L1 偏斜链、D0 fork/join、D1 不对称竞争前沿、D2 跨 job 偏斜、D3 group 顺序及多 sink 六个场景。L0/L1 也是 DAG，不使用旧线性 GPU schema。

默认 pilot seeds 为 9101–9103，formal seeds 为 8101–8105。正式目标为 6 场景 × 5 seeds × 5 repeats × 7 arms，即 150 个配对块、1,050 次 replay；这是配置目标，不是完成声明。pilot 重复次数应显式指定，例如 3 次。

D1 要证明同一决策时刻出现可区分的多个合法候选，且 FIFO/LTF 有不同选择机会；不能只观察两次 replay 顺序不同。L1 检查静态 HOL 与动态绕过。参数修订另起 revision，不能按结果选择有利 seed 或覆盖失败样例。

### 3.3 标定、诊断与资格

- 通信 profile：按 backend、设备、group 大小、collective 参数等匹配真实无竞争通信测量。
- GPU compute profile：按 program 输入输出形状、操作、repeats、UUID 与软件条件匹配，覆盖正式候选和 pilot 所需签名。
- backend/bare qualification：检查机制证据、六场景语义和故障路径；不能只用成功退出判断。
- measurement：A/A 与 minimal/full 配对测量，检查观测开销与语义一致性。
- recovery：启动冲突、中断恢复、整块重试和源码/hash 防串用。
- interference、job-order pilot：用于解释性能或默认顺序敏感性，不充当正式收益结论。

七类 readiness gate 是 software、backend、mechanism、measurement、rehearsal、recovery、budget。`check --status` 读取已有证据，不执行补测；缺证据必须补齐，不能制造空 audit。源码、profile、输入改变会影响证据绑定。

当前 `qualify.py` 的 audit 明确列出网络断连、重复 epoch 不在该检查覆盖范围内。资格通过只覆盖记录的检查集合，不代表所有分布式故障均已验收。

### 3.4 保留的 Gloo 实验

Phase 1/2 使用旧 Workload、Plan builder 与 replay worker，可比较裸发、静态 FIFO/LTF、ready-first、容量与 polling。Phase 3 Gloo 可以运行新 runtime 的线性输入或 schema-v1 DAG，研究静态/动态策略、Lookahead、就绪偏斜及协议开销。

历史 batch 在 `runner_batch.py`、`gloo_phase3_batch.py`、`compact_suite.py` 中，shared/isolated 诊断在 diagnostics。它们保留各自的测量合同，不把 CPU sleep 和 GPU program 当作等价性能样本。

## 4. 使用方法

所有命令从仓库根目录执行，使用已有 `.venv`；不需要为目录整理安装或升级依赖。以下为操作示例，不代表本文已执行实验。

### 4.1 环境与帮助

```bash
source .venv/bin/activate
export PYTHONPATH=src:.
python -m examples.jobpacer.scripts.run_phase3 --help
python -m examples.jobpacer.runtime.replay_launcher --help
nvidia-smi --query-gpu=index,name,uuid --format=csv
```

继承已分配的 `CUDA_VISIBLE_DEVICES`，不要覆盖它。当前 examples 需要仓库在 Python 搜索路径上；`pyproject.toml` 打包发现范围仍为 `src`，安装核心包不等于安装了 examples。

### 4.2 最小 Gloo replay

新 runtime：

```bash
python -m examples.jobpacer.runtime.replay_launcher \
  --policy fifo --workload balanced --backend gloo \
  --world-size 2 --timeout 20 --output /tmp/jobpacer-new-gloo.json
```

旧 scheduler：

```bash
python -m examples.jobpacer.scripts.run_phase2 \
  --mode scheduler --policy fifo --workload balanced --backend gloo \
  --world-size 2 --max-outstanding 1 --output /tmp/jobpacer-old-gloo.json
```

通信标定并应用于旧线性 replay：

```bash
python -m examples.jobpacer.scripts.run_comm_profile \
  --workload balanced --backend gloo --world-size 2 \
  --warmup 10 --iterations 50 --output /tmp/jobpacer-gloo-profile.json
python -m examples.jobpacer.scripts.run_phase2 \
  --mode scheduler --policy ltf --workload balanced --backend gloo \
  --world-size 2 --max-outstanding 1 \
  --comm-profile /tmp/jobpacer-gloo-profile.json --output /tmp/jobpacer-old-ltf.json
```

### 4.3 最小 GPU DAG replay

```bash
python -m examples.jobpacer.runtime.replay_launcher \
  --dag benchmark/phase3/experiments/dag-semantics/smoke/gpu-v2-fork-join.json \
  --backend nccl --world-size 2 --comm-engine new --policy fifo \
  --warmup-iterations 5 --setup-timeout 60 --timeout 30 \
  --output /tmp/jobpacer-gpu-dag.json
```

这是语义 smoke，不是正式性能对照。old 静态路径使用 `--comm-engine old --policy static_fifo`；现役 bare 使用 `--comm-engine bare --policy bare`。LTF 比较应使用严格匹配的通信和计算 profile，并遵守 suite 的共同顺序与输入约束。

### 4.4 准备新的 GPU 输入与 profile

为每轮使用新的目录，避免覆盖冻结产物。以下以 `/tmp/jobpacer-doc-demo` 为示例根目录。

```bash
python -m examples.jobpacer.scripts.run_phase3 prepare generate \
  --scope pilot --output-dir /tmp/jobpacer-doc-demo/pilot
python -m examples.jobpacer.scripts.run_phase3 prepare generate \
  --scope formal --output-dir /tmp/jobpacer-doc-demo/formal
python -m examples.jobpacer.scripts.run_gpu_compute_profile \
  --suite-input-dir /tmp/jobpacer-doc-demo/pilot \
  --suite-input-dir /tmp/jobpacer-doc-demo/formal \
  --warmup 5 --iterations 30 --output /tmp/jobpacer-doc-demo/compute.json
python -m examples.jobpacer.scripts.run_comm_profile \
  --dag /tmp/jobpacer-doc-demo/pilot/inputs/D1-asymmetric-frontiers/workload-9101.json \
  --backend nccl --world-size 2 --warmup 5 --iterations 30 \
  --output /tmp/jobpacer-doc-demo/comm.json
python -m examples.jobpacer.scripts.run_phase3 prepare freeze \
  --input-dir /tmp/jobpacer-doc-demo/pilot \
  --compute-profile /tmp/jobpacer-doc-demo/compute.json \
  --comm-profile /tmp/jobpacer-doc-demo/comm.json \
  --calibration-input-dir /tmp/jobpacer-doc-demo/formal
python -m examples.jobpacer.scripts.run_phase3 prepare freeze \
  --input-dir /tmp/jobpacer-doc-demo/formal \
  --compute-profile /tmp/jobpacer-doc-demo/compute.json \
  --comm-profile /tmp/jobpacer-doc-demo/comm.json
```

上面的通信 profile 示例依赖当前生成器 D1 覆盖所需通信签名；修改 workload 后必须重新检查签名覆盖，不能继续假设一个输入足够。profile 及其 sidecar 必须一起保存。

### 4.5 机制检查、资格与 pilot

```bash
python -m examples.jobpacer.scripts.run_phase3 check --kind mechanism \
  --output /tmp/jobpacer-doc-demo/mechanism --repeats 5 --warmup 5
python -m examples.jobpacer.scripts.run_phase3 qualify \
  --suite-manifest /tmp/jobpacer-doc-demo/pilot/suite-manifest.json \
  --mechanism-evidence /tmp/jobpacer-doc-demo/mechanism/mechanism-summary.json \
  --output-dir /tmp/jobpacer-doc-demo/qualification --warmup 5
```

`qualify` 产出及关联的 suite 状态须实际通过后再创建包含 bare 的 pilot。批次计划和执行是分开的命令：

```bash
python -m examples.jobpacer.scripts.run_phase3 run --plan-only \
  --suite-manifest /tmp/jobpacer-doc-demo/pilot/suite-manifest.json \
  --batch-dir /tmp/jobpacer-doc-demo/pilot-batch --repeats 3
python -m examples.jobpacer.scripts.run_phase3 run \
  --batch-dir /tmp/jobpacer-doc-demo/pilot-batch --allow-pilot
python -m examples.jobpacer.scripts.run_phase3 analyze \
  --batch-dir /tmp/jobpacer-doc-demo/pilot-batch
```

中断后使用同一批次加 `--resume`，仅当源码、输入和环境仍符合冻结合同。失败块按整块恢复，不将旧 attempt 的部分成功结果混入新块。

### 4.6 测量、恢复和正式封存

measurement 明确区分不启动 replay 的 plan/analyze 与实际执行的 run：

```bash
python -m examples.jobpacer.scripts.run_phase3 check --kind measurement --action plan \
  --suite-manifest /tmp/jobpacer-doc-demo/pilot/suite-manifest.json \
  --output /tmp/jobpacer-doc-demo/measurement --pairs 5
python -m examples.jobpacer.scripts.run_phase3 check --kind measurement --action run \
  --output /tmp/jobpacer-doc-demo/measurement
python -m examples.jobpacer.scripts.run_phase3 check --kind measurement --action analyze \
  --output /tmp/jobpacer-doc-demo/measurement
python -m examples.jobpacer.scripts.run_phase3 check --kind recovery \
  --suite-manifest /tmp/jobpacer-doc-demo/pilot/suite-manifest.json \
  --output /tmp/jobpacer-doc-demo/recovery
```

完整准备不是只执行上述两个检查。还需按合同收集 software、backend、mechanism、rehearsal、budget 等对应证据；CLI 不会自动运行全部补测。

后续顺序如下，路径参数使用实际产物，不用占位 audit 通过检查：

1. `finalize pilot --suite-dir <formal候选目录> --pilot-batch-dir <已通过pilot目录>`：把有效 pilot 关联到正式候选。
2. `check --status --suite-manifest <formal清单> --qualification <资格文件> --audit GATE=PATH ...`：只读检查各类证据。
3. `finalize formal --suite-manifest <formal清单> --qualification <资格文件> --audit GATE=PATH ...`：提供全部七类 audit，封存正式输入。
4. `run --plan-only --suite-manifest <formal清单> --batch-dir <新正式目录> --repeats 5`：创建正式计划，不执行。
5. `run --batch-dir <正式目录> --allow-formal-matrix`：显式启动长批次，仍需满足正式资格。
6. `analyze --batch-dir <正式目录>`：独立校验和统计，不启动通信。

`prepare order` 可以单独生成顺序表，`prepare preview --order <顺序表> --history-summary <历史CSV或runs.jsonl>` 可估算预算。没有历史覆盖时预算可为未知，不能把未知当作零。

### 4.7 历史工具和独立诊断

```bash
python -m examples.jobpacer.scripts.run_phase1_2_experiments --help
python -m examples.jobpacer.experiments.gloo_phase3_batch --help
python -m examples.jobpacer.experiments.compact_suite --help
python -m examples.jobpacer.diagnostics.bare_nccl_mechanism --help
python -m examples.jobpacer.diagnostics.gpu_interference_profile --help
python -m examples.jobpacer.diagnostics.layered_fifo_pilot --help
python -m examples.jobpacer.diagnostics.interleaved_isolated --help
python -m examples.jobpacer.analysis.visualize_phase3 --help
python -m examples.jobpacer.analysis.visualize --help
```

历史脚本可能依赖本机归档输入或迁移映射；帮助可用不证明旧批次可直接复现。历史 `run_phase3 --policy ...` 命令应使用当时归档源码，当前单次 replay 请调用 `runtime.replay_launcher`。

## 5. 验证、产物和交接

```bash
# 不主动启用真实集成实验的回归；注意 skipped
PYTHONPATH=src:. python -m pytest -q

# 真实 Gloo
PYTHONPATH=src:. RUN_JOBPACER_RUNTIME_REPLAY=1 \
  python -m pytest -q tests/integration/test_runtime_replay.py

# 确认两张分配的可见 GPU 后执行
PYTHONPATH=src:. RUN_JOBPACER_RUNTIME_NCCL=1 \
  python -m pytest -q tests/integration/test_runtime_replay_nccl.py

git diff --check
```