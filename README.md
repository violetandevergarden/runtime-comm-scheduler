# Runtime Communication Scheduler

本仓库研究真实 PyTorch collective 的运行时通信调度：在多个 job 共享通信资源时，利用在线就绪状态和工作量估计改变通信准入顺序，观察对应用完成时间与各 job 完成时间的影响。

JobPacer 已从 **Phase 1 无调度基线、Phase 2 静态 Plan**，推进到 **Phase 3 中心化在线 runtime 与 DAG 执行模型**。Gloo 保留线性 workload 和 DAG；GPU/NCCL 统一使用 schema-v2 DAG，执行真实 CUDA program 和 all-reduce。目前的实验支持动态准入缓解静态队首阻塞，但**尚未证明新 runtime 相对旧 scheduler 或 bare 有稳定、普遍的净性能收益**。

本文以 2026-09-30 的代码结构和仓库结果报告为依据。实现能力、特定批次的验收和性能结论分别说明；历史实验使用各自归档源码，不能把其通过记录直接视为当前工作树的验收。

## 1. 三个阶段实现了什么

| 阶段 | 研究问题 | 已实现路径 | 主要边界 |
| --- | --- | --- | --- |
| Phase 1 | 没有通信准入调度时，多 job 怎样竞争资源？ | 多 job/rank replay、真实异步 all-reduce、应用时间线和数值校验 | 历史主实验为 CPU/Gloo，计算由 sleep 模拟 |
| Phase 2 | 固定通信顺序、优先级和在途容量会产生什么影响？ | 旧 `AdmissionScheduler`、静态 `Plan`、FIFO/LTF，以及实验中的 SRJF；本地在途容量控制 | 顺序预先确定，队首未就绪时不能在线跳过；容量与新 runtime 的全局完成语义不同 |
| Phase 3.1 | 根据跨 rank 在线状态决定下一次通信，能否减少等待？ | 独立的新 runtime、中心 coordinator、TCP 控制协议、FIFO/LTF/StaticOrder/有界前瞻策略、异步 handle 和物理完成探测 | 当前 all-reduce、单个全局通信容量 `max_inflight=1` |
| Phase 3.2 | 在分支、汇合和计算/通信依赖下怎样推进任务？ | 上层 DAG 模型、合法性校验、异步计算节点、通信节点和依赖推进；GPU 显式 CUDA program、buffer 与应用终点 | 合成 workload，尚未完成真实 Megatron 训练接入验证 |

### Phase 1：建立无调度基线

历史 Gloo replay 为每个 job 建立通信组，在各 rank 上并行推进 job。线性输入描述 producer、collective 和 consumer 时间：producer 表示通信之前的工作；collective 使用真实 `torch.distributed` 调用；历史 consumer 通过 sleep 表示可与通信并行的独立计算，随后再等待通信完成。

这里的 consumer 名称不意味着它在读取 collective 的结果。这一历史语义不能直接移植成 GPU 上依赖通信结果的 tensor 运算。当前 GPU 用 DAG 的依赖、buffer 读写和终点明确表达这些关系，不再沿用 GPU 线性输入或 `cuda-matmul` 兼容分支。

### Phase 2：旧 scheduler 与静态 Plan

应用先构造通信任务和固定 Plan，各 rank 按共同顺序提交。FIFO、LTF 等改变 Plan 的顺序；`max_outstanding` 改变本地允许保留的在途任务数。通信 profile 为静态估计提供依据，必须匹配 backend、设备、通信签名和实验环境。

这一阶段建立了 Plan 一致性、实际 launch 顺序和 tensor 结果检查，也暴露出静态队首阻塞：排在前面的任务尚未就绪时，即使其他任务可以执行，也必须等待。旧核心保留在包根目录，作为历史实现和基线；新 runtime 不依赖旧 `Plan`、`TaskKey` 或 `ScheduledWork`。

### Phase 3.1：中心化在线通信准入

```text
应用 / DAG runner
  │ submit(TaskSpec, LocalBinding, TaskHint)，返回 handle，不等待准入
  ▼
各 rank RankRuntime ── OFFER ──▶ 中心 coordinator
                                  │ 成员请求到齐、group 顺序、容量校验
                                  ▼
                                eligible 候选 → policy → GRANT
                                  │
各 rank 唯一 launch worker ◀───────┘
  │ 按共同 grant 的本地投影发射真实 collective
  ├─ SUBMITTED
  └─ 物理完成探测 → COMPLETED → 全部成员完成后释放全局容量
```

`TaskSpec` 保存共同身份和 collective 规范，`TaskHint` 保存策略估计；tensor、ProcessGroup、CUDA event 和执行 closure 留在本地 binding。控制消息走独立 TCP 通道，不借用被调度的 collective 做同步。

Coordinator 负责合法性，policy 只从合法候选中选择或有界等待。FIFO 按首次 eligible 的次序排序；动态 LTF 使用通信和后续工作估计；StaticOrder 严格等待预先冻结的队首。Grant 是不可撤销的决定，实际发射不能重排。通信容量由全部成员的物理完成释放，不依赖应用什么时候调用 `wait()`；NCCL 的 host 等待和 CUDA stream 依赖也分别处理。

### Phase 3.2：DAG 与 GPU 执行

DAG runner 位于通信 runtime 上层，负责计算节点、通信节点及分支/汇合的推进。图校验同时考虑显式依赖与 group 内规范顺序，避免两套约束组合后形成环。计算完成通过执行 receipt 观察，不能把 CUDA kernel 提交返回当作计算完成。

GPU workload 只有一套 schema-v2 DAG 语义：显式描述 execution program、资源、buffer 关系和 `application_terminals`。GPU 模块负责 CUDA 资源与计算绑定，共同 DAG runner 接入 bare、旧 scheduler 或新 runtime adapter。线性形状也直接用 DAG 表达，不再先构造 GPU 线性 workload 再 bridge。

架构、状态转换和逐文件说明见 [核心包 README](src/runtime_comm_scheduler/README.md)；workload 到实验产物的数据流见 [JobPacer README](examples/jobpacer/README.md)。

## 2. 当前 GPU 七臂比较的含义

| Arm | 执行方式 | 要回答的问题 |
| --- | --- | --- |
| `bare-ordered` | 共同固定发射顺序，多在途；没有在线准入策略和逐项中央 grant | 无调度准入时的直接执行参考 |
| `old-static-fifo` / `old-static-ltf` | 共同 DAG runner 接旧 scheduler，执行冻结的 FIFO/LTF Plan | 旧路径的静态基线 |
| `new-static-fifo` / `new-static-ltf` | 新 runtime 执行对应冻结顺序，队首不可跳过 | 新路径的静态基线及迁移成本参照 |
| `new-dynamic-fifo` / `new-dynamic-ltf` | 新 runtime 根据当前 eligible 集合在线选择 | 在线绕过等待，以及优先级选择的增量价值 |

当前 bare 仍需遵守 NCCL 的顺序和设备执行约束。它使用与估计无关的分层轮转规则冻结通信投影，由单一发射入口执行；前项尚未物理完成时可以继续发射后项，但队首尚未可提交时仍须等待。它既不是各 rank 任意抢发，也不保证每个场景都出现多在途。实际 backend 资格还受版本、配置和执行路径约束。

因此，bare 与受限 scheduler 的比较同时包含**准入控制、在途容量、顺序和执行路径**的差异，不能全部归为 policy 收益。静态与动态策略的主要比较应在新 runtime 内完成；旧路径与新路径比较也包含本地/全局完成语义的差异。

## 3. Phase 1/2：容量、顺序和目标函数

历史 CPU/Gloo 容量批次完成 720 次 replay，轮询批次完成 80 次，SRJF 批次完成 1,040 次。这些是不同批次，不能跨批次拼成一张公平排名表。以下保留其中能说明问题的结果，详细统计见 [Phase 1–3 汇总](docs/JobPacer/result/phase1-3-summary-and-gpu-readiness-20260926.md)和 [Phase 2 结果](docs/JobPacer/result/phase2.md)。

通信密集场景含 3 个 job，每 job 4 次 16 MiB 通信、producer 0.5 ms。各 arm 的 workload makespan 中位数如下，单位 ms；这是各 arm 中位数，非配对差值。

| bare | FIFO 容量 1 | FIFO 容量 2 | FIFO 容量 3 | FIFO 不限容量 | LTF 容量 1 | LTF 容量 3 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 85.368 | 179.142 | 106.784 | 92.316 | 92.427 | 174.952 | 88.998 |

这一场景中，放宽在途容量的影响大于 FIFO/LTF 标签的差异；FIFO/LTF 的 Plan 实际相同，耗时差异不能解释为排序收益。长计算重叠场景则没有显示同样清晰的容量收益：在 0.1 ms 轮询配置下，FIFO 各容量的中位数都约为 44 ms。更多在途是否有用，取决于通信空档、依赖与计算余量。

SRJF 实验进一步表明，**makespan 与平均 JCT 是不同目标**。通信密集场景容量为 1 时，SRJF−FIFO 的配对中位差为 makespan −0.076 ms、平均 JCT −46.034 ms；容量为 3 时，两者变为 +65.136 ms、+19.082 ms。优先短 job 可以改善某些 job 的完成时间，也可能因固定顺序降低实际并发、恶化整体结果。

这一阶段的启发是：必须同时报告容量、实际发射顺序、makespan 和逐 job JCT，不能只凭策略名称评价算法，也不能把 bare 的优势直接解释成“调度一定无效”。

## 4. Phase 3 CPU：在线机制有效，但净收益尚未确立

### 动态绕过静态等待已经观察到

2026-09-23 compact suite 完成 660 次有效 replay。在新 runtime 内，静态 FIFO 与动态 FIFO 的配对 speedup（静态耗时 / 动态耗时）为：

| 场景 | Speedup | 95% CI | 解释 |
| --- | ---: | --- | --- |
| L0，基准场景 | 1.045 | 0.999–1.100 | 未显示稳定收益 |
| L1，静态队首错位 | 1.224 | 1.198–1.283 | 支持动态绕过队首等待 |
| L5，成员就绪偏斜 | 1.200 | 1.154–1.269 | 支持利用在线成员状态 |

但动态 LTF 相对动态 FIFO 没有显示进一步稳定收益。多候选优先级和前瞻等待的正向机制 gate 也没有全部通过：若绝大多数 dispatch 只有一个合法候选，实验并没有充分测试 LTF 的选择能力。早期筛选中的较大加速不能替代后续严格验收。来源：[CPU 汇总与机制审核](docs/JobPacer/result/phase1-3-summary-and-gpu-readiness-20260926.md)。

后续采用 tensor/binding 预创建的独立 Phase D 批次，L1 的 new Static FIFO → new FIFO speedup 为 1.232，95% CI 为 1.161–1.281；但 new FIFO 相对 old FIFO 的配对 makespan 仍慢 0.571 ms。L5 相对 old FIFO 的 speedup 为 1.011，区间 0.973–1.016，跨过无收益点。这一批没有同批 bare，不能补入早期 compact 的 bare 数字作排名。来源：[Phase D 结果与结论收窄](docs/JobPacer/result/phase1-3-summary-and-gpu-readiness-20260926.md)。

### 约 2 ms 是执行闭环差值，不是 policy 算法耗时

零计算、32 项串行通信链中，新静态路径比旧 FIFO 多约 65–71 ms，折合约 2.03–2.21 ms/项。细分诊断中 policy 函数约 3 µs；差值涉及请求到齐、发送锁、线程交接、发射、完成探测、完成反馈和应用继续推进，尚未找到一个独占这 2 ms 的函数。

预创建 binding、减少独立 DECLARE 等干预有局部改善；更频繁轮询或提前唤醒 completion thread 则不保证应用加速，还可能增加 CPU 和观测成本。因此，这不是已证明无法优化的“中心化下限”，也不能承诺简单合并消息就能消除。早期 40–48 ms 的 TCP 反馈问题与修复后约 2 ms 的闭环差值应分开理解。来源：[runtime 开销实验](docs/JobPacer/result/phase3-runtime-overhead-20260924.md)与 [Phase 3 修复记录](docs/JobPacer/result/phase3.12fix.md)。

## 5. Phase 3 GPU：历史发现与最新实验边界

### 2026-09-27 首轮双卡实验

该批次使用 2× RTX 4090、Python 3.12、PyTorch 2.13.0+cu126、CUDA runtime 12.6、NCCL 2.29.3。这是历史批次环境，不代表当前机器探测结果。

固定 32 项通信链的 makespan 中位数如下，单位 ms：

| 通信大小 | 旧路径 | 新 runtime |
| --- | ---: | ---: |
| 4 KiB | 21.955 | 172.699 |
| 1 MiB | 27.366 | 177.526 |
| 16 MiB | 56.242 | 228.950 |

归一化差值约为 4.7–5.4 ms/项，比历史 CPU 数值更大；这仍是整体执行路径差异，不能直接称为网络或纯 coordinator 成本。L1 中，新静态 FIFO → 动态 FIFO 的 speedup 为 1.421，95% CI 为 1.316–1.494，但仍未证明相对旧路径的净收益。

真实 GPU 计算诊断没有建立可重复的 kernel overlap 证据。它不能证明 GPU 计算/通信一定不能重叠，也不能直接和 CPU sleep 的“重叠时间”比较。首轮之后，GPU workload 已改为统一显式 DAG 语义；早期线性/bridge 的依赖和应用终点问题使这些批次不能直接与当前输入配对。来源：[首轮 GPU 报告](docs/JobPacer/result/phase3-gpu-20260927.md)与 [统一 DAG 修正结果](docs/JobPacer/result/phase3-gpu-unified-dag-correction-20260928.md)。

### 2026-09-29 至 09-30：七臂 v2 准备批次

最新仓库报告记录 revision-1 pilot 完成 **54 个七臂块、378 次有效 replay**，矩阵和语义校验通过；对应归档版本的 bare/backend 资格已有通过记录，不能继续沿用早期“bare 尚未通过资格”的状态。但该 v2 批次的正式 1,050 次采集**尚未启动**，原因是 D1 机制门槛未通过：

| 验收项 | 结果 |
| --- | --- |
| L1 静态队首阻塞与动态合法绕过 | 三个 seed 均为 3/3，通过 |
| D1 多候选、分数不同且 FIFO/LTF 分歧 | 三个 seed 分别 1/3、0/3、2/3；未达到至少两个 seed 各 2/3 的门槛 |
| 软件、backend、测量、彩排、恢复、预算审计 | 对应 revision 的六项通过 |
| 正式采集放行 | mechanism 未通过，不能放行 |

D1 并非完全没有发生策略分歧，而是触发频率不足。它限制的是优先级实验的有效性，不能拿 L1 的成功代替，也不能把完整执行 pilot 等同于正式实验完成。此处的 v2 状态与历史使用 raw 替代臂的批次分别记录。

辅助零计算链还给出了“bare 总是最快”的反例：8 次 collective、每路径 3 次重复时，4 KiB 的 bare/old/new makespan 中位数为 22.75/9.28/51.11 ms，64 MiB 为 65.30/50.90/95.89 ms。样本小且无计算重叠，这只能作为执行路径诊断，不能确立一般排名。A/A 的中位绝对差约为 0.8–1.5 ms，也提示毫秒级小收益必须结合噪声和配对区间判断。

上述事实来自 [七臂 v2 验收报告](docs/JobPacer/result/phase3-gpu-seven-arm-preparation-20260929.md)。资格与源码/profile 绑定，目录整理后的代码仍需按变更范围复核，不能继承归档快照的全部验收结论。

## 6. 实验带来的研究启发

以下判断综合已有结果与 [discussion](docs/JobPacer/plan/discussion.md)。其中关于带宽、训练代表性和架构改进的解释仍需独立实验；discussion 中早期的执行状态以对应后续结果报告为准。

1. **先区分调度机会与系统净收益。** L1 已证明在线准入可以减少静态等待；证明 LTF 价值还需多个合法候选和不同优先级。最终收益必须覆盖关键路径上新增的请求、反馈和执行成本，不能只看调度顺序更合理。
2. **Bare 的潜在优势不仅是少一层调用。** 多在途可能减少通信空档，并利用独立计算尚未完成的时间服务其他 job。即使某个 collective 变慢，应用终点也可能不变。但带宽利用、计算干扰和关键路径余量都需要测量；多在途、设备 kernel overlap 和吞吐提高不是同一个事实。Consumer 尚未结束也不等于 runtime 仍扣着通信容量。
3. **大消息不自动让固定成本无关紧要。** 如果通信大部分已被计算隐藏，新增几毫秒仍可能落在暴露的关键路径上。应研究成本与 ready 偏斜、通信服务时间、计算余量、候选集合规模的关系，找到何时收益能抵消成本。
4. **降低决策频率值得研究，但必须保留共同顺序。** 元数据复用、稳定区间复用计划、批量提交或区间授权可能减少逐项闭环；等待凑批也会引入延迟。当前证据尚未证明中心 coordinator 本身是瓶颈，去中心化仍需解决各 rank 视图不同、共同决定和失败恢复，不能预设一定更快。


下一步应先修订并冻结能稳定触发 D1 的实验设计，重跑受影响的资格与 pilot；随后完成同一输入、profile、扰动和观测口径下的配对矩阵，再研究净收益边界与真实训练代表性。Phase 5 的部署扩展和 Phase 6 的多资源/并发选择仍是后续方向。

## 7. 仓库结构与阅读入口

| 路径 | 职责 |
| --- | --- |
| [`src/runtime_comm_scheduler/`](src/runtime_comm_scheduler/README.md) | 核心实现；`runtime/` 提供在线协议和执行基础，`dag/` 提供图模型与推进，包根目录保留旧 scheduler |
| [`examples/jobpacer/`](examples/jobpacer/README.md) | workload 适配、实验编排、分析和命令行入口 |
| `examples/jobpacer/gloo/` | 历史 Gloo 线性 workload、builder、profile 应用层与 adapter |
| `examples/jobpacer/gpu/` | GPU DAG 的 CUDA 资源、计算 profile 和设备辅助 |
| `examples/jobpacer/runtime/` | DAG adapter、rank worker、replay 启动，以及旧 Plan builder/replay worker |
| `examples/jobpacer/scripts/` | 实验入口；七臂统一通过 `run_phase3` 子命令操作 |
| `examples/jobpacer/experiments/` | 输入生成、冻结、资格、批次执行和封存 |
| `examples/jobpacer/analysis/` | 结果读取、校验、统计与可视化 |
| `examples/jobpacer/diagnostics/` | 测量扰动、bare/NCCL、干扰与恢复等诊断 |
| `benchmark/` | 本地实验输入与产物工作区；原始结果不随普通 Git 提交分发 |
| `tests/` | 单元测试及 opt-in Gloo/NCCL 集成 |
| `docs/JobPacer/plan/`、`process/`、`result/` | 分别存放设计计划、实施过程、已验证结果 |

理解设计先读 [讨论总结](docs/JobPacer/plan/discussion.md)、[Phase 3.1 计划](docs/JobPacer/plan/phase3.1.md)和 [Phase 3.2 计划](docs/JobPacer/plan/phase3.2.md)。完整使用方法、各模块职责和实验命令见 JobPacer README；协作约束见 [AGENTS.md](AGENTS.md)。

## 8. 使用与复现

从仓库根目录使用已有 `.venv`。项目声明 Python ≥3.10；PyTorch、CUDA 和 NCCL 按目标机器环境配置，不以历史报告中的版本代替运行时探测。

```bash
source .venv/bin/activate

# 单次新 runtime Gloo replay
PYTHONPATH=src:. python -m examples.jobpacer.runtime.replay_launcher \
  --policy fifo --workload balanced --backend gloo \
  --world-size 2 --timeout 20 --output /tmp/jobpacer-phase3-gloo.json

# Phase 1/2 历史实验入口
PYTHONPATH=src:. python -m examples.jobpacer.scripts.run_phase1 --help
PYTHONPATH=src:. python -m examples.jobpacer.scripts.run_phase2 --help

# GPU 七臂统一入口：prepare / qualify / check / run / analyze / finalize
PYTHONPATH=src:. python -m examples.jobpacer.scripts.run_phase3 --help
```

GPU 单次 replay 使用 `replay_launcher --dag <schema-v2.json> --backend nccl`；完整参数与七臂流程见 [JobPacer 使用说明](examples/jobpacer/README.md)。七臂流程包含输入生成/profile 冻结、资格、pilot、审计、正式输入封存、正式执行与分析；子命令名称列表不表示可以跳过放行检查直接启动正式实验。

```bash
# 核心单元检查与全仓测试
PYTHONPATH=src python -m pytest -q tests/unit/runtime
PYTHONPATH=src python -m pytest -q

# 真实双 rank Gloo；需要本地 TCP socket 权限
PYTHONPATH=src RUN_JOBPACER_RUNTIME_REPLAY=1 \
  python -m pytest -q tests/integration/test_runtime_replay.py

# 双卡 NCCL：先核对可见设备，保留资源分配的 CUDA_VISIBLE_DEVICES
nvidia-smi --query-gpu=index,name,uuid --format=csv
PYTHONPATH=src RUN_JOBPACER_RUNTIME_NCCL=1 \
  python -m pytest -q tests/integration/test_runtime_replay_nccl.py
```

这些是验证入口，不代表本次 README 更新运行了测试。Skipped 不算通过，协议正确、GPU 物理完成、故障退出和性能收益需要分别验收。

复现实验必须保存输入、随机 seed、重复编号、arm 顺序、profile、源码快照、环境、命令、manifest、raw 与哈希。`benchmark/phase3/results/` 被 Git 忽略，普通 clone/commit 不携带原始结果；交接需单独归档。不能用当前源码直接重跑历史语义，也不能把不同 revision、不同观测模式或 raw/bare 定义的数据合并。报告中的批次数字是历史事实，不是当前全仓测试通过声明。
