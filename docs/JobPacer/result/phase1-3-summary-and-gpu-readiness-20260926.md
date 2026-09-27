# JobPacer Phase 1–3：实现、CPU 实验结论与 GPU 测试准备

日期：2026-09-26。覆盖 Phase 1、Phase 2、Phase 3.1/3.2，以及截至 2026-09-25 的 runtime-overhead A–F 实验。

本文是现状综述和 GPU 测试交接材料。第 1–6 节记录实现、已有实验及解释；第 7–10 节是**尚未执行的 GPU 建议方案**，不是 GPU 验收结果。按用户要求集中放在 `result/`；后续正式实施时应将冻结后的执行计划放入 `plan/`，过程放入 `process/`，实测事实再追加到 `result/`。

本次只核对代码、文档及已有统计，不修改运行时、不重跑性能或测试。文中的通过数来自对应历史记录，不代表在当前工作树重新执行。工作区已有未提交修改；历史批次以其 manifest/源码摘要为准，不能把当前 HEAD 当成全部实验实现。

## 1. 核心结论

1. **Phase 1/2 已建立真实 collective 的裸发和静态 Plan 对照。** 容量扫描表明，在当前 CPU/Gloo 通信密集负载上，单在途限制可能比 FIFO/LTF 排序本身更影响 makespan。SRJF 能改变 job 完成时间分配，但并非普遍改善平均 JCT 或总 makespan。
2. **Phase 3.1 已完成 CPU/Gloo 的在线协调、共同 grant、唯一 launch 入口和全成员完成释放容量。** Phase 3.2 已完成 CPU sleep DAG 推进及真实 Gloo 通信组合验收；这不等于 GPU 计算图或真实训练验收。
3. **动态调度有机制价值，但新系统没有取得普遍净收益。** L1 队首错位、L5 成员偏斜中，新动态 FIFO 相对新 Static FIFO 有约 19%–23% 的配对 speedup；与旧 scheduler 比，优势大多被额外执行路径成本抵消。L0 中新增成本尤其明显。
4. **约 2 ms/collective 不是 policy 函数算出来的。** 当前证据指向串行闭环中的应用推进、DECLARE/OFFER、成员到齐、线程交接、完成探测及回执处理的合并影响；policy 计算中位约 3 µs。尚未完成互不重叠的精确因果分解。
5. **下一步应转入 GPU 的语义验收和小规模边界测量，而非复制完整 CPU 矩阵。** 先补齐/验证 producer、consumer、物理完成与计时边界，再测固定开销和动态收益能否抵消成本。不能仅把 `--backend gloo` 改成 `nccl` 就当作公平性能实验。

## 2. 当前实现：四条执行路径与三个阶段

### 2.1 四条路径的实质区别

| 路径 | 谁决定跨 job 顺序 | 准入与发射 | 容量及完成 | 主要用途 |
| --- | --- | --- | --- | --- |
| Phase 1 bare | 各 job 本地实际推进/线程到达 | job 线程直接调用异步 collective | 无 scheduler 全局容量；应用调用 backend `Work.wait()`，另有完成观测 | 真实无调度路径参考，不是零开销理想值 |
| Phase 2 旧静态 Plan | 离线 FIFO/LTF/SRJF | rank-local `AdmissionScheduler` 按固定 Plan 准入，由 scheduler worker 发射 | `max_outstanding` 在各 rank 本地限制；普通静态路径没有全成员完成屏障 | 静态排序、容量扫描及历史系统基线 |
| Phase 3 新 StaticOrder | 离线 task ID 序列 | 完整在线 OFFER/GRANT 协议；静态头不 eligible 就等待；唯一 launch worker | 全局 `max_inflight=1`，全部成员物理完成回执到齐才释放 | 控制执行路径不变的静态对照 |
| Phase 3 新 runtime 决策 | coordinator 按在线合法候选调用 FIFO/LTF/Lookahead | 与新 StaticOrder 相同执行路径，只改变选择/有界等待 | 与新 StaticOrder 相同 | 检验在线状态适应能力 |

四组并不是逐级只增加一个开关：

- bare → old static：混合了调度执行路径、排序和并发限制；不是纯 scheduler CPU 开销。
- old static → new static：包含控制协议、全局完成语义、应用等待及线程结构变化；不是动态策略计算成本。
- new static → new dynamic：最适合评估同 runtime 内在线选择的价值，但 LTF 还需注意第 2.4 节的评分差异。
- bare/old static → new dynamic：回答系统净效果，不能只用 new static 这个较慢分母证明新系统优越。

### 2.2 Phase 1：无调度 multi-job replay

当前入口为 [`run_phase1.py`](../../../examples/jobpacer/scripts/run_phase1.py)，与 Phase 2 共用
[`replay_worker.py`](../../../examples/jobpacer/runtime/replay_worker.py) 和线性 workload。

- 每 rank 一个进程，同进程内多个 job 线程，每 job 独立 ProcessGroup。
- 每段语义为 `producer → async all_reduce → independent consumer compute → wait/消费`。这里 consumer compute 是允许与通信重叠的独立工作，不是必须在通信完成后才能开始的依赖消费者。
- 现有 CPU 实验以 sleep 表示计算窗口，collective 是真实 PyTorch all-reduce，不是模拟通信。
- tensor 准备在应用 release 之前；数值扫描在应用结束、通信排空之后。准备、应用、drain、validation 和 harness 总耗时分开记录。
- bare 不经过 Plan admission；为了字段/身份一致而生成的 policy 标签或 key 不代表它执行该 Plan。
- 独立完成观测使裸发也带有 harness 成本。观测在途峰值不是精确设备并发度，更不是物理带宽利用率。

Phase 1 的 CPU/Gloo 成功不能证明多个 job 线程自由切换多个 NCCL communicator 安全；GPU bare 必须单独审查，不能作为自动可用的第一组实验。

### 2.3 Phase 2：旧 scheduler 与静态 Plan

入口为 [`run_phase2.py`](../../../examples/jobpacer/scripts/run_phase2.py)，主要组件：

- [`plan_builder.py`](../../../examples/jobpacer/runtime/plan_builder.py)：确定性 TaskKey、FIFO/LTF/SRJF 序列和评分诊断。
- `src/runtime_comm_scheduler/plan.py`、`scheduler.py`、`intent.py`、`work.py`：旧 Plan、提交、准入、底层 Work 绑定和应用等待。
- [`executor.py`](../../../src/runtime_comm_scheduler/executor.py)：CPU direct executor；历史 CUDA executor 按 group 建立 gate stream、桥接 producer event。
- [`comm_profile.py`](../../../examples/jobpacer/comm_profile.py) 保存签名、估计及摘要；[`run_comm_profile.py`](../../../examples/jobpacer/scripts/run_comm_profile.py) 执行独立校准。二者分别是数据模型/应用逻辑与运行入口，不是重复 profiling。

静态 FIFO 是预先固定轮转，不是在线 ready FIFO。LTF/SRJF 在每个 job 当前离线候选中选择最长/最短估计剩余路径；不抢占，不根据真实执行中的新 ready 时刻重排。

当前共同评分为：

```text
remaining(i) = max(estimated_comm_i, consumer_compute_i)
             + Σ后继j [producer_compute_j + max(estimated_comm_j, consumer_compute_j)]
```

它假设当前候选已经 ready、准入延迟为零，用 `max` 表示通信与独立 consumer compute 的重叠；不是精确的通信完成后 tail。

`max_outstanding=1/2/3/0` 分别表示有限本地容量/不主动限额。同一静态 Plan 在不同容量下可能达到不同实际并发；相同容量的不同 Plan 也可能因队首阻塞而达不到相同并发。

历史 `ready_first_serial/unbounded` 是额外诊断对照：通过专用控制 ProcessGroup 协调全局 ready，serial 模式另带完成 barrier。不能把它当成普通旧 FIFO，也不能把它与 bare 的差简单解释为串行化成本。

### 2.4 Phase 3.1：独立的新 runtime

核心在 [`src/runtime_comm_scheduler/runtime/`](../../../src/runtime_comm_scheduler/runtime/)，不依赖旧 scheduler 核心：

| 组件 | 当前职责 |
| --- | --- |
| `model.py` | GroupSpec、TaskSpec、CollectiveSpec、TaskHint、LocalBinding；规范/估计/本地对象分离 |
| `protocol.py`、`transport.py` | 版本化消息、独立 TCP 控制面、序号、接收队列及有序发布；已修复 TCP_NODELAY 问题 |
| `coordinator.py` | 单事件循环维护成员 OFFER、group 顺序、eligible、全局容量、完成与失败 |
| `policy.py` | StaticOrder、FIFO、LTF、BoundedLookahead；只选择合法候选或等待 |
| `runtime.py` | 本地提交、控制接收、单 launch worker、独立 completion thread、排空/失败 |
| `handle.py` | 异步 handle、grant/binding/completed 状态、host 等待与消费依赖接口 |
| `executor.py`、`telemetry.py` | direct launch、Work completion query、最小/诊断事件观测 |

主要不变量已经在 CPU/Gloo 路径验收：任务身份与 group_seq 跨成员一致；OFFER 全到且顺序合法才可 grant；grant 不撤销；实际 launch 是共同 grant 的本地投影；SUBMITTED 先于 COMPLETED；全成员完成才释放容量；输入关闭排在已接受提交之后；失败唤醒等待者并有界退出。

FIFO 按首次 eligible 序号选择，而非注册顺序。Lookahead 只考虑安全前沿、比较等待收益并使用固定 deadline，事件不能无限刷新等待预算。StaticOrder 队首不满足条件时不能动态跳过。

`submit()` 不等 grant，但当前实现仍同步执行本地校验、保存和控制消息发送；“异步提交”不等于零 host 成本或无发送锁等待。

当前线性 CLI 默认：`binding-preparation=precreate`、completion poll 1 ms、`declaration-mode=before-producer`、提交唤醒完成探测关闭。不同 runner 的观测模式应显式记录，不能假定全部默认 minimal。precreate 只把资源准备移出应用计时，不删除其成本；DAG 没有自动套用这项线性优化。

**本次代码核对发现的评分边界：**线性新 Static LTF 通过 adapter 调用旧 Plan builder，仅取 task ID 顺序；这是 examples 层的对照适配，不是新核心复用旧 scheduler。但新动态 LTF 的 `runtime_adapter.remaining_tail()` 实际为：

```text
tail(i) = consumer_compute_i
        + Σ后继j [producer_compute_j + estimated_comm_j + consumer_compute_j]
```

动态 LTF 按该 tail 排序，静态 LTF 使用上面的 `max` 评分，两者不完全相同；动态线性 tail 还把后续可重叠部分相加。Phase 3.1 计划要求统一估计定义，当前代码尚不能视为满足这一点。因此线性 Static LTF → LTF 的差异同时含在线候选与评分变化；GPU 第一轮优先 FIFO，不以该对照证明“相同 LTF 仅在线化”的收益。以后应独立修正/冻结评分并重新生成对应批次，不能追改历史结果。

### 2.5 Phase 3.2：DAG 应用层推进

当前代码已拆分到 [`dag/model.py`](../../../src/runtime_comm_scheduler/dag/model.py) 和
[`dag/runner.py`](../../../src/runtime_comm_scheduler/dag/runner.py)；历史文档中的 `dag.py` 是旧位置。

- 有限手写 DAG 支持 compute/comm、分叉、join、多前驱、多通信前沿。
- 图校验包含重复/缺失依赖、环、group_seq 连续性，以及图依赖与 group 规范顺序叠加后的环。
- 每 job 一个推进循环和串行 compute worker；ready 通信交给 RankRuntime，不将整个计算图放入 coordinator。
- compute callable 正常返回视为完成；通信以 runtime handle 的完成状态解锁后继。该 compute 契约目前针对 CPU 同步 callable，直接换成异步 CUDA kernel launch 会过早完成节点。
- DAG tail 按最长后继路径递推：`tail(v)=max(duration(u)+tail(u))`，不含当前节点自身时长。静态/动态 DAG LTF 使用该摘要，比线性历史公式更明确。
- Lookahead 只声明可安全预测的直接依赖前沿；未知后继不会提前参与等待。
- 固定 group 顺序仍约束候选：DAG-ready、OFFER 到齐、全局 eligible 不是同一事件。

已验收的是 CPU/Gloo 依赖、节点全集、launch 投影、tensor、预测边界和失败路径。没有完成 GPU compute 资源模型、真实训练图捕获、一般 DAG bare 性能基线、多通信在途或跨 host 部署。

## 3. Phase 1/2：实验结果与启发

### 3.1 可采用的数据批次

以 [`benchmark/phase1.2/README.md`](../../../benchmark/phase1.2/README.md) 的最终批次及勘误为索引：

| 批次 | 矩阵 | 成功数 | 范围 |
| --- | --- | ---: | --- |
| 容量 `20260919T143653Z-9536db` | 8 workloads × 9 arms × 10 repeats | 720/720 | bare；FIFO/LTF 各 k1/k2/k3/unbounded |
| 灵敏度 `20260920T015224Z-ec2fad-poll-sensitivity` | 1 workload × 8 arms × 10 repeats | 80/80 | overlap-window-long，poll 0.1 ms |
| SRJF `20260920T021723Z-6f899d` | 8 workloads × 13 arms × 10 repeats | 1040/1040 | bare；FIFO/LTF/SRJF 各四容量 |

八个负载为 comm-heavy、staggered-compute、long-short-chains、mixed-message-sizes-large-first/last、overlap-window-short/medium/long。固定输入重复按 repetition 交错，不是 Phase 3 多扰动 seed 的相同统计设计。P10–P90 为样本分布，不是置信区间。

### 3.2 容量效应可能大于策略效应

容量批次 comm-heavy：3 jobs × 4 次 16 MiB all-reduce，每次 producer 0.5 ms。下表是**各 arm makespan 中位数**：

| bare | FIFO k1 | FIFO k2 | FIFO k3 | FIFO unbounded | LTF k1 | LTF k3 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 85.368 ms | 179.142 ms | 106.784 ms | 92.316 ms | 92.427 ms | 174.952 ms | 88.998 ms |

同 repetition 配对的 FIFO k2−k1 为 −70.719 ms，k3−k1 为 −82.386 ms，unbounded−k1 为 −88.445 ms。本负载 FIFO/LTF Plan 相同，其运行差异不能解释为策略排序收益。

启发：单在途会损失潜在并发；提高容量后 scheduler 可接近 bare。但这不是“GPU 一定应该并发”的结论，也不能拿旧路径 k3 去证明新 runtime k1 的 policy 低效。应分别报告实际并发、顺序和执行路径。

长 overlap 窗口中则不同：0.1 ms 复核下 FIFO k1/k2/k3/unbounded 中位数约为 43.847/44.278/44.150/44.229 ms；k2−k1 配对中位 +0.409 ms，P10–P90 为 −0.809 至 +0.748 ms。计算/重叠已经主导时，不能硬排容量胜负。

来源：[容量分析](../../../benchmark/phase1.2/results/capacity/20260919T143653Z-9536db/analysis.md)、[灵敏度分析](../../../benchmark/phase1.2/results/polling/20260920T015224Z-ec2fad-poll-sensitivity/analysis.md)。容量报告结尾部分曾把“独立中位数之差”误称 paired median；本文采用前面的配对表，不沿用该误标。

### 3.3 SRJF：目标函数、队首阻塞与并发利用的取舍

以下来自独立 SRJF 正式批次，均为 SRJF−FIFO 的同 repetition 配对中位差，单位 ms：

| workload / 容量 | makespan 差 | run 内平均 JCT 差 | 解释 |
| --- | ---: | ---: | --- |
| comm-heavy / k1 | −0.076 | −46.034 | 总完成时间近似不变，先完成部分 job，平均 JCT 改善 |
| comm-heavy / k3 | +65.136 | +19.082 | 静态顺序压低实际并发，连平均 JCT 也变差 |
| mixed-large-first / k1 | −0.952 | −41.948 | 小 job 先完成，平均 JCT 改善明显 |
| mixed-large-first / k3 | +22.375 | −13.729 | 平均 JCT 与总 makespan 发生取舍 |
| long-short-chains / k1 | +5.058 | −5.410 | 短 job 优先有代价，不是整体免费加速 |
| staggered-compute / k1 | +52.236 | +22.098 | 静态短剩余路径不等于在线 ready-first，可能等待未 ready 队首 |
| overlap-window-long / k1 | +27.512 | +13.762 | 相同容量不保证顺序合理或重叠充分 |

例如 comm-heavy 的 k3 场景中，FIFO 的观测 admission/launched 峰值中位为 3/3，SRJF 为 2/2。该观测支持“策略改变实际可用并发”的解释，但不是精确链路利用率证明。

对 GPU 的意义：先确定优化目标是 makespan、平均 JCT、最慢 job 还是 slowdown；不要只展示获益 job，也不要默认 LTF/SRJF 标签必然代表更优策略。SRJF 尚不是新 runtime 当前已实现的在线策略。

来源：[SRJF 分析与配对表](../../../benchmark/phase1.2/results/priority/20260920T021723Z-6f899d/analysis.md)。

## 4. Phase 3：机制与主对照结果

### 4.1 先区分功能完成与性能完成

Phase 3.1 的早期 Gloo 验收完成协议、FIFO eligible 顺序、Lookahead deadline、正常关闭及故障传播；早期四个 replay 只是功能样本，不能用其中约 0.2 s 的耗时代表当前性能。

早期控制通道存在约 40–48 ms 完成反馈空档，后续设置 TCP_NODELAY 并修正应用/校验边界后，同类单次应用耗时降到约 23 ms。该异常已修复；当前约 2 ms/项问题是后续路径残余，不能把二者混称为同一个未修复问题。

Phase 3.2 历史验收记录为全仓 116 passed/30 skipped、启用 opt-in 的真实双 rank Gloo 25 passed，包含 DAG 五策略、外部静态序列、成员偏斜和故障。F 阶段最新记录为全仓 221 passed/51 skipped、真实 Gloo declaration/failure 定向 10 passed/37 deselected。这些属于不同时点与不同测试集合，不能相加；也不代表 GPU 通过。

### 4.2 精简 suite：机制 gate 的收获

2026-09-23 compact suite 共 660 次真实 replay，全部 validation=ok，并核对记录与 raw SHA：机制 90、主矩阵 480、isolated 30、噪声 60。

| 问题 | 观测 | 当前判断 |
| --- | --- | --- |
| L1 静态队首错位 | 新 static 保留队首等待，dynamic 可先服务其他 eligible | 能研究避免静态 HOL |
| L5 成员偏斜 | OFFER spread 机制条件 10/10 | 能研究在线成员到齐影响 |
| L2/G2 优先级竞争 | FIFO/LTF 的严格研究候选竞争 gate 各 0/5 | 不扩 180 次主矩阵；不能声称 LTF 优先级收益已验收 |
| L3 准时 Lookahead | 0/5 | 不扩可选 120 次性能矩阵 |
| L4/G4 deadline fallback | 各 5/5 | 回退机制有效，不等于主动等待产生正收益 |
| G0 linear/DAG bridge | 两边各 5/5，样本/launch 桥接检查通过 | 证明特定输入桥接，不证明一般 DAG 更快 |

早期 screening 的大幅收益需要保留勘误：零 jitter seed 不代表不同计算样本；G0 静态顺序质量、线性/DAG overlap 映射不同；L2/G2 的候选竞争检查过宽。不能用旧约 1.7× 图表证明 DAG 或动态算法普遍胜出。

### 4.3 compact 主结果：完整四路径历史参照

环境：i7-14650HX、WSL2 Linux 6.18.33.2、Python 3.13.15、PyTorch 2.13.0+cu129、CPU/Gloo、双 rank、affinity 0–23、OMP/MKL/OpenBLAS/NumExpr 均 1 线程。profile p50：4 KiB 0.902 ms、1 MiB 2.552 ms、16 MiB 19.566 ms。

L0/L1 各 7 arms × 10 seeds × 3 repeats；L5 为 2 arms × 10 seeds × 3 repeats，jitter=0.3。以下为各 arm makespan 中位数，单位 ms；**该批早于 precreate 调整，仅作历史完整四路径参照**：

| 场景 | bare | old FIFO | old LTF | new Static FIFO | new Static LTF | new FIFO | new LTF |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| L0 | 16.16 | 18.51 | 19.08 | 28.63 | 28.17 | 27.44 | 27.64 |
| L1 | 25.63 | 29.81 | 30.45 | 39.11 | 40.26 | 32.29 | 32.83 |

关键配对 speedup（baseline/candidate，>1 表示 candidate 更快）：

| 对照 | L0，95% seed-block bootstrap CI | L1，95% CI |
| --- | --- | --- |
| bare → old FIFO | 0.872（0.846–0.917） | 0.811（0.783–0.863） |
| old FIFO → new Static FIFO | 0.662（0.613–0.687） | 0.750（0.725–0.799） |
| new Static FIFO → new FIFO | 1.045（0.999–1.100） | 1.224（1.198–1.283） |
| new FIFO → new LTF | 1.003（0.951–1.046） | 1.000（0.986–1.028） |

L5 new Static FIFO → new FIFO 为 1.200（1.154–1.269）。因此在线绕过静态等待有价值，但 LTF 相对 FIFO 未显示进一步稳定收益，新动态整体也未超过旧路径。

独立零 jitter A/B 噪声 pilot 的 seed-block 绝对相对差中位数 5.99%、P95 11.96%；用于保守胜/平/负标签的阈值，不是显著性检验，也不能移植到 GPU。

交错 isolated 复核中，FIFO job-0/job-1 slowdown 中位 1.366/1.095，LTF 为 1.300/1.151；只有一个 seed、五重复。此前分批 isolated 得到约 0.5 的反常分母比值已降级，不作为竞争收益结论。

来源：[Phase 3 实验结果与历史勘误](phase3experiments.md)、[结果目录索引](../../../benchmark/phase3/results/README.md)。

### 4.4 调整 tensor/binding 后：Phase D 净收益

precreate、poll 1 ms、jitter 0.3、每场景 5 seeds × 3 repeats，D 共 180 次。差值为 candidate−baseline：

| 场景 | 对照 | 配对 makespan 差 | speedup（95% CI） |
| --- | --- | ---: | --- |
| L0 | old FIFO → new FIFO | +5.150 ms | 0.767（0.707–0.786） |
| L0 | new Static FIFO → new FIFO | −1.368 ms | 1.063（0.978–1.109） |
| L1 | old FIFO → new FIFO | +0.571 ms | 0.975（0.911–0.985） |
| L1 | old LTF → new LTF | +0.534 ms | 0.977（0.929–0.989） |
| L1 | new Static FIFO → new FIFO | −5.880 ms | 1.232（1.161–1.281） |
| L1 | new Static LTF → new LTF | −6.523 ms | 1.243（1.185–1.285） |
| L5 | old FIFO → new FIFO | −0.291 ms | 1.011（0.973–1.016） |
| L5 | new Static FIFO → new FIFO | −5.500 ms | 1.192（1.168–1.234） |

这轮没有同批 bare，不能把 compact bare 拼进 D，给出“最新四组公平排名”。D 也没有证明动态收益完全超过迁移成本：L1 仍略慢于 old，L5 区间跨 1。原报告“改善超过额外路径成本”的文字应按这一数值事实收窄；本文不改写历史报告。

逐 job 也有取舍：new FIFO−old FIFO 的 job-0/job-1 配对 JCT 差，L0 为 +8.142/+4.374 ms，L1 为 +3.846/−12.438 ms，L5 为 +2.654/−13.716 ms。解救被阻塞 job 不等于所有 job 都加速。

## 5. A–F 开销探索：约 2 ms 从哪里来

完整来源为 [runtime-overhead 结果报告](phase3-runtime-overhead-20260924.md) 和
[实施记录](../process/phase3experiments.md)。A–D 共 420 次正式 replay、另有 C smoke 6 次，全部成功；E/F 是后续独立批次，不与 A–D 池化。

### 5.1 A/B：准备时机与轮询确实有影响，但都不是全部原因

| 干预 | 已观测效果 | 限制/当前决定 |
| --- | --- | --- |
| A：new LTF on-ready → precreate | L0 配对 −1.819 ms，L1 −3.013 ms；speedup 1.079/1.117 | 移出创建关键路径；准备中位仍有 5.378/6.944 ms，不是删除成本；线性默认采用 |
| B：poll 1 → 0.2 ms | L0 −1.517 ms，CI 支持本批改善；L1 −0.684 ms，CI 跨 1 | 每 rank CPU 约增 24%/17%，probes 和上下文切换增加；未改默认 |

L0 precreate 相对同批 old LTF 仍慢约 6.396 ms；L1 为 −0.412 ms、CI 跨 1。A 与 D 是不同批次，不应挑选其中更有利的 old/new 数字作为最终结论。

### 5.2 C：固定开销随串行通信数积累

零计算、单 job 通信链，old FIFO 对 new Static FIFO；每配置固定输入五重复。下表为各 arm 中位数：

| 消息 | 项数 | old FIFO | new Static FIFO | arm 中位数差 |
| --- | ---: | ---: | ---: | ---: |
| 4 KiB | 1 / 8 / 32 | 0.689 / 6.689 / 25.963 ms | 2.922 / 25.463 / 96.544 ms | 2.233 / 18.774 / 70.581 ms |
| 1 MiB | 1 / 8 / 32 | 2.413 / 16.038 / 58.708 ms | 4.527 / 32.349 / 129.505 ms | 2.114 / 16.311 / 70.797 ms |
| 16 MiB | 1 / 8 / 32 | 16.396 / 124.190 / 494.906 ms | 18.254 / 139.142 / 559.867 ms | 1.858 / 14.952 / 64.961 ms |

32 项的差约 65–71 ms，即约 2.03–2.21 ms/项。与 payload 成比例增长不符，更像每轮控制/执行/反馈的固定合并成本；但仍包含新旧完成与应用推进差异，不是纯网络开销。

### 5.3 E：policy 很短，重复等待发生在整个闭环

E2 在 diagnostic 模式下，4 KiB×32 和 1 MiB×32 的配对 new−old 中位差分别为 75.893/73.952 ms，即约 2.372/2.311 ms/项。各输入 old/new 各五重复，共 20 次通过。

| 新路径观测区间 | 4 KiB | 1 MiB | 可以说明什么 |
| --- | ---: | ---: | --- |
| 本地 submit → 收到 grant | 1.211 ms | 1.243 ms | 包含送达、成员到齐、中央处理和返回，不是纯 policy |
| 收到 grant → collective 开始 | 0.178 ms | 0.184 ms | 存在 launch worker handoff |
| collective API 返回 → 观察完成 | 1.086 ms | 2.045 ms | 包含 backend 工作和探测，不是纯 poll delay |
| 首次 completion probe 延后 | 0.696 ms | 0.722 ms | 周期探测可能错过即时完成窗口 |
| policy 函数 | 约 3 µs | 约 3 µs | 不支持它导致约 2 ms 的解释 |
| 完整 decision processing | 约 30 µs | 约 30 µs | 比 policy 函数大，但仍不是主要毫秒量级来源 |

E2 trace 复核进一步显示，容量释放时最后 OFFER 尚未进入中央队列的转移为 141/155、137/155；两输入均没有候选在释放之前已经 eligible。容量释放→首次 eligible 约 415/420 µs，而容量与 eligible 都满足后→decision start 约 3/4 µs。

这说明该零计算链经常是“下一请求还没到齐”，不是“合法工作和空闲容量都已有，policy 却长时间不调度”。最后 OFFER 的到达也不能称为单向网络时延，因为它受远端应用推进、DECLARE、发送锁和接收线程影响。

补测 DECLARE 中位为 rank 0/1 的 319/170 µs，发送锁等待约 69/60 µs，常与上项 COMPLETED 持锁发送重叠。收到 grant→worker dequeue 约 176–185 µs，dequeue→调用约 11–12 µs。慢 OFFER 不总来自同一个 rank。

可以用下面的依赖链理解积累，但**箭头不表示各段中位数可直接相加**：

```text
物理完成 → probe 观察 → 唤醒应用 ─→ 下一项 DECLARE/producer/submit/OFFER ─┐
                     └→ COMPLETED → 中央释放容量 ──────────────────────┤
                                             两个条件均满足 → 决策 → GRANT
                                                                  → 本地发射 → 下一次完成
```

成员反馈与应用推进有并行和锁竞争；新旧数据区间包含共同的 backend 成本；不同 rank/中央时钟也不能直接相减。准确结论是**闭环系统成本**，不是已经找到一个独占 2 ms 的函数。

E3 只测试提交后唤醒 completion thread：首次 probe 提前约 0.52 ms，却常在 backend 尚未完成时多做一次 probe。4 KiB 五对全部更慢，配对中位 +4.851 ms；1 MiB 方向混杂、3/5 较快，不能靠两 arm 中位数相减决定配对收益。未启用默认，按 gate 不跑 E4。此结果否定该具体候选的稳定收益，不是否定所有 completion 优化。

### 5.4 观测扰动本身不可忽视

E1 最初两版诊断使通信链额外增加约 50 ms；缩减后 minimal/diagnostic 中位仍为 88.545/99.764 ms，差 11.219 ms/32 项。后续 DECLARE 补测的 diagnostic−minimal 配对中位为 +8.919 ms。

所以性能结论优先 minimal；diagnostic 解释机制；设备 profiler 再单列。不能把日志成本当固定常数从曲线中扣掉，也不能跨观测级别拼接耗时账本。

### 5.5 F：删除独立 DECLARE 有边际收益，但不是应用问题的总解

只比较默认 before-producer 与直接 on-submit，后者 OFFER 仍含完整 spec/hint，不改容量或完成协议。F1/F2/F3/F4 共 90 次通过；F4 60 次、30 个配对块计算样本完全一致。

| 项目 | 结果 |
| --- | --- |
| F1，4 KiB×32 minimal | 五对全更快；on-submit 配对中位 −13.970 ms |
| F3，1 MiB×32 minimal | 五对全更快；配对中位 −12.871 ms |
| F2 diagnostic | wait-return→next submit 缩短约 540/286 µs；中央等最后 OFFER 缩短约 550 µs |
| F2 竞争转移 | rank 0 OFFER 发送锁等待增加约 147 µs，5/5 |
| F4 L0 | seed 汇总差中位 −0.354 ms；范围 −0.963 至 +2.223，3/5 seed 更快 |
| F4 L1 | seed 汇总差中位 +0.406 ms；范围 −0.316 至 +1.461，1/5 seed 更快 |
| F4 资源 | 两场景 process CPU 配对中位均约 −2.4 ms；voluntary switches 减少 45/47 |

链上约 0.40–0.44 ms/项的改善不意味着把原来的 2 ms 全部消除；F4 的 L0/L1 只有四项通信，存在 producer/consumer 窗口、跨 job 重叠和扰动，节省不一定位于最终关键路径。相同计算样本也不等于相同绝对 ready 时刻。不能单凭 OFFER 锁增加就把 F4 没收益唯一归因于该锁。

**F4 统计口径澄清：**当前脚本实际计算的是 `median_seed(median_repeat(candidate−baseline))`，不是 `median_seed(median_repeat(candidate)−median_repeat(baseline))`。原过程/结果中“先各组取中位数再配对”的表述不精确；CSV 相邻的 arm median 列相减并不等于 delta 列。例如 L1 seed 5301 的两个 arm 中位数之差为 −1.590 ms，但 repeat 配对差中位为 +0.406 ms。本文沿用实际配对估计量，不切换口径挑有利结论。

结论：保留 before-producer 默认，不继续扩 DECLARE 策略矩阵。既有证据支持减少控制消息能降低 CPU 成本，不支持 L0/L1 的稳定应用加速。

### 5.6 还有哪些值得怀疑

| 因素 | 目前证据 | 后续如何区分 |
| --- | --- | --- |
| 全成员完成释放 vs rank-local 完成释放 | 两系统语义确实不同 | 保留历史整系统对照；另做明确命名的完成契约对齐对照 |
| 应用 wait 契约 | old 等 backend Work；new 等完成探测唤醒 | GPU 分 host wait 和 stream dependency 两条 lane |
| Python/GIL/线程唤醒、发送锁、事件队列 | 有等待、CPU/context-switch 证据，无精确因果拆分 | 小规模相同输入/相同观测单变量消融，不先重写线程架构 |
| 轮询与实际通信时长比例 | B 有部分收益，E3 具体候选无稳定收益 | GPU 重新 profile 和做一次 poll 敏感性，而非复制 CPU 最佳值 |
| 静态与动态 LTF 估计定义 | 当前线性公式不一致 | 优先 FIFO；评分独立修正和重测 |
| 预热、tensor 初始化、binding 生命周期 | A 已证明会进入应用关键路径 | GPU 需证明初始化完成或建立事件依赖；记录内存成本 |
| 图表统计边界 | 蓝条是观察完成，代表图不是配对样本 | 同 seed/repeat 双 rank 本地时间线，另报统计分布 |
| DAG compute/comm 双层观测 | CPU runner 有独立完成收集和轮询 | GPU DAG 单独测，不能把全部延迟归到中央 scheduler |

## 6. 已有结果能回答和不能回答的问题

能够回答：CPU/Gloo 的协议和 DAG 机制可运行；静态 HOL 是真实问题；动态 FIFO 可利用在线 ready 状态；新 runtime 的逐项闭环有显著成本；tensor 准备和 DECLARE 是其中可测部分；policy 函数不是毫秒级瓶颈。

尚不能回答：GPU 上差距是否仍约 2 ms；NCCL 的真实设备完成是否被正确识别；真实 GPU compute overlap 下的净收益；Lookahead 的稳定正收益；LTF 在严格共同候选上的优先级收益；多 communicator 并发的安全性和收益；新 runtime 多在途、多资源、多机、真实训练性能。

对“2 ms 是否重要”，应看关键路径上的通信次数与可隐藏程度，而不是仅看单项：零计算串行链会近似积累；长计算窗口可能隐藏；大量小 collective 的训练可能非常敏感。这里只是机制推断，不能用 `总 collective 数 × 2 ms` 直接预测真实训练退化。

## 7. GPU 准备：必须先处理的实现与测量差距

### 7.1 为什么现在值得切到 GPU

CPU 排查已能排除“大部分时间花在 policy 选择”这种解释，也完成两项具体候选的闭环验证。继续扩大同一 sleep/Gloo 微基准的策略矩阵，不能回答真实目标平台上的主要问题。

GPU 可能出现相反结果：通信更快使固定 host 开销占比更高；真实计算重叠可能隐藏开销；SM/显存带宽/PCIe/NVLink 竞争又可能改变串行化的价值。因此下一阶段应测**适用边界**，不是预设 GPU 会让 scheduler 获胜。

### 7.2 代码级优先项

| 优先级 | 现状证据 | GPU 准备要求 |
| --- | --- | --- |
| P0 | new worker 无论 backend 均构造 `DirectExecutor()`；其 launch 只是调用 closure | 新核心需要独立的 CUDA device/stream 执行契约；不引入旧 Plan 依赖 |
| P0 | `LocalBinding` 有 producer_event，但 runtime 在 executor 不声明支持时明确拒绝 | 明确 producer 依赖如何跨线程传到 collective；不能只填 event 字段 |
| P0 | `RuntimeHandle.wait_on()` 当前把 backend Work 传给 `stream.wait_stream()`，fallback 也未显式进入传入 stream 上下文 | 不能将其视为合格 CUDA consumer bridge；在新 API 上实现/验证真正的目标 stream 依赖 |
| P0 | completion probe 直接调用 `Work.is_completed()`，并声明支持物理完成 | 对目标 PyTorch/NCCL 版本独立验收；类属性或 mock 通过不是证明 |
| P0 | old `ScheduledWork.wait()` 保留 backend CUDA 等待契约，new `wait_host()` 等 host 可见完成 | 应用终点先统一为正确消费完成，不能把旧 host 提交结束与新物理结束直接比较 |
| P0 | 当前 bare 是多 job 线程直接使用多个 group | 先审查多 communicator 顺序要求；无合格 bare 时标未验收，不强跑、不伪称公平四组 |
| P1 | 线性 precreate 用 `torch.full`，初始化在 GPU 上可能异步 | 初始化必须在 release 前完成，或有明确依赖；准备与内存成本单列 |
| P1 | `run_phase3._start()` 按 rank 覆盖 `CUDA_VISIBLE_DEVICES`，子进程 LOCAL_RANK 固定为 0 | 检查是否符合目标机器/集群分配的可见设备集合；不能默认物理 GPU 0/1 就是获准使用的两卡 |
| P1 | DAG compute future 返回就解锁后继 | 真实 GPU compute 要用实际完成事件或已定义的设备依赖，不能把 kernel enqueue 当完成 |
| P1 | CPU 诊断入口硬编码 Gloo/world size 2 | `run_control_path_diagnostic.py` 不能直接当 GPU 批处理入口；先做明确的 runner 适配 |
| P1 | 线性 LTF 评分不一致 | 在进入 LTF 性能矩阵前统一契约、测试及配置摘要 |

旧 [`tests/gpu/test_cuda_semantics.py`](../../../tests/gpu/test_cuda_semantics.py) 可借鉴测试思想，但其中使用旧 scheduler，fixture 为 world size 1；通过也不能替代新 runtime 双 rank NCCL 验收。

官方文档强调 CUDA collective 的等待/stream 语义与 CPU 不同，跨 stream 使用结果需要显式依赖；实际物理完成探测还应核对目标版本实现和设备事件证据。参见 [PyTorch collective 同步语义](https://docs.pytorch.org/docs/2.14/distributed.html#synchronous-and-asynchronous-collective-operations)。本次查询页面是 2.14 文档，并非此前 CPU 环境 2.13 的版本证明。

多个 NCCL ProcessGroup 还要求跨 rank 一致的执行顺序及适当同步；具体机制受 NCCL/CUDA 版本与配置影响，不能仅凭 group 内顺序一致或存在单 host worker 就宣称安全。参见 [PyTorch groups 警告](https://docs.pytorch.org/docs/2.14/distributed.html#groups)、[NCCL 多 communicator 说明](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/communicators.html#using-multiple-nccl-communicators-concurrently)。这些是验收依据，不是要求升级依赖或开启某个环境变量。

### 7.3 GPU 必须分开两种应用契约

**H lane：host 完成链诊断。** 应用每项通信消费点要求本地物理完成，下一项依赖任务才推进。四路径若参与该 lane，应显式对齐该应用契约。旧路径若增加 host completion 等待，应新命名为 `old-static-host-complete`，不能覆盖历史 old 行为。该 lane 测闭环成本，不代表异步训练最优实现。

**S lane：stream 依赖与真实 overlap。** producer/consumer 都有实际 GPU 工作，消费通过目标 stream 依赖保证正确，不要求每项阻塞 host；应用终点是末端依赖计算完成。新 runtime 仍独立观测真实完成并按全成员回执释放容量。host 提交结束、设备终点、drain 分别记录。

两 lane 不混算。先通过 H lane 和 stream 正确性，再做 S lane 性能；不能给 old 用较弱 host 终点、给 new 用物理完成终点后报告性能落后。

预创建仅针对 buffer/绑定结构；真实 producer 产生的数据和依赖不能被提前“造好”绕过。固定 buffer 大小/数量、保留引用直至安全释放，记录峰值显存。

## 8. 建议的 GPU 实验顺序与门槛（未执行）

第一轮限定单机两块 GPU、每 rank 一 GPU、真实 all-reduce、新 runtime 全局单在途。沿用独立 TCP 控制面，不增加多机、框架接入、在线 SRJF、多资源或新 runtime k>1。

### G0：环境盘点与正确性验收

记录 GPU 型号/UUID、驱动、PyTorch build、CUDA runtime、实际 NCCL 版本、两卡拓扑、NUMA/CPU affinity、线程设置、显存余量、后台负载、功耗/频率状态、相关 NCCL 环境变量。保留目标环境，不擅自安装或升级组件。

先按单 group，再按两个 group 严格串行的顺序验证：

1. rank-device 映射、group 创建顺序、collective 数值、实际 launch 投影。
2. producer 在非默认 stream 写入可区分数据；未完成依赖不能被 collective 越过。
3. consumer 在另一个非默认 stream 读取结果，必须依赖通信；不能靠最后 validation 的隐式同步“补正确”。
4. 把 host wait 返回、stream 依赖建立、completion query、设备末端事件分开观测，验证不会提前发 COMPLETED。
5. 人为推迟应用消费，确认通信完成仍独立上报；人为控制一个成员的提交/反馈，确认中央未收齐完成不能放下一 grant。
6. 两 group 顺序切换、tensor 生命周期、初始化与释放；不存在借默认 stream 偶然串行才通过的测试。
7. 元数据冲突、缺失成员、launch/probe 异常、输入关闭和超时；后续任务停发、等待者唤醒、子进程有界回收。不用硬件破坏性故障注入。

对物理完成的独立证据，可以在已正确接上 NCCL 完成依赖的 stream 记录尾事件；事件若没有依赖 NCCL，不能用它当证明。设备 trace 只在少量诊断运行采集。不要用每项 `cuda.synchronize()` 作为隐藏修补来完成 S lane。

**Gate：任一 producer/consumer/完成/顺序条件不成立，停在实现修复，不产出策略性能结论。** GPU 不足两块或 backend 不可用时，记录限制；不以单 GPU 或 Gloo 代替双 GPU 验收。

### G1：目标平台重新校准与观测扰动

先覆盖 4 KiB、1 MiB、16 MiB，float32 sum、相同成员。每签名建议 warmup 5、测量 30；显存及耗时允许且 16 MiB 仍过短时，可额外探测 64 MiB，但不默认扩主矩阵。

- 记录 host API、调用至确认完成、合法设备事件区间；profile 使用独立无竞争测量，不能复用 Gloo 数值。
- 所有策略共享冻结 profile；不能根据一次实际 future jitter 回填 policy hint。
- 选择小消息链做 minimal/diagnostic 五个交错配对，共 10 次；设备 profiler 另做少量运行，不计入性能样本。
- 在目标 GPU 重测噪声，不能使用 CPU 的 11.96% 阈值。

这里 profile 的 30 个测量样本不等于 30 个 workload replay。每阶段分别报告启动开销、采样时间和总墙钟，避免矩阵规模误判。

### G2：固定通信链——GPU 上还剩多少逐项开销

主问题：在完成契约对齐后，old static 与 new static 的差是否仍近似随通信项数增长？

第一块保持最小：

| 维度 | 设置 |
| --- | --- |
| 输入 | 单 job、单 group、零计算、32 项串行链 |
| 消息 | 4 KiB、1 MiB、16 MiB |
| arms | old static FIFO（H lane）与 new Static FIFO（H lane） |
| 重复 | 固定输入 5 个随机顺序配对；3 × 2 × 5 = 30 次 |
| 固定项 | precreate、同 warmup、最小观测、初始 poll 1 ms；DECLARE/wakeup 保持已记录默认 |

先报告每个 pair 的应用完成差、链总时间/项数、CPU、probe 次数与末端完成边界。只有三个大小才不能拟合消息大小的通用规律；固定输入五重复只作探索。

若差距清晰且需要确认是否按项数累积，**只选一个代表大小**补 N=1/8 的两 arm 五配对，增加 20 次，而不是重跑所有尺寸×长度。用 `ΔT(N)≈a+bN` 作描述，明确小样本不构成通用常数模型。

若观察到量化在 poll 周期附近的空档，优先做一个代表配置的 1 ms/0.2 ms、五配对（10 次）敏感性，保持其他变量不变。仍需同时报告 CPU，不能先开 DECLARE、wakeup、低 poll 三个优化再解释收益。

判断方向：

- 设备通信更短但 host gap 仍大：重点是控制闭环/完成观测，不是换更复杂 policy。
- 设备通信本身变长：查 launch/stream 依赖和干扰；不能把 host 图蓝条直接当设备通信。
- 较大消息只降低开销占比、绝对 gap 不变：说明成本摊薄，不表示 runtime 固定成本已消失。

### G3：最小多 job 净收益矩阵

首先 L0 负对照 + L1 队首错位；FIFO 为主，避免同时引入 LTF 评分问题。

| arm | 作用 | 前提 |
| --- | --- | --- |
| old static FIFO | 历史系统基线/完成契约对齐基线，名称分开 | 对应 lane 的消费语义已验收 |
| new Static FIFO | 固定顺序的同 runtime 对照 | 不允许跳头 |
| new FIFO | 在线选择 | 与 new static 相同执行与容量 |
| raw/bare 参考 | 检查系统相对低调度开销路径的净效果 | 多 communicator 顺序与完成契约独立合格 |

如果原始 bare 不能在目标 NCCL 配置下安全复现，应如实记为“不支持该并发参考”；可另建受控共同发射顺序的 raw 参考，但必须明确它已经不同于 CPU 原始裸发，不能用新名字掩盖额外协调/排序成本。

建议两层规模：

1. 机制 pilot：每 workload 每核心 arm 3 次，检查静态 HOL 与 dynamic 绕过，而非检验胜负；失败场景修改输入后重新编号，不事后筛选成功触发的运行。
2. 性能探索：冻结输入及 profile，5 execution seeds × 3 repeats。L0/L1 × 3 核心 arms 为 90 次；如 bare 已验收，则四 arm 为 120 次。先完成这两场景，不一次启动 L0–L5 全矩阵。

输入应按 GPU profile 校准 ready 错位窗口，pilot 可调、正式批次必须冻结；保留绝对时长和“错位/通信时长、控制成本/通信时长”比例。不能原封不动照搬 CPU 的等待预算或靠无限加大偏斜制造胜出。

主对照为 new Static FIFO→new FIFO（机制价值）与 old/raw→new FIFO（系统净效果）。若机制不触发，先修场景；若机制触发但净收益不稳定，报告边界，不立即扩 LTF/Lookahead。L5 仅在需要确认成员偏斜是否改变结论时追加一场景。

### G4：真实 GPU compute overlap 与 DAG

这是比扩大 CPU 风格 sleep 矩阵更重要的第二步，但需要新的 compute 完成契约，不能只替换 sleep。

- 先做线性 producer/通信/独立计算/依赖消费；使用预分配 buffer 和固定 GPU 算子序列。校准算子规模，不在正式实验里以 wall-clock 忙循环硬追时长。
- 至少分短 overlap 和长 overlap 两档；算子/工作量样本固定，实际 GPU duration 因争用发生变化应保留，不强行固定绝对 ready 时间。
- 同时测 application device-complete makespan、逐 job JCT、计算实际耗时、通信相关设备区间、host 控制间隔。计算变慢和通信变慢要分别可见。
- 先取一个代表消息大小、两个 overlap 档、old static/new static/new FIFO，采用 3 seeds × 3 repeats，共 54 次探索；无明确问题不要追加全尺寸交叉积。
- 再做 linear→DAG bridge 和一个 diamond；确认 consumer overlap 没被错误串行成通信完成后的计算、join 等所有真实前驱完成、group 顺序仍成立。
- DAG 性能先在新 runtime 内比较 static/FIFO；一般 DAG 没有合格 bare/旧 Plan 对应物时，不伪造跨阶段对照。

同 GPU 上多个 job 并没有各自独占算力；compute 与 NCCL 可能争用设备资源。本阶段要实际测这种影响，而不能继续把 CPU sleep 的独立窗口视为真实算力重叠。

### G5：后置研究方向与停止条件

| 方向 | 进入条件 | 首先要回答的问题 |
| --- | --- | --- |
| LTF | 评分定义统一；指定两个候选同时 eligible 的 gate 稳定 | 相同执行路径下，改变选择是否改善关键路径/JCT |
| Lookahead | 预测窗口按 GPU 时标校准；准时/迟到可稳定区分 | 避免未来阻塞的收益是否超过主动空等和预测误差 |
| DECLARE/完成唤醒等优化 | G2/G3 明确指出对应路径是主限制 | 单变量是否既改变机制区间又改善 minimal 应用结果 |
| 新 runtime k>1 | 单在途基线完成；另行定义并验收设备顺序/容量协议 | 串行化代价是否大到值得牺牲在线反应空间 |
| 真实训练/trace 接入 | S lane、DAG 数据依赖及端到端指标可信 | 能否把合成场景收益迁移到真实通信序列 |

这些不是当前一轮自动执行项。尤其不能因为 Phase 2 k3 较快，就在 Phase 3 中直接放开 inflight；那是独立设计/验收范围。

## 9. GPU 统计、产物与验收标准

### 9.1 统一统计合同

1. 主指标取每 rank 本地 release→约定应用终点的 duration，再取参与 rank 最大值。它是 max rank-local duration，不冒充跨 GPU 全局同步时钟跨度。S lane 应用终点包含末端依赖计算完成。
2. 每个 job 同样用明确的本地 release/结束边界汇总；报告每 job、run 内平均 JCT、最慢 job。slowdown 的 isolated 分母必须同环境、同契约、交错测量。
3. 同 seed/repeat/input 配对，记录并随机化 arm 顺序，replay 串行运行。5 seeds×3 repeats 不等于 15 个独立 seed；零 jitter 多 seed 不自动形成不同计算样本。
4. 建议主要估计量明确冻结为 `median_seed(median_repeat(candidate−baseline))`；配对 ratio 另列。独立 arm 中位数、arm 中位数之差、配对差中位数使用不同字段名。
5. bootstrap 按 seed block 重采样，保留块内所有 repeats；报告方法、次数、随机种子和有限 seed 限制。五 seed 是探索性证据，不能仅靠区间宣布平台通用结论。
6. 若需正式扩测，优先增加预先约定的新 seeds，例如扩至 10 seeds×3 repeats，而非同 seed 无止境重复；先冻结扩测条件，保留全部结果，不“跑到显著为止”。
7. paired 差、绝对 ms、百分比、全部原始点同时报告；P10–P90 不是 CI。收益区间跨零/一时明确“不确定”，不只数胜出的 repeat。
8. 当前只做单环境结论；更换机器、PyTorch/NCCL、拓扑、poll 或观测模式后另建批次，不池化。

### 9.2 时间线和时钟

- host 本地时间、coordinator 时间、各 GPU event elapsed 分开。不能从任意 rank send 时间减另一个 rank receive 时间计算网络延迟。
- collective API start→completion observation 仍包含观测等待，不标“纯通信”。CUDA event 必须接在正确依赖链上，也可能包含排队；精确 kernel 区间仅在诊断 profiler 中解释。
- 同 seed/repeat 选配对运行，rank 0/1 分面；可以用 task ID/消息箭头表达因果，但不得把未经同步的横轴当同时钟。
- makespan 最接近中位数的各 arm 代表图只用于可视化，不用于直接成对归因；写明源文件、seed/repeat 和选择规则。

### 9.3 产物组织

沿用现有目录边界；下面是建议新目录，本文未创建或运行：

```text
benchmark/phase3/
  experiments/gpu-readiness/
    semantics/       # G0 契约与正确性输入
    calibration/     # G1 签名与测量配置
    communication-chain/
    readiness/       # L0/L1，按需 L5
    compute-overlap/
    dag-bridge/
  results/gpu-readiness/
    <相同语义分类>/<batch-id>/
      inputs/ raw/ tables/ figures/
      manifest.json runs.jsonl analysis.md
```

执行脚本仍在 `examples/jobpacer/scripts/`；分析在 `examples/jobpacer/analysis/`；本地绑定/harness 在 `examples/jobpacer/runtime/`；新 GPU 执行/完成语义属于新 runtime 边界，不回连旧核心。

manifest 除输入、命令、退出码、哈希外，还需包括源码快照/patch、dirty 状态、lane、实际 executor/probe、消费契约、初始化与预热边界、显存、GPU 拓扑/版本、poll、观测级别和 workload/profile digest。只有 SHA 而没有可恢复源码不够长期复现。

结果目录沿用仓库忽略规则，raw 是否纳入 Git 与是否已持久保存是两件事。不能因为报告链接存在就假设其他 checkout 含 raw；归档时同时打包 manifest 指向的输入/原始数据/源码快照，不覆盖旧批次。

### 9.4 最小交付与停止标准

- G0：所有必需正常/异常语义项通过；明确 skipped、未支持的 bare/stream/backend 边界。
- G1：目标设备 profile 与观测扰动报告完备。
- G2：30 次最小链实验有完整配对、正确性和成本量级结论；追加项只按预定问题触发。
- G3：至少 L0/L1 核心三 arm，同输入样本、明确完成契约、逐 job JCT 和净收益结论。结果为“没有净收益”也算有效交付。
- G4 以后只有在前述数据/语义可信且资源允许时进入，不为凑全矩阵而运行。

任何 tensor 错误、grant/launch 不一致、提前完成、无界挂起都阻止性能验收；不能删去失败样本后继续给出只含成功运行的结论。环境失败保留原记录，补跑独立标记原因和归属。

本轮准备文档的检查范围：复核 compact L0/L1 CSV 的 arm 中位数、F4 汇总代码和 seed 表、文内本地链接及 Markdown 空白；没有重新逐文件核验全部历史 raw 的哈希。历史批次的全量验证情况引用原结果报告，GPU 环境盘点和 G0–G5 均尚未执行。

## 10. 推荐下一步的具体落点

第一优先不是再消掉几十微秒的 policy/DECLARE，而是完成**新 runtime 的 CUDA producer/consumer/physical-completion 契约审查与双 rank 验收**，同时厘清旧路径与 bare 的 GPU 等待、顺序和计时契约。

通过后，按 **目标 GPU profile → 30 次单 group 通信链 → L0/L1 三或四路径小矩阵 → 真实 compute overlap → DAG** 的顺序推进。这样可以分别回答：

1. GPU 路径是否正确；
2. 当前固定 host 闭环成本在目标平台有多大；
3. 避免静态 HOL 的收益是否足以抵消成本；
4. 真实计算重叠和设备争用是否改变结论。

若 new dynamic 只优于 new static、仍落后合格 old/raw，应如实总结为“在线机制有效，但当前系统成本或串行限制抵消净收益”；若只有特定偏斜/消息/overlap 区间有效，就报告该边界，而不是继续扩矩阵寻找一个普遍获胜的叙述。

## 附：主要证据入口

- [架构讨论](../plan/discussion.md)、[Phase 3.1 计划](../plan/phase3.1.md)、[Phase 3.2 计划](../plan/phase3.2.md)。
- [Phase 2 初始实现/校准验收](phase2.md)、[Phase 1/2 最终批次与勘误](../../../benchmark/phase1.2/README.md)。
- [Phase 3.1 CPU 验收](phase3.1.md)、[Phase 3.2 CPU/DAG 验收](phase3.2.md)。
- [Phase 3 pilot/screening/compact 及勘误](phase3experiments.md)、[A–F 开销结果](phase3-runtime-overhead-20260924.md)。
- [E2 分段复核及 DECLARE 锁等待](../../../benchmark/phase3/results/runtime-overhead/instrumentation/20260925-E1-declare-send-lock/followup-analysis.md)。
- [F4 统计与原始批次](../../../benchmark/phase3/results/runtime-overhead/control-path/20260925-F-declaration-mode/application-L0-L1/)、[F4 汇总代码](../../../examples/jobpacer/scripts/run_control_path_diagnostic.py)。

以上本地 raw/派生统计链接依赖当前工作区保留的忽略目录；历史测试数和耗时以其原批次为准。本文新增结论中的代码差距属于 2026-09-26 阅读现状所得，未声称已经修复或 GPU 实测复现。
