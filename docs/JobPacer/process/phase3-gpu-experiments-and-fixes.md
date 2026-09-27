# Phase 3 GPU 实验与现有问题修复：详细实施方案

日期：2026-09-26。状态：**待实施，本文不是修复完成或 GPU 验收记录。**

依据：[Phase 1–3 总结与 GPU 准备](../result/phase1-3-summary-and-gpu-readiness-20260926.md)、[设计讨论](../plan/discussion.md)、[Phase 3.1 协议与验收要求](../plan/phase3.1.md)、[Phase 3.2 DAG 计划](../plan/phase3.2.md)、[CPU A–F 结果](../result/phase3-runtime-overhead-20260924.md)。按本次要求将修复和实验的执行细节集中放在 `process/`；不改写历史结果。

本文中的新类、CLI 参数、测试文件、输入和阶段命令均为**拟实现接口**，除明确标注的现有命令外，不表示已存在。后续实施必须更新完成清单并记录实际接口、命令、环境和结果路径。

## 1. 目标、范围与执行顺序

本轮回答四个问题：

1. 新 runtime 是否正确处理 CUDA producer、NCCL collective、consumer stream 和物理完成？
2. 完成契约对齐后，GPU 上是否仍存在随 collective 数积累的固定闭环开销？
3. L1 静态队首错位中，动态调度的机制收益是否足以抵消新增成本？
4. 真实 GPU compute overlap 是否改变 host sleep replay 的结论？

范围：单机、两 GPU、每 rank 一个进程/一 GPU、float32 sum all-reduce、新 runtime 全局 `max_inflight=1`、独立 TCP 控制面。保持现有 Gloo 路径和历史基线可运行。

不做：新 runtime 多在途、多机、多资源、跨进程多 job GPU 共享、在线 SRJF、框架插件、自动 trace 捕获或新通用执行框架。GPU 缺失时只做 CPU 单测与 Gloo 回归，明确 GPU 未验收，不以单 GPU smoke 代替双 GPU 结论。

```text
R0 版本/环境/实验合同
  → R1–R5 GPU 执行、等待、完成、计时、启动与基线修复
  → C0 CPU/Gloo 回归 → G0 双 GPU 语义与故障验收
  → G1 profile、观测扰动与噪声
  → G2 单 group 固定通信链
  → G3 L0/L1 多 job 净收益
  → R8 + G4 真实 GPU compute overlap
  → R9 + G5 DAG bridge/diamond

R6 LTF 评分修复、R7 统计修复可独立开发；R7 在任何性能批次前通过，
R6 在 LTF/Lookahead 性能测试前通过，不阻塞 FIFO 的 G2/G3。
```

每个 gate 允许结论为“无性能收益”；只有正确性、输入或测量问题才阻止解释性能。不能为了让策略获胜不断调整正式输入。

## 2. R0：冻结基础合同和可恢复版本

### 2.1 实施前记录

- 保存当前 HEAD、dirty 状态、相关源文件 SHA，以及可恢复的源码快照或 patch。已有未提交修改属于现有工作，不覆盖、不自动提交。
- 保存 CPU A–F 的配置与结果索引；不重建或覆盖历史 raw/manifest。
- 检查 `pyproject.toml` 的 Python 下限；新增代码保持兼容。
- 在目标 GPU 主机只读盘点：可见设备、UUID、拓扑、显存、驱动、PyTorch build、CUDA runtime、NCCL 实际版本、CPU/NUMA、affinity、线程环境、后台负载及相关环境变量。
- 不安装/升级依赖，不自行调整共享机器的功耗、频率或其他进程。

输出：环境 JSON、源码快照、`contracts.json`；未知字段写 unknown，不能套用 CPU 主机的参数。

### 2.2 两条完成契约

| 契约 | 应用推进 | 消费安全 | 应用结束 |
| --- | --- | --- | --- |
| H：host-complete | 在每项通信的消费点等待本地物理完成 | 之后才能进行依赖消费 | 全部 job 的末端工作完成；不含结果扫描 |
| S：stream-dependent | host 可继续安排独立工作，通信消费通过设备依赖连接 | 目标 consumer stream 等待已绑定通信的完成事件 | 所有 job 的末端设备事件已完成；不能使用 host enqueue 结束 |

H 用于 G2/G3 与 CPU 机制接续，S 用于 G4 真实 overlap。两条 lane 不池化、不同图例，不用旧 `Work.wait()` 的 host 返回对比新 `wait_host()` 的物理完成。

H 中旧基线命名 `old-static-fifo-host-complete`，显式增加应用侧物理完成等待；旧 scheduler 本地容量语义不改成中央全成员容量。历史行为另标 `old-static-fifo-legacy-wait`，必要时单独诊断，不混入 H 主表。

### 2.3 OFFER 的 ready 含义

GPU event 已记录不表示 producer 已完成。显式记录：

- `physical-ready`：应用侧观察 producer event 完成后 OFFER。
- `dependency-enqueued`：producer event 已记录且依赖可以传递，允许 OFFER，executor 在设备侧等待 producer。

**本轮 H 主实验及 S 首轮均固定 physical-ready**，便于比较实际 ready 调度；仍必须测试 executor 对未完成 producer event 的正确桥接。将来研究提前 OFFER 时单列 dependency-enqueued 消融，因为它可能提前占用全局容量、改变 HOL 与 Lookahead，不是透明性能优化。

physical-ready 的 producer query 等待计入应用推进并记录；采用每 job 有界事件查询/现有推进循环，不创建每个 task 一个永久线程。不得在 coordinator 或控制接收线程阻塞等待 GPU。

## 3. 修复清单和文件边界

| 编号 | 优先级 | 当前问题 | 主要拟修改位置 | 完成条件 |
| --- | --- | --- | --- | --- |
| R1 | P0 | new worker 总用 DirectExecutor，producer event 不受支持 | `runtime/executor.py`、`runtime/model.py`、`runtime/runtime.py`、新 worker | 新 CUDA executor 能传递依赖；CPU direct 不退化 |
| R2 | P0 | `wait_on()` 把 Work 当 Stream，且 fallback 未进入目标 stream | `runtime/handle.py`、新 CUDA execution receipt | 正确目标 stream 依赖，有失败/超时边界 |
| R3 | P0 | 通用 Work query 未经目标 NCCL 物理完成验收 | 新 CUDA receipt/probe、`runtime/runtime.py` | 设备完成证据先于 COMPLETED；SUBMITTED 顺序保持 |
| R4 | P0 | 初始化、应用终点、旧新 wait 含义不一致 | 两 worker、measurement、runtime_results、visualize_phase3 | H/S 终点明确，校验不进入应用关键路径 |
| R5 | P0 | launcher 覆盖可见 GPU；多 group bare 不保证安全 | phase2/phase3/profile 启动、共享启动辅助模块、基线 harness | 使用获准设备映射；不启动未合格 bare |
| R6 | P1 | 线性 static/dynamic LTF 评分不一致 | examples adapter 与静态序列构造、单测 | 新 static/dynamic 共用有版本的 tail；历史 old 不改写 |
| R7 | P0（统计） | arm median difference 与 paired median 命名混淆 | analysis、诊断/批处理脚本、报告生成 | 估计量、失败样本、block 数可审计 |
| R8 | P1（S 前） | 当前 compute 是 host sleep | examples GPU compute binding、两 worker | 固定 GPU 工作量、正确消费、终点和采样 |
| R9 | P1（DAG 前） | compute callable 返回被当作设备完成 | DAG runner/adapter、GPU compute result | 未完成 kernel 不解锁完成依赖 |

这里的 `runtime/` 指 `src/runtime_comm_scheduler/runtime/`。核心不得依赖 examples 或旧 Plan/scheduler；共享 workload/统计/启动辅助代码放 examples。避免为本次修复建设插件系统。

### 3.1 R1：新 CUDA executor 和执行回执

推荐在新核心中增加具体的 `CudaCollectiveExecutor` 和本地 `CudaCollectiveWork`（名字实施时可调整）。后者持有 backend Work、明确 device、通信完成事件和生命周期引用，向 runtime 暴露已有 Work-like 最小能力及显式 stream dependency 能力；不让 coordinator 知道 CUDA 对象。

发射路径：

1. executor 构造时绑定确切 device index，不能只保存字符串 `cuda`。核对 tensor、producer event、consumer stream 的设备归属；跨设备误用提前拒绝。
2. 由唯一 launch worker 调用 executor；显式进入对应 `torch.cuda.device` 和本地 gate stream 上下文。
3. gate stream 先 `wait_event(producer_event)`，再调用真实异步 all-reduce；没有 producer event 时要求初始化依赖已由 harness 满足。
4. 在目标版本验证过的语义下，将 backend Work 完成依赖接回 gate/专用完成 stream，再记录独立 `comm_done` event。不能在没有接 NCCL 依赖的默认 stream 上随手 record event。
5. 返回持有 Work 和 event 的回执；接入既有 bind→SUBMITTED→active 顺序，不向 coordinator 多发一种“假完成”消息。

初版按 group 保留 gate stream，避免以后错误继承其他 group 的 producer 依赖；但全局容量仍为 1，不据此声称支持 communicator 并发。可用已注册 ProcessGroup 本地身份作键，无需把 ProcessGroup 序列化进协议。

关于步骤 4：候选实现是在 gate stream 上调用 backend 的非 host-blocking 完成依赖接口，再 record event；例如目标版本已验证的 `Work.wait()` stream 语义。若配置开启 blocking wait 或该接口确实阻塞 host，应拒绝 S lane 资格或单独标为 blocking 配置，不能默默改变发射路径。不要以 `wait(timeout=很小值)` 轮询 NCCL，它可能是失败/abort 边界。

CPU DirectExecutor 保持简单，不宣称支持 producer event；CUDA binding 误接 DirectExecutor 应 fail closed，不等出错数值后才发现。

回归：设备不匹配、无效 Work、producer event 传递、同 group 发射顺序、launch 异常、无 Torch/CUDA 的 CPU 导入、支持标记不能绕过真实能力检查。

### 3.2 R2：修复 `RuntimeHandle.wait_on()`

删除“检测 stream 有 wait_stream 就把 backend Work 传进去”的鸭子类型分支；Work 不是 CUDA Stream。

拟定契约：

- 等待 handle 成为 BOUND 或 FAILED；未绑定等待允许阻塞应用线程，但不能阻塞控制/完成线程。
- 使用本地 execution receipt 的显式能力：进入调用者指定 device/consumer stream，执行 `consumer_stream.wait_event(comm_done)`。
- 方法成功返回只表示消费依赖已建立，不将 handle 标 COMPLETED、不发完成回执、不释放容量。
- 不插入 device-wide synchronize，不等待 unrelated stream。
- 增加明确的有限 binding wait deadline；超时不取消已提交 collective，也不把超时 handle 伪装为成功。建议与 `wait_host` 一致返回成功布尔值；调用方必须检查，兼容性变更记录在文档/测试中。
- 后端没有 stream-dependency 能力时显式报“不支持”，不 fallback 到含义不确定的 `work.wait()`。
- 多次调用可为多个合法 consumer stream 建立依赖；失败时唤醒阻塞调用者，完成后仍可建立消费依赖。

要覆盖“handle 完成后 binding 已释放，再调用 wait_on”的情况：完成事件/执行回执不能随 binding 过早失效。不能只在常见调用顺序上通过。

### 3.3 R3：物理完成、生命周期及失败边界

推荐首版对 CUDA 使用 executor 建立的 `comm_done.query()` 作为明确物理完成证据；不把通用 Work `is_completed` 的名称/布尔属性当跨 backend 保证。backend Work 查询如保留，记录作用并验收其失败传播，不能与 event 的含义混称。

保持：

```text
backend launch 返回有效回执
  → bind handle
  → SUBMITTED 发送成功
  → 加入 active
  → comm_done 已完成且未进入失败状态
  → 本地 COMPLETED / 唤醒 host / COMPLETED 消息
  → coordinator 收齐全部成员 COMPLETED 才释放容量
```

即使设备 event 在发送 SUBMITTED 前已完成，也不能让 probe 越过 active 边界。`wait_on` 或 producer event 完成均不能作为通信完成。

- input tensor、producer event、backend Work、comm_done 至少活到通信安全完成；consumer 仍读取 tensor 时由应用持有引用。keepalive 与 PyTorch allocator/stream 使用关系要明确，不能只清空 Python binding 就假设安全。
- 每任务 event 不复用到尚未结束的下一任务；每 repeat 输入重置完成后才能开始新测量，防止 in-place all-reduce 数值递增污染。
- backend 错误/query 异常传播到 runtime fail-stop；event 永不完成也有 epoch/harness timeout。event.query 不是完整的 NCCL 故障恢复系统。
- 若底层失效导致线程无法安全回收，父进程在 deadline 后终止本次子进程、记录失败；不宣称 GPU collective 被 admission 取消。
- 完成日志记录 `completion_source`、executor/probe 类型、版本及验证批次，而不是只有 `supports_physical_completion=true`。

R1–R3 联合验收前，不运行任何可用于性能结论的 GPU replay。

### 3.4 R4：准备、预热、应用计时与校验

线性路径全部 precreate，但分开两件事：预分配 tensor/binding 结构，与真实 producer 写入本轮数据。后者不能因 precreate 被绕过。

共同阶段：

1. 初始化进程组、建立控制连接和注册 group。
2. 预分配 buffer/event/stream；warmup 覆盖每个 group/签名，使用独立 warmup buffer 或随后重置正式数据。
3. 初始化和 warmup 在 release 前确认完成。这里允许 setup 级设备同步，必须计入 preparation，不能带进 per-task 热路径。
4. 所有本地 job 线程准备好，执行共同启动 barrier；记录本 rank release 后释放应用。
5. H：消费点等待物理完成；S：排入消费依赖/实际消费算子，末端事件完成后记录 job/application end。
6. 独立记录 protocol/communication drain、deferred validation、harness end。

旧 H 基线在 harness 层用经验证的本地完成回执等待，不能直接把底层 CUDA Work.wait 返回当 host complete；不改旧 ScheduledWork 默认语义。S 中旧/新消费位置、独立计算工作量一致，未绑定等待单独记录。

新增元数据建议：`lane`、`offer_readiness_mode`、`application_end_definition`、`consumer_dependency_impl`、`completion_source`、`preparation_s`、`warmup_count`、`peak_allocated_bytes`、`peak_reserved_bytes`。

修正 executor 包装后的 API 计时边界：当前 runtime 的 `collective_call_start/return` 包住整个 executor 调用；新增 CUDA bridge 后，这会额外包含 device/stream 切换、完成依赖建立和 event record。应另记 `executor_start/return`，真正的 `collective_api_start/return` 在 backend 调用附近记录；旧字段保留 schema/version 解释，不能把 wrapper 总耗时继续称为纯 collective API。H/S 的桥接成本属于真实执行成本，仍计入应用时间，不能从 makespan 扣除。

验证：人为增加结果扫描工作只增加 validation/harness；人为增加预热只改变 preparation；未完成的正式 producer 仍阻止正确消费。不能通过后台 validator 抢 GPU 影响应用。

### 3.5 R5：设备映射、控制通道和 bare 安全性

**设备映射修复同时覆盖 Phase 2、Phase 3 和 profile；Phase 1 复用的启动路径也一并检查。** 当前这些 launcher 有按 rank 覆盖 `CUDA_VISIBLE_DEVICES` 的逻辑。

推荐统一：父进程确定获准的可见设备集合，子进程继承原可见集合，以 `LOCAL_RANK=rank` 选择逻辑 ordinal。若提供显式列表，它应选自当前可见集合；不要把物理数字、CUDA 逻辑 ordinal、UUID/MIG token 混用。profile worker 当前固定 device 0 的逻辑必须同步修正。

父进程在启动前检查可见 GPU 数量≥world size、映射无重复、NCCL 可用；子进程记录 rank→逻辑 index→UUID，并核对无意重复占卡。不绕过集群资源分配。单元测试覆盖可见集合为空、仅一张、重排序列表、UUID、重复映射和父子一致性。

控制面继续用 TCP；setup/finalization barrier 在测量边界外。调度热路径不得借 job NCCL collective 做 rendezvous。

bare 分级：

- 单 group、单 job raw：顺序自然明确，可在 G2 作为可选底层参考。
- 原始多 job/thread 多 group bare：默认不进入 GPU 主矩阵。独立审查是否满足目标 PyTorch/NCCL 全局顺序要求；有限次不挂不能证明一个本来无约束的顺序正确。
- 受控 raw 参考：若需要，单独实现共同发射序列及必要设备同步，命名 `raw-ordered-*` 并计入协调成本；它不再是原始 bare，不能用来宣称“动态超过原始无调度并发”。

首轮核心三 arm 足以回答迁移成本与在线收益；没有安全 bare 时留缺口，不强行凑四组。旧多 group k1 同样需要设备执行顺序验收，不能仅因有静态 host Plan 就自动放行。

### 3.6 R6：统一新路径 LTF 评分，而不篡改旧基线

问题：旧 Plan/新线性 Static LTF 当前采用含 overlap 的 remaining-score；新动态线性 LTF 使用简单相加 tail。新核心的 `remaining_tail_s` 契约应保持“当前通信完成之后的估计剩余时间”，不能只把旧 score 塞进 tail 字段就说统一。

建议冻结新线性评分为 `linear-postcompletion-tail-v2`，在零准入延迟、当前通信与独立 consumer 同时开始的估计假设下：

```text
c_i = estimated_comm_i
u_i = independent_consumer_compute_i
tail_v2(i) = max(u_i - c_i, 0)
           + Σ后继j [producer_compute_j + max(c_j, u_j)]
```

这是估计，不使用实际 ready、已实现扰动或未来真实耗时；实际准入延迟可能使独立计算已完成更多，v2 不是精确在线剩余工作。若后续要估计随状态更新，另立实验，不能混在此修复。

实施：

1. 在 examples 的线性估计工具中提供独立纯函数，新 static LTF 序列构造和新动态 hint 共用它；新 static 不再借旧 LTF builder 定义新评分。
2. 新 static builder 保持 job 内顺序、覆盖全部 task，明确稳定 tie-break；动态核心继续在合法候选中按 tail 优先，原有 eligible tie-break 保留并记录。
3. 旧 Phase 2 FIFO/LTF/SRJF 及其 score_definition 保持原样；新旧 LTF 比较标“系统与策略定义共同变化”。FIFO 基线顺序也不因本项变更。
4. DAG longest-successor tail 保持原定义；分支 sibling overlap 与线性近似并不天然同义，不强行声称线性/DAG LTF 评分相等。桥接首轮用 FIFO。
5. 输出 estimator version、逐 task c/u/tail 和静态序列；Lookahead 使用新 tail 后重跑确定性选择/截止测试。

测试至少覆盖 u<c、u=c、u>c、多后继、零时长、不同 payload、固定 tie-break、静态/动态相同候选 tail 一致，以及构造一个旧公式/新公式顺序不同的样例。历史 LTF 数据不回写；修复后 CPU 小回归和 GPU 新批次分开标版本。

### 3.7 R7：统计与报告修复

主估计量固定为：`median_seed(median_repeat(candidate - baseline))`。各字段分名：

- `baseline_arm_median`、`candidate_arm_median`：描述性 arm 中位数。
- `arm_median_difference`：两 arm 中位数之差。
- `paired_delta_median_within_seed`：同 seed 内 repeat 配对差中位。
- `seed_block_median_delta`：上项跨 seed 中位。
- `paired_ratio`：每 pair baseline/candidate，再按规定分层汇总。

对旧 F4 字段保持可读取，可加 alias/说明，不悄悄改其值。为原报告补注明勘误时保留原数字及估计量；不覆盖历史 raw。把 `[1,100,101]` 对 `[2,3,102]` 这样的两种中位数差不同的样本固化为回归。

最小观测仍保留任务全集、数值、顺序、失败校验；所需 aggregate CPU/probe 计数要先测开销，不为收集 counters 偷开 full trace。diagnostic、device-profiler 与 minimal 分批。

失败数据保存，批次标 incomplete/failed；预期样本不足时拒绝给出正式性能验收。环境失败如要补跑，保留原 ID，整配对块重新编号，不拿新的单 arm 与漂移前样本强行配对。

### 3.8 R8：真实 GPU compute 与线性推进

新增具体 GPU compute binding，不引入通用计算调度器。首版选预分配矩阵上的固定数量矩阵乘/点算子，校准 workload 大小，不用正式运行中的忙循环追目标 wall time。

每 job 至少区分 producer 与 consumer/独立计算 stream；明确数据流：

```text
producer 写通信输入 → producer_done → collective
                   └→ independent_compute ─┐
collective_done ────────────────────────────┤
                                           → dependent_consumer → next producer / job end
```

独立计算必须在可能阻塞的 wait_on/binding 等待之前入队，否则会因未获 grant 而丢失本应存在的 overlap。dependent_consumer 需要实际读取 collective 结果，不能只排空 event 然后到 validation 才读。

固定 `(seed,epoch,job,task,rank,segment)` 对应算子规模/次数；GPU 实际 duration 允许因争用变化。CPU sleep 扰动时长和 GPU 工作量不是同一数据类型，manifest 分别标记，不声称二者逐微秒等价。

首轮仍采用 physical-ready OFFER；若 job 为下一项提前入队 producer，它必须依赖上一消费结果，host 只有在该 producer_done 已完成后才提交下一通信。只预分配资源，不提前提交整条通信链消除应用反馈。

### 3.9 R9：DAG compute 完成契约

保留 CPU callable 返回即完成；对 GPU 新增具体 completion token/result，至少持有 device、done event 和 keepalive。future 返回 token 只代表提交结束，runner 必须等待 token 的完成条件，才能 `_complete(node)` 解锁完成依赖。

不用每节点线程等待 GPU；沿用每 job compute worker 与推进循环，有界 query。GPU query/失败不能在持有中央状态锁时执行。停止时未完成设备工作不能假装取消，harness 有界处理。

首版 GPU DAG 使用物理完成依赖，可能比全 stream 图更保守；不在本轮将所有 DAG 边改为异步提交依赖。报告这个额外推进成本。通过 diamond、多个终点、同 group 顺序冲突、多个 group 和迟到 compute 的测试后再做 DAG 性能。

## 4. C0/G0：测试设计与修复验收

### 4.1 C0：CPU 可完成的回归

现有命令从仓库根目录执行；具体通过/跳过数由实施记录填写：

```bash
PYTHONPATH=src pytest -q tests/unit/runtime
PYTHONPATH=src pytest -q
env PYTHONPATH=src RUN_JOBPACER_RUNTIME_REPLAY=1 pytest -q tests/integration/test_runtime_replay.py
git diff --check
```

新增 mock/可控 event 测试验证控制交错、绑定超时、错误唤醒、SUBMITTED/COMPLETED 顺序、device 校验、CPU 无 CUDA 导入回归。采用 Event/Barrier 控制并发，不靠长 sleep 或大量随机重复。

真实 Gloo 回归仍必须运行；socket 权限受限按权限流程处理，不写成逻辑失败。测试缺 GPU 是 skipped，不是 passed。

### 4.2 G0：新增双 rank NCCL 验收矩阵

建议新增独立 opt-in `tests/integration/test_runtime_replay_nccl.py`，使用动态端口、父进程超时及完整产物；旧单 GPU `tests/gpu/test_cuda_semantics.py` 只做辅助。拟定 opt-in 名 `RUN_JOBPACER_RUNTIME_NCCL=1`，实施前它尚不存在。

| ID | 测试内容 | 必须检查的证据 |
| --- | --- | --- |
| Q01 | 两卡映射、单 group FIFO | UUID 无重复；tensor 正确；group launch 投影一致 |
| Q02 | 新 Static FIFO/动态 FIFO 两 group | 共同序列、有序实际调用、没有依靠 job 线程自由排序 |
| Q03 | 非默认 producer 延迟写入 | bridge 正确；无依赖实现的对照能被断言识别，而非偶然通过 |
| Q04 | 非默认 consumer 实际读结果 | wait_on 接入指定 stream；host 无需等物理完成 |
| Q05 | 无关长工作 stream | wait_on 不引入全设备等待；诊断检查 unrelated 依赖不存在 |
| Q06 | 设备已完成、SUBMITTED 尚未发完 | active 边界阻止 COMPLETED 越过 SUBMITTED |
| Q07 | 一成员完成回执被可控延迟 | 中央不释放容量、不发下一 grant；用中央因果序列检查 |
| Q08 | 应用推迟消费 | 完成回执不依赖应用 wait_on/wait_host 的调用时机 |
| Q09 | handle 完成后建立 stream 依赖 | done event 仍有效；binding 清理不导致悬空引用 |
| Q10 | device/event/stream 不匹配 | collective launch 前明确拒绝，不死锁其他 rank |
| Q11 | metadata/missing-task/launch/probe/断连 | fail-stop、等待者唤醒、后续停发、父进程有界退出 |
| Q12 | H/S 末端与 validation 隔离 | 应用完成不靠 validation 隐式同步；扫描工作变化不进入应用指标 |
| Q13 | warmup、buffer 重置、重复 epoch | 无残余在途、无重复 reduction 污染、引用释放时机安全 |

GPU 时序测试可用有界计算核形成窗口，但控制路径交错仍用显式事件/屏障；不得用无限 GPU spin 构造不可回收挂起。真正设备完成与延迟发送 COMPLETED 是不同测试，Q07 只验中央回执门控，不能独自证明设备 probe 正确。

通过条件：每项适用的断言成立；记录实际 NCCL/CUDA/PyTorch 版本。对目标 backend 的执行回执/完成事件构造须有独立 stream/设备证据。不能拿两次 replay 没挂当作 Q03–Q09 的替代。

## 5. G1：profile、噪声与观测扰动

前提：C0/G0 通过，R7 可正确汇总，源码冻结。

### 5.1 校准

- 4 KiB、1 MiB、16 MiB；float32 sum；相同成员与设备；每签名 warmup 5、正式样本 30。
- API duration、host 调用到物理完成、设备依赖区间分别命名；策略估计明确使用哪一个服务时间字段。
- profile 边界外允许同步/汇总，不能把每次跨 rank 汇总 collective 算进被测通信耗时。
- 如 warmup 5 后仍有明显初始化漂移，只重做校准并冻结更合适的次数，主实验不用数据驱动地逐 arm 调整 warmup。
- 16 MiB 仍过短时可在 pilot 加 64 MiB；是否进入主实验先决策并记录，不自动增加全部尺寸矩阵。

生成目标机器 profile，严格签名匹配；CPU profile 绝对时长不复用。三种尺寸×30 个服务样本不是 90 次独立 replay。

### 5.2 观测与噪声

1. new Static FIFO、4 KiB×32、H、固定输入：minimal/diagnostic 各五次，交错配对，共 10 replay。比较 makespan、CPU、context switches 和 probe 数。
2. new FIFO、L0、H、零 jitter：相同配置标 A/B，五个配对块，每块 A/B 各三 repeats，共 30 replay。这里 block 是运行噪声块，不宣称产生五种计算扰动。
3. device profiler 仅少量单独诊断，不加入上两组统计。

主性能固定 minimal；diagnostic 有扰动就记录，不减常数修正。噪声仅用于解释分辨率和预设实际意义阈值，不把噪声 P95 冒充显著性检验。

## 6. G2：固定通信链，先测平台成本

固定单 job、单 group、零 producer/consumer 计算、32 项、precreate、H、minimal、poll 1 ms。两 arm：old-static-fifo-host-complete 与 new-static-fifo-host-complete。默认 DECLARE before-producer、wakeup off。

| 消息 | 固定输入配对 repeats | replay 数 |
| --- | ---: | ---: |
| 4 KiB | 5 | 10 |
| 1 MiB | 5 | 10 |
| 16 MiB | 5 | 10 |
| 合计 | 15 pairs | 30 |

不同 replay 串行；每 pair 随机 arm 先后。不要把链中 32 项当作 32 个独立统计样本。H 旧基线等待机制的新增成本单列，不能声称与历史 Gloo old 完全相同。

必报：每对总时间差、`ΔT/32`、all-rank 最大本地应用 duration、CPU、task API/handoff、probe 统计、完成源和 protocol drain。设备事件区间若只来自 diagnostic，则不与 minimal 跨运行相减。

条件追加，最多先选一项：

- **长度检查**：若额外成本可分辨，选 1 MiB 补 N=1/8、两 arm×5 pairs=20 replay，描述 `ΔT≈a+bN`；不拟合为普适常数。
- **poll 检查**：若诊断显示明显周期探测量化，选 4 KiB×32 的 new Static FIFO，1/0.2 ms×5 pairs=10 replay。只改变 poll，先不动 DECLARE、锁或 wakeup。
- **底层 raw 参考**：仅当需要区分旧 wrapper 成本，单 group raw-H 与 old-H 按一个代表大小五配对=10 replay；原始已测 old 样本不能拿来拼配对。

任何新增优化若测试，只能作为新 arm，要求机制区间和 minimal 端到端同时评估；不因为局部下降就改默认。G2 没看到 2 ms 也应如实记录平台变化，不为复现 CPU 数字调输入。

## 7. G3：L0/L1 的同 runtime 收益与跨系统净收益

### 7.1 输入与机制 pilot

以已有 L0/L1 线性语义为模板，保留两 job、各两项通信。第一轮使用 H + host sleep；这是 GPU 通信/host 到达控制实验，不叫真实 GPU 训练计算。

消息先用 G1 的 1 MiB；L0 不额外设 rank 偏斜。L1 只增加静态队首 job 首 producer 的延迟。pilot 可依据 `t_service` 与 G2 闭环时间设置偏斜，例如在 2×、4×代表通信周期两档中检查机制；每次输入新 revision，最多两次调参后决定进入或暂停，不无限放大到赢为止。

机制 pilot：L0/L1 × old static/new static/new FIFO × 3 repeats=18 replay。L1 验收检查：

- static 的指定队首未 eligible 时存在另一个合法候选，并保持等待；
- dynamic 确实在队首 ready 之前服务另一个候选；
- 各成员实际 launch 一致、数值正确，固定任务全集；
- 同输入三次均能观察到目标机制，再冻结用于性能。判据是机制而非 speedup。

若需要调整偏斜，不从已有性能样本筛“机制触发成功”的子集；原 pilot 完整保留。未通过 gate 的场景不扩正式 repeats。

### 7.2 正式探索矩阵

5 execution seeds×3 paired repeats，jitter=0.3；同 block 中固定所有 producer/consumer 样本，arm 随机顺序，源码/profile/input 均冻结。

| 场景 | 核心 arms | replay 数 |
| --- | --- | ---: |
| L0 | old static FIFO H、new Static FIFO H、new FIFO H | 45 |
| L1 | 同上 | 45 |
| 总计 | 30 blocks，每 block 3 arms | 90 |

如合格 raw/bare 参考在该批开始前已确定，可一开始按四 arm 执行，共 120；不能完成三 arm 后把后来的 raw 与旧数据声称同批配对。

三个预设对照：new static→new FIFO（在线收益）、old static→new static（整体路径差异）、old static→new FIFO（净效果）。后两者仍有 rank-local/global 容量语义不同，不称纯控制开销。

逐 job JCT、平均 JCT、最慢 job 与 makespan 一起报告。正式样本全部保留，即使有些 block 没触发 HOL；触发率作为结果解释，不能事后剔除。

L5 不是默认矩阵；只有 L1 后需要研究慢成员是否改变结论时，固定一场景×3 arms×5 seeds×3 repeats=45 次。LTF/Lookahead 不随本轮自动扩测。

## 8. G4/G5：真实 GPU 计算与 DAG

### 8.1 G4：S lane 计算重叠

前提：R8 通过 GPU 正确性，R1–R5 已验收 S lane，冻结算子与数据依赖。

选一个代表消息大小（默认 1 MiB）；独立 compute 两档，校准目标约为同平台无竞争通信服务时间的 0.5×、4×。实际以离线选定的算子规模/次数表达，不强行把运行耗时精确匹配到该比例。

arms：old static FIFO S、new Static FIFO S、new FIFO S。physical-ready、相同输入工作量、precreate、相同 poll；两档×3 arms×3 seeds×3 repeats=54 replay。三个 seed 是探索规模，不支持强鲁棒性声明。

重点指标：device-complete 应用时间、每 job 末端完成、独立计算是否真正 overlap、通信/计算各自的设备观测区间、host 提交/等待、CPU、显存。没有设备 profiler 的证据不要仅凭两个异步 API 就声称真正并行。

若新增 compute 与 NCCL 争用，区分“计算变慢”“通信相关设备区间变长”“host 控制空档”三种信号。GPU 工作量按 seed 配对，不按测得 duration 配对；调度引起的持续时间变化属于结果。

### 8.2 G5：DAG bridge 与 diamond

前提：R9 完成。先做语义，不先铺五策略矩阵。

- linear→DAG bridge：同一两 job 负载、相同 GPU 算子与通信参数，明确额外执行顺序是否保留；先 FIFO 三次配对（6 replay）。比较节点/task 全集、数据结果、依赖、group 投影和工作量，不要求线程时序/JCT完全相同。
- diamond：两前驱 join，含实际 GPU compute 和 comm；Static FIFO/FIFO 各三次（6 replay）。确认 comm 完成和独立计算完成共同解锁 join。
- 另以针对性测试覆盖多个终点、迟到 compute、错误 group_seq、依赖与 group 顺序组合环、compute query 失败。

若两项机制合格且有性能研究必要，只给一个 DAG 输入增加 new static/new FIFO×3 seeds×3 repeats=18 replay。一般 DAG 没有 old/bare 等价物时不新增虚假的跨阶段对照。

DAG 采用物理完成推进，可能有额外事件观察开销；它不是已优化的框架 stream graph。CPU/Gloo DAG 完成和本阶段 GPU DAG 完成分别记录。

## 9. 后置策略与优化实验：进入条件

| 项目 | 必要条件 | 最小下一步 |
| --- | --- | --- |
| LTF | R6 已通过；两个指定研究候选共同 eligible 可稳定出现 | 固定候选竞争机制，FIFO/LTF 同 runtime 配对，不直接全 L0–L5 |
| Lookahead | R6、预测前沿/固定 deadline GPU 时间尺度测试通过 | 准时、晚到各一个输入；先看触发/回退，不用 CPU 20 ms 预算硬套 |
| DECLARE on-submit | G2/G3 指向声明推进的可见成本 | 一个输入单变量消融，保持 Lookahead/DAG 限制 |
| wakeup/低 poll | GPU 完成观测确实限制关键路径 | 同时报告 CPU 和 makespan；不照搬 E3 失败候选为默认 |
| 控制线程/发送锁改造 | 已有 trace 证明对应等待主导且能独立复现 | 先确定性回归，再一次一项修改；不删锁破坏消息顺序 |
| 新 runtime 多在途 | 单在途目标平台验收完成、单独方案获确认 | 新容量/设备排序验收，不属于本方案自动实施 |

原来的约 2 ms 是系统级差距，不列为“修掉一个 2 ms bug”的验收项。R1–R9 修的是明确语义/实现/测量问题，是否更快由后续实验决定。

## 10. 统计、规模上限和停止规则

### 10.1 指标与时钟

- `application_makespan=max_rank(local_application_end-local_release)`；H/S 分别使用约定终点。不是未经同步的全局 wall-clock span。
- 每 job 从相同本地 release 计算；平均 JCT 先在 run 内平均再汇总，不能平均各 job 独立中位数冒充 run 均值。
- CUDA event elapsed 只在同设备有效依赖链上解释；coordinator 的接收时刻不能减 rank 原始时间戳。sendall 返回不是远端收到。
- API return→completion observation 含设备执行和轮询，不能标纯通信或纯 poll delay；分段中位数不能相加还原全部差距。
- representative timeline 标选择规则；因果诊断用同 seed/repeat 双 rank 分面，不能把独立中位代表图当成一对。

### 10.2 重复与统计决策

- 固定输入链：五次配对，报告全部差值、median/range，不伪称五独立 workload seeds。
- 多 job：repeat 内先配对，seed 内取配对差中位，跨 seed 汇总；95% bootstrap 按 seed block 重采样，建议 2000 次、固定分析 seed。
- 同时报告绝对 ms、配对 ratio、原始点、样本/失败数。P10–P90 是分布，不是 CI。
- G3 五 seeds 为探索。若需要正式确认，事先登记追加五个新 seeds，各三 repeats，并报告初始与新增批次各自结果及共同配置；不循环追加直到显著。
- 统计显著与实际意义分开：在主批次前结合目标应用和 G1 噪声冻结实际意义阈值；未给定应用预算时不自封“2% 就有价值”。
- 不按策略事后分别选择最优 poll、最优容量或最有利 message size 来做主比较。

### 10.3 默认执行量

| 阶段 | replay 数 | 是否首轮必跑 |
| --- | ---: | --- |
| G0 | 按语义测试参数，不计入性能样本 | 是，运行适用项 |
| G1 观测扰动 | 10 | 是 |
| G1 噪声 | 30 | 是 |
| G1 profile | 3 签名×30 测量样本，另有 warmup | 是，不按 90 replay 计 |
| G2 通信链 | 30 | 是 |
| G3 机制 pilot | 18 | 是，最多两轮输入 revision |
| G3 主矩阵 | 90；预先含合格 raw 时 120 | gate 后执行 |
| G4 compute overlap | 54 | 第二阶段，R8/S gate 后 |
| G5 bridge/diamond | 12 | 第二阶段，R9 gate 后 |
| G2 长度/poll/raw | 20 / 10 / 10 | 条件追加，先只选一项 |
| L5 / DAG 性能 | 45 / 18 | 条件追加 |

G1–G3 默认最小为 **178 replay**（10+30+30+18+90），不含 profile、G0、补测和输入 revision；含初始合格 raw 的主矩阵则为 208。不是一次全部启动的任务：每阶段 gate 后再继续。G4/G5 独立第二阶段，不自动合并进首轮预算。

需要跨契约新增 H/S 功能 smoke、R6 修复后的 CPU 小回归时，单列为验证成本，不从上述性能 repeats 中挪用，也不把 smoke 作为正式样本凑数。

先用 smoke 实测每 replay 的 setup/应用/总墙钟，以中位和 P90 估算批次时长并写 manifest。不得把毫秒级应用时间乘次数当成实验墙钟预算。支持阶段 resume，但只复用 manifest 标记且哈希有效的已完成 run，不扫描目录混入旧文件。

立即停止性能阶段的条件：tensor/顺序/完成边界错误、不可回收挂起、GPU 映射错误、样本不配对或统计失败。保存产物后修复，以新源码/批次重跑；不能删除坏样本继续宣布全通过。

性能无收益、CI 跨 1、噪声较大不是代码正确性失败，也不自动授权扩矩阵或架构重写。报告边界与未决问题即可。

## 11. 脚本、配置和产物

保留现有入口职责：`run_phase1/2/3.py` 单 replay，`run_comm_profile.py` 校准。拟新增一个薄的 GPU 阶段编排入口 `examples/jobpacer/scripts/run_phase3_gpu_experiments.py`，复用 `run_experiments.py` 的命令/manifest/顺序逻辑，不复制整套 batch runner。

不要把硬编码 Gloo 的 `run_control_path_diagnostic.py` 默默切成 NCCL；如抽公共分析/配对能力，保留 E/F 原命令与输出含义。

拟新增阶段接口（只有实现并通过测试后才可执行）：

```text
python -m examples.jobpacer.scripts.run_phase3_gpu_experiments
    --stage G1|G2|G3-pilot|G3-main|G4|G5
    --config <冻结的阶段 JSON>
    --output <新的 batch 目录>
```

配置至少写明：backend/world_size、设备映射规则、lane、OFFER ready 模式、arms、inputs、profile、estimator version、seed/repeat/order seed、warmup、poll、DECLARE/wakeup、observation、timeout、预期 run 数及 gate 输入。

建议路径（待创建）：

```text
benchmark/phase3/
  experiments/gpu-readiness/
    semantics/ calibration/ communication-chain/
    readiness/ compute-overlap/ dag-bridge/
  results/gpu-readiness/
    <同名语义分类>/<batch-id>/
      inputs/ raw/ tables/ figures/ logs/
      manifest.json runs.jsonl analysis.md
```

脚本只放 examples；benchmark 只放输入和结果。GPU 上的新文件保持原有语义命名，不把所有模块改成 phase 名。

每 batch 保存完整命令、stdout/stderr、退出码、wall time、输入/profile/source SHA、可恢复源码、设备/版本、实际 executor/probe、group/Plan/launch 序列、计算样本摘要、配对完整性、失败/skip、图的源 trace。results 被 Git 忽略时，额外说明归档位置和恢复方法；链接存在不代表 raw 在其他 checkout 可用。

### 11.1 每阶段产出

| 阶段 | 必交产物 |
| --- | --- |
| R/C0/G0 | 修复差异、命令、正常/异常测试结果、GPU 能力矩阵与未验收项 |
| G1 | profile、噪声/观测扰动表、冻结的测量配置 |
| G2 | paired chain 表、按项描述、CPU/probe、必要设备证据 |
| G3 | paired makespan/逐 job JCT、机制触发率、输入样本一致性、系统净收益解释 |
| G4 | 真实算子配置、S lane 依赖与末端证据、计算/通信/host 分层分析 |
| G5 | 节点全集、join/group 正确性、bridge 差异及 DAG 附加成本 |

结果报告拟写 `docs/JobPacer/result/phase3-gpu-<date>.md`，只记已验证事实。本文实施时追加状态，不把未执行条目打勾。

## 12. 执行完成清单

- [ ] R0：环境、源码快照与 H/S/ready 合同冻结。
- [ ] R1：新 CUDA executor 与本地执行回执。
- [ ] R2：wait_on 目标 stream、超时、失败及完成后调用语义。
- [ ] R3：物理完成、SUBMITTED 边界、生命周期与有界失败。
- [ ] R4：precreate/init/warmup、旧新 H/S 计时与 deferred validation。
- [ ] R5：设备映射、profile 同步修复、多 group/raw 资格判定。
- [ ] R6：新 LTF tail v2 统一，旧历史评分不变。
- [ ] R7：统计字段、block 配对、失败样本及报告勘误。
- [ ] C0：CPU 全仓和真实 Gloo 回归，记录 skipped。
- [ ] G0：双 GPU/NCCL 全套适用语义/故障项通过。
- [ ] G1：目标 profile、观测扰动、噪声。
- [ ] G2：30 次最小通信链及结果解释。
- [ ] G3：L0/L1 pilot 与 gate 后主矩阵。
- [ ] R8/G4：真实 GPU compute 与 S lane 探索。
- [ ] R9/G5：GPU DAG 完成契约与 bridge/diamond。
- [ ] 结果、源码、输入、原始数据可恢复归档；未验收边界明确。

## 13. 参考语义与本次编写检查范围

[PyTorch collective 同步/异步语义](https://docs.pytorch.org/docs/2.14/distributed.html#synchronous-and-asynchronous-collective-operations) 与 [groups 安全要求](https://docs.pytorch.org/docs/2.14/distributed.html#groups) 是 CPU wait、CUDA stream 依赖和多 group 使用不能混称的依据；[NCCL 多 communicator 说明](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/communicators.html#using-multiple-nccl-communicators-concurrently) 用于目标版本的设备排序审查。必须使用目标安装版本重新核对，不能因为链接到 2.14 就宣称历史 2.13 环境已经验证。

上述官方页面已在前一份总结编写时查阅；本次再次访问网络失败，未据此宣称获得新的版本确认。实施时应保存目标版本信息及相关文档/源码依据。

本次仅新增方案，复核当前 executor、handle、launch/completion、设备启动/profile 和旧 GPU 测试代码；没有修复 R1–R9、没有运行测试或 GPU 实验。特别是 NCCL event completion 路径属于拟实施方案，必须通过 G0 后才成为已支持能力。
