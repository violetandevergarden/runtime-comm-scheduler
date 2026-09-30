# Phase 3 GPU 七臂实验计划：v2 当前合同

2026-09-29 修订。下一轮以[完整实施要求](../process/phase3-gpu-seven-arm-implementation-20260928.md)为执行合同。软件准备不等于 backend、机制、测量或正式资格已通过。

- 七臂固定为 `bare-ordered`、`old-static-fifo`、`old-static-ltf`、`new-static-fifo`、`new-static-ltf`、`new-dynamic-fifo`、`new-dynamic-ltf`。`raw-ordered-static-fifo` 仅用于同顺序串行诊断。
- Bare 使用共同分层轮转默认顺序、唯一异步 dispatcher、独立完成观察，不施加 admission 单在途限制；旧路径为本地单在途，新 runtime 为全成员物理完成释放的全局单在途。
- 新输入位于 `benchmark/phase3/experiments/gpu-seven-arm-v2/revision-0/`。L0/L1 是链式 DAG，D0–D3 为不同一般 DAG；三 pilot seeds 为 9101–9103，五 formal seeds 为 8101–8105。真实执行 repeats 与名义 profile 签名独立保存。
- 六场景三 pilot seeds 三 repeats 的六 scheduler 机制诊断为 54 块、324 replay；七臂全覆盖则为 378 replay。D1/L1 各要求至少两个 seeds 在三次中至少两次触发。输入参数最多修订两轮，另起目录与源码 revision。
- 正式仍是 150 完整块、1,050 replay，采用经诊断的 minimal 模式；A/A 与 minimal/full 独立诊断不并入正式统计。缺任一 readiness gate 时拒绝正式启动。
- 当前实施和验证边界见[准备记录](../process/phase3-gpu-seven-arm-preparation-20260929.md)。没有启动正式采集，不能将生成输入、通过单元测试或一次 smoke 写成有效实验或净收益。

## 历史：2026-09-28 串行 raw 替代版本

以下保留旧批次定义与研究上下文，其 raw 替代授权、单 seed pilot 和旧资格流程不适用于 v2。旧原始结果需使用各自归档源码解释，不与 v2 拼接。

# Phase 3：GPU 七配置重测与线性 / DAG 实验计划

日期：2026-09-27；修订：2026-09-28。状态：**已修订第七臂定义；原始 bare 已有直接提交实现和双卡诊断，但没有通过全局顺序安全验收，仍不纳入本版矩阵。执行事实以 preflight 结果记录为准。** 本修订本身不代表替代 arm 已通过双卡资格或启动正式矩阵。

实现状态更新（2026-09-28）：`BareDagAdapter` 已在共同 DAG 路径上实现绕过 admission 的 ready-task 直接提交，并通过真实双卡 L1/D1/D3 诊断。L1 的一轮实测出现两 rank 不同的全局 launch 序列；数值检查通过不消除这一顺序安全失败。原始 bare 因此仍为未资格认证配置；正式矩阵保留修订后的 `raw-ordered-static-fifo`。

2026-09-29 顺序基线补记：为避免手写 DAG 的确定性 tie-break 把一个 job 的通信链全部排在另一个 job 之前，预检冻结 `topological-layer-job-round-robin-v1`：完整 DAG（含 `group_seq` 和 `submit_after`）决定层级，先浅层，同层按固定输入 job 顺序轮转，同 job 按输入节点顺序打破平局，再取通信投影。bare 与旧/新 static FIFO 共用该序列；dynamic FIFO 仍按首次 eligible 排序。L1/D1 双卡 pilot 的默认和反转 job 起始顺序共 16 次 replay 已通过顺序与数值检查，详细记录见[分层轮转 pilot 结果](../result/phase3-gpu-layered-fifo-pilot-20260929.md)。此规则不宣称最优、公平无偏或代表 Megatron 实际调用顺序；pilot 不覆盖历史记录、不解除 D1 机制门槛，也不自动放行正式矩阵。

配套设计：[GPU workload、执行契约与通信调度](phase3-gpu-workload-and-scheduling.md)。核心约束继续遵循 [Phase 3.1](phase3.1.md) 和 [Phase 3.2](phase3.2.md)。历史依据为[首轮 GPU 实验](../result/phase3-gpu-20260927.md)，历史 raw 与结果不覆盖。

## 1. 目标与范围

在新的、明确描述真实 GPU 工作量的输入上重新建立七种配置的完整对照：

1. `raw-ordered-static-fifo`：共同静态 FIFO 全序、无 policy admission 的受控参考。原始 `bare` 已实现但未通过顺序安全资格，不纳入本版矩阵。
2. 旧 scheduler + 静态 FIFO Plan。
3. 旧 scheduler + 静态 LTF Plan。
4. 新 scheduler + static FIFO。
5. 新 scheduler + static LTF。
6. 新 scheduler + 动态 FIFO。
7. 新 scheduler + 动态 LTF。

正式矩阵采用 **2 个线性场景 + 4 个 DAG 场景**，每场景 **5 个随机工作量样例，每样例每配置重复 5 次**。共 `6 × 7 × 5 × 5 = 1,050` 次正式 replay。这里“5 次”指五次独立启动的正式测量，不包含 warmup、失败重试或 profiler。

用户要求至少三个随机样例和每配置至少五次测量；本计划采用五个样例，增加 workload 覆盖。3 seeds × 5 repeats 的 630 次仅作为可能的预算缩减方案，不是默认执行矩阵，不能执行后悄悄降低完成标准。若需更改规模，在正式批次前更新计划与 manifest。

本轮研究 GPU 内部的系统与策略差异，不与 CPU/Gloo sleep 结果配对、合并或比较绝对性能。CPU 检查继续承担协议与旧路径回归用途。是否出现 kernel overlap 不作为实验成功条件。

## 2. 研究问题与主要对照

| 问题 | 主要对照 | 可得结论的范围 |
| --- | --- | --- |
| 共同静态发射顺序下、绕过 admission policy 的系统表现如何 | raw-ordered-static-fifo 与 old/new scheduler 各 arm | 含固定全局发射次序的受控参考；不能推断原始裸发行为 |
| 新旧执行路径有什么差别 | old-static-FIFO vs new-static-FIFO；old-static-LTF vs new-static-LTF | 在共同 Plan / 合同下的系统差异；容量差异仍须说明 |
| 在线选择有没有价值 | new-static-FIFO vs new-dynamic-FIFO；new-static-LTF vs new-dynamic-LTF | 同 runtime 下的在线准入选择效果 |
| LTF 是否优于 FIFO | 同一执行类别内 FIFO vs LTF | 给定估计定义与目标指标下的策略效果 |
| 对不同 GPU 工作量是否稳健 | 同场景不同 workload seed | 有限样例范围内的稳定性，不外推平台普适结论 |

预先将 new static→同策略 new dynamic 的 makespan、run 内平均 JCT 配对结果列为主要对照。新旧静态与 raw-ordered 受控参考列为系统基线；原始 bare 诊断单独报告，不属于本版正式 arm。其他交叉比较列为次要探索，不能从所有两两组合中事后挑一个“赢家”代替主要结论。

同时报告逐 job JCT，避免平均值改善掩盖某个 job 被持续推迟。LTF 是启发式，不预设它一定改善平均 JCT 或 makespan。

## 3. 七种配置的执行合同

### 3.1 共同设置

首版为单机两张获分配的 GPU，每 rank 一个设备，真实 NCCL float32 SUM all-reduce。所有 arm 使用相同：

- 规范 workload、计算与通信任务全集、group 成员和 group_seq。
- producer / independent / dependent 的算子、shape、dtype、repeats、输入数据与依赖。
- 初始化、预分配、预热、consumer 消费与应用末端定义。
- profile、估计版本、precision 设置、CPU affinity/线程设置及采样模式。
- workload seed、执行样例、超时合同与事后数值校验。

线性路径使用“producer physical-ready → 提交通信请求 → 提交 independent → 两者汇合 → dependent/terminal → 下一片段”。通信请求接口须及时返回，不能在 job 推进线程等静态队首或 grant，阻塞本应独立的计算。

所有 GPU 阶段由固定工作量程序执行，不把旧 producer/consumer 秒数当实际工作量。正式输入中不隐含 host sleep 作为主要计算；确需控制成员到达的机制诊断单独命名，不混入正式性能样本。

### 3.2 配置表

| arm ID（拟议） | 提交与选择 | 必须记录 |
| --- | --- | --- |
| `raw-ordered-static-fifo` | 绕过 admission policy；所有 rank 按同一静态 FIFO 全序直接提交通信 | 全序 hash、逐 rank 实际发射投影、串行 dispatcher 与 NCCL 顺序成本、峰值在途数 |
| `old-static-fifo` | 旧 scheduler 执行预构造 FIFO Plan | Plan hash、旧容量参数、实际 group 投影 |
| `old-static-ltf` | 旧 scheduler 执行预构造 LTF Plan | Plan hash、评分版本、旧容量参数 |
| `new-static-fifo` | 新 runtime 忠实执行同一规范 FIFO 序列 | 序列 hash、静态队首等待、全局容量 |
| `new-static-ltf` | 新 runtime 忠实执行同一规范 LTF 序列 | 序列 hash、评分版本、全局容量 |
| `new-dynamic-fifo` | 按首次 eligible 顺序选择 | eligible 次序、选择、等待原因 |
| `new-dynamic-ltf` | 从合法 eligible 候选按版本化 LTF 评分选择 | 候选评分、选择及估计来源 |

新 runtime 保持全局 `max_inflight=1`；旧 scheduler 固定既有 `max_outstanding=1` 作为首轮配置。两者不能仅因参数都为 1 就宣称容量等价，需检查实际释放条件与跨 rank 约束。raw-ordered 不使用 admission policy，但保留共同静态全序及串行 dispatcher；它的容量和执行时间差异须按受控参考解释。

### 3.3 原始 bare 的资格门槛与本版替代定义

裸发是本轮必须解决的基线，不默认为已有可用能力。此前入口仅支持 bare + S lane；本轮已增加 DAG 直接提交适配器，但多 job/thread、多 group 的自由发射仍没有跨 rank 全局顺序保证。

实施前先交付 bare 合同说明，回答：

1. 谁决定各 rank 的跨 group 发射顺序？如何避免线程到达顺序不同？
2. group 内身份与序号如何保持一致？DAG 分支同时 ready 时如何处理？
3. 目标 PyTorch/NCCL 配置要求的 host/device 同步由谁建立？
4. 若本地 launcher 排队，是否阻塞计算推进？同步与协调成本是否进入应用指标？

若目标环境无法安全运行原始多线程自由裸发，不能硬跑或悄悄改名。当前已实现的 `BareDagAdapter` 在真实双卡 L1/D1/D3 诊断中出现跨 rank 全局 launch 顺序分歧，仍不合格。可采用带共同发射顺序的 `raw-ordered` 受控参考，明确它有顺序控制但无 policy admission；受控 raw 不等于原始 bare，不能声称完成了后者的验收。

**2026-09-28 正式计划修订：**根据 [bare 合同评估](../process/phase3-gpu-seven-arm-bare-contract-20260928.md)，当前目标环境的 NCCL 多 communicator 使用要求各设备上的 host 发射顺序一致；shared DAG runner 的不同 job 由独立线程推进，无法在不加共同全序的情况下证明此条件。加共同全序会得到已有 `raw-ordered` 语义。因此本矩阵的第七臂正式改为 `raw-ordered-static-fifo`，研究问题相应收窄为“无 admission policy 的共同顺序参考”。本修订不把替代臂标为 qualified；它仍须通过 P3 目标双卡资格与 P4 七臂彩排。随后实现并诊断的原始 `bare` 曾观测到跨 rank 全序分歧，故仍不符合本矩阵安全合同；所有结果需明确区分直接裸发诊断与正式替代臂。

没有合格 bare 时可以继续其余配置的语义开发，但不得把六配置批次写成七配置完整交付。不得以复用新 coordinator、再关闭日志的方式伪造裸发。

### 3.4 旧 scheduler 的 DAG 接入

当前旧 Plan builder 面向线性 Workload。为覆盖 DAG，需在 workload 层增加旧路径通信绑定，让同一 DAG runner 执行相同计算程序与依赖：

```text
统一 workload / DAG runner
    ├─ raw-ordered 静态全序直接提交绑定
    ├─ 旧 scheduler 通信提交与完成绑定
    └─ 新 RankRuntime 通信提交与完成绑定
```

绑定只转换通信身份、submit、错误和完成回执，不重写 DAG 推进、不把新核心连接到旧 Plan。旧 TaskKey 可在 adapter 中确定性映射，但所有成员必须相同，group 投影必须可核对。

静态 Plan 只规定通信顺序，不能把完整 DAG 展平成“计算→通信→等待→计算”的串行链。旧 scheduler 静态等待期间，同 job 的其他独立计算仍应能够推进。

首版统一使用既有 DAG 的完成依赖推进，避免某 arm 采用全 stream 图而另一 arm 采用 host 完成轮询。线性与 DAG 不要求耗时相等；等价 bridge 只用于验证工作量、依赖和终点。

## 4. FIFO、LTF 与静态 Plan 的共同定义

### 4.1 FIFO

静态 FIFO 采用输入稳定顺序，在满足完成依赖与 group 顺序的前提下生成一个确定性的通信全序。线性可按 ordinal/job 稳定次序，DAG 采用共同的稳定拓扑构造；构造规则与 tie-break 写入版本。

动态 FIFO 按中央首次 eligible 顺序选择，不改成输入注册顺序。静态 FIFO 和动态 FIFO 不是同一种排序规则，这正是对照内容。

同场景、同 seed 的 old-static-FIFO 与 new-static-FIFO 使用同一份规范通信序列，分别转换为旧 Plan 和新 StaticOrder；比较序列 hash 及实际投影。

### 4.2 LTF

先实现配套设计中的 GPU 工作量 profile 与 tail 版本，再进入 LTF 实验。线性 tail 明确 independent 与通信汇合；DAG tail 按本 job 合法后继关键路径估计，并计入必要的同 job group 顺序约束，不能把并行分支时长直接相加。

静态 LTF 在启动前从合法拓扑前沿按同一评分规则生成通信序列；动态 LTF 在真实 eligible 集合上应用同一版本的评分。二者可以面对不同候选，但不应暗中采用不同 tail 定义。图、group 顺序及静态全序联合校验无环，避免队首等待其后继造成死锁。

为隔离新旧执行路径，本轮 `old-static-ltf` 指“旧执行引擎执行本轮共同 GPU LTF Plan”，输出标记 `plan_origin=gpu-shared-estimator`，不称为未修改的历史旧 LTF 算法。历史旧 builder 保持不变，不回写历史评分或结果。如果还需研究历史算法本身，单列追加 arm，不替代七配置中的共同 Plan 对照。

profile 的估计与实际执行扰动分离。首版使用固定、版本化启发式；没有在线计算进度消息就不能声称评分考虑了精确的独立计算剩余量。记录估计偏差，不用本次实际未来时长回填。

## 5. 六个场景

每场景包含至少两个 job，产生真正的跨 job 通信选择空间。每 job 的计算与通信规模保持小而可解释；正式节点数、矩阵形状与消息尺寸经 pilot 后写入冻结 manifest，不在此凭空规定毫秒目标。

| 场景 | 结构与主要变化 | 预期检查 |
| --- | --- | --- |
| L0-balanced | 两个线性 job，各至少 3 个通信片段；上游与独立计算量接近 | 无明显队首错位的负对照，观察系统成本；不要求动态更快 |
| L1-skew-tail | 两个线性 job；静态前项 producer 较长，另一个 job 较早 ready；后续路径长短有别 | 静态 HOL、动态绕过；另设能同时 eligible 的决策窗口检查 LTF/FIFO 区别 |
| D0-fork-join | 两个 job 均有 producer 分叉为通信与独立计算，再汇合；join 后还有后继通信 | producer/consumer 数据依赖、两前驱 join、独立推进 |
| D1-asymmetric-frontiers | 每 job 含不等长分支、不同 group 的多个通信前沿，最终汇合 | 至少一次多个合法候选 tail 不同；检查 LTF 选择及对关键路径/JCT 的影响 |
| D2-cross-job-skew | 多 job，分支 producer 工作量错位，不同 group 竞争全局通信容量 | 静态队首未 ready 时其他任务可服务；记录实际 HOL 触发率 |
| D3-order-and-sinks | 同 group 通信规范顺序与跨 group 独立前沿混合，至少一个 job 有多个终点 | 后项早 ready 也不越序；其他 group 可推进；job 完成等待所有终点 |

D3 不构造无法调度的组合环作为性能输入；非法图属于单独负向测试。不同 group 不能只是每个通信随意新建一个 communicator 来人为消除所有排序约束。

上述四类 DAG 均使用真实 GPU 计算。D0 中独立计算量可短于或长于通信，但不为了提高重叠比例反复调到某一策略获胜。

### 5.1 Pilot 的机制门槛

- L0：任务/工作量相同、执行正确即可，允许策略几乎无差异。
- L1/D2：诊断记录中至少出现静态队首等待且其他合法候选存在；动态能选择其他候选。
- D1：出现真实的候选竞争，不是所有决策都只有一个 eligible；记录评分与选择。
- D0/D3：检查分支、join、group 顺序、多终点的因果关系。

不以 speedup 作为冻结输入的标准。每场景最多进行两轮参数 revision；若机制仍未触发，停止该场景并解释原因，不无限放大偏斜“调到赢”。pilot 的全部版本与失败保留，不能混进正式统计。

## 6. 随机样例与重复测量

### 6.1 三种随机种子

| 种子 | 控制什么 | 不控制什么 |
| --- | --- | --- |
| workload seed | 每节点固定执行工作量，如有界整数 repeats 扰动 | 不固定任务绝对 ready 时间，不依赖实际调度结果 |
| tensor seed | 同一计算程序的确定性输入值 | 不能仅换数值就称为不同调度样例 |
| order seed | 配置顺序与 block 顺序 | 不改变 workload 或策略估计 |

拟议正式 workload seeds 为 `8101, 8102, 8103, 8104, 8105`，order seed 为 `20260927`；pilot 使用另一段种子。生成规则和 PRNG 版本与输入摘要一起归档。seed 值不是现有已执行批次编号。

固定各场景拓扑，对基准 repeats 使用预先冻结的有界离散扰动，例如 `{0.75, 1.0, 1.25}` 乘基准后以明确规则转为正整数。校验五个样例确实生成不同工作量；rounding 导致重复时调整生成规则并版本化，不能按性能重抽种子。tensor 值由稳定键 `(scenario, workload_seed, job, node, rank, stage)` 派生。

扰动幅度在 pilot 冻结。首轮默认不加入隐藏 rank 偏斜；如某场景需要成员错位，写入显式 rank 工作量表并记录，不混用未记录 sleep。

### 6.2 配对块与运行顺序

一个 block 为 `(scenario, workload_seed, repeat_index)`。同 block 七个 arm 使用完全相同的输入、tensor seed、profile 与估计。正式 repeat_index 为 0–4。

运行前生成完整顺序表：随机交错场景/seed/repeat 的 block 顺序，并用 order seed 生成尽量均衡各位置的 arm 排列。不要连续跑完一个 arm 的所有样本后才跑另一个，以免温度、频率和背景负载漂移偏向某配置。

所有 replay 串行，每次独立启动 rank 进程，GPU 上不得残留本轮上一 replay 的工作。其他用户进程不擅自终止，背景负载与设备状态记录；出现明显干扰按预先定义的规则暂停，不按策略输赢决定剔除。

不同 repeats 的输入保持相同；若 CLI 的 epoch 会影响随机输入派生，必须固定 workload 的 sample epoch，或拆开 run identity 与数据 seed。不能把 repeat 编号无意编码进输入数据后声称固定输入重复。

### 6.3 估计与扰动的隔离

先对基准程序独立 profiling，得到共享估计。用于预测误差研究的随机执行扰动只进入实际程序，策略不读取扰动后的未来耗时或未来工作量摘要。

运行器为配对核验保存实际工作量完整 manifest，但 adapter 向 policy 提供的仅是冻结估计视图。若另做“已知工作量、未知运行干扰”实验，应另标合同，不能混合两种信息条件。

## 7. 分阶段执行与预算

| 阶段 | 内容 | 数量与进入条件 |
| --- | --- | --- |
| E0 | 环境、H 回归、GPU workload 与七 arm 实现、bare 和 DAG 适配 | 按测试用例计，不计性能 repeats |
| E1 | 目标环境计算/通信 profile、warmup 与运行墙钟 pilot | 覆盖实际签名；建议起始 warmup 5、测量 30，稳定后冻结 |
| E2 | 六场景七 arm 的机制 pilot | 每场景一个独立 pilot seed × 3 repeats，即每 revision 最多 126 replay |
| E3 | 七配置正式线性矩阵 | `2 × 7 × 5 × 5 = 350` replay |
| E4 | 七配置正式 DAG 矩阵 | `4 × 7 × 5 × 5 = 700` replay |
| E5 | 有问题指向的 profiler/机制诊断 | 单独登记最小规模，不进入 1,050 正式样本 |

E0 未完成不能直接进入正式性能阶段；特别是 LTF 未接入真实 GPU 估计、bare 或旧 DAG 未验收时，不以少跑 arm 替代七配置目标。

正式运行之前用实际 smoke 的完整进程墙钟中位数/P90 估算总预算：启动、建 group、预分配、warmup、应用、validation、drain 均计入墙钟。记录预计磁盘用量、失败上限与超时。

采用新目录分批运行，支持按 manifest 恢复。恢复检查 input/profile/source/contract/raw hash，不扫描目录寻找“看起来成功”的文件凑样本。正式期间源码改变则建立新 revision，不合并不兼容样本。

## 8. 正确性验收与停止规则

每个 arm 都要检查实际 launch 投影、通信任务全集、每 group 成员覆盖、producer/independent/dependent 数值及 GPU 完成边界，不能只看进程退出码或 grant 日志。bare 没有 grant 时，检查其规范顺序合同与成员实际序列。

共同检查包括：

- 初始化 stream 到首次使用的依赖、producer 完成后通信、consumer 等全部必需输入。
- 完成探测不依赖应用调用 wait；SUBMITTED 先于 COMPLETED。
- 静态 Plan 合法且不动态跳过队首，动态策略不越 group 顺序。
- 独立计算不因通信等待绑定而停止；DAG 同一节点不重复执行，多终点全部完成。
- 重复 epoch 的 buffer 重置、无残余工作及资源生命周期。
- 缺失任务、元数据冲突、launch/probe/compute query 失败、网络断连的停发与有界退出。

数值校验在应用终点后执行，但真实 dependent compute 必须在应用内读取通信结果，不能靠 validation 隐式同步补依赖。matmul producer 的 all-reduce 参考不能继续沿用 rank 常量求和公式。

出现 tensor 错误、投影不一致、提前完成、不可回收挂起或错误 GPU 映射，立即停止相关性能批次，保留全部产物，修复后新版本重跑。性能无提升、CI 跨 1 或没有 kernel overlap 不属于正确性停止条件。

### 8.1 失败与重试

所有 attempted run 写入不可覆盖 ledger，记录退出码、异常、超时、raw 路径和 block/attempt。不能按结果慢或策略输赢重跑。

预定义可重试环境错误（例如确认的 rendezvous 端口冲突）最多重试整个七 arm block 一次；原 attempt 全部保留，成功替代 block 单独标识。第二次仍失败则暂停该场景排查，不反复运行到全绿。

成功完整 block 用于配对性能统计，同时报告全部尝试的失败率、被替代块及缺失块。程序正确性错误不适用环境重试规则；也不能仅用成功块宣称系统无失败。

## 9. 指标、统计与解释

### 9.1 主要指标

| 指标 | 定义 |
| --- | --- |
| application makespan | 每 rank 本地 release→所有 terminal 完成观察的 duration，再取 rank 最大值 |
| job JCT | 同一 release→本 job 全部 terminal 完成；成员 rank duration 取最大 |
| mean JCT | 先在单次 replay 内对 job JCT 求平均，再跨运行汇总 |
| failure rate | 按 arm/场景报告全部 attempted runs 的失败与原因 |
| 全程序墙钟 | 父进程从启动到子进程回收，另列 setup/应用/drain 等阶段 |

补充报告各 arm 的 CPU 时间、实际在途峰值、显存、通信/计算设备区间及工作量摘要。没有同环境 isolated 分母时不报告规范化 slowdown。

profiler 只作因果诊断；同设备 kernel 区间求交才是设备 overlap，CUDA event 区间不能自动当 kernel 活动时间。host、coordinator 与 GPU 时钟域分别分析，不未经校准跨域相减，也不把分段中位数相加解释总差值。

### 9.2 配对估计量

对 baseline B 与 candidate C：

```text
每 repeat：delta = T_C - T_B；ratio = T_B / T_C
每 seed：取五个 paired delta / ratio 各自的中位数
场景汇总：取五个 seed-block 中位数的中位数
```

delta < 0、ratio > 1 表示 candidate 较快。另列每 arm 的原始 duration 分布；arm 中位数之差与 paired delta 中位数是不同估计量，不混用。

报告五个 seed 的配对结果、25 个 repeat 点、median/range、失败与缺失计数。95% bootstrap 按 seed block 重采样，建议固定分析 seed、2,000 次；五个 seed 仍属有限探索，不因 25 次测量就宣称充分覆盖所有 workload。

正式 seed 的机制不触发时也保留样本，并报告 HOL/多候选实际触发率，不能筛掉“没发生动态优势”的点。实际意义阈值在主批次前结合噪声与目标预算冻结；未给定阈值时只报告效应量，不自封某个百分比具有应用价值。

不循环追加 seeds 直到显著。若结果不确定，需要新一轮预先登记的确认批次，初始与新增结果分开呈现。大量次要比较不作为单个未经多重比较说明的显著性结论。

## 10. 产物与可复现执行

拟议路径，实施时创建：

```text
benchmark/phase3/experiments/gpu-seven-arm/
  linear/L0/ linear/L1/
  dag/D0/ dag/D1/ dag/D2/ dag/D3/
  suites/                         # 冻结后的矩阵、seed、顺序和合同
benchmark/phase3/results/gpu-seven-arm/<batch-id>/
  inputs/ source/ profiles/ raw/ logs/ tables/ figures/
  manifest.json runs.jsonl analysis.md
```

batch manifest 至少包含：七 arm 实际实现与语义（含第七臂的计划修订及资格证据；明确 original bare 未支持）、静态 Plan/估计版本、环境与 GPU UUID、源码归档及 hash、输入与 profile hash、完整运行顺序、seed 和 PRNG 规则、warmup、poll、timeout、目标数量和阶段 gate 结果。

每 run 保存完整命令、stdout/stderr、耗时、退出码、run/block/attempt ID、实际 contract、验证摘要、原始任务数据及 hash。源码不能只记 dirty HEAD；要包含未提交实现和未跟踪依赖文件。

脚本放在 `examples/jobpacer/scripts/`，复用现有 batch ledger、命令与分析基础设施；必要时增加一个薄的 suite 编排入口，不在 benchmark 内复制实验逻辑。未实现的 CLI 不写成可执行命令，实施后在 process 文档给出经过 smoke 验证的完整命令。

结果目录被 Git 忽略，独立备份 source/inputs/profile/raw/manifest/logs。归档校验时区分“顶层 SHA 清单通过”和“逐 raw 全量校验通过”。

## 11. 交付与完成标准

- 七配置在六个场景上使用明确且可解释的 GPU 执行合同；本版 `raw-ordered-static-fifo` 替代及容量差异明确标识，原始 bare 不得宣称已测。
- 两个线性场景、四个 DAG 场景均有五个可复现工作量样例，各 arm 每样例五次完整正式测量。
- 1,050 次有效正式 replay 的配对完整性与全部尝试 ledger 一致；缺失时明确批次未完成，不以均值填补。
- 新旧静态 Plan 序列一致，GPU LTF 估计版本与实际程序匹配，历史旧算法保持可追溯。
- 正常/异常、数据与顺序、设备完成边界分别验收，正确性结果和性能结果分别报告。
- 报告包括全部场景的原始点、主要配对结果、逐 job JCT、失败与重试、样例限制和未支持范围。
- 无收益也是有效实验结果；没有重叠不自动归因 runtime 缺陷；不把有限场景结论推广为 GPU 普遍规律。

实施过程写在 `docs/JobPacer/process/`，最终验收写在 `docs/JobPacer/result/`。本计划中的 DAG/旧 scheduler 适配、替代 arm 的 GPU 资格与 GPU LTF 工作优先于启动正式矩阵；此前设计先 FIFO 后 LTF 的实施顺序仍成立，但本轮最终交付包含全部七配置（原始 bare 除外）。
