# GPU workload 统一 DAG：具体修正实施计划

日期：2026-09-28。状态：R1–R5 主体实现及 D0 语义 smoke 已完成；独立 GPU-linear/bridge 执行循环已退役；R6 与完整 G5/G7 验收及正式实验仍未完成。本文最初作为实施合同编写，现追加实际落地记录；下述计划要求与已验证事实分开阅读。

设计依据：[修订后的 GPU workload 计划](../plan/phase3-gpu-workload-and-scheduling.md)与[七配置实验计划](../plan/phase3-gpu-seven-arm-experiments.md)。[此前逐文件实施说明](phase3-gpu-workload-implementation.md)保留，涉及新增 linear runner、bridge runner 的后续路线由本文取代；前轮验证事实仍见[结果记录](../result/phase3-gpu-workload-implementation-20260927.md)。

## 1. 修正目标与约束

本轮不是重写 GPU 功能，而是把已经实现的固定工作量、数据校验、profile、事件回执接回现有 JobPacer DAG 和实验接口。

最终主路径只有：`load_dag → DagInput → 共同资源准备 → DagRunner → 通信 adapter → 共同汇总/实验 runner`。线性 job 用链式 DAG 输入表达；不再新增第四套图模型或执行循环。

保留旧 `Workload` 与 Phase 1/2 历史入口，不强制它们全部内部迁移。七配置中的旧 scheduler 是执行引擎适配，不是让新 DAG 退回旧线性 Workload。新 runtime 核心不得依赖旧 Plan；所有跨引擎转换位于 examples 层。

## 2. 当前改动的保留与收敛清单

| 文件/对象 | 处理方式 | 完成判据 |
| --- | --- | --- |
| `workloads.py` 的 Workload/Job/CollectiveComm | 保留历史接口和采样规则，不再为新 GPU 主路径扩字段 | 原输入与相关回归仍可运行 |
| `runtime_adapter.py` 的 DagInput/ReplayExecutionConfig | 扩展现有结构、load_dag/parse_dag，统一规范化和摘要 | 新旧 DAG 格式都返回同一 DagInput |
| `gpu_workload.py` | 保留具体 GPU 参数/估计元数据解析；去除主路径对 GpuLinearInput/GpuSegment 的依赖 | 无第二份 jobs/groups/graph 状态；节点 GPU 绑定可复用 |
| `gpu_compute.py` | 保留底层算子、init 依赖、receipt、验证；解耦 GpuSegmentResources | 程序由节点描述及 buffer 表构造，不要求固定 producer/comm/join 片段 |
| `gpu_compute_profile.py`、`run_gpu_compute_profile.py` | 保留 profile 能力，输入改为枚举 DAG 节点实际签名 | 通用节点可校准；旧 profile 显式兼容或报版本不匹配 |
| `src/.../dag/model.py` | 复用图校验、tail、静态序列；必要的提交门控校验另有明确边界 | 不重复实现拓扑和评分算法 |
| `src/.../dag/runner.py` | 唯一 DAG 推进器，增加最小提交门控和通信接口约束 | CUDA receipt、完成依赖、失败、deadline 共用一套状态机 |
| `gpu_linear_runner.py` | 新路径不再调用；迁移完测试后退役重复执行逻辑 | 不再维护独立线性 GPU 状态推进 |
| `gpu_dag_bridge.py` | 可保留离线转换校验；退役 run_gpu_bridge_job 循环 | bridge 不再是 worker 执行模式 |
| `gpu_linear_worker.py` | 可复用的准备/验证逻辑搬回共同 harness 或小助手；不保留独立 rank 生命周期 | 新 GPU 都由共同 worker 启动、关闭和汇总 |
| `run_phase3.py` | 使用 --dag 主入口，收敛 _run_gpu_linear 特殊汇总 | 七 arm 可比较的输出 schema 一致 |
| `run_experiments.py` | 扩展既有 arm 与 DAG 命令构造 | 不建立新 ledger、配对算法和结果目录扫描逻辑 |

“退役”分阶段进行：先新增并验证共同路径，再撤除主入口引用，最后清理重复实现。输入、raw、历史文档不删除；如保留兼容 CLI，只允许薄转换后进入同一 runner，不继续维持旧新两套状态机。

## 3. R0：基线盘点与迁移清单

1. 保存当前 dirty diff、未跟踪源码及相关输入列表，不覆盖其他工作。现有 GPU 新文件和 DAG 相关修改可能尚未进入 Git 索引；迁移前逐项确认，不通过 checkout、clean 或批量覆盖重置它们。
2. 阅读前轮结果，确认已修复 H 问题和仍未归因的 Phase 2 超时状态；必要的复测单独记录，不自动写“已消除”。
3. 建立 import/caller 清单，覆盖 GPU schema、linear/bridge runner、profile、CLI、测试和结果分析消费者。
4. 选取前轮 GPU fill/matmul/null independent/null consumer 样例，列出预期任务、算子、数据和末端，不用旧性能数字作为迁移判据。
5. 明确旧接口仍支持的范围及 GPU 主路径待退役入口，防止通过全局替换破坏历史回归。

交付：实施记录中的文件迁移表和基线检查结果。此阶段不重跑性能矩阵。

## 4. R1：统一输入与 execution 配置

### 4.1 runtime_adapter.py

- 在既有 DAG loader 中增加 v2 分派；v1 保持原字段和语义，均返回 DagInput。
- 扩展 ReplayExecutionConfig：compute_programs、buffers、comm_bindings、submit_after、sample_id、compute_model、profiles。Python 层可拆小数据类，但不新建 GPU graph/job 类。
- GPU parser 不再要求 compute_duration_s 存在，也不能用“默认 0 秒”掩盖缺失程序。每个 ComputeNode 必须有恰好一个程序绑定，每个 CommNode 有恰好一个数据绑定；拒绝多余条目。
- 统一 canonical JSON/hash 包含图、执行合同、程序、数据 seed 和 profile 引用。另保存实际扰动工作量摘要与策略估计视图摘要，避免泄漏未来扰动。输入身份、执行样本、估计视图分别计算摘要；不得用包含 run/epoch 的输出文件 hash 充当配对输入 hash。
- apply_dag_profile 更新图估计时保留全部 execution 元数据，不只重建旧 compute_duration_s。
- make_replay_compute 按 mode 选择 sleep 或 CUDA 程序。当前读取 base_s 在 GPU 分支之前的做法须改为分支内部读取，避免 GPU 仍被迫伪造 sleep 数据。

### 4.2 GPU 描述与数据校验

复用 gpu_workload.py 的数值、shape、算子校验，重构为“节点程序 + job 本地 buffer 引用”。先支持 fill/matmul/sum_join，不增加插件注册系统。

验证数据引用与完成依赖：读取某次运行时写入结果必须有对应 writer 的完成路径；原地 all-reduce 是 writer；未排序写写/读写和未知别名拒绝。只做首版明确支持的 whole-buffer 所有权，复杂 view/alias 暂不支持。

`submit_after[compute] = [comm, ...]` 表示 compute 要等这些 comm 的 `submit()` 成功返回 handle；首版只允许同 job 引用。每个被引用 comm 的**传递完成前驱**必须包含在 compute 的传递完成前驱内，且 comm 本身不得是 compute 的完成前驱；据此拒绝等待未来节点与隐含循环。仅比较直接 `deps` 不足以覆盖传递依赖。图/group 联合校验继续复用现有函数。

测试：v1 兼容、v2 规范化、缺失/额外映射、错误 shape/repeats、buffer hazard、非法门控、数据 seed 与 repeat 分离。示例 JSON 语法通过不等于 parser 验收。

## 5. R2：解耦 GPU 程序与资源所有权

把 GpuSegmentResources 的可复用内容组织为 job 本地 buffer 表和按 node ID 索引的具体程序。无需引入资源管理服务；普通数据对象及现有 CUDA receipt 足够。

- release 前分配/初始化，记录 init event；各使用 stream 显式等待初始化。
- 程序 submit 只排固定次数工作，返回 completion receipt；query/elapsed/keepalive 契约沿用。
- producer 输出、通信原地输出和消费输入通过 buffer 表连接，不能再靠“当前 segment”寻找 tensor。
- 计算与通信引用保留到所有消费者安全结束；首版可让 job 资源活到该 job/epoch 完成，避免过早追求内存复用。
- 精确算子 warmup 后重置正式 buffer；数值参考放在应用计时外。matmul producer 使用对应参考，不套 rank 常量公式。
- 保留旧 CudaMatmulProgram/legacy S 所需接口，迁移时不改写历史执行合同。
- profile 按实际算子和 buffer signature 查找；角色/节点名变化不能隐式导致 profile 用错。更改签名格式则升级版本并显式转换或拒绝。

测试：非默认 init stream、warmup=0/非零、输出接通信、消费读结果、null 算术阶段、多终点、重复 epoch 与 query 异常。真实 GPU 测试使用有界工作，先单程序再双 rank。

## 6. R3：只扩展既有 DagRunner

### 6.1 计算与完成模型

继续使用一个 compute worker 和一个 active compute receipt/job。future 返回 receipt 不算完成，receipt.is_completed 成功才释放计算通道和完成边。不同 job 可独立推进，通信不占该计算通道。

本修订选择物理完成边：consumer 节点等通信和独立计算都完成才入队，不复刻旧 bridge 的提前 consumer stream 排队。必须在输出中记录 `dependency_mode=physical-completion`、`compute_model=one-active-compute-per-job`，避免误认为全 stream DAG。

不在本轮把单计算通道偷偷改成无限并行。需要并发计算时后续统一扩展，所有 arm 同时采用。

### 6.2 submit_after 门控

在 runner 构造参数中接收经校验的普通 node ID 映射，不依赖 examples 的 GPU 类型。ready compute 的启动增加“相关通信已成功返回 handle”条件；该门控不改变 DAG 的完成计数。

事件测试必须阻塞 grant/通信完成，确认 independent 在 submit 返回后仍启动；同时覆盖通信 submit 失败时不启动受门控节点。不要靠遍历通信在计算之前的偶然顺序替代显式检查。

### 6.3 通信 adapter 契约

盘点 runner 实际使用的接口：submit、failure、abort、handle 的状态/非阻塞物理完成查询；Lookahead 另需 declare。用最小 Protocol 或现有鸭子类型标注，不扩成插件框架。worker 负责注册 group、start/finish/close，避免把生命周期回调散入节点逻辑。

新 RankRuntime adapter 首先接通；bare/旧适配后续复用同一 runner。核心模块不 import examples 或旧 Plan。

测试：与现有 CPU DAG 回归一起覆盖 ready 顺序、GPU receipt、通信等待期间的计算推进、失败、固定 deadline、全部 sink 完成。不要复制原测试仅换类名；保留能复现错误交错的断言。

## 7. R4：共同 worker、输入和输出

在 runtime_worker.py 的现有 DAG 路径扩展资源准备与 make_binding，使用规范 CommNode 对应的 buffer。共用 group 创建、`_run_jobs`、异常处理、应用时间边界和结果结构；**控制通道只属于新 runtime adapter**，bare 与旧 scheduler 走各自的通信初始化和关闭流程。共同 worker 不是让三种引擎共用新 coordinator。

将 gpu_linear_worker.py 中必要的设备信息、profile 校验、deferred validation 和准备计时搬入共同流程或小助手；不要整文件复制另一份。公共助手按实际重复点提取，避免提前通用化所有 lifecycle。正常结束固定为：所有本地 job 的终点完成 → 记录应用终点 → 引擎关闭输入并完成通信 drain → 记录 drain 终点 → 校验数值和结果 → 关闭本地资源/输出。校验不得被用来补应用内缺失的设备依赖；对于仅在校验中需要同步的对象，其同步时间单列且不回写应用时长。失败时停止新提交、通知同 rank 其他 job、调用对应引擎的 abort/关闭一次，并按共同 deadline 有界回收；已经提交的 CUDA/NCCL 工作仍按真实状态记录。

run_phase3.py 主入口使用 --dag；新 v2 与旧全局 compute-matrix-size/repeats 冲突时明确报错。gpu-linear/gpu-bridge 的过渡入口若保留，离线/入口转换成 DagInput 后走同一路径，不再调用专用 runner。

输出复用既有 validation/performance/ranks 结构，新增程序/数据摘要、完成合同和估计版本。既有分析不理解新字段时显式升级解析，不另建仅 GPU linear 能读的汇总。

新增一个链式 DAG、一个真正非片段式分叉/多前沿 DAG 做 smoke。后者必须无法仅靠串接 GpuSegment 表达，防止名为 DAG、实际仍依赖 segment 的实现通过验收。

## 8. R5：旧 scheduler、bare 与共同静态计划

七配置是本轮目标，不将旧/bare 写成“以后可能需要”。接入位置为 examples 层的薄通信 adapter，可用一个具体模块保存三个实现，暂不做注册框架。

### 8.1 旧执行引擎

- 用 dag model 的 build_static_order 构造 FIFO/LTF 规范序列；旧 Plan 与新 StaticOrder 映射回规范 ID 后必须相同。
- 规范 task/group 身份确定性映射到旧 TaskKey，不能假定 group_id==job_id 或每 job 只有一个 group。
- 复用旧 AdmissionScheduler 的发射能力；通过 adapter 提供满足 runner 契约的物理完成回执、错误和本地资源保活。
- 旧 wait 的 CPU 返回不当作 GPU 完成，使用已验收的 CUDA 回执桥接。计算不迁移到旧 replay 的线性循环。
- 保持旧历史 Plan builder 不变，本轮输出明确共享 GPU Plan 来源。

### 8.2 bare

独立确认多 communicator 的共同发射/设备顺序合同，提交接口不能阻塞其他 ready 计算。真实 backend 安全要求未满足时，不运行七 arm 正式矩阵。

若只有 raw-ordered 可支持，记录共同顺序、协调/同步成本和未支持原始 bare 的缺口；不把新 coordinator 改名 bare。该选择在正式 manifest 冻结前明确，不默认为本次文档已批准替代。

### 8.3 共同策略估计

复用 compute_tails 与 build_static_order。DAG tail 是当前节点完成后的最长后继路径，LTF 评分在静态、动态及 Lookahead 前沿统一为 `estimated_comm_s + remaining_tail_s`；线性 tail 的构造仍保留其 overlap 估计。DAG tail 仍忽略 sibling 残余、计算通道排队和争用。GPU profile 更新节点估计；固定执行扰动不泄漏给 policy。评分公式固定在策略中，不进入逐任务 hint。

## 9. R6：复用批次框架与完成六场景接入

run_experiments.py 扩展 --dag 的 old/bare arm 支持及统一命令构造；原来的“old arms require linear workload”限制只在旧入口继续需要时保留，不能挡住新的共同 DAG 模式。

复用 ARM_SPECS、order seed、run ledger、配对统计、失败归属与恢复校验。静态 Plan hash、执行工作量 hash、估计视图 hash、contract 及 sample ID 纳入每个 block 的一致性检查。

构造 L0/L1 两个链式 DAG 和 D0–D3 四个一般 DAG；五个工作量样例、每样例五次重复的具体矩阵不变。重复 run 的协议 epoch 变化不能改变 tensor seed 或工作量；生成器固定 sample_id，保存展开后的实际 manifest。输入根字段 `seed` 明确定义为 **workload seed**。`tensor seed` 由版本化稳定哈希从 `(scenario, sample_id, workload seed, job_id, buffer_id 或 node_id, rank, stage)` 派生，不单独由 CLI 随机生成；`order seed` 只在 batch runner 排列 block/arm 顺序，不进入 DAG 输入。`repeat_index`、`run_id`、协议 `epoch` 只标识执行，不参与工作量或 tensor 值派生。每个 sample 在启动任一 arm 前先展开并冻结实际 repeats、输入种子及共同估计视图；七 arm 和同 sample 的五次 repeat 读取同一份展开 manifest。

先用小规模命令检查七 arm 均可启动、结果可由同一分析解析，再按七配置计划执行机制 pilot。不能只接通 new FIFO 就启动 1,050 次矩阵。

## 10. R7：兼容收尾与退役重复代码

共同路径通过之后，逐项处理 gpu_linear_runner、run_gpu_bridge_job、专用 worker 和 _run_gpu_linear 汇总分支：

1. 将有价值的初始化、错误、数值和时序测试迁移到共同节点程序/runner/adapter 测试。
2. 保留需要的输入转换测试，不保留重复状态机作为转换工具。
3. 旧入口要么薄转换并给出合同迁移说明，要么明确标记为归档复现入口；不能继续作为七配置主路径。
4. 更新实际使用的 README/命令和新 process/result 记录，历史结果保持原文并关联修订说明。
5. 通过 import/caller 检查确认没有内部依赖仍指向待移除代码，再清理重复实现。历史 manifest、raw、归档源码不删除。

2026-09-28 已执行 R7 的重复执行路径清理：删除专用 runner、worker 和 bridge 推进状态机；移除旧 replay CLI；将 compute/communication profile 命令收敛到 schema-v2 DAG。历史输入与历史结果保留，旧代码可从对应源码归档复现。清理后的验证记录见第 18 节。

## 11. 分层验证与产物

| 阶段 | 最小验证与证据 |
| --- | --- |
| R1 | v1/v2 parser、schema/数据依赖/门控负例、规范摘要与随机键回归 |
| R2 | GPU 节点单测及真实算子数值/事件；profile 签名匹配 |
| R3–R4 | 原 DAG/Gloo 回归、事件控制交错、双卡链式和一般 DAG 数值/顺序/末端 |
| R5 | 旧/bare/new 相同图与工作量、静态 Plan 对齐、容量与完成差异、正常/失败路径 |
| R6 | batch 小 smoke、配对 hash 校验、失败保留与恢复；六场景七 arm 支持矩阵 |
| R7 | 旧入口回归、全仓测试与 skips、无悬空引用、文档命令可运行 |

通用验证命令从仓库根目录执行，按改动阶段选择，不擅自安装或升级环境：

```bash
PYTHONPATH=src .venv/bin/python -m pytest -q tests/unit/runtime tests/unit/test_jobpacer_dag.py
PYTHONPATH=src .venv/bin/python -m pytest -q
PYTHONPATH=src RUN_JOBPACER_RUNTIME_REPLAY=1 .venv/bin/python -m pytest -q tests/integration/test_runtime_replay.py
PYTHONPATH=src RUN_JOBPACER_RUNTIME_NCCL=1 .venv/bin/python -m pytest -q tests/integration/test_runtime_replay_nccl.py
git diff --check
```

新建测试和 smoke 命令在实现后补入实施记录；不能认为上述现有测试已经覆盖新合同。Socket 权限限制按工具流程处理，不改代码绕过；GPU 不可用不以 CPU 通过替代。

每阶段记录源码/输入/profile hash、环境、命令、通过/失败/跳过、时长和产物路径。性能结果需新 batch；统一物理完成 DAG 与前轮 S/bridge 的执行合同不同，不合并原始性能样本。

完成标志：新 GPU 主路径不依赖 GpuLinearInput 和专用 bridge 执行循环；六类场景均由同一 DagRunner 推进，七配置仅替换通信执行/选择；历史工作受保护。正式矩阵的执行完成是后续实验验收，不由本修正方案自动宣称。

## 12. 逐文件落地边界与依赖顺序

以下是对 R0–R7 的具体化。路径均相对于仓库根目录。每一步先建立当前行为的回归，再修改调用链；每个阶段结束时记录实际改动文件与预期表的偏差。

| 阶段 | 文件 | 具体修改 | 不应放入该文件的职责 |
| --- | --- | --- | --- |
| R1 | `examples/jobpacer/runtime/runtime_adapter.py` | 扩展 `ReplayExecutionConfig`、`parse_dag`、canonical 文档、`apply_dag_profile`、`make_replay_compute`；v1/v2 都返回 `DagInput` | CUDA tensor/ProcessGroup、中心状态机 |
| R1 | `examples/jobpacer/runtime/gpu_workload.py` | 提取可复用的算子规格、参数检查、稳定 seed 派生；旧 `GpuLinearInput` 暂留给历史入口 | 第二份 DAG 拓扑或 group 顺序 |
| R1 | `src/runtime_comm_scheduler/dag/model.py` | 只在现有图/group 校验与静态排序确有缺口时修改；复用 `compute_tails` 和规范任务 ID | GPU 程序、buffer、`submit_after` 的 examples 执行配置 |
| R2 | `examples/jobpacer/runtime/gpu_compute.py` | 增加 job 本地 buffer 所有者和 node 程序构造/回执；保留旧 `CudaMatmulProgram` 与 S lane 可调用接口 | admission、DAG ready 状态 |
| R2 | `examples/jobpacer/runtime/gpu_compute_profile.py`、`examples/jobpacer/scripts/run_gpu_compute_profile.py` | 用实际 node 算子签名枚举与校准，核对设备/软件/精度和版本 | 从本次真实扰动反推策略估计 |
| R3 | `src/runtime_comm_scheduler/dag/runner.py` | 在已有完成依赖推进上加 `submit_after` 检查；将 runtime/handle 使用收敛到最小可满足接口 | 三种引擎的生命周期、具体 GPU 数据 |
| R4 | `examples/jobpacer/runtime/runtime_worker.py` | 共同资源准备、按 `CommNode` 绑定 buffer、共同 job 执行和阶段计时；新 runtime 生命周期仍在新分支内 | bare/旧引擎借用新 coordinator |
| R4 | `examples/jobpacer/scripts/run_phase3.py` | `--dag` 接 v2、冲突参数报错、共用汇总/validation；保留 v1 和历史 CLI | 第二份 GPU linear 汇总协议 |
| R5 | 拟新增 `examples/jobpacer/runtime/dag_comm_adapters.py`，及必要的 worker 启动分支 | 分别实现 bare、旧 scheduler、新 runtime 到 runner 的 submit/handle/failure/abort 契约；引擎初始化明确分支 | 反向修改新 runtime 核心以适配旧 Plan |
| R6 | `examples/jobpacer/scripts/run_experiments.py`、六场景输入生成/manifest | 七 arm 命令、配对 block、hash/ledger、source snapshot、恢复与结果解析 | 独立于原 batch 框架的第二套账本 |
| R7 | `gpu_linear_runner.py`、`gpu_dag_bridge.py`、`gpu_linear_worker.py` 与对应测试/调用方 | 在共同路径验收后撤除专用运行循环；需要的离线转换和旧合同复现入口明确保留 | 提前删除历史源码/输入/结果 |

实现顺序不能把 R5 的旧/bare 适配挪到 R3 前面：先让新 runtime 通过 v2 节点数据流，才能判断 adapter 差异。R6 的正式命令构造可提前开发，但七 arm 语义 gate 未过前只能跑 smoke，不能写成功批次 manifest。

## 13. 输入、随机性和数据流的可执行合同

### 13.1 v1/v2 解析与规范化

`schema_version=1` 保持 `execution.compute_duration_s` 和可选 `linear_sample_keys` 的现有含义；现有 v1 canonical JSON/hash 的行为不因 v2 新字段改变。`schema_version=2` 使用设计稿的 `execution.mode=cuda-program`、`compute_programs`、`buffers`、`comm_bindings`、`submit_after`、`compute_model`、`sample_id` 和 `profiles`。首版 v2 只接受 `cuda-program`；如果需要 v2 sleep，应单列用途并给出解析检查，不能用缺省 sleep 值自动兜底。`estimated_duration_s` 与 `estimated_comm_s` 仍在图节点上，执行程序不能覆盖这些估计。

解析流程固定为：检查根版本和字段集合 → 构造规范 `DagGraph` 并 `validate_graph` → 按全局 `job_id/node_id` 核对 compute/comm 绑定全集 → 校验 buffer、程序和提交门控 → 构造稳定 canonical 文档与摘要。解析只读 JSON 值，不分配 CUDA 资源。错误应指出具体 job/node/buffer；未知字段、漏绑定、重复绑定和未使用绑定均报错。`sample_id` 只标识冻结样本，不参与运行次数或 epoch 选择。

当前 `apply_dag_profile` 通过旧 `compute_duration_s` 重建 canonical 文档；改动后必须从完整 execution 配置重建，并区分原始输入 hash、应用 profile 后的估计视图 hash。对于同一冻结样本，所有 arm 看到完全相同的 graph 估计、profile 版本和 tail；实际 repeats 只能从执行 manifest 读取，不写回策略可见 `TaskHint`。

### 13.2 三种 seed 与 manifest

每个 `(scenario, workload seed)` 先生成一次规范 v2 DAG 与展开后的执行 manifest；整数 repeats 扰动在此时一次性确定并落盘，正式 run 不再抽样。`seed` 是 workload seed，控制 repeats；tensor 值由固定版本的哈希与 PRNG 从该 seed、`sample_id`、job、buffer/node、rank、stage 派生。派生算法、整数映射、PRNG 名称/版本和浮点初始化范围要写入 manifest；不能使用 Python 进程内置 `hash()`。rank 在 tensor seed 中保留，因为 all-reduce 的各成员输入可以不同；同一 rank、同一 sample、同一程序在七 arm/五次 repeat 中必须完全相同。

order seed 仅控制 `(scenario, seed, repeat)` block 和其中 arm 的执行顺序。`epoch` 继续用于通信身份、group/TaskSpec 和控制协议；它不传入 v2 tensor/工作量派生。一个配对 block 至少记录 `input_hash`、`execution_sample_hash`、`estimate_view_hash`、`profile_hash`、`source_hash`、`contract_version`。上述五项在同 block 七 arm 间必须相等；run/epoch/order 信息另列，不参与配对相等性检查。若 profile 或源码改变，建立新 revision，不混入旧 block。

### 13.3 buffer 所有权与校验算法

首版 buffer 键是 `(job_id, buffer_id)`，每个 rank 有自己的 tensor 实例；`comm_bindings` 指向本 rank 的实例，不能把 tensor 放在 JSON 或中央控制消息中。buffer 规格固定 shape/dtype/init，通信绑定须与 `CollectiveSpec` 的 shape、numel、bytes、dtype 一致。只允许 whole-buffer 读写，拒绝 view、部分更新、隐式别名和不同 job 共享；后续如需复用存储，另定义版本。

对每个 buffer 建立操作访问集合：初始化是 release 前 writer；`fill`/`matmul`/`sum_join` 按实际输入输出列出 reader/writer；all-reduce 是同一节点上的读写操作。每个运行时 reader 必须能从其**传递完成前驱**中定位最近且唯一的 writer；未排序的两个 writer、reader 与 writer、或仍可能被消费的旧值被覆盖，都在加载时拒绝。对于 comm 原地更新，后继读取归因于 comm writer，不能仅依赖 producer。校验使用完成依赖的可达关系，不把 JSON 数组顺序或 `submit_after` 当作数据完成边。初始化 buffer 可以被无前驱的节点读取；`empty` buffer 在首个 writer 完成前不可读取。

程序 submit 前检查输入/输出 tensor 形状与设备；数值 reference 在应用 release 前由同一冻结输入建立。warmup 使用相同签名，但正式运行前重置所有会被写入的 buffer，并使 init/reset event 对 compute stream 和 collective stream 都可见。`sum_join` 必须实际读取通信输出与独立分支输出；末端验证至少包含通信结果、独立输出和 dependent 输出，不能只查 all-reduce 常量和。

## 14. DagRunner 与通信 adapter 的最小接口

### 14.1 runner 内部状态与门控

现有 `DagRunner.run()` 每轮先收计算/通信完成，再提交 ready comm，最后选 ready compute；增加门控后仍保留这个顺序，但正确性不能依赖它。构造时传入已校验的 `submit_after` 子映射。对每个 ready compute，只有其 `remaining_deps==0`、所列 comm 全部在 `handles` 且对应 `submit()` 已成功返回、该 job 计算槽空闲时才能入队。被门控的 compute 仍保持 READY，不能提前标记 RUNNING；grant 或物理完成均不作为门控条件。没有门控的 compute 按原 node 顺序选择。

`submit()` 抛错时不记录 handle，不解锁任何门控节点，沿现有失败传播终止本 job；其它 job 由 worker 的 stop/abort 协调。compute future 返回 CUDA receipt 只表示已排队；只有 `receipt.is_completed()` 成功才 `_complete()` 并释放计算槽。comm handle 的 `wait_host(0)` 必须代表设备物理完成的 host 证明，而非底层 `Work.wait()` 的普通 CPU 返回。固定 deadline 在每轮检查，事件唤醒可重算但不得刷新预算。提交、完成和失败的 event 均带规范 `job_id/node_id/task_id`，便于检查重复与顺序。

### 14.2 adapter 能力与生命周期

runner 的最小依赖是 `submit(spec, binding, hint) -> handle`、可读取的 `failure`、`abort(exc, ...)`；仅 Lookahead 使用 `declare(spec, hint)`。handle 至少提供 `wait_host(0)`、可用于诊断的 `state`，并持有本地 binding/Work 直到物理完成。新 adapter 可以直接由 `RankRuntime` 满足这些能力。定义 Protocol 时只描述这些成员，不把旧 `TaskKey` 或 CUDA 类型塞入 `src/dag`。如果 bare/旧 arm 不支持 `declare`，它们的 worker 必须拒绝 Lookahead；七臂不包含 Lookahead。

三种引擎的启动与结束分开列成明确分支：

| adapter | release 前 | `submit()` 返回条件 | 物理完成证明 | 正常 drain |
| --- | --- | --- | --- | --- |
| new | 创建 group、启动 coordinator/client/RankRuntime，注册本 rank 使用的 group | RankRuntime 保存 OFFER 并返回 handle；不等 grant | 已验收的 CUDA 完成探测与 handle 状态 | `finish_epoch(remaining_deadline)` 后关闭控制资源 |
| old | 由共同静态序列构造旧 Plan/TaskKey，启动旧 scheduler；不启动新 coordinator | 旧请求进入 scheduler 并返回本地 handle；不等计划发射 | 对实际 Work 建 CUDA host 完成回执，不能直接把 `ScheduledWork.is_completed()` 当证明 | 旧 scheduler 排空已接受任务并按其 API 结束 |
| raw-ordered 受控参考 | 建立共同静态顺序；不启动新 coordinator 或旧 admission | 本地请求进入唯一有序发射入口并返回 handle；不等实际 launch | CUDA event query 对应的物理完成回执 | 等所有已接受请求真实完成后关闭发射入口 |

原始 `bare` 尚未实现是本过程文档初次执行时的记录。后续已在共同 DAG 路径增加直接提交 `BareDagAdapter` 并完成真实双卡 L1/D1/D3 诊断；L1 出现两 rank 全局 launch 顺序差异，因此仍不能作为合格 arm。raw-ordered 始终保持独立名称，不能作为 bare 通过证据。

旧 scheduler 的旧 Plan 与新 StaticOrder 都由 `build_static_order(graph, policy, tails)` 的同一个规范任务 ID 序列生成；转换前后比较**规范序列 hash**，旧 `Plan.digest()` 另记为旧格式身份，不能要求它与新 StaticOrder 的原始 hash 字节相等。旧 `TaskKey` 有 `iteration/microbatch/parallelism/process_group_id/layer_id/bucket_id/ordinal` 字段，adapter 应冻结一张从规范 `(epoch, job_id, task_id, group_id, group_seq)` 到这些字段的确定性映射表，并保留反向映射供结果核对；不得用进程随机 `hash()`、各 rank 的线程到达次序或假定 `group_id==job_id` 编号。旧 scheduler 若现有 `submit` 或完成接口不能满足表中的非阻塞/物理完成条件，先实现适配或判为未支持，不能在 runner 中等待 grant 或把校验时同步当完成。

bare 的多 communicator 顺序需要独立证明。实现记录实际发射顺序及必要 host/device 同步；不能把新 runtime 的 grant 顺序藏到 bare 内。若只能实现 raw-ordered，arm 名称、同步开销、在途能力和与原始 bare 的差别写进冻结 manifest；它仍需通过两卡真实 NCCL 的顺序、数值和故障路径检查，不能仅凭有限次未挂起认定安全。

## 15. worker、结果和批次的统一时间线

`runtime_worker.py` 的 schema-v2 DAG 分支现已按 job 分配 GPU buffer 表，`ComputeNode` 和 `CommNode` 共享同一资源所有者，runner 推进后再 defer 数值检查。旧 v1 的 `_prepare_dag_gpu_compute`/sleep 分支继续按原合同执行，v2 使用显式模式分支。

共同 worker 时间线如下；不同 adapter 只替换引擎生命周期步骤，不改变 GPU 数据、DAG、release 和指标定义：

```text
load/validate input + profile → 确定 rank 逻辑 cuda:rank/UUID
→ 创建必要 ProcessGroup → 分配/初始化 buffer、准备 reference
→ 精确算子 warmup → reset buffer 并建立跨 stream init/reset 依赖
→ 初始化所选通信 adapter → 共同 release barrier 与本地起点
→ 所有本地 job 的 DagRunner 结束，所有 sink 已物理完成
→ 记录 application_end → 关闭输入并 drain 已接受通信
→ 记录 drain_end → 数值与投影校验 → 记录 validation_end
→ 释放本地资源、归档结果并回收进程
```

release barrier 使用独立控制 group，不能用被调度的 job collective 做 rendezvous。正常路径关闭输入恰好一次，并且排在所有已接受 submit 后；父进程不能因为某 rank 应用先结束便提前撤销其他 rank 已提交任务。失败路径为 `stop_event` → 禁止新 submit → 对应引擎 abort/唤醒等待者 → 在原固定 deadline 内回收 → 输出失败阶段和未完成任务；不得在终态无限零等待轮询。应用时长只取各 rank 自己的 release/end duration 再取最大，不跨 rank 相减原始时间戳。

`run_phase3.py` 的共同结果保留 `validation`、`metrics/performance`、`ranks`，增加 `execution_contract`、三个输入/样本/估计 hash、profile/estimator 版本、实际 arm/adapter、每 rank 设备 UUID、每 job 全部 sink 和物理完成来源。旧分析消费者若依赖 `measurement_lane` 或 linear segment 列，升级为读取共同字段并显式区分 v1/v2；不能用缺失字段默认“通过”。结果检查按规范通信 task ID 计算成员实际 launch 投影、group_seq、静态序列、tensor 数值与 terminal 覆盖；new 额外核对 grant 前缀，bare 则核对其已声明的发射合同。

`run_experiments.py` 保留原 ledger/恢复逻辑，schema-v2 DAG 可经共同 worker 启动 old/new 配置；raw-ordered 仅可显式选择为受控参考，bare 可通过 `run_phase3.py` 的独立诊断入口选择，但七臂 suite 按正式计划只采用 raw-ordered。DAG batch 记录并冻结输入、compute profile、源码摘要及 arm 配置；目前没有六场景样本生成与全场景配对合同，因此不能据此启动或宣称完成 1,050 次矩阵。

## 16. 分阶段完成门槛与实施记录模板

阶段 gate 是进入下一阶段的条件，不是声称目前已经通过。每个 gate 都应在**最后一次相关代码修改后**重新检查。

| Gate | 必须观察到的事实 | 失败时停在哪里 |
| --- | --- | --- |
| G1 输入 | v1 canonical/运行回归保持；v2 正负例覆盖绑定全集、传递依赖、buffer hazard、seed/epoch 分离；展开 manifest 的七臂 hash 相同 | 不创建 GPU 正式样本 |
| G2 GPU 节点 | fill/matmul/sum_join 的真实 tensor 数据流正确；warmup/reset 与 init stream 依赖正确；compute/comm 共享同一个 buffer 实例；profile 精确签名可追溯 | 不接旧/bare adapter |
| G3 runner/new | grant 未返回时 submit 已返回，independent 可推进；提交失败不解锁门控；计算/通信仅在物理完成后解锁后继；固定 deadline 和异常有界退出 | 不宣称统一 DAG 已验收 |
| G4 共同 worker | 链式与非片段式分叉 DAG 经同一入口运行；终点、drain、validation 时序和结果结构一致；旧 H 与新 S 回归都保持 | 不退役专用 worker |
| G5 三 adapter | 同一图/数据/估计，旧与新静态**规范序列 hash**一致；bare 顺序/完成经真实双卡证明；正常与故障路径均能回收 | 不启动七 arm 正式矩阵 |
| G6 批次 | L0/L1/D0–D3 的机制 pilot 通过七配置计划的门槛；配对 ledger、恢复、失败保留、source/manifest 可复现 | 不冻结正式 1,050 次输入 |
| G7 收尾 | 主调用链无独立 linear/bridge runner；历史输入/结果仍可定位；全仓及 opt-in GPU 结果、跳过和环境分别记录 | 不宣称迁移完成 |

检查落点沿用现有文件：`tests/unit/test_jobpacer_runtime_adapter.py` 覆盖 v1/v2 解析、规范摘要、profile 保留和错误输入；`tests/unit/test_jobpacer_gpu_workload.py`、`tests/unit/test_jobpacer_gpu_compute_profile.py` 覆盖 seed、算子签名与版本；`tests/unit/test_jobpacer_dag.py` 覆盖 `submit_after`、完成依赖、失败与固定 deadline；`tests/unit/test_jobpacer_runtime_worker.py` 覆盖 worker 阶段时序；`tests/unit/test_jobpacer_phase3_experiments.py` 覆盖七 arm 命令、manifest/ledger 与恢复。新的 buffer/adapter 测试可分别增设 `tests/unit/test_jobpacer_gpu_dag_resources.py`、`tests/unit/test_jobpacer_dag_comm_adapters.py`，避免把所有模拟塞进 worker 测试。真实通信以 `tests/integration/test_runtime_replay.py` 和 opt-in `tests/integration/test_runtime_replay_nccl.py` 扩展链式/分叉 DAG、旧/bare/new 投影与数值；需要额外 integration 文件时沿用同一 opt-in 约束，不把 GPU 检查伪装成默认通过。并发交错用事件/屏障驱动，异常路径检查实际停止与有界回收；格式、mock 或单次未挂起不足以代替真实 backend 验收。

每个阶段在新的 process 实施记录中至少写：源码 revision 与 dirty diff 摘要、文件变更/调用链、输入和 profile hash、命令、环境与 GPU UUID、通过/失败/跳过、耗时、raw 路径、未满足 gate。结果文档只写已实测事实；CPU/Gloo、真实 NCCL 语义和性能收益分开判定。正式批次需按七配置计划另行执行并归档，不能把此实施计划或旧 GPU batch 的数字当作完成证据。

## 17. 2026-09-28 实施与复核记录

R0 在代码修改前保存了当前工作区快照：`benchmark/phase3/results/gpu-unified-dag-r0-baseline-20260928-025624/`。其中包含 tracked dirty diff、未跟踪路径清单和源码归档；该目录及本节列出的新 GPU 结果都位于 Git 忽略的 `benchmark/phase3/results/`，需要随源码单独归档。

实现落点如下：

- R1/R2：`runtime_adapter.py` 接入 schema-v2 `cuda-program`，校验程序/通信绑定全集、shape、buffer hazard、`submit_after` 传递依赖，分离输入和 estimate-view 摘要。`gpu_dag_resources.py` 负责 job-local buffer、稳定 tensor seed、fill/matmul/sum_join、warmup/reset、物理完成 receipt 和应用外数值检查。GPU compute profile v3 把 shape、dtype、layout 和算子参数纳入签名。
- R3/R4：既有 `DagRunner` 增加 submit-return 门控；schema-v2 GPU DAG 经 `runtime_worker.py` 和 `run_phase3.py` 的共同路径执行。`application_end`、communication drain 与 deferred validation 分别记录。
- R5：examples 层 `dag_comm_adapters.py` 提供旧 `AdmissionScheduler` adapter 和 raw-ordered adapter。旧路径由共同 DAG `build_static_order` 生成 Plan，并用确定映射连接 `TaskKey` 与规范 task ID；NCCL 物理完成经 CUDA event probe 观察。old/raw 使用 rendezvous Store 传播 peer failure，并在销毁 ProcessGroup 前对已提交 Work 做有界 failure drain/teardown 握手。raw-ordered 不是 bare。
- R6：`run_experiments.py` 的 DAG 命令已映射六个配置到共同 runner，支持 compute profile、batch preview、profile/input/source snapshot 和 ledger resume。一个 old-static-FIFO batch smoke 完成。六场景 L0/L1/D0–D3 生成、5 workload seeds 的展开、样本级 hash 一致性审计和成组恢复尚未完成；没有创建正式矩阵 manifest。
- R7：已删除 `gpu_linear_runner.py`、`gpu_linear_worker.py`、`gpu_dag_bridge.py` 及其专用单测和 `runtime_adapter.py` 中仅供该路径使用的适配函数；`runtime_worker.py` 不再分派到旧 worker，`run_phase3.py`/`runtime_worker.py` 不再提供 `--gpu-linear` 或 `--gpu-bridge`。GPU compute/communication profile 入口均以 `--dag` 读取 schema-v2 输入。历史 schema-v1 输入目录与历史结果未删除。全量单测仍有 3 个缺失历史迁移 fixture 的失败，因此 G7 的完整验收未通过。

D0 输入为 `benchmark/phase3/experiments/dag-semantics/smoke/gpu-v2-fork-join.json`，包含两个 job 的 producer→communication/independent fork→join→后继通信。它不是 L0/L1/D0–D3 正式样本。输入文件 SHA-256 为 `1827a1d60044ee2bafc743513715333756509d14aedd0a9340a2ef26e2bac9df`，规范 `dag_input_hash` 为 `21296905a3ed972665f59c4e966cade6c827a16d3d4914486fd41a6ab49ac6ae`。compute profile 文件 SHA-256 为 `014b0d7969f3497d8e9cf2572fa945416726d301e2a2d7fb88d7809350ab06be`，profile content digest 为 `649e481e2229dc642b4dee6ac37ff5fb75accbf51ace32ce09b78662e56dc051`。

在 2× NVIDIA GeForce RTX 4090 上，rank 0/1 UUID 分别为 `c024d768-866d-952a-36f6-899a9e5844e0` 和 `a376a831-0ace-b841-7b91-6b962b9c2398`；软件为 Python 3.12、PyTorch 2.13.0+cu126、CUDA 12.6、NCCL 2.29.3。六个已支持配置各运行一次，D0 的通信 task 集合、两 rank launch 投影、所有 GPU buffer/comm 数值检查和全部 sink 均通过。old/new FIFO 的共同 static sequence digest 相同；old/new LTF 的共同 static sequence digest 相同。raw-ordered 另运行一次并通过相同正确性检查。这些是语义 smoke，单次 application duration 不用于估计策略收益。

old 和 raw-ordered 各注入一次 `binding_failure`。首次 old 故障复现出 rank 1 `-11`，随后根据证据补充 peer failure signal、backend wait 和 teardown 握手；最终复核中两 rank 都以正常 Python error exit 返回，没有信号退出或 NCCL 堆损坏。该证据只覆盖 binding 故障，不覆盖网络断连、所有 adapter 故障类型或重复 epoch。

复核命令结果：

- `PYTHONPATH=src:. .venv/bin/python -m pytest -q tests/unit`：271 passed、3 failed。三个失败是 `tests/unit/test_benchmark_paths.py` 依赖的已迁移旧输入/结果 fixture 在该工作树不存在；新改动相关测试均通过。
- 最后一次针对性检查 `PYTHONPATH=src:. .venv/bin/python -m pytest -q tests/unit/runtime tests/unit/test_jobpacer_dag.py tests/unit/test_jobpacer_runtime_adapter.py tests/unit/test_jobpacer_runtime_results.py tests/unit/test_jobpacer_phase3_experiments.py tests/unit/test_jobpacer_gpu_compute_profile.py`：130 passed，4.84 秒。
- `PYTHONPATH=src RUN_JOBPACER_RUNTIME_REPLAY=1 .venv/bin/python -m pytest -q tests/integration/test_runtime_replay.py`：47 passed，129.03 秒。
- `env PYTHONPATH=src RUN_JOBPACER_RUNTIME_NCCL=1 .venv/bin/python -m pytest -q tests/integration/test_runtime_replay_nccl.py`：4 passed，24.27 秒。
- `py_compile` 覆盖两个 adapter/worker/validator/CLI 文件，编译通过；最终 `git diff --check` 通过。
- DAG batch preview 报告六个 arm、1 个 seed、每 arm 1 次，共 6 次；单 old-static-FIFO batch smoke 为 1/1 成功。没有启动性能矩阵。
- 最后一次 CLI help 文案调整后的针对性单测 `PYTHONPATH=src:. .venv/bin/python -m pytest -q tests/unit/test_jobpacer_phase3_experiments.py tests/unit/test_jobpacer_runtime_results.py`：30 passed，2.73 秒；同轮 `py_compile examples/jobpacer/scripts/run_experiments.py` 与 `git diff --check` 通过。

逐 rank 输出、两个 failure injection、profile、batch smoke、命令/环境摘要和 D0 执行时的源码归档保存在 `benchmark/phase3/results/gpu-unified-dag-correction-20260928/`，其 `SHA256SUMS.txt` 覆盖目录内所有产物。本次已从各归档目录分别执行 `sha256sum -c SHA256SUMS.txt`，R0 基线快照与结果归档中的所有条目均校验通过。该源码归档对应 D0 smoke 当时的代码；其后的 R7 清理未重跑 GPU smoke。G5 仅在 old/new/raw-ordered 的 D0 正常路径及 old/raw binding 故障上得到有限证据；由于没有原始 bare，G5 未完成。G6 与 G7 的完整验证未完成，1,050 次正式配对实验、六场景机制 pilot、网络断连及重复 epoch 均未验收。

## 18. 2026-09-28 R7 清理后复核

用户指出专用 GPU-linear/bridge 执行路径仍留在当前树后，已清理重复实现并检查调用方。删除 `examples/jobpacer/runtime/gpu_linear_runner.py`、`gpu_linear_worker.py`、`gpu_dag_bridge.py`、对应 runner/bridge 状态机单测，以及 `runtime_adapter.py` 中仅供该路线使用的 group/task/order/tail/hint helper。`run_phase3.py` 与 `runtime_worker.py` 移除了旧 manifest replay 参数和分派；`run_gpu_compute_profile.py` 与 `run_comm_profile.py` 改为直接读取 DAG。`benchmark/phase3/experiments/gpu-linear/` 下输入仍保留，但 README 已注明它们仅是历史输入。旧版 compute-profile 结构解析仍留作文件兼容，不再有旧 manifest 校准或 replay 入口。

清理后，`rg` 对 `examples/` 与 `tests/` 扫描未发现旧 runner、bridge 执行函数、旧 replay 参数或 `GpuSegmentResources` 引用。针对性命令 `PYTHONPATH=src:. .venv/bin/python -m pytest -q tests/unit/test_jobpacer_runtime_worker.py tests/unit/test_jobpacer_gpu_workload.py tests/unit/test_jobpacer_gpu_compute_profile.py tests/unit/test_jobpacer_phase3_experiments.py tests/unit/test_jobpacer_runtime_results.py tests/unit/test_jobpacer_runtime_adapter.py tests/unit/runtime tests/unit/test_jobpacer_dag.py`（含迁移后的 drain-before-validation 检查）为 141 passed，4.89 秒。Gloo replay 为 47 passed，133.63 秒；NCCL replay 为 4 passed，23.42 秒。D0 schema-v2 输入的 `run_comm_profile --dag` 双卡最小 profile smoke 成功。全量 `tests/unit` 为 246 passed、3 failed；3 个失败仍是 `tests/unit/test_benchmark_paths.py` 所依赖的历史迁移 fixture 缺失，与删除的 GPU 路径无关。相关文件 `py_compile`、三个 profile/replay CLI 的 `--help` 检查和 `git diff --check` 通过；replay 只提供 `--workload`/`--dag`，compute-profile 要求 `--dag`，communication-profile 支持通用 `--workload` 或 `--dag`，旧 `--gpu-linear` 参数已移除。

## 19. 2026-09-28 别名拒绝与失败 deadline 复核

后续审查发现两个未覆盖点，已在首版 GPU 程序与 old/raw 失败收尾中修复：

- `_parse_compute_programs()` 现在在 hazard 分析前拒绝任意计算程序的 output 出现在 inputs 中，覆盖 `matmul` 和 `sum_join`。解析回归包含 `sum_join(inputs=["z"], output="z")`，避免输出清零后再读自身。
- old adapter 用同一把锁包住 scheduler submit 与 handle 登记，并在 abort 时先关入口、再快照已接受 handle；逐个 Work 等待和 scheduler worker join 都只取共同绝对 deadline 的剩余时间。
- raw adapter 在 release 时接收 worker 的 replay deadline，更新 dispatcher 的固定 deadline；abort 的 dispatcher join、所有已绑定 backend Work 的等待以及后续 close 共用这个 deadline。worker 的 old/raw failure teardown rendezvous 也直接以同一个绝对 deadline 截止，不再追加 `timeout + 3s`。
- 确定性交错测试覆盖 old submit 正在进入 scheduler 时 abort 开始，以及 raw dispatcher join 期间才绑定的第三个 Work；逐 Work timeout 递减并且所有操作总预算不增长。

修复后的针对性命令 `PYTHONPATH=src:. .venv/bin/python -m pytest -q tests/unit/test_jobpacer_runtime_adapter.py tests/unit/test_jobpacer_dag_comm_adapters.py tests/unit/test_jobpacer_runtime_worker.py tests/unit/test_scheduler.py tests/unit/test_jobpacer_dag.py tests/unit/runtime`：最终代码 125 passed，3.99 秒。`PYTHONPATH=src RUN_JOBPACER_RUNTIME_REPLAY=1 .venv/bin/python -m pytest -q tests/integration/test_runtime_replay.py`：47 passed，134.06 秒（在最后仅影响 old/raw close 的边界调整前；该 Gloo 路径不使用这两个 adapter）。随后在两张 RTX 4090 上针对 D0 输入分别运行 old/raw-ordered `binding_failure`；两个 worker 均按预期以 Python 错误退出，没有父进程超时或信号退出。最终 close 边界调整后再次以 `static_fifo` 正常运行 old 与 raw-ordered，两者均通过两 rank 通信/缓冲区数值校验，launch 投影与共同静态序列一致。这些只是双卡语义与故障 smoke，不是性能实验；不证明多 Work backend drain 的所有 NCCL 故障交错，后者由确定性 adapter 测试覆盖。

## 20. 2026-09-30 退役 GPU 线性执行路线

当前边界为：Gloo 保留旧 `Workload` 线性 replay 和 DAG replay；GPU/NCCL replay 只接受 DAG 输入。历史 Phase 2 `run_phase2.py` 与 `runtime/replay_worker.py` 限定 Gloo，移除了 S lane 的 CUDA producer/independent/consumer streams、warmup、完成事件及输出字段。新 `runtime_worker.py` 保留 Gloo 线性和 GPU DAG；直接以 NCCL 启动线性 workload 会在 ProcessGroup 初始化前拒绝。`run_phase3.py` 对 `--backend nccl --workload ...` 作同样的前置拒绝。

移除了 `gpu/gpu_workload.py` 中的 GPU 线性模型和 schema-v1 parser，并删除对应专用单测；`gpu_compute_profile.py` 现在只接受签名 mapping，不再依赖 `GpuComputeSpec`。旧 compute profile v2 的校验形式保留为通用 mapping 校验，DAG profile v3 的签名、加载与校验保持不变。`gpu_compute.py` 与 `_prepare_dag_gpu_compute()` 的旧 DAG matmul 能力、schema-v2 `gpu_dag_resources.py` 均保留。`run_experiments.py` 源码快照清单和 workload 注释已同步。

`benchmark/phase3/experiments/gpu-linear/` 的输入、历史运行结果及 2026-09-27 实施/结果记录保留。早期实施说明已标为历史；根 README 与 JobPacer README 改为推荐 DAG/NCCL 用法。

最终复核：`PYTHONPATH=src:. .venv/bin/python -m py_compile` 覆盖本轮改动的 GPU profile、两个 worker、Phase 2/3 CLI 和对应测试，退出码为 0；四个入口的 `--help` 均正常，Phase 2/replay worker help 只列出 Gloo backend，Phase 3 help 不再含 `--lane`。`git diff --check` 通过，源码扫描未发现 `GpuComputeSpec`、`GpuLinearInput`、旧 `gpu_segments` 或线性 S-lane 分支。

- 针对性命令 `PYTHONPATH=src:. .venv/bin/python -m pytest -q tests/unit/test_jobpacer_gpu_compute_profile.py tests/unit/test_jobpacer_runtime_worker.py tests/unit/test_jobpacer_gpu_route_retirement.py tests/unit/test_jobpacer_phase1.py tests/unit/test_jobpacer_phase3_experiments.py tests/integration/test_jobpacer_profile.py tests/integration/test_runtime_replay_nccl.py`：46 passed、5 skipped，27.59 秒。通过项包括 compute-profile v2/v3、worker 直接拒绝 NCCL 线性、Phase 2 Gloo profile replay 和 Phase 3 入口拒绝测试；其余 opt-in NCCL 用例在该命令中跳过。
- `PYTHONPATH=src RUN_JOBPACER_RUNTIME_REPLAY=1 .venv/bin/python -m pytest -q tests/integration/test_runtime_replay.py`：47 passed，117.64 秒，覆盖保留的 Gloo 新 runtime 线性与 DAG 路径。
- 双卡 NCCL 的 `test_two_rank_gpu_dag_waits_for_compute_event_before_dependent_comm`：1 passed，5 deselected，6.71 秒。RTX 4090 ×2、PyTorch 2.13.0+cu126、CUDA 12.6、NCCL 2.29.3；保留的 DAG `cuda-matmul` 计算回执检查通过。
- 另以 schema-v2 `gpu-v2-fork-join.json` 执行一次 `run_phase3 --policy fifo --dag ... --backend nccl` smoke：validation `ok`、2 ranks、collective 数值正确；结果 `/tmp/jobpacer-gpu-dag-retirement-smoke.json`。该次是语义检查，不作性能结论。
- 完整 `tests/unit`：310 passed、3 failed，7.27 秒。3 个失败仍是 `tests/unit/test_benchmark_paths.py` 依赖的历史迁移映射/结果 fixture 缺失（Phase 3 旧路径、Phase 1.2 旧批次目录），不涉及本次 GPU 路线修改；未补造或改写历史产物。

## 21. 2026-09-30 schema-v2 GPU 合同与模块归位

本轮按“GPU/NCCL 只接受 schema-v2 `cuda-program` DAG；Gloo 保留线性与 DAG”执行整理。

- Phase 3 父入口、直接 rank worker、NCCL 通信 profiling 和 GPU compute profiling 均检查 schema-v2 GPU DAG 合同；该合同要求输入显式列出每个 job 的 `application_terminals`。旧输入不转换成新语义。NCCL 集成继续覆盖旧线性/schema-v1 输入的前置拒绝。
- 退役 `gpu_compute.py`、`CudaMatmulProgram` 和 `_prepare_dag_gpu_compute()`。所有 GPU DAG 计算仅通过 `GpuDagResources` 的输入绑定执行。compute profile 按输入 `compute_programs` 的精确签名匹配，只更新估计时长；移除了旧 `nominal_compute_repeats` 对 program repeats 的运行时覆盖。七臂生成器将每个 matmul 的 repeats 保存在该节点 program 中。
- `workloads.py`、`workload_builder.py`、`plan_builder.py`、`replay_worker.py` 分别移入 `gloo/`。旧 Gloo `Workload` 映射和线性采样绑定移入 `gloo/runtime_adapter.py`；通用通信 profile 数据模型移入 `runtime/comm_profile.py`，Gloo Workload 应用层留在 `gloo/comm_profile.py`。
- 共享路径工具移入 `examples/jobpacer/paths.py`。Phase 3 replay launcher 实现位于 `runtime/replay_launcher.py`；旧 Phase 3 与 compact-suite 批次实现移入 `experiments/`，GPU Phase 3 统计和证据校验位于 `analysis/phase3.py`，Phase 1.2 批次实现移入 `experiments/runner_batch.py`。`analysis/` 和 `diagnostics/` 不再导入 `scripts/`。
- 为保留 NCCL 多 communicator 的真实检查，新建 `gpu-v2-multi-group.json` schema-v2 DAG smoke。历史 GPU-linear 输入、旧报告和既有结果未改写。源码目录与源码快照清单已同步；旧 source digest 的资格不能用于当前树。本轮没有创建/执行正式采集矩阵。

本轮验证：针对性回归 107 passed；完整单元测试 315 passed、3 skipped（checkout 不包含历史迁移 map/结果夹具）；Gloo 双 rank runtime 集成 47 passed；双卡 RTX 4090 NCCL 语义套件 7 passed；六个七臂子命令、公开 Phase 3/Phase 1–3 命令的 `--help` 检查通过。初次全量 `pytest -q` 曾递归收集 `benchmark/phase3/results/**/source_snapshot/tests` 下的历史源码副本并触发重复模块名；`pyproject.toml` 现将默认 `testpaths` 限定为 `tests/`，历史快照保持原样。最终 `PYTHONPATH=src:. .venv/bin/python -m pytest -q` 为 327 passed、52 skipped，33.82 秒。没有执行正式实验矩阵；最终 `git diff --check` 与源码扫描通过。

## 22. 2026-09-30 删除薄入口

删除了 `scripts/run_phase3.py`、`scripts/run_experiments.py` 和 `scripts/run_compact_suite.py` 三个只转调内部 `main()` 的 wrapper。Phase 3 replay 直接通过 `python -m examples.jobpacer.runtime.replay_launcher` 启动；批次和 compact suite 分别通过 `examples.jobpacer.experiments.gloo_phase3_batch` 与 `examples.jobpacer.experiments.compact_suite` 调用。诊断、资格检查、批次执行及 Gloo/NCCL 集成测试均更新为直接调用这些实现，Phase 3 源码快照不再列入已删除的 wrapper。根 README、JobPacer README、协作指南和当前命令示例已同步。

删除后验证：三个内部 CLI 的 `--help` 和 `compileall` 通过；`PYTHONPATH=src:. .venv/bin/python -m pytest -q` 为 327 passed、52 skipped（36.30 秒）；双 rank Gloo runtime 集成为 47 passed（129.37 秒）；双卡 RTX 4090 NCCL 集成为 7 passed（33.00 秒）。`git diff --check` 通过，Python 源码和测试不再引用被删除的模块或路径。没有启动实验批次。

## 23. 2026-09-30 将 Gloo runtime 实现归位

按职责将 `gloo/plan_builder.py` 和 `gloo/replay_worker.py` 移入 `runtime/`，将线性 workload 映射适配器移为 `runtime/gloo_runtime_adapter.py`，避免与已有的通用 DAG `runtime_adapter.py` 重名。Gloo workload 数据模型、builder 和 profile 应用代码仍留在 `gloo/`。更新了 Phase 2 CLI、Phase 1/2 批次实现、Phase 3 线性/DAG 适配调用、测试导入和源码快照清单。

## 24. 2026-09-30 将七臂 GPU study 命名为 Phase 3 实验

用户入口由 `scripts/run_gpu_seven_arm.py` 改为 `scripts/run_phase3.py`；原六个子命令及各自执行语义保持不变。实现包从 `experiments/seven_arm/` 移至 `experiments/phase3/`，批次分析从 `analysis/seven_arm.py` 移至 `analysis/phase3.py`。`runtime/replay_launcher.py` 继续负责单次底层 replay，因此顶层 Phase 3 实验命令与 replay CLI 含义分开。原 CPU/Gloo Phase 3 批次模块另改名为 `experiments/gloo_phase3_batch.py`，与 GPU Phase 3 study 区分。七臂作为当前 Phase 3 study 的 arm 设计保留在 schema、manifest 和历史结果术语中。源码快照和当前操作说明已同步。

验证：`PYTHONPATH=src:. .venv/bin/python -m examples.jobpacer.scripts.run_phase3 --help` 显示六个实验子命令；`gloo_phase3_batch --help`、`compileall` 和 `git diff --check` 通过；`PYTHONPATH=src:. .venv/bin/python -m pytest -q` 为 327 passed、52 skipped（36.05 秒）。当前 Python 源码、测试与操作说明不再调用旧入口名或旧包路径；历史过程记录保留旧名称作为当时状态。

## 25. 2026-09-30 将 Gloo adapter 放回 Gloo 模块

应用户要求，将 `runtime/gloo_runtime_adapter.py` 移回 `gloo/runtime_adapter.py` 并恢复原模块名。通用 DAG 适配器仍为 `runtime/runtime_adapter.py`；两者分别负责旧 Gloo workload 映射与 DAG schema/runtime 绑定。Phase 3 worker、replay launcher、测试和源码快照均引用新的 Gloo 路径。

验证：完整测试为 327 passed、52 skipped（38.35 秒）；`compileall` 与 `git diff --check` 通过，代码和测试不再引用 `runtime.gloo_runtime_adapter`。

验证：`PYTHONPATH=src:. .venv/bin/python -m pytest -q` 为 327 passed、52 skipped（34.65 秒）；最终 `compileall`、Phase 2 `--help`、Python 调用路径扫描与 `git diff --check` 通过。全仓初次收集发现两个单元测试仍从旧 `gloo` 包导入 `replay_worker`；已更新为 `runtime.replay_worker`，随后完整测试通过。
