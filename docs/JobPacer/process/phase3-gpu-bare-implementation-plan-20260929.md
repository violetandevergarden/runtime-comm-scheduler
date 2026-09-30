# Phase 3 GPU bare 修正执行计划

日期：2026-09-29。状态：**bare-ordered-v1 的实现和专项验证已完成；分层轮转修订版 bare-ordered-v2-layered-round-robin 已完成有限双卡 pilot，正式七臂接入仍待完整资格和其他实验门槛。**v1 结果见[bare 实施结果](../result/phase3-gpu-bare-implementation-20260929.md)，v2 pilot 见[分层轮转结果](../result/phase3-gpu-layered-fifo-pilot-20260929.md)。

本计划落实本轮讨论：bare 应表示不经过 JobPacer 性能调度器的框架执行路径，同时满足 PyTorch/NCCL 正确性要求，允许多个通信在途。当前自由到达顺序的 bare 实现不合格，不意味着应放弃 bare；当前逐项等完成的 raw-ordered 也不能完整替代这一目标。

## 1. 依据、历史修正与范围

- [9 月 14 日实验计划](../260914项目计划讨论/JobPacer实验计划.md)提出直接调用 torch.distributed、观察多 job 通信竞争，再与完成串行的 scheduler 对照。
- 同目录的[仓库状态说明](../260914项目计划讨论/项目仓库%20Current%20State.md)已经提到 overlapping communicators 的循环 launch order；原实验计划没有完整衔接这一执行条件。
- [9 月 28 日 bare 合同评估](phase3-gpu-seven-arm-bare-contract-20260928.md)及[preflight 结果](../result/phase3-gpu-seven-arm-preflight-execution-20260928.md)记录了直接裸发的数值成功与跨 rank 发射顺序分歧。这些历史事实保留，不改写成 GPU 数值失败，也不改写成新 bare 已经通过。
- 复用[统一 DAG 设计](../plan/phase3-gpu-workload-and-scheduling.md)、[修正实施路线](phase3-gpu-unified-dag-correction.md)和[七臂实验计划](../plan/phase3-gpu-seven-arm-experiments.md)的输入、计算、采样及统计合同。

本轮讨论修正了两项过度限定：必要的执行顺序保障不自动等于性能调度；没有 scheduler 不意味着线程可以在各 rank 任意重排通信。9 月 14 日研究目标保留，GPU 执行方案需补完整。历史两项 all-reduce 并发测量是否采用共同 host 顺序，必须查原脚本，不能从结果表反推。

本轮按 bare-only 范围完成 bare 示例适配层及必要的共同收尾接口：不重写 DagRunner，不引入另一套 linear/bridge，不修改新 coordinator 的 max_inflight=1，不扩展多资源、动态权限、跨进程独立 job 或完整 Megatron 集成。现有工作区修改、历史代码快照和原始结果均保留。

## 2. 目标合同与命名

### 2.1 Bare 的定义

```text
共同 DagRunner / CUDA 程序 / buffers / 依赖 / 终点
    ├─ bare：默认通信调用顺序 → 异步直接执行 → PyTorch/NCCL
    ├─ old：旧 Plan / AdmissionScheduler → PyTorch/NCCL
    └─ new：RankRuntime / coordinator / policy → PyTorch/NCCL
```

Bare 保留任务身份、组内顺序、框架默认调用顺序、producer/consumer 依赖、buffer 保活、完成观察和错误退出；不调用新 coordinator、旧 AdmissionScheduler 或 FIFO/LTF 的在线选择，不按通信估计或 tail 调整顺序，不施加全局单在途性能准入。

本地 submit 接受并保存请求后返回 handle，不等待排到发射位置、backend 提交或通信完成。默认顺序的队首尚未 ready 时，dispatcher 可以等待，但不能堵住 DAG 的独立计算推进，也不能由各 rank 自行跳过队首。此等待属于基线执行顺序的成本，纳入应用时间并单独观测。

### 2.2 三种路径不得混淆

| 路径 | 顺序来源 | 在途合同 | 用途 |
| --- | --- | --- | --- |
| 历史自由到达 bare | 各 rank 的线程到达与 group_seq | 不设全局容量限制，已观察到跨组顺序分歧 | 保留诊断证据，不进入正式矩阵 |
| 当前 raw-ordered-static-fifo | 共同静态全序 | dispatcher 逐项等本地物理完成 | 串行执行参考，保留既有结果 |
| 本计划 bare-ordered | 工作量定义的默认调用顺序 | 不因前项未完成而额外阻止后项提交；仍遵守依赖和 backend 条件 | 新 bare 候选，待独立资格验证 |

arm ID 为 `bare-ordered`。本次分层轮转合同版本为 `bare-ordered-v2-layered-round-robin`。CLI 继续使用 `--policy bare --comm-engine bare`，输出包含 execution contract/version；不能将历史自由到达 bare 或 v1 顺序解释成当前合同。

本 bare 代表有确定执行结构的框架式 replay，不代表互相独立的真实训练进程完全自由竞争。多在途能力、实际设备重叠和性能收益是三个不同结论。某个合法 DAG 的峰值在途为 1 不自动失败；但专门的独立通信资格用例必须证明执行器没有隐含单在途门控。

## 3. B0：盘点历史证据与当前调用链

实施前记录 HEAD、dirty patch、相关源文件 hash 和可恢复快照，确认当前 CLI、测试和 preflight 状态；不能依据本文中的历史状态跳过检查。

检索 9 月 14 日两项 all-reduce 实验的原脚本及结果，核对：调用线程与进程布局、ProcessGroup 成员、跨 rank 发射顺序、异步调用与 wait 的位置、CUDA stream/event、NCCL 配置、计时与时钟域。如果只找到表格，明确记为“执行方法尚未核实”，不阻塞新实现，但不宣称已复现历史方法。

复核 Gloo bare：是否每 job 独立 ProcessGroup、是否只检查组内顺序、是否记录跨组交错。Gloo 通过仅证明其已测路径；若未记录跨组全序，不能声称其全序一直相同。当前 NCCL 差异来自 backend 约束和适配实现，不能反向判定 CPU 数据无效。

至少检查以下实际调用链：

- `examples/jobpacer/runtime/dag_comm_adapters.py`：BareDagAdapter、RawOrderedDagAdapter、handle、completion observer 与 failure signal。
- `src/runtime_comm_scheduler/runtime/executor.py`：producer gate、backend Work.wait、完成 event 与 keepalive。
- `src/runtime_comm_scheduler/dag/`：节点推进、submit 接受门、物理完成依赖与线程模型。
- `runtime_worker.py`、`run_phase3.py`、七臂 suite/qualification/batch：engine 选择、校验、结果合同与 formal 门槛。

交付：简短现状清单和源码快照；不因历史代码缺失而虚构 bare 的原始实现。

## 4. B1：冻结框架默认通信顺序

手工 DAG 没有唯一的算子执行顺序，需要在输入展开阶段生成并保存默认通信调用序列。此顺序是 baseline 的显式假设，不是性能最优顺序，也不是实际到达 FIFO。

优先复用已有图校验、拓扑工具与序列校验。对完整 DAG 依赖、`group_seq` 和必要的 `submit_after` 接受约束计算拓扑层级；先取较浅层，同层 job 按固定输入顺序轮转，每个 job 内按输入节点顺序打破平局，最后取通信投影并重新验证合法性。规则不读 profile、tail、计算/通信时长、未来扰动或线程到达。旧、新 static FIFO 共用这一冻结序列；dynamic FIFO 继续按首次 eligible 排序。

基线规则标识为 `topological-layer-job-round-robin-v1`。层级只离线决定固定调用顺序，不是执行屏障：不等一层的任务完成才进入下一层，DAG 计算照常推进，bare 的通信仍异步提交并可多项在途。它是便于复现的手写 DAG 调用顺序约定，不代表最优调度、公平性保证或 Megatron 的真实调用序。诊断时可交换固定 job 起始顺序，不能按运行性能挑选默认项。此规则仍有静态队首阻塞：队首未 ready 时，后续已 ready 任务会等待；消除这种等待需要在线协调，需另行计入协议成本。

输入可显式携带默认顺序，但须具备同等校验。冻结内容至少包括规则版本、完整 task ID 序列、group 成员、输入 hash、sequence hash。所有 rank 在应用计时前核验共同输入和顺序；非成员执行本地投影，不能要求不参与者发出 collective。首版只宣称覆盖当前双 rank 多 group 合同，其他成员拓扑另行验证。

启动前校验必须覆盖：

1. 通信节点全集精确覆盖，无缺失、重复、非法身份或参数不匹配。
2. 每组投影满足 group_seq；不同组的通信顺序与完整 DAG 的传递依赖兼容。
3. 将调用顺序约束、compute/comm 完成边、submit_after 接受门及现有计算推进约束一起检查，避免队首需要的计算反过来等待后序通信被接受或完成。
4. 非法顺序在任何 collective 发射前拒绝，不能靠运行超时发现，也不能将通信投影直接转成计算的完成串行边。

通过标准：固定输入得到固定序列；profile/epoch/repeat 不改变默认顺序；非法依赖与重复任务有回归用例。

## 5. B2：实现顺序发射与独立完成观察

### 5.1 复用与状态边界

保留共同 DagRunner、binding 和 CUDA 程序。复用已有 bare handle、完成观察、失败传播与 drain；借用 raw-ordered 的队列及顺序校验思路，不复制其逐项完成等待。以小型具体适配器实现，不增加插件框架。

建议本地状态为 accepted → launching → bound → physically-completed，错误可进入 failed；不伪造新 runtime 的 GRANT。只有真正完成 backend 调用并建立 receipt 后，才发布 bound 并允许观察器查询。实际 launch 日志由唯一 dispatcher 记录。

### 5.2 发射循环

```text
submit(task, binding):
    校验身份、成员、重复提交及终态
    保存请求并返回 handle

dispatcher:
    取共同序列的下一个本地任务
    等它的请求到达或等待失败/固定 deadline
    经已验收的 backend 顺序机制异步发射
    发布 receipt，前移调用序列游标
    继续下一项；不在这里循环等待前项物理完成

completion observer:
    独立查询所有 bound receipt
    标记物理完成，唤醒 DAG，按资源合同回收
```

队首等待不能持有阻止 submit 或 completion 更新的锁。dispatcher 阻塞不能阻塞 job 的独立计算。关闭输入后先排空已接受请求；发生失败后停止新发射、唤醒全部等待者，使用固定清理期限退出。已发出的 NCCL 工作不能伪装成可取消请求，保留首个根因，复用现有外层 worker 超时兜底。

移除应用级全局单在途限制，不代表无限预分配所有 tensor：资源由现有 workload 管理，输入规模需有界，不在本阶段增建吞吐控制器。任务到达、backend API 返回、设备完成、应用 terminal 和最终 drain 分开计时。

## 6. B3：选择并验证 NCCL 多 communicator 执行机制

这是进入多在途实验的前置条件，不能只检查 host launch 序列相同。参考 [NCCL 2.29 系列 communicator 文档](https://docs.nvidia.com/deeplearning/nccl/archives/nccl_2293/user-guide/docs/usage/communicators.html#using-multiple-nccl-communicators-concurrently)和 [PyTorch 分布式接口](https://docs.pytorch.org/docs/stable/distributed.html)。文档版本、实际 PyTorch ProcessGroupNCCL 实现、目标运行版本需一起核对。

候选机制是共同 host 调用顺序配合已支持的 NCCL implicit launch ordering；需确认目标版本是否支持、如何启用、有效配置及 PyTorch 是否施加其他依赖。不能假定默认启用，不能认为设置环境变量就完成安全证明，也不能为获得并发擅自升级库。

逐项检查 producer event → collective、跨 communicator launch ordering、collective → completion event、completion → dependent compute 的关系。保留 backend Work.wait 在目标 stream 上建立完成依赖的用途，不能因其名字含 wait 就删除；同时检查每 group gate stream 是否因不必要的共享依赖串行化其他通信。配置若导致 host 阻塞或不兼容现有完成桥，启动前明确拒绝。

要求说明 backend 为什么保证进展，再用真实双卡验证，不以“重复多次没挂”代替机制论证。若当前 PyTorch/backend 路径只能证明完成串行安全，应报告“多在途候选尚未支持”，保留串行参考，不偷换 bare 合同或直接删除安全检查。

涉及共享 executor 的修改须回归旧 H/S lane、新 runtime 和 raw-ordered；若可通过小型可选执行配置实现，不改变其他 arm 的默认行为。多在途安全验收不授权放宽新 coordinator 的容量。

## 7. B4：分层回归与 GPU 资格

### 7.1 确定性交错回归

新增或扩展 `tests/unit/test_jobpacer_dag_comm_adapters.py` 等实际相关测试。优先使用事件/屏障控制交错，验证：

- 两个模拟 rank 的 ready 到达相反，实际发射仍是同一共同序列；组内顺序正确。
- B 先到、A 后到时，B 的 submit 已返回；等待 A 不阻塞独立计算，且不偷发 B。
- A receipt 被人为保持未完成，独立且已 ready 的 B 仍能发射；捕获隐含的单在途回归。
- 有真实 A→B 完成依赖时，B 不能提前发射；submit_after 只等待请求被接受。
- 观察器不能访问尚未绑定 receipt；buffer 在物理完成前保活。
- 缺任务、错误顺序、重复 submit、binding/launch/probe 失败、等待队首时 close、带在途的异常退出均有固定期限和首因保留。

单元测试不作为 GPU 安全证据。新增测试名、命令和实际结果由实施记录填写，不预报通过数量。

### 7.2 最小双卡机制实验

先准备两项独立且已 ready 的 all-reduce，所有 rank 固定 A→B 调用顺序；比较 A 单独、B 单独、完成串行、允许多在途四种执行。固定 tensor/消息大小/设备/预热和相关 backend 配置，使用相同工作量，至少五次重复用于初步稳定性观察，不从该小实验推断普遍收益。

记录每项 API 起止、receipt 发布、CUDA 物理完成、本地观察到的在途数量、应用总时间。观察器计数可能因轮询滞后夸大在途重叠，不能仅凭 peak inflight>1 判定设备并行；另用设备时间线核对 kernel 区间。主性能样本与 profiler 诊断分开。

通过门槛是正确性、共同顺序和无隐含完成串行门控；实际 kernel overlap 单独报告，不以“必须比串行快”为资格标准。若 GPU 执行没有重叠，区分输入太短、资源竞争、backend 序列化和实现等待，不通过无限加大 workload 挑选有利结果。

### 7.3 共同 DAG 与失败路径

按当前候选输入先跑 L1、D1、D3，再覆盖其余 L0、D0、D2；每项检查任务/参数/成员、group 投影、默认序列投影、buffer 数值、producer 与 join 因果、所有 sink、terminal 和物理 drain。保留 ready 偏斜与多个独立前沿，不以消除扰动掩盖顺序问题。

在目标双卡做 binding、launch、completion-probe 故障和缺失请求的有界失败检查，外层 timeout 记录真实原因；网络断连和同进程重复 epoch 单列已测/未测，不能借原 raw-ordered 的成功继承资格。

按风险运行相关 unit、真实 Gloo 回归及 opt-in NCCL 套件；共有 worker/executor 末次修改后重做受影响检查。使用仓库 .venv，继承 CUDA_VISIBLE_DEVICES，不安装升级依赖、不占用未分配设备、不停止其他用户进程。大规模实验不属于这些资格检查。

## 8. B5：实验矩阵、文档与历史兼容

实施时同步修订七臂计划和 qualification/batch 合同：目标七臂恢复为新 bare-ordered、old static FIFO/LTF、new static FIFO/LTF、new dynamic FIFO/LTF。当前 raw-ordered 保留为额外诊断参考，不自动把正式矩阵扩为八臂；其既有结果不删除。

不改变六场景、五 workload seeds、每样例五 repeats 的正式设计；新 bare 通过资格并完成七臂彩排后，才可在后续正式批次选入。本文不解除既有 D1 机制、profile、预算、源码冻结等其他 gate；实施前读取最新实际状态，不沿用旧报告宣布其通过或失败。

新增 arm 合同/顺序 hash/backend 配置后，重新冻结批次与源码，旧 raw-ordered、旧自由裸发、bare-ordered-v1 与 v2 不拼成一个统计 arm。v2 的 old/new static FIFO 必须共用同一顺序文件；不得把同层起始顺序带来的变化全部解释为执行系统开销。

结果分析分开回答：

1. 相同顺序下，完成串行与允许多在途的差异。
2. 相同新 runtime 单在途路径下，static/dynamic 与 FIFO/LTF 的差异。
3. Bare 与 scheduler 的整体净效果，承认顺序、容量和控制开销同时不同，不能唯一归因。

此前 bare-only 执行完成 bare-ordered-v1 实现和双卡资格验证，没有注册正式 arm。随后按用户确定的分层轮转规则实现 v2，并运行独立两场景 pilot；正式 qualification、七臂彩排和矩阵仍未启动。

GPU 资格结果写入 result/，实施过程写入 process/；discussion、七臂计划、bare 合同说明、README 的术语与当前状态同步更新，采用有日期的勘误保留上下文。正式原始产物、manifest、命令、环境、设备 UUID、源码/输入 hash、失败与重试 ledger 在被忽略的结果目录另行归档。

## 9. 完成交付与停止条件

| 交付 | 完成要求 |
| --- | --- |
| 历史与合同说明 | 区分 9 月 14 日目标、历史测量未知项、自由裸发缺口及新定义 |
| 默认顺序 | 可复现、可校验、无性能估计驱动，公布其队首等待影响 |
| 适配器 | 共同 DAG、异步接受、唯一顺序发射、独立完成观察，无单在途性能门控 |
| Backend 资格 | 有执行机制依据和目标双卡证据，配置可复现 |
| 回归与故障 | 最终代码的相关检查完成，未验收边界单列 |
| 批次接入 | 新 arm/version 正确隔离历史，其他 formal gate 保留 |

任一 backend 安全条件不明、数据/顺序错误或失败退出无界时，停止推进正式实验，定位对应根因。不能通过取消检查、挑选相同顺序的成功样本、退回单在途后继续称为新 bare 来完成交付。没有测出加速本身不构成实现失败；正确的基线也可能比 scheduler 慢。

## 10. 本轮执行状态

| 项目 | 状态 | 证据或边界 |
| --- | --- | --- |
| B0 历史与调用链 | 完成 | 9 月 14 日原始执行脚本/数据未找到，方法仍记为未核实；当前源文件快照和哈希已归档。 |
| B1 默认顺序 | v2 实现并通过有限双卡 pilot | L1/D1 默认和反转 job 起始次序共 16 个 replay 全部通过顺序及数值检查；完整 G1 仍待 D3 和故障注入。 |
| B2 顺序发射 | 完成 | rank-local 单 dispatcher 按共同顺序发射，独立 completion observer 不形成单在途门控；提前 close 会失败化所有未完成 handle 并唤醒等待者。 |
| B3 NCCL 机制 | 完成本轮范围 | 双卡使用 NCCL 2.29.3 与 `NCCL_LAUNCH_ORDER_IMPLICIT=1`；本轮验证 host 发射顺序和多在途调用路径，没有设备 kernel 时间线。 |
| B4 回归与 GPU | v1 完成；v2 pilot 完成 | v1 的历史回归保持原报告；v2 两场景正常路径完成，结果见[分层轮转 pilot](../result/phase3-gpu-layered-fifo-pilot-20260929.md)。 |
| B5 正式矩阵接入 | 未执行 | v2 仅完成两场景顺序 pilot；尚未修改正式 arm/qualification/batch 合同，也未启动正式矩阵或七臂彩排。 |
