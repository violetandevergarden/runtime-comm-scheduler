# JobPacer Phase 3.1：动态 runtime 重写计划

更新时间：2026-09-16

设计依据：[讨论总结](discussion.md)。本计划替代此前以现有代码复用为前提的 Phase 3.1 计划。

## 1. 目标与完成边界

从零设计并重写通信 runtime，根据实际任务到达和完成状态，在线决定下一项 collective，或有界等待即将到达的任务；保证各 ProcessGroup 成员提交相同的 collective 序列。

不要求兼容现有 Plan、scheduler、Work、executor 或七字段 TaskKey，也不以现有实现、测试结构作为架构限制。历史实验输入和结果可作对照资料，新核心不依赖旧代码。

本阶段完成：

1. 两 rank、多个线性 job 的在线提交与真实 collective 执行。
2. 中心化 coordinator、独立控制通道、本地单 launch worker。
3. 在线任务注册、成员匹配、有序 grant、独立完成反馈、失败与结束握手。
4. StaticOrder、DynamicFIFO、DynamicLTF、BoundedLookahead 四种策略。
5. 确定性机制验收，以及同一执行路径下的小规模静态/动态对照。

首版限制：全局一个通信容量，max_inflight=1；首先支持 all-reduce；CPU/Gloo 先验收，GPU 可用后补两 rank NCCL。线性 workload 由应用推进，不做任意 DAG executor。

不实现多 collective 在途、跨 host 部署、多进程共享 GPU、Job Agent 服务、多资源调度、抢占、重连恢复或大规模性能研究。完整 Phase 2 workload 重测和多 seed 性能统计在机制验收后另行安排，不作为本次完成条件。

## 2. 最小系统结构

```text
每 rank 的多个 job 线程
        │ declare / submit → handle
        ▼
本地 Runtime ─── OFFER / SUBMITTED / COMPLETED / FAILED ─► Coordinator
        ▲                                                     │
        │                                            安全检查 → Policy
        └────────────────── GRANT ────────────────────────────┘
        │
单 launch worker → async collective → 独立完成探测
```

每 rank 一个进程，GPU 场景每进程一块 GPU，进程内托管多个 job 线程。Coordinator 位于 rank 0 进程内，以单事件循环拥有全局可变状态；rank 0 同样通过普通本地 runtime 参加协议。

采用一种独立 TCP 控制通道。接收线程向事件循环投递消息；状态变更、策略调用、有序 grant 发布与等待计时由事件循环统一处理。不得使用 job ProcessGroup collective 做调度 rendezvous，不建设可插拔传输框架。

核心放在 src/runtime_comm_scheduler/，整体重写；examples/jobpacer/ 只负责 workload、启动和结果输出。职责可组织为 model、runtime、coordinator、policy、transport、executor、telemetry；允许合并文件，不要求每项职责对应一个可替换抽象层。

## 3. 数据模型与应用接口

### 3.1 对象边界

| 对象 | 本阶段内容 |
| --- | --- |
| GroupSpec | epoch 内稳定 group_id、成员 endpoint 集合，启动时注册校验 |
| TaskSpec | epoch、job_id、task_id、group_id、group_seq、collective 参数 |
| TaskHint | 预计剩余 ready 延迟、通信耗时、通信后的 remaining tail；不参与任务身份匹配 |
| LocalBinding | 本地 tensor、实际 ProcessGroup、producer dependency、执行入口、生命周期引用 |

task_id 在 epoch 内唯一，由 adapter 按共同语义产生。group_seq 从 0 开始按 group 的规范序列编号，不能按本地线程到达顺序生成。job 与 group 独立标识，即使首批 workload 每 job 只有一个 group。

首个 all-reduce backend 校验算子、shape、dtype、元素数/字节数、reduction 等匹配参数；tensor 值不要求相同。成员集合由注册的 GroupSpec 定义，提交者不能任意改写。

Phase 3.1 不提供通用 DAG schema。线性计算依赖由 workload 推进；group_seq 只表达通信顺序，不代表未来所有依赖都必须等物理完成。

### 3.2 API 草案

```python
runtime.declare(spec, hint=hint)        # 可选：声明未来任务
handle = runtime.submit(spec, binding) # 无 declare 也可注册并提交
handle.wait_host(timeout=...)          # 等待主机可观察的完成
handle.wait_on(consumer_stream)        # GPU：建立消费 stream 依赖
runtime.finish_epoch(timeout=...)      # 关闭本地输入并等待全局结束
```

方法名可调整，语义不能改变：submit 完成本地校验和保存后返回，不等待 grant；本地对象不发往 coordinator；声明不是 ready，也不是执行许可。

Hint 在声明时提供，必要的预计 ready 信息可随 OFFER 更新，不实现在线学习服务。相对 ready 延迟转换为 coordinator 时钟上的预测并记录依据，不直接比较远端原始时钟。

声明任务必须最终提交；首版不提供撤销，关闭输入后仍缺失就报错。动态路径不要求完整任务全集；结束由输入关闭和排空判断，不靠耗尽静态列表。

## 4. 协议与一致性

### 4.1 状态和事件

本地状态：PENDING → OFFERED → GRANTED → SUBMITTED → COMPLETED；执行错误进入 FAILED。可选声明作为注册信息单独管理，不强迫直接提交经过虚构的生产阶段。

上行事件：DECLARE、OFFER、SUBMITTED、COMPLETED、INPUT_CLOSED、FAILED。下行事件：GRANT、FINISHED、FAILED。携带 epoch、endpoint、递增 event_seq 及相关 task_id。

Coordinator 检查旧 epoch、重复提交、非成员、元数据冲突、非法转移和 group 序号冲突。允许提前声明后项，但缺口存在时不能发射；输入关闭或超时报告缺失项。

### 4.2 决策与有序发布

容量可用或相关状态改变时：

1. 处理事件，更新成员状态。
2. 构造成员已全部 OFFER、group 顺序合法的 eligible。
3. Policy 根据快照返回 Dispatch(task) 或 Wait(target, deadline)。
4. 校验返回值；Dispatch 时先占用容量、提交不可重排的决定，再向成员发布 grant。

Grant 包含全局 decision_seq 和每 endpoint 连续的 delivery_seq。单一发布路径有序发送，本地单队列校验接收，由唯一 worker 提交；不交给 job 线程自行发射。

同编号重复 grant 不重复执行；冲突重复、序号缺口、非法状态触发失败。首版不重传或重连，连接异常有界退出。

全局至多一个 grant 尚未完成。收齐所有成员的物理完成回执才释放容量；已 grant 尚未提交也占容量。完成回执独立于应用消费。

### 4.3 不变量

- 成员对任务身份、group_seq 和参数匹配后才能 grant。
- 每任务至多 grant/launch 一次，实际提交记录是共同 grant 序列的本地前缀。
- 同 group 各成员可以快慢不同，不能跳项、交换或重复提交。
- 已 grant 不撤销；已提交不假装可取消；受管理 group 不允许绕过 runtime 发射。
- 单事件循环与有序 worker 保证提交顺序；全局完成屏障是本阶段的容量策略，语义上与顺序机制分开。

未来多在途仍需这些机制，并单独验收 backend 设备端排序。本阶段可用模拟 grant 序列验证本地顺序不依赖完成时间，不实现真实多在途模式。

## 5. 执行、完成与结束

CPU producer sleep 完成后才 OFFER。GPU producer event 已记录可表示依赖已建立、允许接受 grant，但不是 producer 已物理完成；executor 必须将依赖传递给实际通信执行路径。

SUBMITTED 表示异步调用已成功交付 backend、返回有效句柄，不等于入队或完成。Runtime 独立探测物理完成，更新 handle 并发送 COMPLETED。

wait_host 与 wait_on 分开定义；不得仅凭底层 wait 的 CPU 返回认定物理完成，完成探测方式需经过 backend 验收。未绑定 handle 的消费等待可等发射绑定，但不能阻塞控制循环。

Runtime 在安全释放前保持 tensor 等对象存活，消费位置由 workload 定义。集成测试检查真实 tensor 结果。

正常结束：所有端 INPUT_CLOSED、声明无缺失、已接受任务全部完成，coordinator 发布 FINISHED 后各端关闭连接。

异常结束：EOF、协议错误、launch/完成探测错误或超时停止新 grant，通知其他端、唤醒 pending handle、有界退出。记录 task、缺失成员、最后决定及 pending/inflight。首版用连接和 epoch 超时，不实现心跳服务或恢复；底层无法安全退出时 harness 有界终止进程并标记失败，不能报告任务成功完成或已取消。

## 6. 策略

四种策略共用控制与执行路径，只处理可序列化快照。

| 策略 | 行为 |
| --- | --- |
| StaticOrder | 预先给定 task_id 序列，下一项不 eligible 就等待，不能绕过 |
| DynamicFIFO | 按首次满足成员及顺序条件、进入 eligible 的中央序号选择，ID 仅破同分 |
| DynamicLTF | 选择预计通信后 remaining tail 最大的 eligible，ID 破同分 |
| BoundedLookahead | 基于 LTF，允许为当前前沿中预计很快 ready 的任务有界等待 |

StaticOrder 接受静态 FIFO/LTF 离线序列；其完整输入要求不得扩散到其他策略。Remaining tail 统一指当前通信完成后的预计剩余 job 时间，不计当前已结束的 producer；静态与动态使用相同定义及估计。

Lookahead 首版采用两任务局部估计：比较“立即执行当前候选，再执行未来目标”与“等待目标，先目标再当前候选”，分别计算预计通信完成时间加各自 remaining tail 后的最大值。仅当等待方案更优且预计等待不超过 wait_budget 时等待。这是可解释的启发式，不宣称全局最优。

anticipated 只含线性 job 当前可推进的下一任务，不等待必须先服务当前候选才会出现的后继；不读取真实未来扰动。首次等待建立固定 deadline，事件可提前重算，但不得刷新同轮预算。到期且有 eligible 时强制服务一个候选后才允许新一轮主动等待；无候选则等事件或 epoch 超时。记录目标、估计、分数、截止与回退理由。

## 7. 日志与最小效果对照

本地记录 submit、OFFER、grant 接收、实际提交、完成探测、消费等待。中央记录 OFFER 到齐、首次 eligible、决定、完成回执到齐。中央接收时间与本地/设备事件分开；不把采样到的完成时间当成精确 kernel 结束时间。

决策日志保存候选、anticipated、估计值和理由；区分无候选、静态队首阻塞、主动等待、容量满。

机制验收后完成小规模对照：相同 workload、固定估计、计算扰动样本及 max_inflight=1，运行 StaticOrder 和动态策略，输出每 job 时长、总体时长与等待分解。至少覆盖无扰动和静态队首晚到；不要求动态策略处处获胜。

扰动由 (seed, epoch, job, task, rank) 确定，不依赖线程次序。固定计算时长样本，不固定请求绝对 ready 时间。事件日志用于纯决策重放；相同 seed 的真实多线程运行不保证事件次序完全相同。

旧系统整体对照、多 seed 统计、真实链路收益、bare 并发基线留到后续性能实验。CPU/Gloo 结果不能提前证明 GPU 收益。

## 8. 里程碑与验收

| 里程碑 | 工作 | 通过条件 |
| --- | --- | --- |
| M0：模型与协议 | 身份、GroupSpec、API、事件、grant、结束和错误语义 | 时序明确，无旧 Plan 依赖，动态任务无需全集 |
| M1：纯协调和策略 | 状态机、匹配、顺序、串行容量、四策略 | 可注入时钟的事件检查覆盖选择、静态阻塞、截止回退、直接 submit |
| M2：执行和控制 | 重写 runtime、handle、worker、完成探测、TCP、结束握手 | 两 rank Gloo 结果正确；submit 不等 grant；消费延迟不阻止完成上报 |
| M3：机制与故障 | 确定性场景及日志 | 下述矩阵通过，错误有界退出，无静默跳过 |
| M4：最小对照 | 相同路径和条件运行静态/动态 | 输入可核对，等待和选择差异可解释 |
| M5：GPU 补充 | 两 rank NCCL、producer/consumer stream、完成语义 | tensor 正确，依赖有效，真实完成后释放容量；记录实际软件配置 |

验收矩阵：

- 无 declare 直接 submit，后续任务可在线注册。
- 两 rank 对不同 job 的到达顺序相反，实际 group 提交序列仍匹配。
- 动态服务其他 ready job，StaticOrder 忠实等待未到达的队首。
- 同一候选快照下 FIFO/LTF 作出预期不同选择。
- Lookahead 目标在期限内/外到达时分别提前服务/截止回退；重复事件不延长预算。
- OFFER 未到齐不 grant；上一项全部成员完成前不释放容量。
- 应用晚消费不延迟 runtime 完成上报。
- 模拟不同成员集合与非成员过滤；两 rank 单成员 group 可补真实过滤检查。
- 元数据冲突、非成员、重复提交、重复/冲突 grant、旧 epoch、缺失任务、断连、执行失败明确报错。
- 输入关闭后排空握手，超时报告缺失成员和最后状态。
- 决策事件日志可重放，不要求真实线程调度跨轮完全一致。

M0–M4 通过后标记 CPU 机制完成。GPU 不可用则 M5 明确待验收，不阻塞 CPU 工作，也不宣称 NCCL 路径完成。

## 9. 后续阶段的最小衔接

| 阶段 | 当前边界 | 后续实现 |
| --- | --- | --- |
| Phase 3.2 DAG | job/task 独立；提交顺序不等于完成依赖；计算推进在 adapter | 多前驱、多候选、DAG executor、关键路径估计 |
| Phase 4 trace | 规范/估计与本地绑定分离 | 捕获、标准化、真实 workload 转换 |
| Phase 5 多进程 | 中央只收序列化信息；endpoint 与 job rank、GPU 区分；对象留本地 | 独立部署、跨 host、Job Agent、GPU controller、跨进程提交交接 |
| Phase 6 多链路 | 候选合法性、策略选择、容量检查职责分开 | 资源需求、多资源原子预留、并发集合选择 |

不提前建设资源图、分层调度器或插件框架；现在只实现可验证的 Phase 3.1，后续允许针对需求修改 API。

## 10. 交付物和完成判据

交付重写的核心、四策略、线性 harness、协议/策略测试、两 rank Gloo 验收、最小对照及结果文档。实施时明确旧代码替换范围，无需内部兼容，保留历史实验资料。

完成判据：应用通过新 API 在线提交真实 collective；中央按实际状态改变跨 job 顺序或有界等待；group 序列、结果、完成边界和失败收尾正确；静态与动态能在相同执行条件下对照并解释差异。后续阶段的功能不进入本阶段完成条件。
