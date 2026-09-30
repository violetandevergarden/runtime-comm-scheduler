# Phase 3：完整有效的 GPU 七臂实验实施要求

重写日期：2026-09-29。本文是下一轮实现与实验的执行合同，不是历史实施记录。按用户要求替换原有流水账，以研究目标、需要交付的能力和可核验的准入条件组织工作。本文本身不表示任何门槛已经通过，不授权自动启动长时间实验。

本轮目标是：在同一真实 GPU DAG 工作量上，比较无 JobPacer 性能调度的多在途执行、旧静态 scheduler、新 runtime 的静态和动态策略；同时分清策略选择收益、执行路径成本和并发限制的影响。完整矩阵、语义正确、机制触发和净性能收益分别验收。允许结果为负，但不能把没有选择机会的运行解释为 LTF 效果。

## 1. 距离有效实验还必须补齐的能力

以下是交付要求，不按旧代码是否存在判断完成；已有接口满足要求就复用，仅对缺口实现和验证。

| 必须补齐或重新证明的事项 | 需要的实现与证据 | 不接受的替代 |
| --- | --- | --- |
| 合法的无调度多在途基线 | bare 的共同顺序、异步提交、独立完成观察及真实 NCCL 资格 | 用单在途 raw-ordered 充当 bare |
| 有实际调度机会的 benchmark | 重建输入生成器；在实际决策点记录多个 eligible 候选、不同分数和策略分歧 | 图看起来有分叉、只在离线模拟中存在竞争 |
| 对策略有区分力的场景 | 分开构造静态队首失配与 FIFO/LTF 选择差异，保留无收益对照 | 单候选时动态 LTF 比静态快就称为 LTF 胜利 |
| 可解释的默认与静态顺序 | 分层拓扑、同层 job 轮转；bare 与 old/new static FIFO 共用；LTF 共用另一冻结序列 | 每个 engine 自己排一遍或整 job 优先而不披露 |
| 可比的测量 | 统一应用终点、低扰动性能模式、噪声与观测消融 | full trace 时间直接视为部署性能或固定扣除日志成本 |
| 可复现且不可混批的数据 | 输入/profile/order/源码冻结，完整块重试和恢复，独立校验 | 不同合同、源码或 attempt 的样本拼接 |
| 明确的正式准入 | 软件正确性、backend、机制、测量、七臂彩排逐项通过 | 仅进程退出成功或矩阵计数完整 |

本文明确下一轮第七臂为多在途 bare。它取代既有七臂计划中将串行 raw-ordered 作为替代臂的下一轮目标；实施时同步修订计划、CLI、suite、资格工具和分析合同，不覆盖旧批次定义。其他阶段约束继续遵循[统一 GPU DAG 设计](../plan/phase3-gpu-workload-and-scheduling.md)、[Phase 3.1](../plan/phase3.1.md)与[Phase 3.2](../plan/phase3.2.md)。

## 2. 固定研究问题与七臂

| arm ID | 顺序与决策 | 执行与容量 |
| --- | --- | --- |
| bare-ordered | 冻结框架默认顺序，无性能准入策略 | 直接异步发射，不施加全局单在途限制 |
| old-static-fifo | 共同静态 FIFO Plan | 旧 scheduler，配置本地单在途 |
| old-static-ltf | 共同静态 LTF Plan | 同一旧执行路径、本地单在途 |
| new-static-fifo | 共同静态 FIFO 序列，队首未 eligible 不跳过 | 新 runtime，全成员完成释放全局单在途容量 |
| new-static-ltf | 共同静态 LTF 序列，队首未 eligible 不跳过 | 同一新 runtime |
| new-dynamic-fifo | 按首次进入 eligible 的顺序选择 | 同一新 runtime |
| new-dynamic-ltf | 按统一 comm+tail 估计选择 | 同一新 runtime |

裸发允许必要的身份、默认调用顺序和 CUDA 依赖保障；无性能调度不等于任意跨 rank 交错。bare 不得通过新 coordinator 或旧 scheduler 再关闭日志实现。旧本地完成与新全成员完成不是同一容量合同，必须记录实际释放与在途行为；old/new 差距称为整体执行路径差异，不称为纯 policy 或纯中央网络成本。

主要对照在冻结前指定：new static FIFO→dynamic FIFO、new static LTF→dynamic LTF，分别看 makespan 与 mean JCT。系统净效果另行固定报告 old static→对应 new dynamic、bare→各 scheduler。动态 FIFO→动态 LTF 用于有竞争证据场景的策略辨别，不能从所有交叉组合中事后挑赢家。

保留少量同顺序的串行直接执行诊断，与 bare 对照以分离在途限制影响；不自动扩成第八个正式 arm。研究目标不要求 LTF 一定胜过 FIFO，也不要求 scheduler 一定胜过 bare。

## 3. 共同执行与 bare 的必要实现

所有场景直接使用共同 DagInput/DagGraph/DagRunner。线性场景也是链式 DAG，不另建 linear runner 或桥接执行循环。计算为固定工作量 CUDA 程序，使用同样的 buffer、初始化、warmup、dtype、precision、依赖和终点；不要以 host sleep 假装 GPU 计算。

任务规范、策略估计和本地 tensor/event/closure 分离。保留首版每 job 一个活跃 compute 的资源约定，多个 job 可以并发推进；真实 GPU 资源竞争进入测量。若这项约定限制了候选生成，应修改场景或另立执行合同，不能只让某个 arm 使用另一计算推进方式。

Bare 要求：

- 启动前核验共同输入、task 身份、成员、组内 group_seq 和默认通信序列；默认顺序不得依赖 rank 本地 ready 到达或真实未来扰动。
- submit 接受保存请求后返回 handle；唯一 dispatcher 按本地投影发射，队首等待不持有阻止计算、提交或完成观察的锁。
- dispatcher 不循环等待前项物理完成才发后项；独立 completion observer 更新应用依赖，保留 buffer 至正确完成边界。
- 明确多 communicator 的 backend 顺序机制。核对实际 PyTorch/NCCL/CUDA 版本、有效配置与调用路径；不能只凭 host 顺序相同或环境变量字符串宣称安全。
- 如果采用 implicit launch ordering，启动前检查支持条件；不支持时拒绝该配置，不暗中退回串行并继续标记 bare。
- 提前 close、缺失队首、binding/launch/probe 失败均停止新发射、唤醒 handle、保留首因并有界退出。已提交通信不伪装取消，外层进程 timeout 作为最后兜底。

Bare 的必要顺序约束可能产生队首等待，必须计入应用时间。多在途 API 能力、同一时段未完成 receipt、设备 kernel overlap 和最终性能收益分开报告。peak inflight 受观察轮询影响，不足以单独证明设备并行。

## 4. 重建 benchmark：目录、输入与生成器

新建独立版本目录，例如 `benchmark/phase3/experiments/gpu-seven-arm-v2/`，采用 template、pilot、formal、profile/order manifest 的清晰结构；原始结果另存 `benchmark/phase3/results/gpu-seven-arm/<batch-id>/`。重建不是删除旧数据或重复实现公共 runner。旧输入和结果留在各自版本，不作为新合同的默认输入。

生成器必须输出可单独运行的完整 manifest，不依赖运行时临时改参数。每个样例至少包含：scenario/sample ID、graph、group 成员与序号、buffer 定义、逐节点 compute program、comm binding、submit_after、默认顺序规则版本、工作量样本 hash。profile 后另生成估计视图与静态顺序，记录依赖关系和 hash。

拓扑、计算时长、通信大小和偏斜应是明确参数，场景不能只是更换名称或随机微调同一条图。每个参数说明它改变哪种机制。以 bytes、算子形状与 repeats 表达真实工作量，记录实测微秒/毫秒范围，不把 GPU 校准值当作绝对不变的执行时间。

随机性要求：

- 默认五个正式 workload seeds（8101–8105），至少覆盖三个不同实际工作量样例；五个样例必须公布结构/工作量差异，不能只有 seed 字段不同。
- Pilot 使用独立的至少三个 seeds（建议 9101–9103）；正式 seeds 不用于挑选参数或调到出收益。
- tensor 初始化用稳定、版本化 domain hash，从 scenario/sample/job/buffer-or-node/rank/stage 派生；workload、tensor 和运行顺序随机流分离。
- run ID、epoch、repeat 不改变工作量或 tensor 数值；七臂读取同一展开样例。扰动施加到实际工作量，不固定所有任务绝对 ready 时间。
- 不把真实执行扰动泄露到策略 hint。策略只读冻结的独立估计；需要研究预测误差时显式分离 nominal 与 execution 参数。

## 5. 重建 benchmark：六个场景的机制合同

最少两个链式 DAG、四个一般 DAG。下表给出必须覆盖的研究作用；节点数量不是门槛，实际可观察的执行机制才是门槛。

| 场景 | 构造要求 | 必须检查的机制 |
| --- | --- | --- |
| L0 稳定链对照 | 两个左右 job，近似均衡、可预测的计算通信链 | 静态序列合理，无人为强制收益；测执行成本和噪声，允许没有多候选或加速 |
| L1 静态队首失配 | 不同 job 的 producer/workload 偏斜使静态队首晚 ready，另一任务能执行 | 新 static 确有容量空闲但队首未 eligible 的等待；dynamic 在对应条件下推进其他合法任务 |
| D0 分叉汇合与重叠 | producer 后分出通信和独立 GPU 计算，join 消费正确结果 | 独立计算不被准入阻塞，join 等真实依赖；至少有跨 job 独立通信用于 bare 多在途检查 |
| D1 优先级竞争 | 至少两个合法通信前沿同处决策点，comm+tail 不同，FIFO 先到项与 LTF 最大分项不同 | 真实双卡上多候选、分数区分、FIFO/LTF 对同一候选快照给出不同选择，实际 LTF 选择符合规则 |
| D2 动态错位与后继链 | 一般 DAG，有跨 job 工作量偏斜及不同后继计算，多个通信阶段 | 静态序列失配与后继计算解锁可追踪；用来检验 D1 以外的选择机会是否存在和是否影响关键路径 |
| D3 顺序与多终点 | 同 group 规范顺序、多 group、多分支及多个 sink | 不跳 group_seq、不漏 sink；候选即使受组内限制也不制造非法并发，终点涵盖完整应用工作 |

不能只让图含 fork/join 就认定测试了通信选择。同一 group 的后续任务可能不合法，不得将它计为第二个 eligible 候选。不能强行放宽核心 group_seq、容量或物理完成合同来通过机制门槛。

D1 应优先使用一个有真实工作量的前驱/占用阶段，使独立前沿在下一次容量释放前有机会准备并被接收；这些必须是 workload 的真实节点，所有 arm 共同执行。具体需要多大通信、多少计算工作量，由独立 profile 与控制路径尺度决定，不沿用 CPU sleep 数字。不要为某个策略增加专用 barrier、延迟反馈、注入 OFFER、隐瞒完成或读取未来 ready 时刻。

机制级确定性测试可用事件/屏障强制候选同时到达，以检验 policy 正确性；这种合成测试不充当正式应用 benchmark。正式 D1 必须在真实 DAG 执行中自然形成足够稳定的竞争。

## 6. 默认顺序、静态计划与图合法性

默认顺序采用版本化的“完整约束图拓扑分层，同层 job 轮转，同 job 依输入节点顺序”规则。Bare 与 old/new static FIFO 读取同一冻结通信投影；LTF 的 old/new static 读取同一独立冻结 LTF 序列。复用图工具和 adapter 接口，不为每个 engine 重建排序算法。

层级是离线排序规则，不是运行屏障。不能等整层完成才允许下一层推进，不能把调用序列变成计算完成串行边。输入验证需同时考虑 DAG 传递依赖、group_seq、submit_after 接受关系和计算推进限制，拒绝队首依赖后序通信、缺项、重复和成员不一致。

FIFO 按首次 eligible 选择，LTF 的 comm+tail 定义和 tie-break 明确固定；静态估计与动态 hint 来自同一估计视图。声明或节点输入次序不能代替 dynamic FIFO 的首次 eligible 次序。

Pilot 用少量默认/反转 job 起始顺序对照检查基线偏置；不按速度挑较有利的顺序。正式冻结一种默认规则，并公布其等待影响。将来真实框架 trace 提供调用顺序后，可另建版本替换手工默认约定。

## 7. Profile、时间尺度与机制门槛

先在分配的目标双卡上测通信签名和 CUDA compute program，记录 shape/dtype/repeats、设备 UUID、精度、版本、预热、样本分布及原始值。只测独立服务时间不足以预测竞争，另用少量代表组合观察计算与通信互相拖慢的幅度。不得临时安装升级 PyTorch/CUDA/NCCL 或覆盖设备分配。

使用少量零计算链估计本机逐轮执行成本，选择至少能区分“控制开销主导”和“设备工作较长”的诊断尺度。正式输入参数在 pilot 结束前冻结；不通过盲目放大消息或计算直到出现 speedup。记录 profile 与运行中时长差异，缺少必要签名时拒绝正式执行，不静默使用默认估计。

机制资格至少包括：

1. 所有场景、所有 arm 的任务、数据、依赖和发射语义通过。
2. D1 在至少三个独立 pilot seeds、每 seed 三次机制诊断中，至少两个 seeds 各有两次及以上出现真实多候选、不同分数、FIFO/LTF 反事实选择分歧，且实际选择符合所用 policy。此门槛预先固定，用于避免偶然一次竞争；不要求任何加速。其余未触发样例全部保留。
3. L1 在同样的 pilot 样本范围中，至少两个 seeds 各有两次及以上证实静态队首阻塞和动态合法绕过；通过单时钟事件链说明，不能只用总时间差代替。
4. 专用独立通信用例证明 bare 没有前项完成门控；D0 检查真实计算/通信分叉与终点。设备重叠有无分别报告，不作为策略必须获益的门槛。
5. 对六场景的全部动态 dispatch 统计候选数量分布；D2 是否出现优先级选择如实记录。不得仅检查 D1 后概括其余场景也有调度空间。

新增机制分析器将 decision 与对应 policy snapshot、候选分数、选择、capacity release、OFFER/eligible 和实际 launch 关联。离线在同一 snapshot 上计算 FIFO/LTF 选择，识别真正的策略分歧；比较两个独立 replay 的顺序不同不能单独证明策略造成差异。

Pilot 每个场景最多两轮预先记录的参数修订；仍不过则停止该场景晋级，回到图结构/执行路径根因分析，另立设计 revision，不以无限 pilot 调参或绕过 gate 取得“有效正式”标签。显式要求带未通过 gate 运行时，只能另标描述性批次。

## 8. 测量合同与噪声控制

应用时间统一为 release 到所有应用 sink 物理完成的本地时长；makespan 按既定 rank 聚合规则取最大，mean JCT 先按 job 聚合 rank 再平均。独立保存通信 drain、准备、warmup、校验及完整进程 wall time，不能把进程启动时间混入应用指标。不同 rank 未同步的绝对时间不相减。

正式性能使用经验证的 minimal 模式，仍保存身份、实际 launch 投影、数值/终点校验和低成本计数。详细 full/profiler 另起配对诊断样本，不并入主性能统计，不从 full 结果扣一个假定日志成本。确保首次完成时间不会被后续轮询覆盖。

正式前测两类扰动：

- 同一 arm 的 A/A 标签重复：选代表场景，至少五个交错配对，检查输入一致、标签无行为差异、运行顺序漂移和测量噪声。
- 同输入同 arm 的 minimal/full 配对：至少一个 raw/bare、一个旧路径、一个新路径，每个至少五对；记录时间、CPU、事件量和上下文切换。统一模式名称不意味着统一扰动。

这些是最小诊断量，不构成精确噪声界。若策略差异小于观测到的波动，应报告精度不足，优化测量或按预先规则补独立样本；不能要求统计显著后才允许冻结，也不能重复直到显著。Full 中可见的机制需用低成本计数确认 minimal 执行下没有系统性消失。

逐项记录 API 提交、设备完成观察、容量释放和应用消费，不能把观察延迟当作纯 kernel 时长。必要时使用已有可用的 PyTorch profiler 等设备 trace；没有 nsys 不等于只能凭 peak inflight 推测重叠。

## 9. 批次工具、正式采样与独立校验

Suite、CLI、资格工具、runner、analyzer 必须共同识别七个真实 arm ID 和合同版本，拒绝以 raw-ordered 替代 bare。裸发容量单独记为不施加 admission 单在途限制；不能在所有 arm 顶层统一写 max_inflight=1。

正式默认六场景 × 五 workload seeds × 五 repeats × 七臂，共 150 个完整配对块、1,050 次 replay。Warmup、资格、pilot、profiler、失败尝试不计入这一数量。所有正式 seeds 和 repeats 保留；正式样例没有触发某机制也不得剔除，并报告触发率下降与解释边界。

同 block 的七臂读取同一输入/profile，运行顺序由独立 order seed 平衡并冻结，逐 run 串行启动以避免实验之间资源竞争。冻结源码、依赖环境、设备、group、输入、profile、估计视图、顺序、观测模式、timeout、warmup、规则版本和命令；源码改变即新 batch revision。

启动与恢复要求：

- 发射前预留 raw/log/attempt 路径，运行 ledger 可靠追加；已有冻结目录不可静默覆盖。
- 仅对预注册的启动环境故障做有界重试；端口冲突重试须清理整个子进程组，确认没有遗留 communicator 后换新 rendezvous。部分 rank 已进入应用时不得当成未启动直接重跑。
- 失败 attempt 完整保留；成对分析只用同一 block attempt 的七臂，不跨 attempt 拼接。中断恢复重新检查源码与输入 hash。
- 明确外层整块重试与内层启动重试的合计上限，避免嵌套重试无限放大；保留所有端口和返回码。

独立校验不只读取 raw 的 status：验证输入/源码/raw hash、任务全集、成员、参数、组内和跨组实际 launch 投影、静态序列、buffer 数值、终点、退出码、超时与失败归类。需有一次可控启动故障后的恢复彩排及中断恢复检查；不能以正常运行未触发重试证明恢复正确。

正式启动前做至少两个代表场景、每场景一个完整七臂块的彩排；其余四个场景也须有七臂语义覆盖。先测 wall time/磁盘预算，再执行正式批次。运行中采集设备占用与环境变化，不依据输赢剔除样本。

## 10. 分析要求：同时回答策略与系统净效果

所有 arm 给出六场景的原始分布、seed 内 repeat 中位数、跨 seed 汇总、mean/逐 job JCT、失败率和实际完成规模，不只给新 static/dynamic 的比值。

对配对块先计算 delta 与 ratio，再每 seed 汇总 repeats，最后按 seed block 构造区间。默认 bootstrap 2,000 次，固定分析随机种子；每场景五个 workload seeds 是有限独立样本，不把 25 repeats 或跨场景 150 对当作同一总体的大样本。预先列明主要/次要比较；未做多重比较校正的区间只作探索性证据，不能从众多区间挑选普遍优势。

新 dynamic 是否抵消执行成本，逐配对使用：

```text
旧 static 时间 - 新 dynamic 时间
= (新 static 时间 - 新 dynamic 时间)
  - (新 static 时间 - 旧 static 时间)
```

这是同一配对样本的时间恒等式，不是完整因果分解，不能拿各 arm 独立汇总中位数相减代替。裸发与 scheduler 还同时改变容量与顺序等待，应明确解释范围。

每个性能结论同时给出机制证据：候选数、分数差、选择分歧、静态 HOL、容量空档、后继计算解锁、观测扰动。没有多候选的场景只支持 ready-driven 执行与固定顺序的比较；LTF 没有获得选择机会就不报告 LTF 优先级收益。

原始产物需包含机器可读 gate report，分开字段：`matrix_complete`、`semantic_checks_passed`、`mechanism_gates_passed`、`measurement_checks_passed`、`net_performance_result`。不以一个 complete=true 混合这些含义。

## 11. 执行顺序与完成标准

| 阶段 | 要完成的工作 | 晋级证据 |
| --- | --- | --- |
| E0 合同统一 | 七臂真实身份、baseline 顺序、容量与计时定义、scope 明确 | schema/CLI/manifest/analyzer 一致，不存在替代臂冒名 |
| E1 软件与 backend | bare 多在途与关闭/失败修正、共同 runner 回归、版本检查 | 单元及目标双卡正常/异常检查；CPU/Gloo 回归分列 |
| E2 重建 benchmark | 六图模板、确定性生成、独立 pilot seeds、真实 profile | 输入可复现、估计不读未来、所有依赖与顺序合法 |
| E3 机制与测量 pilot | D1 竞争、L1 HOL、D0 分叉、全场景候选统计、A/A 与观测消融 | 达到预设触发门槛，噪声与记录口径可解释 |
| E4 冻结与彩排 | 全部正式输入、七臂接入、版本和预算冻结、恢复彩排 | 全部必要 gate 通过，失败/中断可追溯恢复 |
| E5 正式采集 | 冻结的 150 块、1,050 replay | 完整块和逐项语义独立校验，失败完整保留 |
| E6 分析归档 | 主要配对、系统净效果、逐 job、机制与边界报告 | 命令/源码/输入/raw/hash 可恢复，结论不超出证据 |

每阶段产出简短机器可读检查结果与执行记录，遇到根因先修复并重做受影响验证，不必重复无关的大规模检查。记录 pytest 的通过/失败/跳过及实际命令，最终源码变更后旧记录不能替代当前回归。使用已有 .venv，不擅自安装升级依赖。

必须停止正式晋级的条件：bare 不满足 backend 合同、数据或依赖错误、D1 必需机制未触发、测量口径不一致、源码/input 漂移、七臂身份错误、失败无法有界退出或恢复可能混批。单个策略更慢、设备没有实际 overlap、统计区间跨零不是失败理由，而是应如实报告的结果。

本文按用户要求不保留此前实施流水账；不删除其他历史文档或 benchmark 产物。本轮只重写执行要求，不在此宣称 benchmark 已重建、代码已实现或实验已通过。
