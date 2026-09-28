# Phase 3 GPU workload 与调度：逐文件实施方案

日期：2026-09-27。状态：M1/M2/M4/M5 已实现并完成受限 smoke；M0 的 H 回归已修复，Phase 2 profile replay 复跑通过，但一次超时未找到稳定根因；M3 配对性能实验未启动。执行事实、命令、环境与未验收项见[本任务结果记录](../result/phase3-gpu-workload-implementation-20260927.md)。依据 [设计提案](../plan/phase3-gpu-workload-and-scheduling.md)、[Phase 3.1](../plan/phase3.1.md)、[Phase 3.2](../plan/phase3.2.md)、[已有 GPU 实施合同](phase3-gpu-experiments-and-fixes.md)与[首轮结果](../result/phase3-gpu-20260927.md)。实际能力以工作树代码和本文记录的重新验证为准。

## 1. 基线、边界与交付顺序

目前 `runtime_worker.py` 的旧 S 路径使用 `fill_(rank + 1)` producer、统一矩阵尺寸与次数的 independent matmul，且 independent 在 `runtime.submit()` **之前**入队；`replay_worker.py` 也保留旧 S 合成路径。现有 `CudaMatmulProgram` 只有独立 matmul；旧 `Workload` 的 `consumer_compute_s` 原本就是提交通信后、等待通信前的独立计算。现有 DAG compute 回执在设备完成后才解锁节点，但 `DagRunner` 的普通完成边及 ready 遍历不承诺本提案的 host 提交次序。首轮 G4/G5 的事实和限制继续保留，不能将其改名为新合同的验收。

按以下依赖顺序交付，每一阶段冻结输入、命令和源码摘要后再进入下一阶段：

1. **M0：恢复基线**。确认 H 路径回归修复和 Phase 2 超时归因；重跑受影响的 H、旧 S、Gloo 检查，记录仍失败的既有 fixture。这里不改新 schema 的语义。
2. **M1：输入、程序和校准**。加入独立 GPU schema、具体 CUDA 程序和 compute profile；先完成纯解析、设备数据流和数值验收。
3. **M2：线性 GPU 执行**。接入新 worker 分支，仅 Static FIFO / Dynamic FIFO；真实双卡验证依赖、异常、重复 epoch 和结果全集。
4. **M3：小规模配对对照**。固定 manifest、profile、seed block、warmup、poll、观测模式和终点，先机制 smoke 后正式对照。实验批次大小另行冻结；本文不启动实验。
5. **M4：等价 DAG bridge**。同一片段程序和 seed 映射，显式保证 host 提交顺序并逐项核对数据依赖及 terminal。
6. **M5：GPU LTF / Lookahead**。在校准和误差记录可用后接入版本化估计；必要时再独立设计在线摘要更新。

每个里程碑的实际通过、失败、跳过、时长与产物写入 `result/` 新文档；本文件只规定实施和判定方式。保留已有未提交工作、历史输入和原始数据，不覆盖首轮结果。

## 2. 数据合同与文件修改

### `examples/jobpacer/runtime/gpu_workload.py`（新增）

建立独立的 `jobpacer-gpu-linear` / `schema_version=1` 解析器与冻结数据类，不扩展旧 `Workload.from_dict()`。根对象包含 `name`、`execution_seed`、`groups`、`execution_contract`、`jobs`；每个 segment 包含稳定 `segment_id`、`group_id`、全 epoch 同 group 的 `group_seq`、`producer_compute`、`collective`、可为 null 的 `independent_compute` / `dependent_compute` 及 `estimates`。解析后生成按键排序、数值规范化的 canonical JSON 与 SHA-256，保留输入路径和原始摘要。epoch 是运行参数，不由 rank 到达次序生成。Task ID 固定为 `job_id/segment_id`，跨 rank 不变。

只接受 `fill` / `matmul` producer、`matmul` independent、`sum_join` dependent；首版 float32、连续二维矩阵、SUM all-reduce、`max_inflight=1`、`physical-ready`、`comm-request-before-independent`、`terminal-event-complete`。`fill` 的常数由规范 seed 键计算；`matmul` 的 `(m,n)` 必须等于通信 shape，`k`、repeats 为正整数。`numel` 与 `num_bytes` 从 shape/dtype 复核，拒绝自相矛盾值。`sum_join` 只能引用存在的通信输出或独立输出；null independent 时不能引用它。拒绝未知字段/算子/角色、重复 job/segment/group/task/序号、缺失成员、非连续 group 序号、布尔冒充整数、非有限估计、危险 buffer 别名和不支持的合同。各 rank 的共同通信规范来自同一 canonical manifest；本地 rank 差异首版只体现在确定性输入值，暂不接受任意 rank 工作量覆盖。

`estimates` 保存 profile 引用或显式标记的 FIFO 占位版本，不保存实际执行时长。解析器不允许 `_compute_s` 出现在新 schema 的执行字段。保留旧 sleep schema 原样；不能将旧 S 的全局 `--compute-matrix-size/--compute-repeats` 静默覆盖新片段参数。

### `examples/jobpacer/workloads.py`（保持旧格式）

只在模块注释或文档中明确 `consumer_compute_s` 是旧独立计算及旧 schema 的适用范围；不迁移旧 manifest、不改变 `linear_execution_duration()` 的历史采样键。新入口直接调用 `gpu_workload.py`。这样旧 H/旧 S 与新合同能在结果里可靠区分。

### `examples/jobpacer/runtime/gpu_compute.py`

保留 `CudaMatmulProgram` 供旧 S 和旧 DAG 使用；新增具体 `FillProducer`、`MatmulProducer`、`IndependentMatmul`、`SumJoinConsumer` 及每片段 `GpuSegmentResources` / compute receipt。资源对象拥有通信 buffer、私有矩阵、独立输出、标量 join 输出、producer/compute/consumer stream、初始化与阶段完成事件、keepalive，以及用于数值参考的冻结输入摘要。producer matmul 最后一次 `torch.mm(..., out=comm_buffer)` 直接成为 all-reduce 输入；repeats 是固定覆盖次数，不把前一次输出递归输入下一次。independent 只读私有输入，输出不与 in-place all-reduce buffer 别名。`sum_join` 在 consumer stream 上读取通信输出及存在的 independent 输出，保存可验证标量；null dependent 仍记录 join/terminal event。

资源创建在 release 前完成。初始化使用显式 init event，让实际读写的非默认 stream 等待；warmup 精确覆盖将用的算子签名，并在 release 前确认结束、重置被改写的正式 buffer。派生 seed 使用 `(execution_seed, epoch, job_id, segment_id, rank, stage)`，不使用 DAG node 的偶然名称或线程次序。程序重复使用前确认上次 terminal 完成。每阶段 submit 返回带 done event 的回执；host `query()`/`elapsed_ms()` 的用途分开，query 异常按失败处理。准备、warmup、数值参考构造与正式应用计时分别记录。固定 PyTorch matmul precision / TF32 配置并写入结果。

### `examples/jobpacer/runtime/runtime_adapter.py`

新增针对 `GpuLinearInput` 的 `TaskSpec`、`LocalBinding` 与 hint 构造函数，不把 tensor/event 放入 `TaskSpec`。`LocalBinding.tensor` 使用 producer 的通信 buffer，`producer_event` 使用 `producer_done`；设备和 ProcessGroup 只在本地绑定。group 建立顺序来自 canonical groups，而不是 job 顺序。Static FIFO 在生成 bridge DAG 后，按 job 片段依赖边和 group 序号边做确定性拓扑排序，并校验全集与边约束；不同 group 的 `group_seq` 不作全局比较。Dynamic FIFO 的 eligible 次序仍由 coordinator 根据实际 OFFER 判定。

M2 的 `TaskHint.ready_after_s=None`，不在 producer 前 DECLARE；FIFO 可以使用严格通信 profile 的 `estimated_comm_s` 与 `gpu-fifo-unused-v1` 零 tail，占位版本必须写入输出，配置层拒绝用于 LTF/Lookahead。M5 增加独立纯函数 `gpu_linear_static_tail_v1()`：`segment_estimate_i=p_i+max(c_i,u_i)+d_i`，`tail_after_comm_i=max(u_i-c_i,0)+d_i+sum(segment_estimate_j)`（后继片段）；静态与动态 GPU LTF 共用它。估计来源必须是先冻结的 compute/comm profile，不读取本轮未来扰动、设备实际完成或旧 sleep 字段。旧 `linear-postcompletion-tail-v2` 保持历史含义。

### `examples/jobpacer/runtime/gpu_compute_profile.py`（M1 新增）

保存计算 profile 的版本化签名和解析/校验：算子、m/n/k、tensor shape、dtype、layout、repeats、输出角色、precision/TF32、设备 UUID、PyTorch/CUDA 版本及采样方法。fill 与 sum_join 签名必须包含实际 tensor shape、dtype、layout；schema v2 拒绝缺少尺寸或不匹配的旧签名。分别记录 device event 区间、host enqueue、准备成本与样本分布；不把 event 区间叫纯 kernel 时间。由 `examples/jobpacer/scripts/run_gpu_compute_profile.py`（新增）执行离线校准并输出 JSON 与摘要；正式 replay 仅加载已冻结 profile，签名或环境不匹配时明确拒绝估计驱动策略。通信校准仍复用 `comm_profile.py`，但校验 group 成员、backend 和消息签名。

## 3. 线性执行与命令入口

### `examples/jobpacer/runtime/runtime_worker.py`

在 `run_rank()` 最早的输入分派处加入 GPU schema，不能让 `load_workload()` 或 `load_dag()` 猜测格式。`_new_groups()`、warmup、预分配、静态顺序、预期 task 集合和结果汇总增加 `GpuLinearInput` 分支；所有 rank 在 release 前核对 canonical 摘要与设备映射。rank 使用继承可见空间的逻辑 `cuda:rank`，记录 UUID，不覆盖 `CUDA_VISIBLE_DEVICES`。

新增 `_prepare_gpu_linear_segments()` 与 `_run_gpu_linear_job()`，与旧 `_run_linear_job()` 并列，避免悄悄改变旧 S 结果。每个 job 顺序执行片段，job 之间可并行；共同固定 deadline 由 epoch/replay 传入，不因事件更新而刷新。单片段严格执行：

```text
前片段 terminal query 成功 → producer 入队 → producer_done physical-ready
→ runtime.submit(spec,binding,hint) 返回 handle
→ independent 在自己的 stream 入队并记录 independent_done（可为 null）
→ handle.wait_on(consumer_stream) 建立通信依赖
→ consumer_stream.wait_event(independent_done)（若存在）
→ dependent 或空 join → terminal_event → host query terminal
```

因此需把当前 S 路径的 independent-before-submit 写法留给 legacy 分支。submit 后立刻入队 independent，不等 grant；`wait_on()` 可能等待 BOUND，必须排在 independent 后。`wait_on()` 返回只说明依赖已入队，terminal query 才是片段应用终点。通信完成反馈由 runtime 的探测线程独立上报，不能由应用消费触发。null dependent 仍需要 terminal event；null independent 时只依赖通信。失败时调用公共 `runtime.abort()`、停止新节点和通信、唤醒等待者；不把已 grant 的 collective 当成取消。输入关闭只由 rank 共同拥有者在全部本地 job terminal 后调用一次，随后单独等待协议 drain。

结果结构为每段输出 task/segment ID、合同版本、程序签名和 seed 摘要、host submit/return、producer physical-ready、独立入队、consumer 依赖、terminal、各阶段 device event 时长、handle/collective 状态和数值校验。校验只在所有本地 job 到达 application end 后执行，并记录 task/rank 的独立 validation 时间区间；父进程拒绝 validation 与 application 重叠。minimal 只留正确性及必要聚合；diagnostic/profiler 独立运行。应用 makespan 采用 `max_rank(local_end-local_release)`；不相减不同 rank 的原始时钟。数值参考在计时外构造，matmul producer 不再套用旧 rank 常数公式；通信、独立输出和 dependent 标量分别断言。

### `examples/jobpacer/scripts/run_phase3.py`

增加互斥 `--gpu-linear <manifest.json>`，保留 `--workload` / `--dag` 与历史参数。新输入只允许 NCCL、两张可见 GPU、S lane、precreate、on-submit、`max_inflight=1` 与 M2 的 `static_fifo/fifo`；与 `--compute-matrix-size`、`--compute-repeats`、host jitter、旧 sleep/DECLARE 配置冲突时报错。若 argparse 默认参数会被误判为用户覆盖，应以参数是否显式传入或独立配置对象判定，而非比较默认值。M5 验收后才开放 `ltf/lookahead` 并要求匹配的 profile/estimator version。把 schema、合同、输入/源码/profile 摘要和每 rank UUID 送入父子结果；失败时保存部分结果与退出原因。

### `examples/jobpacer/runtime/replay_worker.py` 与旧启动脚本

修复 H lane 在 host-sleep 分支漏设 `application_wait_start_ts` 的路径；Phase 1 Gloo 回归复跑通过。Phase 2 profile→replay 的一次初跑发生双 rank timeout，随后带线程栈诊断的直接 replay 和集成回归均通过，未复现稳定根因，因此本轮不声称已归因或消除该偶发超时。M3 若需要旧新 GPU 对照，旧 worker 必须直接复用 `gpu_workload.py` 与 `gpu_compute.py` 的同一资源和提交/消费合同，再命名为新 old-runtime GPU arm；不能拿现有旧 S 合成路径当配对 arm。无此需求时仅保留 legacy 标识和已有功能。`run_phase2.py`、`run_experiments.py` 不接收新 schema；现有 CUDA 设备辅助模块沿用已获准映射。

## 4. DAG bridge 的具体落点（M4）

### `examples/jobpacer/runtime/gpu_dag_bridge.py`（新增）

从同一个 `GpuLinearInput` 生成 `DagGraph`：`previous_terminal→producer`，`producer→comm/independent`，`comm+independent→dependent_or_join→next_producer`。每个 DAG node 附有指回规范 `(job_id,segment_id,stage)` 的本地映射，复用同一 `GpuSegmentResources` 构造、seed、初始化与校验代码；不从 DAG node ID 重新采样。桥接输出保存生成图 canonical 摘要及线性输入摘要，并用 `validate_graph()` 校验 group 顺序与联合环。生成图的 compute 估计来自相同 profile，但 M4 只比较 FIFO。

### `src/runtime_comm_scheduler/dag/runner.py` 与 `examples/jobpacer/runtime/runtime_worker.py`

普通 `DagRunner` 仍使用完成依赖。GPU bridge 由示例侧 DAG 推进器消费已校验 `DagGraph`：producer CUDA event 完成才解锁 comm/independent；comm `submit()` 返回后才启动 independent；comm 物理完成由 runtime handle 的物理完成状态单独观察；independent 用自己的 CUDA event 查询；join 在通信 stream 依赖与 independent event 均已排入后进入 consumer stream，terminal event 完成才解锁下一片段 producer。每个图节点在推进时直接记录提交/完成事件，不在任务结束后回填；父进程校验 node event 全集与完成状态。桥接器保留 GPU event 依赖语义，不把 GPU workload 塞进 CPU `DagRunner`，DAG 模型仍位于核心上层。该实现覆盖当前线性桥接图，不代表任意 DAG 执行器或性能等价已验收。

`dag/model.py` 的 `TaskSpec`、group 校验及节点类型原则上不变；`dag/__init__.py` 只在新增公共接口时更新导出。线性与 bridge 的预期通信任务、group launch 投影、程序签名、输入值摘要、join 条件、terminal 定义逐项比较；JCT 数值无需相同。旧 G5 bridge 继续标注为旧输入。

## 5. 通信核心、策略与结果处理

`src/runtime_comm_scheduler/runtime/{model,coordinator,policy,runtime,executor,handle}.py` 首版不为新 schema 改协议：继续同 group 共同身份/序号匹配、单事件循环维护容量、共同 grant、有序本地 launch、SUBMITTED 先于 COMPLETED、全成员物理完成才释放容量。`TaskSpec` 不携带 GPU 程序。若 M2 发现具体回执或生命周期缺陷，先写能复现根因的检查，再做最小修复并重验 H/S、Gloo、NCCL 异常路径；不得用 consumer terminal 释放通信容量。StaticOrder 队首未 eligible 必须等待，FIFO 容量满时继续维护首次 eligible 顺序。

M5 在 `runtime_adapter.py` 构造 GPU tail/ready hint；静态 LTF、动态 LTF 和 Lookahead 前沿排名固定使用 `estimated_comm_s + remaining_tail_s`，共享 `runtime.policy.ltf_score()`，不由任务字段选择公式。线性和 DAG 的 tail 构造可以不同，但 tail 都排除当前通信，所以分数显式加回当前通信。Lookahead 只对已经启动且尚未 physical-ready 的 producer 声明有来源的预计剩余时间；后继等待未完成通信时不声明有限 ready。固定等待 deadline、晚到回退、profile 误差和控制传输/轮询误差分别验收。估计出错不能通过读取本轮实际未来时长修正。

`examples/jobpacer/analysis/runtime_results.py`、`visualize_phase3.py` 与批处理入口在 M3 只接受同合同/输入/profile/seed block 的配对；输出应用 JCT/makespan、submit→grant、grant→调用、物理完成、terminal、失败与跳过。保留不同 clock 域名称；device event 时长不称纯 kernel 时间，profiler trace 单独诊断。`benchmark/phase3/experiments/gpu-linear/`（新增）保存冻结的 smoke、错位到达、可选 null 阶段和 bridge 输入；`benchmark/phase3/results/` 的 raw、manifest、命令、源码快照及 SHA-256 独立归档，因为目录被 Git 忽略。

## 6. 验证位置、条件与退出准则

| 修改/新增位置 | 针对性检查与必须观察的事实 |
| --- | --- |
| `tests/unit/test_jobpacer_gpu_workload.py` | schema 正反例、规范摘要、group 全 epoch 序号、shape/bytes、角色、null、估计版本、旧字段拒绝 |
| `tests/unit/test_jobpacer_gpu_compute_profile.py` 与双卡 compute profile smoke | profile 签名、软件/UUID 校验、固定 repeats、初始化/warmup/复用门控及设备数据流 |
| `tests/unit/test_jobpacer_runtime_adapter.py` | TaskSpec 跨 rank 一致、FIFO 占位不能进入 LTF、profile 签名拒绝、LTF comm+tail 公式及静态/动态一致性 |
| `tests/unit/test_jobpacer_runtime_worker.py` | 用可控 Event/Barrier 与假 runtime 证明 producer 完成→submit 返回→independent 入队；grant 阻塞不阻止 independent；wait_on 与 terminal 顺序；失败停发与关闭一次 |
| `tests/unit/test_jobpacer_dag.py` | bridge 图、host 顺序、共同 seed/data map、null join、group 顺序与联合环；旧 DAG runner 回归 |
| `tests/integration/test_runtime_replay.py` | H 与旧 S 共用 worker 修改后的 Gloo/旧功能回归，预期 task 全集与 group 实际 launch 投影 |
| `tests/integration/test_runtime_replay_nccl.py`、`nccl_semantics_worker.py` | 双卡真实 producer→NCCL 数据、consumer 双依赖、null 阶段、terminal、延迟消费与容量分离、warmup=0/非零、重复 epoch、故障/断连有界退出 |

先运行纯输入与顺序检查，再运行针对性 runtime 单测、全仓检查及 opt-in 双 rank Gloo/NCCL。最终代码变更之后重跑相关检查，不能引用改动前的通过记录。GPU 不可用时将设备项标为未验证；socket 沙箱拒绝按权限流程处理，不改逻辑绕过。NCCL 验证采集实际软件版本、两张 UUID、命令、耗时、通过/失败/跳过与产物路径；不安装或升级 PyTorch/CUDA/NCCL，不覆盖分配的可见设备。

M2 退出条件是共同 task/spec 匹配、逐 rank 实际 launch 为 grant 投影前缀、数值与 stream 依赖正确、SUBMITTED/COMPLETED 顺序及容量正确、所有故障有界收尾。M3 退出条件是合同冻结与配对原始点可审计；不要求净性能收益或观察到 kernel overlap。M4 退出条件是工作量、数据流、host 提交次序和终点全部对齐。M5 退出条件是 profile 签名、公式输入、候选选择、deadline 与估计误差都能解释；性能结论单独记录。

## 7. 本轮实施与验收状态

`gpu_workload.py`、`gpu_compute.py`、`gpu_compute_profile.py`、两个校准入口、GPU rank worker、linear runner、DAG bridge、profile 加载与 policy hints、`run_phase3.py` 双入口及对应单测/输入已落地。新入口要求两 rank NCCL，继承 `CUDA_VISIBLE_DEVICES` 并映射逻辑 `cuda:rank`；父子结果包含 manifest/profile/source 摘要、UUID、launch/grant 投影和逐张量校验。实现过程中真实设备测试发现 scalar `torch.sum(out=...)` 需要显式 reduction dimensions，已修复正式 join 与 warmup。

此前一轮 GPU smoke 的 `L0-smoke.json` 包含 fill/matmul producer、独立 matmul、sum join、null independent/dependent 和三个共同 group 序号；该版本的 dynamic FIFO、Static LTF、Lookahead 与 FIFO 两卡运行及六种 fault injection 均通过。首轮 backend launch 故障使 peer NCCL Work 卡在未匹配 collective，随后添加失败路径对本地 NCCL groups 与 WORLD 执行 abort。六种路径均在有限时间内收到双 rank 错误或 deadline 错误退出，没有把 grant 伪装成取消。本次审查修复后没有重跑 fault-injection batch。

审查修复将 linear 数值校验延后到本地所有 job terminal 且 `finish_epoch()` 协议 drain 成功之后，验证期间独立记录 rank/task validation 区间；对应测试验证协议关闭先于数值校验。Static FIFO 改由 job/group 依赖边拓扑排序，Static/Dynamic GPU LTF 共用 `ltf_score()`。bridge 改为实际按 DAG 节点状态推进，并记录独立的 producer、通信物理完成、independent 和 join/terminal 完成事件；此前结果文档中的 bridge 单次 makespan 来自只做图映射、复用线性 runner 的旧实现，不作为真实 DAG bridge 证据，以修复后结果为准。

2026-09-28 补充修复：compute profile 升为 schema v2，L0/L1 输入都引用独立的 `gpu-compute-v2.json`，保留原 v1 profile 作为历史产物；`max_inflight` 必须是非布尔整数。v2 profile 重校准、L0 Static LTF/Lookahead smoke 与本轮回归的事实见结果文档补充记录。

M4 真实双卡重跑结果及当前验证摘要见[结果文档的审查修复复验](../result/phase3-gpu-workload-implementation-20260927.md#审查修复复验)。M3 正式配对对照、足量 profile、噪声 block、长期/重复 epoch、NCCL 断连及跨多 group/backend 的完整矩阵仍未执行。此前全仓测试中的三个 `test_benchmark_paths.py` 失败来自此工作区缺少 Git 忽略的历史 migration map 与迁移后 raw/manifest；没有伪造历史产物。此前全仓其余测试通过；Phase 1/Phase 2 target integration 与 Gloo/NCCL 的验证时间点见结果文档。
