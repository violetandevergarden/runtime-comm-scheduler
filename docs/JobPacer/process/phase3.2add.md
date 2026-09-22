# Phase 3.2 增补：固定 DAG 下的动态 tail 实施方案

日期：2026-09-22。状态：待实施；本文不代表功能或实验已验收。

依据：[设计讨论](../plan/discussion.md)、[Phase 3.1](../plan/phase3.1.md)、
[Phase 3.2](../plan/phase3.2.md)、[结构整理验收](../result/phase3.12fix.md)。
按本次用户要求在 process 中记录实施方案；完成后的事实另写 result/phase3.2add.md。

## 1. 目标与边界

保持 DAG 拓扑、执行依赖及 LTF 定义不变，根据已经发生的计算观测修正后续计算耗时估计，
重新计算 tail，并让已经声明或排队的通信能够向 coordinator 更新评分。

本次必须完成：

- 固定估计与在线估计两种模式，默认保留固定估计。
- rank/job 本地估计状态、显式同类节点关联、计算耗时采样、tail 重算。
- pending 通信的估计更新协议及其 grant/close/failure 竞态处理。
- 可追溯日志、确定性排序变化检查和双 rank CPU/Gloo 回归。

本次不做：动态 DAG、通信身份或 group_seq 修改、中央 DAG、资源竞争预测、分支进度感知的
job 完成时间预测、跨 job 共享估计、多通信在途、后台学习线程、GPU 测量或框架适配。
首版通信耗时保持原估计，只在线学习计算耗时；通用评分更新接口可以携带通信估计，但不据此宣称
通信在线学习已完成。跨迭代观测可复用本地估计对象，但不新增多迭代训练 harness 或持久化服务。

## 2. 当前代码依据

- `dag/model.py::compute_tails()` 按 job 计算最长后继路径，复用本 job 数据依赖和 group 顺序边。
- `dag/runner.py` 保存初始化 tail，在 DECLARE/OFFER 时构造 hint，尚无在线更新。
- `examples/jobpacer/runtime_worker.py` 每 rank 计算并共享初始 tail map；在线模式不可原地修改这份共享 map。
- `runtime/runtime.py::submit()` 会替换已有本地任务记录，扩展时必须保留新增的估计版本状态。
- `runtime/coordinator.py::_accept_task()` 接受 DECLARE/OFFER 的 hint，但不允许重复声明或提交。
- coordinator 当前按成员 hint 的最大 tail 构造候选；LTF 比较 tail，FIFO 使用首次 eligible 序号。
- 计算 callable 由 adapter 提供，runner 通过 Future 观察完成；样本配置、seed 和 sleep 留在 examples。

实施前重新检查实际调用链和工作区。编写本文时 `dag/model.py`、`runtime/transport.py` 已有用户修改；
本次只填充本文件，不覆盖上述改动，也不把历史测试记录作为本次验证。

## 3. tail 语义与估计更新

### 3.1 不改变策略定义

```text
tail(v) = max(estimated_duration(u) + tail(u), u 为 v 的直接后继)
无后继时 tail(v) = 0
```

不包含当前节点自身耗时，不减去 admission 等待时间，不把已完成节点简单置零来伪装剩余路径模型。
同 job 的 group 顺序边保持原语义，不引入其他 job 的工作量。
当前 tail 仍是最长依赖路径启发式，未完整模拟计算 worker 串行化、分支汇合等待和跨 rank 同步。

### 3.2 已完成观测如何影响未来

仅修改已经完成节点的权重，通常不能改变未来通信的后继 tail。因此显式提供
`estimate_keys: Mapping[qualified_compute_node_id, str]`，由 adapter 指定可共享绝对耗时估计的节点。
首版 key 只在同 rank、同 job 内共享；未配置的节点各自独立。节点同属 compute 不代表可共享。

每个 key 使用指数平滑：

```text
第一次观测：new = (1 - alpha) * observed_node_initial_estimate + alpha * sample
后续观测：  new = (1 - alpha) * previous_key_estimate + alpha * sample
```

初始时每个节点保留自己的原估计；第一次观测后，该 key 对应的未来节点采用 new。
因此 key 必须表示可共享绝对耗时的工作，而不是只有名称相似、规模不同的算子。
alpha 要求有限且 0 < alpha <= 1，建议初始默认 0.2，记录到配置；确定性检查可用 alpha=1。

样本只来自成功完成的执行；失败、取消、非有限或负值不能进入估计。
更新只应用到尚未开始的计算节点，正在执行节点保持开始时的估计；不改变执行状态或依赖计数。
没有同类节点映射的一次性 DAG 允许没有可用的未来修正，不凭空推断全 job 慢速系数。
相同节点跨迭代学习要求调用方显式复用估计对象，不能默认当前 replay 已具备多迭代能力。

### 3.3 观测口径

runner 在计算 worker 内包装现有 `compute_fn(node, stop_event)`：调用前后读取本地单调时钟，
成功时通过 Future 返回 elapsed；runner 线程消费样本并更新估计。保持外部 compute_fn 返回 None 的接口。
不使用 pool.submit 到 runner 发现完成的时间，因为它混入 worker 排队和轮询延迟。
CPU host elapsed 仍可能包含 OS 调度等待，日志必须如实命名，不解释为纯设备计算时间。

replay 的执行配置、预采样时长不得作为估计器输入；它们只负责实际模拟和离线校验。
不从 `submit()` 到 `wait_host()` 的跨度学习通信耗时，其中包含策略排队。
未来通信学习需独立定义实际 launch 到完成探测的本地时间，并区分探测误差；GPU 必须另行定义设备测量。

## 4. 文件与关键函数设计

只新增一个本地估计模块，不建设插件接口或学习服务。以下为计划 API，实施时可按现有风格微调。

| 文件 | 修改内容 |
| --- | --- |
| `src/runtime_comm_scheduler/dag/model.py` | `compute_tails(graph, *, duration_estimates=None)` 支持覆盖权重；无覆盖时结果不变 |
| `src/runtime_comm_scheduler/dag/estimates.py`（新增） | 具体的 `DurationEstimator`，维护 key 映射、平滑值及观测记录，不读取 replay 配置 |
| `src/runtime_comm_scheduler/dag/runner.py` | worker 内测量、runner 消费观测、重算本 job tail、发布变化的评分 |
| `src/runtime_comm_scheduler/dag/__init__.py` | 仅导出确需调用方使用的估计对象，不增加历史模块兼容层 |
| `src/runtime_comm_scheduler/runtime/runtime.py` | `update_estimate()`，本地任务估计 revision、发送顺序及生命周期处理 |
| `src/runtime_comm_scheduler/runtime/protocol.py` | 新事件 `UPDATE_ESTIMATE` 及版本化消息约定 |
| `src/runtime_comm_scheduler/runtime/coordinator.py` | 成员估计版本、更新校验、忽略过期/已 grant 更新、决策日志 |
| `src/runtime_comm_scheduler/runtime/model.py` | 必要的评分字段有限值校验，不将估计加入 TaskSpec 身份 |
| `examples/jobpacer/runtime_adapter.py` | 独立估计配置解析与校验，计算样本路径继续与估计器隔离 |
| `examples/jobpacer/runtime_worker.py` | 按 job 构造估计器，保留共享初始 tail；配置和结果传递；适配 fault wrapper |
| `examples/jobpacer/run_runtime_replay.py` | CLI 参数、完整转发、启动前预检 |
| `examples/jobpacer/runtime_results.py` | 观测/更新统计及版本追踪，不改变现有正确性检查 |

估计器最小接口：

```python
class DurationEstimator:
    def observe(self, node_id: str, elapsed_s: float) -> bool: ...
    def duration_estimates(self) -> dict[str, float]: ...
```

构造时接收本 job 初始计算估计、key 映射和 alpha；observe 返回估计是否变化。
runner 在应用快照时过滤已开始节点。所有可变估计状态由该 job runner 线程拥有；worker 不直接写估计器。
固定模式不构造学习状态、不发送更新；需要公平测量时可启用同样的采样但不学习。

`compute_tails` 覆盖表使用 qualified node ID，缺失项回退原值，未知 ID 和非法数值拒绝。
提取复用的单 job tail helper，由全图函数和在线单 job 重算共用，避免在线每次重算所有 job。
每次相关观测最多重算本 job 一遍，复杂度 O(V+E)，暂不做增量缓存算法。

## 5. 更新协议与竞态

### 5.1 接口与消息

```python
runtime.update_estimate(
    task_id,
    estimated_comm_s=...,
    remaining_tail_s=...,
) -> bool
```

revision 由 runtime 在现有条件锁内分配，不要求 adapter 同时管理传输版本。
返回 True 仅表示已接受并发送，不代表中心采纳；返回 False 表示本地已知 grant 或评分无变化。
未知任务、关闭后调用和非法数值明确报错；失败按现有 runtime 失败传播路径处理。

UPDATE_ESTIMATE payload 包含 task_id、estimate_revision、estimated_comm_s、remaining_tail_s。
不包含 ready_after_s，不修改 DECLARE 时间锚点。版本范围为 (epoch, endpoint, task_id)。
DECLARE/OFFER 同样携带 estimate_revision，首次评分版本为 0，随后每次发布递增。
DECLARE 后 UPDATE 再 OFFER 时，runner 必须传入最新评分；OFFER 发布新版本，runtime 保留版本计数，
不得因替换 `_LocalTask` 丢失版本。所有发送遵守同一个条件锁和 event_seq 顺序。

该字段扩展改变控制协议：升级 PROTOCOL_VERSION 并同步更新解析、测试和所有端点。
不承诺新旧进程混跑；历史结果 JSON 和 DAG 文件不因此重写。

### 5.2 中心规则

1. 校验 epoch、event_seq、任务存在、成员身份、该成员已 DECLARE/OFFER、revision 类型和值及评分数值。
2. 较旧 revision 记录并忽略；相同 revision 相同值视为无效重复，相同 revision 不同值报告协议冲突。
3. 对已 grant 的合法更新记录 `ignored_after_grant`，不撤销、不重排、不视为运行失败。
4. 对 pending 任务替换该成员评分，保留其 ready_after_s 与 declared_at；仍取成员最大值。
5. 保持 eligible_seq、group 顺序、容量和已发布 grant 不变；通过现有事件循环重新评估策略。

rank 不需要等其他 rank 的估计版本一致；成员可以使用不同观测。决策日志记录实际使用的各成员版本。
估计更新不是全员同步点，不新增 ACK 往返；中心采纳/忽略结果通过日志核验。
对本地未知的 grant 竞态，以中心决定为准。中心终态、过期 epoch 和非法身份仍遵循原协议，不借忽略竞态放宽校验。

### 5.3 关闭、失败与 lookahead

更新必须与 submit/finish_epoch 共用锁，保证已接受 UPDATE 排在 INPUT_CLOSED 前。
输入关闭后禁止继续更新；更新发送失败沿用 abort、唤醒 handle 和有界退出逻辑。
不以估计更新重置 epoch watchdog 或 active_wait deadline；持续更新也必须触发正常超时检查。
对 lookahead 的评分变化可以重评估，但不能借换目标反复开启新的等待预算。
StaticOrder 不重新生成顺序；FIFO 不改变首次 eligible 排名。

## 6. runner 推进顺序

```text
成功计算 Future 完成
  → 读取 worker 内 elapsed
  → observe 更新同类节点估计
  → 在尚未开始节点应用估计、重算本 job tail
  → 合并并发送已声明/已 OFFER 通信的变化评分
  → 正常完成节点、解锁后继；新 DECLARE/OFFER 使用最新 tail
```

每轮合并同一任务的更新，只发数值发生变化的评分，不加定时线程。
尚未注册任务仅保留本地新值；已经 grant 的任务即使 runner 尚未观察完成，也由 runtime/中心安全过滤。
不扫描或学习其他 job 的执行样本，不修改 worker 共享的初始 tail map。
确定性 unit 测试可以注入观测值；正式 replay 只能使用成功执行后的实测值。

## 7. 配置、日志与对照

建议 CLI：`--tail-estimation fixed|online`，默认 fixed；online 需要 `--tail-estimation-config PATH`。
配置独立于现有 DAG JSON，避免改变历史 canonical digest，例如：

```json
{
  "alpha": 0.2,
  "estimate_keys": {
    "job-a/compute-0": "same-size-block",
    "job-a/compute-2": "same-size-block"
  }
}
```

验证 key 仅引用存在的 compute 节点；相同字符串在不同 job 中不共享状态。
结果记录独立配置的 canonical digest、模式、alpha、初始 tail 和最终估计摘要；旧 DAG digest 不变。
拒绝线性 workload 使用 online，直到显式实现线性适配，不静默忽略参数。
Static/FIFO 可以用 online 检查不变量，但其原调度规则不变。

新增事件记录：样本节点/口径/elapsed、key、估计旧值/新值、task tail 旧值/新值、revision、中心采纳/忽略原因。
决策记录应能关联到使用的成员版本和聚合 tail；不能只有最终估计表。
报告更新次数、消息量、重算耗时、估计误差、job duration 及原有等待分解。
预测误差优先比较节点开始前的预测与该节点实测；不能将“job 完成减去通信完成”的跨度直接当作纯 tail 真值，
该跨度包含后续调度等待、资源竞争等。不同 rank 的原始时钟不直接相减。

固定/在线 LTF 对照保持 DAG、初始估计、实际执行样本、seed、backend、成员、容量与探测间隔一致。
固定的是样本而非绝对 ready 时间。先做机制和误差检查，不默认在线学习改善 makespan。
不覆盖历史批次或哈希 manifest；新测试输入单独放在 benchmark/phase3 下。

## 8. 验证设计

### 8.1 单元与协议

- `tests/unit/test_jobpacer_dag.py`：覆盖权重的 tail、缺省等价、group 顺序语义、同类映射和平滑；
  无映射不传播、未来样本不可见、正在执行节点不重估、失败/取消不学习、非法值拒绝。
- `tests/unit/runtime/test_runtime_core.py`：构造容量暂占用、两个候选 A/B，A 的 tail 从 2 更新为 8，B 为 5；
  释放容量后固定模式选 B、更新模式选 A；验证取成员最大值、成员版本独立和旧版本不覆盖。
- `tests/unit/runtime/test_rank_runtime.py`：DECLARE→UPDATE→OFFER 保留版本、更新与 grant 交错、
  更新与 finish 交错、传输失败、关闭后拒绝、非法身份、已 grant 更新不导致故障。
- 在现有协议测试中覆盖新版本/消息字段；在 lookahead 测试中覆盖持续 UPDATE 不能延期，FIFO 排名不变。
- adapter/worker/results 测试：配置预检、CLI 完整转发、独立配置 digest、固定模式回归、日志可关联。

并发用 Event/Barrier 控制交错，不用短 sleep 制造必现排序。
排序翻转的 coordinator 单测使用明确事件序列；不能要求普通 OS 调度下每次集成运行得到同一 grant 序列。

### 8.2 CPU/Gloo

新增小型固定 DAG：前段计算提供观测，后续同 key 计算位于待调度通信的 tail 中，另有独立计算分支，
使某通信排队时仍可能收到计算观测。核验真实更新采纳与结果正确性；若更新因 grant 太快未生效，
日志必须显示原因，不把“发送过更新”当成排队任务更新成功。
需要保证交错的机制用例采用受控测试入口，不向正常 CLI 注入绝对 ready 时间控制。

保留单一在途，检查共同 grant 的本地 launch 投影、成员覆盖、tensor 结果、失败有界退出。
旧线性和 DAG 集成场景继续回归。GPU/NCCL、跨 host、通信在线估计和性能普遍收益不在验收范围。

实施时从仓库根目录执行并记录实际结果：

```bash
PYTHONPATH=src pytest -q tests/unit/runtime tests/unit/test_jobpacer_dag.py tests/unit/test_jobpacer_runtime_adapter.py tests/unit/test_jobpacer_runtime_worker.py tests/unit/test_jobpacer_runtime_results.py
PYTHONPATH=src pytest -q
env PYTHONPATH=src RUN_JOBPACER_RUNTIME_REPLAY=1 pytest -q tests/integration/test_runtime_replay.py
git diff --check
```

socket 沙箱限制按权限流程处理；不改代码绕过。记录软件环境、passed/failed/skipped、时长与产物路径，
不预填测试数量，不自动安装依赖或启动大规模实验。

## 9. 实施顺序与完成标准

1. 保存当前工作区边界和基线结果；确认用户已有修改与本方案重叠处。
2. 扩展纯 tail 函数与具体估计器，先验证观测确实能影响未来路径。
3. 实现 UPDATE_ESTIMATE 协议及版本/关闭/grant/失败回归，再接入 runner。
4. runner 完成测量、学习、重算和发布闭环，接入独立配置及日志。
5. 完成确定性排序变化、双 rank Gloo 和历史路径回归。
6. 新写 result/phase3.2add.md，区分机制通过、估计误差、单次性能观测和未验收边界。

完成标准：固定模式保持原语义；online 使用已发生观测改变未来估计；中心能在 grant 前采纳排队任务的新 tail；
版本与生命周期竞态安全；动态更新不修改合法候选判定、group 顺序、容量、grant 不可撤销性和等待上限；
测试与日志能证实上述行为，而不只是增加了配置开关。
