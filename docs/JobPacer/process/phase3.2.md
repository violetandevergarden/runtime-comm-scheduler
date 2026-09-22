# JobPacer Phase 3.2 详细实现方案

日期：2026-09-20。状态：完成（CPU/Gloo）；NCCL/GPU 未验收。

> 2026-09-21 结构迁移说明：本文主体记录 Phase 3.2 首次实现时的设计与路径，原文路径不回写。
> 之后的模块迁移和重新验收见 [Phase 3.1/3.2 结构整理](phase3.12fix.md) 与
> [结构整理验收](../result/phase3.12fix.md)。当前 DAG 库位于 `src/runtime_comm_scheduler/dag/`；
> replay schema、采样和绑定位于 `examples/jobpacer/runtime_adapter.py`。

本文把 [Phase 3.2 计划](../plan/phase3.2.md) 落到当前代码结构上，给出数据格式、
校验算法、本地推进状态机、策略衔接、文件改动、测试矩阵和验收顺序。设计基础是当前
`src/runtime_comm_scheduler/runtime/` 的 Phase 3.1 实现及
[Phase 3.1 修复后结果](../result/phase3.1.md)，不是重新设计一套通信 runtime。

实施前基线：

- `RankRuntime` 已提供 `declare()`、非阻塞 `submit()`、`RuntimeHandle`、单 launch worker、
  completion probe、`finish_epoch()` 和 fail-stop；
- coordinator 已按所有成员 OFFER、`group_seq` 和 `max_inflight=1` 构造 eligible，并支持
  Static、FIFO、LTF 和 bounded Lookahead；
- `runtime_worker.py` 仍按每 job 的线性 `for communication` 循环推进，且默认一个 job
  对应一个 group；
- `runtime_adapter.remaining_tail()` 仍是线性累加，不能用于 DAG；
- `run_runtime_replay.py` 的校验仍主要从每个 job 的已返回 task 反推 group 投影，尚不能
  拒绝“预期任务缺失但现有结果全正确”的空集合情况；
- 2026-09-20 在当前工作树执行 `PYTHONPATH=src pytest -q tests/unit/runtime`，结果为
  `32 passed`。这只是实施前回归基线，不是 Phase 3.2 验收结果。

## 1. 实施原则与完成边界

Phase 3.2 的最小可行改动是：在 JobPacer 应用层增加一个手写 DAG 模型和本地 runner，
把 ready communication 映射成现有 `RankRuntime.submit()`；通信核心继续只负责跨 rank
成员匹配、共同 grant、group 顺序、launch 和完成反馈。

本阶段完成时必须具备：

1. 从 JSON 加载有限、执行前已知的多 job DAG；
2. 启动通信前拒绝非法图、非法 group 序号和联合约束环；
3. 每 job 一个串行 compute 通道，计算节点和通信节点均按真实完成事件推进依赖；
4. 同一 job 可以同时产生多个通信前沿，并全部提交给现有 coordinator 选择；
5. FIFO、LTF、StaticOrder 和安全 Lookahead 使用同一 DAG 执行路径；
6. 两 rank Gloo 下验证 diamond、计算/通信重叠、多 group、rank skew 和失败收尾；
7. 输出能够证明预期节点全集完成、各 group 成员看到一致的 launch 投影、tensor 正确；
8. 原有线性 Phase 3.1 replay 和 32 项 runtime 单测继续通过。

本阶段不实现：GPU/NCCL DAG 语义、多通信容量、多计算资源调度、跨 job 数据依赖、
在线增删节点、同一 group 内动态重排、框架自动捕获、跨 host、重连或最优 DAG 调度。
这些能力不需要预留抽象类、插件注册器或新协议版本。

## 2. 总体结构与最小改动面

```text
DAG JSON
  │
  ├─ parse / normalize / validate / digest
  │
  ├─ DagWorkload + 每 job DagRunner ── compute future
  │              │
  │              └─ ready comm ──> RankRuntime.submit()
  │                                      │
  │                                  Coordinator / Policy
  │                                      │
  └─ execution config / binding ──> LocalBinding / Gloo Work
```

按最少文件原则，首版只新增一个正式库实现文件和 benchmark 输入目录：

| 文件 | 改动 |
| --- | --- |
| `src/runtime_comm_scheduler/dag.py` | 新增具体数据类、JSON 加载、联合校验、tail、静态顺序、runtime 映射和 `DagRunner`；保持 `runtime/` 核心不反向依赖 DAG |
| `examples/jobpacer/runtime_adapter.py` | 保留历史线性 workload 的小型映射 |
| `examples/jobpacer/runtime_worker.py` | 增加 DAG 分支，复用 group、runtime、server、tensor 和进程生命周期 |
| `examples/jobpacer/run_runtime_replay.py` | 增加 `--dag`、DAG 参数传递、预期集合校验和 DAG 指标汇总 |
| `src/runtime_comm_scheduler/runtime/runtime.py` | 增加公开 `abort()`，内部仍复用 `_fail()` |
| `benchmark/phase3/*.json` | 三个确定性手写 benchmark 输入 |
| `tests/unit/test_jobpacer_dag.py` | 模型、校验、tail、静态顺序和 runner 因果测试 |
| `tests/unit/runtime/test_rank_runtime.py` | 公共 abort 的最小回归测试 |
| `tests/integration/test_runtime_replay.py` | 增加可选的双 rank Gloo DAG 用例 |
| `docs/JobPacer/result/phase3.2.md` | 实施后记录命令、产物、通过项、观测结果和限制 |

首版不把 DAG 类型塞进 `src/runtime_comm_scheduler/runtime/model.py`，而是作为
`runtime_comm_scheduler.dag` 的上层库模块；不修改控制协议，也不新增 `dag/` 包。只有当单个
`dag.py` 实际变得难以维护时再拆分。

## 3. 手写 DAG 输入格式

### 3.1 顶层结构

首版 schema version 固定为 1：

```json
{
  "schema_version": 1,
  "name": "diamond",
  "seed": 0,
  "groups": [
    {"group_id": "group-a", "ranks": [0, 1]},
    {"group_id": "group-b", "ranks": [0, 1]}
  ],
  "execution": {
    "compute_duration_s": {
      "job-0/producer": 0.004,
      "job-0/independent": 0.003,
      "job-0/join": 0.0
    }
  },
  "jobs": [
    {
      "job_id": "job-0",
      "nodes": [
        {
          "node_id": "producer",
          "kind": "compute",
          "deps": [],
          "estimated_duration_s": 0.004
        },
        {
          "node_id": "comm-a",
          "kind": "comm",
          "deps": ["producer"],
          "group_id": "group-a",
          "group_seq": 0,
          "estimated_comm_s": 0.001,
          "collective": {
            "op": "all_reduce",
            "numel": 1024,
            "num_bytes": 4096,
            "dtype": "float32",
            "shape": [1024],
            "reduction": "sum"
          }
        },
        {
          "node_id": "independent",
          "kind": "compute",
          "deps": ["producer"],
          "estimated_duration_s": 0.003
        },
        {
          "node_id": "join",
          "kind": "compute",
          "deps": ["comm-a", "independent"],
          "estimated_duration_s": 0.0
        },
        {
          "node_id": "comm-b",
          "kind": "comm",
          "deps": ["join"],
          "group_id": "group-b",
          "group_seq": 0,
          "estimated_comm_s": 0.001,
          "collective": {
            "op": "all_reduce",
            "numel": 1024,
            "num_bytes": 4096,
            "dtype": "float32",
            "shape": [1024],
            "reduction": "sum"
          }
        }
      ]
    }
  ]
}
```

字段语义：

| 字段 | 语义 |
| --- | --- |
| `schema_version` | 只接受整数 1；未知版本直接拒绝 |
| `name/seed` | 运行身份和可重复扰动输入；seed 不参与节点身份 |
| `groups` | epoch 无关的 group ID 与全局 rank 成员；epoch 由命令行注入 |
| `job_id/node_id` | 稳定身份；生成全局节点 ID `job_id/node_id` |
| `deps` | 只允许引用同 job 节点，且表达“前驱真实完成后才可启动” |
| `estimated_duration_s` | compute 的策略估计；用于 DAG tail 和 Lookahead |
| `execution.compute_duration_s` | qualified compute node ID 到 CPU sleep 基值的映射；策略不得读取 |
| `estimated_comm_s` | communication 的策略估计；真实执行仍由 collective 决定 |
| `group_seq` | 整个 epoch 内该 group 的规范通信序号，不按 job 重新编号 |
| `collective` | 直接构造现有 `CollectiveSpec` 的字段 |

`join` 继续用零时长 compute 表示，不增加 barrier/join 节点类型。首版要求每个 job
至少有一个 comm；该 job 引用的所有 group 必须具有完全相同的 rank 集合，该集合就是
job 的参与成员。这样每个参与 rank 都拥有相同 job 图，不引入 rank-local 缺失节点。

`execution` 是 replay binding 配置，不属于 DAG 节点描述。每个 compute 必须恰好有一个
执行值，不能缺失或包含非 compute 的额外 key。后续若增加 jitter，只允许从 execution
基值和稳定 seed 生成本轮样本，不能把实际样本回填给策略。通信节点不配置“实际通信
时长”，以免用 sleep 替代真实 collective。

### 3.2 具体 Python 模型

`dag.py` 使用冻结 dataclass，不建设继承层次：

```python
@dataclass(frozen=True)
class ComputeNode:
    node_id: str
    deps: tuple[str, ...]
    estimated_duration_s: float

@dataclass(frozen=True)
class CommNode:
    node_id: str
    deps: tuple[str, ...]
    group_id: str
    group_seq: int
    estimated_comm_s: float
    collective: CollectiveSpec

Node = ComputeNode | CommNode

@dataclass(frozen=True)
class DagJob:
    job_id: str
    nodes: tuple[Node, ...]

@dataclass(frozen=True)
class DagWorkload:
    schema_version: int
    name: str
    seed: int
    groups: tuple[GroupSpec, ...]   # 加载时注入 epoch
    jobs: tuple[DagJob, ...]

@dataclass(frozen=True)
class ReplayExecutionConfig:
    compute_duration_s: Mapping[str, float]

@dataclass(frozen=True)
class DagInput:
    workload: DagWorkload
    execution: ReplayExecutionConfig
```

图描述、replay 执行配置和运行态三者分开；都不保存 tensor、ProcessGroup、callable、
future 或 handle。运行态由 `DagRunner` 单独维护。解析仅接受文档中列出的字段；首版
未知字段报错，避免拼写错误被静默忽略。所有耗时必须是有限、非负的实数，bool 不按
整数接受。

节点的 communication task ID 固定为 `f"{job_id}/{node_id}"`。`job_id` 和 `node_id`
不得包含 `/`，从而保证 ID 可逆且无需转义规则。manifest 中不再单独允许覆盖 task ID。

### 3.3 规范化与摘要

加载后按以下方式产生 canonical document：

1. group 按 `group_id` 排序，ranks 使用 `GroupSpec` 的升序结果；
2. job 按输入顺序保留，node 也按输入顺序保留，作为稳定 FIFO 破同分键；
3. deps 去重校验后按字符串排序；
4. 使用 `json.dumps(..., sort_keys=True, separators=(",", ":"), allow_nan=False)`；
5. 对完整规范化输入（图描述和 execution 配置）计算 SHA-256，输出 `manifest_digest`。

所有 rank 返回 digest；父进程要求 digest 完全一致。摘要用于证明输入相同，不替代
coordinator 对 GroupSpec、TaskSpec 和成员 OFFER 的逐项检查。

## 4. 执行前校验

校验必须在创建线程、连接 coordinator 和发起 collective 之前完成。错误包含 workload、
job、node/group 和具体字段，不能等 epoch timeout 才发现。

### 4.1 局部结构校验

逐层检查：

- workload name 非空，job/group ID 唯一且非空；
- node ID 在 job 内唯一，kind 只允许 compute/comm；
- deps 不重复、不包含自身、全部存在于同 job；
- job 至少一个 comm，引用 group 均存在，且这些 group 的 ranks 相同；
- group ranks 非空、唯一、非负，并在运行时小于 `world_size`；
- comm task ID 在 epoch 内唯一；
- `CollectiveSpec` 继续复用现有 shape、numel、bytes、dtype 和 op 校验；
- compute 和 comm 的估计、execution mapping 中的 sleep 基值均为有限非负值；
- execution mapping 恰好覆盖所有 compute qualified ID，且不引用 comm 或未知节点；
- 同一 `(group_id, group_seq)` 只能对应一个 comm。

### 4.2 原始 DAG 环检测

为每个 job 构造 `node -> deps` 映射，调用标准库 `graphlib.TopologicalSorter.prepare()`。
捕获 `CycleError` 后把涉及节点转换为 qualified node ID 报出。这里不自行实现 DFS。

### 4.3 group 连续性和联合约束环

对每个 group：

1. 收集所有 job 的 comm；
2. `group_seq` 必须恰好为 `0..N-1`，不允许缺口；
3. 按 seq 给相邻通信增加规范顺序边 `seq(n) -> seq(n+1)`；
4. 把这些边和全部 job 数据依赖边放入一个全局 qualified-node 图；
5. 再运行一次 `TopologicalSorter.prepare()`。

第二次检查会在启动前拒绝“DAG 要求 B 在 A 前完成，但 group 顺序要求 A 在 B 前发起”
一类联合死锁。跨 job 共享 group 的顺序边只用于合法性和准入，不变成跨 job 数据依赖。

### 4.4 静态通信顺序校验

显式静态顺序由 `--static-order <json>` 提供，文件内容是 task ID 数组。校验：

- 与 manifest 中全部 comm task ID 集合完全相等；
- 长度相等且无重复；
- 任意 comm 的所有通信祖先在数组中更早出现。通信祖先由联合约束图的一次拓扑动态规划
  得到，不对每对节点重复做图搜索。

若 DAG 模式运行 `static_fifo/static_ltf` 且没有显式文件，则本地确定性生成顺序：对联合
图做 Kahn 推进，先立即消化所有 ready compute，再从 ready comm 中选择一项。FIFO 使用
manifest 的 `(job_index, node_index, task_id)`，LTF 使用
`(-remaining_tail_s, job_index, node_index, task_id)`。每选择一个 comm 才把它加入静态
计划。生成结果仍经过上述完整性和祖先顺序校验。

这只是合法、可复现的静态对照，不模拟计算完成时刻，也不宣称是最优拓扑排序。

## 5. DAG tail 与策略输入

### 5.1 节点 duration

策略只读取估计：

```text
duration(compute) = estimated_duration_s
duration(comm)    = estimated_comm_s
```

对每个 job，将该 job 内同 group 的规范顺序边加入原 DAG；跨 job group 边不纳入某个 job
的 tail，以免一个 job 的优先级包含另一个 job 的计算。按逆拓扑序计算：

```text
tail(v) = max(duration(u) + tail(u) for u in successors(v))
tail(v) = 0                                           if v is a sink
```

comm 节点传给 `TaskHint.remaining_tail_s` 的值是 `tail(comm)`：不包含当前 comm 自身耗时，
并行分支取最大值而不是相加。该定义替代 `runtime_adapter.remaining_tail()` 的线性累加；
两种指标不混用。

单元测试至少固定一个 diamond：一条后继分支 3 ms、另一条 7 ms，前驱 comm 的 tail
必须取 7 ms 路径，不能得到 10 ms。

### 5.2 FIFO 与 LTF

runner 会把所有 DAG-ready comm 提交给 runtime，不在本地先选一个。现有 coordinator：

- DynamicFIFO 继续按首次全部成员 OFFER 且 group 当前 seq 合法时的 `eligible_seq`；
- DynamicLTF 继续读取 `TaskHint.remaining_tail_s`；
- `group_seq` 尚未轮到的 comm 即使已经 OFFER，也不会成为 eligible；
- 全局 `max_inflight=1` 保持不变。

因此 `policy.py` 和 `coordinator.py` 首版不改。新增 DAG 单测若暴露通用 runtime 缺陷，
应在共享实现修复并给 Phase 3.1 增加回归测试，不能在 runner 内模拟 coordinator 选择。

### 5.3 Lookahead 安全声明

每当 compute 启动或节点完成后，runner 检查尚未声明的 comm。只有以下条件全部成立才
调用 `runtime.declare()`：

1. comm 尚未 ready、未声明、未提交；
2. 所有已完成依赖已经是 `COMPLETED`；
3. 每个尚未完成的直接依赖都是已经 `RUNNING` 的 compute；
4. 该任务仍受 coordinator 的 group next-seq 规则约束；runner 不预测它会越序执行。

安全性只按目标通信的直接依赖判断；job 内无关的 PENDING/READY compute 或 RUNNING comm
不取消此预测。它们不是该通信的依赖，且通信准入仍完全由 coordinator 控制。

预测值在首次声明时固定：

```text
remaining(pred) = max(0, estimated_duration_s - elapsed_since_compute_start)
ready_after_s   = max(remaining(pred) for each running predecessor)
```

首版每 job 只有一个 compute worker，通常最多一个运行前驱；公式仍按集合写，避免把安全
条件绑死在这个事实。节点真正 ready 后调用 `submit()`，其 OFFER hint 使用
`ready_after_s=0`，但 estimated comm 和 tail 不变。已声明节点不重新声明，也不更新旧
预测；日志保存声明依据、预计 ready 和实际 ready，供结果分析误差。

如果任一未完成前驱是 communication，则不提前声明。这条规则防止 Lookahead 等待一个
必须先服务当前通信才能到达的后继。

## 6. 本地 `DagRunner`

### 6.1 运行状态

描述对象不可变；每个 runner 维护：

```python
class NodeState(Enum):
    PENDING = "pending"
    READY = "ready"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"

remaining_deps: dict[str, int]
successors: dict[str, tuple[str, ...]]
state: dict[str, NodeState]
ready_compute: set[str]
ready_comm: set[str]
compute_future: Future | None
compute_node_id: str | None
comm_handles: dict[str, RuntimeHandle]
declared: set[str]
```

状态只由该 job 的推进线程修改，因此不需要为每个字典加锁。compute future 和 runtime
handle 只作为完成信号读取。节点只能执行一次；`_complete(node_id)` 先断言当前为 RUNNING，
再逐个减少 successor 的 `remaining_deps`，首次降为零时进入 READY。

job 完成条件是所有节点均 COMPLETED，不要求单一 sink。任何 FAILED 都终止该 job，且不
再启动或提交后继。

### 6.2 推进循环

实现顺序固定为：先收完成、再提交通信、再启动一个计算、最后声明安全 Lookahead。

```text
initialize zero-dependency nodes as READY
deadline = monotonic + job_timeout

while completed_count != node_count:
    1. 检查 runtime.failure
    2. 收割已完成 compute future
    3. 对 RUNNING comm 调用 handle.wait_host(0)，收割完成或抛出失败
    4. 按稳定 node key 提交全部 ready comm，不等待 grant
    5. compute 通道空闲时，启动一个最小稳定 key 的 ready compute
    6. 为满足安全前沿规则的 comm 做一次 declare
    7. 若本轮无状态变化，以 dag_poll_interval 有界等待
    8. 到达同一个绝对 deadline 时报告诊断并失败
```

通信提交步骤对每个节点：先创建 tensor/binding，再记录 `comm_submit_call`，调用
`runtime.submit()`，保存 handle，状态改为 RUNNING，记录 `comm_submit_return`。submit
返回、GRANT、底层 Work 绑定均不解锁后继；只有 `wait_host(0)` 观察到 runtime handle
COMPLETED 才调用 `_complete()`。

compute 使用 `ThreadPoolExecutor(max_workers=1)`，且仅在通道空闲时提交一个 callable，
不提前把多个节点塞进 executor 队列。样例 callable 使用可中断的 `stop_event.wait(duration)`；
正常返回才表示完成。ready compute 用 manifest 稳定 key 选择，避免 set 迭代顺序影响结果。

`dag_poll_interval` 与 runtime 的 `completion_poll_interval_s` 分开配置和记录，默认都为
1 ms。前者只影响 runner 观察 handle/future 的延迟，后者影响 runtime 观察 backend Work
的延迟；结果文档不得把任一观察时间称为精确设备完成时间。

### 6.3 独立推进语义

如果 `producer` 同时解锁 `comm-a` 和 `independent-compute`，runner 在同一轮先 submit
`comm-a`，再启动 compute；之后不等待 comm grant/complete。两者可以重叠。若另一个
ready comm 属于不同 group，也在同一轮提交。这里的“全部提交”是让中央策略看见完整
当前前沿，不表示多个 collective 会同时 launch；`max_inflight=1` 仍由 coordinator 保证。

每 job 一个 compute worker 意味着同 job 的两个 ready compute 会串行；不同 job runner
各自拥有一个 worker，可能在 CPU 上并行。结果必须写明这是 replay 资源模型，不能外推为
共享 GPU 上的真实计算并行。

## 7. DAG 库映射、group 和真实 binding

### 7.1 DAG 到 runtime 模型的映射

在 `src/runtime_comm_scheduler/dag.py` 中提供三个直接函数：

```python
dag_task_spec(job, comm, *, epoch) -> TaskSpec
dag_task_hint(job, comm, tails, *, ready_after_s) -> TaskHint
dag_group_specs(workload, *, epoch) -> tuple[GroupSpec, ...]
```

`dag_task_spec` 不复用线性 ordinal：task ID 来自 job/node，group_seq 来自 manifest，
CollectiveSpec 直接来自节点。`examples/jobpacer/runtime_adapter.py` 只保留原有
`group_spec/task_spec/task_hint/all_specs`，确保线性 replay 无需迁移，也避免正式库依赖 examples。

LocalBinding 仍在 worker 创建，因为它需要本 rank 的 tensor、真实 ProcessGroup 和
`dist.all_reduce` closure。首版每个 comm 创建独立 float32 tensor，keepalive 至少包含
tensor；预期 all-reduce 值按该 comm 的真实 group ranks 计算，不能按 job ID 或 world
size 猜测。

### 7.2 ProcessGroup 创建与注册

所有 rank 都按规范化 manifest 的 group 顺序调用 `dist.new_group(ranks)`，防止创建顺序
不一致。只有属于该 group 的 rank 才调用：

```python
runtime.register_group(GroupSpec(epoch, group_id, ranks), process_group)
```

只为 rank 所参与的 job 启动 `DagRunner`。由于首版要求一个 job 引用的 group 成员完全
一致，不会出现某 rank 只执行该 job 的部分节点。销毁时按创建逆序 destroy group，最后
销毁 default group，沿用现有 worker 的 `finally`。

## 8. Runtime 的唯一必要 API 改动

当前 `runtime_worker.py` 在 job 异常时直接调用私有 `runtime._fail()`。Phase 3.2 增加：

```python
def abort(self, error: BaseException, *, stage: str = "application", **details: Any) -> None:
    self._fail(error, stage=stage, **details)
```

要求：

- 复用现有首次错误获胜、失败所有 handle、停止 launch、发送 FAILED 的实现；
- 多个 job 同时 abort 幂等，不覆盖首个根因；
- CREATED、RUNNING、INPUT_CLOSED 和已 FAILED 状态行为有测试；
- 线性 worker 同步改用 `runtime.abort(...)`，不保留新的私有调用者。

不为 DAG 增加 completion callback、每通信 waiter 线程、批量 submit 或 coordinator 图
注册协议。runner 对公开 handle 做有界轮询已经满足首版规模和语义。

## 9. Worker 与命令行接入

### 9.1 CLI

`run_runtime_replay.py` 增加并完整转发：

```text
--dag PATH
--static-order PATH
--dag-poll-interval FLOAT     default 0.001
--compute-jitter FLOAT        default 0.0
--poll-interval FLOAT         保留并真正转发给 RankRuntime
--epoch INT                   保留并真正转发
```

`--dag` 与 `--workload` 互斥；两者均未提供时继续使用线性 `balanced`，保证旧命令兼容。
`--static-order` 只允许 DAG 的 static policy。`compute-jitter` 必须在 `[0, 1)`，用稳定
hash `(seed, epoch, job_id, node_id, rank)` 产生 execution duration 样本；0 表示严格使用
manifest 值。

当前父进程 `_start()` 没有转发 `epoch` 和 `poll_interval`，本阶段一并修正，因为 DAG
输出必须准确记录运行配置。不要为参数转发新建配置框架，继续显式构造 argv。

### 9.2 Worker 分支

`run_rank()` 的公共部分保留：初始化 dist、创建 coordinator/server/client/runtime、
barrier、finish、close 和 group 销毁。加载输入后分为：

```text
linear workload -> 现有 run_job()
DAG workload    -> 每个本地 job 创建 DagRunner.run()
```

每个本地 job 仍使用一个顶层线程，使多个 job 可以独立产生前沿。用一个
`threading.Barrier(local_job_count)` 在记录 `job_start` 后统一释放，避免线程创建先后被
误算为策略差异；barrier 只同步当前 rank，跨 rank 到达差异仍由 runtime 正常处理。

所有 join 共用一个绝对 replay deadline。不能对 N 个线程各 `join(timeout)` 导致总等待
扩大为 N 倍。任一 runner 异常时：保存首个错误、设置共享 stop、调用
`runtime.abort(error, stage="dag_runner", job_id=..., node_id=...)`；其他 runner 在下一轮
观察 stop/runtime failure 后退出。

全部 runner 成功后，worker 只调用一次 `runtime.finish_epoch()`。runner 自身不得关闭
共享输入。若 compute callable 卡死，Python 线程不能安全取消；父进程 `_collect()` 到达
整体 timeout 后 kill rank，并把运行标记为失败。

## 10. Telemetry 和结果格式

### 10.1 本地 DAG 事件

每 rank 增加一个共享 `EventLog(source="dag", endpoint=rank)`，记录：

| 事件 | 必要字段 |
| --- | --- |
| `job_started/job_completed/job_failed` | job_id、状态、耗时或错误 |
| `node_ready` | job_id、node_id、kind、完成前驱数 |
| `compute_started/compute_completed` | 估计时长、实际 sample |
| `comm_declared` | task_id、ready_after、tail、预测依据节点、`prediction_base_us`、`predicted_ready_at_us` |
| `comm_submit_call/comm_submit_return` | task_id、group_id、group_seq |
| `comm_completed_observed` | task_id、handle state |
| `node_failed` | job_id、node_id、异常类型和文本 |
| `job_timeout` | 所有未完成节点、remaining deps、运行 future/handle 状态 |

runtime 自己的 declared/offered/grant/launch/completion 事件继续保留，不复制进 DAG 日志。
本地时间只在同 rank 内相减；跨 rank 阶段仍使用 coordinator records。

### 10.2 rank 输出

每个 rank 输出至少增加：

```json
{
  "input_mode": "dag",
  "dag_name": "diamond",
  "manifest_digest": "...",
  "groups": [{"group_id": "group-a", "ranks": [0, 1]}],
  "expected_task_ids": ["job-0/comm-a", "job-0/comm-b"],
  "jobs": [{
    "job_id": "job-0",
    "status": "ok",
    "node_count": 5,
    "completed_node_ids": ["..."],
    "tasks": [{"node_id": "comm-a", "task_id": "job-0/comm-a", "correct": true}]
  }],
  "dag_events": [],
  "runtime_events": []
}
```

`completed_node_ids` 和 task 列表按 manifest 稳定顺序输出，而不是按线程/字典完成顺序。
真实完成顺序从事件时间读取。

### 10.3 父进程校验

`_validate_results()` 接收规范化 DAG 的预期集合，逐项验证：

1. rank 数等于 world size，所有 rank status 为 ok，digest 相同；
2. 每个 rank 的 expected task/node 集合与其 group/job membership 一致；
3. 实际 task 集合与预期集合完全相等，数量相等，无重复和非预期 task；
4. 每个预期 comm 都有 `correct=true`，不能用空列表上的 `all()` 判成功；
5. 每 rank `grant_sequence == launch_sequence` 的成员投影；
6. 对每个 group，仅比较该 group 成员 rank 的 launch 投影，且都等于按 `group_seq`
   排序的最终完整序列；
7. 全部预期 DAG node 都恰好完成一次；失败节点或未排空 handle 使验证失败；
8. coordinator 的 dispatch task 集合与全局预期 comm 集合相同。

线性模式也复用“显式预期集合、拒绝空成功”的通用校验修正，但不改变历史字段含义。

### 10.4 指标口径

新增或明确：

- job duration：本 rank runner 从统一释放到所有节点完成；跨 rank job JCT 取成员最大值；
- coordinator epoch duration：首个 coordinator 控制事件（通常为 `group_registered`）到 `FINISHED`；该值包含控制连接/group 注册，不等于 replay makespan。没有中央 replay-start 时间戳时，不把它解释为统一启动到完成的 replay 时长；
- node ready wait：ready 到 compute start 或 comm submit call；
- runtime admission wait：offer 到 grant，继续使用各自 rank 本地事件；
- predicted-ready error：实际 DAG-ready 减 `predicted_ready_at_us`，该时间在调用 `declare()` 前按同一 monotonic 时钟固定，仅同 rank 计算；不使用 `comm_declared.time_us` 反推，避免混入控制发送耗时；
- DAG critical-path estimate：静态输入估计，不称为实际 critical path；
- completion timestamp：明确称“完成探测时间”，不称物理 kernel 精确结束时间。

不把不同进程 monotonic clock 直接相减，不把 compute 排队、group 顺序等待、成员未到齐、
容量等待和 probe 延迟未经证明相加成完整 makespan 分解。

## 11. 失败和资源生命周期

| 失败点 | 行为 |
| --- | --- |
| JSON/图/group/static order 非法 | 启动 dist/runtime 前失败，父进程收到明确解析错误 |
| compute callable 抛错 | 节点 FAILED，停止该 job，公共 `runtime.abort()` 传播整个 epoch |
| binding/tensor 创建失败 | 对应 comm FAILED，尚未 OFFER，abort 并有界退出 |
| submit/transport 失败 | runtime 保存首错并唤醒所有 handle；runner 观察同一错误 |
| launch/probe/backend 失败 | 沿用 RankRuntime fail-stop，不执行新节点 |
| job/replay timeout | 输出未完成节点和 handle 状态，abort；父进程最终 kill 卡死进程 |
| 已声明节点未提交 | 不发送成功 INPUT_CLOSED；作为运行失败收尾，不引入撤销协议 |

tensor、binding closure 和 predecessor 输出至少存活到对应 comm 完成及所有消费者完成。
`RankRuntime` 在完成后可以释放 binding，但 runner 若还需要 tensor 校验或下游读取，必须
在自己的 node runtime record 中保留引用。校验完成后再释放，不把全 tensor 校验放进
coordinator 或 completion worker。

正常结束顺序：所有 runner 完成 -> 校验本地节点/任务全集 -> `finish_epoch()` -> 收到
FINISHED -> 采集日志 -> `runtime.close()` -> destroy subgroups -> destroy default group。

## 12. 确定性样例

在 `benchmark/phase3/` 提供三份小输入：

### 12.1 `linear.json`

两个 job，每个为 `compute -> comm -> compute -> comm`，各自一个 group。用途：证明 DAG
路径可表达 Phase 3.1 的完成依赖链，并用于 FIFO/Static 兼容烟测。注意它不是旧线性
`submit -> independent compute -> wait` overlap 语义的机械等价转换。

### 12.2 `diamond.json`

```text
producer ─┬─ comm-a ───────┐
          └─ compute-b ────┤
                           join -> comm-c
```

用途：验证分叉、汇合、通信与独立计算重叠、join 等待全部前驱、tail 取最大分支。

### 12.3 `multi-group.json`

两个 job、三个 group；job-0 在 producer 后同时 ready group-a/group-b 两个 comm，并有不同的
非零后继 tail（21 ms / 3 ms）；job-1 的 group-c 前沿有 51 ms tail，预测 ready 与实际
execution 可用 jitter 分别验证按时到达和 deadline fallback。用途：验收多前沿、FIFO/LTF
选择、Lookahead 等待及 group 投影。

如果需要覆盖跨 job 共享 group 的联合环，使用单元测试内最小 dict 构造非法输入，不增加
第四份仅用于失败的样例文件。

## 13. 测试计划

测试优先使用 event/fake handle，时间断言只检查因果关系，不用 1 ms sleep 竞争判断正确性。

### 13.1 模型和校验单测

| 用例 | 断言 |
| --- | --- |
| schema round-trip/digest | 规范化稳定，不同 key 顺序摘要相同 |
| duplicate/missing/self dep | 启动前明确拒绝 |
| local cycle | 报出 job 和 cycle 节点 |
| duplicate task/group seq | 启动前拒绝冲突 |
| group seq gap | 缺少 0 或中间序号时拒绝 |
| inconsistent job memberships | 一个 job 引用不同 ranks 的 group 时拒绝 |
| combined cycle | 原 DAG 无环但加入 group 边成环时拒绝 |
| invalid duration | 负数、NaN、Inf、bool 拒绝 |
| static order | 缺失、重复、非预期、违反祖先顺序均拒绝 |
| tail | diamond 取最长分支；非对称链验证 successor duration；不读取 execution duration |
| multi-group strategy | 用 DAG 生成的 tail/hint，在容量占用期间积累两个前沿，断言 FIFO 与 LTF 选择不同 |

### 13.2 Runner 单测

建立只实现 `declare/submit/failure` 的 FakeRuntime 和可手动完成的 FakeHandle：

| 用例 | 断言 |
| --- | --- |
| chain | 前驱 handle 完成前后继不 submit |
| fork | 两个 ready comm 都 submit，runner 不等待第一个 grant |
| diamond | join 在两个分支都完成后且仅执行一次 |
| compute serialization | 同 job 两个 ready compute 不同时运行 |
| independent progress | comm pending 时独立 compute 可以完成 |
| safe lookahead | 仅直接未完成前驱为运行 compute 才声明；无关 pending compute / running comm 不阻挡 |
| prediction | ready_after 随已运行时间减少，只 declare 一次；预测绝对时刻锚定在 declare 调用前 |
| duplicate completion defense | 节点不会二次解锁或出现负 remaining deps |
| compute/submit failure | abort 一次、后继不运行、诊断包含 node |
| deadline | 单一绝对预算，不因状态变化重置 |

### 13.3 Runtime 回归

在 `test_rank_runtime.py` 增加公共 `abort()` 测试：所有非终态 handle 失败并唤醒、队列中的
后续任务不 launch、重复 abort 保留首错。现有 32 项 runtime 单测全部保留。

### 13.4 双 rank Gloo 集成

保留环境变量 `RUN_JOBPACER_RUNTIME_REPLAY=1` 的 opt-in 方式，增加：

| policy / DAG | 重点 |
| --- | --- |
| FIFO / linear | DAG 基本路径、任务全集、tensor 正确 |
| FIFO / diamond | 分叉汇合和 compute/comm overlap |
| LTF / multi-group | 多前沿候选、tail 选择和 group 投影 |
| Static FIFO / multi-group | 静态顺序合法且忠实等待队首 |
| Static LTF / multi-group | 自动生成顺序与外部 `--static-order` 均合法并执行 |
| Lookahead / multi-group | Gloo 验证目标声明、非零预测及预测误差锚点；CoordinatorState 因果测试验证 `ACTIVE_LOOKAHEAD` 与目标到达 |
| Lookahead / deadline | CoordinatorState 到期测试验证 deadline 记录和 `LOOKAHEAD_DEADLINE_FALLBACK` |
| FIFO / rank skew | 两 rank OFFER 到达不同但共同 grant 正确 |

另做一个 failure 参数化用例覆盖 compute failure、非法 manifest、missing submit 和 binding
failure；所有 subprocess 必须在测试 timeout 内退出，不能只断言返回码非零。

## 14. 分阶段实施顺序

### M0：冻结 schema 和校验

实施：

1. 在 `dag.py` 加 dataclass、严格 JSON parser、canonical document 和 digest；
2. 实现局部拓扑、group 连续性、联合约束图和静态 order 校验；
3. 添加 `linear/diamond/multi-group` 三份样例；
4. 完成模型和非法输入单测。

通过条件：任何会导致依赖或 group 顺序死锁的手写输入在启动 runtime 前失败；合法样例
在任意 rank 产生相同 digest、task IDs 和 group specs。

### M1：tail、静态顺序和本地 runner

实施：

1. 计算 DAG tail 和稳定节点 key；
2. 生成/校验 static FIFO、static LTF order；
3. 实现 NodeState、remaining deps、单 compute future、全部 ready comm submit；
4. 实现 safe Lookahead declare、绝对 deadline 和 DAG EventLog；
5. 用 FakeRuntime/FakeHandle 完成因果测试。

通过条件：不启动 torch/distributed 即可稳定证明 chain、fork、diamond、独立推进、一次执行、
失败停止和预测边界。

### M2：接入真实 worker

实施：

1. 增加 DAG adapter；
2. worker 按 manifest 创建/register 多 group，并启动本地 runners；
3. 添加 `RankRuntime.abort()` 并迁移线性 worker 的私有调用；
4. 父进程完整转发 epoch/poll/DAG 参数；
5. 输出 expected sets、DAG events 和 digest。

通过条件：两 rank Gloo 的 FIFO/diamond 与 FIFO/multi-group 均有界成功，所有 collective
结果正确，预期节点/任务全集相等，各 group 成员 launch 投影一致。

### M3：多策略与 Lookahead

实施：

1. DAG static order 分支接入现有 StaticPolicy；
2. FIFO/LTF 使用同一 runner，只替换 policy；
3. Lookahead 使用安全前沿声明，记录预测误差与 fixed deadline；
4. 加入 rank skew、static head blocked 和 lookahead fallback 集成用例。

通过条件：日志中的候选、tail、声明依据和 decision 可解释；不存在为未服务通信后继主动
等待的场景；所有策略使用相同 DAG、compute samples、group 和 completion probe。

### M4：失败、回归和结果文档

实施：

1. 故障注入与缺失集合校验；
2. 运行完整 unit、原 Phase 3.1 integration 和新增 DAG integration；
3. 对三个样例运行 FIFO、Static FIFO、Static LTF、LTF、Lookahead 烟测；
4. 将实际命令、JSON 产物、通过项和局限写入 `result/phase3.2.md`。

通过条件：失败均有界、原线性 replay 不回归、DAG 机制矩阵全部通过。性能胜负不是本阶段
完成门槛，不用一次运行的 makespan 宣称某策略稳定优于其他策略。

## 15. 实施后的验证命令

```bash
PYTHONPATH=src pytest -q tests/unit/runtime tests/unit/test_jobpacer_dag.py

RUN_JOBPACER_RUNTIME_REPLAY=1 PYTHONPATH=src \
  pytest -q tests/integration/test_runtime_replay.py

PYTHONPATH=src python examples/jobpacer/run_runtime_replay.py \
  --policy fifo \
  --dag benchmark/phase3/diamond.json \
  --backend gloo --world-size 2 --timeout 20 \
  --output /tmp/jobpacer-phase3.2-diamond-fifo.json

PYTHONPATH=src python examples/jobpacer/run_runtime_replay.py \
  --policy ltf \
  --dag benchmark/phase3/multi-group.json \
  --backend gloo --world-size 2 --timeout 20 \
  --output /tmp/jobpacer-phase3.2-multigroup-ltf.json

PYTHONPATH=src python examples/jobpacer/run_runtime_replay.py \
  --policy lookahead \
  --dag benchmark/phase3/multi-group.json \
  --backend gloo --world-size 2 --timeout 20 \
  --output /tmp/jobpacer-phase3.2-multigroup-lookahead.json

PYTHONPATH=src python examples/jobpacer/run_runtime_replay.py \
  --policy static_fifo \
  --dag benchmark/phase3/multi-group.json \
  --backend gloo --world-size 2 --timeout 20 \
  --output /tmp/jobpacer-phase3.2-multigroup-static.json

PYTHONPATH=src python examples/jobpacer/run_runtime_replay.py \
  --policy static_ltf \
  --dag benchmark/phase3/multi-group.json \
  --static-order docs/JobPacer/result/phase3.2-validation/static-ltf-order.json \
  --backend gloo --world-size 2 --timeout 20 \
  --output docs/JobPacer/result/phase3.2-validation/static-ltf-external-multi-group.json
```

还需重新运行原线性四例，证明默认 CLI 和 Phase 3.1 入口未改变：

```bash
PYTHONPATH=src python examples/jobpacer/run_runtime_replay.py --policy fifo --workload balanced --backend gloo --timeout 20
PYTHONPATH=src python examples/jobpacer/run_runtime_replay.py --policy static_fifo --workload delayed --backend gloo --timeout 20
PYTHONPATH=src python examples/jobpacer/run_runtime_replay.py --policy ltf --workload tail --backend gloo --timeout 20
PYTHONPATH=src python examples/jobpacer/run_runtime_replay.py --policy lookahead --workload delayed --backend gloo --timeout 20
```

## 16. 最终验收清单

- [x] DAG schema、实际 replay 时长和策略估计边界已写入样例及结果文档；
- [x] 重复/缺失依赖、局部环、group 序号缺口和联合环均在启动前拒绝；
- [x] 所有 rank 的规范化 digest、TaskSpec 和 GroupSpec 一致；
- [x] comm 后继只由 runtime handle 的 COMPLETED 解锁，不由 submit/grant/bind 解锁；
- [x] fork 的多个 ready comm 全部 OFFER，未退化为本地逐项 wait；
- [x] 每 job compute 串行、多个 job 可独立推进，资源模型已明确记录；
- [x] DAG tail 对分支取 max，且不读取 execution sample；
- [x] 非对称后继时长验证 tail 递推；多前沿的 DAG tail 令 FIFO 与 LTF 选择可区分；
- [x] Lookahead 只按直接依赖声明，记录预测绝对时间，并验收 `ACTIVE_LOOKAHEAD`、目标到达和 deadline fallback；
- [x] Static LTF 自动计划和外部 `--static-order` 均有双 rank 集成验收；
- [x] StaticOrder 覆盖全部 comm、无重复并满足联合祖先顺序；
- [x] 预期节点/任务全集与实际集合完全相等，空结果不能通过；
- [x] 每个 group 的所有成员 launch 投影一致并按 group_seq 完整递增；
- [x] tensor 结果按真实 group ranks 校验，而不是按 job 或 world size 推断；
- [x] 任一 runner/runtime 失败会唤醒其他等待者并在整体 deadline 内退出；
- [x] `finish_epoch()` 每 rank 仅由共享 owner 调用一次，且发生在全部本地 job 完成之后；
- [x] 原 Phase 3.1 runtime 单测和四个线性 Gloo replay 不回归；
- [x] `docs/JobPacer/result/phase3.2.md` 及固定 validation batch 记录实际命令、持久 JSON、源码摘要和 CPU/Gloo 限制。

Phase 3.2 的完成标志是“手写 DAG 能在现有 runtime 上正确、可观测、可失败地推进真实
Gloo 通信，并让中央策略看到多个合法前沿”。它不以某个策略在三份小样例上的 makespan
胜出为完成条件，也不把 CPU sleep/probe 结果当作 GPU 框架适配已经成立。
