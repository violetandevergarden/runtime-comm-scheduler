# Phase 3.1 / 3.2 结构整理实施计划

日期：2026-09-21。状态：已完成；实际改动和验收见 [结果记录](../result/phase3.12fix.md)。

实施前依据为 `runtime/`、包根目录 `dag.py`、JobPacer replay 及 [Phase 3.2 实施说明](phase3.2.md)。遵循 [设计讨论](../plan/discussion.md)、[Phase 3.1 计划](../plan/phase3.1.md) 和 [Phase 3.2 计划](../plan/phase3.2.md) 的职责边界。历史验收见 [Phase 3.1 结果](../result/phase3.1.md)、[Phase 3.2 结果](../result/phase3.2.md)，不能代替重构后的回归。

## 1. 目标、非目标与不变量

本次不是重写 runtime，而是整理已经存在的能力：

1. 将 DAG 图模型和依赖推进从 CPU replay 配置中分离。
2. 收敛联合图构建、job 线程生命周期和结果分析中的职责重叠。
3. 明确正式库、新 replay 与历史静态实现的入口及依赖方向。
4. 保持命令行、已有输入格式、通信协议和实验语义稳定。

不增加通用插件、执行后端注册器、Job Agent、资源图、在线改图、多通信在途或 GPU 支持。不能为减少文件数重新使用旧 Plan/TaskKey 作为新 runtime 的内部模型。

必须保持：group 规范顺序和跨成员匹配、唯一有序 launch、SUBMITTED 先于 COMPLETED、全成员物理完成释放容量、FIFO 首次 eligible 顺序、Lookahead 固定 deadline、输入关闭顺序、失败唤醒和有界退出。

线性 replay 与 DAG replay 暂时保留各自的单 job 执行逻辑。线性 submit 后独立计算、消费等待和旧 tail 不能在结构重构中悄悄替换成不同的 DAG 语义。

## 2. 实施前证据与处理决定

| 当前位置 | 现状 | 本次处理 |
| --- | --- | --- |
| `src/runtime_comm_scheduler/dag.py` | 图解析、联合校验、tail、静态序列、sleep 采样、runner 混在一个模块 | 改为 `dag/` 包；图与 runner 留库内，replay 配置和采样移到 examples |
| `DagInput` / `DagRunner` | 通用推进必须携带 seed、execution 和实际耗时 | runner 改为接收计算 callable，不读取 replay 配置 |
| `parse_dag/build_static_order/_joint_predecessors` | 重复构造数据依赖与 group 顺序联合图 | 统一构图函数；保留局部/全局校验区别 |
| `runtime_worker._run_linear/_run_dag` | 两套 barrier、线程、deadline、stop、异常汇聚 | 共用一个具体的本地 job 运行函数，单 job 逻辑保持独立 |
| `run_runtime_replay.py` | 启动、验收、指标汇总、CLI 混合 | 验收与指标移到 `runtime_results.py` |
| 新 replay 的线性静态分支 | 通过旧 plan builder 生成 Plan，再转 task ID | 暂留为显式历史桥接，集中到 adapter，禁止依赖扩散 |
| 包根历史模块与 runtime 同名模块 | 名称相似，完成与准入语义不同 | 不合并、不删除；文档明确主线与历史路径 |

`dag.py` 位于 `runtime/` 之外本身不是依赖错误；真正需要解决的是模块职责。新 `dag/` 与 `runtime/` 平级，仍属于通信 runtime 的上层调用者。

## 3. 预期目录结构

下列树列出本次涉及的正式模块、实验入口、历史文件及测试；未列出的仓库文件保持原位。

```text
src/runtime_comm_scheduler/
├── __init__.py                   # 保留旧导出，说明主线导入入口
├── runtime/
│   ├── __init__.py
│   ├── model.py
│   ├── protocol.py
│   ├── coordinator.py
│   ├── policy.py
│   ├── runtime.py
│   ├── handle.py
│   ├── executor.py
│   ├── transport.py
│   └── telemetry.py
├── dag/                          # 替代同名 dag.py，不同时保留两者
│   ├── __init__.py
│   ├── model.py
│   └── runner.py
├── adapters/
│   ├── __init__.py                # 现有历史适配
│   └── megatron.py
├── executor.py                   # 以下为历史路径，暂不迁移
├── intent.py
├── plan.py
├── scheduler.py
├── telemetry.py
├── validate.py
└── work.py

examples/jobpacer/
├── __init__.py
├── README.md
├── runtime_adapter.py            # 扩充：实验输入、模拟与绑定
├── runtime_worker.py             # rank 生命周期、共享 job harness
├── runtime_results.py            # 新增：结果校验与指标
├── run_runtime_replay.py          # CLI、子进程启动与汇总
├── workloads.py                  # 以下历史输入/入口继续保留
├── plan_builder.py
├── replay_worker.py
├── run_replay.py
├── comm_profile.py
└── profile_communication.py

benchmark/phase3/
├── README.md
├── linear.json
├── diamond.json
└── multi-group.json

tests/unit/
├── test_jobpacer_dag.py           # 保留图、runner 的现有回归，可内部按职责分组
├── test_jobpacer_runtime_adapter.py  # 新增输入、模拟配置、绑定与导入检查
├── test_jobpacer_runtime_worker.py   # 新增共享 job harness 的并发/失败检查
├── test_jobpacer_runtime_results.py  # 新增结果校验与指标纯函数检查
└── runtime/                     # 现有核心测试保持
tests/integration/
└── test_runtime_replay.py         # 保留线性、DAG、故障与静态序列回归

docs/JobPacer/
├── process/phase3.12fix.md        # 本实施计划
└── result/phase3.12fix.md         # 实施完成后新建的验收事实
```

依赖方向为 `examples → dag → runtime`，examples 也可直接调用 runtime；runtime 不导入 dag，src 不导入 examples。历史静态桥接仅保留在 examples adapter，不进入 dag/runtime。

## 4. 正式库逐文件设计

### 4.1 `dag/__init__.py`

仅导出图模型、图算法和 runner，不导出 CLI、JSON 文件读取、sleep 模拟或实验 digest 函数。保留 `runtime_comm_scheduler.dag` 这个模块入口，但其中属于 replay 的名称迁移到 examples；仓库内调用方同步修改，不做从 src 反向转发到 examples 的兼容层。

拟导出 `ComputeNode`、`CommNode`、`DagJob`、`DagGraph`、`validate_graph`、`build_static_order`、`validate_static_order`、`DagRunner`、`NodeState`。其他函数优先保持模块内使用，避免无意扩大公共 API。

### 4.2 `dag/model.py`

内容：冻结节点/图数据类、验证、拓扑关系、tail、静态顺序与通信规范映射。它可以依赖 runtime 的模型，但不依赖 torch、文件系统或实验 sleep 配置。

关键对象与函数草案：

```python
@dataclass(frozen=True)
class DagGraph:
    groups: tuple[GroupSpec, ...]
    jobs: tuple[DagJob, ...]

def validate_graph(graph: DagGraph, *, world_size: int | None = None) -> None: ...
def joint_predecessors(graph: DagGraph) -> dict[str, set[str]]: ...
def compute_tails(graph: DagGraph) -> dict[str, float]: ...
def build_static_order(graph: DagGraph, policy: str) -> tuple[str, ...]: ...
def validate_static_order(order: tuple[str, ...], graph: DagGraph) -> tuple[str, ...]: ...
def dag_task_spec(job: DagJob, comm: CommNode, *, epoch: int) -> TaskSpec: ...
def dag_task_hint(comm: CommNode, *, tail_s: float,
                  ready_after_s: float | None = 0.0) -> TaskHint: ...
```

设计约束：

- `validate_graph()` 也服务 Python 直接构图，不将全部合法性检查藏在 JSON parser 中。
- 保留原节点和 job 的输入顺序及当前破同分规则。不要在移动代码时将其改成另一种 ID 排序。
- `joint_predecessors()` 是全图数据依赖和 group 规范边的唯一构造点。解析后的全局环检查、静态计划生成和静态序列校验共用它。
- group 连续性检查必须保留，不能因共用构图函数只剩下环检查。
- tail 的作用域仍是 job 内数据依赖加同 job group 顺序。可共享“加入 group 相邻顺序边”的小函数，但不能简单截取全局图而漏掉跨 job 间隔两侧的同 job 顺序关系，也不能把其他 job 的计算计入 tail。
- 原 `DagInput` 中的 `tails/node_order/group_ranks` 属于派生索引，不与独立可修改的模型重复存储。初期按需计算，在一次初始化中复用；没有测量证据不引入缓存失效体系。
- 有限非负估计、成员相容、ID 唯一、依赖完整性等仍严格校验。
- 不增加通用 graph interface、策略注册器或自定义拓扑框架，继续使用标准库。

### 4.3 `dag/runner.py`

内容：`NodeState`、单 job 推进、compute future、通信 handle、Lookahead 前沿、完成解锁、错误及超时诊断。

拟接口：

```python
class DagRunner:
    def __init__(self, graph: DagGraph, job: DagJob, runtime: RankRuntime, *,
                 epoch: int, rank: int,
                 compute_fn: Callable[[ComputeNode, threading.Event], None],
                 make_binding: Callable[[CommNode], LocalBinding],
                 deadline: float, poll_interval: float = 0.001,
                 enable_lookahead: bool = False,
                 stop_event: threading.Event | None = None,
                 event_log: EventLog | None = None): ...

    def run(self) -> dict[str, Any]: ...
    def _check_stop(self) -> None: ...
    def _collect_compute(self) -> bool: ...
    def _collect_comms(self) -> bool: ...
    def _declare_safe_frontier(self) -> bool: ...
    def _complete(self, node_id: str) -> None: ...
```

调整重点：

- 去掉 `compute_jitter`、seed、execution mapping 和默认 sleep；必须提供 compute callable。真实模拟耗时不传入策略或 runner。
- 保留一个串行 compute worker 和非阻塞通信推进，保持现有调度顺序及轮询方式。
- `compute_fn` 正常返回表示 CPU 节点完成，抛错表示失败。不得把这个接口当成已支持异步 GPU kernel 完成。
- compute 样本与采样日志由 adapter 记录，runner 只记录实际启动/完成及估计依据；样本在结果层合并。
- deadline 使用同一绝对单调时钟期限；收敛当前循环内和 `_check_stop()` 两套超时详情构造，避免重复分支漂移。
- 建立 `node_by_id` 索引，替代 `_declare_safe_frontier()` 和 `_complete()` 中重复的线性查找。
- 清除仅 `try: future.result(); except: raise` 一类无附加行为包装。
- 保留现有完成 handle 和 binding 的可读取结果，便于迁移期间 tensor 校验；明确其至少保留至结果校验完成，不在这次重构顺带改变释放时点。
- runner 出错仍调用公共 `runtime.abort()`；外层因其他 job 失败再次 abort 必须是幂等的。可以减少重复日志，但不能移除独立使用 runner 时的错误传播。

### 4.4 `runtime/` 各文件

本次不重排核心文件，仅按必要性做小清理：

| 文件 | 保留内容与关键函数 | 本次动作 |
| --- | --- | --- |
| `__init__.py` | 正式 runtime 模型、策略、handle、RankRuntime 导出 | 核对导出，无 DAG/replay 导出 |
| `model.py` | GroupSpec、TaskSpec、TaskHint、LocalBinding | 保留对象边界，不塞入计算节点 |
| `protocol.py` | encode/decode、envelope 校验 | 无协议改动 |
| `coordinator.py` | apply/tick、eligible、grant、完成与结束 | 保持单事件循环状态机，不与 DAG 静态排序合并 |
| `policy.py` | FIFO/LTF/Lookahead 的 decide | 保持在线策略，不调用离线 plan builder |
| `runtime.py` | submit/declare/abort/finish_epoch、launch/completion 循环 | 复用公共 abort；不加入 job 线程管理 |
| `handle.py` | wait_host、状态与失败通知、wait_on | 不借重构修改完成契约，GPU 仍未验收 |
| `executor.py` | DirectExecutor、WorkIsCompletedProbe | 核对 GlooExecutor 无行为子类的导出/调用；无需要时删除并更新导出，否则保留并说明，不建立新 executor 层 |
| `transport.py` | CoordinatorServer、ControlClient、有序收发 | 不与 replay 的进程启动工具合并 |
| `telemetry.py` | EventLog、monotonic_us | 只保留事件记录，不搬入实验指标分析 |

### 4.5 包根与历史模块

`__init__.py` 保留现有旧接口导出并更新说明，推荐新代码显式导入 `.runtime` 或 `.dag`。不同时大规模改成惰性导入或重命名旧 API。

根目录 `executor.py/intent.py/plan.py/scheduler.py/telemetry.py/validate.py/work.py` 及 `adapters/megatron.py` 暂时原位保留。它们与新 runtime 的名字重复不等于语义重复；不跨代合并完成探测、状态机或任务对象。`legacy/` 迁移另立任务，不能顺手破坏旧实验入口。

## 5. 实验层逐文件设计

### 5.1 `runtime_adapter.py`

职责：将手写 replay 输入和旧线性 workload 转换为正式库对象，提供实际计算模拟、通信绑定和确定性执行样本。保留现有 `group_spec/task_spec/task_hint/remaining_tail` 的线性语义。

新增/迁移内容：

```python
@dataclass(frozen=True)
class ReplayExecutionConfig:
    compute_duration_s: Mapping[str, float]

@dataclass(frozen=True)
class DagInput:
    graph: DagGraph
    name: str
    seed: int
    execution: ReplayExecutionConfig
    manifest_digest: str
    canonical_json: str

def load_dag(path, *, epoch=0, world_size=None) -> DagInput: ...
def parse_dag(raw, *, epoch=0, world_size=None) -> DagInput: ...
def load_static_order(path, graph: DagGraph) -> tuple[str, ...]: ...
def sample_compute_duration(seed, epoch, job_id, node_id, rank, base_s, jitter) -> float: ...
def make_replay_compute(job, execution, *, seed, epoch, rank, jitter,
                        samples, event_log): ...
def make_collective_binding(spec: TaskSpec, process_group, *, rank, device) -> LocalBinding: ...
def linear_static_order(workload, policy) -> tuple[str, ...]: ...
```

具体要求：

- JSON schema version、允许字段、execution 全覆盖校验和 canonical SHA-256 规则保持。拆分内部模型不改变历史输入摘要。
- parser 处理格式/字段；构造图后调用 `validate_graph()` 处理图语义，两层不复制相同图算法。
- `make_replay_compute()` 返回具体闭包，采样仍使用现有稳定 key 和公式，并通过 stop event 等待；不创建新的 ComputeBackend 类。
- `make_collective_binding()` 集中正常 tensor 构造和 all-reduce 调用。按实际 CollectiveSpec shape/dtype/reduction 构造；现有不能兑现的参数明确拒绝或作为独立修复验收，不能静默忽略。
- torch 只在绑定所需位置导入，确保模型解析、输入摘要检查和通用 DAG 库不因示例绑定引入强制 torch 依赖。
- 故障注入保留在 worker 的包装中，不进入正式库或正常绑定函数。
- `linear_static_order()` 暂时封装旧 plan builder 的转换，标明只为旧线性基线服务。不让 worker、dag 或 runtime 直接依赖旧 TaskKey。
- `all_specs()` 等函数先搜索调用；仅确认无使用且非需要保留的接口后移除，不凭名称判死代码。

### 5.2 `runtime_worker.py`

职责：一个 rank 的环境、ProcessGroup、coordinator/client/runtime 生命周期，调用单 job 逻辑，故障注入和结果组装。

关键函数设计：

```python
def _run_jobs(jobs, run_one, *, runtime, deadline, stop_event,
              thread_name_prefix) -> list[dict[str, Any]]: ...
def _run_linear_job(job, *, runtime, groups, deadline, stop_event, ...) -> dict: ...
def _run_dag_job(job, *, dag_input, runtime, groups, deadline, stop_event, ...) -> dict: ...
def _new_groups(group_specs) -> dict[str, Any]: ...
def run_rank(args) -> dict[str, Any]: ...
```

这里 `...` 表示实施时按当前绑定需要保留的具体参数，不是新增配置容器。不要为消除参数列表引入跨层的全能 context。

`_run_jobs()` 统一负责：

1. 空 job 集合直接返回；否则创建本地 barrier。
2. 每个 job 一个线程，在同一 deadline 内通过 barrier 后调用 `run_one(job)`。
3. 按输入顺序保存结果，记录首个异常，设置 stop 并打断 barrier，调用公共 abort 唤醒通信等待。
4. 所有线程的 join 使用同一 deadline 的剩余时间；不能每线程重新获得完整 timeout。
5. 出错不执行 finish_epoch，不把未完成线程记为成功；不可取消的 callable 仍由父进程超时兜底。

单 job 函数不再启动线程或维护独立错误列表。DAG job 负责组装 runner、注入模拟 compute、检查 tensor 和提取结果；线性 job 保留原声明、producer、submit、独立计算、wait 的次序。

`run_rank()` 仍拥有唯一 finish_epoch 调用和资源关闭。初始化失败也需按已成功创建的资源收尾，避免只给执行阶段加 finally。涉及关闭行为的调整必须补故障回归，不以代码缩短作为安全证明。

`_FailingProbe/_DropOneSubmit/_DroppedHandle` 暂留 worker，标注故障注入用途；不新增 fault plugin 系统。

### 5.3 `runtime_results.py`

职责：离线解释返回结果；不启动进程，不创建 runtime，不导入 torch。

```python
def expected_dag_results(graph: DagGraph, world_size: int) -> dict: ...
def validate_results(results, world_size, *, expected, expected_nodes=None,
                     digests=None) -> dict: ...
def metrics(results) -> dict: ...
```

从当前 `_validate_results()` 和 `_metrics()` 迁移实现，先保持参数与输出字段，确有需要再逐步简化。

保留预期全集、重复/缺失任务、参与成员、group_seq、实际 launch 与 grant 投影、中央 dispatch 覆盖、tensor correct、digest 等校验。不能将“进程返回成功”或空集合 `all()` 当成功。

指标继续区分本地与中央时钟、完成观察与真实完成；不重新解释已有时间字段。抽取函数不是把指标塞进 runtime telemetry 的理由。

### 5.4 `run_runtime_replay.py`

仅保留 CLI、输入预检、子进程启动/回收、调用结果函数和写输出：

- `main()`：保留现有命令和参数，调用 adapter/results。
- `_start()`：完整传递 epoch、poll interval、DAG 参数、静态序列和 fault。
- `_collect()`：保持子进程超时、终止及错误输出收集。
- `_free_port()`：当前本地启动工具，暂不增加通用进程管理库。

直接脚本执行与包导入两种方式继续支持。参数校验的父进程和 worker 边界保留；必要时共享小校验函数，但不能因父进程已检查而取消 worker 的独立防御。

### 5.5 其他实验文件

| 文件 | 整理事项 |
| --- | --- |
| `examples/jobpacer/__init__.py` | 不重导出整个正式库，不引入启动副作用 |
| `examples/jobpacer/README.md` | 明确新线性/DAG 命令、旧静态命令、profiling 的用途及不同完成语义 |
| `workloads.py` | 保留旧线性输入及 overlap 解释，不强制改为 DAG |
| `plan_builder.py` | 保留历史静态计划，不移入 runtime；新路径只经 adapter 的历史桥接调用 |
| `replay_worker.py/run_replay.py` | 保留历史基线，不在本轮合并其生命周期实现 |
| `comm_profile.py/profile_communication.py` | 保留 profiling 能力和旧调用关系，不因文件名相似删除 |
| `benchmark/phase3/README.md` | 更新库/入口路径说明和命令，输入语义不改 |
| `benchmark/phase3/*.json` | 三个现有输入保持原样及摘要，新增重构测试输入使用测试内小样例 |

## 6. 迁移映射与兼容策略

| 原实现 | 目标 |
| --- | --- |
| `dag.py` 的 ComputeNode/CommNode/DagJob | `dag/model.py` |
| `dag.py` 图校验、tail、静态排序、TaskSpec/Hint 映射 | `dag/model.py` |
| `dag.py` 的 NodeState/DagRunner | `dag/runner.py` |
| `dag.py` 的 DagInput/ReplayExecutionConfig/load/parse/canonical/sample | `examples/jobpacer/runtime_adapter.py` |
| worker 的正常 tensor/collective 绑定 | `runtime_adapter.make_collective_binding` |
| 两套 job 线程收尾 | `runtime_worker._run_jobs` |
| replay 的预期集合、校验与指标 | `runtime_results.py` |

迁移 `dag.py → dag/` 时使用可审查的补丁并核对所有导入；同一路径下不同时保留 module 和 package，也不留下完整重复实现。删除旧文件仅发生在内容迁移完成并验证后，属于搬迁而非丢弃历史能力。

保留 `runtime_comm_scheduler.dag` 的通用名称；replay 专属名称的导入变化同步改 tests、worker、launcher 和文档。若发现仓库外调用需求，再单独决定兼容范围；禁止为兼容让 src 导入 examples。

命令行、JSON schema、样本生成算法、canonical digest、结果字段及静态排序破同分默认保持。必要行为修复单独列出，不能和文件搬迁混成不可解释的实验差异。

## 7. 其他整理事项

1. 更新过时的“仅 Stage 3.1”“single-job skeleton”模块说明，明确库核心、应用上层与实验入口。
2. 清理已确认未使用变量，例如静态排序中的 `jobs_by_id`；无调用函数或导出删除前先检查整个仓库及文档。
3. 区分同算法复制与不同边界验证：父进程、worker、coordinator 的必要验证不因去重被删掉。
4. 不合并旧/new EventLog 或 WorkIsCompletedProbe，仅因同名不足以证明契约相同。
5. 保持 Python >=3.10；不用本机 3.13 专有语法，兼容性检查与本机测试分开报告。
6. 不重新生成或覆盖历史 validation batch 的 manifest、源码 SHA 和结果；重构验收建立新产物。
7. `process/phase3.2.md` 保留历史实现上下文，可追加迁移说明链接；`result/phase3.2.md` 不改写历史通过数字来冒充当前验收。
8. 仓库中未找到 `process/phase3.1&2fix.md`，本次不创建、不删除或合并该文件；以用户指定的本文路径为实施计划入口。
9. 实施前逐文件检查工作区状态和重叠；本次开始时 worktree 干净。保护已有修改，不 reset、不提交 commit、不删除历史 benchmark。

## 8. 实施顺序

| 步骤 | 工作 | 检查点 |
| --- | --- | --- |
| S0 基线 | 记录工作树、导入、命令、固定输入摘要和当前测试结果 | 区分已有失败与重构回归，不把历史文档计数当新基线 |
| S1 图与模拟分离 | 创建 dag 包，迁移 replay 配置，更新调用与导出 | 图校验/tail/静态序列、输入摘要、样本不变；src 无 examples 依赖 |
| S2 runner 精简 | 注入计算 callable，清理重复超时和节点查找 | 独立分支、完成解锁、安全 Lookahead、计算失败回归 |
| S3 harness 去重 | 抽取共享 `_run_jobs`，移动绑定与历史桥接 | barrier/首错/timeout/abort/关闭异常检查，线性语义不变 |
| S4 结果与文档 | 抽取 results，更新入口与历史说明，清理小冗余 | 结果字段一致、缺失/重复/成员错误仍拒绝 |
| S5 最终验收 | 全仓回归、真实 Gloo 和新结果文档 | 测试与持久产物可追溯，未验收边界明确 |

每步形成独立可审查 diff；不要求提交 commit。机制回归未通过前不继续扩大整理范围。

## 9. 测试文件与关键检查

| 测试文件 | 关键覆盖 |
| --- | --- |
| `test_jobpacer_dag.py` | Python 直接构图验证、联合环、group 序号、非对称 tail、静态顺序；runner 无 execution 配置也能使用自定义 compute；完成解锁和 Lookahead |
| `test_jobpacer_runtime_adapter.py` | JSON round-trip/摘要、未知字段、execution 全覆盖、采样稳定性；绑定参数与旧线性桥接；核心导入不依赖 examples |
| `test_jobpacer_runtime_worker.py` | 空 job 集合、barrier、首个异常传播、其他线程唤醒、共享 deadline、不执行错误后的 finish、部分初始化收尾 |
| `test_jobpacer_runtime_results.py` | 正确产物、空/缺失/重复任务、成员缺失、launch 投影不一致、digest 不一致；既有指标字段及时间锚点 |
| `tests/unit/runtime/` | 既有协议、顺序、容量、结束、abort 和失败回归，不因重构删测试 |
| `test_runtime_replay.py` | 原线性、三个 DAG、静态生成/外部序列、rank skew、故障和非法输入；新导入与 CLI 参数传递 |

并发单测用事件/屏障控制交错，避免依赖短 sleep 碰运气。静态顺序和 digest 可以精确比较；动态真实运行不要求与某次历史 grant 次序、耗时或时间戳逐字相同。

拟执行命令（从仓库根目录）：

```bash
PYTHONPATH=src pytest -q tests/unit/runtime tests/unit/test_jobpacer_dag.py
PYTHONPATH=src pytest -q tests/unit/test_jobpacer_runtime_adapter.py tests/unit/test_jobpacer_runtime_worker.py tests/unit/test_jobpacer_runtime_results.py
PYTHONPATH=src pytest -q
env PYTHONPATH=src RUN_JOBPACER_RUNTIME_REPLAY=1 pytest -q tests/integration/test_runtime_replay.py
git diff --check
```

根据权限流程运行真实本地 TCP 测试，沙箱 socket 拒绝不是逻辑回归。不要安装或升级 PyTorch/CUDA/NCCL。GPU 不可用则明确未验收，不通过重构扩大实验规模。

## 10. 完成判据

- 正式 DAG 库不再依赖 sleep 样本、seed、JSON 文件位置或 examples；runtime 不反向依赖 DAG。
- 数据依赖与 group 顺序的共同构图逻辑只有一份，局部 tail 和全局合法性的作用域仍正确。
- 两条 replay 路径只共享生命周期，不混淆其执行/消费语义。
- 通用 runner 可通过自定义 compute callable 使用，不要求伪造 replay execution 配置。
- 输入摘要、采样算法、静态排序与结果 schema 不发生未说明变化。
- 原线性、DAG、故障及真实 Gloo 检查通过；新增回归覆盖本次改变的边界。
- `docs/JobPacer/result/phase3.12fix.md` 记录实际命令、环境、通过/失败/跳过、运行时长、产物位置与源码状态，GPU 等未验收范围单列。

此次交付是更清晰、少重复的同一套能力，不宣称重构自动带来性能收益或真实框架接入完成。

## 11. 后续问题修复（2026-09-21）

对初次重构验收中发现的边界补充修复，保留上文原实施记录：

1. 增加独立 `--setup-timeout`。ProcessGroup/group/control 初始化按 setup deadline 收敛；初始化完成后只创建一次 replay 绝对 deadline，runner、通信等待和 `finish_epoch()` 都使用其剩余时间。Coordinator 的 epoch watchdog 从首个控制事件启动，预算设为 setup + replay，避免在 rank setup 尚未结束时抢先超时；父进程回收期限为两段预算之和再加 5 秒。
2. 将指标字段改名为 `coordinator_epoch_duration_s`，明确它是首个 coordinator 记录到 `FINISHED`，包含注册阶段；不再称作 replay makespan。
3. `_collective()` 在 DAG 输入预检时只接受 `sum` reduction，并以 CLI 集成测试验证错误在 rank worker 启动前被拒绝。
4. 补 DAG runner 的重复完成、submit 失败 abort/后继不运行、deadline 诊断/不刷新预算测试；补 worker 错误后不执行 `finish_epoch()` 的直接测试，以及 Gloo launch/probe failure 集成覆盖。
5. 将 worker 并发测试中的 Timer、短 sleep 和耗时区间断言替换为 Event 与受控 join deadline。
6. 删除未使用的 `compute_ids`，去掉 `DagInput.tails` 的重算属性；每 rank 对 DAG 只算一次 tails，并传给静态排序和所有 job runner 复用。

这些修正的最终命令、实际测试结果和独立产物记录在结果文档的后续修复附录中。
