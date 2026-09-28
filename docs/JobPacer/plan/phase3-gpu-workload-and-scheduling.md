# Phase 3：统一 DAG workload 的 GPU 执行与通信调度设计

创建：2026-09-27；修订：2026-09-28。状态：修订设计，待按新路线实施。本次仅修改设计，不表示代码已经完成收敛，也不启动性能矩阵。

依据：[总体讨论](discussion.md)、[Phase 3.1](phase3.1.md)、[Phase 3.2](phase3.2.md)、[GPU 七配置实验计划](phase3-gpu-seven-arm-experiments.md)。具体迁移见[统一 DAG 修正实施计划](../process/phase3-gpu-unified-dag-correction.md)。

## 1. 本次修正与历史边界

2026-09-27 初稿提出独立 `jobpacer-gpu-linear` schema、线性执行器，再通过 bridge 转成 DAG。其目的是核对 sleep 到 GPU 的迁移语义，但将验证步骤变成了第二套长期执行架构。本轮实现已有 GPU linear runner、worker 和单独 bridge 推进循环；它们不适合作为七配置、六场景实验的共同主路径。

本修订取代初稿中“先 linear、再 bridge”的架构、schema、tail 和实施顺序要求：

1. 新 GPU 实验统一使用现有 `DagInput / DagGraph / DagRunner`，线性只是 DAG 的一种输入拓扑。
2. 扩展 `ReplayExecutionConfig` 表达 GPU 算子、buffer 数据流和执行合同；不再扩展 `GpuLinearInput` 为完整图模型。
3. 保留旧 `workloads.py` 的线性 `Workload`、旧输入与历史执行路径，用于复现和回归，不承担新 GPU 主模型。
4. 保留本轮新增的 CUDA 程序、初始化依赖、回执、profile 和数值校验能力，解除固定 segment 绑定后复用。
5. 所有七配置共用输入、计算推进、数据绑定、结果与统计，仅切换通信执行适配。
6. bridge 降为可选的离线输入转换与迁移校验，不再是必经阶段或独立实验模式。

[前轮实施说明](../process/phase3-gpu-workload-implementation.md)和[前轮结果](../result/phase3-gpu-workload-implementation-20260927.md)保留为历史事实；其中 M1/M2/M4/M5 的完成不等于本修订已完成。H 回归修复和 Phase 2 复测以该结果记录为准，不能继续将初稿中的旧故障状态当作当前实测结果。

七配置实验计划的七 arm、两个线性场景、四个 DAG 场景、五 seeds × 五 repeats 不变。其中引用旧设计的片段公式、线性接口或 bridge 前置要求，以本修订为准；DAG 估计和共同执行接口按本文统一。

## 2. 范围与成功标准

本轮支持单机两张获分配 GPU、真实 float32 SUM all-reduce、进程内多个 job。新 runtime 继续中心化管理、独立 TCP 控制面、全局 `max_inflight=1`。不加入多在途、中央计算资源调度、通用算子插件框架或跨机恢复。

最终目标是同一 DAG 输入可在 bare、旧静态 FIFO/LTF、新 static FIFO/LTF、新动态 FIFO/LTF 上执行。旧路径和 bare 需适配及独立验收，不能因有统一接口就声称已经支持。

不要求新策略获胜或 kernel 必须重叠。正式比较限于统一 GPU workload；CPU/Gloo 用于协议和历史回归，不作为 GPU 性能配对对象。

## 3. 当前能力与缺口

| 对象 | 可复用能力 | 尚需修正 |
| --- | --- | --- |
| `DagInput / DagGraph` | jobs、groups、节点、依赖、规范身份、通信参数、估计、全集和校验 | execution 目前以 sleep duration 为主，缺少通用节点 GPU 程序与数据引用 |
| `DagRunner` | ready/running/completed、完成依赖、异常传播、通信 handle、compute receipt | 明确计算通道、提交顺序合同；解除接口对单一通信执行路径的不必要假设 |
| `gpu_workload.py` | GPU 算子参数、profile 引用、seed 派生及校验 | `GpuJob.segments / GpuLinearInput` 不作为新主模型；GPU 描述改按节点绑定 |
| `gpu_compute.py` | 固定工作量、设备事件、初始化和校验 | `GpuSegmentResources` 拆解为节点程序及 job 本地 buffer 所有权，避免强制四阶段结构 |
| `gpu_dag_bridge.py` | 可借鉴输入转换与数据一致性检查 | 重复的节点状态机不进入统一路径 |
| `run_experiments.py` | arm、配对顺序、ledger、重试/分析基础 | 解除旧 arm 仅支持线性的限制，接统一 DAG worker，不再另建 batch runner |

上述为工作树阅读所得，不宣称已运行本修订的验收。

## 4. 统一模型与职责

```text
同一个 DagInput
  graph：节点、完成依赖、group/collective、策略估计
  execution：节点程序、tensor 数据流、seed、执行合同
                         ↓
                  同一个 DagRunner
              compute_fn / make_binding
                         ↓
     bare adapter | old scheduler adapter | new runtime adapter
                         ↓
               相同应用指标、校验与批次分析
```

### 4.1 图结构和调度规范

复用 `ComputeNode`、`CommNode`、`DagJob`、`DagGraph`，不再建 GPU 专用 job/group/graph 类。TaskSpec 和 group_seq 仍由规范输入决定，跨成员一致；同 group 的序号作用域是整个 epoch。

producer、independent、dependent 是实验节点的角色标签，不是核心必选节点类型。一般 DAG 可以有多个 producer、多条前沿、多终点，没有固定 segment 层。线性场景生成链式 DAG JSON 后走同一 parser/runner。

### 4.2 执行配置

扩展现有 `ReplayExecutionConfig`，按规范化的 `job_id/node_id` 索引：

| 内容 | 拟议字段 | 含义 |
| --- | --- | --- |
| 计算实现 | `mode` | `host-sleep` 或 `cuda-program`；与 Gloo/NCCL backend 分开 |
| 节点程序 | `compute_programs` | 算子、shape、dtype、repeats、输入/输出 buffer |
| 本地数据规格 | `buffers` | shape、dtype、初始化与 seed 引用；不含真实 tensor |
| 通信数据绑定 | `comm_bindings` | 每个通信节点使用哪个本地 buffer |
| 额外提交约束 | `submit_after` | 指定 compute 需等哪个通信请求被接受；不是等待通信完成 |
| 计算资源模型 | `compute_model` | 首版每 job 一个计算通道，所有 arm 相同 |
| 校准来源 | `profiles` | compute/comm profile 与估计版本 |
| 随机输入身份 | `sample_id` | 与协议 epoch、run/repeat 编号分离 |

GPU 程序参数与策略预计时长分离，实际 GPU duration 只作为输出。`estimated_duration_s / estimated_comm_s` 沿用现有节点字段，profile 在计时前生成或核验估计视图；不要在执行字段中复制第二份会漂移的 tail 状态。

旧 sleep `compute_duration_s`、`linear_sample_keys` 继续按旧 schema 解析。建议将新字段纳入 DAG schema v2，v1 仍保持原语义；不是新增 `gpu-dag` 根模型。v2 的确切字段在实现前通过解析测试冻结。

### 4.3 数据流必须显式

完成边 `A → B` 只说明 B 等 A，不能代替 tensor 引用。buffer ID 在 job 内唯一，输入输出引用由执行配置绑定。

- producer matmul 可直接写通信 buffer；输出 shape/dtype 必须匹配 collective。
- all-reduce 原地更新该 buffer，属于数据写入。读通信结果的节点必须有该通信的完成依赖。
- independent 程序使用独立 buffer，不在 all-reduce 在途期间读取或改写同一存储。
- 多次写入同一 buffer 必须由依赖明确排序；前一个值仍被下游需要时不能覆写。首版拒绝无法证明安全的复用和 view 别名。
- 初始化可作为 release 前已完成输入；运行时 producer/consumer 的因果关系仍必须显式。

校验输入引用存在、单节点形状合法、读写与依赖一致、跨成员通信规范一致、图/group/静态顺序无组合环。不要把“不重复分配”误当作允许任意共享存储。

## 5. DAG v2 输入草案

以下仅展示数据结构，不是当前可运行输入。预计时长为示例值；正式实验由冻结 profile 给出。省略 profiles 表示只能做非估计驱动 smoke，不能作为 LTF 正式样本。

```json
{
  "schema_version": 2,
  "name": "gpu-dag-example",
  "seed": 8101,
  "groups": [{"group_id": "g0", "ranks": [0, 1]}],
  "jobs": [{"job_id": "job-0", "nodes": [
    {"node_id": "p", "kind": "compute", "deps": [], "estimated_duration_s": 0.001},
    {"node_id": "c", "kind": "comm", "deps": ["p"], "group_id": "g0", "group_seq": 0,
     "estimated_comm_s": 0.001,
     "collective": {"op": "all_reduce", "shape": [512, 512], "numel": 262144,
                    "num_bytes": 1048576, "dtype": "float32", "reduction": "sum"}},
    {"node_id": "u", "kind": "compute", "deps": ["p"], "estimated_duration_s": 0.001},
    {"node_id": "d", "kind": "compute", "deps": ["c", "u"], "estimated_duration_s": 0.001}
  ]}],
  "execution": {
    "mode": "cuda-program",
    "sample_id": "L0-seed8101",
    "compute_model": "one-active-compute-per-job",
    "buffers": {
      "job-0": {
        "a": {"shape": [512, 512], "dtype": "float32", "init": "seeded-random"},
        "b": {"shape": [512, 512], "dtype": "float32", "init": "seeded-random"},
        "x": {"shape": [512, 512], "dtype": "float32", "init": "empty"},
        "y": {"shape": [512, 512], "dtype": "float32", "init": "empty"},
        "z": {"shape": [], "dtype": "float32", "init": "empty"}
      }
    },
    "compute_programs": {
      "job-0/p": {"op": "matmul", "inputs": ["a", "b"], "output": "x", "repeats": 2},
      "job-0/u": {"op": "matmul", "inputs": ["a", "b"], "output": "y", "repeats": 4},
      "job-0/d": {"op": "sum_join", "inputs": ["x", "y"], "output": "z"}
    },
    "comm_bindings": {"job-0/c": {"buffer": "x"}},
    "submit_after": {"job-0/u": ["job-0/c"]}
  }
}
```

shape 由 buffer 规格确定，matmul m/n/k 由输入输出校验推导，不再维护冲突的双份 shape。repeats 表示固定次数覆盖同一输出，不表示递归更新。`fill`、`matmul`、`sum_join` 复用本轮具体程序；不新增通用插件系统。没有 dependent 算术时可直接以通信和独立计算为多个终点，不为凑四阶段强造节点。

## 6. DagRunner 的统一执行合同

### 6.1 采用已有物理完成推进

首版保留现有完成依赖模型：计算 receipt 的设备事件完成才解锁后继；通信 handle 的物理完成才解锁后继。调用返回或 event 入队不能视为完成。

因此上述 d 节点在 c/u 均物理完成后才入队。它与初稿“提前给 consumer stream 建依赖、等待末端 event”的执行方式不同，是本次明确的合同修正，需新 batch，不要求复现旧 S/bridge 的数值耗时。所有 arm 必须共享此模型，不能一个 arm 用 host 完成推进、另一个用提前 stream 图。

`wait_on` 能力继续保留并单独验收，但不是统一 DAG 首版的必经阻塞调用。未来若要让后继在前驱未完成时提前排设备依赖，另行设计图边语义，不藏在某个通信 adapter 中。

### 6.2 计算通道

首版沿用每 job 一个 active compute：从节点入队到 receipt 物理完成前不启动该 job 下一个 compute，job 间可并行，通信仍可独立推进。这是 runner 的本地资源模型，不代表每 job 独占 GPU。

独立计算可与通信并行，但同 job 两个独立计算分支可能串行；D1 等场景必须记录这个限制。若后续需要多个计算 stream 同时在途，统一修改 runner 的显式资源参数、所有 arm 一起验收；不恢复另一套 GPU DAG 状态机。

### 6.3 通信请求与独立计算的提交顺序

线性拓扑若需保留原 JobPacer 合同，用 `submit_after[u]=[c]` 明确要求 u 在 c 的 submit 成功返回后入队。它不要求 c 获 grant，也不要求 c 物理完成，更不要求 NCCL kernel 先开始。

仅给现有 runner 增加具体的提交门控：compute 的完成依赖已满足、所列通信已有 handle、计算通道空闲，才可启动。不能仅依赖“当前代码先遍历通信”保证将来仍成立。

首版约束此门控只引用同 job 通信，且该通信的全部完成前驱必须已包含在 compute 的完成前驱集合中；拒绝自依赖、非法类型和产生循环的配置。该限制覆盖 producer→comm/independent 模板，不建设通用多类型依赖框架。一般 DAG 不需要该关系时省略，不自动对所有 ready 通信施加全局门槛。

### 6.4 应用完成、容量与关闭

job 完成等待所有图终点物理完成，包含没有算术 consumer 的通信 sink；应用完成取所有本地 job 完成观察时刻。validation 在应用终点后执行，protocol drain 单独记录。

新 runtime 的通信容量从 grant 占用，到全部成员物理完成释放；不等待下游计算或应用消费。SUBMITTED 先于 COMPLETED，实际 launch 始终是共同 grant 的本地投影前缀。

只有 rank 共同拥有者在所有本地 job 结束后关闭输入一次。失败停止新节点与通信、唤醒等待者、按固定 deadline 有界退出；已提交 CUDA 工作不能伪装成取消。

## 7. 三种通信执行适配与七配置

复用 runner 的 `submit(spec,binding,hint)`、handle 完成查询及错误传播语义。为旧 scheduler 和 bare 添加薄适配，真实对象留在本地。可用最小结构协议标注接口，不建设通用 runtime 插件框架。

- 新 adapter 委托现有 RankRuntime，不更改 coordinator 的合法性和容量。
- 旧 adapter 将规范通信身份确定性映射到 TaskKey，并装载统一静态通信序列；不重新推进计算节点。
- bare adapter 绕过 admission policy，但仍需目标 backend 的跨成员发射顺序与设备依赖保证。若只能提供受控 `raw-ordered`，明确命名及成本，不能替代未验收原始 bare 而不说明。

所有 adapter 的 submit 不得等待 grant 或物理完成，避免阻塞 DAG 计算推进。旧 Work 的 stream wait 不能当作 host 物理完成；统一 handle 契约须由实际回执支持。epoch/ProcessGroup 生命周期在 worker 管理，不塞进 DagRunner 节点逻辑。

旧核心、新核心保持独立，适配位于 examples/workload 层；核心不得反向依赖 examples。现有 batch runner 扩展同一 DAG 输入的 arm 选择，不再为 GPU linear 独建统计路径。

## 8. 策略估计与共同静态 Plan

复用 `compute_tails`、`build_static_order`、`validate_static_order` 及 `dag_task_spec/hint`。DAG 统一 tail 为：

```text
tail(v) = max(duration(u) + tail(u) for u in successors(v))
tail(sink) = 0
```

继续遵循现有本 job 的依赖及 group 顺序定义；跨 job group 约束用于全局合法性，不把其他 job 的工作计入本 job tail。静态/动态 LTF 共用 tail 优先的评分合同，稳定 tie-break 单独记录；不再使用 `gpu_linear_static_tail_v1` 作为统一模型的公式。

这个 tail 是结构关键路径启发式，不计汇合的其他分支残余、单计算通道排队、准入等待与设备争用，不声称精确剩余 JCT。需要验证选择与局限，不提前建设在线估计更新协议。

旧静态 FIFO/LTF 与新 static 对应使用同一通信序列，并分别转换为旧 Plan 和新 StaticOrder。旧历史 LTF builder 不修改；本轮旧引擎执行共同 GPU Plan，结果注明 plan 来源和 estimator version。

FIFO 可先 smoke，但七配置正式矩阵必须完成 LTF 估计与校准；Lookahead 不在七配置必选集合，不应先扩它而遗漏 bare/旧 DAG。未来 Lookahead 复用已有安全前沿与固定 deadline 规则，单独验收。

## 9. 校准、随机性与计时

计算 profile 保留实际算子/shape/dtype/layout/repeats、precision、设备与软件签名；通信 profile 复用现有工具。节点 role 可作注释，不应成为不可替代的 segment 身份。估计视图与执行工作量视图分开，随机未来执行扰动不向 policy 泄漏。

数据 seed 按 `(scenario/sample_id, workload_seed, job_id, node_id或buffer_id, rank)` 派生，与 repeat、run ID、协议 epoch 分离。重复测量只改变运行身份，不改变数据；旧采样键留在历史路径，不强制重写。

正式 release 前预分配、初始化、精确算子 warmup，并建立 init stream 到使用 stream 的依赖。warmup 后恢复被 reduction 修改的 buffer；重复 epoch 不积累数值污染。准备、校准、validation、drain 单列。

应用 makespan 为 `max_rank(local_end-local_release)`；逐 job JCT 等待所有终点。CUDA event 时间不等于纯 kernel 活动，profiler 单独诊断；host/coordinator/device 时钟不未经校准相减。

具体 1,050 次矩阵、配对块、失败和统计规则遵循七配置计划。缺失 arm 时明确未完成，不改总数凑验收；CPU sleep 不作为性能基线。

## 10. 迁移与验收要求

| 层次 | 修正后的验收 |
| --- | --- |
| 输入 | v1 历史兼容，v2 GPU 节点映射完整；拒绝危险数据别名、缺失依赖、非法 submit_after |
| 执行 | 只有一个主 DagRunner；阻塞 grant 时独立计算仍能启动；compute query 错误按现有失败路径传播 |
| GPU | producer/通信/消费数值、初始化依赖、终点、重复 epoch、错误设备与回执分别验收 |
| adapter | 三种通信执行路径同一提交/完成/错误合同；裸发顺序单独验证，不靠有限次不挂推断 |
| 策略 | 静态序列全集/合法性与新旧一致，静态/动态 LTF 同估计，FIFO 仍按首次 eligible |
| 实验 | 两个链式 DAG + 四个一般 DAG 均走相同入口、runner、结果和批次协议 |
| 历史 | 保留旧 Workload、输入、raw、结果及旧路径回归；新旧合同不混入同批统计 |

并发回归尽量用事件/屏障控制交错，真实通信结论需要双 GPU 验证。格式或 unit 通过不替代设备完成验收。

实施顺序为：输入与程序解耦 → 统一 DagRunner 新 runtime 路径 → 旧/bare 适配和共同 Plan → batch 七配置接通 → 六场景 smoke → 冻结正式矩阵。不再单设 GPU linear 或 bridge 性能里程碑。保留何种兼容入口及何时退役重复代码，按 process 修正方案执行，不删除历史产物。
