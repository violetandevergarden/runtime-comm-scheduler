# JobPacer Phase 3.1 / 3.2 CPU 实验结果

日期：2026-09-22。本文只记录本次实际执行的 CPU/Gloo pilot，不把单次运行写成稳定性能结论。

## 执行环境与输入

- CPU：Intel Core i7-14650HX，12 physical cores / 24 logical CPUs，1 NUMA node。
- OS：WSL2，Linux 6.18.33.2；Python 3.13.15；PyTorch 2.13.0+cu129；Gloo 可用。
- world size：2；backend：Gloo；dtype：float32；reduction：sum；新 runtime `max_inflight=1`。
- profile：`/tmp/jobpacer-phase3-profile.json`，4 KiB、warmup 5、measurement 30，group `[0, 1]`。
  实测 p10/p50/p90 为 `0.0005807/0.0006600/0.0008042 s`；策略使用冻结 p50。
- 线性运行固定 `epoch=0`、`compute_jitter=0.3`、`wait_budget_s=0.02 s`、completion poll `1 ms`，
  旧路径和新路径使用同一 workload、profile 与确定性 producer/consumer 样本。
- DAG 运行使用现有 `benchmark/phase3/{linear,diamond,multi-group}.json`，同样使用
  `compute_jitter=0.3`、`epoch=0`。没有把这些机制样例扩写成 G0–G4 的正式参数化矩阵。

新增批处理入口为 `examples/jobpacer/run_phase3_experiments.py`，每次运行保存 command、退出码、
stdout/stderr、原始 JSON，并生成 `manifest.json`、`runs.jsonl` 和 `summary.csv`。本次产物位于：

- `/tmp/jobpacer-phase3-batch-final-20260922/`：线性 8 组；
- `/tmp/jobpacer-phase3-dag-linear-20260922/`、`/tmp/jobpacer-phase3-dag-diamond-20260922/`、
  `/tmp/jobpacer-phase3-dag-multigroup-20260922/`：每个 DAG 五策略。

## 线性共同子集

下表为 `summary.csv` 中的 rank-local job duration 最大值定义的 makespan；每个配置只有一个
seed/repeat，单位秒。新 runtime 的“平均/最慢 job JCT”列实际来自 rank-local job duration，
不是跨 rank 同步 release 后的严格 JCT，不能据此声称全局时钟跨度。所有 8 条运行 `status=ok`，
tensor 校验与 runtime validation 均通过。

| 组 | policy | makespan | 平均 job JCT | 最慢 job JCT |
| --- | --- | ---: | ---: | ---: |
| B0 old bare | fifo label | 0.013782 | 0.012656 | 0.013638 |
| B1 old scheduler | fifo | 0.016161 | 0.015114 | 0.015989 |
| B2 old scheduler | ltf | 0.013796 | 0.013095 | 0.013672 |
| S0 new static | static_fifo | 0.244383 | 0.220251 | 0.244383 |
| S1 new static | static_ltf | 0.241855 | 0.219806 | 0.241855 |
| D0 new dynamic | fifo | 0.249703 | 0.225205 | 0.249703 |
| D1 new dynamic | ltf | 0.240584 | 0.219778 | 0.240584 |
| D2 new dynamic | lookahead | 0.241002 | 0.219916 | 0.241002 |

该表只能说明本次机器和小消息 pilot 的一次观测。新旧路径的完成探测、控制协议和容量实现不同，
因此 B0/B1/B2 与 S/D 的差值是整体系统差异，不能归因成 policy 收益；S/D 之间也没有足够
重复次数支持显著性或稳定排序。新路径本次明显慢于旧路径，记录为控制/完成路径开销信号，后续需
用独立 profile、重复和更长通信重新确认。

## DAG 五策略 pilot

下表为三个现有 DAG 样例的一次五策略运行，列出 makespan 秒；每行 5/5 成功。

| DAG | static_fifo | static_ltf | fifo | ltf | lookahead |
| --- | ---: | ---: | ---: | ---: | ---: |
| linear | 0.183963 | 0.181333 | 0.190094 | 0.177208 | 0.140602 |
| diamond | 0.093985 | 0.093039 | 0.094469 | 0.099905 | 0.089654 |
| multi-group | 0.280708 | 0.285452 | 0.267401 | 0.289098 | 0.233908 |

这些数字不是跨图比较的性能结论。机制 trace 提供了以下可复核事实：

- `multi-group/static_fifo` 首次运行记录 `STATIC_HEAD_BLOCKED ≈ 0.05305 s`，实际 dispatch
  序列保持静态队首；
- `multi-group/static_ltf` 首项为 `job-1/comm-c0`，没有跳过 group 内顺序；
- `multi-group/lookahead` 记录 `ACTIVE_LOOKAHEAD ≈ 0.000220 s`，并保留容量满区间；
- 三种 DAG 的每个策略均检查了预期节点/任务集合、每个 rank 的 grant/launch 投影和 all-reduce
  结果，未发现顺序或数值错误。

## 验证命令与结果

实验支撑改动后的针对性检查：

```text
PYTHONPATH=src pytest -q tests/unit/runtime tests/unit/test_jobpacer_comm_profile.py
41 passed
```

最终工作树全仓检查：

```text
PYTHONPATH=src pytest -q
170 passed, 35 skipped

env PYTHONPATH=src RUN_JOBPACER_RUNTIME_REPLAY=1 \
  pytest -q tests/integration/test_runtime_replay.py
30 passed in 85.46s
```

其中 35 个 skipped 包括默认关闭的本地 TCP replay 参数和不可用硬件路径；开启 opt-in 后的
真实双 rank 集成以及故障注入均通过。实验批次完成线性 8 条和 DAG 15 条真实 Gloo replay，
均为退出码 0、validation ok。

## 已完成的实验支撑改动

- 新 runtime 支持严格匹配的 `--comm-profile`，并在结果中记录 profile digest、环境和估值来源。
- 新旧线性 replay 都支持固定 `--compute-jitter` 与 `--epoch`；样本键包含 workload seed、epoch、
  job、communication、rank 和 producer/consumer segment，实际样本不会写入 policy hint。
- 新 runtime 暴露并记录 `--wait-budget-s`。
- 批处理入口顺序运行新旧线性组或同一 DAG 的五策略，失败运行保留原始输出，不从统计中静默删除。
- 新增确定性样本键回归检查；没有引入额外调度抽象或依赖。

## 未验收范围

本次没有完成：4 KiB/1 MiB/16 MiB 多尺度 profile、单 job isolated 分母、L0–L5 正式多 seed
矩阵、30 seed × 3 repeat 主结果、bootstrap 区间、rank-specific L5 straggler 输入、G0–G4
正式 DAG 参数化输入、一般 DAG bare、GPU/NCCL、跨 host、多 inflight、多资源和长期性能研究。
因此本文不回答“动态策略稳定优于静态”或“新 runtime 整体优于旧系统”。

## 完成反馈延迟修正复测（2026-09-22）

旧 pilot 原始 trace 显示，`completion_observed`/`completed_sent` 到 coordinator 收齐
`COMPLETED` 之间存在约 40--48 ms 的空档；本地 collective launch 到完成观测只有约 1.5--2.7 ms。
将逐通信 tensor 校验后置后该空档仍存在，线程数对照也未改变结果。检查控制通道后发现 client
和 server socket 均未设置 `TCP_NODELAY`；修正后同时保留 tensor 引用，将 correctness scan
统一移动到 `runtime.finish_epoch()` 之后，并增加 application release、drain、validation
边界和统一 `performance` 汇总字段。

修正后的单次 CPU/Gloo 复测：

- 命令：`run_runtime_replay.py --policy ltf --workload balanced --backend gloo --world-size 2 --compute-jitter 0.3 --comm-profile /tmp/jobpacer-phase3-profile.json`。
- 产物：`/tmp/jobpacer-phase3-followup-ltf.json`。
- `validation=ok`，两 rank 的 all-reduce 结果正确。
- 六项通信的 coordinator `all_submitted_to_all_completed` 为约 1.31--2.05 ms，中位数约 1.56 ms。
- application makespan 为 23.018 ms，communication drain makespan 为 23.943 ms；时间均按各 rank 本地单调时钟先求差，再取成员最大值。

该复测只证明本次完成反馈异常在同配置下消失，不替代上面的单次 pilot，也不支持动态策略性能收益结论。
正式 30 seed × 3 repeat、多尺度 profile、isolated 基线和持久化正式批次仍未完成；后续性能结果
必须基于修正后的代码重新生成。修正后的有限 block 重复见下节，不替代正式统计设计。

## 修正后小批重复

为检查修正后的测量链路，运行了 5 个新 runtime 策略、`seeds=0,1`、每个 seed 两次 repeat，
共 20 条真实 CPU/Gloo replay。产物目录为
`/tmp/jobpacer-phase3-followup-small-20260922/`；20/20 退出成功且 validation 为 `ok`。
该批次使用 `--order-seed 20260922` 按 `(seed, repeat)` 随机化策略 block 顺序，manifest schema
为 2，并保存 17 个相关源码文件的 SHA-256 摘要；source snapshot digest 为
`489f2e6c5b33a88ce5d1281cbc3d3d4c010df2a6d3afdc2282a6b46560c345b0`。

每个策略有 4 个样本，统一 application makespan 的中位数（秒）为：

| policy | 样本数 | makespan 中位数 |
| --- | ---: | ---: |
| static_fifo | 4 | 0.023292 |
| static_ltf | 4 | 0.021082 |
| fifo | 4 | 0.020799 |
| ltf | 4 | 0.020838 |
| lookahead | 4 | 0.021206 |

该小批只验证修正后的反馈、计时和 block 产物链路，不足以证明策略排序、统计显著性或新 runtime
相对旧路径的收益；正式矩阵、isolated 分母、多尺度 profile 和 DAG 参数化实验仍未完成。
