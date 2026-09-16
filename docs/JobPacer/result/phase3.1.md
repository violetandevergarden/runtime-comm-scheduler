# JobPacer Phase 3.1 实现与验收结果

日期：2026-09-16

已按 [讨论方案](../plan/discussion.md)、[执行计划](../plan/phase3.1.md) 和
[实际实现方案](../process/phase3.1.md) 完成 CPU/Gloo 版 Stage 3.1。此前的
`examples/jobpacer/dynamic/` 原型和旧动态 replay 入口已删除；新 replay 使用独立
`src/runtime_comm_scheduler/runtime/`，没有复用旧 Plan、AdmissionScheduler 或
TaskKey。

## 实现内容

新增 runtime 核心：

- `model.py`：`GroupSpec`、`CollectiveSpec`、`TaskSpec`、`TaskHint` 和 rank-local
  `LocalBinding`。group 直接使用规范化的 `ranks: tuple[int, ...]`，包含成员、序号、
  shape/dtype/bytes 校验和 JSON round-trip。
- `protocol.py`：版本化 NDJSON envelope、HELLO、上行事件和控制消息校验。
- `policy.py`：`StaticOrder`、DynamicFIFO、DynamicLTF 和 bounded lookahead 的纯策略决策。
  四种返回动作统一为 `Action = Dispatch | Wait | Idle | Done`。
- `coordinator.py`：单 event-loop 所有权的任务注册、成员 OFFER 匹配、group 序号、单
  inflight 容量、连续 event/delivery 序号、完成释放、正常结束和 fail-stop。
- `handle.py`：submit 立即返回的 `RuntimeHandle`，分离 grant、绑定、host wait、消费
  依赖和完成状态。
- `transport.py`：rank 到 rank 0 的独立 TCP 控制通道；连接启动有界重试，不使用受调度
  ProcessGroup 做控制 rendezvous。
- `runtime.py`：每 rank 的 control reader、单 launch worker、completion probe 和
  多 job 共享的 `RankRuntime`。
- `executor.py`、`telemetry.py`：Gloo/direct launch 边界、物理完成探测和 JSON-safe
  事件记录。

新增 replay：

- `examples/jobpacer/runtime_adapter.py` 把 workload 映射成独立 group/task/hint。
- `examples/jobpacer/runtime_worker.py` 创建真实 ProcessGroup、多个 job 线程和
  `RankRuntime`；job 线程只负责 producer、submit 和消费位置的 `wait_host()`。
- `examples/jobpacer/run_runtime_replay.py` 启动两 rank、分配 rendezvous/control
  端口、汇总 grant 序列和 all-reduce 正确性。
- `tests/unit/runtime/test_runtime_core.py` 覆盖模型、策略、成员匹配、容量释放、元数据
  冲突、静态队首阻塞和缺失静态任务。

`src/` 只新增 `src/runtime_comm_scheduler/runtime/`，旧 Phase 2 文件未修改。

## 验收配置

| 项目 | 值 |
| --- | --- |
| backend | Gloo |
| device | CPU |
| world size | 2 |
| collective | `all_reduce`，默认 4096 bytes |
| capacity | `max_inflight=1` |
| control plane | rank 0 TCP server + 每 rank ControlClient |
| completion | `Work.is_completed()` 独立轮询 |
| group | 每个 job 一个独立 ProcessGroup，成员由 workload ranks 决定 |

## 真实 replay

以下命令均在放行本地 TCP socket 的环境中完成，两个 rank 的 tensor 结果和 grant
投影均正确：

```bash
PYTHONPATH=src python examples/jobpacer/run_runtime_replay.py \
  --policy fifo --workload balanced --backend gloo --timeout 20

PYTHONPATH=src python examples/jobpacer/run_runtime_replay.py \
  --policy static_fifo --workload delayed --backend gloo --timeout 20

PYTHONPATH=src python examples/jobpacer/run_runtime_replay.py \
  --policy ltf --workload tail --backend gloo --timeout 20

PYTHONPATH=src python examples/jobpacer/run_runtime_replay.py \
  --policy lookahead --workload delayed --backend gloo --timeout 20
```

| policy / workload | 两 rank grant 顺序 | 结果 |
| --- | --- | --- |
| FIFO / balanced（一次运行观测） | `job-1/0, job-0/0, job-1/1, job-0/1, job-1/2, job-0/2` | 通过 |
| Static FIFO / delayed | `job-0/0, job-1/0, job-0/1, job-1/1` | 通过，队首阻塞保持 |
| LTF / tail | `job-1/0, job-1/1, job-1/2, job-0/0, job-0/1, job-0/2` | 通过 |
| Lookahead / delayed | `job-1/0, job-0/0, job-1/1, job-0/1` | 通过，记录主动等待/回退 |

两 rank 的 `grant_sequence` 完全一致；任务线程完成顺序可以不同，不能作为调度顺序。
rank 0 输出的 `decision_records` 保存 eligible、anticipated、decision_seq、等待和
完成事件，供后续纯决策重放。

## 测试结果

```text
PYTHONPATH=src pytest -q tests/unit/runtime tests/unit
61 passed
```

另外通过：

```text
python -m py_compile src/runtime_comm_scheduler/runtime/*.py \
  examples/jobpacer/runtime_*.py examples/jobpacer/run_runtime_replay.py
git diff --check
```

当前沙箱直接执行本地 socket 会被系统拒绝；真实两 rank replay 在允许本地 TCP 的提权
环境中通过。已有仓库 integration/gpu 测试未被本阶段替换；NCCL producer/consumer
stream 和物理完成边界仍未验收。

故障注入也已执行：

```bash
PYTHONPATH=src python examples/jobpacer/run_runtime_replay.py \
  --policy fifo --workload balanced --backend gloo --timeout 8 \
  --fault metadata_mismatch

PYTHONPATH=src python examples/jobpacer/run_runtime_replay.py \
  --policy fifo --workload balanced --backend gloo --timeout 8 \
  --fault missing_task
```

两次命令都以失败状态有界退出。前者报告 `task metadata mismatch for
job-1/comm-0`，后者由 coordinator 广播 `missing_offer` 并报告
`job-1/comm-2`；两者都没有报告成功或继续发射后续 collective。

## 边界

本阶段仍只支持线性 job、`all_reduce`、一个全局 inflight 资源和单机两 rank replay。
未实现 DAG、多 collective 在途、多资源、多 host、重连恢复、在线估计学习和 NCCL
验收。`estimated_comm_s` 仍是 workload hint，用于策略比较，不被伪装成 CPU 实测传输
时间。
