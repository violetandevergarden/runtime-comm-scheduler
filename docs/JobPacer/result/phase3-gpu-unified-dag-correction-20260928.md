# Phase 3 GPU 统一 DAG 修正：实现与语义复核

日期：2026-09-28。范围：schema-v2 GPU DAG 的共同 worker、old/new 通信 adapter、raw-ordered 参考和 D0 双卡语义 smoke。本文不报告策略性能收益。

实施合同与未完成 gate 见[统一 DAG 修正过程](../process/phase3-gpu-unified-dag-correction.md)。R0 修改前工作区快照、原始 rank 结果、故障注入、profile、batch smoke、命令、环境及源码快照保存在 `benchmark/phase3/results/gpu-unified-dag-correction-20260928/`；该目录被 Git 忽略，校验清单为其中的 `SHA256SUMS.txt`。

## 已验证范围

输入 `benchmark/phase3/experiments/dag-semantics/smoke/gpu-v2-fork-join.json` 包含两个 job 的 producer 分叉、all-reduce 与独立 matmul 分支、join、fill 及后继通信。输入文件 SHA-256：`1827a1d60044ee2bafc743513715333756509d14aedd0a9340a2ef26e2bac9df`；规范 `dag_input_hash`：`21296905a3ed972665f59c4e966cade6c827a16d3d4914486fd41a6ab49ac6ae`。

执行环境是 2× NVIDIA GeForce RTX 4090、Python 3.12、PyTorch 2.13.0+cu126、CUDA 12.6、NCCL 2.29.3。rank 0/1 UUID 为 `c024d768-866d-952a-36f6-899a9e5844e0`、`a376a831-0ace-b841-7b91-6b962b9c2398`。profile 由该 DAG 以 warmup=1、iterations=3 校准；profile 文件 SHA-256 为 `014b0d7969f3497d8e9cf2572fa945416726d301e2a2d7fb88d7809350ab06be`，profile content digest 为 `649e481e2229dc642b4dee6ac37ff5fb75accbf51ace32ce09b78662e56dc051`。

下表每配置仅运行一次，目的是检查语义与共同序列，不用于比较时间：

| 配置 | 结果 | 规范静态序列摘要 |
| --- | --- | --- |
| old static FIFO | 通过 | `37cc742d…6433669c` |
| new static FIFO | 通过 | `37cc742d…6433669c` |
| old static LTF | 通过 | `37cc742d…6433669c` |
| new static LTF | 通过 | `37cc742d…6433669c` |
| new dynamic FIFO | 通过 | 动态选择，无静态全序 |
| new dynamic LTF | 通过 | 动态选择，无静态全序 |
| raw-ordered static FIFO 参考 | 通过 | `37cc742d…6433669c` |

所有配置都核对了通信 task 集合、rank launch 投影、group 顺序、GPU buffer 与通信数值、以及 DAG 节点/终点完成。FIFO 与 LTF 在这个对称小图上产生相同静态顺序，不代表它们对一般工作量的选择相同。应用 duration 是单次 smoke 数据，受 warmup、系统噪声和执行路径影响，不作性能解释。

old 和 raw-ordered 各执行了一次 rank-0 `binding_failure`。首次 old 故障暴露 rank-1 NCCL 未匹配操作在 teardown 时崩溃；最终代码使用 rendezvous Store 发布 peer 失败、对已提交 Work 做有界 backend wait，并在销毁 ProcessGroup 前执行 teardown 握手。最终两 rank 都返回带来源的错误退出，没有 `SIGSEGV` 或 allocator corruption。此结果只覆盖 binding 故障，不代表网络断连或所有异常路径均已通过。

## 回归与 batch 检查

- R7 清理后的 `PYTHONPATH=src:. .venv/bin/python -m pytest -q tests/unit`：246 passed、3 failed。失败都来自 `tests/unit/test_benchmark_paths.py` 所需的历史迁移 fixture 缺失：`benchmark/phase3/multi-group.json` 与 `benchmark/phase1.2/result/batches/20260919T143653Z-9536db/`。
- 清理后的针对性回归（包含新的 drain-before-validation 时序测试）：141 passed，4.89 秒。
- `PYTHONPATH=src RUN_JOBPACER_RUNTIME_REPLAY=1 .venv/bin/python -m pytest -q tests/integration/test_runtime_replay.py`：47 passed，133.63 秒；`env PYTHONPATH=src RUN_JOBPACER_RUNTIME_NCCL=1 .venv/bin/python -m pytest -q tests/integration/test_runtime_replay_nccl.py`：4 passed，23.42 秒。
- D0 schema-v2 输入的 `run_comm_profile --dag` 双卡最小 profile smoke 成功，输出写到 `/tmp/jobpacer-comm-profile-dag-cleanup-smoke.json`。
- 相关文件 `py_compile`、三个入口的 `--help` 检查与 `git diff --check` 通过；当前 `run_phase3` 不提供旧 GPU-linear/bridge 参数，compute-profile 要求 `--dag`，communication-profile 支持 `--workload` 或 `--dag`，不再提供 `--gpu-linear`。
- R0 快照与 D0 结果归档的 `SHA256SUMS.txt` 均逐项校验通过。归档源码对应 D0 smoke 当时的工作树；R7 删除重复执行路径后未重新运行 GPU smoke。
- `run_experiments.py --preview` 显示 D0 的六个已支持配置、一个 seed、每 arm 一次，共 6 次；单独 old-static-FIFO batch smoke 为 1/1 成功。

## 尚未验收

原始 bare 没有实现；raw-ordered 是有共同静态发射顺序和串行 dispatch 的受控参考，不能代替 bare，也不能据此称七配置实验完成。L0/L1/D0–D3 六场景样本生成、5 个 workload seed 的冻结展开、六场景机制 pilot、正式 1,050 次配对矩阵、重复 epoch 和网络断连仍未执行。现有结果只证明一个 D0 固定输入上的小规模双卡语义，不证明策略收益、稳定性能或完整故障覆盖。

R7 清理已删除独立 GPU-linear/bridge replay worker、runner 和 DAG 推进循环；旧 schema-v1 输入留作历史资料，不能再由当前 replay/profile CLI 启动。清理只移除了重复执行路径，没有修改 schema-v2 DagRunner 或通信 adapter；上述 D0 双卡结果仍指向清理前归档源码，清理后的共享路径有 Gloo/NCCL 集成回归覆盖。

## 后续实现缺陷修正

解析器现拒绝计算程序输入/输出别名，回归直接覆盖 `sum_join(inputs=["z"], output="z")` 和 matmul 输出覆盖输入。old/raw abort、已接受 Work 的逐项等待、dispatcher/scheduler close，以及 worker 的 failure teardown rendezvous 均使用同一个绝对 replay deadline。old adapter 还将 submit 接受与 handle 登记同步到 abort 快照，避免并发提交漏出清理集合。

最终针对性回归 `PYTHONPATH=src:. .venv/bin/python -m pytest -q tests/unit/test_jobpacer_runtime_adapter.py tests/unit/test_jobpacer_dag_comm_adapters.py tests/unit/test_jobpacer_runtime_worker.py tests/unit/test_scheduler.py tests/unit/test_jobpacer_dag.py tests/unit/runtime`：125 passed，3.99 秒；Gloo 集成 `PYTHONPATH=src RUN_JOBPACER_RUNTIME_REPLAY=1 .venv/bin/python -m pytest -q tests/integration/test_runtime_replay.py`：47 passed，134.06 秒（该测试运行于仅影响 old/raw close 的边界调整前，Gloo 不走这两个 adapter）。确定性交错检查包含多个已登记 old Work、abort 开始时仍在 scheduler submit 中的 old Work，以及 raw dispatcher join 期间绑定的第三个 Work；所有等待收到递减的剩余时间。两张 RTX 4090 上针对 D0 输入分别运行 old 与 raw-ordered `binding_failure`；两个 worker 都以预期 Python error 退出，父 runner 未超时，也没有信号退出。最终 close 边界调整后，old 与 raw-ordered `static_fifo` 正常双卡 smoke 再次通过：两者均为 `validation.status=ok`，所有 collective/缓冲区数值正确，两 rank launch 序列一致。以上均为语义/故障 smoke，不是性能实验；故障 smoke 不等于所有真实 NCCL 网络/多 Work 故障路径验收。

## 2026-09-28 LTF 评分合同补充

本结果上文的 D0 双卡 Static/Dynamic LTF 记录来自当时的策略实现。此后移除了 TaskHint 的
`estimator_version` 字段及 Candidate/Anticipated 中的逐任务版本分支，统一 Dynamic LTF、
Lookahead 与 DAG Static LTF 使用 `estimated_comm_s + remaining_tail_s`。D0 对称小图上原有
Static LTF 顺序不变，但上文 GPU 结果没有以该次源码版本复跑；
不能将它当作新评分合同的双卡验收。当前回归只证明共用公式和可区分候选下的静态/动态选择，
该双卡 D0 smoke 不覆盖新公式。下节补充了专门区分两种评分结果的小 DAG NCCL smoke；其他
workload 的策略顺序和性能批次仍须按新公式重新运行。

### 当前公式的回归与双卡语义 smoke

2026-09-28 本次改动后重新运行针对性单测：
`PYTHONPATH=src:. .venv/bin/python -m pytest -q tests/unit/runtime tests/unit/test_jobpacer_dag.py tests/unit/test_jobpacer_runtime_adapter.py tests/unit/test_jobpacer_phase3_experiments.py`
为 **121 passed in 4.32 s**。新增覆盖 TaskHint wire fields 中没有逐任务公式字段、coordinator 面对
不同旧版本字符串仍以 `comm+tail` 选择，以及静态/动态 DAG 和 Lookahead 使用相同公式。
双 rank Gloo replay 命令
`PYTHONPATH=src RUN_JOBPACER_RUNTIME_REPLAY=1 .venv/bin/python -m pytest -q tests/integration/test_runtime_replay.py`
为 **47 passed in 131.20 s**。

另在两张 RTX 4090（PyTorch 2.13.0+cu126、NCCL 2.29.3）上，用一次性两组 DAG 输入做
Static LTF 与 Dynamic LTF NCCL 语义 smoke。两个通信候选分别为 A：`0.001 + 0.001s tail`，
B：`0.003 + 0s tail`，所以 tail-only 会选 A，而当前统一公式选 B。两次 replay 的
`validation.status=ok`、所有 collective 正确，rank 0/1 launch 序列均为
`job-b/comm-b → job-a/comm-a`。临时输入摘要为
`a355c58c0e76447bab51b4c1b274e1519899ad78110c0ae00510bc4409c910cb`；JSON 位于
`/tmp/jobpacer-ltf-score-static-gpu.json` 与
`/tmp/jobpacer-ltf-score-dynamic-gpu.json`。这两次各一轮的 smoke 只验证真实 NCCL 下的顺序投影
与数值，不比较时长、不作为收益证据，也不替代 M3 配对实验。

命令（输入是临时构造的 schema-v1 DAG）：

```bash
env PYTHONPATH=src:. .venv/bin/python -m examples.jobpacer.scripts.run_phase3 \
  --dag /tmp/jobpacer-ltf-score-smoke.json --policy static_ltf --backend nccl \
  --world-size 2 --timeout 30 --setup-timeout 30 --observation-mode minimal \
  --output /tmp/jobpacer-ltf-score-static-gpu.json

env PYTHONPATH=src:. .venv/bin/python -m examples.jobpacer.scripts.run_phase3 \
  --dag /tmp/jobpacer-ltf-score-smoke.json --policy ltf --backend nccl \
  --world-size 2 --timeout 30 --setup-timeout 30 --observation-mode minimal \
  --output /tmp/jobpacer-ltf-score-dynamic-gpu.json
```

最终 `py_compile` 与 `git diff --check` 通过。该结果只表示当前代码和上述小输入已验收；
其他历史输入的 LTF 顺序按相应归档源码解释，需要性能对比时应冻结新 manifest 后重新运行。

## 后续状态更正（2026-09-28）

本报告中“原始 bare 没有实现”是本报告执行时的状态。随后已添加 `BareDagAdapter` 和共同 worker/runner 入口，并在真实双卡 L1/D1/D3 输入执行诊断。数值与 buffer 校验通过，但 L1 出现两 rank 全局 launch 顺序分歧；bare 仍未通过安全资格。正式替代臂 `raw-ordered-static-fifo` 的 G1 故障路径后来以 first-writer-wins failure signal 修复并重新验收通过。当前完整状态见[七臂 preflight 执行报告](phase3-gpu-seven-arm-preflight-execution-20260928.md)；P2 的 D1 多候选 gate 仍失败，正式矩阵没有启动。
