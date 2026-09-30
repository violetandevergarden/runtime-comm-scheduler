# 七臂 v2 实验前验收（2026-09-29 至 2026-09-30）

本记录对应[执行合同](../process/phase3-gpu-seven-arm-implementation-20260928.md)。正式 1,050 replay 尚未启动。revision-1 pilot 与所有放行预检查已经完成；由于 D1 mechanism gate 未通过，当前不能放行正式 batch。

## 环境与归档

- 使用仓库 `.venv`：Python 3.12，PyTorch `2.13.0+cu126`，CUDA 12.6，NCCL 2.29.3。
- 两张 RTX 4090：`GPU-c024d768-866d-952a-36f6-899a9e5844e0`、`GPU-a376a831-0ace-b841-7b91-6b962b9c2398`；继承可见设备映射。
- 原始产物根目录：`benchmark/phase3/results/gpu-seven-arm/preparation-v2-20260929/`，被 Git 忽略，需独立保存。
- revision-0 是最初冻结输入。第一次续跑发现 TCPStore 端口冲突分类过严后，修复分类并按要求新建 revision-1 输入与 batch；本报告的最终验收以 revision-1 为准。
- Pilot 及后续配对诊断冻结 `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1`，减少应用计时以外 CPU 参考准备的线程争用；各 arm 相同。未覆盖 `CUDA_VISIBLE_DEVICES`。

## revision-0 首轮准备记录（历史）

| 检查 | 结果 | 时间 / 产物 |
| --- | --- | --- |
| compute profile：48 输入、44 GPU 签名记录 | 完成；warmup 5、iterations 30 | `profiles/compute.json` 与 sidecar |
| communication profile：D1 涵盖的通信签名 | 完成；warmup 5、iterations 30 | `profiles/communication.json` |
| bare 独立检查：A、B、完成串行、多在途各 5 次 / rank | 数值、backend、B 发射前 A 仍未完成的直接查询均通过 | `bare-mechanism/mechanism-summary.json` |
| bare 六场景正常与四类异常路径 | 全部通过，有界失败退出 | `qualification/qualification-audit.json` |
| 针对性 unit | 172 passed | 7.07 秒；`/tmp/jobpacer-seven-arm-targeted-final.log` |
| CPU/Gloo runtime 集成 | 47 passed | 125.09 秒；`/tmp/jobpacer-seven-arm-gloo.log` |
| opt-in 双卡 NCCL 集成 | 4 passed | 21.57 秒；`/tmp/jobpacer-seven-arm-nccl.log` |
| 全 unit 检查 | 309 passed、3 failed | 7.27 秒；三项历史 benchmark 文件缺失，见下文 |

GPU 集成和 bare 检查验证的是相应执行路径；没有据此宣称设备 kernel overlap 或净性能收益。多在途查询证据与延迟观察完成的 peak 计数分别保存。

全 unit 的失败是 `tests/unit/test_benchmark_paths.py` 中三个历史路径用例，引用缺失的 migration map/结果产物；本轮没有伪造历史结果或放宽断言。全 unit 通过数与最后针对性回归版本不同，不能合并成一个通过数。

## 最终接手清单执行结果（2026-09-30）

revision-0 续跑在 178 条 ledger 记录处遇到真实 `EADDRINUSE`。原分类器要求错误文本含字面量 `TCPStore`，但 PyTorch 返回 `DistNetworkError: ... EADDRINUSE`，因此错误地终止为应用/清理失败。修复分类器改为核验结构化 rendezvous startup attempts、无 rank 结果、非超时、非零退出及进程组退出；新回归覆盖实际错误文本和负例。针对性回归 189 passed。源码变化后按要求另建 revision-1，未把 revision-0 结果拼入新 batch。

- revision-1 source digest：`31c6524c63fbaaf6989991bb76a7f1583c9e42c13f8cb0f60ee911b7ef77931a`
- revision-1 pilot suite SHA256：`4eac6791dcf6cc80a51526ffd7b49272508c095b1cbce5ff4b5913ed985ab25c`
- 输入位于 `benchmark/phase3/experiments/gpu-seven-arm-v2/revision-1/`；原始结果位于 Git 忽略的 `benchmark/phase3/results/gpu-seven-arm/preparation-v2-20260929/`。

### Pilot 与机制审核

revision-1 完成 54/54 个七臂块、378/378 个有效 replay；ledger 保留 385 条 attempt 记录，包含一次启动前真实端口冲突及整块重试。矩阵完整、语义检查通过、validation errors 为 0。实际冲突发生在 `D3-order-and-sinks / new-static-fifo`，被分类为 `environment_port_conflict`，rank 未启动；随后七个 arm 均以 block attempt 2 成功。

| 机制场景 | seed 9101 | seed 9102 | seed 9103 | 结果 |
| --- | ---: | ---: | ---: | --- |
| D1 多候选、分数不同且 FIFO/LTF 分歧 | 1/3 | 0/3 | 2/3 | **未通过**；规则要求至少两个 seed 各达到 2/3 |
| L1 静态队首阻塞与动态合法绕过 | 3/3 | 3/3 | 3/3 | 通过 |

D1 的缺口是预注册的逐 seed 触发频率不足；实际出现分歧的 dispatch 选择校验正确。不能用 L1 或独立 A/A 诊断替代 D1 gate。D0 的逐事件链审计覆盖 9/9 个 D0 块、36 个 dispatch 与 36 条 DAG join 链；按 task/decision ID 关联 OFFER/eligible、dispatch、容量释放、实际 rank launch，未作跨 rank 时钟相减。报告为 `d0-event-chain-audit-revision-1.json`。

### 辅助诊断

- 默认/反转 job 顺序诊断共 16 次 replay，全部数值、任务和终点校验通过。单 seed、每种顺序每 arm 仅一次，结果只作顺序敏感性诊断，不按耗时挑默认规则。静态新 adapter 记录了队首等待；旧 adapter 没有该计时字段，报告为未测量而不是零。逐次结果见 `auxiliary-diagnostics-revision-1.json`。
- 零计算链完成 24/24 次 replay：4 KiB 与 64 MiB 张量、8 个串行真实 collective、四种执行路径各 3 次。workload makespan 中位数如下；它包含执行路径与 NCCL 成本，不能解释为纯 coordinator 或网络开销。

| 张量 | bare | raw-ordered | old | new |
| --- | ---: | ---: | ---: | ---: |
| 4 KiB | 22.75 ms | 15.64 ms | 9.28 ms | 51.11 ms |
| 64 MiB | 65.30 ms | 58.53 ms | 50.90 ms | 95.89 ms |

- 组合干扰诊断全部数值检查通过，16 个配置单元，每单元 10 个 rank 样本（两 rank × 五次保留重复）。64 MiB all-reduce 单独执行中位数约 4.44 ms；96 次小矩阵乘的设备时间约 1.10 ms；二者 concurrent/serial 的墙钟样本相近。该诊断不证明 kernel overlap，完整统计在同一辅助诊断 JSON。

### A/A 与 minimal/full 测量

60/60 次 replay 完成；bare、old、new 各有五组交错 A/A 与五组 minimal/full 配对，30 组配对均通过语义和 hash 检查，measurement gate 通过。A/A 中位绝对时间差：bare 1.167 ms、old 1.456 ms、new dynamic LTF 0.812 ms。Minimal 下，D1 样本中 new dynamic LTF 有 3/5 组观察到 FIFO/LTF 分歧；full 下为 2/5。CPU 时间、事件量和上下文切换逐 pair 保存在 measurement audit 与辅助诊断中，没有从 full 时间中减去假设的日志成本。

measurement 工具的 plan 将 NCCL 版本 JSON 序列化成 list，而运行时版本对象是 tuple，原 CLI 因类型不一致在首个 replay 前安全退出。源码未作更改；以归档启动记录中的内存 JSON 规范化方式重跑，plan 的环境值和当前环境相同，60 次 replay 正常完成。启动记录是 measurement gate 的底层 artifact。

### 恢复彩排、预算与七项审计

可控恢复彩排通过：真实 TCPStore 端口占用在 rank/app 启动前触发，有界清理、换新端口、整块重试、中断 reservation 恢复、source/input/raw hash 检查全部通过。详细审计为 `recovery-revision-1/recovery-audit.json`。

revision-1 正式顺序预览为 150 块 / 1,050 次 minimal replay。基于七臂 pilot 实际历史估算：总墙钟 p50 6,725 秒（约 1.87 小时）、p90 8,507 秒（约 2.36 小时），raw 约 228,894,450 bytes（约 229 MB）。检查时文件系统可用 2,835,973,484,544 bytes（约 2.58 TiB）。这是资源预算预览，不是正式计划，也未启动 replay。

七份 source/profile 绑定审计位于 `benchmark/phase3/results/gpu-seven-arm/preparation-v2-20260929/qualification/revision-1/`：software、backend、measurement、rehearsal、recovery、budget 通过；mechanism 未通过（仅 `seed_thresholds=false`，对应 D1）。所有审计的 artifact 路径和 SHA256 均已重新校验。revision-1 汇总清单 `benchmark/phase3/results/gpu-seven-arm/preparation-v2-20260929/SHA256SUMS-revision-1.txt` 覆盖 2,866 个输入、快照、raw、ledger、日志和审计文件；清单自身 SHA256 为 `48f08cfce2af2a7c703f6a3262076b21ea5b7eac9ab7e9d6a1cfd0773dbcfa70`，旁置校验文件为 `SHA256SUMS-revision-1.txt.sha256`。总体状态是 **not ready for formal**。没有执行 `seal-formal-inputs`，没有运行 finalize，也没有创建正式 batch plan。

全仓验证记录：

- `PYTHONPATH=src:. .venv/bin/python -m pytest -q` 在被忽略的历史 source snapshot 测试副本间触发 24 个同名模块 collection 冲突。
- `PYTHONPATH=src:. .venv/bin/python -m pytest -q --ignore=benchmark/phase3/results`：324 passed、48 skipped、3 failed；失败是 `tests/unit/test_benchmark_paths.py` 的三个旧迁移映射/历史 fixture 缺失用例。
- 针对性单元：189 passed；Gloo：47 passed；双卡 NCCL：4 passed；`git diff --check`：clean。

## 放行结论

当前实验准备工作与审计已完成，但正式采集没有达到预注册放行门槛。D1 未通过，不能冻结并创建正式 batch。下一步须先作出针对 D1 实验设计的明确修订决定；若改变输入或源码，应创建新 revision 并重跑受影响的资格和 pilot。不得根据当前耗时挑选 D1 参数或重排默认顺序。
