# 七臂 v2 放行检查记录（2026-09-29 至 2026-09-30）

## 2026-09-30 统一入口

Phase 3 GPU 实验统一从仓库根目录使用 `python -m examples.jobpacer.scripts.run_phase3`：顶层命令为 `prepare`、`qualify`、`check`、`run`、`analyze`、`finalize`。准备动作使用 `prepare generate/freeze/order/preview`；`run --plan-only` 只建批次计划，`run --resume` 恢复同一批次；pilot 封存和 readiness 封存分别使用 `finalize pilot`、`finalize formal`。

2026-09-30 后续目录整理将测量、恢复、bare 机制、层序敏感性等实现移入 `examples/jobpacer/diagnostics/`，退役 CPU/Gloo 控制路径诊断入口并归档其源码，且将诊断目录加入源码快照。旧 revision-1 审计仍保留作历史证据，但其源码摘要不匹配当前树，不能用于当前正式放行；按当前源码重新审核。

`check --status --suite-manifest PATH` 只读现有资格与放行证据；测量、恢复和 NCCL mechanism 诊断必须显式选择 `check --kind ...`。`analyze` 只分析已有批次。正式数据仍未放行，本记录中的 D1 门槛结论不变；这些入口不自动封存或启动正式采集。

## 当前结论

接手清单的预正式检查已完成。revision-1 pilot、七项 source/profile 绑定审计、GPU 辅助诊断、恢复彩排、测量检查、预算和回归结果均已归档。

**正式实验暂不放行：mechanism gate 中 D1 未达到预注册触发频率。** 其余六项 gate 通过。没有封存正式输入、没有 finalize，也没有创建或启动 150 块正式 batch。不要把 54 个 pilot 块算作正式采集。

完整事实与指标见[最终准备结果](../result/phase3-gpu-seven-arm-preparation-20260929.md)。所有原始 JSON、ledger、输入快照、验证日志和审计位于被 Git 忽略的 `benchmark/phase3/results/gpu-seven-arm/preparation-v2-20260929/`，需要独立备份。 revision-1 的 2,866 个文件由 `SHA256SUMS-revision-1.txt` 覆盖，清单 SHA256 为 `48f08cfce2af2a7c703f6a3262076b21ea5b7eac9ab7e9d6a1cfd0773dbcfa70`。

## 批次修订

revision-0 续跑在 178 条 ledger 记录处遇到真实 TCPStore `EADDRINUSE`。最初分类器错误要求异常文本包含字面量 `TCPStore`；实际 PyTorch `DistNetworkError` 文本不含该词。分类器改为结合结构化 startup attempts、rank 结果、退出码、超时及子进程组退出状态判定，并新增回归用例。

源码变化后，新建 revision-1 suite 和 pilot batch，没有把 revision-0 数据合并进来。revision-1 source digest 为：

```text
31c6524c63fbaaf6989991bb76a7f1583c9e42c13f8cb0f60ee911b7ef77931a
```

revision-1 输入：`benchmark/phase3/experiments/gpu-seven-arm-v2/revision-1/{pilot,formal}/`。

## 已完成的清单项

| 项目 | 结果 | 证据 |
| --- | --- | --- |
| 七臂 pilot | 54/54 配对块、378/378 有效 replay；ledger 385 条，含真实端口冲突及整块重试；矩阵与语义通过 | `pilot-seven-revision-1/analysis.json` |
| D1 机制 | 未通过：seed 9101 为 1/3、9102 为 0/3、9103 为 2/3；门槛是至少两个 seed 各达到 2/3 | `pilot-seven-revision-1/analysis.json`、`qualification/revision-1/mechanism-audit.json` |
| L1 机制 | 通过：三个 seed 均 3/3 触发静态 HOL 与动态合法绕过 | 同上 |
| D0 事件链 | 9/9 D0 块、36 个 dispatch 与 36 条 DAG join 链关联通过 | `d0-event-chain-audit-revision-1.json` |
| job 顺序诊断 | 16/16 replay 校验通过；默认/反转各一次/arm/场景，作为敏感性诊断，不按性能挑顺序 | `reverse-order-revision-1/analysis.json`、`auxiliary-diagnostics-revision-1.json` |
| 零计算链 | 24/24 replay 校验通过：4 KiB、64 MiB，8 个串行真实 collective，bare/raw/old/new 各 3 次 | `zero-compute-revision-4/zero-compute.json` |
| 组合干扰 | 16 个配置、每格 10 个 rank 样本，数值检查通过 | `interference-revision-1/interference.json` |
| A/A 与 minimal/full | 60/60 replay；30 组配对；五对/arm/kind；semantic、hash 与 minimal 机制检查通过 | `measurement-revision-1/measurement-audit.json` |
| 恢复彩排 | TCPStore 端口冲突、有界清理、fresh rendezvous、整块重试、中断恢复、hash 检查通过 | `recovery-revision-1/recovery-audit.json` |
| 回归 | targeted unit 189 passed；Gloo 47 passed；NCCL 4 passed；`git diff --check` clean | `verification/` |
| 全仓 pytest | 排除归档快照后 324 passed、48 skipped、3 failed；3 项为旧 benchmark 路径映射/fixture 缺失 | `verification/revision-1-full-suite-ignore-results.log` |
| 预算 | 1,050 次估计 p50 6,725 秒、p90 8,507 秒、raw 约 229 MB；检查时可用约 2.58 TiB | `budget-preview-revision-1.json`、`verification/revision-1-disk-space.txt` |

measurement CLI 的环境比对存在 tuple/list 序列化问题，首次尝试在 replay 前退出。源码未改；在运行函数入口将 live 环境记录 JSON 规范化后，plan 值校验相同，60 次 replay 完成。具体记录见 `qualification/revision-1/measurement-launch-record.json`。

## 七项放行审计

| Gate | 状态 |
| --- | --- |
| software | 通过 |
| backend | 通过 |
| mechanism | **未通过（D1 seed threshold）** |
| measurement | 通过 |
| rehearsal | 通过 |
| recovery | 通过 |
| budget | 通过 |

审计文件、输入 artifact SHA256 和综合状态：`benchmark/phase3/results/gpu-seven-arm/preparation-v2-20260929/qualification/revision-1/`。`readiness-status.json` 记录每个 audit 的哈希、profile hash 和 source digest；artifact 完整性检查无错误。

## 下一步边界

D1 的触发频率不足，故当前状态为 `not_ready_for_formal`。不得用辅助测量或其他场景替代 D1 门槛。若对 D1 workload/触发条件作预注册修订，应新建 revision，并按影响重做输入 profile、bare qualification、pilot 与绑定审计；不要按耗时挑选参数。只有七项 gate 都通过后，才封存正式输入并创建 150 blocks / 1,050 runs 的计划。正式采集仍需另行启动。
