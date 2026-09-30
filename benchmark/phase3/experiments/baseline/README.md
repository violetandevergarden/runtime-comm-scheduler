# Baseline 输入

L0 稳定均衡负载用于基础开销和跨阶段对照。比较时冻结 workload、profile、扰动样本、成员、warmup 与并发限制；
分别报告旧/新执行路径和新 runtime 静态/动态策略，不把整条路径差异解释成纯调度器收益。

- 输入：[`L0-balanced.json`](L0-balanced.json)
- 编排：[`../suites/compact.json`](../suites/compact.json)
- 验收：所有 arm 完成且 validation 为 `ok`；配对统计按 seed-block 计算。
