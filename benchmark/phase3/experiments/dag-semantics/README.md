# DAG semantics 输入

G1 是无候选竞争的分叉/汇合负对照；G3 检查 group 顺序约束。两者用于验证 DAG 语义，不因执行成功就解释成策略
性能收益。`smoke/` 收纳迁移前位于 `benchmark/phase3/` 根目录的三个小样例。

- [`G1-diamond.json`](G1-diamond.json)
- [`G3-group-order.json`](G3-group-order.json)
- [`smoke/`](smoke/)：`linear.json`、`diamond.json`、`multi-group.json`
