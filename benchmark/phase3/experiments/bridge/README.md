# Bridge 输入

G0 对照线性与 DAG 执行器。线性输入与 DAG compute 节点有显式映射，使用共同 seed/采样键和冻结交错顺序，且不含
consumer overlap。报告中需分开说明各 runner 自身启动与 DAG 推进开销。

- [`L0-no-overlap-bridge.json`](L0-no-overlap-bridge.json)
- [`G0-linear-bridge.json`](G0-linear-bridge.json)
- [`G0-interleaved-order.json`](G0-interleaved-order.json)
