# Lookahead 输入

L3/L4/G4 分别覆盖主动等待目标、迟到预测和 DAG 安全前沿。验收必须读取 trace 中的目标、预测 deadline、成员到达、
等待时长及 fallback；未形成等待或目标时序的 run 不算机制触发。

- [`L3-lookahead.json`](L3-lookahead.json)
- [`L4-late-prediction.json`](L4-late-prediction.json)
- [`G4-prediction-frontier.json`](G4-prediction-frontier.json)
- 性能扩批受 suite 中预注册的机制门槛约束。
