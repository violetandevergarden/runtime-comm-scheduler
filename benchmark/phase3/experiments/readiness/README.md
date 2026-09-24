# Readiness 输入

L1 检查静态队首未 ready 时是否确有其他 eligible 工作；L5 检查指定成员的到达偏斜，并区分单 rank ready 与全成员
eligible。未观测到对应条件的 run 只算正确性样本，不作为策略机制收益证据。

- [`L1-head-misalignment.json`](L1-head-misalignment.json)
- [`L5-member-skew.json`](L5-member-skew.json)
- 批次编排见 [`../suites/compact.json`](../suites/compact.json)。
