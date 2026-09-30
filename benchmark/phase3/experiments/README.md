# Phase 3 实验输入

输入按研究问题分类，文件名和 L/G 编号保持稳定。suite 中的路径相对于本目录；命令行和运行记录中的
仓库文件路径以仓库根目录为基准。多项实验复用输入时只通过路径引用，不复制 JSON。

| 类别 | 研究问题 | 输入与主要验收 |
| --- | --- | --- |
| `calibration/` | 通信尺度、串行/并发校准 | 多尺度和 two-groups；profile 需记录 backend、成员和环境 |
| `baseline/` | 稳定负载及跨阶段基础开销 | L0；八组执行路径使用相同 workload/profile/样本 |
| `readiness/` | 静态队首错位、成员偏斜 | L1 检查队首阻塞时的其他 eligible 项；L5 检查成员 OFFER spread |
| `priority/` | tail 优先级、多前沿竞争 | L2/G2 需在首次候选决策时证明同时 eligible |
| `lookahead/` | 主动等待、预测失准、安全前沿 | L3/L4/G4 检查 wait、deadline、到达和回退证据 |
| `dag-semantics/` | DAG 语义与负对照 | G1/G3；`smoke/` 保留旧小样例 |
| `bridge/` | 线性/DAG 执行器桥接 | G0 输入、显式映射、共同采样键及冻结交错顺序 |
| `isolated/` | 单 job 分母 | 输入须与 shared job 的 profile、计算样本和环境一致 |
| `noise/` | 固定 workload 的 A/B 系统噪声 | 通过配置引用 L0，不复制输入；不把噪声重复当成扰动样本 |
| `runtime-overhead/` | 准备、完成轮询与剩余控制路径开销 | 阶段 A/B 的固定参数和命令见子目录 README；阶段 C 为门控项 |
| `suites/` | 跨类别编排 | `compact.json` 是当前 compact runner 支持的格式；其他清单保留原用途并不自动视为可执行 |

`estimated_comm_s` 是输入占位估计；正式实验需使用严格匹配的 `--comm-profile`。运行成功不等于机制成立，
结果报告必须同时给出触发率和未触发/跳过状态。
