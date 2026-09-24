# Communication chain diagnostics — Phase C (gated)

本组用于在 A/B 后仍存在新旧基础路径差距时，观察差值对 collective 次数和消息大小的依赖。输入全部为两
rank、Gloo、单 job/group、零 producer/consumer 计算；同一配置内每项 collective 使用相同的消息大小。

| bytes per collective | 1x | 8x | 32x |
|---:|---|---|---|
| 4 KiB | `4KiB-1x.json` | `4KiB-8x.json` | `4KiB-32x.json` |
| 1 MiB | `1MiB-1x.json` | `1MiB-8x.json` | `1MiB-32x.json` |
| 16 MiB | `16MiB-1x.json` | `16MiB-8x.json` | `16MiB-32x.json` |

使用 `benchmark/phase3/results/calibration/multi-scale/phase3-compact-20260923/profile.json` 注入已校准
的通信估计。主比较为 `old-fifo` 对 `new-static_fifo`，new 使用 `precreate`，poll interval 固定 1 ms，
每一输入固定 seed、重复 5 次；每个 seed/repeat 块内随机化 arm 次序。总量 9 配置 × 2 arm × 5 repeat =
90 replay。该重复是固定输入重复，不是多 seed 扰动实验。

执行前先对三种消息大小的 1x 输入各做一次 CPU/Gloo smoke，并根据实测单次耗时估计全批预算。九个配置分别
写入 `results/runtime-overhead/communication-chain/20260924-phase-C-main/<配置名>/`，不覆盖既有产物。

解释时观察差值是否主要为固定偏移、是否随 collective 次数增长、是否随消息大小增长，以区分可能的启动/唤醒、
逐轮控制和 backend/观测影响。不能仅据新旧差值断言纯控制通道成本；完成观测不是物理完成时刻，且多个计时区间
可能重叠。
