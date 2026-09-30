# GPU 七臂 v2 实验前准备

2026-09-29。本记录落实[执行要求](phase3-gpu-seven-arm-implementation-20260928.md)，保留进入工作时已有未提交实现；起始 tracked patch 和七臂脚本副本暂存 `/tmp/jobpacer-seven-arm-preparation-start/`。未提交 commit，未删除历史输入/结果，未启动正式矩阵。

## 实现

- v2 manifest/CLI/batch/analyzer 使用真实 `bare-ordered`，拒绝 raw 替代臂和旧合同；容量按 engine 分别记录。bare 读取冻结 FIFO 文件并验证其等于默认规则生成的顺序。
- 新建六场景生成器、18 pilot 与 30 formal 展开输入。L0/L1 变为真正链式 DAG，D1 增加真实大通信占用 job，D3 跨组多终点。执行扰动与 nominal compute repeats 分离，静态及动态估计使用相同 nominal profile。
- compute profiler 同时收集实际/名义签名，保留两 GPU 原始设备/host 样本。冻结检查接受包含正式全集和额外可验证 pilot 输入的 profile，缺签名仍拒绝。
- full 分析按同一 coordinator timestamp 关联 snapshot/dispatch，记录 FIFO/LTF 反事实选择与 tie-break；D1/L1 按 seed 分别计数。minimal 保存候选直方图、分数区别、策略分歧及匹配计数。
- DAG 保存首次节点物理完成观察时刻，应用结束取各 job 的最后完成，避免元数据构造和线程清理进入应用时间；drain、CPU、校验仍单列。
- A/A 和 minimal/full 诊断有独立 plan/run/analyze 工具，覆盖 bare/old/new 各五对；完整块工具冻结正式 minimal 模式，保留源码/环境/input/profile/order/attempt，禁用内部启动重试，外层最多两块 attempt。
- 资格工具改为 bare 六场景和四类失败路径，子进程组有外层期限。正式晋级需 source/profile 绑定、artifact hash 可复核的各项 gate；不因性能方向或置信区间跨零而拒绝。
- 分析增加 bare→scheduler、old static→new dynamic、dynamic FIFO→LTF 配对以及逐块执行成本恒等式，按 scenario/seed 汇总，不将所有场景合并为同一总体。

## Backend 依据

当前执行器每 ProcessGroup 一个 gate stream：producer event→gate→backend collective；`Work.wait()` 在 gate 上插入 NCCL 完成依赖，之后记录完成 event，observer 查询该 event。裸发唯一 dispatcher 在共同 host 顺序下继续发下一项，不在 dispatcher 轮询前项完成。

NCCL 2.26 起支持 implicit launch ordering；CUDA 12.3 起其 launch completion events 允许 communicator kernels 并行。该性质要求所有设备 host 发射顺序一致，不能仅凭 peak receipt count 判断 kernel overlap。见 [NCCL communicator 文档](https://docs.nvidia.com/deeplearning/nccl/archives/nccl_2293/user-guide/docs/usage/communicators.html#using-multiple-nccl-communicators-concurrently)。本合同启动前拒绝旧 NCCL/CUDA、未启用 implicit 或 blocking-wait。

针对本机 PyTorch git `cf30153c4c131c8164ee7798e5022d810682e2cb`，已核对 [ProcessGroupNCCL.cpp](https://github.com/pytorch/pytorch/blob/cf30153c4c131c8164ee7798e5022d810682e2cb/torch/csrc/distributed/c10d/ProcessGroupNCCL.cpp)：`syncStream` 将 current stream event 传给 NCCL stream；`synchronizeStream` 反向插入完成依赖；无 blockingWait 且未传 timeout 的普通 collective `wait()` 不进入 CPU 完成轮询。版本检查、调用链依据和实际双卡验证分开记录。

## 验证与待晋级边界

最终命令、结果、产物路径见[准备验证记录](../result/phase3-gpu-seven-arm-preparation-20260929.md)。初始参数和工具可用于后续独立 pilot；D1 的多 seed 触发率、L1 HOL 频率、组合干扰/零计算尺度、完整 A/A 和观测消融、所有七臂彩排以及真实启动故障/中断恢复资格，均需其对应证据后才能晋级。一次短 smoke 不替代这些 gate。
