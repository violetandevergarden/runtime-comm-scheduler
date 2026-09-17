总体分层方向是对的，但现在还不能认为 Phase 3.1 已完整验收：我确认了提交/完成之间的竞态、FIFO 顺序
  错误和 lookahead 截止处理问题，这些会影响运行正确性或实验结论。

  现有单元测试 68 passed。我另外用独立模拟脚本复现了三个问题；没有修改仓库代码，也没有改动你正在编辑的
  handle.py。本次未重新执行两 rank/GPU 实验。

  下面按优先级说明。

  1. 高优先级：COMPLETED 可能先于 SUBMITTED 发出

  位置：src/runtime_comm_scheduler/runtime/runtime.py:211

  Launch 线程先 handle.bind(work)，然后发送 SUBMITTED。但一旦 bind，completion 线程就能看到 BOUND，并发
  送 COMPLETED。

  我通过控制线程交错复现了：

  OFFER → COMPLETED → SUBMITTED

  Coordinator 会将其判为非法 completed transition，正常任务因此偶发失败。Transport 的发送锁只能保证消息
  完整，不能保证两个线程调用 send 的先后。

  建议明确一个“已上报提交、允许探测完成”的边界。对于当前单在途模式，也可以由同一个执行线程完成 launch、
  发送 SUBMITTED、轮询完成，直接消除这类跨线程竞态。

  2. 高优先级：Dynamic FIFO 实际可能按声明顺序选择

  位置：src/runtime_comm_scheduler/runtime/coordinator.py:394

  _decide() 遇到 inflight 就返回，导致在通信执行期间，新任务即使 OFFER 到齐，也不会立即获得
  eligible_seq。等容量释放后，才按照 tasks 字典顺序统一编号。

  我复现的场景：

  X 正在执行
  A 先声明，B 后声明
  B 先 OFFER 到齐，A 后 OFFER 到齐
  X 完成后，实际选择 A

  按设计的 FIFO 应选 B。这会直接污染静态/动态对照。

  应把“更新 eligible 状态”与“容量允许时选择任务”分开。容量满只能阻止发 grant，不能阻止记录任务何时成为
  合法候选。

  3. 高优先级：lookahead 的等待期限并不可靠

  位置：src/runtime_comm_scheduler/runtime/transport.py:130、src/runtime_comm_scheduler/runtime/
  coordinator.py:394

  有两个相互叠加的问题：

  - tick() 只有在输入队列连续空闲 50 ms 后才调用。毫秒级等待可能明显超时；持续有消息时，epoch timeout
    也可能一直得不到检查。

  - _must_dispatch 仅由 tick() 设置。消息在 deadline 后到达时，apply() 可以再次返回 Wait，而不是强制回
    退。

  模拟中，原截止时间为 1.005，在 1.010 输入新事件后，系统重新建立了第二轮等待，仍未发射已有候选。

  建议每次事件处理都检查截止时间；队列等待时长取最近 deadline 与当前时间的差。等待期间出现新候选，也应
  允许重算，保留原预算，而不是只检查原目标是否 ready。

  4. 高优先级：wait_host() 绕开了 runtime 自己的完成和失败状态

  位置：src/runtime_comm_scheduler/runtime/handle.py:94

  Handle 进入 BOUND 后，wait_host() 就直接阻塞于底层 work.wait()：

  - 之后 handle.fail() 的条件变量通知无法唤醒底层等待。
  - 底层返回后没有再次检查 handle 是否已失败，可能在 runtime 已失败时返回成功。
  - 捕获 TypeError 后调用无超时的 work.wait()，会丢掉调用者指定的 timeout。

  既然 runtime 已有独立完成探测，wait_host() 更适合等待自身的 COMPLETED/FAILED 状态。这样完成定义、超时
  和错误唤醒只有一条路径，也不必再猜不同 backend 的 wait() 语义。

  5. 中优先级：lookahead 的未来 ready 估计基准不正确

  位置：examples/jobpacer/runtime_worker.py:95、src/runtime_comm_scheduler/runtime/coordinator.py:360

  Replay 在 job 开始时声明所有任务，但每项 ready_after_s 只填自己的 producer 时间。Coordinator 将它解释
  为“从声明时刻起多久 ready”。

  因此，第二项及之后忽略了前序通信和计算。等它们成为 frontier 时，预测时间往往已经过去，策略就把它们当
  成马上会 ready 的任务。

  本阶段最直接的处理是：在每项 producer 开始前声明该项，用那个时刻作为预测基准。另需避免把
  ready_after_s=None 当成零延迟；未知时间不等于立即到达。

  6. 中优先级：执行状态、输入关闭和协议校验还有缺口

  位置：src/runtime_comm_scheduler/runtime/runtime.py:103、src/runtime_comm_scheduler/runtime/
  coordinator.py:118

  具体包括：

  - INPUT_CLOSED 后没有统一禁止新的 DECLARE/OFFER；coordinator 已 finished 后也没有完整拒绝新事件。
  - _fail() 标记 handle 失败，但没有停止 launch worker。已排队的任务仍可能先调用 backend，再在 bind 时
    发现 handle 已失败。

  - 本地处理 GRANT 只取 task ID 和 decision sequence，没有校验完整 TaskSpec、目标 epoch、group 序号。
  - register_group(..., process_group) 的实际对象参数被忽略；submit 也没有验证 tensor 与 TaskSpec 的
    shape/dtype/bytes 一致。

  建议建立明确的 runtime 生命周期，并在执行前检查失败状态；协议入口和本地绑定入口分别做校验。否则“两边
  声明一致”不等于“两边实际提交参数一致”。

  7. 当前日志和验收不足以支持“效果比较已完成”

  位置：examples/jobpacer/runtime_worker.py:123、结果文档 (docs/JobPacer/result/phase3.1.md)

  目前：

  - submit_ts 实际记录在 producer sleep 之前，不是提交时间。
  - EventLog 已定义但没有接入执行路径。
  - 中央部分事件没有时间戳，缺少策略完整估计输入。
  - 没有完整的 job 完成时间、grant→launch 延迟和等待分解。
  - 汇总程序输出 grant sequence，却没有独立验证各 group 的实际 launch 投影。
  - 新 runtime 的单元测试只有 6 个，主要覆盖模型和纯 coordinator；68 个通过不意味着新 runtime 的线程、
    transport、handle 已充分覆盖。

  因此，现在的结果更准确地说是“若干 CPU/Gloo 正常路径和故障示例通过”，尚不足以支撑计划中的完整机制验收
  和最小性能对照。

  效率与代码结构上，我最建议做三件事：

   改进                                          原因
  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   用独立 active/inflight 集合做完成探测         现在每毫秒扫描全部历史任务，成本随任务总数增长
  ────────────────────────────────────────────  ───────────────────────────────────────────────────────
   完成后释放 runtime 持有的 LocalBinding        当前 _tasks 一直保留 tensor、closure、keepalive，内存
                                                 随 epoch 累积
  ────────────────────────────────────────────  ───────────────────────────────────────────────────────
   定时器按 deadline 唤醒，日志按状态变化记录    同时减少空轮询、重复 idle 日志，并改善等待精度

  此外，GPU 路径应继续明确标为未完成：producer_event 尚未使用，wait_on() 将 Work 传给
  stream.wait_stream() 也不是完整的消费依赖实现。现阶段不应因为 CLI 接受 --backend nccl，就把它视为可用
  backend。

  我建议先修 回执竞态、FIFO、deadline、handle 等待，补上对应的确定性回归测试；然后补生命周期和日志，再
  重做静态/动态对照。现有分层可以保留，不需要再扩建架构。