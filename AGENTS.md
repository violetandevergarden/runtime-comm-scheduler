# 仓库协作指南

## 项目与当前阶段

本仓库研究真实 PyTorch collective 的运行时通信调度。当前主线是 JobPacer Phase 3.1：用中心化 coordinator 根据在线状态决定通信准入，并比较静态顺序与动态策略。

开始工作前阅读与任务相关的文档：

- `docs/JobPacer/plan/discussion.md`：设计讨论与后续阶段边界。
- `docs/JobPacer/plan/phase3.1.md`：Phase 3.1 目标、协议、验收要求。
- `docs/JobPacer/process/`：实现与修复过程。
- `docs/JobPacer/result/phase3.1.md`：已记录的实验结果与未验收范围。

根目录 README 和部分早期设计文档仍描述旧的单 job、静态 Plan 路线，不能据此覆盖新的 JobPacer 计划。用户当前明确要求优先；文档描述的是目标或历史结果，当前能力需结合代码与验证判断。发现冲突时明确说明，不把目标当作已完成事实。

## 代码边界

- `src/runtime_comm_scheduler/runtime/`：新 runtime 核心，包括模型、coordinator、policy、本地执行、控制通道和观测。
- `examples/jobpacer/runtime_adapter.py`：workload 到新模型的映射。
- `examples/jobpacer/runtime_worker.py`、`run_runtime_replay.py`：新 runtime 的 rank harness 与启动、汇总入口。
- `tests/unit/runtime/`、`tests/integration/test_runtime_replay.py`：新 runtime 的单元与双 rank 集成检查。
- 包根目录的 `plan.py`、`scheduler.py`、`work.py` 等及旧 replay 属于历史路径，仍可用于基线和回归。

新 runtime 不以兼容或复用旧 Plan、TaskKey、AdmissionScheduler、ScheduledWork 为目标。不要为复用重新引入旧核心依赖，也不要未经任务要求删除历史实现、实验输入或结果。核心不得反向依赖 examples；示例负责构造 workload 和启动实验，不承担核心协议或调度状态机。

## 实现范围与原则

Phase 3.1 以线性 job、all-reduce、单个全局通信容量（`max_inflight=1`）、中心化 coordinator 为范围，优先完成 CPU/Gloo 机制；GPU/NCCL 需独立验收。

- 根据实际调用链和失败路径修复问题，不只针对一个示例补丁。
- 保持任务规范、策略估计和本地执行绑定分离。tensor、ProcessGroup、CUDA event、closure 留在本地。
- Coordinator 构造合法候选并提交共同决定；policy 只选择或等待，不能绕过成员、顺序和容量校验。
- 单事件循环拥有中央可变状态；各 rank 的实际 collective 提交由唯一入口按 grant 顺序执行。
- 简单具体的实现优先，不提前建设通用插件框架、资源图或多层调度服务。
- 保持 `pyproject.toml` 声明的 Python 版本兼容性；不要仅因本地解释器较新就使用更高版本语法。
- 修复用户指出的问题时补能复现根因的回归检查；并发检查尽量用事件/屏障控制交错，不靠大量 sleep 或重复碰运气。

后续衔接只保留必要边界：Phase 3.2 的计算 DAG 推进位于 adapter/workload 层；Phase 5 保持中心化管理，扩展进程部署与执行端归属；Phase 6 再增加多资源与并发选择。本阶段不提前实现这些功能，允许后续按需求修改 API。

## 必须保持的语义

1. 同一 group 的任务身份、规范序号和 collective 参数必须跨成员匹配，序号不能由各 rank 的线程到达顺序独立产生。
2. 每个 rank 的实际提交记录是共同 grant 序列的本地投影前缀。不得跳项、重排或重复发射；序号校验不能替代有序发布和有序执行。
3. Grant 是不可撤销的决定提交点。已提交 collective 不可在 admission 层伪装成已取消。
4. `submit()` 保存请求后返回 handle，不等待 grant。OFFER、GRANT、SUBMITTED、物理完成与应用消费是不同事件。
5. 对同一成员同一任务，SUBMITTED 必须先于 COMPLETED 上报；完成探测不得越过提交回执边界。
6. 严格串行容量从 grant 起占用，到全部成员物理完成才释放，不能取决于应用何时调用等待。
7. Host 完成等待与 consumer stream 依赖分别定义。不能默认底层 `Work.wait()` 的 CPU 返回就是 GPU 物理完成。
8. FIFO 按首次进入 eligible 的顺序选择，容量满时也要更新该顺序；不能用注册次序代替实际 ready 次序。
9. 主动等待使用固定 deadline；新事件可以触发重算但不得无限刷新预算。超时检查不能只在消息队列空闲时执行。
10. 输入关闭必须排在已接受提交之后。正常结束消息应可靠送达；失败应停发、唤醒等待者并有界退出，终态不得继续零等待空转。

控制消息不使用受调度的 job collective 作为 rendezvous。多 communicator 的 host 顺序一致不单独证明设备端并发安全；放宽在途容量前需要实际 backend 的独立验证。

## 验证命令

以下命令从仓库根目录执行，使用当前项目环境；不要擅自安装或升级 PyTorch/CUDA/NCCL。

```bash
# 新 runtime 的针对性检查
PYTHONPATH=src pytest -q tests/unit/runtime

# 全仓测试；注意报告 skipped 项
PYTHONPATH=src pytest -q

# 双 rank 真实 CPU/Gloo 集成，需本地 TCP socket 权限
env PYTHONPATH=src RUN_JOBPACER_RUNTIME_REPLAY=1 \
  pytest -q tests/integration/test_runtime_replay.py

# 单次 replay，输出存放在临时目录
PYTHONPATH=src python examples/jobpacer/run_runtime_replay.py \
  --policy fifo --workload balanced --backend gloo \
  --world-size 2 --timeout 20 --output /tmp/jobpacer-runtime-review.json

git diff --check
```

按改动风险选择验证范围。协议、线程、完成或关闭流程修改必须检查相应异常路径；需要真实通信的结论不能只靠 mock。Socket 被沙箱拒绝属于环境限制，按工具权限流程处理，不改代码绕过、不报告为逻辑回归。GPU 不可用或路径未验收时如实标注，不默认切换到其他设备或扩大实验规模。

## 实验与文档

- 静态/动态策略的主要对照共用新 runtime 执行路径；StaticOrder 队首未 eligible 时必须等待，不能动态跳过。
- 比较时固定 workload、profiling 估计、扰动样本、group 成员、backend、并发限制及计算/消费语义。
- 策略估计不得读取真实未来扰动。固定计算时长样本，不固定所有任务绝对 ready 时间。
- 不直接相减未同步的不同 rank 时钟。区分本地执行时间、中央接收时间、完成探测时间及设备完成时间。
- 检查实际 launch 投影、成员覆盖和 tensor 结果，不能只检查 grant 日志或进程退出码。
- 记录命令、软件/硬件环境、通过/失败/跳过、时长及结果路径。一次运行的顺序或性能不能写成稳定结论。
- 结果文档只记录已验证事实。单元测试通过、Gloo 集成通过、GPU 语义通过和性能收益分别报告。
- 计划写在 `plan/`，实施说明写在 `process/`，验收事实写在 `result/`；修正历史结论时保留上下文并说明勘误。

## 协作方式

默认使用中文解释，代码标识遵循现有 Python 风格。审查请求先给证据、位置、影响和优先级，不擅自实现修复；用户要求修改时完成实现与适当验证。

保护已有工作区修改，不覆盖无关内容。未明确要求时不提交 commit、不删除历史资料、不启动大规模或长时间实验。避免把固定测试计数、临时故障清单写成永久规则；完成后说明实际改动、验证结果和仍未验证的边界。
