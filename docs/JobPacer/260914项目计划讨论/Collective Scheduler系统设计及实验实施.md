## 1. 项目背景

研究问题：LLM training场景下的流量调度，也即通过控制训练过程中产生流量的调度顺序、传输路径、发送速率、网络优先级等，实现降低整体makespan、降低网络拥塞、提高资源利用率等优化目标

前期进展：
- **流级模拟器**：基于SimAI项目，实现了一个P2P flow-level的LLM训练/推理任务模拟器，包含了DAG workload生成、多任务模拟、可插拔的自定义流级调度策略、通信计算窗口可视化等功能。可以用来模拟调度算法在大规模集群中的表现，目前基于此已经复现了cassini、hermod、puppeteer等已有工作。
- **DAG调度理论分析**：基于DAG模型，对LLM training进行了数学建模与理论分析。关键结论表明，在单链路模型中，最优调度一定可以表示成这样的形式：维护当前活跃的传输流集合S，每当S发生变化（新流加入或旧流完成）时调度器介入，决定下一段时间执行传输的流让其独占带宽传输。这一发现决定了我们后续schedule mechanism的设计（难点：如何提供一种通用的调度机制，能够尽可能支持多种多样的自定义策略）。
- **多job并行链调度的建模**：由于通用DAG调度模型的NP困难性，我们转而考虑一种多并行链的特例情形，证明了一定的近似性结论，探索了某些启发式调度策略。尽管LLM training任务往往并不是多并行链的结构，但我们注意到这种模型特别适用于multi-tenant场景下的多job并行的简化建模（例如cassini等很多先前工作把单job建模为通信-计算交替的单链），因此我们决定先从这个具体的场景入手，进行实验探索。

当前阶段：**真实training stack中的调度系统设计**
- 为了使我们的工作成为LLM训练栈中真实可用的系统，而非仅仅停留在conceptual idea或者是模拟器的层级，我们需要在当前的LLM训练框架中实现一个scheduler层，提供通用的调度机制，支持后续多种多样的策略设计。
- 之前的流级模拟器在调度机制上，采用的是per-flow带宽分配模型，虽然能提供足够的自由度供用户定义调度策略，但真实 PyTorch/NCCL stack 不会向上层暴露这样的逐 flow 事件和控制接口，因此这些策略不能直接移植。真实训练stack中应当如何设计新的流量调度机制成为我们接下来需要考虑的问题。

## 2. Training Stack 与控制边界

通信请求的通常调用流程示意

```text
Megatron / training framework
  -> torch.distributed API
  -> c10d ProcessGroup
  -> ProcessGroupNCCL
  -> NCCL communicator、channels、kernels
  -> NVLink / PCIe / NIC / network
```

调度自由度自顶向下递减：越靠近训练框架，越能改变"何时、以何种粒度发射"通信；越靠近网络，越能改变"每条流实际怎么传"，但接口暴露越来越少。

| Training Stack 层                                        | 可观察事件                                                             | 可控制行为（流调度机制）                                             |
| ------------------------------------------------------- | ----------------------------------------------------------------- | -------------------------------------------------------- |
| ① 训练框架层<br>Megatron / training loop + framework adapter | iteration、layer、microbatch、DP/TP/PP 等训练语义信息                       | 将训练语义信息传递到下层，辅助scheduler决策                               |
|                                                         | tensor、梯度bucket何时ready                                            | 调整梯度 bucket 划分与大小<br>                                    |
|                                                         | 通信-计算依赖关系以及overlap窗口                                              | 在不破坏依赖关系的前提下，基于训练语义延迟、合并、拆分、调换 collective请求的顺序           |
| ② 集合通信 API 层<br>torch.distributed                       | c10d collective API 调用及参数（op 类型、size/shape/dtype、PG、async handle） | 将c10d API参数作为元信息传递到下层，辅助scheduler决策                      |
|                                                         | 异步 `Work` 对象的完成状态                                                 | 延迟或准入 collective API 调用                                  |
|                                                         | 训练代码中的 `wait()` / 同步点位置                                           | 合并（coalescing）或者拆分（chunking）collective请求，创造更细的调度粒度       |
| ③ ProcessGroup 层<br>c10d                                | collective 提交与完成（通过底层 `Work`、future 或 CUDA event）                 | 同一 PG 内尚未提交 collective 的顺序<br>有条件可控制：所有成员 rank 必须保持一致    |
|                                                         | 各 PG 内部的 collective 序列                                            | 不同 PG 之间的发射时机交错<br>有条件可控制：不得破坏各 group 的诱导顺序与训练依赖         |
| ④ NCCL 实现层<br>ProcessGroupNCCL / communicator / kernel  | CUDA stream / event 依赖关系                                          | 将 collective 绑定到不同 CUDA stream，用 stream/event 显式管理依赖与并发  |
|                                                         | communicator 状态与 channel 建立                                       | communicator splitting、channel 数量                        |
|                                                         | ncclProfiler 提供的 chunk / channel 级进度                              | algo / proto（ring、tree、NVLS）选择，甚至按chunk级粒度自定义各种集合通信算子的实现 |
| ⑤ 数据面 / 网络层<br>NVLink / PCIe / NIC / network            | packet 或 flow 的进入 / 退出<br>                                        | 每条 flow 的带宽分配、优先级控制                                      |
|                                                         | per-flow 队列、拥塞、RTT 等指标                                            | 路径 / 路由选择、拥塞控制算法、网络信号反馈                                  |

## 3. 系统设计与长期目标

目前决定在以下几个层级入手：

**应用层（Training Framework & pytorch）**
- 主要机制：Admission Scheduler，它接收上层应用提交的集合通信请求并且放置到自己内部的任务队列中，根据一定的调度策略决定请求的执行顺序以及何时释放每个请求；
- 预期目标策略：在multi-job多并行链调度场景，根据预先计算出的调度结果，依次释放各个job的通信请求，手动延迟某些已经ready的请求来避免冲突；

**CCL层（NCCL等通信库）**
- 主要机制：基于组里之前的工作theseus，根据scheduler层拿到的信息，动态决定集合通信算法的选择
- 预期目标策略：例如，给并发的通信任务选取不同的网络路径从而最大化利用网络带宽资源；

**网络层（NVLink / PCIe / NIC / network）**
- 主要机制：交换机上的拥塞控制算法或者优先级队列
- 预期目标策略：根据上层传下来的训练语义信息或者profiling阶段预先计算好的调度结果，决定每条流的物理优先级，或者是拥塞控制算法的参数；

目前我计划主要完成应用层的调度机制设计，更底层作为可选项，视后续项目进度和需求，决定是否需要实施。
## 4. 当前阶段基础

针对应用层的调度机制，已经实现了一个初步版本artifact，作为本阶段的开始基础。项目仓库位于[仓库链接](https://github.com/hovering-clouds/runtime-comm-scheduler)，以下简要介绍当前已完成的模块设计：

1. CommIntent
	- 包装原本的torch.distributed请求，用于附带一些标准的元数据（例如iteration、layer、microbatch、DP/TP/PP 等训练语义信息），作为通信任务的基本单元传递给scheduler进行统一调度；
2. AdmissionScheduler
	- 提供submit()接口，training应用可以包装好CommIntent后调用此接口提交通信请求；
	- 内部有一个专门的worker线程，不断检查内部的任务队列，根据预先定义的规则决定接下来lauch哪一个任务，或者进行延迟等动作。目前调度策略固化在AdmissionScheduler内部，还未整理成可插拔的模块，自定义新的策略可能需要重新实现别的scheduler类；
3. ScheduledWork
	- AdmissionScheduler内部的任务队列存放的任务对象，包装了torch.distributed任务实际调用后返回的work对象，提供wait(), is_completed()等异步调用的方法
4. LaunchExecutor
	- executor负责ScheduledWork对象的实际执行。
	- 目前实现了两个执行后端，一个是基于Gloo通信库的DirectLaunchExecutor，用于CPU环境，通常可以在单机上建一堆进程的方式来模拟多机通信；另一个是基于NCCL通信库的TorchProcessGroupExecutor，用于GPU环境，里面包含了某些cuda stream同步操作。
5. Plan
	- 一个带版本区分的全局任务序列，AdmissionScheduler会严格按照Plan中指定的顺序依次launch通信任务。
	- 设计它的主要原因有两个，首先是为了测试使用，在接入真实的megatron训练框架之前，需要有某种按照一定顺序提交通信任务的机制来生成测试序列，因此可以通过构造一个Plan对象来指定这样的通信任务序列。其次，在某些调度场景中，这个全局任务序列可以作为调度策略的一部分。例如对于multi-job多并行链调度场景，可以预先计算出通信任务的调度顺序，然后提供一个Plan对象来强制按照顺序执行。

详细介绍可以阅读[[项目仓库 Current State]]

更多信息参见仓库里的`README`和`docs/`下面的文档。关于如何使用这些类来进行端到端的通信任务执行，可以去`examples/`下面找那些`run*`开头的脚本，参考一下它们的逻辑。记得善用ai进行代码分析、环境配置和任务执行。

## 5. 接下来的任务

1. 在当前初版仓库的基础上，进行**多job并行链调度**的实验；参见[[JobPacer实验计划]]
2. 继续开发，研究当前调度机制如何适配到megatron框架；参见[[Megatron框架适配计划]]

## 6.注意事项

- 两个任务分别开新的git分支来实施，但是一个任务并不局限在一个分支上，可根据需求多开分支；
- 开发过程中留足文档记录，包括整体design、开发阶段规划、每一步骤实施和测试记录，以便后续进行代码审查或者故障回溯；
- 开发过程主要依赖agent帮忙，但是建议每完成一个小步骤之后进行一次人工介入，大致浏览一下新增的改动、项目结构等，关于代码的有任何不懂的地方找agent讨论确保自己理解，看看是否符合自己的预期，确保自己有对开发进程有全局把控；
- 设计和规划阶段要用顶尖模型，如gpt-sol，astra等，上层建筑没弄好后面实施阶段会比较麻烦；
- 关于Megatron / pytorch / python多线程可能不熟悉，可以在开发过程中边遇到边学，善问ai，善查文档。