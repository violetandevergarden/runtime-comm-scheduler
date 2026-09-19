## 1. 背景与目标

当前仓库已经可以安全地构造、延迟并提交真实的 PyTorch collective，适合先在受控
workload 上研究 admission policy。本阶段计划采用“从 Megatron 观察真实 workload，再在本仓库重放”的路线：
```text
Megatron profile / instrumentation
        -> 语义化 trace
        -> trace normalization + validation
        -> multi-tenant replay workload
        -> scheduler policy evaluation
```

目标是在保留Megatron计算-通信结构、collective 大小、ready/consumer 依赖和 communicator 顺序的前提下，建立可重复、可控、可比较的 multi-job 通信调度实验平台。主要研究以下问题：
- 多个训练 workload 同时运行时，通信请求在何种资源上发生竞争，竞争如何影响每个 tenant 的 step time、throughput、tail latency 和计算通信重叠？
- 对给定 workload，FIFO baseline 以及之前探索的各种heuristic调度顺序的效果如何？收益主要来源于何处？
- scheduler 可以安全控制的“host submission order / admission time”机制，在当前任务场景下是否足够完善？这决定了后续scheduler是否需要往更底层走。

## 2. 可能的阶段划分

**Phase1：手工构造multi-job workload进行replay**
- 将每个job简单视作通信-计算交替的序列，这里的计算既可以用sleep来代替，也可以插入真实的tensor计算语句，而通信则是由torch.distributed发起的真实集合通信操作，并且共享PCIe等资源。手工构造若干条这样的序列进行replay。
- 单进程多线程replay：可以将每个job放在一个replay进程的多个线程中执行它的通信-计算序列。这比较方便因为不同线程的job可以共享同一个scheduler对象。
- replay阶段可以不涉及scheduler，直接用torch.distributed接口裸发通信请求就行。
- 验收结果：本阶段作为baseline，展现完全不协调job之间流量时，可能产生的冲突与竞争。除了每个job的makespan以外，还可以观察每个job的具体执行trace，看看通信任务的发出和结束时间点，分析可能的竞争行为。

**Phase2： 接入scheduler进行调度**
- 将上一阶段中，各个job线程发送的通信请求，改用scheduler.submit()接口
- 尝试通过规定Plan中通信任务顺序的方式，实现FIFO、longest-tail-first等多种调度策略，使用非抢占式调度，保证上一个通信任务完成后才发送下一个通信任务
- 验收结果：与上面的baseline进行比较，看看避免一条链路上同时发出多个通信任务的情况下，是否会比baseline随意发送要好，充分研究好与坏结果出现的原因

**Phase3-1：runtime动态性调度**（可选）
- 之前的实验中，通过预先profiling确定每个job的通信-计算交替的窗口长度，但实际上无论是计算还是通信的执行时间都可能受一定的扰动，因此通过Plan规定的静态顺序不一定好
- 本阶段可以考虑在代表计算窗口的sleep时长上增加一点随机性，然后考虑在这种模型下，如何在scheduler中进行runtime决策，决定下一步是应该释放哪个ready task，又或者等待某个更重要的但是还没提交的task
- 可能需要抛弃原本的Plan设计，在不违反每个ProcessGroup内所有rank提交collective顺序一致的要求下，如何设计调度机制以支持上述runtime决策
- 验收结果：确定一种runtime调度机制，以及探索各种调度算法与baseline的效果比较

**Phase 3-2：DAG模型调度**（可选）
- 每个job只考虑线性计算-通信序列并不足以反映真实情况。本阶段要求重新按照一般的DAG模型分析调度策略的设计
- scheduler需要决策，下一步应当选择哪个job的ready task执行，才最有利于全局的推进
- 注：初期可以手动构造DAG，在每个job线程上实现一个简单的DAG执行器，每个任务需要满足所有前置依赖才允许submit给scheduler。在phase4-6完成后才能考虑更符合实际情况的多维度并行下真实的DAG结构。

**Phase4: Megatron trace capture & replay**
- Trace capture 的职责是观察真实训练，可结合框架级插桩和PyTorch profiler/NVTX，产出每 rank 的结构化记录。记录至少应使 replay 能知道“哪个job 的哪个计算阶段产生了哪个 collective、何时可发射、何时必须消费、它属于哪个 communicator，以及传输了多少数据”
- 初期阶段只需记录每个通信、计算操作的时间窗口，用于重构上述phase所需的通信-计算交错序列。扩展阶段可以考虑同时记录任务之间的依赖关系，用于下一phase的DAG模型调度。
- 验收结果：给定Megatron model和并行配置，能够生成该job的通信-计算时序；此外，使用这些新的workload重新执行phase1-2观察效果。

**Phase 5：多进程模型下的控制面与执行面分离**
- 在真实框架中，每个training job并不可能都放在相同的卡上执行。相应的scheduler不可能与所有job在同一个进程中，甚至不一定在同一个host上。在这种情况下，必须考虑将scheduler设计成每host上单独的进程。为了能让schedule层获取全局的pending task信息以便进行调度，需要设计一种新的schedule模式，需要考虑跨进程/host之间的信息交换；
- 参考SDN设计，可以采取控制面-执行层分离的架构设计，控制面负责收集全局信息然后计算调度决策，执行层负责收集局部的submit tasks送往控制面，以及执行控制面的决策；
- 分布式 & 中心化设计取舍：控制面可以考虑中心化设计，放在一个节点上，以避免复杂的consensus一致性保证

**Phase6：多链路竞争**
- phase1-3都局限在单链路场景中，一个真实可用的系统（尤其是phase5考虑不同job放在不同rank之后）必须考虑连接多rank的多链路模型。某些job使用的rank和链路之间可能完全没有交集，scheduler完全可以同时launch这些job的通信任务而不引发任何链路竞争
- 需要重新思考调度策略的设计，例如greedy选取占据链路不交的最大通信任务集合
- 注：依赖phase5，此外还依赖有多机多卡的实验环境

## 3. 一些前期结果

实验设计：两张GPU通信场景、两个allreduce任务A、B同时发出、依次发出所消耗的时间，与每个任务单独所需时间的比较。每组实验进行50轮取中位数展示。

实验环境：

| 项                  | 值                                      |
| ------------------ | -------------------------------------- |
| GPU                | 2× NVIDIA GeForce RTX 3080 Ti，12 GiB   |
| PyTorch            | 2.12.1+cu130                           |
| CUDA runtime       | 13.0                                   |
| NCCL               | 2.29.7                                 |
| GPU topology       | `NODE`，无 NVLink                        |
| GPU P2P read/write | `CNS`（chipset not supported）           |
| NCCL transport     | `SHM/direct/direct`                    |
| NCCL topology path | `PHB`，模型带宽约 12 GB/s，理论带宽上限15.75GB/s    |
| NCCL channels      | 每 communicator 2 个 collective channels |

结果：
下表时间均为两 rank 全局 makespan 的中位数：

| 每个 all-reduce |    A 单独 |    B 单独 | 单独耗时和 |  强制顺序 |      并发 | 并发/单独和（P10–P90） | 并发/顺序 |
| --------------: | --------: | --------: | ---------: | --------: | --------: | ----------------------: | --------: |
|           1 MiB |  0.259 ms |  0.260 ms |   0.521 ms |  0.442 ms |  0.391 ms |   0.748（0.716–0.796） |     0.881 |
|          16 MiB |  2.508 ms |  2.519 ms |   5.027 ms |  4.945 ms |  4.349 ms |   0.865（0.855–0.876） |     0.880 |
|          64 MiB |  9.705 ms |  9.703 ms |  19.405 ms | 19.364 ms | 16.705 ms |   0.861（0.854–0.869） |     0.863 |
|         256 MiB | 38.335 ms | 38.323 ms |  76.682 ms | 76.444 ms | 65.453 ms |   0.854（0.839–0.866） |     0.855 |

结论：两个任务同时执行总时间 < 顺序执行总时间 $\approx$ 单个任务执行时间之和，这是因为单个任务很多时候没法打满全部带宽，因此之前的单链路建模与实际模型具有一定差异。这一结果表明”通信任务在同一链路上的竞争会导致性能恶化“并不一定成立。