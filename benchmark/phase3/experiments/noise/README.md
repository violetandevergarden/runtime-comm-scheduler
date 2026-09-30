# Noise 对照

`zero-jitter-ab.json` 冻结 2026-09-23 的固定输入 A/B 噪声 pilot：L0、零计算扰动、单个 `new-fifo` arm，
10 个 seed、每 seed 3 次重复，两份拷贝按 seed 奇偶交替先后。先对每份拷贝的重复取中位数，再计算同 seed 的相对差。

它估计固定配置下的系统噪声，不是扰动抽样，也不是显著性检验。该配置是可追溯记录，当前 compact suite runner
不直接执行此 A/B 文件；执行能力由原始批次日志和 suite 编排保留。
