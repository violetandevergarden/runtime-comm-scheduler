# Isolated 分母

这里的 job ID、通信签名、计算段和 seed 与 L0 一致。对其他线性场景计算
slowdown 时，应从对应场景拆分生成单 job 文件，不得混用这两个 L0 分母。
分母必须使用与 shared run 相同的 backend、runner group、policy、profile、seed/repeat
和 warmup 参数。
