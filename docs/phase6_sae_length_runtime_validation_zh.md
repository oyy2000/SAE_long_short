# C31 新环境：特征 24086 长度控制验证

## 当前结果

2026-09-07 在 C31 的现有 allocation 279213 内实际执行了预检。使用用户指定的 `activate_runtime.sh` 和该目录下 `envs/sft/bin/python`。前者只设置缓存与运行变量，单独 source 不会切换 Python。

新环境可导入 PyTorch 2.10.0+cu128、Transformers 4.48.3，CUDA 可用，可见四张 GPU。三项现有测试通过：KV 缓存复制互不修改；缓存分支与重新生成的贪心输出一致，首个续写 logit 的 hook 正确触发；合成 SAE 上目标抑制、扰动范数上限及拒绝错误 prefill 符合预期。这些是兼容性单元测试，不能证明真实特征可以控制输出长度。

**真实教师生成尚未运行。** BeeGFS 当前未挂载，原 SAE 检查点、Phase 5 特征选择证据和原 corpus 路径不可读。C31 新环境目前缓存的是 Qwen2.5-1.5B-Instruct，而该特征来自 Qwen2.5-7B-Instruct 的第 17 层，不能替代使用。

预检证据位于：

- [机器可读报告](../results/phase6_sae_length_runtime_preflight_v1/exploratory/attempt_20260907_1220/preflight.json)
- [测试日志](../results/phase6_sae_length_runtime_preflight_v1/exploratory/attempt_20260907_1220/compatibility_tests.log)
- [GPU 快照](../results/phase6_sae_length_runtime_preflight_v1/exploratory/attempt_20260907_1220/nvidia_smi.txt)

报告记录时间、节点、allocation、环境版本、配置与源码哈希、逐项输入路径及缺失原因。不产生实验完成标记。

## 必须恢复的输入

1. `checkpoints/phase2_sae_sampling_ablation_v1/full_trace_balanced/sae_model.safetensors`。原训练标记记录 SHA256：`f26ccf37289e025ab53199b6a5d09c59e1145d1028c5d66159f316b475cb356c`。特征编号必须与这份字典绑定，重新训练的字典不能沿用 24086 的含义。
2. Phase 5 的 `feature_gate/selected_features.json`、manifest 与完成标记，用于重新验证选择来源。目前可读的项目说明记录主特征为 24086，但原始结果证据尚不可访问。
3. 原 `mixed_trajectories.jsonl` 或经验证的完整题目排除名单，用于避免把已分析题目当成独立新题。
4. Qwen2.5-7B-Instruct，固定 revision `a09a35458c702b33eeacc393d103063234e8bc28`。模型可重新下载；不能替代缺失的私有 SAE 检查点。

## 恢复后的验证方案

保留 Phase 5 对照覆盖率未通过的历史结论，在独立目录注册新方案。先验证特征来源、模型维度和数值读回，再冻结生成协议；当前仅完成环境预检，未冻结或实现以下完整生成矩阵。

- 主特征固定为 24086，不依据新的输出重新选特征。先验证双向目标读回、实际 BF16 扰动量和非目标变化。
- 对照设计需要解决旧随机特征自身激活稀疏造成的覆盖率差异。计划用同一无干预参考轨迹记录目标触发位置和幅度，再对目标与预先固定的两个随机方向重放同一时间表。不同分支局部隐藏范数和 EOS 会影响实际可施加扰动，必须记录并审计这种差异，不能把请求幅度相同写成实际幅度完全相同。该方案估计的是参考轨迹时间表下的方向效应，与分支自身动态门控不同。
- 独立抽取 64 道 GSM8K train 新题，排除旧 corpus；每题两个采样重复。无干预、目标正负方向、两个随机方向各正负，共七个条件；起始位置为首个输出 token 与共同前缀 64 tokens 后，总计 1,792 条计划输出。
- 使用同一前缀的独立 KV 缓存和配对采样种子。现有缓存工具只支持正长度前缀；零前缀分支需补充缓存 prompt 最后一个 token 前状态的实现与验证，不能直接套用。
- 显存足够的每张 GPU 一个进程，每次一条序列，独立分片；逐有效 token 读回，避免 EOS 后污染。每次启动前执行共享显存准入检查，不修改或终止现有占用进程。
- 汇总完整输出、答案正确率、有效输出 token 数、EOS/截断比例、实际干预覆盖率与扰动范数。按题配对 bootstrap，随机对照比较作多重校正。准确率损失区间过宽应报告证据不足；不得将这轮小样本直接解释为安全等效。

## 复现预检

配置：[phase6_sae_length_runtime_preflight_v1.json](../configs/phase6_sae_length_runtime_preflight_v1.json)。有可访问备份时，在独立配置中改写路径并保留原检查点哈希。预检只检查可用性，不代替完整生成协议的来源审计。

在有效的 C31 allocation 中运行；将下面的 allocation ID 换成当前有效值，并为每次预检使用新的输出目录：

```bash
srun --jobid=279213 --overlap --nodes=1 --ntasks=1 --cpus-per-task=2 bash -c '
  source /mnt/local/youyang7/SAE_long_short_c31/activate_runtime.sh
  export OMP_NUM_THREADS=2
  cd /home/youyang7/projects/SAE_long_short
  /mnt/local/youyang7/SAE_long_short_c31/envs/sft/bin/python \
    scripts/6_0_check_sae_length_runtime.py \
    --project-root /home/youyang7/projects/SAE_long_short \
    --config configs/phase6_sae_length_runtime_preflight_v1.json \
    --output results/phase6_sae_length_runtime_preflight_v1/exploratory/attempt_new
'
```

退出码 2 表示存在阻塞输入或兼容性失败，详细原因见 `preflight.json`；不应将其解释为已完成长度实验。
