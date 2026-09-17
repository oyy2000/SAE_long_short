# Phase 13：统一数学教师候选接口

2026-09-13 更新：本文保留首次开发配置的边界和执行记录。学生题池已在独立的 [B0/B1/B2 源生成协议](phase13_raw_student_generation_v1.md) 中启用；新协议不修改这里的开发产物。统一 DAP/TokenSkip 开发结果见 [文本压缩说明](phase13_unified_text_compression_dev_v1.md)，实际作业状态见 `PROJECT_STATUS.md`。

本接口承接已冻结的 MATH 角色和主 Qwen-7B ASC 校准，服务后续 B0–B7 学生比较。当前配置只允许 smoke 与 development 生成；6,723 道学生候选题的输入已保存，但尚未启用生成或学生 SFT。开发集产物不得进入学生训练。

## 输入与实现

配置为 `configs/phase13_unified_math_candidates_dev_v1.json`，入口为 `scripts/13_29_generate_math_candidates.py`，提交复用 `scripts/13_23_submit_sae_readback.py` 的注册入口和 NCSU 路由。源码、模型文件哈希和输入冻结于 `results/phase13_baseline_expansion_v1/preparation/unified_math_candidates_dev_v1`。

直接复用 `sae_norm_intervention` 的逐题随机流、cache 生命周期和 nucleus sampler，提取 `generate_condition_raw` 将实际 token 解码与评分分开。旧 `generate_condition` 保留原数值评分包装和默认采样设置，已经启动的历史作业使用自己的不可变快照。

数学候选使用完整题目上下文和已审计的 typed grader。新增 `FixedVectorController` 只执行一次已拟合方向的解码位置加法，不运行 SAE 编码器；绝对模式与 ASC 的 BF16 加法一致，相对模式与既有相对隐藏状态范数公式一致。当前 B1/B2 不注入向量；B3/B4/B7 后续接入时必须额外绑定方向文件、完成标记、层位及强度。

## 当前生成设置

主教师是固定 revision 的 Qwen2.5-7B-Instruct。64 道开发题与 MATH ASC 使用相同题号和顺序，另以固定 hash 从中取 8 道作接口 smoke；两者均不属于学生池。

| 设置 | 数值或规则 |
| --- | --- |
| 每题每生成条件候选 | 4 |
| 生成条件 | B1 未干预；B2 简短提示 |
| 最大新 token | 4,096 |
| temperature / top-p / top-k | 0.7 / 0.95 / 20 |
| repetition penalty | 1.0 |
| 批次大小 | 8 |
| development 分片 | 2，题号按固定顺序取模 |
| 随机流 | 基础 seed 2026091315，每候选增加 100003，再按题号派生 |
| 终止 | tokenizer EOS；不在答案标题处提前停止 |

Qwen 模型默认 `top_k=20`，旧 SAE sampler 默认为不做 top-k。这里将 top-k 显式纳入协议，在 nucleus 前过滤并保留阈值并列项。所有统一生成条件使用同一采样实现；它与上游 Transformers multinomial 的随机抽样实现不同，不能声称逐 token 复现上游随机输出。模型参数、采样分布配置与实际随机算法分别登记。

B1 提示使用 step-by-step 和 boxed 答案。B2 仅增加 `as concisely as possible`，保留相同问题和最终答案格式。各条件从原始 prompt 独立开始；同题同候选使用相同均匀随机数。8 题 smoke 对两个条件的候选 0 额外重放，要求实际 token 完全一致。

## B0/B1 选择、覆盖与成本

B0 与 B1 共用 B1 的四候选池。先保留正确且未截断的输出，再按候选索引保留最早出现的相同去空白文本，复用 `ncsu_multi_answer.unique_correct`。B1 按实际采样 token 数、候选索引选择最短；B0 以固定逐题随机种子在同一去重正确池中均匀选择。B2 使用相同最短正确选择规则。

保存各方法完整问题池覆盖、零正确支持题号、唯一正确候选数量分布、各自支持上的选择，以及目前可用方法的共同交集。当前 B0/B1/B2 交集不称作八方法共同支持，不能提前据此固定正式训练题目数。

每个批次的时间独立保存，合并时只计一次。B0 和 B1 不重复计算共享生成成本；额外 smoke 重放单独登记。新接口没有机制实验中的 SAE 编码读回及在线标题识别开销，但尚未完成 DAP 改写、TokenSkip 压缩、方向拟合和学生训练的完整成本汇总。

## 执行记录与剩余工作

CPU `821887` 的 13 项检查通过，覆盖历史缓存/窗口行为、真实小模型贪心批次结果、随机流行顺序不变、top-k、绝对向量与 ASC 一致性、相对范数、重复/缺失候选、正确选择与完整池覆盖。准备 `821888` 完成，10 项输入/父绑定和 418 项源码绑定核验通过。

H200 smoke `821936` 和 CPU 选择审计 `821937` 已完成，64 条输出、两次逐 token 重放检查和 15 项直接绑定均核验通过。GPU peak allocated 为 16,094.75 MiB，生成批次合计 179.35 秒，额外重放 66.85 秒。B0/B1/B2 的完整与共同正确支持均为 5/8；没有为了增加覆盖而重新抽取更容易的 smoke 题。两个 development 作业 `821938/821939` 已在 smoke 审计后运行，预期共 64×4×2=512 条；合并审计 `821940` 依赖两个分片。这些状态需以 Slurm 和完成标记为准。

后续需要基于这些实际新教师输出接入 DAP/TokenSkip，完成 ASC/SAE/dense 在数学开发题上的控制参数核验，再冻结正式候选和共同支持 SFT。原设置方法检查、这次开发集选择文件和已排队作业都不代表统一学生实验完成。
