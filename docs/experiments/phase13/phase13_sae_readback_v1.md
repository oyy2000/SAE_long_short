# Phase 13：SAE 新题区域与同状态读回协议

本阶段落实实验计划 S1b 的区域反证、同状态特征读回、dense/答案格式/三个随机 SAE 对照。入口为 `scripts/13_22_measure_sae_readback.py`，配置为 `configs/phase13_sae_readback_v1.json`，调度入口为 `scripts/13_23_submit_sae_readback.py`。源码和可执行配置在 GPU 作业开始前复制到独立快照。历史 SAE checkpoint、八个 short features 和旧实验产物保持不变。

## 数据与隔离

- 复用 `sae_mechanism_inputs_v1` 冻结的新题：8 smoke、400 discovery、300 dev、300 confirmation，已排除可追溯历史 SAE 问题及测试近似题。
- discovery 提取官方参考、Answer→Result、移除答案标题、Entry 注释、Answer 注释、已核验的单处算术结果变更，以及未干预模型输出。算术变体仅在准备阶段可构造时存在，不代表验证了整条解答的错误性。
- dense 和答案格式方向仅从 400 discovery 题拟合。dev 用全部 300 题比较登记的方向及剂量，不根据正确率筛题。
- 本入口拒绝 confirmation 特征提取或读回。后续生成剂量、特征数、时间窗口和统计协议需另行冻结，才能进入独立 confirmation。

## 计算契约

教师为固定 revision 的 Qwen2.5-7B-Instruct。复用原 SAE 的第 17 个零起始索引 transformer block、TopK-64、activation mean/scale 和 BF16。`MeasuredSAEController.sparse_codes` 将原 `feature_values` 内部的 TopK→ReLU 运算提取为共享方法，保留实际 TopK 成员关系；旧冻结快照不修改。

SAE 方向遵循 NCSU 原干预代码：decoder 先转 BF16，再以 FP32 求选中特征向量的和并归一化。设置 δ=ρ‖h‖d，然后将 h+δ 转回 BF16；报告实际舍入后的相对扰动范数。

对每个原始生成前缀，先缓存除最后一个 token 外的全部前缀。每个方向/剂量使用独立复制的 cache，重新运行最后一个 token。钩子只接受一个 token，并逐分支验证输入 h 完全相同。真实后续 transformer 层和 lm_head 给出下一 token 的全词表概率；这里不使用线性 logit 近似。

零剂量是每个位置的基准。记录 Enc(h)、Enc(h+δ)、八个 short 特征激活总量、非目标激活 L1 变化、正激活 TopK 成员进入/退出/Jaccard、full-vocabulary forward KL、Answer token 集和模型 EOS 集的概率。非目标包括字典中除八个 short 特征以外的全部正激活编码。FP32 KL 在零附近可能有约 1e-7 的负舍入误差，保留原值并检查容差。

## 固定方向与位置

dev 比较六个方向，各用 ρ∈{0.05,0.1,0.2,0.3}，另有零剂量基准：

1. 历史八特征 SAE 方向。
2. 三个在输入准备时冻结的随机八特征 SAE 方向，分别保留结果。
3. dense：逐题官方参考响应平均状态减原始生成响应平均状态，再对题等权平均、归一化。这仍可能混入内容、风格和长度差异。
4. answer-format：同一参考正文后 Answer 与 Result 标题末 token 状态之差，对题等权平均、归一化。

固定前缀位置为预测首个响应 token 的 prompt 末位、正文中点、标题前一位、标题末位和响应末位。没有显式标题的输出保留，标题相关位置记录为不可用，不能用其他位置补齐。位置名称指已处理的 token，logit 始终预测其后一个 token。

官方参考保持冻结 token/span 对齐；未干预生成在输入阶段没有保存采样 token IDs，因此本阶段对完整解码文本重新分词，明确记录这一限制。分区为 reasoning body、answer marker、answer suffix，跨界 token 归入较后区域。注释中的字面 Answer 另行统计。

## 已执行和验收

- CPU `821312`：11 项数值、BF16、TopK、缓存独立性及位置检查通过。
- CPU 准备 `821317`：配置及 389 项源码文件绑定已核验。
- L40S `821322`：8 题、56 个参考/生成变体提取完成，峰值 allocated GPU memory 14,853.32 MiB。
- L40S `821323`：680 条同状态读回完成，零剂量保持不变，峰值 14,882.29 MiB。其排队依赖曾误录为 `821318`，已在运行前通过 `scontrol` 修正为真实提取作业 `821322`；原记录和修正记录均保留。
- 四个 discovery 提取作业 `821326`–`821329` 以及方向拟合 `821330` 已完成。
- dev 作业 `821335`–`821338` 全部完成，合计 37,500 条测量。CPU 分析测试 `821342` 的 2 项检查通过，完整分析 `821343` 已完成；共 2,794 条 discovery 变体，错误算术变体缺失 6 题。分析/方向/八个分片的 71 项直接绑定已核验，图已查看。[完整报告](../../../results/phase13_baseline_expansion_v1/exploratory/sae_readback_analysis_v1/report_zh.md)。

分析入口 `scripts/13_24_analyze_sae_readback.py` 使用独立源码快照，复核全部变体和方向×剂量×位置记录、原始响应身份、相同前缀及零剂量，并生成区域图、完整局部效应图、逐题指标和完成标记。discovery 的四组格式/算术对比、两个激活指标为一个八项 Holm 校正家族；dev 局部图为描述性均值。前缀局部效应不等于整条生成的长度/准确率效应。

后续仍需完整生成的剂量检查、top1/2/4/8、单特征及 leave-one-out、早期/延迟/答案后窗口，并在冻结策略后做独立 confirmation。当前读回完成也不能替代统一教师 B0–B7 学生 SFT。
