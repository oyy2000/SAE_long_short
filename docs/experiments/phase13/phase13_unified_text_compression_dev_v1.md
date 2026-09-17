# 统一主教师：DAP 与 TokenSkip 开发集适配

本阶段使用已冻结的 Qwen2.5-7B-Instruct 数学候选，完成真实文本压缩和选择审计。它是统一 B5/B6 数据管线的开发验证，尚未训练学生，也不替代已完成的 TokenSkip 原设置 SFT。

## 来源与方法

64 道 MATH development 题与 ASC 开发队列相同。B1、B2 各生成四个候选，共 512 条；B0 从 B1 的去重正确池中随机选择，B1 选择最短正确候选。B1、B2 分别有 56 道题可用，共同支持为 55 道；所有零支持题保留在覆盖率分母中。

DAP 改写 B1 中全部 194 条唯一、正确、未截断的候选，保留原问题、完整源解答和难度感知 system/user 消息。每条源轨迹改写一次，完整 Thought/Solution 输出保留，再用冻结 typed grader 验证，最后按题选择最短正确改写。此处改写模型为统一 Qwen-7B，区别于原论文完整 DeepSeek-R1 设置。

TokenSkip 使用 56 道题的 B1 最短正确轨迹及固定 revision 的 LLMLingua-2。开发检查覆盖 0.5/0.6/0.7/0.8/0.9/1.0 六比例，共 336 条；每题只有一个预分配比例可用于后续同类训练协议。比例在完整 64 题池上按哈希平衡分配，正确过滤后不重新分配，因此过滤后的六档数量是 10/11/9/10/9/7。保留作者比例条件 prompt/completion，末尾答案来自已验证的源预测，不从 gold 偷补；附回答案的正确性不验证压缩推理的语义。

## 完成证据

17 项接口、采样、选择、来源及比例检查 `822078` 通过，输入准备 `822080` 完成。真实 DAP/TokenSkip smoke `822110/822111` 及合并 `822112` 完成；四个完整开发分片 `822113`–`822116` 和合并 `822117` 完成。合并核对源身份、实际 token/EOS、逐条 typed grading、比例条件、完整分片键和批次成本；主合并的 10 项直接绑定已复核。

| 测量 | 结果 |
| --- | ---: |
| DAP 全部配对源平均 tokens | 547.40 |
| DAP 全部改写平均 tokens | 479.12 |
| DAP 改写最终答案正确 / cap hit | 194/194；0 |
| B1 最短正确监督平均 tokens（56 题） | 526.02 |
| B5 最短正确监督平均 tokens（同 56 题） | 442.13 |
| B5 / B6 可用题数 | 56 / 56 |
| 原始 B1 候选生成批次总时间 | 686.75 秒 |
| 额外 DAP 改写批次总时间 | 471.84 秒 |
| TokenSkip 六比例开发压缩总时间 | 8.54 秒 |

194 条轨迹不是 194 道独立题。上表长度为本开发支持的描述统计；未据此声称学生准确率或统计非劣。B1 原始生成成本只计一次并由 B0/B1/B5/B6 共享；额外 smoke、重放、准备、验证和模型加载须在完整成本报告中另计。TokenSkip 的六档开发扫描成本也不能冒充只生成训练所需单档的成本。

CPU 合并 `822117` 在高负载节点启动缓慢，但重查时已成功完成，因此没有取消或重跑它。

## 实现与后续

- 配置：[phase13_unified_text_compression_dev_v1.json](../../../configs/phase13_unified_text_compression_dev_v1.json)。
- 共用逻辑：[unified_text_compression.py](../../../src/length_budget_distill/unified_text_compression.py)；入口：[13_31_compress_math_candidates.py](../../../scripts/13_31_compress_math_candidates.py)。
- 完整产物：[merged/development](../../../results/phase13_baseline_expansion_v1/preparation/unified_text_compression_dev_v1/merged/development)。

下一阶段将与 B3 dense、B4 ASC-CES、B7 SAE 的统一采样开发曲线一起冻结运行强度，再生成独立的 6,723 题学生候选池、审计八方法共同支持并执行学生 SFT。当前所有压缩样本仍属于 development，不能进入学生训练。
