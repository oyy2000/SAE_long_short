# SAE reasoning distillation manuscript draft

本稿按照用户提供的 [研究思路图](<../../image (1).png>) 撰写，保留如下主线：同一教师、同一学生、相同题目且答案正确时，短轨迹的蒸馏表现更好；比较长短轨迹的 SAE 特征，解释长轨迹增加了什么内容，再通过特征干预生成更有用的学生训练数据。

## 文件

- `sae_reasoning_distillation_abstract.tex`：英文摘要，可直接用于主稿。
- `sae_reasoning_distillation_introduction.tex`：英文 introduction，依据当前 abstract 展开，并引用推理蒸馏、轨迹压缩和激活干预相关工作。
- `sae_reasoning_distillation_related_work.tex`：英文相关工作，按蒸馏、压缩、激活控制和特征解释组织。
- `sae_reasoning_distillation_experiments.tex`：英文实验设置、已完成探索性结果及待补比较。
- `sae_reasoning_distillation_references.bib`：BibTeX。
- `sae_reasoning_distillation_main.tex`：匿名、普通 article 预览入口。投稿时换入目标年份的官方 ICLR 模板，保留正文 input 文件。

## 写作取舍与证据

短轨迹优于长轨迹作为初步观察保留，依据用户图示和 [历史配方复现实验说明](../experiments/phase1/phase1_legacy_trace_sae_distillation.md) 中已记录的三种长度组结果。本次写作未重新核验 BeeGFS 上的逐例结果，因此没有在摘要中新增数值或显著性结论。

“长 CoT 对蒸馏冗余”写为解释该观察的研究假设，而不是所有长推理都无效的普遍结论。具体方法围绕同题正确轨迹、共享 SAE 字典、特征解释、双向生成干预和学生 SFT 展开；没有将本项目改写为学生梯度效用预测或最小长度匹配方法。

补充的细节包括词汇及位置控制、不同 SAE 种子的一致性、共享前缀与缓存复制、目标激活读回、随机方向对照，以及等样本和等监督 token 的独立比较。这些表述描述方法和评价设计，不代表它们都已执行或通过。较新的 [Phase 5 记录](../experiments/phase5/phase5_sae_clean_feature_causal_plan_zh.md) 和 [新环境预检](../experiments/phase6/phase6_sae_length_runtime_validation_zh.md) 尚不支持“新特征已稳定缩短输出并提升学生”的结论。

图中的 SFT 每题一条保留为主线；RL 每题十六条保留为后续扩展方向。尚未给出 RL 算法及采样策略，因此没有虚构 GRPO、PPO、on-policy 训练或 RL 性能结果。

完整生成和学生评估结束后，再把摘要末尾的方法性总结替换为实际结果，至少同时报告长度变化、教师正确率、学生准确率和相应对照。当前稿件未声称提升幅度、机制发现已经成立、跨任务泛化或优于已有方法。

## 参考文献

背景中的推理轨迹和蒸馏表述分别引用 [Wei et al.](https://arxiv.org/abs/2201.11903) 与 [Hsieh et al.](https://arxiv.org/abs/2305.02301)；SAE 分解及 TopK 方法引用 [Cunningham et al.](https://arxiv.org/abs/2309.08600) 与 [Gao et al.](https://arxiv.org/abs/2406.04093)；GSM8K 引用 [Cobbe et al.](https://arxiv.org/abs/2110.14168)。均于 2026-09-07 核对原始论文页面。

## 编译

```bash
cd docs/latex
pdflatex -interaction=nonstopmode -halt-on-error sae_reasoning_distillation_main.tex
bibtex sae_reasoning_distillation_main
pdflatex -interaction=nonstopmode -halt-on-error sae_reasoning_distillation_main.tex
pdflatex -interaction=nonstopmode -halt-on-error sae_reasoning_distillation_main.tex
```

当前环境未找到 `pdflatex` 或 `tectonic`；本次仅检查正文引用、文件关联与基础 LaTeX 结构，未生成或验证 PDF。

## 2026-09-13 Introduction 更新

依据当前 abstract 重写引言，围绕冗余监督假设、同题长短正确轨迹、共享 SAE、生成干预和学生 SFT 展开。补入 Small Models Struggle、LiteCoT、Compress-Distill、ASC 和 Steering Vector Distillation 的定位；来源已在本次会话核对。正文未将待补对照描述为已完成实验，未加入未确立的学生准确率收益，研究范围保持 GSM8K。此前提到的跨 SAE seed、共享前缀缓存和 RL 扩展等细节不再写入本版 Introduction。

用户 abstract 原文保留，其中结果句仍含 `xxx` 和省略号，且存在重复的 `\begin{abstract}`；完整稿件编译前需修复该环境配对。

## 2026-09-13 Experiments 与 Related Work 更新

实验章节采用已完成的 `ncsu_phase12_reproduction_v1` 与 `ncsu_sae_intervention_v1`，分别报告自然长度组和后续干预，不混用两轮 base。训练、采样与 SAE 设置来自冻结协议及配置；学生数值与配对区间对照 `student_followup/analysis/summary.json`，完整教师池数字来自已交付状态报告。本次读回验证总完成标记与学生完成标记直接绑定的 33 项文件哈希，没有重新运行完整实验审计。

复用 `student_followup/analysis/student_comparison.pdf`，保留已有生成代码和配色，不新增平行绘图实现。结果图路径相对于 `docs/latex`，请在该目录编译。教师完整候选池使用紧凑表格，因为需要精确比较正确率、平均长度和触及上限数量。

明确最终入选为短特征增强，联合方向扫描不等于已证明双向干预有效；等 token 的轨迹重复不匹配步数；自然 short 的候选池和选样规则不同。ASC、提示、稠密方向、文本压缩、长度匹配均列为尚未完成比较。多答案扩展仅写登记设计，未使用部分评估作为最终结果。全部结果保持探索性 GSM8K、一个 SAE seed 的边界。

Related Work 新增 SAE 推理干预、策略控制和特征反证引用；主文件已接入两节并加载 graphicx。仅删除 abstract 重复的环境起始标记，摘要正文及结果占位符保留。本文仍缺完整 Method 章节，标题中的 Better Students 及摘要结果句也需与最终证据统一，不是可直接投稿的终稿。
