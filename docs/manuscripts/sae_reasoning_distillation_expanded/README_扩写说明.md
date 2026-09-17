# SAE reasoning distillation：baseline 与 SAE 分析扩写版

本版基于用户提供的 `sae_reasoning_distillation_concise.zip`，围绕讨论的 baseline 和 SAE 分析扩写。原压缩包保持不变；来源校验值记录在 `source_archive.json`。

## 使用入口

将本目录所有文件上传至同一 Overleaf 工程，设置 `sae_reasoning_distillation_main.tex` 为主文件。也可在本目录执行：

```bash
latexmk -pdf sae_reasoning_distillation_main.tex
```

或依次执行 pdflatex、bibtex、pdflatex、pdflatex。所有输入均在包内，不依赖项目外部图表或实验目录。主文件新增 amsmath 和 tabularx 包以支持公式和 baseline 表格。

## 本次扩写

- **Introduction**：保留精简结构，增加简单长度控制与答案格式替代解释两个研究问题，引出后续比较和分析。
- **Related Work**：扩写同池轨迹选择、文本压缩、激活控制、SAE 语义解释与行为迁移；区分普通均值差与正式版 ASC、普通改写与完整 DAP。
- **Experiments**：保留已有自然长度、教师生成和学生蒸馏结果。新增 `Baseline Comparison Design (Proposed)`，列出 random-correct、concise prompt、dense direction、answer-format direction、多个随机 SAE 集合、ASC 和文本压缩；说明候选数、共同支持、生成 token、训练预算、开发集校准与长度匹配的比较原则。
- **SAE Analysis**：新增独立正文文件 `sae_reasoning_distillation_sae_analysis.tex`。从现有八个 short 特征的 Answer 激活上下文出发，讨论平均激活的长度分母效应，并设计区域/词汇/位置控制、同状态目标读回、非目标扰动、特征冗余、数量消融、干预时机、跨 seed 稳定性及学生学习分析。
- **Appendix**：保留原实验细节，新增六 SAE 的真实留出指标表，进一步明确前缀预测的信息限制、格式反例、子空间分解、读回度量和跨 seed 匹配方法。

## 已有证据和新增设计的边界

摘要、标题、参考文献文件均保留原精简版。教师/学生结果和已有比较的统计结论没有改写成正面收益。

本次引入的已有诊断只有两类：原特征报告中的 Answer 相关高激活上下文，以及六 SAE 审计中的解释方差和测试支持上未激活比例。来源路径与哈希列于 `evidence_sources.json`；这些元数据用于溯源，不是编译依赖。

所有新增 baseline、语义反证、读回、数量/时机消融、跨 SAE seed 和长度匹配 SFT 都写为 proposed。没有为它们填写虚构指标、显著性、图表或完成状态，也没有启动新模型实验。多答案结果沿用原精简稿的证据截止状态，本次没有重新审计其运行进度。

特别明确：当前 no-steering 蒸馏已使用 shortest-correct，不能称 random-correct；历史 natural-short 不是同池选择对照；同范数不是同 KL；测试集未激活不是训练中死亡；三个学生 seed 不是三个 SAE seed；事后长度匹配不能单独证明内容中介机制。

## 文件和版本

- `sae_reasoning_distillation_main.tex`：编译入口。
- `sae_reasoning_distillation_*.tex`：各正文与附录模块。
- `sae_reasoning_distillation_references.bib`：原有 13 条参考文献，保留引用键。
- `sae_reasoning_distillation_main.pdf`：扩写版编译预览，以本版构建记录为准。
- `README_原精简版说明.md`：原包说明的历史存档，其中篇幅、页数和编译描述仅指原精简版。
- `source_archive.json`、`evidence_sources.json`、`build_verification.json`：来源、诊断证据与本版验证记录。

本稿仍为单 teacher/student、单 SAE seed、GSM8K 的探索性研究稿；新增设计需实施并审计后才能改写为结果章节。未新增独立 Method 章节，正式成稿仍需结合最终实现补齐。

## 本版构建验证

已使用 Tectonic 0.17.0 编译更新后的 PDF，共13页：主文8页、参考文献1页、附录4页。引用、标签、环境配对和包内文件关联检查通过；最终日志无缺失引用、缺失字符或 overfull/underfull box。XeTeX 忽略 inputenc 的提示是引擎兼容性提示，不影响输出。

已查看PDF首页、baseline表、SAE分析公式、训练预算及六SAE质量表所在页面。原精简版摘要与参考文献逐字保持；原实验与附录内容作为新增章节前的完整前缀保留。压缩包不包含编译缓存，解压后可独立编译。
