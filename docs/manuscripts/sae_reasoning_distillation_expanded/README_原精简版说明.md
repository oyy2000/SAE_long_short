# SAE reasoning distillation：精简修改版

入口文件：`sae_reasoning_distillation_main.tex`。原有文件名保留，新增 `sae_reasoning_distillation_appendix.tex`；原始上传文件没有被覆盖。

## 篇幅变化

以下为 TeXcount 的 text-word 统计，不计标题、图表标题、表格内容、公式和参考文献。它是源文件正文的统计口径，不等同于 PDF 中全部可见英文词数。

| 章节 | 原版 | 修改版主文 |
| --- | ---: | ---: |
| Abstract | 86 | 106 |
| Introduction | 708 | 229 |
| Related Work | 462 | 169 |
| Experiments | 1,447 | 456 |
| 主文合计（含摘要） | 2,703 | 960 |

主文减少 **64.5%**。附录另有 601 词；连同附录，修改版共 1,561 词，比原版减少 **42.2%**。摘要略有增加，因为原稿含未完成的占位结果句，修改版补成了完整摘要。

## 结构与内容调整

引言收束为研究动机、现有方法与问题、SAE 干预思路、评估设计，去掉重复的结果复述。相关工作由四个长段压成三个短段，保留与本文定位最相关的区别。

实验主文保留自然长度比较、teacher 干预和 student 蒸馏的主要结果，以及理解结论必须知道的控制条件。完整采样与 SAE 网格、筛选规则、LoRA 超参数、训练预算账目、未完成的多答案扩展移入附录。附录也经过压缩，并非原文直接搬运。

原稿的全部 13 个引用键继续使用，`.bib` 文件逐字保留。本次仅编辑用户提供的稿件，没有另行核查外部论文的最新版本或元数据。

## 与证据一致的修改

标题从 `Shorter Traces, Better Students: Sparse Autoencoder Steering for Reasoning Distillation` 改为 `SAE-Guided Trace Shortening for Reasoning Distillation`，避免在标题中预先宣称学生表现提升。

摘要的占位结果句已替换为原实验中的 teacher/student 长度结果和准确率不确定性。没有补造新实验，也没有将不显著的准确率差异改写成显著收益。

区分“框架筛选过的联合干预”与“最终选中的短特征增强”。原稿中的两次 base 准确率不一致、自然短样本的选择规则差异、近似 token 匹配不等于 step 匹配、单 SAE seed 和开发集复用等限制均保留。

原稿引用的 `student_comparison.pdf` 没有随此次上传提供。修改版以正文已报告的四组、两种预算的 seed-mean 准确率生成表格，不重建未提供的逐 seed 点、误差条或原图样式；不再依赖原稿的外部图路径。

原上传材料没有独立的 Method 文件，本次没有增写未经提供的实现细节。

## 编译与使用

上传整套源文件后，将 `sae_reasoning_distillation_main.tex` 设为主文件即可。已在本文所用本地模板中编译，并检查引用、表格和页面排版。

```bash
latexmk -pdf sae_reasoning_distillation_main.tex
```

也可依次执行：

```bash
pdflatex sae_reasoning_distillation_main.tex
bibtex sae_reasoning_distillation_main
pdflatex sae_reasoning_distillation_main.tex
pdflatex sae_reasoning_distillation_main.tex
```

压缩包中的 PDF 是本地 `article` 模板预览：3 页主文、1 页参考文献、2 页附录。正式投稿时仍应使用对应会议的官方模板，页数会随模板变化。合并至已有工程时，保留 `booktabs` 支持，并将新增附录接在该工程的 `\appendix` 后。
