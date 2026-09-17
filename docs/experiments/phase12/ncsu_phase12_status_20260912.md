# NCSU Phase 1/2 实验状态

更新时间：2026-09-12（America/New_York）  
实验：`ncsu_phase12_reproduction_v1`

## 当前状态

**新教师轨迹生成、short/medium/long 学生训练与评估、六组 SAE 训练及特征分析均已完成。本轮未复现 short SFT 相对原始学生的准确率提升。**

最终作业 `811975` 于 2026-09-12 03:06:37 完成，Slurm 状态为 `COMPLETED`，退出码 `0:0`。本次查询时，当前 NCSU 用户队列中没有运行或排队作业。

本报告仅覆盖当前工作区可核验的 NCSU 实验，不将历史 C31 Phase 7/8 分支的记录作为本轮执行证据。

## 完成范围

| 阶段 | 已完成产物 |
| --- | --- |
| 教师生成 | 881 题 × 16 候选，共 14,096 条新轨迹；13,937 条答案正确，20 条触及生成上限 |
| 学生数据 | short、medium、long 各 881 题；使用正确、去重、未触及上限的候选 |
| 学生训练 | 三种长度 × seeds 17/42/73，共 9 个 adapter |
| 学生评估 | Base + 9 个 adapter，共 10 组；每组完整评估 GSM8K `test[50:1319]` 的 1,269 题 |
| 激活提取 | 四个分片，覆盖审计无重复或缺失；每层 3,527,285 tokens |
| SAE | 第 10/17/23 层 × TopK 32/64，共六组；每组训练 1,500 步，seed 17；训练和特征评分均完成 |

新数据分别取下 20% 长度带代表、中位代表、上 20% 长度带代表，平均 completion 长度为 214.69 / 243.65 / 285.70 tokens。这是方法复现，不是历史极端最短/最长样本的逐字复现。

## 学生结果

学生固定为 Qwen2.5-1.5B-Instruct。下表准确率来自完整逐题评估；SFT 均值覆盖三个训练 seed。

| 条件 | Seed 17 | Seed 42 | Seed 73 | 平均准确率 | 相对 Base（百分点） |
| --- | ---: | ---: | ---: | ---: | ---: |
| Base | — | — | — | 67.45% | — |
| Short | 66.67% | 66.82% | 66.51% | 66.67% | −0.79 |
| Medium | 66.51% | 66.35% | 66.12% | 66.33% | −1.13 |
| Long | 65.01% | 63.83% | 63.75% | 64.20% | −3.26 |

差值根据未四舍五入的准确率计算。

![学生准确率比较](../../../results/ncsu_phase12_reproduction_v1/exploratory/analysis/student_comparison.png)

按训练 seed 与配对题目进行 10,000 次交叉 bootstrap：

| 预设比较 | 差值（百分点） | 未校正 95% 区间 | Holm 校正 p |
| --- | ---: | ---: | ---: |
| Short − Base | −0.79 | [−3.20, +1.68] | 1.0000 |
| Short − Medium | +0.34 | [−1.50, +2.15] | 1.0000 |
| Short − Long | +2.47 | [+0.29, +4.65] | 0.0852 |

Short 的点估计高于 Long，但三项比较经多重比较校正后均未达到 0.05 显著性水平。Short 相对 Base 没有观察到提升，不能用本轮结果支持“短轨迹 SFT 提升原始学生”。

## SAE 结果

六组 SAE 共确认 90 个模型内候选特征；该数量不是跨模型去重的语义概念数。主 SAE（第 17 层、TopK 64）的 24 个开发集候选中，留出集确认了 8 个 short-associated 和 8 个 long-associated 特征。

检验按题目配对、按 token 归一化，并要求前 64 个 token 的方向一致。当前结果说明存在长度关联特征，尚不构成因果干预或学生收益证据；本轮没有完成 SAE 干预后的学生蒸馏对照。

![SAE 留出集特征验证](../../../results/ncsu_phase12_reproduction_v1/exploratory/analysis/sae_figures/03_heldout_feature_validation.png)

## 执行异常与恢复

H100 完成教师生成及全部学生训练、评估。健康 L40S 完成激活提取及部分 SAE 训练和评分。期间存在 smoke 失败，以及 L40S 非法 CUDA 访问或停滞；后续重试已完成对应阶段，失败记录保留，没有将失败或部分分片计作完成。

## 证据核验

原交付审计记录 `passed`：104 个绑定产物及冻结源码验证通过，10 组评估各有 1,269 条记录。本次状态查询重新核对了 Slurm 最终作业、总完成标记直接绑定的 29 个文件，以及中文报告和总完成标记的 SHA256，全部一致。本次没有重新执行完整的 104 项产物及模型审计。

- [总完成标记](../../../results/ncsu_phase12_reproduction_v1/exploratory/EXPERIMENT_COMPLETE.json)
- [交付审计](../../../results/ncsu_phase12_reproduction_v1/exploratory/setup/final_delivery_audit.json)
- [完整中文报告](../../../results/ncsu_phase12_reproduction_v1/exploratory/analysis/report_zh.md)
- [学生统计](../../../results/ncsu_phase12_reproduction_v1/exploratory/analysis/student_summary.json)
- [生成审计](../../../results/ncsu_phase12_reproduction_v1/exploratory/generation/generation_audit.json)
- [激活覆盖审计](../../../results/ncsu_phase12_reproduction_v1/exploratory/setup/activation_coverage_audit.json)
- [SAE 特征分析](../../../results/ncsu_phase12_reproduction_v1/exploratory/sae_features/analysis/analysis_report.md)
- [方法与调度记录](ncsu_phase12_reproduction.md)
- [模型目录](../../../checkpoints/ncsu_phase12_reproduction_v1/)

## 结论边界与后续

本轮保持探索性 GSM8K 定位，完成标记为 `formal_claim_allowed=false`。学生有三个训练 seed，SAE 只有一个 seed；等样本数不等于等监督 token 或等计算预算。

建议后续优先核对新旧数据的长度选择规则、监督 token 数及训练实现差异，解释本轮与历史 short 涨点结果的不一致，再设计独立的特征干预验证。以上为后续建议，本次仅整理状态，未提交新作业。
