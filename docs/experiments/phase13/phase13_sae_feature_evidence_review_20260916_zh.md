# SAE 长短特征：已有消融、结论与证据缺口

盘点日期：2026-09-16。对应[新项目](phase13_sae_feature_separation_project_zh.md)。

我们已做过长短 feature 发现、采样消融、格式清理、同状态读回、强度/数量/窗口消融、保留题生成确认和跨 SAE seed 复验。尚未得到同时满足“正文语义清楚、跨 seed 稳定、因果特异”的长短 feature 集合；缺口集中在这条证据链上。

## 1. 本次证据等级

下列 NCSU Phase 12/13 报告已在当前项目读取。本次盘点报告、标记及引用路径，不重算全部预测或大文件哈希；原报告的审计数量只表示原审计范围。只读检查确认 readback、开发生成、确认生成、seed feature、seed generation 分析的 `COMPLETE.json` 存在，分别引用 20/23/24/75/18 项现存路径；未重新验证这些路径的内容哈希。Phase 12 六组 scoring 和 analysis/audit 标记存在，原 audit 状态为 passed。

Phase 2 sampling、Phase 5 clean-feature、Phase 6 的原 `results/` 目录在当前工作区不存在，历史结论据保留协议、报告及账本回顾。本次未重审旧集群激活、权重和预测，不能将历史完成状态等同于当前 NCSU 输入已验证可复用。

## 2. 消融与结果总表

| 实验/消融 | 主要观察 | 结论边界 |
| --- | --- | --- |
| Phase 2/12：layer10/17/23 × k32/64 | NCSU 六 SAE 共 90 个特征通过原 held-out 规则；主 layer17/k64 为 short8、long8 | 有统计关联，未建立语义或因果身份 [E1] |
| Phase 2：token-uniform / trace-balanced / prefix64 | 历史抽样量与长度 Spearman 0.768→0.006；三个字典严格稳定匹配候选为 0 | 采样暴露得到控制，稳定语义未证实 [H1] |
| 历史六字典跨 seed | full/prefix-balanced × 17/42/73；原组合规则下稳定特征为 0 | 取消 first64 准入的旧 B3 重筛选没有可据此认定完成的新结果 [H1] |
| Phase 5：固定 64 个清理后早期 token | 2,047 轨迹、131,008 状态；11 个候选通过，主 F24086 | 有值得恢复的候选；ID 仅对该历史字典有效 [H2] |
| Phase 5：正负读回及随机对照 | 主特征通过；v2 随机 F3739 覆盖率 4.15%<5% | 按门槛结束，后续生成/SFT 未触发 [H2] |
| Phase 3：干预与加强 sweep | 旧 24 adapters 无稳健收益；sweep 768 输出、零条件通过缩短筛选 | 加大力度没有现成成功保证；不是后续有效配方 [H1] |
| Phase 6：采样、数量、强度、时机、joint、随机 | dev 选 joint16/rho0.3/start0；test 261.61→198.75 tokens，正确数 63/64→62/64 | 具体方向可缩短，不是所有 short 增强均有效 [H3] |
| Phase 6：学生预算 | 等样本 SAE 69.66% vs 无干预 69.35%；等 token 70.37% vs 70.53% | 未确立学生收益，与当前 3B pilot 分开 [H3] |
| Phase 13：区域/词汇 | 八 short 的 Answer 激活质量 52.77%–98.04%；whole d=2.173–2.775，body d=-0.102–0.129 | 强格式关联，不能解释成已发现简洁推理 feature [E2] |
| Phase 13：格式反例与同状态读回 | 400 discovery、300 dev、37,500 局部测量；reference short 激活 0.13506，no_marker 0.00111 | 对标题敏感，局部测量不替代生成或学生结果 [E3] |
| Phase 13：剂量、特征、窗口 | 300×25 剂量、64×24 消融；top1/2/4/8、single、leave-one-out、四类窗口 | 完整开发比较已做，最高点不能追认为独立确认 [E4] |
| Phase 13：保留题生成确认 | 300×7 加 64×24，共 3,636 输出；SAE 290.60→249.95 tokens，287/300→279/300 | 正文与总体缩短，未确立优于格式或准确率非劣 [E5] |
| Phase 13：四 SAE seeds | short 数 8/3/5/7；top3 方向 cosine 约 0.11–0.14；严格描述性匹配无合格 short 对 | 新增三个 seed 均无法构造原八特征 [E6] |
| Phase 13：top3 seed 生成 | 64 题相对无干预长度差 -17.36/+17.98/-8.31/-8.69，依次为 17/42/73/101 | seed42 变长，未支持稳定缩短，属于事后适配 [E7] |
| Phase 13：Qwen 3B B0–B7 pilot | 1,024 训练题、14 adapters、1,269 测试题；B7 85.19%，B1 86.05%，base 86.84% | 单种子闭环已完成，SAE 学生优势未确立 [E8] |

历史看板 B1–B4 是旧任务编号，不等于 Phase 13 baseline 的 B1–B4，引用时必须写明分支。

## 3. 最关键的三组证据

### 原本很大的差异，去掉格式后明显减小

| 历史 feature | Answer 激活质量 | whole paired d | body paired d |
| --- | ---: | ---: | ---: |
| F24148 | 98.04% | 2.340 | -0.057 |
| F7339 | 52.77% | 2.775 | -0.053 |
| F11240 | 97.42% | 2.392 | 0.129 |
| F26001 | 58.33% | 2.173 | -0.102 |

其余四个见[区域报告][E2]。这说明为什么旧图有很大的统计差异，却难以解释成不同推理行为。只删字面 Answer 也不够，标点、标题与后缀需独立处理。原支持 1,035/1,036 轨迹有标题，但没有标题在 first64 内出现；原 first64 同向准入不要求早期效应显著。

### 确认了缩短，没有确认独有语义优势

| 300 题确认条件 | 正确数 | 生成 tokens | 正文 tokens |
| --- | ---: | ---: | ---: |
| 无干预 | 287 | 290.60 | 282.33 |
| SAE rho0.3 | 279 | 249.95 | 242.93 |
| Answer-format rho0.3 | 277 | 246.47 | 240.17 |
| Dense rho0.3 | 276 | 204.90 | 198.84 |

SAE 相对无干预及三个随机方向的长度检验通过原 18 项 Holm 校正，相对 Answer-format 的差异未通过。正确率数值下降 2.67 pp，未显著不代表非劣。确认题旧 baseline 已观察，独立性限于未用于拟合/选点的新干预效果；答案复核由项目助手执行，不称独立人工标注。[E5]

开发窗口消融中，before-marker 为 261.19 tokens，after-marker 为 301.72，未干预为 301.41。这值得继续检查标题前的作用，但不能直接将八特征命名为正文语义 feature；完整条件及正确率见[开发报告][E4]。

### 跨 seed 是实质性限制

四 SAE 的测试 MSE 都约 0.224，但 short 数量、方向和生成效果不稳定，重构质量近似不能替代 feature 稳定性。seed101 的 F19908 虽无 Answer 激活质量，却有 96.25% 的质量在冒号，仍是格式关联反例。[E6]

top3 补充使用相同已观察 64 题，不能因为改成三个 feature 后可以生成，就说原八特征方法通过复验。seed feature 分析标记中的 `generation_stability_complete=false` 只界定该阶段；后续生成有自己的分析标记，而该生成结果未支持稳定缩短。[E7]

## 4. 现在走到哪一步

机制线：旧方向的发现、格式反证、读回、生成消融、保留题确认及 seed 复验已有结果，可定位为“可控制长度的格式关联方向，尚缺稳定正文语义解释”。Phase 5 是另一条正文候选路线，停在对照准入且原始输入待恢复。

学生线：Qwen 3B 单种子 GSM8K pilot 已完整审计，B7 相对 B1 为 -0.87 pp，配对 95% CI [-2.52,+0.79]；七项比较 Holm 均未达 0.05，额外训练 seeds 未过门槛。[E8] KD v2 已有完成记录，两 seed 增益为 +2 pp/0 pp，平均区间跨零，未通过放大门槛。这是已观察开发集的 B1 小试，不是 SAE feature 证据，详见[独立 KD 决策](../../../results/phase13_baseline_expansion_v1/exploratory/token_kd_small_v2/analysis/decision.json)。旧配方对齐分支的最终状态仍以自身证据为准。

规模线：MATH 7.5K/25K 和跨数据集主实验尚未形成完整结论；旧队列取消不等于主实验完成。本次只读查询时当前用户队列为空，队列为空本身不能区分任务成功、失败或取消，不能据此更新实验完成结论。

下一步是新项目 P1：核验 NCSU corpus/activation/checkpoint/tokenizer，盘点观察历史与自然长短配对支持；再做同一支持上的正文重评分和 feature cards。历史 F24086 权重无法恢复就重新发现，不能移植编号。新项目目前仅立项完成，尚无新 feature 结果。

## 证据入口

- E1：[NCSU 六 SAE 长短特征报告][E1]。
- E2：[Answer 区域诊断][E2]。
- E3：[格式反例与同状态读回][E3]。
- E4：[完整开发剂量与消融][E4]。
- E5：[保留题确认报告][E5]及[协议](phase13_sae_confirmation_v1.md)。
- E6：[SAE seed 特征复验][E6]及[训练、匹配、失败记录](phase13_sae_seed_stability_v1.md)。
- E7：[SAE seed 新生成分析][E7]。
- E8：[Qwen 3B pilot 审计总结][E8]。
- H1：[项目历史任务看板][H1]，历史 A1/B2/C1/C2 需标明各自分支。
- H2：[Phase 5 协议及停止位置][H2]。
- H3：[Phase 6 完整报告 v2][H3]，包含随机对照修订与训练执行差异限制。

[E1]: ../../../results/ncsu_phase12_reproduction_v1/exploratory/sae_features/analysis/analysis_report.md
[E2]: ../../../results/phase13_baseline_expansion_v1/exploratory/answer_marker_diagnostic_v1/report_zh.md
[E3]: ../../../results/phase13_baseline_expansion_v1/exploratory/sae_readback_analysis_v1/report_zh.md
[E4]: ../../../results/phase13_baseline_expansion_v1/exploratory/sae_generation_analysis_v1/report_zh.md
[E5]: ../../../results/phase13_baseline_expansion_v1/confirmation/sae_generation_analysis_v1/report_zh.md
[E6]: ../../../results/phase13_baseline_expansion_v1/exploratory/sae_seed_analysis_v1/report_zh.md
[E7]: ../../../results/phase13_baseline_expansion_v1/exploratory/sae_seed_generation_analysis_v1/report_zh.md
[E8]: ../../../results/phase13_baseline_expansion_v1/exploratory/gsm8k_qwen3b_sae_pilot_v1/final_review_v1/report_zh.md
[H1]: ../../../PROJECT_STATUS.md
[H2]: ../phase5/phase5_sae_clean_feature_causal_plan_zh.md
[H3]: ../phase6/phase6_local_length_controlled_strength_report_zh_v2.md
