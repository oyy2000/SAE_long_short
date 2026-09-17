# Phase 13：成熟压缩基线、多模型蒸馏与 SAE 机制实验计划

登记日期：2026-09-13。用户要求将 ASC、DAP、TokenSkip 纳入真实实验，按阶段从 pilot 推进至较大规模，并明确批准沿用 NCSU 现有账户与分区。

2026-09-16 新增独立[长短 SAE 特征分离与可解释性项目](phase13_sae_feature_separation_project_zh.md)，并整理[已有消融与证据缺口](phase13_sae_feature_evidence_review_20260916_zh.md)。新分支从正文语义、格式反例和跨 seed 稳定性推进，不把旧八特征的生成缩短视为语义解释已成立。下表保留原分支阶段记录，最新执行状态以根目录 PROJECT_STATUS.md 和各实验完成证据为准。

本文是新分支的执行计划。`configs/phase13_baseline_expansion_v1.json` 是机器可读计划，尚不是任一训练阶段的冻结协议；每个阶段实施前绑定输入、源码、模型、数据集版本与完成条件。原 878/881 题 GSM8K、三种子 factorial 和所有历史结果保持原位。新分支允许用户指定的跨数据集研究，但论文中的新性能结论须等待对应完整证据。

## 研究终点

在相同学生与可核算的预算下，检验 SAE 构造的监督相对成熟压缩方法是否具有额外价值。分别报告：

- 相近学生输出长度下的准确率；相近准确率下的实际成本。
- 教师与学生准确率、生成长度、延迟、cap-hit、有效题目覆盖。
- 一次性 SAE/方向构建成本、数据生成与改写成本、SFT 成本和字典复用后的边际成本。

不把“教师变短”或“学生继承短输出”单独当作蒸馏质量收益；统计不显著不等于准确率非劣。

## 一、当前完成状态

| 阶段 | 本次进度 | 已交付/下一个验收条件 |
| --- | --- | --- |
| S0 文献/源码和矩阵准备 | 已完成源码核对与84格矩阵构建 | 固定三个官方仓库 commit、文件哈希与正式论文；校验48+12+12+12，未提交这些训练 |
| S1a 既有 Answer 区域诊断 | 已完成 | 132题、1,036轨迹、16特征；稀疏事件重建原全序列和first64均值通过，输出区域分解、配对统计和图 |
| S1b 新题语义反证及目标读回 | discovery/dev 与保留题确认完成；三个新 SAE 训练/原规则筛选及探索性 top-3 生成、复核、分析完成，未支持原八特征复验或跨 seed 稳定缩短 | 保留题角色、真实模型、格式/dense/多个随机对照、同状态激活与logit读回；保留不利结果与解释边界 |
| S2 ASC/DAP/TokenSkip实现及小规模核验 | TokenSkip原设置完整SFT/23条件评测完成；DAP长轨迹及统一教师194条改写完成；R1 native CoT开发集压缩出现，MATH ASC完整生成/分析完成且未确立长度优势 | 方法组件齐全、命名准确；完整报告控制曲线、失败设置与准确率损失 |
| S3 新数据与跨数据集评测准备 | 来源、grader、题目角色、四模型cap及R1两题扩展完成；两个学生的8K合成SFT接口已验证，最终方法预算待冻结 | 版本、去重去污染、开发/评测划分、grader、生成cap和环境都冻结 |
| S4 7.5K来源池完整核心基线 | 修订 ASC/dense 与 DAP/TokenSkip 开发重跑完成；新 steering 开发及原 raw 生成中，旧下游继续 hold | 完成新开发选点、学生 steering 与全 raw 评分迁移，再完成共同支持及分阶段 SFT |
| S5 25K主实验 | 训练待S4及成本评估；OpenThoughts完整来源、64,946个预筛候选的参考解析/跨源一致性及内部近似审计已完成；参考语义仍待复核 | 预筛和旧评分自比较不等于可用训练数据；达到真实唯一可用题数目标后，执行主48次SFT |
| S6 精简泛化与token控制 | 待S5 | 跨家族12、跨teacher12、等token12，完整评测与最终审计 |

执行证据：

- [官方源码核对](../../../results/phase13_baseline_expansion_v1/preparation/source_audit_v1/source_audit.json)
- [84格参考矩阵](../../../results/phase13_baseline_expansion_v1/preparation/reference_matrix_v1/matrix.json)
- [Answer区域诊断](../../../results/phase13_baseline_expansion_v1/exploratory/answer_marker_diagnostic_v1/report_zh.md)

### S1a 的实际发现及对后续的影响

八个短特征的最高激活质量 token 均为 `Answer`，该单词解释各特征总激活质量的 **52.77%–98.04%**。全序列的同题 paired d 为2.173–2.775，去掉答案标题及后缀、仅分析正文后为−0.102至+0.129。只删除字面 `Answer` 对部分特征仍不够，说明标题标点和答案区域也需要控制。

检测到显式答案标题的轨迹为1,035/1,036，本支持中没有标题在前64 token内出现。因此不能把本轮 first64 同向检查解释为“前64已经包含答案标题”；其同向条件本身也不要求first64效应显著。后续读回和控制须单独检验早期稀疏激活的含义。

这是已观察题集的事后描述，不是独立确认，更不是“Answer导致全部缩短”的因果证明。当前八特征保留为原pilot的格式关联集合。正式分支在S1b后固定其语义定位或另行发现新集合，任何新集合必须用新版本命名，不覆盖原特征与失败记录。

## 二、文献依据与实现差异

### ASC-CES：正式生成端基线

- [正式论文](https://aclanthology.org/2026.findings-acl.1828/)；[官方仓库](https://github.com/ArminAzizi98/ASC)。100对verbose–concise用于学习/校准方向，不是学生蒸馏规模。
- 已固定官方仓库 commit `27f02ed4a1a935cbe3a2736cc275ecf92e89ee0d`。所核对 `extract_steering_vector.py` 仍提取长短末位hidden差，当前仓库Python实现没有正式CES/KL训练过程。
- 行动：按正式论文方法段恢复目标、长度归一化和可微KL约束，逐项记录公式与实现位置；首先验证有限loss/gradient、约束响应和生成控制。此项未完成前B4不可称已复现。
- B3 Dense Mean-Difference独立保留。迁移教师、改变学生训练方式以及方向校准数据均写清楚。

### DAP：难度感知文本改写基线

- [论文](https://arxiv.org/abs/2505.19716)；[官方仓库](https://github.com/Evanwu1125/LiteCoT)。25K同题长短控制作为规模参考，不拿已发布Liter分数直接对比。
- 已固定 commit `86a7ef32361f60958e0e626c5a58c2fe1d553868`。公开仓库重点为数据、SFT和评测；论文§3.1给出easy/medium/hard难度判断及条件化改写。
- 行动：恢复并冻结带完整原始解答的难度判断/改写流程，记录难度、prompt、改写模型、原始和改写token、验证和失败。若需要依据论文重构prompt，应明确是方法级适配，不声称原作者逐字prompt复现。
- 重写教师先采用主teacher以便统一比较，额外强模型仅作为另登记变体，不隐性加入成本。

### TokenSkip：比例条件化基线

- [正式论文](https://aclanthology.org/2025.emnlp-main.165/)；[官方仓库](https://github.com/hemingkx/TokenSkip)。保留LLMLingua-2、比例条件化训练与推理。
- 已固定 commit `7b958cdb98e0b2a84056e26e885b260987f2e64f`。代码为每题分配一个压缩档位，不是每题自动扩成六条监督；公开输入格式还需按模型chat template正确适配。
- 行动：实现teacher trace到压缩数据的适配，保留最终答案处理与六个保留比例0.5–1.0。主等样本设置按题号哈希平衡分配一个档位，冻结到各训练seed共同使用；同一adapter在多个比例上评估。实际压缩长度可能不等于请求比例。
- 原配方LoRA/epoch用于小规模faithfulness检查，主表使用统一学生SFT配方并称TokenSkip-style distillation。不得只调用压缩器就宣称复现完整TokenSkip。

Small Models Struggle用于设计两个学生规模及同teacher mixed-length补充；Compress-Distill用于原始/压缩/answer-only/长度匹配截断和成本分析。后两类消融放第二层，不挤掉B4/B5/B6。RL或偏好优化方法暂不加入固定SFT主矩阵。

## 三、模型、数据与训练矩阵

| 维度 | 新分支设置 |
| --- | --- |
| 主teacher | Qwen2.5-7B-Instruct |
| 第二teacher | DeepSeek-R1-Distill-Qwen-7B；单独收集激活并训练/校准SAE，不直接套用普通Qwen的字典 |
| 主students | Qwen2.5-1.5B-Instruct、Qwen2.5-3B-Instruct |
| 跨家族student | Llama-3.2-3B-Instruct；先检查模型访问与chat template |
| 学生seeds | 17、42、73 |
| SAE稳定性 | 主seed17之外新增42/73/101三个SAE seeds；旧seed17仅在数据/配置/源码完全匹配时复用，否则四个均重训 |
| 起点 | 保留历史约1K pilot；7,500题MATH原始训练池先做核心比较 |
| 主规模目标 | 25,000唯一可用数学问题；固定OpenThoughts版本及抽样清单 |
| 评测 | GSM8K、MATH-500、GSM8K-Hard、AQuA-RAT、OlympiadBench英文纯文本数学 |

7,500是来源池规模，不是自动保证7,500条最终训练数据。开发保留、测试去污染、格式验证、正确过滤和共同支持都可能降低实际训练题数，必须据实报告。25K目标也不能靠重复训练记录凑数；必要时扩大预先登记的候选问题池，记录所有排除与补充。

先冻结当前训练配方的可复现实现，随后在共同开发问题上检查学习率/轮数与长序列适配。小规模原方法检查和统一主表是两套记录。不得凭不同方法的最终测试表现逐方法选择训练超参数。R1生成cap与学生sequence length须重新校准，不能静默沿用512/2048而截掉大部分长推理。

| 模块 | 方法/模型组合 | SFT次数 |
| --- | --- | ---: |
| 主结果 | 主teacher × B0–B7 × 两Qwen学生 × 三seed | 48 |
| 跨学生家族 | 主teacher × B1/B4/B5/B7 × Llama学生 × 三seed | 12 |
| 跨teacher | R1 teacher × B1/B4/B5/B7 × Qwen1.5B × 三seed | 12 |
| 等监督token | 主teacher × B1/B4/B5/B7 × Qwen1.5B × 三seed | 12 |
| 一个完整规模的参考总量 | 不含前置pilot、额外压缩点SFT、SAE拟合或模型原配方验证 | 84 |

7.5K阶段先执行关键4条件12次，再补其余4条件12次。25K全矩阵的84次不能同时充当7.5K和25K两个完整矩阵的总次数；新增压缩曲线训练点须另计预算。TokenSkip一个adapter改变推理比例不算新SFT，其额外评测次数仍计入成本。

## 四、固定主表B0–B7

| ID | 数据构造 | 核心控制 |
| --- | --- | --- |
| B0 | 无干预Random-Correct | 与B1同一原始候选池，固定随机选样 |
| B1 | 无干预Shortest-Correct | 当前no-steering对应此项 |
| B2 | Concise Prompt + Shortest-Correct | 统一最终答案格式 |
| B3 | Dense Mean-Difference + Shortest-Correct | 同题、同层、题目等权的长短方向 |
| B4 | 正式ASC-CES生成 + SFT | 不是旧版均值差 |
| B5 | DAP改写 + SFT | 固定原始候选与难度感知流程 |
| B6 | TokenSkip-style + SFT | 六档位条件化，每题一条训练记录 |
| B7 | SAE生成 + Shortest-Correct + SFT | 在S1后冻结特征、方向、强度和时机 |

Base student另列，不计入8个训练条件。多个随机SAE方向、answer-format方向、same-teacher mixed-length、answer-only和截断监督作为机制/第二层对照，明确其额外作业数。

## 五、分步骤执行和验收

### S1b：先验证格式解释

从GSM8K train抽取约1K新问题，排除原发现/调参问题及可追溯历史重复，划分发现、开发和确认部分；实际ID、去重规则和划分种子在GPU运行前冻结。初始小批真实模型验证通过后再扩全量，不使用锁定学生测试题调参。

2026-09-13 执行状态：已固定 400 discovery / 300 dev / 300 confirmation，8 题 smoke 与 1,000 条原始单候选轨迹均完成。统一答案复核为 smoke 8/8、主输入 941/1,000；不得将这组未干预生成准确率当作 SAE 机制效果。区域提取与控制方向拟合已完成；8 题真实 GPU 的 680 条局部读回和 11 项数值/缓存测试通过，300 题 dev 的六方向×四剂量对照已完成 37,500 条测量及审计，报告见 [新题读回](../../../results/phase13_baseline_expansion_v1/exploratory/sae_readback_analysis_v1/report_zh.md)。实现和边界见 [同状态读回协议](phase13_sae_readback_v1.md)。

完整 dev 生成、特征数/窗口消融，以及 [300 题保留干预确认](phase13_sae_confirmation_v1.md) 已完成新文本复核和分析。确认结果支持相对未干预/随机方向的总体及正文缩短，尚未确立 SAE 优于 Answer-format，亦不证明准确率非劣。保留题旧 baseline 已观察，独立性仅限于未用于方向拟合、剂量选择及干预效果分析。

[三个新 SAE seeds](phase13_sae_seed_stability_v1.md) 已完成同样本训练、原规则特征筛选、候选匹配及词汇区域分析。seed17/42/73/101 分别仅有 8/3/5/7 个 confirmed short，原八特征构造在新增 seeds 上均不可行；保留这一限制，不用未确认特征补齐。补充同 H200 的四 seed top-3 生成是在看到入选数量后另行冻结的探索性实验，不替代原方法复验或新增独立确认。

1. 文本区域分为reasoning body、answer marker、answer suffix。构造逻辑不变的格式替换、格式不变的错误计算、非推理词汇注入，并保留token/span对齐。
2. 用同一个hidden state进行Enc(h)与Enc(h+delta)读回，严格复用NCSU实际八特征组合、层索引、缩放和BF16；报告目标/非目标激活、TopK更替、真实next-token KL、Answer/EOS概率。
3. 比较原SAE方向、dense、answer-format、至少三个预抽取随机SAE集合。每种方向先开发集剂量校准，所有随机集合和失败条件均报告。
4. 固定总扰动范数做top1/2/4/8、单特征与leave-one-out；采用同前缀独立cache和早期/延迟/答案后窗口，确定缩短发生在哪里。
5. 若效果主要是格式/终止控制，按实际结果重新定位B7；若另发现正文相关特征，用新协议和独立问题确认。不能通过降低历史门槛“修复”原语义结论。

验收：来源/代码/模型哈希、全部输出、正确率与覆盖率、区域读回、随机对照、报告和完成标记齐全。通过是指实现和证据完整，不预设必须得到有利结果。

### S2：先检查方法忠实性，再统一比较

ASC须具备正式目标和KL约束及其数值测试；DAP须记录难度判断与完整原轨迹条件；TokenSkip须有ratio条件化的数据、训练和评测。原配方小规模验证与统一teacher/student适配分开登记。

每种方法先检查参数改变是否产生预期控制趋势、是否有异常截断/格式失败/正确率崩溃，再对约1K开发规模做统一候选池比较。缺少关键组件时该方法状态为“实现未完成”，不能用简单替代静默填主表。

当前 TokenSkip 已完成完整比例条件化 SFT 和作者/replica 各 11 条件、base 1 条件评测；固定 cap 曲线与 scaled cap 对照分别报告。DAP 已在 64 个正确长源轨迹上完成真实 Qwen-7B 改写，完整输出平均缩短 83.77%、63/64 正确，原论文改写模型差异单独标注。ASC 的低强度收益区间跨零，一致 native CoT 提示和 boxed 参考答案的重新校准与拟合已完成，完整生成仍运行；目标函数与 KL 配方保持正式版。已有 31,091 条输出经统一评分审计，原分数与逐项修正保留；新生成记录需要另外审计，不能直接套用已有题号的评分。

### S3：数据、评测与cap冻结

- 绑定数据集revision、原始ID、语言/领域/模态过滤和抽样种子；保留原始题目与答案。OpenThoughts同来源同规模不等于LiteCoT原始25K样本。
- 对五评测集合做规范化精确重复、近重复及可用来源ID检查；报告检查方法的局限。原有benchmark预训练污染无法由本次去重排除。
- DAP 长轨迹检查使用的 64 个 MATH 题已登记在 `results/phase13_baseline_expansion_v1/preparation/dap_paired_sources_v2/reserved_math_development_ids.jsonl`；冻结学生题池时必须排除，计入真实留出规模。公开长短对应题不等于恢复了 LiteCoT 原始 25K 样本 ID。
- OpenThoughts 固定 revision 的 12 个 metadata 分片已通过来源及完整性审计：89,120 个唯一数学问题，词汇/参考结构/保留集预筛后 64,946 个候选。参考 box 不自动构成可靠 gold，尤其 22,794 条原数学记录含证明措辞；见 [来源结果](phase13_openthoughts_inventory_v1.md)。
- 在 25K 选题前执行 [全部候选参考审计](phase13_openthoughts_reference_audit_v1.md)：分别报告自比较解析覆盖、精确 MATH 同题参考一致性、全部数学来源的 near 连通组及保留集隔离传播。复核失败和 32 个事先固定的来源质量样本；不以解析自比较替代数学正确性验证，也不将 DeepSeek 生成解答当作 gold。
- [MATH 参考完整性检查](phase13_math_reference_integrity_v1.md) 已发现最后一个 box 丢失多答案等问题：7,987 题扫描标记 629 个复核 case。原文本准备已暂停；应完成语义/类型修订，检查实际方向及开发输入影响，并重评分完整候选后恢复主流程。保留原轨迹和历史 v2 评分，不通过直接删除所有标记题绕过检查。
- 2026-09-13 21:43 EDT：完整 cohort v2 已通过 7,987 题与 8,894 项检查；648 个显式复核定义、7,339 个自动转换定义，原数据字段保持不变。5,018 条真实校准/开发输入输出审计发现 6 条标签变化，均已全文复核；其中 07414 的首个候选应进入 ASC/dense 100 对校准集并替换 06821。已 hold 28 个待运行旧方向分片与合并，保留运行中和历史生成；文本入口 822579 继续 hold。下一步重新冻结校准输入、拟合方向、重跑受影响开发选点及全池重评分，再迁移 S4 下游，不能直接沿用旧方向或释放旧入口。
- MATH-500是MATH的留出评测；GSM8K-Hard是GSM8K衍生鲁棒性集合，保留父题关联，不能当作五个完全独立OOD证据。
- GSM8K保持既定smoke/locked split。AQuA按选项、MATH/OlympiadBench按数学表达式与题型判分；对grader建立真实答案格式、等价表达式、无效输出和超时的回归检查。
- 开发集预先校准主teacher与R1的生成cap和SFT最大长度。比较时统一该teacher各方法cap，报告所有cap-hit；完整轨迹优先，不把截断结果当作正常正确监督。

2026-09-13 22:21 EDT 更新：修订 ASC-CES 完成 3,000 步，新旧向量 cosine 0.925484；dense 已重算。DAP/TokenSkip 194/336 条开发重跑及全部分片合并通过。新 steering、原生采样 ASC、开发选点、学生准备和全 raw 评分审计已分别登记依赖，详见 [续跑步骤与证据](phase13_reviewed_baseline_continuation_v1.md)。当前没有正式统一学生 SFT；旧历史方向和压缩入口继续 hold。

### S4–S6：生成、训练与审计

生成端先每题4候选，统一验证、去重和shortest-correct（B0明确随机例外）。B5/B6消费同一B1源池，原始生成及额外压缩调用全计成本；压缩后的正确性/格式再验，不能因附回正确final answer就认证中间推理正确。

训练端同一共同题集做等样本，报告全部池覆盖率；等target-token独立登记实际tokens、steps和题目重复权重。跨tokenizer不直接把token总数当成可比FLOPs。

压缩水平在开发集校准；主表预先固定一个operating point，教师曲线可用多个生成档位，学生多点曲线需要对应数据集和SFT成本另计。TokenSkip同一adapter的多个比例须报告全部，不能从最终评测挑最优比例充主分数。

准确率区间按训练seed及配对题目重采样，多个计划对比的校正族在执行前登记。衍生benchmark的父题关系单独处理；不按结果选择数据集或种子。非劣主张需要事先明确margin、设计和检验。

最终模型、配置、训练源码、launcher与输入哈希、各shard manifest、重复/缺失审计、逐例预测、成本和汇总齐全后写阶段完成标记。历史和新规模严格分目录。

## 六、成本与规模推进决策

总成本分为activation/SAE或基线方向拟合、候选生成、改写/验证、学生训练四部分，学生服务推理另报告。分别展示硬件与batch相同的延迟、GPU时长和显存峰值。

若方法有更高的一次性成本但更低单次推理成本，按真实测量计算摊销次数；分母无节省时明确无可用摊销点。格式和控制器开销不能从“短了多少token”推断出来。

7.5K阶段结束后的25K推进报告需包含：已完成方法、共同支持与损失、最大显存/时长、预估84格资源、正确率—成本曲线，以及当前SAE解释。规模推进不以“必须显著涨点”为唯一门槛，也不在证据缺失时直接扩矩阵。

## 七、代码复用与当前运行方式

- 计划：`configs/phase13_baseline_expansion_v1.json`。
- 矩阵构建：`scripts/13_1_build_baseline_matrix.py`，逻辑为`src/length_budget_distill/baseline_plan.py`；只生成元数据，不提交训练。
- 已执行诊断：`scripts/13_2_analyze_answer_marker.py`与`configs/phase13_answer_marker_diagnostic_v1.json`；可复用逻辑在`src/length_budget_distill/answer_marker_diagnostic.py`。
- 诊断沿用既有token-event统计和蓝/红配色，扩展为动态feature数量、答案区间剔除和保存均值的重建审计；复用`sae_feature_analysis.py`题内统计，避免拷贝旧脚本中固定5个short feature的历史说明。
- GPU后续复用`ncsu_reproduction.py`、`sae_norm_intervention.py`、`ncsu_intervention.py`的分片、cache、范数和memory-fit准入；上游方法依赖隔离，不能破坏已冻结legacy TRL环境。
- 资源：用户`youyang7`，GPU账户`jekml_gpu`；H200/L40S使用`gpu_partners/short_gpu`，分别请求`gpu:h200:1` / `gpu:l40s:1`；H100使用`gpu/gpu`和`gpu:h100:1`。每次launch重新核对Slurm分配GPU、nvidia-smi、剩余显存与非本任务进程，临时文件和中间checkpoint使用管理员指定的`/share/jekml/youyang7/tmp`下作业独立目录。

S0和S1a已完成；S2已进入真实GPU复现，当前结果与活跃作业见[执行台账](../../../PROJECT_STATUS.md)和[方法实现说明](phase13_baseline_reproduction.md)。MATH的7,500题来源池及五个评测集已完成第一轮数据审计，12 个近似训练题与重复已排除、17 项答案修复及全部 GSM-Hard parent 映射已登记；typed grader 的 11,453 条参考自比较和 12 组正反例检查通过，题目角色已冻结为 6,723 学生候选、400 校准、300 开发、64 DAP 保留；三个 Qwen 的 32 题 cap 检查完成，R1 在 8K 仍有 2/32 cap hit，已启动仅两题的 16K/32K 扩展；最终方法/SFT 上限仍待核验。准备标记、完整向量拟合及文本压缩记录均不等于学生蒸馏已完成。继续推进S1b/S2并完成数据冻结后，再进入正式规模训练。

矩阵准备说明：`reference_matrix_v1/config_snapshot.json` 保留首次生成84格SFT矩阵时的计划快照。后续将SAE稳定性明确为主seed17之外新增三个seed；该调整不改变84格学生SFT矩阵，新增SAE训练成本单独核算。

统一数学候选接口见 [执行说明](phase13_unified_math_candidates_v1.md)：B1/B2 使用相同四候选和明确 top-k 设置，B0 由 B1 同池随机选择；当前只执行开发集，最终训练池仍等待全部方法和共同支持冻结。

2026-09-13 16:24 更新：统一 DAP/TokenSkip 的实际开发压缩已完成，见 [执行说明](phase13_unified_text_compression_dev_v1.md)。完整 SAE 生成控制显示 SAE 与答案格式方向均可缩短正文，当前没有 SAE 优于格式控制的证据；保持格式关联解释与全部失败/退化条件。MATH B3/B4/B7 的共享采样四候选开发矩阵登记于 `phase13_unified_steered_math_candidates_dev_v1.json`，先固定可用运行点，再冻结学生池生成；原 84 格学生终点保持不变。此前段落为阶段性历史状态，最新作业状态以执行台账和完成产物为准。

2026-09-15 storage correction: the current launcher uses administrator-designated scratch. Historical frozen launchers and previously submitted Slurm scripts retain their original paths and require separate migration verification.
