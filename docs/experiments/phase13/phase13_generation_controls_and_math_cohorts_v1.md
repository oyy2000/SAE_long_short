# Phase 13：完整生成对照与 MATH 题目角色

本阶段承接 [SAE 同状态读回](phase13_sae_readback_v1.md)，推进 S1b 的完整生成控制和 S3 的题目角色、长度预算检查。历史结果和源码快照保留。这里的完成标记不代表统一 B0–B7 学生 SFT 已完成。

## SAE 完整生成

配置 `configs/phase13_sae_generation_controls_v1.json` 在生成前冻结。入口为 `scripts/13_25_run_sae_generation_controls.py`；复用 `scripts/13_23_submit_sae_readback.py` 提交，配置中的 entrypoint 区分局部读回和完整生成。生成逻辑复用 `sae_norm_intervention.generate_condition`、`NormMatchedController` 和逐题固定均匀随机数的 nucleus sampler；只增加可选 token 观察回调和具名方向/窗口控制。

教师、SAE、BF16、字典缩放和特征身份保持 discovery/dev 读回的设置。每题每条件一个候选，cap 1024、temperature 0.7、top-p 0.95、batch size 8，seed 2026091308。这是新配对采样，不能套用早期未干预轨迹的评分或 token 身份。各条件重新从 prompt 开始生成，共享逐题随机均匀数；采样 EOS 计入实际 token ID 列表，`generated_tokens` 排除 EOS。

| 生成模块 | 问题 | 条件 | 预期输出 |
| --- | ---: | ---: | ---: |
| 完整剂量 | 全部 300 道机制 dev 题 | 零剂量＋六方向×四剂量 = 25 | 7,500 |
| 数量与窗口消融 | 固定 hash 抽取的 64 道 dev 题 | 24 | 1,536 |
| GPU smoke | 原 8 道独立 smoke 题 | 11 个接口覆盖条件 | 88 |

六方向为历史八特征 SAE、discovery-only dense、答案格式和三个冻结随机八特征集合；四剂量为 0.05/0.1/0.2/0.3。消融固定历史 rho=0.3，不根据本次完整生成结果选择：八特征完整方向、top1/2/4、另外七个单特征、八个 leave-one-out、四种窗口及零基线。top1 同时是第一个单特征，不重复计算。所有非零方向归一化后匹配总相对扰动范数；`count=8` 表示读回的历史目标数，实际注入成员另外存为 `injected_feature_ids`，避免将 top1 误解为八特征注入。

四窗口为前 64 个预测位置、从第 64 个位置开始、首个显式答案标题完成前和完成后。标题完成以在线解码文本匹配登记正则为准，观察到冒号之后的下一 logit 才启用 after-marker。没有标题的输出保留，after-marker 对该行不干预。窗口按行控制，不改变已结束行的状态；缓存从每个条件的原始 prompt 重新建立。直接验证 delayed 的首 64 个 token 和 after-marker 的标题前 token 与同批零基线相同。

CPU `821357` 的七项窗口、成员、范数和采样回归检查通过；准备 `821363` 完成。L40S smoke `821366` 完成 88 条输出，零重放、窗口边界、非零作用和范数容差均通过，峰值 15,268.54 MiB，零 cap hit。自动评分仅用于记录，不作为 smoke 的效果门槛。

完整剂量作业为 `821369/821371/821373/821375`，对应的消融为 `821370/821372/821374/821376`，每片消融依赖同片剂量作业成功，限制同时运行的任务量。`generation_dag.json` 的首次数组汇总因通用 writer 只接受 object 而失败；八个实际提交都成功，空文件保留，`generation_dag_v2.json` 从逐作业提交记录恢复，没有重启作业。

每个输出保留真实 token IDs、完整文本、实际扰动统计、标题出现时间、重分词后的 body/marker/suffix 计数、批次时长及硬件 UUID。区域 token 数来自解码文本重新分词，不冒充生成时 token ID 的精确分区。批次时长在批次级单独记录，不把同一批时间重复求和为逐样本成本。所有条件均加载 SAE 并进行编码读回，计时还包含在线文本解码/标题识别；这些机制测量开销不能作为部署时延或完整端到端效率结果。

新疑难答案审计配置 `phase13_uniform_answer_review_v3.json` 包含上述 9,124 条 SAE 输出、ASC native CoT 192 条评估和 103 条校准尝试，共 9,419 条。准备作业 `821638` 依赖生成完成；评审队列隐藏方法与 gold，实际正误决策完成后才能把该评分用于最终对照。未审查自动分数不决定最终方向/剂量。

## MATH 固定题目角色

`scripts/13_26_freeze_math_cohorts.py` / `configs/phase13_math_cohorts_v1.json` 复用来源审查、规范化、near-match 和 typed grader。CPU 测试 `821367` 的三项分组/配额/排除检查及准备 `821368` 均完成，25 项直接产物绑定已核验。

原来源 7,500 题，经先前 12 个评测近似题和 1 个重复排除后为 7,487。对全部剩余 MATH 题计算词汇 5-gram Jaccard≥0.8，得到 51 个无向近似题对、7,445 个连通组，最大组 5 题。DAP 保留题及其连通组先划出，再以固定种子和确定性 subset-sum 分配完整组件：

| 角色 | 唯一问题数 |
| --- | ---: |
| 学生候选池 | 6,723 |
| 校准 | 400 |
| 开发 | 300 |
| DAP 开发保留 | 64 |

此次 DAP 64 题没有额外达到阈值的剩余近邻；跨角色阈值内近似边为零。保留所有题的学科、难度、来源 revision、答案修正和组件 ID。近似检查不保证消除全部语义重复或预训练污染。

五个固定评测集的 ID、typed answer spec 和 GSM8K-Hard 父题映射同时保存。GSM8K smoke 为前 50 题，正式评测仍为后 1,269 题。6,723 是候选生成池，最终共同正确支持和实际 SFT 题数必须另行审计，不重复样本凑成 7,500。

## MATH 长度预算检查

`configs/phase13_math_cap_calibration_v1.json` / `scripts/13_27_calibrate_math_caps.py` 固定从上述 300 dev 题中 hash 抽取 32 题，覆盖五个已知难度和七个学科。两个教师为 Qwen2.5-7B-Instruct、R1-Distill-Qwen-7B；两个学生为 Qwen2.5-1.5B/3B-Instruct，所有 revision 和模型文件哈希固定。教师采样沿用各自登记参数，学生 greedy；统一采用 native chat 的 step-by-step / boxed 答案提示。

每题生成最多 8,192 tokens，保留实际 token 序列，离线比较 1,024/2,048/4,096/8,192 四个截断预算。模型若设置 forced-EOS 则拒绝使用这种前缀模拟。每个模型另外对前两题直接生成 1,024 cap，要求 token 序列与完整输出的对应前缀完全一致。原始完整生成和额外检查的时长分别保存；离线较小 cap 没有实际时延测量。

CPU `821379` 的两项 EOS 边界和真实小模型采样/greedy 前缀测试通过，输入准备 `821380` 完成。实际模型作业为 `821643`（Qwen-7B/H100）、`821644`（R1-7B/H100）、`821645`（Qwen-1.5B/L40S）、`821646`（Qwen-3B/L40S）。

自动建议取 cap hit≤5%、准确率相对 8K 降低不超过 1/32 的最小候选 cap；该建议不是尾部概率保证，也不是干预教师、微调学生或其他评测集的最终上限。完整 S3 仍需结合方法检查、后续覆盖与截断审计冻结实际候选生成和 SFT 配置。独立 SAE confirmation、SAE seed 稳定性和统一 B0–B7 学生实验仍须继续。


## 已完成检查与后续执行

截至 2026-09-13 15:27 EDT，三个 Qwen cap 作业均完成：2K/8K 正确数分别同为 27/32（7B）、21/32（1.5B）、24/32（3B）；2K cap hit 分别为 0、1、0。六次额外直接解码都与保存前缀一致，但这些实际样本没有超过 1K；不要把它们说成实际长输出截断的验证。R1 作业仍运行。

MATH 主教师 ASC 已由 `math_baseline_calibration.py` 接入，11 项测试和输入准备完成，396 个合格原参考进入新校准候选，另 4 个 gold 修正原参考排除。H100 校准 `821686` 在运行，后续 `821687/821688/821689` 对应 8 步检查、完整拟合和 64×5 开发评估。

SAE 两个 H200 剂量分片及对应消融已完成，两个 L40S 剂量分片仍运行。完整分析入口 `13_28_analyze_sae_generation.py` / `phase13_sae_generation_analysis_v1.json` 已冻结；三项审计测试通过。它要求全部八个完成分片和 v3 finalized review，按完整预测身份关联评分，重新审计 token/EOS、文本、区域、范数、题目支持与批次成本，再生成剂量曲线和特征/窗口对照图。配对 bootstrap 是开发集探索性区间，不是同时覆盖的多重检验，也不证明 student 效果。当前最终分析尚未执行。
