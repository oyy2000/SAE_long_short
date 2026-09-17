其实我们的论文最终目标就是想要做到，我们在已知较短的老师输出可以有更好的对于小模型sft效果的情况
  下，展示我们的SAE方法产生的输出要好于别的方法在Qwen 3B model上，所以我们的

# 项目管理：SAE 长度混淆消融与学生蒸馏

## 2026-09-16：用户授权扩大 KD，与八组 baseline 配对比较

用户新指令“KD把实验做完整，大规模数据和baseline对比”授权另立扩展协议；256 题小试仍为未通过效果门槛，不改写为成功。先推进 GSM8K 1,024 题八 baseline × SFT/KD × seeds17/42/73，共 48 个 adapter，Base 同设置重评；同时恢复 MATH 7,500 题来源池数据链。GSM 两个评测 cohort 已被观察，保持 exploratory；旧 B7 明确标为 Answer-associated SAE，清理后 F20319 因覆盖率门槛失败不进入本矩阵。

当前 GSM 配置 `phase13_token_kd_baselines_gsm1024_v2.json` 保持小试的 3 epochs、r8、0.5 CE + 0.5 T² KL、T2，并沿用已冻结的 baseline 选点。四路 H200 工作链必须先通过数据/词表、数值、最长输入 smoke 和显存/时长门槛，最后审计完整预测、配对曝光、哈希与统计。 v2 的 856443/856444 已完成 22 项测试与八组数据准备；856445 smoke 上次回读为 QOSGrpGRES，856446 等待其成功。v1 的 21 项测试通过，856423 准备因重复 exclusive-create B1 文件失败，856424/856425 自动取消，无训练；修复与回归测试在独立 v2，旧快照和故障保留。

按用户新要求，GPU 配额排队时先检查可立即执行且符合显存/时限的其他已批准卡型；仅无合适替代时排队，规则已写入 `AGENTS.md`。同科学设置的独立 L40S 路由 `phase13_token_kd_baselines_gsm1024_l40s_v1.json` 已通过 856509 测试、856510 准备、856511 最长输入 KD smoke（实测峰值预留 26,376 MiB、4 步，训练部分约 30 秒）和 856512 矩阵门槛；856592–856639 共 48 个训练作业已提交，Base 856640、分析 856641、独立审计登记 856516/审计 856644 均有依赖。首个 L40S 训练作业目前因 `QOSGrpGRES` 等待；H100 健康节点 gpu17 的四张卡均已分配、gpu16 不可用，H200 也无共享配额，因此当前排队是必要的。经再次核对三个作业均为本用户且仍 PENDING，已取消旧重复 H200 链 856445/856446/856461；[路由切换记录](results/phase13_baseline_expansion_v1/exploratory/token_kd_baselines_gsm1024_l40s_v1/protocol/route_switch_h200_to_l40s_v1.json)保留旧新作业 ID 与证据。48 个训练和效果结果仍未完成。

MATH 原 32 个 raw 分片齐备，但 822575 合并报 `Candidate grading differs`，824506 取消。新 `kd_math_source_preparation_v2` 提交 856437 候选归档 → 856438 stored/v2/reviewed 评分差异审计；归档核验 token/text、题目网格、批成本和哈希，保留旧评分但不认证其可复现性。后续需审核差异、完成 reviewed raw 与 DAP/TokenSkip、共同支持和长输入 smoke，尚未提交 MATH KD 训练。实际提交脚本使用指定 scratch，旧作业和证据保留。

详见[扩展协议与进度](docs/experiments/phase13/phase13_token_kd_baselines_expansion_zh.md)。这里记录实现和启动，不是完整 KD 效果结果。

MATH 原始评分审计后续：856438 对 53,848 条记录完成 stored/v2/reviewed 比较，216 条 correctness 改变，旧 v2 correctness 零差异；216 条逐条复核为 209 条新版识别正确、2 条确错、5 条保守解析漏判。292 条 reviewed 解析异常保留为不合格并公开分类。856471 对 53,784 条 student-pool 候选的全部原评分字段复算，零差异；不能据此证明旧 822575 失败的瞬时原因。受控原合并重试 856477 → reviewed raw 迁移 856480 → 文本压缩测试/准备 856484/856485 → 压缩 DAG 注册 856490 → 全八方法共同数据注册 856494 已提交成功依赖。未放行 MATH KD/GPU 学生训练或跨数据集结果。独立 GSM 矩阵末端审计注册 856461 等待 856446 放行矩阵成功。


## 2026-09-16：启动 Answer-free SAE 重训链

**修订状态：v2 按读回门槛结束。** v1 准备 853536 因代码把同题标签集合误判为必须恰好 `{short,long}` 而失败；原题还含其他候选标签。afterok 自动取消 853537/853538，无 SAE 产物；原冻结快照和故障记录保留。v2 修正为同题同时包含 short/long，3 项回归测试通过，独立冻结并提交 853543 准备 → 853544 CPU 样本 → 853545 H200 训练，均为 COMPLETED/0:0，训练完成标记的 6 项文件哈希核验通过。14,096 条父轨迹中保留 6,769 条配对 short/long 轨迹（train/dev/test 4,739/1,020/1,010），排除 169 条合格位置不足及 21 条无标题。109 个原激活 chunk 均核验；新样本 train/dev/test 为 250,000/50,000/50,000 token，各轨迹配额 train52–53、dev49–50、test49–50。TopK64 SAE 完成 1,500 步，最后 dev MSE 0.2264、解释方差 0.7733。四个评分分片 853657–853660 与分析 853661 均 COMPLETED/0:0，2,030 条轨迹、129,920 个固定正文 token 审计零缺失/重复。16 个候选中 4 个通过清理后门槛，dev 预定主特征为新字典 F20319；此处旧 dev/test 已被观察，只是探索性复现。另冻同状态读回协议：853699 冻结、853700–853702 三特征 L40S 读回、853703 汇总分析均 COMPLETED/0:0，五份实际 batch 与冻结指定 scratch 脚本逐字节一致。24,240 条逐状态记录零缺失/重复；F20319 的 test 激活覆盖仅 2.40%（要求≥5%），虽正负目标变化及非目标比值合格，整体失败。随机特征 7540/8589 各自读回通过，但 8589 的 dev 频率为主特征 2.44 倍，超过注册的两倍匹配范围。最终判定 `stop_target_or_control_engagement_failed`，教师生成与学生训练未触发。见[筛选结果](results/phase13_sae_feature_separation_v1/exploratory/answer_free_sae_v2/screen/feature_gate/selected_features.json)、[读回审计](results/phase13_sae_feature_separation_v1/exploratory/answer_free_sae_v2/screen/engagement_analysis_v1/engagement_analysis_manifest.json)与[停止记录](results/phase13_sae_feature_separation_v1/exploratory/answer_free_sae_v2/screen/final_decision.json)。

按用户要求，新分支先从已有 NCSU 第 17 层激活中剔除 `Answer:` 及后续 token，并排除正文末 32 token、格式 token，按轨迹平衡重采样后重训 TopK64 SAE。无标题或不足 64 合格 token 的轨迹按规则排除；旧 F24086/八特征编号不迁移。v1 失败记录保留；v2 SAE 与特征筛选已完成，F20319 双向同状态读回因 2.40% 覆盖率低于 5% 而停在生成前。若开启独立新协议，必须先在 dev 实际读回位置预筛目标覆盖率和随机对照两倍匹配，再以未观察新题验证；不能改写本轮门槛。见[协议](docs/experiments/phase13/phase13_sae_feature_separation_project_zh.md)与[冻结配置](results/phase13_sae_feature_separation_v1/exploratory/answer_free_sae_v2/protocol/frozen_config.json)。

## 2026-09-16：3B 配方对齐三条件 × 三种子完整完成

独立 `gsm8k_qwen3b_recipe_alignment_v1` 已完成九个 adapter、含 Base 的十个评测条件、40 个分片和 12,690 条预测；850206 分析与各登记训练/评测作业均 COMPLETED/0:0。独立复核 236 个标记、6,925 项哈希与逐条件准确率/长度/cap-hit，零缺失或重复。每个学生保留同一组 1,024 题，1 epoch、256 优化步；本轮为旧配方对齐 SFT，并非 KD。

| 条件 | seed17 | seed42 | seed73 | 平均准确率 | 平均输出 tokens |
| --- | --- | --- | --- | --- | --- |
| B1 无干预最短正确 | 83.29% | 82.74% | 83.22% | 83.08% | 277.68 |
| SAE 0.1 | 84.63% | 84.87% | 83.92% | 84.48% | 271.34 |
| SAE 0.3 | 82.74% | 83.14% | 82.51% | 82.79% | 250.86 |

SAE 0.1 相对 B1 三个 seed 均正增益，平均 +1.39 pp、输出缩短 2.29%；题目与训练 seed 交叉 bootstrap 95% CI [−0.11,+2.86] pp，四比较 Holm 调整 p=0.204，尚不能确认优于 B1。SAE 0.3 相对 B1 平均 −0.29 pp、输出缩短 9.66%，CI [−1.89,+1.39] pp。相同 cap/解码下 Base 为 81.56%；SAE 0.1 相对 Base +2.92 pp、调整 p=0.0128，但论文关心的相对 B1 优势仍未获统计支持。

评测为已观察过的 GSM8K test[50:1319] 共 1,269 题、greedy cap512、repetition penalty1.1，属于 exploratory 配方敏感性，不能与 cap1024 的原 pilot 或 KD 开发集准确率直接混比，也不构成新独立确认。见 [结果图](results/phase13_baseline_expansion_v1/exploratory/gsm8k_qwen3b_recipe_alignment_v1/analysis/student_accuracy.png)、[完整比较](results/phase13_baseline_expansion_v1/exploratory/gsm8k_qwen3b_recipe_alignment_v1/analysis/comparisons.json)、[独立复核](results/phase13_baseline_expansion_v1/exploratory/gsm8k_qwen3b_recipe_alignment_v1/verification/RESULTS_VERIFIED.json)。

## 2026-09-16：独立建立长短 SAE 特征分离与可解释性项目

用户要求把“long/short trace 有明显、可解释的 feature 差异”作为重要方向，并梳理已有消融。已建立[项目计划](docs/experiments/phase13/phase13_sae_feature_separation_project_zh.md)、[证据回顾](docs/experiments/phase13/phase13_sae_feature_evidence_review_20260916_zh.md)及[机器可读计划](configs/phase13_sae_feature_separation_v1.json)。仅 P0 文档立项完成，P1 输入/题目角色审计待执行；配置为 planning_not_frozen，本次未提交新作业。

NCSU 六 SAE 有 90 个原规则确认特征，主字典 short/long 各八个；八 short 的 Answer 激活质量为 52.77%–98.04%，正文 paired d 仅 -0.102 至 +0.129。300 题确认支持缩短，但未确立优于 Answer-format 或准确率非劣。新增 SAE seeds 的 short 数为 3/5/7，原八特征不可构造，top3 生成也未稳定缩短。历史 Phase 5 的 11 个清理候选值得恢复，但该轮停在随机对照覆盖率门槛，未进入生成/SFT。Phase 2/5/6 原结果目录当前本地缺失，本次仅据保留文档回顾；NCSU 标记及引用路径检查不等于全部哈希重审。

新路线先做同题自然长短、正文固定支持、格式反例，再决定 body-balanced 重训、跨 seed 与未观察题确认，最后进入特异因果和学生实验。特征可解释性、教师缩短与 Qwen 3B 学生收益分别验收；其他分支和不利结果保留。

## 2026-09-16：KD 256 题小试 v2 完整完成，未通过放大门槛

修正版 `token_kd_small_v2` 四组训练/开发评测 850219–850222、分析 850223 均 COMPLETED/0:0。每组 256 题训练、96 优化步、每题曝光 3 次，300 题开发预测；共 1,200 条预测零重复/缺失。独立复核 172 个完成/协议标记与 6,040 项哈希，重算准确率和长度一致，SFT/KD 监督 token 与曝光匹配，实际中间输出均在指定 scratch。

seed17：SFT 269/300（89.67%），KD 275/300（91.67%），+2.00 pp；seed42：两者均 269/300（89.67%），差 0。平均 +1.00 pp，按问题配对、条件于两个固定学生的 95% bootstrap CI 为 [−1.17,+3.33] pp。KD 平均输出长度分别为 286.65/286.51 tokens，对应 SFT 为 275.53/272.15；全部 cap-hit=0。长度门槛通过，但 seed42 未正增益且 CI 跨零，故不放大、不据此宣称 KD 稳定有效。当前没有提交本分支大规模实验。

该结论仅适用于已观察过的 GSM8K 开发集上的探索性 B1 小试，不是锁定测试、独立确认或 SAE 有效性结论。v1 仍作废保留。详见 [完整比较图](results/phase13_baseline_expansion_v1/exploratory/token_kd_small_v2/analysis/sft_vs_kd.png)、[门槛结果](results/phase13_baseline_expansion_v1/exploratory/token_kd_small_v2/analysis/decision.json)、[独立复核](results/phase13_baseline_expansion_v1/exploratory/token_kd_small_v2/verification/RESULTS_VERIFIED.json)。

## 2026-09-16：新增 256 题 token-KD 小试，先验证再扩大

**实现修订：当前以 token_kd_small_v2 为准。** 在查看任何开发准确率前发现 v1 的自定义损失忽略整组梯度累积 token 归一化，旧四次训练不能作为效果证据。已停止剩余 v1 评测/分析 850151–850155、保留产物并写入 IMPLEMENTATION_INVALID。修复后补充两个原生损失/累积梯度回归；相同科学设置重新冻结，提交 850209 测试 → 850210 准备 → 850211 smoke → 850212 小试门槛。四个实际 batch 与冻结 scratch 启动器逐字节一致。配置见 [v2](configs/phase13_token_kd_small_v2.json)。以下 v1 启动记录仅为历史，不是有效实验结论。

用户要求在现有 SFT 之外增加 KD，先小规模确认有效性。新增独立 `token_kd_small_v1`：固定 Qwen2.5-7B 教师与 3B 学生，抽取 B1 的 256 道既有共同训练题；seeds 17/42 各比较匹配的 SFT 与 `0.5 CE + 0.5 T² KL(teacher||student)`，T=2，共四次训练。两臂保持题目、文本、token、LoRA 与训练预算一致；本小试采用原 pilot 的 3-epoch/rank-8 配方，与另一条旧配方对齐复跑分开。教师冻结、completion-only causal mask、在线概率监督，不保存完整 logits。两 tokenizer 有效 token-ID 完全一致，KL 排除不对应实际 token 的填充输出维度并记录概率质量。

开发集为此前已观察的 300 题，greedy/1024 cap；不读 locked-test 预测作选点。放大需两 seed 均有正向准确率差、平均至少 +1 pp、按问题配对的 95% CI 下界大于零，且长度和 cap-hit 门槛通过。未过门槛不扩展，不据小试宣称 SAE 有效。当前仅提交 850144 数值测试 → 850145 数据准备 → 850146 H200 KD smoke → 850147 小试启动门槛，未提交完整规模。四个实际 Slurm batch 已回读确认指定 scratch 路径。

详见 [KD 小试协议](docs/experiments/phase13/phase13_token_kd_small_v1_zh.md)、[配置](configs/phase13_token_kd_small_v1.json)、[冻结源码与协议](results/phase13_baseline_expansion_v1/exploratory/token_kd_small_v1/protocol/FROZEN.json)。本条是实现和启动登记，尚非 KD 效果证据。

启动验收：850144 的 17 项检查、850145 数据/词表准备、850146 H200 smoke、850147 小试准入均 COMPLETED / 0:0。主代理复核 586 项输入/源码绑定；smoke 四步成功、teacher_frozen=true、峰值 allocated/reserved 为 22,939/24,380 MiB。850151/850152 为 SFT/KD seed17，850153/850154 为 seed42，850155 等待全部预测做配对分析。五个后续实际批处理脚本与冻结 scratch 启动器一致。见 [smoke 与提交核验](results/phase13_baseline_expansion_v1/exploratory/token_kd_small_v1/verification/SMOKE_AND_SUBMISSIONS_VERIFIED.json)。GPU smoke 不等于 KD 已有效，放大仍需完整效果门槛。

## 2026-09-16 03:42 EDT：3B 对齐旧 1.5B 配方，首组通过、剩余矩阵已提交

用户确认保留当前 1,024 题和教师 trace。教师仍为同 revision 的 Qwen2.5-7B-Instruct；3B 改用旧配方：1 epoch、batch 4 / accumulation 1、LR 2e-5、linear / warmup 0.03、LoRA r4 / alpha16 / dropout0.05 / all-linear、训练上限2048；greedy评测batch32、cap512、repetition penalty1.1。最后一项来自旧1.5B实际模型默认值，由独立execution_v2显式绑定。当前题集、提示词、显式completion labels和判分器保留，不能称为旧878题的严格复现。

首阶段固定 B1、B7 rho0.1、B7 rho0.3，各 seeds17/42/73，共9个adapter；Base按本轮解码设置重评。850139准备完成，26项检查通过，三组数据与父实验逐字节一致。850141训练完成256步/1epoch/每题曝光1次，850148完成修订后318题评测分片。旧评测850142与释放850143/850149已取消、产物保留；旧控制器850140正常退出。

测量释放850158完成，另3项控制回归检查通过；剩余8次训练、39个评测分片及最终分析850206已提交为850159–850206。首组reserved显存37920MiB，加8192MiB余量后不安排L40S，后续使用H200。当前四个就绪作业等待QOSGrpGRES，其余等待成功依赖；其他kd_pilot作业保留。850150为本分支恢复控制器。本轮尚未完成；原pilot结果和独立MATH作业824506保留。

详见[复跑协议](docs/experiments/phase13/phase13_gsm8k_qwen3b_recipe_alignment_v1.md)及[生效冻结配置](results/phase13_baseline_expansion_v1/exploratory/gsm8k_qwen3b_recipe_alignment_v1/execution_v2/protocol/frozen_config.json)。51个修订提交的实际batch与冻结启动器一致；已核验真实Trainer output_dir和评测scratch位于管理员指定目录。

## 2026-09-16：GSM8K Qwen3B pilot 完整训练、锁定评测与独立审计完成

首轮 2,048 题来源提供十四条件 1,936 题共同正确支持，固定抽取同一组 1,024 题完成 14 次 seed-17 SFT；每次 3 epochs、384 优化步、每题曝光 3 次。300 题开发集冻结强度/比例和公共 1,024-token cap。全部 14 个训练条件加 base 在 GSM8K test[50:1319] 完成 1,269 题、60 分片、19,035 条逐题预测。分析 848972 和独立审计 849972 均 COMPLETED / 0:0；独立复核 254 个标记、5,037 项哈希绑定，零重复/缺失，重算指标一致。当前无活跃 pilot 作业；旧 heartbeat 的运行状态不是现状。

主方法按开发集预先选定 B3/B4/B7 强度 0.1、TokenSkip 比例 1.0。锁定测试：base 86.84%，B0 85.82%，B1 86.05%，B2 82.51%，B3 85.19%，B4 85.19%，B5 83.77%，B6 84.71%，B7 SAE 85.19%。SAE 教师监督相对 B0 缩短 15.06%，学生输出相对 B1 缩短 4.84%，但准确率相对 B1 为 −0.87 pp，题目配对 95% CI [−2.52,+0.79]；七项 SAE–baseline 检验经 Holm 校正均未达到 0.05。不能宣称 SAE 已优于主要基线、等价或非劣。

开发收益门槛未通过：SAE 88.67%，B0/B1 90.00%，最强竞争 Dense 91.67%；按协议不触发额外 seed 42/73。当前阶段为“单种子 GSM8K pilot 闭环已完成，进入结果解释与方案修订”，不是多种子/MATH/跨数据集主实验完成。下一步先诊断监督质量、错误类型及 base→SFT 差异，再冻结独立确认方案；本轮锁定测试已观察，不据此继续调参并称为独立验证。未提交新 GPU 实验。

见 [中文总结](results/phase13_baseline_expansion_v1/exploratory/gsm8k_qwen3b_sae_pilot_v1/final_review_v1/report_zh.md)、[原实验完成标记](results/phase13_baseline_expansion_v1/exploratory/gsm8k_qwen3b_sae_pilot_v1/EXPERIMENT_COMPLETE.json)、[独立审计](results/phase13_baseline_expansion_v1/exploratory/gsm8k_qwen3b_sae_pilot_v1/final_review_v1/COMPLETE.json)。所有结论保持 exploratory / GSM8K-only / 单训练 seed 边界。MATH 独立 raw-impact 审计 824506 仍暂停，后续需另行恢复。

## 2026-09-16 00:34 EDT：取消 99 个已被修订协议淘汰的旧排队作业

按用户要求清理旧提交。逐项复核当前 owner、PENDING/priority=0、旧冻结启动器及依赖后，取消 822579、822752–822780、823702–823770，共 99 个旧 MATH steering、DAP/TokenSkip 和 SFT 准备作业。它们属于旧参考/来源协议，已由 reviewed 协议淘汰；清理提交不代表放弃 MATH 后续研究。取消限定当前用户和 PENDING 状态，命令成功；独立只读 sacct 回读确认 99/99 为 CANCELLED。

独立 reviewed raw-impact 审计 824506 仍保持 JobHeldAdmin；GSM execution_v7 作业、恢复控制器与其成功依赖保留。回读时 848155、848157、848158、848160 及控制器 848124 运行中，848159/848161 等待自身依赖。全部历史配置、日志、预测、checkpoint 保留，未删除 scratch。见 [取消清单与终态核验](results/operations/20260916_obsolete_queue_cleanup_043251/COMPLETE.json)；此完成标记仅指队列清理。

## 2026-09-15 23:55 EDT：指定 scratch 迁移完成测试，GSM pilot 从 SFT 准备续跑

按用户要求建立项目 `tmp -> /share/jekml/youyang7/tmp`。冻结 execution_v7：作业独立 scratch、TMPDIR/TMP/TEMP、HF dataset/编译缓存及显式 Trainer output_dir 统一使用管理员指定位置，删除项目目录 fallback；逐作业检查实际路径、容量和 GPFS Share01 用户/组配额。保留现有模型缓存和 ABI preload，训练/生成参数不变。旧 execution_v6 和历史 held MATH 作业均未修改或释放。

修复 842496 的 `IndexError: no such group`：答案区域诊断兼容无命名分组的 pilot 正则，历史具名分组语义保持。复用并重核已完成共同支持审计，失败的两份 B0 文件归档至 `execution_v7/preserved_failed_842496`。848103 在计算节点通过 56 项测试，实际临时 adapter 写入指定 scratch；测试标记和 569 项新源码/协议绑定已核验。

848123 为新的 support/SFT 准备，848124 为新恢复控制器，仅授权 execution_v7 作业；原 STOP 已归档，842496 登记为 superseded。已回读这两个作业及测试的 Slurm-spooled batch，确认与冻结新启动器一致，实际提交不再引用旧启动器。此处为恢复提交，尚未确认学生训练完成。

证据：[冻结修订](results/phase13_baseline_expansion_v1/exploratory/gsm8k_qwen3b_sae_pilot_v1/execution_v7/protocol/FROZEN.json)、[测试完成](results/phase13_baseline_expansion_v1/exploratory/gsm8k_qwen3b_sae_pilot_v1/execution_v7/protocol/TEST_COMPLETE.json)、[续跑提交](results/phase13_baseline_expansion_v1/exploratory/gsm8k_qwen3b_sae_pilot_v1/execution_v7/verification/RESUME_SUBMITTED.json)、[实际批处理脚本核验](results/phase13_baseline_expansion_v1/exploratory/gsm8k_qwen3b_sae_pilot_v1/execution_v7/verification/resume_batch_audit.json)。

续跑验收：848123 已 COMPLETED / 0:0，用时 1 分 47 秒，完整生成 14 条件 × 1,024 道共同训练题的数据；45 项绑定哈希、唯一问题 ID 和所有条件问题顺序已复核。848146–848160 为 base 开发评估加 14 个 seed-17 学生任务，848161 依赖全部首阶段任务做开发选点；15 个实际提交脚本逐一与 execution_v7 核对一致。848124 控制器运行中。详见 [SFT 数据及提交核验](results/phase13_baseline_expansion_v1/exploratory/gsm8k_qwen3b_sae_pilot_v1/execution_v7/verification/SFT_READY_VERIFIED.json)；此验收不代表学生训练或最终评测完成。

23:57 EDT 实际运行确认：848146–848149 已在 gpu38 的 H200 上运行，B0/B1/B2 日志已有训练 loss，约推进至 epoch 0.17。主代理进一步核验三个真实 run_config 的 Trainer output_dir 和逐阶段 storage admission，均位于对应作业的指定 scratch；恢复 heartbeat 已由新控制器 848124 更新。见 [实际路径与训练启动核验](results/phase13_baseline_expansion_v1/exploratory/gsm8k_qwen3b_sae_pilot_v1/execution_v7/verification/RUNTIME_MIGRATION_VERIFIED.json)。剩余条件按四条 lane 的成功依赖继续，尚无最终学生比较结论。

## 2026-09-15：管理员报告 gpu18 系统盘写满，修正临时存储并暂停旧路径续跑

管理员报告 SAE 作业写入 `/var/tmp` 导致 gpu18 的 `/var` 空间耗尽、节点不可用。只读核查确认 823088/823089 均在保存 SAE checkpoint 时报 `No space left on device`；旧 frozen launcher 硬编码 TMPDIR，训练沿用该路径。将系统临时目录作为实验 scratch 的运行策略有误。

当前源码中的 13 个 NCSU Slurm 入口已改为管理员指定的 `/share/jekml/youyang7/tmp` 下作业独立目录（用户名由 `id -un` 获取），统一 TMPDIR/TMP/TEMP。13 项 shell 语法及环境路径检查和指定 scratch 的微量写入检查通过；更新 AGENTS.md 及当前运行文档。历史冻结快照、实验协议、预测和 checkpoint 保留原状。本次未访问计算节点清理残留，不能声称 gpu18 已恢复。

初次核查仍有旧 execution_v6 恢复控制器 847156 运行、旧冻结入口作业排队。已通过 recovery/STOP 请求停止自动续提，并对当前用户项目内仍引用 `/var/tmp` 的排队作业执行可逆 hold，保留原有 hold 和依赖，不取消作业。恢复前须生成单独冻结的 scratch 修订并核验实际提交脚本；源码修复不等于旧作业已迁移，也不代表实验完成。最终逐项状态见 [处置记录](results/operations/20260915_scratch_incident/containment.json) 和 [只读审计](results/operations/20260915_scratch_incident/audit.md)。 后续核验：847156 于 23:07:09 正常退出（COMPLETED/0:0），未发现后继控制器；100 个旧路径排队作业均回读确认 hold，其中 71 个为本次新增暂停。见 [停止核验](results/operations/20260915_scratch_incident/followup_20260915.md)。

## 2026-09-15 10:07 EDT：TokenSkip ABI 修复复核通过，压缩流水线持续推进

本次重新检查确认已有 execution_v6 修复生效：冻结源码、协议、原生库及 smoke 产物共 593 项哈希/一致性检查通过。842428 的 51 项测试、842429 的 4/4 条 H200 TokenSkip smoke 和 842430 smoke 合并均为 COMPLETED / 0:0。未改动包版本或冻结快照；补充了 [环境故障与修复说明](docs/environment/ncsu_sft_environment.md)。

14:07 UTC 只读调度快照：DAP 842431–842442 已在调度器完成，842443–842446 在 gpu38 运行；TokenSkip 完整分片 842463–842494 等待依赖，完整合并 842495、共同支持 842496 同样等待依赖。控制器 842497 在 c207n01 运行，heartbeat 更新至 14:07:20 UTC。旧 840483 的 NEEDS_ATTENTION 已有 RESOLUTION 记录，属于已处理的历史故障；本次未发现新的 ABI 阻塞。当前等待 DAP 压缩链推进，尚未完成全部压缩、学生 SFT 或评测。

证据：[593 项复核](results/phase13_baseline_expansion_v1/exploratory/gsm8k_qwen3b_sae_pilot_v1/execution_v6/verification/recheck_20260915T140728Z/hash_audit.json)、[当前作业状态快照](results/phase13_baseline_expansion_v1/exploratory/gsm8k_qwen3b_sae_pilot_v1/execution_v6/verification/recheck_20260915T140728Z/status_snapshot.json)。分片调度完成及部分完成标记不代表整个实验审计完成。

## 2026-09-15 09:48 EDT：TokenSkip 原生库 ABI 故障修复，恢复完整压缩阶段

本次延误不是持续计算：steering全部32分片及合并已完成，DAP smoke840408完成15条；TokenSkip840409于03:42因NLTK→sqlite3导入加载节点旧libstdc++，缺少CXXABI_1.3.15而失败，840478/840480/840483三次重试同错后达到上限，阻塞约6小时。旧报告中的CPU controller RUNNING不能当作GPU实验运行。

冻结execution_v6，通过启动器opt-in LD_PRELOAD现有sft环境的libstdc++.so.6；绑定该库及libgcc哈希，未安装/升级包，未修改旧执行快照。登录节点已验证torch→sqlite3/nltk/llmlingua导入成功；新增计算节点回归用例。ImportError明确列为确定性依赖错误，不再盲目消耗三次GPU重试。

新链842428回归 → 842429 H200 TokenSkip smoke → 842430 smoke合并 → 842431–842494 DAP/TokenSkip各32分片（4路H200） → 842495完整合并 → 842496共同支持/SFT。新控制器842497继续只读Luna之外的自动恢复，使用独立abi_recovery前缀续期，避免旧controller周期命名冲突。旧840410–840476全为本用户未运行pending，逐项保存superseded后取消；已完成教师轨迹、DAP smoke保留，失败TokenSkip目录归档，840483警报有明确RESOLUTION。主结果仍未完成SFT或收益比较。

## 2026-09-15 00:00 EDT：steering 改为32分片、四路混合GPU，并启用Slurm自动恢复

用户明确要求根据可用GPU尽快恢复、失败重试。原steering smoke 832680完成，832681根据16分片推算9,642秒超过short_gpu 2小时限制，阻塞下游。现冻结 [execution_v5](results/phase13_baseline_expansion_v1/exploratory/gsm8k_qwen3b_sae_pilot_v1/execution_v5/protocol/frozen_config.json)：新steering输出根 `round_1/steered_split32`，每片64题，保留题目/四候选/种子/强度；原raw完整结果和steering smoke复用。4路轮转H200、H200、L40S、L40S；当前H200节点idle、L40S mixed，H100无idle。H200真实吞吐待新任务测量，不据此声称已测得更快。

839091回归 → 839092新smoke门槛 → 839093–839124共32个GPU分片 → 839125合并 → 839126压缩入口。旧832682–832699全为当前用户未运行的PENDING，已逐项superseded后取消；保留旧失败记录。

839127为Slurm CPU恢复控制器，每分钟检查本execution的终态GPU失败，最多额外3次重试。先保留不完整输出，再单次提交，按实时Slurm依赖重接后续；提交不确定时不盲重提，CANCELLED不重试，已完成adapter有hash标记则复用。TIMEOUT使用登记的H100四小时时限重试。只对generate/compress/train-dev/train/evaluate-dev/evaluate-test阶段自动处理；CPU、未知阶段、已有完成标记但状态异常、达到重试上限的情况写NEEDS_ATTENTION，不自动改变协议。每55分钟安排下一控制器周期（上限144），采用1CPU/2GiB；Luna/medium每小时在控制器中只读审阅。4项新增单元检查通过；模拟集成验证失败归档、一次重提、依赖更新及重复扫描不重复提交。启动验收已确认839091的48项测试、839092的32分片smoke门槛完成；主代理验证源码/测试/raw复用/新smoke标记，保存execution_v5/verification/STARTUP_VERIFIED.json。839127恢复控制器已在c207n01运行并写出heartbeat；待其Luna调用成功再停止旧monitor v3，避免监控空窗。

## 2026-09-14 09:08 EDT：GSM8K 完整新题比较恢复，移除统一本地 scratch 门槛

用户确认保持新题池、四候选、B0–B7 全比较。根据更新后的 AGENTS.md，新增按实际阶段写盘估算与作业独立共享存储回退，冻结 [execution_v4](results/phase13_baseline_expansion_v1/exploratory/gsm8k_qwen3b_sae_pilot_v1/execution_v4/protocol/frozen_config.json)。不再以固定4/8 GiB本地空间拒绝GPU；保留显存admission及真实容量检查，训练预算由encoded输入和LoRA尺寸计算。方向提取复用已完成校准/ASC与100对输入；未修改旧协议或重跑已完成上游。

新增5项存储检查通过，完整回归830248、恢复方向830249、raw begin830250已登记afterok。旧失败830075保留，旧阻塞830076登记superseded后取消。恢复验收：830248 COMPLETED/0:0，44项全部通过；830249已处理100/100对并发布方向完成标记，主代理已校验输出哈希，见 execution_v4/verification/RECOVERY_COMPLETE.json。该任务临时空间预算33,554,432 bytes，输出预算40,147,778 bytes，实际本地空闲6,025,125,888 bytes；按工作量admission成功。候选/SFT尚未完成。监控v3在login04以PID4174071启动，luna/medium每小时，新增hostname、heartbeat、异常记录；v2写STOP。旧v2无hostname，仅在当前节点查不到PID不能证明跨节点进程已死，纠正此前过强表述。

## 2026-09-14 07:42 EDT：GSM8K 小试按实测本地盘容量修订 admission

第二个 H200 任务 830070 同样在模型加载前被原 8 GiB 门槛拦截；记录表明 gpu39 的 `/var` 总容量仅 8,522,825,728 bytes，空闲 7,263,576,064 bytes，不能达到该统一门槛。资源配置单独冻结为 [execution_v3](results/phase13_baseline_expansion_v1/exploratory/gsm8k_qwen3b_sae_pilot_v1/execution_v3/protocol/frozen_config.json)：基础预留 4 GiB，SFT 保留按实际 encoded 输入四份加 1 GiB 的动态门槛；训练关闭周期性 checkpoint。复用通过 39 项检查的 execution_v2 源码，科学设置与已冻结校准输入不变。

新链为 calibration 830071 → ASC smoke 830072 → ASC fit 830073 → directions prepare 830074 → directions extract 830075 → raw begin 830076。五个旧的未运行下游任务已逐项保存 superseded 记录后取消；两次 scratch 失败完整保留。小时监控 v2 继续覆盖主结果根所有新提交。启动验收确认 830071 在 gpu38 RUNNING，已通过两次 GPU admission 并完成四个 Qwen checkpoint shards 加载，未再出现 scratch 错误；校准尚无完成标记，学生训练与比较结果仍未完成。

## 2026-09-14 07:36 EDT：GSM8K Qwen3B SAE 小试已实现并启动独立执行链

按用户要求，当前新增小试回到 GSM8K，Qwen2.5-7B 教师、Qwen2.5-3B 学生；B0–B7 共 14 个 seed-17 SFT 条件，先以相同问题支持比较学生收益，SAE 仅要求比 B0 教师监督更短。目标共同训练题数 1,024，补 seed 42/73 由预注册开发收益门槛决定。MATH 扩展和历史产物不据此改写。详见 [执行协议](docs/experiments/phase13/phase13_gsm8k_qwen3b_sae_pilot_v1_zh.md)。

输入冻结完成：eligible 5,583、首轮来源 2,048、预留至 3,072、开发 300、locked test 1,269；共同训练支持尚未建立。830038 通过最初 38 项回归；830039 完成输入准备。GPU 启动前修复正文长度诊断正则，保留原冻结输入与代码，另建 `execution_v2`；830043 已通过 39 项检查，830044 已成功启动校准阶段。当前执行入口是 [execution_v2 frozen config](results/phase13_baseline_expansion_v1/exploratory/gsm8k_qwen3b_sae_pilot_v1/execution_v2/protocol/frozen_config.json)，后续任务使用其中冻结代码。不能把启动成功记为完成训练或方法收益证据。

CPU 经现有 association 核验采用 compute_partners/short；只替换本小试尚未运行的 bootstrap 830033–830035 与 held begin 830040，取消与 superseded 记录保留。GPU 生成/训练/评测采用已授权 L40S，ASC 采用 H200，最多两路 GPU，下游受 afterok、完整性、显存和 smoke 吞吐门槛控制。

校准首个任务 830046 在 gpu38 因 `/var/tmp` 空闲不足注册的 8 GiB 门槛失败，尚未加载模型。失败日志保留；830070 使用相同 execution_v2 配置重试，增加磁盘字节记录并临时排除 gpu38。ASC smoke 830047 的依赖已回读更新为 830070，后续 fit 830048 顺序不变；两个尚未运行的 ASC 任务同样排除该磁盘不足节点。方向准备/提取和 raw 启动为 830049–830051，仍须等待校准成功。修改记录位于 `launch/calibration_generate_retry1`，未放宽资源门槛。

每小时只读监控已由独立元数据进程启动，PID 1438785；固定 gpt-5.6-luna、medium，每 3,600 秒一次，首次 2026-09-14 12:23:35 UTC / 08:23:35 EDT。小模型实际调用 smoke 已成功；报告写入 [monitor v2](results/phase13_baseline_expansion_v1/preparation/gsm8k_qwen3b_pilot_monitor_v2)，不自动改动作业，不声称推送聊天。原 monitor v1 已写 STOP。进程不能保证主机重启后恢复；命令与 PID 已落盘。

## 2026-09-13 23:16 EDT：修订开发选点完成，新学生池生成已排队

完整开发分片/合并 `824484`–`824492`、分析 `824527`、学生准备 `824529` 均已完成。3,072 条开发候选按原规则选中 B3 rho=0.2、ASC scale=0.25、SAE rho=0.3；对应层为 17/16/17。各方法正确支持 56/64，八方法开发交集 54/64，不能记为学生训练题数。完整合并/分析/学生协议/源码的 35/26/26/531 项绑定通过。

新 H100 smoke `824669`、合并 `824670`、32 个分片 `824671`–`824702` 和完整合并 `824703` 已提交，最多四路学生生成，预期 6,723 × 3 × 4 = 80,676 条候选。35 个当前用户作业最后均为 PENDING，全部依赖逐项匹配；smoke 等待资源。见 [75 项提交绑定与依赖回读](results/phase13_baseline_expansion_v1/preparation/unified_math_candidates_steered_reviewed_launch_v1/generation_chain_v1/verification_v1/COMPLETE.json)。初次未看到分析输出后继续回读原作业，稍后完整文件与标记通过，没有重提已完成分析。

原 raw 八个完整分片共 13,464 条，08–11 仍运行；全池评分影响 `824506` 等待原合并 `822575`。修订文本/SFT 配置已接到新的完整来源路径，保持原科学设置，尚未放行新 SFT。旧方向/文本 30 个 hold 均保留。本轮其余已完成产物见 [23 项进展绑定](results/phase13_baseline_expansion_v1/preparation/math_reference_integrity_gate_v1/reviewed_continuation_progress_v2/COMPLETE.json)；完整执行顺序见 [修订续跑](docs/experiments/phase13/phase13_reviewed_baseline_continuation_v1.md)。目标仍包括全部学生 baseline、25K 和泛化/预算矩阵。

## 2026-09-13 23:02 EDT：原生 ASC 修订评测、学生接口及来源抽样审阅完成

原生 ASC 五强度 `824535` 与分析 `824544` COMPLETED / 0:0，分别 34 分 11 秒与 14 秒；完整 64 × 5 网格 320 条、生成 5 项和分析 116 项直接绑定复核通过。无干预为 52/64、658.48 tokens；scale 0.5 为 49/64、623.36 tokens；scale 1.0 为 36/64、557.14 tokens。0.1/1.0 各一条 cap hit，其余零。该开发检查没有证明保准确率的压缩优势；与统一四候选扫描分开报告，不据此另调学生选点。见 [完整曲线和统计](results/phase13_baseline_expansion_v1/exploratory/asc_qwen_math_reviewed_analysis_v1/report_zh.md) 与 [回读证据](results/phase13_baseline_expansion_v1/exploratory/asc_qwen_math_reviewed_analysis_v1/verification_v1/COMPLETE.json)。

学生修订提示 `824587` 完成 55,524 输入核验，只有 `math500-00296` 的两学生 × 七提示共 14 条改变。统一评测 CPU 检查 `824596` 通过 10 项；L40S `824597/824598` 完成两个学生的 base/合成 LoRA 接口，共 60 条合成输出，实际 penalty 均为 1.0。256-token smoke 的 7/4 次截断保留，不确定正式 cap。见 [评测接口及边界](docs/experiments/phase13/phase13_student_evaluation_interface_v1.md)。

预先抽定的 32 个 OpenThoughts 质量样本已完整读题/参考并绑定保存，10 个最终答案可支持但推导有明显缺陷，另四个有题意/操作数 caveat。有限枚举与有理数检查已保存；助手审阅不是独立人工证明标注，未覆盖原 gold 或放行 25K。见 [来源审阅](docs/experiments/phase13/phase13_openthoughts_reference_audit_v1.md)。

修订 steering 开发仅余分片 `824491` 仍在运行；合并 `824492`、选点 `824527` 与新学生准备 `824529` 尚待成功。原 raw 学生分片 08–11 继续运行，完整 raw 审计仍依赖原合并；旧方向待运行 DAG 保持 hold。尚无正式统一学生 SFT，全部 B0–B7、25K 和泛化/预算矩阵仍未完成。本轮完成实际结果核验和来源审阅，继续保持原目标。

## 2026-09-13 22:28 EDT：修订 steering smoke 已完成，完整开发分片启动

`824482/824483` COMPLETED / 0:0，384 条 smoke 候选完成生成、token 回放和合并；首批完整开发分片 `824484`–`824487` 已开始按资源运行，完整扫描和新选点尚未完成。原生 ASC smoke `824534` 完成（1 分 46 秒），完整五强度生成 `824535` 正在 H200 运行；新分析器用共享修订评分，两项检查 `824543` 通过，分析 `824544` 等待完整生成。

原 raw shard07 `822550` 完成 1,680 条候选，旧 steering shard03 `822751` 完成 2,532 条并作为旧方向证据保留。30 个旧方向/合并/文本待运行作业最后回读均为当前用户的 PENDING / JobHeldUser，没有解除或取消。新阶段直接绑定与六个完成作业状态见 [生成进展](results/phase13_baseline_expansion_v1/preparation/math_reference_integrity_gate_v1/reviewed_continuation_progress_v1/COMPLETE.json)，全部依赖与剩余步骤见 [续跑协议](docs/experiments/phase13/phase13_reviewed_baseline_continuation_v1.md)。没有新正式学生 SFT，目标继续进行。

## 2026-09-13 22:21 EDT：修订 ASC/dense 已拟合，DAP/TokenSkip 开发重跑完成

校准重建 `824462` 完成：395 题来源池、117 题/180 个必要候选、100 对；07414/0 替换 06821/0，184 个原始尝试全部保留。ASC smoke/full `824469/824470` 和 dense 准备/提取 `824471/824473` 完成。ASC 全部 3,000 步与 3,584 参数 checkpoint 一致，H200 拟合 571.58 秒，64 条保留通用文本平均 KL 0.018417；新旧方向 cosine 0.925484。dense cosine 0.999938，SAE/三个随机方向数值不变。KL 仍是 soft penalty，不是逐输入保证。

raw 开发迁移 `824474` 完成 576 条 smoke/development 记录，零正确性标签变化。新 DAP/TokenSkip 49 项检查与完整运行 `824496`–`824505` 全部 COMPLETED / 0:0，分片/合并直接绑定通过。DAP 194 条最终答案全部正确、零 cap hit，平均 547.40→479.12 tokens（缩短 12.47%），194 条文本与旧运行完全相同；TokenSkip 六比例 336 条、56 个预分配比例例子。两者均为已筛选正确源的开发检查，不是独立全池教师正确率或学生结果。学生来源接口 `824521` 通过 17 项检查及两个真实重评分视图验证；H100 raw 上下文 `824526` 完成 64 条 smoke 重评分，未完成全学生池选择。见 [续跑协议与证据](docs/experiments/phase13/phase13_reviewed_baseline_continuation_v1.md)。

统一 steering smoke `824482` 当前在 gpu38 RUNNING；其合并 `824483`、八分片 `824484`–`824491`、完整合并 `824492` 已登记，开发阶段最多四条并发通道。新分析 `824527` 等待完整曲线与压缩合并；学生 steering 测试 `824528` 通过 18 项检查，准备 `824529` 等待新选点。原生采样 ASC smoke/full `824534/824535` 另行排队更新方法检查。完整 raw 评分审计 `824506` 等待原合并 `822575`，没有释放旧 hold 或提交新正式学生 SFT。

本轮完成实际拟合、重跑、来源接口修复及 21 个完成作业的证据复核；[执行复核](results/phase13_baseline_expansion_v1/preparation/math_reference_integrity_gate_v1/calibration_rebuild_execution_v1/COMPLETE.json) 含 23 项直接绑定。下一步完整开发选点与新学生生成、全 raw 变化输出复核及下游迁移。7.5K→25K、跨模型/seed/预算矩阵和全部 B0–B7 学生比较继续保持原目标，未记作完成。

## 2026-09-13 21:43 EDT：完整 MATH cohort 修订完成，真实预测审计发现 ASC 校准入选变化

最后 128 个多答案请求来源已复核，v7 `824314` COMPLETED / 0:0：633 个显式定义、1,509 个检查，全部扫描标记 629/629 均覆盖，另 4 个早期锚点。完整 cohort 初次转换 `824336` 发现 15 个原扫描未标记的来源案例并以 FAILED / 1:0 停止放行；失败目录、10 个失败探针和全文诊断保留。逐题补充复核后，显式定义累计 648 个，补齐遗漏解集、有序长度、选项案例、千分位空格、TeX 单 token 分数/phantom 与显式舍入单位。两个原参考的空 box 对应已有修正计数 0，原文继续保留。

v8 定义验证 `824345`（5 秒）通过 648 定义的 1,555 个检查和 35 项测试；完整 cohort v2 `824410`（14 秒）通过 7,987 题、8,894 个检查，零新增 case、零构造错误。648 个定义由项目助手读题/参考复核，7,339 个由原 typed metadata 自动转换，不能称为全池独立证明审查。原 question/answer/role/near-component 等字段逐项不变，新增 `reviewed_answer` 和来源说明；其他四评测文件原样复制。六个登记的源全文解析局限为 `04253/05341/05343/06922/07225/07321`。v8 启动/结果 122/34 项、cohort v2 启动/诊断/完成 116/30/31 项直接绑定通过。

真实输入/输出影响审计 `824411` COMPLETED / 0:0（10 秒），37 项测试，覆盖 5,018 条记录：184 个 ASC 校准尝试、ASC/dense 各 100 对长短输入、320 个原采样 ASC 开发输出、512 个 raw 开发候选、3,072 个 steering 候选、194 个 DAP 改写及 336 个 TokenSkip 比例输出。全部旧 v2 重算标签与存储标签一致；新评分改变 6 条，全文均已读完并保存最终答案复核，不认证中间推理。原 100 对拟合输入全部仍正确，raw/DAP/TokenSkip 开发最终答案标签不变；124 条仍为解析错误的输出另行保存，不能因无标签变化宣称它们都经人工核验。启动/结果 123/34 项、六例及选择影响的 14 项绑定已核验。

关键结果：`math_train-07414` 的四个校准候选都答全了 -3 与 3/2，且满足更长/非截断/上下文条件。按原首次合格、稳定题序、100 对上限重选，第 0 候选进入校准集，替换末尾 `math_train-06821`；原 source pool 中 `04253` 的全文格式不能满足任选一个的新定义，但其排序位于截点之后。此前“问题未入选旧 100 对”不足以支持方向不受影响。下一步重建完整来源/候选选择协议，重新拟合 ASC-CES 和 dense 方向，重跑受影响开发条件与选点，再接入全池新评分及下游压缩/SFT。

已确认当前用户所有的 28 个旧方向待运行 steering 分片 `822752`–`822779` 及合并 `822780` 全部 user hold，29 个 PENDING / JobHeldUser 回读通过。hold 后结果列表写入因保存接口要求 dict 而失败；未重复调度操作，原不完整文件保留，恢复记录以权威 Slurm 回读验证全部 hold。原文本准备 `822579` 继续 PENDING / Priority=0 / JobHeldUser。运行中的 `822751` steering 和 `822547/822548/822550` raw 保留；`822749` 已 COMPLETED / 0:0，新增 2,532 条历史 steering 候选、5 项绑定通过。尚未重新拟合或释放 SFT；完整 baseline、25K、跨模型/预算矩阵继续进行。本轮属于已执行修订、验证、实际影响发现与调度调整的进展。

详见 [参考修订与下一步骤](docs/experiments/phase13/phase13_math_reference_integrity_v1.md)；当前 cohort 为 `preparation/reviewed_math_cohort_v2`，实际评分为 `preparation/reviewed_math_prediction_impact_v1`，选择复核和暂停证据分别在 `math_reference_integrity_gate_v1/prediction_impact_review_v1` 与 `steered_calibration_revision_hold_v1`。

## 2026-09-13 21:15 EDT：累计 505 个参考定义验证通过，剩余 128 个标记案例

本次新完成 101 个剩余文字参考和 100 个多答案请求来源的逐题复核，分别保存于 `text_context_review_part02_v1` 与 `multiple_request_review_part01_v1`。前者含 25 个单位数量、27 个文字目标、11 个完整列表、37 个选项/角度和一个精确到秒的时刻；后者含 24 个完整列表、21 个区间/并集和 55 个单目标。原 58 个百分比上下文、38 个早期决定的显式 schema 转换、首批 40 个文字/单位案例也已纳入执行记录；38 个转换不是新增问题。累计 505 个唯一问题，其中扫描标记 501/629，另 4 个早期跨源锚点未被扫描标记。剩余 128 个仅为多答案请求：114 个学生池、14 个锁定 MATH-500。见 [来源绑定进度](results/phase13_baseline_expansion_v1/preparation/math_reference_integrity_gate_v1/review_progress_v3.json)。

单位检查已确认旧数值比较会误接受 `13.5 kg` 对应 `13.5 pounds`。独立 grader 现登记允许单位，拒绝错单位；保留无单位的正确数值和明确允许的写法。追加文字列表完整性/顺序、A–Z 选项、显式拼写别名和秒级时刻；只有复核定义登记 pm 上下文的题才接受省略 pm 的十二小时答案，显式 am 不被覆盖。所有来源由项目助手进行目标/类型复核，不是独立证明审查；Asymptote 仅按文本阅读。两个多值函数记号和球坐标约定等来源 caveat 已保留，没有据此宣称原题都经严格证明。

CPU 冻结验证 `824257/824270/824276/824293` 均 COMPLETED / 0:0：264/304/405/505 个定义分别通过 568/750/1014/1209 个检查，最后一轮 27 项测试、4 秒；全部零失败。四轮启动/结果直接绑定分别 114/19、115/22、116/25、117/28，最新进度标记的 11 项绑定也通过。四个全文自动提取局限仍与登记一致：`04253` 原文给出两个焦点但要求任选一个，`06922/07225/07321` 所需多值分布在解释段落。这不等于原答案数学错误，也不证明所有真实模型输出评分可靠。所有 proposal 仍未应用到正式 cohort 或监督选择。

steering 分片 `822748/822750` COMPLETED / 0:0，各 2,532 条，共 5,064 条；各 5 项结果绑定已验证。末查 `822749/822751` steering 和 `822547/822550` raw 在 gpu17 RUNNING，原 `822579` 仍 PENDING / Priority=0 / JobHeldUser。未释放旧文本入口、未提交正式学生 SFT。下一步完成剩余 128 题，冻结完整 cohort/评分 overlay，检查实际方向和开发输出的影响，再重评分全池并迁移下游绑定。完整 baseline、25K、跨模型/预算矩阵仍未完成；本轮为已执行的复核、实现、CPU 验证及新生成证据进展。

## 2026-09-13 20:43 EDT：新增 168 个参考复核，独立评分实现通过 CPU 验证

已读完并保存剩余 105 个多 box 问题的原始参考，另完成 63 个进制上下文/无花括号 box 案例。新增六个需要答全的定义：两组有序交点、两个多项式、两项整数解和两组角度；另登记任选一个单位向量、周期等价相位移、12/24 小时时间及四个输出进制。多数其余 box 是重复解法或中间结果，保持单目标。两条所谓“无 box”来源实际使用合法的 `\boxed 2` / `\boxed 9`。由项目助手进行语义/类型复核，不是独立人工证明审查；所有建议仍未覆盖原 gold。

独立模块 `reviewed_math_grading.py` 和验证入口 `13_47_validate_reviewed_math_definitions.py` 保留旧两个 grader 的字节与全部历史证据。新实现区分单值、答全和任选一个，保留重数/顺序，读取由连接词相连的最终多个 box，支持进制文字包装与单 token TeX；不会把所有中间 box 自动拼接。首轮 CPU `824244`（4 秒）通过 18 项测试及 105 定义的 128 个检查；追加版本 `824247`（3 秒）通过 20 项测试及 168 定义的 191 个检查。首轮 111/10 项、追加 112/13 项启动/结果绑定及两组复核 8/10 项绑定已验证。

原解答 `math_train-06922` 的三个角度分散于解释段落，仍不能由最后连续 box 列表完整提取；该局限预先登记、保留诊断，不能用 canonical 检查通过证明实际模型输出都评分正确。四组复核累计 206 个唯一 MATH 问题，其中 202 个属于 629 个扫描标记；还剩 427 个百分比、文字和多答案请求案例。多 box、进制上下文及无 box 标记已全部复核。见 [参考修订及验证](docs/experiments/phase13/phase13_math_reference_integrity_v1.md)。

Slurm 末查 `822748`–`822751` 四个 H100 steering 作业仍 RUNNING；文本准备 `822579` 仍 PENDING / Priority=0 / JobHeldUser。尚未重评分全池或恢复后续 69 个压缩/数据节点，没有学生 SFT 提交。本轮完成的是参考修订的实际进展，完整 baseline、25K、跨模型/预算矩阵仍未完成。下一步处理剩余 427 个参考，将此前 38 个建议转为相同显式定义，随后冻结完整 cohort overlay、检查真实方向/开发输出影响并迁移下游依赖。

## 2026-09-13 20:15 EDT：来源 near 全审计完成，保留主参考修订门槛

内部近似 `823978` 和完整合并 `823979` 已 COMPLETED / 0:0，分别 19 分 17 秒和 9 秒。全部 89,120 个数学问题产生 157 对 near、88,968 个连通组，最大 3 题；690 个原保留匹配没有新增传递邻居。v2 评分下暂可解析候选 63,218 个，覆盖 63,121 个 near 组。准备/四分片/near/合并的 14/20/6/12 项直接绑定通过；32 个来源质量样本已产出但未完整阅读。该来源阶段完成不解除主参考完整性问题。

主 raw 分片 `822549`（shard06）COMPLETED / 0:0，1,680 条新候选、5 项直接绑定通过。其评分仍属待修订的 v2 来源；生成内容保留。新的 steered 分片 `822751` 已接替 H100，当前四个 steering 分片运行。原文本准备 `822579` 最后回读 PENDING / Priority=0 / JobHeldUser，保持暂停；无学生 SFT 提交。

参考扫描的 400 项启动/11 项结果、hold 的 2 项和两组语义复核各 6 项直接绑定通过。下一动作保持完成剩余 MATH 来源复核及新评分定义，再检查实际输入影响、重评分全部源并迁移文本/SFT 数据依赖；不能直接释放旧入口。完整研究矩阵仍未完成，目标继续保持进行中。

## 2026-09-13 20:08 EDT：主参考提取缺口已确认，旧文本入口保持暂停

四个来源参考分片 `823974`–`823977` 均 COMPLETED / 0:0：64,946 个候选中 63,231 个自比较通过，1,728 个解析/一致性复核 case；5,494 个精确 MATH 对照中 5,478 个一致。16 个差异已复核并保存未应用的 canonical proposals。它们涉及提取完整性和格式，不能解释为 16 道原始数学解答错误。负号例全文为 `-\boxed{15}`，原解答正确，提取丢了框外负号。原近似审计 `823978` 仍 RUNNING，CPU 时间持续增长；合并 `823979` 等待它成功，未重启该活跃作业。

三项测试及完整参考扫描 `824110` COMPLETED / 0:0，7,987 题中 629 个标记案例，其中多 box 122。优先完整阅读校准/开发的 14 个多 box 来源及实际入选数据的其余 8 个标记来源，已保存 22 个决定。确认两个校准多答案、一个开发多答案以及一个校准“任一焦点”定义需要修订；逐 ID 检查，它们均未进入实际 ASC/dense 各 100 对和 64 道开发题。其余实际入选标记参考值完整，尚不因此重训方向；仍需完成所有参考/评分/真实预测的影响审计。见 [发现、复核及恢复条件](docs/experiments/phase13/phase13_math_reference_integrity_v1.md)。

为避免旧 gold 进入 DAP/TokenSkip 与学生数据，`822579` 已在 PENDING、确认 owner 后 user hold，PENDING/Priority=0 回读通过，前后证据保留。主 raw/steered GPU 生成继续；旧全池合并即使完成也保留为 v2 历史评分，不能直接用作修订后的正式结果。下一步必须完成参考语义与类型修订、冻结新版本、重评分完整候选并迁移下游绑定，不能直接解除旧入口或删掉所有标记题。

Llama 后续分支的 gated 401 访问问题仍待用户配置有权 HF 登录，其他工作继续。完整 baseline 目标保持：尚未完成共同支持 SFT、25K、跨模型和训练预算矩阵。本轮新增的完整性发现改变了当前主流程的下一动作，不以旧评分自比较通过证明源 gold 可靠。

## 2026-09-13 19:50 EDT：全部候选参考审计运行中；Llama 访问缺口已识别

新增原参考解析、精确 MATH 同题一致性和来源内部 near 连通组审计，复用既有 typed grader、匹配及组件代码，不改变历史评分。401 项启动绑定已冻结；16 项测试 `823972` 和准备 `823973` 均 COMPLETED / 0:0。四个解析分片 `823974`–`823977` 对全部 64,946 个预筛候选执行检查；内部 near `823978` 使用全部 89,120 个数学问题并传播保留集隔离，五个作业已在 `c026n04` 实际运行，完整合并 `823979` 等待全部成功。自比较通过只表示解析诊断，不能称为参考正确。见 [执行协议](docs/experiments/phase13/phase13_openthoughts_reference_audit_v1.md)。

后续计划中的 `meta-llama/Llama-3.2-3B-Instruct` 无本地权重缓存。公开 metadata 可读取并固定 revision `0cb88a4f764b7a12671c53f0838cd831a0843b95`，但下载该 revision 的 `config.json` 返回 `GatedRepoError` / HTTP 401，未下载权重。访问结果及源码版本记录于 `preparation/llama_student_access_v1`；已请用户在集群配置获权 HF 登录，不在对话传 token。该外部访问问题仅影响后续 Llama 分支，不替换模型或停止当前 Qwen/baseline 工作。

主源四个 H100 作业仍 RUNNING，raw/steering 完整合并、文本压缩、共同支持和学生 SFT 均未完成。完整 7.5K→25K、跨模型和预算矩阵继续保留；来源审计不能替代正式学生对比。

## 2026-09-13 19:39 EDT：OpenThoughts 来源完整审计完成，25K 保留为后续阶段

CPU 测试 `823881`、完整来源审计 `823882` 和完整性复核 `823894` 均 COMPLETED / 0:0。12 个 metadata 分片的上游 LFS SHA-256、大小、schema 与 113,957 行总量通过；42 项直接结果绑定、全部来源 ID/字段哈希、数学行及候选集合重新核对，40 条原 parquet 抽查通过。launch 的 410 项、验证/重叠复核的 3/4 项绑定也已核验。见 [协议及结果](docs/experiments/phase13/phase13_openthoughts_inventory_v1.md)。

数学记录 89,120 条，规范化后均唯一；22,794 条含证明措辞，702 条有视觉词汇标记，32 条有 CJK，109 条缺失或空 box，标记允许重叠。690 个数学问题与评测或校准/开发/SAE 保留题精确或近似匹配；最终 64,946 个预筛候选。该数量没有经过参考正确性、题型/语言可靠验证、内部近似去重或教师正确覆盖，不能作为真实 25K 训练规模。

五个评测来源有两对精确和四对近似匹配。项目助手复读四对 near：三对 OlympiadBench 属同题格式/符号变体，一对 MATH-500 共用邮票材料但问不同量；六题全部继续排除，复核不是独立人工标注。现有 MATH 学生池另有 5,900 对精确和 47 对近似关系，作为信息项报告。来源审计始终没有使用 DeepSeek 生成答案作为 gold。

主 raw/steered 四个 H100 作业在本轮末查仍 RUNNING，后续 69 个压缩/共同数据准备节点继续等待父依赖，尚无正式学生 SFT 结果。本轮完成的是 25K 来源可行性和数据完整性证据；完整 baseline、学生比较和后续模型/预算矩阵保持未完成，目标继续推进。

## 2026-09-13 19:34 EDT：主池压缩流程已提交，25K 来源审计接入

主池后续 69 个作业已真实提交，八项检查 `823697` 已 COMPLETED / 0:0；launch 的 488 项和提交的 144 项直接绑定已核验。DAP/TokenSkip smoke `823702/823703` 依赖原文本准备 `822579`，smoke 合并 `823704` 后接 DAP 32 片 `823705`–`823736`（H200、四 lane）和 TokenSkip 32 片 `823737`–`823768`（L40S、单 lane）。文本完整合并 `823769` 等待全部 64 片；八方法共同数据准备 `823770` 等待 raw `822575`、steering `822780` 和 text 三个完整合并。没有提交 SFT 或测试评测。见 [依赖流程](docs/experiments/phase13/phase13_unified_pipeline_v1.md)。

本轮重新查 Slurm，`822549/822748/822749/822750` 四个 H100 主源作业仍实际运行，后续源合并及压缩保持依赖等待。没有重启或改变已冻结的主生成硬件。原 gpu18 归档 `823126` 仍等待节点恢复，不阻塞其他工作。

OpenThoughts 固定 revision 的 12 个 metadata 文件已下载完成，共 2,469,730,242 bytes；传输完成尚不等于哈希验证。其 `ground_truth_solution` 含证明解答，不能仅因有 box 就计入可评分题数。新增来源清点复用现有提取、规范化和近似匹配，并核查五个评测来源、MATH/SAE 保留题及学生题重叠。410 项来源/代码绑定已冻结，CPU 测试 `823881` 已输出四项通过，审计 `823882` 按成功依赖提交到检查过容量的 `c026n04`。见 [来源审计协议](docs/experiments/phase13/phase13_openthoughts_inventory_v1.md)。不使用 DeepSeek 生成答案作为 gold，不声称已选定 25K 题池。

完整 baseline 复现目标保持进行中：八方法正式共同支持、学生 SFT/评测、25K 以及多模型/预算矩阵仍未完成。本轮是实际实现、冻结及启动来源审计的进展，不以排队记录代替实验结果。

## 2026-09-13 19:06 EDT：四 seed 探索性生成与新评分分析完成

同 H200 的四个分片 `823476`–`823479` 全部 COMPLETED / 0:0，各 128 条，28 项直接绑定复核通过；加上 64 条 smoke，共 576 条新输出。答案准备 `823510`、最终评分 `823565` 和分析 `823610` 完成。项目助手审阅 20 个 case 对应的 15 个最终答案句组合，8 条正误标签变化、无未解决 case；这不是独立人工标注。准备/决定/评分的 16/5/9 项绑定、分析的 18 项绑定及图已核验。

64 道已观察问题上，未干预 63/64、287.61 tokens；四个 top-3 seed17/42/73/101 分别为 60/63/61/62 正确、270.25/305.59/279.30/278.92 tokens，全部零 cap hit。相对未干预长度差为 −17.36/+17.98/−8.31/−8.69；前两者描述性边际区间分别 [−30.36,−4.25] 与 [4.91,31.78]，后两者跨零。原 seed17 八特征参考为 60/64、251.83 tokens，Answer-format 为 62/64、243.56，dense 为 60/64、204.22。此处不作同时检验或非劣主张。结果未支持 top-3 适配跨 seed 稳定缩短；原八特征在新增 seed 上不可构造的结果继续保留。见 [完整报告](results/phase13_baseline_expansion_v1/exploratory/sae_seed_generation_analysis_v1/report_zh.md)。

AGENTS.md 已明确 NCSU H100/H200/L40S 授权、L40S 实际路由和资源检查。baseline 主目标仍在推进：当前 H100 上有三个 steering 学生分片和一个 raw 分片运行；全部源合并、文本压缩主池、共同支持、正式学生 SFT，以及 25K/泛化/预算矩阵尚未完成。失败 gpu18 临时文件归档继续等待 drain 节点恢复，不构成其他已授权工作的阻塞。

## 2026-09-13 18:54 EDT：SAE 确认与三个新 seed 分析完成，原八特征构造未稳定复现

NCSU H100、H200、L40S 已在 AGENTS.md 总授权和具体资源条目中明确；L40S 的 `jekml_gpu / gpu_partners / short_gpu / gpu:l40s:1` 路由与 2 小时 QOS 再次经 Slurm 验证。新增独占进程 GPU 的独立 CUDA preflight 和实测 scratch 容量要求，来自本次实际失败，不改变已冻结实验。

确认生成八个分片 `822804`–`822811`、新答案准备 `822813`、最终评分 `823236`、预登记分析 `823237` 均已完成。300 题 × 7 条件及 64 题 × 24 消融共 3,636 条，另有 88 条 smoke。项目助手审阅 62 个最终答案句组合（113 个唯一 case），对其中 6 个组合的 15 条全文补读，35 条正误标签变化，无未解决 case；并非独立人工标注。确认分析及新评分 24/9 项直接绑定已复核。

确认结果：未干预 287/300、290.60 tokens；SAE 279/300、249.95；dense 276/300、204.90；Answer-format 277/300、246.47，七条件均零 cap hit。SAE 相比未干预少 40.65 个生成 tokens、少 39.39 个正文 tokens，与未干预/三个随机方向的长度检验通过同一 18 项 Holm 校正（p=0.000900）。SAE 相较 Answer-format 没有确立优势；相对未干预准确率数值下降 2.67 pp，Holm p=0.6738 不能证明非劣。正文缩短并不等于高层推理语义成立。见 [确认协议与结果](docs/experiments/phase13/phase13_sae_confirmation_v1.md)。

三个新 SAE `823190/823191/823192` 成功完成 1,500 步、最终权重发布和内部哈希/有限值/decoder 范数验证，复用原 token 样本、归一化和训练代码。特征准备 `823168` 与原规则评分 `823252/823253/823254` 全部完成；原 seed17 与新 seeds42/73/101 分别确认 8/3/5/7 个 short features。新增三个 seed 均不足原方法所需八个，不补入未确认特征。六项测试 `823369` 与描述性对齐/Answer 区域分析 `823370` 完成，四个 top-3 方向跨 seed cosine 约 0.11–0.14。新特征普遍但并非全部集中于 Answer；完整匹配、区域效应与图保留。见 [SAE seed 复验](docs/experiments/phase13/phase13_sae_seed_stability_v1.md)。

原 gpu18 的两次最终保存失败和一次 CUDA 独占进程冲突保留；第一轮恢复的 scratch guard 拒绝、NumPy 不匹配的特征准备失败也保留。恢复没有改变科学参数。gpu18 因 `/var` 满而 drain，限定路径归档 `823126` 仍等待节点恢复，不能报告已归档或清理。

统一 H100 steering smoke `822746/822747` 已完成，96 条候选、5/10 项直接绑定验证通过，峰值 allocated 16,945.18 MiB；真实 steering 学生分片 `822748/822749/822750` 已运行。前四个 raw 分片 `822543`–`822546` 已全部完成，共 6,744 条，各 5 项直接绑定通过，其他分片按 DAG 继续。完整源合并、DAP/TokenSkip 主池、八方法共同支持及主学生 SFT 仍未完成。

另行登记同 H200、64 道已观察问题的四 seed top-3 探索性生成，保留 seed17 原八特征、未干预、格式与 dense 参考，共 512 条，加 64 条 smoke。共享 rollout 重用检查 `823418` 的 18 项测试、准备 `823419` 和 H200 smoke `823475` 均通过：14/474 项协议/源码绑定与 7 项 smoke 绑定核验，smoke 零 cap hit、峰值 allocated 15,309.25 MiB。准备重核了新 seed 分析的 75 项绑定。四个分片 `823476`–`823479` 已在不同 H200 UUID 上运行，576 条新输出的答案复核准备 `823510` 等待完整生成。它是看到入选数后的适配，不作为原八特征成功复现或独立确认。完整研究目标继续保持，包括主 SFT、25K 和模型/预算泛化矩阵。

## 2026-09-13 18:04 EDT：统一 steering 学生源已排队，SAE 保留题确认正在生成

完整 steering 开发合并 `822340`、选择 `822401` 和学生源准备 `822631` 均已完成。最后两片的 10 项、开发合并的 35 项、选择的 26 项直接绑定复核通过，完整曲线已检查。按预登记筛选和配对长度目标，实际选中 B3 rho=0.2 / layer 17、B4 ASC scale=0.5 / layer 16、B7 rho=0.2 / layer 17。三个条件各有 57/64 正确支持，八方法开发交集 53/64；这些不是学生训练样本或学生收益。学生源协议 26 项输入和 442 项源码绑定通过，并逐项匹配 raw 阶段的 H100 路由、32 分片、题序及采样设置。

新的 H100 smoke `822746`、smoke 审计 `822747`、32 个学生分片 `822748`–`822779` 和完整合并 `822780` 已按四 lane 依赖提交，共预期 80,676 条 B3/B4/B7 学生候选。当前 smoke 等待资源；原始源首批四个 H100 分片仍在运行并持续写入，未重启。原始源完整合并 `822575`、文本准备 `822579`、全部八方法共同支持及实际 SFT 仍未完成。见 [steering 学生源](docs/experiments/phase13/phase13_steered_student_generation_v1.md)。

评测输入预检 `822715` 已完成两个测试和两学生 × 3,966 题 × 七提示的 55,524 个 native-chat 输入检查；20 项产物及 461 项 launch 绑定通过。最长输入 1,315 tokens，候选 4K 输出预算下没有上下文溢出。选择题共享字母答案提示，不读取 gold。没有生成测试答案，最终 cap 与完整评测仍待后续冻结。见 [评测输入说明](docs/experiments/phase13/phase13_student_evaluation_inputs_v1.md)。

SAE 确认配置复用原生成/分析代码，明确转入保留的 300 个 confirmation IDs。固定原始参考 rho=0.3、六个方向和全部预定窗口/特征控制；18 项主要双侧比较在同一 Holm 族内，生成前冻结规则与分析源码。旧未干预 baseline 曾生成和评分，独立性只限于新干预效果，不能声称这些问题从未观察。15 项角色、窗口、随机化与校正检查 `822796`、准备 `822797` 均完成；协议 13 项、源码 453 项绑定通过。

H200 smoke `822803` 已完成 88 条、零 cap hit，6 项直接绑定通过，峰值 allocated 15,296.77 MiB。四个完整确认剂量分片 `822804/822806/822808/822810` 已实际运行，两个 H200、两个 L40S；各自完成后接 64 题消融分片 `822805/822807/822809/822811`，最多四片并发。新答案复核准备 `822813` 依赖完整生成；新输出尚未复核或完成确认分析。见 [保留题确认协议](docs/experiments/phase13/phase13_sae_confirmation_v1.md)。

完整目标仍保持：确认分析、额外 SAE seeds、全部真实学生监督与 SFT、后续 25K 和模型泛化/训练预算矩阵尚未完成。历史 seed17 的 SAE 训练入口与当前入口 SHA 一致，既有 layer17/k64 为 1,500 步、90.31 秒；追加 seeds 仍需绑定相同 token 样本、归一化与训练依赖，不能仅修改 seed 后忽略原 sample-manifest 的 config hash。

## 2026-09-13 17:34 EDT：MATH ASC 原采样分析完成，共同 SFT 与 steering 学生接口接入

MATH ASC 五强度 `821689` 已 COMPLETED / 0:0，共 320 条，5 项生成绑定通过。复用原 `baseline_method_analysis`，扩展 typed MATH source-context 评分和 token/EOS/seed 审计；CPU 分析 `822680` 完成，100 项直接绑定及图片已核验，评分变化为 0。未干预为 50/64、639.50 tokens；scale 0.1/0.25/0.5/1 为 47/45/47/35 正确、652.78/659.08/634.56/605.02 tokens，scale 1 有 3/64 cap hits。所有非零强度的配对长度 95% 区间均跨 0；scale 1 准确率差 −23.44 pp，区间 [−35.94,−10.94] pp。没有确立 MATH 原采样压缩优势，不以通用文本 KL=0.01875 宣称任务行为保持。见 [完整分析](results/phase13_baseline_expansion_v1/exploratory/asc_qwen_math_analysis_v1/report_zh.md)。该单候选结果不替代四候选统一主比较。

新增 B3/B4/B7 学生源注册解析，16 项检查 `822630` 通过，准备 `822631` 依赖最终开发分析 `822401`。只接受完整分析实际选出的方向、层和强度，保持与 raw stage 一致的 H100、题序、32 分片及随机流；未先验填入一个“应当更好”的剂量。见 [steering 学生协议](docs/experiments/phase13/phase13_steered_student_generation_v1.md)。

新增 `unified_student_distillation.py` 与 `13_35_distill_unified_students.py`：从全部八种完整源重核选择与共同支持，保留 TokenSkip 比例格式，用已验证的显式 completion 标签和曝光计数执行 S4a 首阶段 12 次 SFT。五项反例检查 `822655` 通过。配置 `phase13_unified_student_sft_math7_5k_v1.json` 尚未绑定真实交集题数，数据准备和实际 SFT 未运行；不能把通过单元测试当作学生比较完成。见 [共同数据与 SFT 协议](docs/experiments/phase13/phase13_unified_student_distillation_v1.md)。

统一开发分片 0/1/2/3/4/6 已完成，各 384 条、共 30 项直接绑定核验通过。尚未启动的 5/7 已改排 H200（原 job ID `822337/822339` 保留），初次 Slurm 更新部分改变 GRES 后返回依赖错误；核对并修正冗余已完成依赖后第二次成功，前后状态完整保留。现在最后两片均在 H200 运行；已运行的 L40S 分片自然完成，最多四个开发分片并发。四个真实 raw 学生分片 `822543`–`822546` 均已运行，其他源分片、完整合并和文本准备继续依赖推进。完整目标保持，未完成共同支持 SFT、确认/SAE seeds、25K 及泛化/预算研究。

## 2026-09-13 17:10 EDT：真实学生题池已产生首批候选

原始源 H100 smoke `822541` 和审计 `822542` 均完成，64 条候选及 5/10 项直接绑定通过；B1/B0 支持 5/8，B2 为 7/8，交集 5/8。峰值 15,507.03 MiB（约 15.14 GiB）。学生池分片 `822543`、`822544`、`822545` 已实际运行，并保存了 `question_role=student_pool` 的首批真实预测；`822546` 等待资源，其他 28 片保持四 lane 依赖。这些部分预测不当作完整实验。

同题同 seed 的旧 H200 smoke 与新 H100 smoke 各方法只有 2/32 逐 token 一致，B1/B2 正误分别改变 0/4 条。已检查 decoder 唯一源码差异为 DAP messages 支持且对这些输入不生效，grader 源码一致；两次硬件/驱动不同，但这不是硬件单因素实验。`smoke_cross_run_audit_v1.json` 保存完整绑定和界限。后续 B3/B4/B7 主学生直接生成需使用同一 H100 路由、题序、32 分片与批次大小，避免主表直接生成条件混入这些调度差异。

## 2026-09-13 17:05 EDT：两个学生训练接口已验证，真实 MATH 学生源生成已接入

新增 `completion_supervision.py`，在共用 TRL wrapper 中支持显式 native-chat completion 标签及训练前核查。真实 batch 保留 EOS、掩蔽 prompt 和 padding，并计数 causal-shift 后监督 token、输入 token、microbatch 与逐题曝光；不修改 TRL 优化目标，当前限定单进程。首个 CPU `822513` 因测试夹具未指定 pinned TRL 要求的 dtype 失败，修正为 CPU float32 后独立冻结的 `822519` 六项检查及输入准备完成。历史冻结训练代码不变。

H200 合成 SFT `822531`（Qwen-1.5B）和 `822532`（Qwen-3B）均完成：每个学生 16 条、最长 8,189 token、两个更新、每条恰好一次、59,326 输入/58,176 监督 token。两套 adapter 的 20 项直接完成绑定核验通过；allocated 显存峰值 18,114.43/21,746.97 MiB，reserved 峰值 26,218/30,038 MiB。实际训练与权重更新通过，但没有使用开发/校准/学生问题，不是学生性能或正式 SFT 结果。见 [训练接口证据](docs/experiments/phase13/phase13_unified_sft_interface_v1.md)。

真实学生源配置 `phase13_unified_math_candidates_raw_v1.json` 已固定：6,723 道既有学生池问题，B1/B2 各四候选，共 53,784 条，B0 复用 B1；生成/选择/提示/随机流必须与已完成开发检查完全一致。15 项检查 `822527`、准备 `822529` 完成，13 项输入/父绑定和 440 项源码绑定通过。H100 smoke `822541` 已运行；审计 `822542`、32 个学生分片 `822543`–`822574`、总合并 `822575` 按四条 lane 成功依赖提交，最多四个学生源分片同时运行。线性开发参考约 34.9 GPU 小时，不是实际成本或八方法总成本。DAP/TokenSkip 主学生数据准备 `822579` 已依赖完整源合并，TokenSkip 主阶段只执行每题预分配比例。见 [真实题池协议](docs/experiments/phase13/phase13_raw_student_generation_v1.md)。

统一 steering smoke `822330` 和合并 `822331` 完成，共 384 条；生成/选择的 5/28 项直接绑定通过，峰值 allocated 16,945.91 MiB，生成 1,188.36 秒、额外重放 323.20 秒。第一批四个完整开发分片 `822332`–`822335` 均实际运行（两 H200、两 L40S），后四片/最终合并/选点保持依赖。MATH ASC 原采样五强度 `821689` 仍运行。B3/B4/B7 学生题池、B0–B7 共同支持及实际主 SFT、确认/SAE seeds、25K 和泛化/预算矩阵仍未完成，完整目标继续保持。

## 2026-09-13 16:37 EDT：统一开发选点审计已接入

新增共享分析 `unified_steering_analysis.py` 与入口 `13_33_analyze_math_steering.py`。在本次完整开发扫描完成前明确配对目标为同题最短正确监督的 method−B1 token 差；保留原准确率、覆盖、cap 筛选及低强度平局规则。两项反例检查 `822400` 已通过；分析 `822401` 按成功依赖等待完整开发合并 `822340`。它会验证相同输入设置、全部四候选格、逐题候选均值的配对区间，并关联 B0/B1/B2/B5/B6 的开发支持，输出八方法共同支持及实际运行点；没有数据则不会生成选点结果。

本次最后核对 `822330` 仍在 H200 运行，MATH ASC 原采样五强度 `821689` 仍在 H100 运行；其余统一开发分片保持依赖等待，未重启活跃 GPU 作业。新方向提取、四个文本压缩分片及当前 smoke 的 GPU 型号/UUID 已核查记录。主学生题池生成、B0–B7 SFT 和后续扩展仍未完成，目标继续保持。

## 2026-09-13 16:31 EDT：MATH dense/SAE 方向完成，统一 steering 开发矩阵已提交

方向测试恢复作业 `822221` 的 7 项检查通过，准备 `822178` 与 L40S 实际提取 `822263` 均 COMPLETED / 0:0。100 对校准、同层 dense 和历史 SAE/三个随机集合方向已封存；提取 10.40 秒、峰值 14,840.33 MiB，dense–SAE cosine −0.02986。准备、源码、方向分别 11/427/6 项绑定复核通过。

统一 steering 准备 `822279` 完成，17 项输入/父绑定和 429 项源码绑定通过。12 条件（B3/B7 各四个 rho、B4 四个 scale）×64 题×四候选的 3,072 条开发输出已提交：smoke `822330` 正在 H200 运行，smoke 审计 `822331`、八分片 `822332`–`822339`、最终合并 `822340` 按成功依赖排队；最多四个完整分片同时运行。方法共享采样、题目、prompt/cap、typed grader 与 B1 参考；运行点选择规则在扫描前记录。见 [执行协议](docs/experiments/phase13/phase13_math_steering_dev_v1.md)。未完成前不计作完整比较，6,723 学生题池仍未生成或 SFT。

CPU 恢复复核发现共享目录可见性/Slurm采样滞后：原 ASC 复核分析 `822183` 在取消交界已写出全部文件、完成标记和最终日志，随后 Slurm 状态为 CANCELLED；替代 `822222` 因目录已存在拒绝覆盖。已核验原 v2 分析的 97 项绑定、192 条预测及图，完整保留该派生分析和两次调度记录，不声称原作业正常 COMPLETED，也不再重跑。其 scale 0.25 / 1 的配对长度差 95% CI 分别为 [−381.34,−146.67] / [−1028.63,−763.72] tokens；已观察开发题的准确率区间不证明非劣或泛化。恢复审计记录于 `preparation/cpu_health_recovery_20260913T2022_v1/COMPLETE.json`。

## 2026-09-13 16:24 EDT：统一 DAP/TokenSkip 完成；SAE 全部开发生成与新答案复核完成

统一 Qwen-7B 候选的 512 条输出、两个分片和选择 `821938/821939/821940` 完成并复核绑定。B1/B2 各有 56/64 题正确支持，共同支持 55 题；B0 与 B1 共用原始正确池。真实文本压缩的 17 项检查 `822078`、准备 `822080`、两个 smoke `822110/822111`、smoke 合并 `822112`、四个完整分片 `822113`–`822116` 及合并 `822117` 均已完成。DAP 改写全部 194 条唯一正确源，194/194 最终答案正确、零 cap hit，平均 547.40→479.12 tokens；同 56 题最短正确监督为 B1 526.02→B5 442.13。TokenSkip 完成 56×6=336 条压缩与 56 个预分配比例样本。原始 B1 生成 686.75 秒、额外 DAP 471.84 秒、六档 TokenSkip 开发压缩 8.54 秒；不是完整端到端成本。见 [统一文本压缩说明](docs/experiments/phase13/phase13_unified_text_compression_dev_v1.md)。当前样本均属 development，不能用于学生 SFT。

SAE 八个完整剂量/消融分片全部结束。新答案准备 `821638`、格式分流 `822126` 和最终评分 `822142` 完成，9,419 条输出、341 个复核案例，137 条正误标签变化。项目助手审阅 114 种完全相同问题/最终答案句/提取规则的组合，覆盖 336 个完整文本 case hash；其中 5 个组合另读全文。其余 5 个案例按达到 cap 且缺少最终答案的格式规则处理。这不是独立人工验证，也不是逐项验证 336 条完整推理语义。输入/决策/最终评分的 31/6/9 项直接绑定核验通过。

完整 SAE 分析 `822151` 完成，23 项直接绑定及两幅图已检查。300 题单候选开发集上，未干预为 93.67% / 298.19 tokens；SAE rho=0.3 为 89.33% / 259.03，dense 为 87.67% / 205.89，answer-format 为 91.33% / 251.92。正文长度也下降，不能仅解释为删除答案后缀；但格式方向同样能影响此前生成，当前结果不支持 SAE 优于格式控制或证明高层推理语义。after-marker 控制的正文保持不变。全部剂量、单特征/leave-one-out/窗口结果及探索性配对区间保留在 [报告](results/phase13_baseline_expansion_v1/exploratory/sae_generation_analysis_v1/report_zh.md)。计时含机制读回，不是部署时延；confirmation 与额外 SAE seeds 尚未完成。

R1 native CoT 生成 `821277` 和初次分析 `821348` 完成；新统一答案复核后的三强度正确数仍为 64/64、64/64、62/64，平均 1455.70、1192.00、564.13 tokens。方向通用文本 KL=0.0224051，略高于 0.02 软目标；这是已观察 GSM8K dev 的方法忠实性检查，尚非独立确认或学生结果。新的复核分析使用独立 v2 目录。MATH Qwen 校准 `821686` 共 184 次尝试、访问 118 题，得到 100 对合格轨迹；8 步 `821687`、完整 3,000 步 `821688` 均完成，held-out WikiText KL=0.0187533，向量 norm=16.6412。五强度生成 `821689` 正在运行。

R1 两题长度扩展 `821927` 完成，原 8K 前缀及原四档统计一致；两条完整输出为 11,417 / 15,676 tokens，开发规则建议 16K，16K/32K 均 26/32 正确、零 cap hit。新增实际生成 589.81 秒（含重复前缀）；不能将该 32 题建议视为全部模型/评测的尾部保证。

MATH B3/B7 方向入口 `13_32_prepare_math_directions.py` 已加入：抽取共用 response hidden-state 逻辑，B3 使用同层、题目等权的 100 对校准状态，B7 保留历史 BF16→FP32 八特征构造和随机方向；完整 source token 绑定及共享解码设置会验证。新 12 条件×四候选 MATH steering 开发配置已登记，尚未启动生成。两个 CPU 作业原 `822177/822183` 所在 20 核节点 CPULoad 达数千、batch CPU 为零；确认目标未写入后取消并保留记录，相同冻结源码改由健康节点执行为 `822221/822222`，准备 `822178` 依赖新测试。`822117` 在取消前重查已完成，完整保留，没有重跑。恢复只改变 CPU 调度，详见 `preparation/cpu_health_recovery_20260913T2022_v1`。

完整目标仍未完成：后续需要主教师各方法统一开发选择、真实学生题池与 B0–B7 SFT、独立确认/SAE seeds，以及分阶段扩展的模型、数据和预算矩阵。


## 2026-09-13 15:51 EDT：统一数学候选接口已运行；R1 上限诊断触发两题扩展

主教师候选接口 `unified_math_candidates.py` 已接入：复用逐题随机流解码和既有正确文本去重，把 task grading 从原数值生成器中分离。B0/B1 共用每题四个未干预候选，分别随机正确/最短正确选择；B2 使用同样四候选、4K cap、0.7 temperature、0.95 top-p、显式 top-k 20 和匹配 boxed 格式。完整池覆盖与当前可用方法共同支持分别保存，不把 B0/B1/B2 支持当作全部八组支持。

13 项缓存/窗口/实际小模型解码/随机流/top-k/向量一致性/选择审计检查 `821887` 通过，准备 `821888` 完成，10 项输入/父绑定及 418 项源码绑定验证通过。H200 smoke `821936` 与选择审计 `821937` 已完成，64 条输出和 15 项直接绑定验证通过。生成批次合计 179.35 秒，额外重放 66.85 秒，peak allocated memory 16,094.75 MiB；B0/B1/B2 各自正确支持及共同支持均为 5/8，三个零支持题完整保留。仅检查其 B1 候选 0 的实际文本，三者均有实质错误；这不是全部候选的独立人工复核。两个 64 题开发分片 `821938/821939` 已实际运行，合并 `821940` 等待二者完成。当前只允许 smoke/development；冻结的 6,723 学生题输入还未用于生成或 SFT。

R1 cap `821644` 已完成，6 项直接绑定验证：1K/2K/4K/8K 正确数为 3/21/25/26（各 32 题），cap hit 为 32/13/5/2。因此原 5% cap-hit 规则没有给出可接受上限。两次额外直接 1K 解码均与实际长输出前缀一致。独立扩展配置 `phase13_math_cap_extension_r1_v1.json` 只重跑原来触及 8K 的 `math_train-02785` 与 `math_train-02140`，其余 30 条已终止输出保留；生成最多 32K，再比较 16K/32K。要求原 8K token 前缀和原四档统计完全一致，全部重复生成计入成本，不据此宣称尾部概率保证。三项检查 `821926` 通过，H100 扩展 `821927` 正在运行。

MATH ASC 校准 `821686`、原 R1 native CoT 生成 `821277` 和剩余 SAE L40S 剂量 `821369` 仍运行；另一个 L40S 剂量 `821373` 已完成并核验 6 项直接绑定，其消融 `821374` 已运行。按原依赖继续，不重启活跃作业。统一答案复核 `821638` 仍等待完整新输出。TokenSkip 原设置 SFT/评测与 DAP 长轨迹检查保持已完成，统一 B0–B7 学生蒸馏仍未完成。

[统一候选执行说明](docs/experiments/phase13/phase13_unified_math_candidates_v1.md)。

## 2026-09-13 15:27 EDT：MATH 角色冻结与三个模型 cap 检查完成；完整生成和主教师 ASC 校准接入

MATH 角色冻结 `821368` 完成，25 项直接绑定复核通过：7,500 来源题经既有排除剩 7,487，按近似题连通组划分为学生候选池 6,723、校准 400、开发 300、DAP 保留 64。51 个近似题对组成 7,445 个组，跨角色阈值内近似边为零；五个评测队列及 GSM8K-Hard 父题映射固定。6,723 仍是候选池，不能当成最终正确共同训练支持。

四模型 MATH cap 输入 `821380` 完成。`821643`（Qwen-7B）、`821645`（Qwen-1.5B）、`821646`（Qwen-3B）均 COMPLETED / 0:0，三个完成标记的 18 项直接绑定、父协议 6 项和 406 项源码绑定已验证。固定 32 道 dev 题上，2K/8K 的正确数分别同为 27、21、24；2K 的 cap hit 分别为 0/32、1/32、0/32。三个模型的规则建议均为 2K。这不保证干预教师、微调学生或其余评测集的尾部覆盖。六次额外直接解码与保存前缀完全一致，但这些实际样本本身都短于 1K，不能声称它们验证了实际长输出的截断；另有 CPU 小模型与 EOS 边界测试。R1 `821644` 仍运行。

ASC native CoT 的 3,000 步拟合 `821276` 已完成，向量 norm=36.6144，held-out WikiText KL=0.0224051，仍略高于 0.02 的软约束目标；6 项直接绑定已验证。完整生成 `821277` 仍运行，自动分析 `821348` 等待依赖，不能把部分答案当作压缩结论。

SAE 完整生成代码、11 条件 L40S smoke `821366` 和 7 项 CPU 检查已完成。smoke 共 88 条、零 cap hit，峰值 allocated GPU memory 15,268.54 MiB；零剂量重放、延迟/答案后窗口边界和实测范数检查通过。完整剂量为 300×25=7,500 条，消融为 64×24=1,536 条。四片剂量 `821369/821371/821373/821375` 及各自依赖的消融 `821370/821372/821374/821376` 已提交，当前部分完成、其余运行/等待；其中 H200 剂量 `821371/821375` 及消融 `821372/821376` 已完成，四个完成标记的 24 项直接绑定复核通过；两个 L40S 剂量仍运行，其消融待依赖。新的 9,419 条答案审计准备 `821638` 依赖这些输出及 ASC 新生成，旧评分不复用到新文本。

已接入 `math_baseline_calibration.py`：复用 CES 拟合/生成代码，改用已冻结 MATH 角色和 typed grader；检查题号及近似组隔离，排除未经推理修复的 gold-corrected 原参考，保留全部排除原因。CPU `821668` 的 11 项 CES/接口/元组/进制/角色检查通过；准备 `821669` 已完成，19 项输入/父结果绑定和 379 项源码绑定复核通过。400 题中 396 个原参考通过，4 个 gold 修正题以逐题理由排除；新的 H100 校准 `821686` 已实际运行并记录 UUID，8 步检查 `821687`、完整拟合 `821688`、64 题五强度评估 `821689` 按成功依赖排队。主 Qwen-7B 新校准注册 4K cap、目标 100 个正确长/短对，从 400 个校准题中按固定顺序取样。4K 比未干预诊断的 2K 建议保守，但最终方法生成设置和主 SFT 协议仍需单独冻结。

完整生成分析入口 `13_28_analyze_sae_generation.py` 已加入并冻结，CPU `821727` 的 3 项重复/条件/EOS/零干预/批次成本检查通过。分析会验证真实 token 与文本、区域重新分词和全部分片覆盖，使用新统一复核评分，输出完整剂量曲线、特征/窗口图及配对区间。当前尚未运行最终分析，因为新输出复核还未完成。

[完整生成与 MATH 角色协议](docs/experiments/phase13/phase13_generation_controls_and_math_cohorts_v1.md)。本阶段生成计时包含 SAE 编码读回、在线答案标题识别等机制测量开销，不能直接作为部署时延或完整端到端效率证据。完整复现目标仍包括统一 teacher 的 B0–B7 候选、学生 SFT，以及后续规模、泛化和预算控制。

## 2026-09-13 14:35 EDT：SAE 新题区域与同状态读回完成；ASC native CoT 完整拟合进行中

S1b 的 discovery/dev 读回已实际完成，尚不等于完整 S1b 或学生主表。CPU 数值/缓存测试 `821312` 的 11 项和分析审计测试 `821342` 的 2 项均通过。8 题 L40S 提取 `821322` 得到 56 个参考/生成变体；局部读回 `821323` 完成 680 条记录，零剂量和同状态检查通过。四个 discovery 作业 `821326`–`821329` 以及控制方向拟合 `821330` 完成；四个 dev 作业 `821335`–`821338`、CPU 分析 `821343` 均 COMPLETED / 0:0。

400 道 discovery 共 2,794 条变体记录：六个必需条件各 400 条，错误计算变体 394 条，缺失 6 条按准备记录保留。参考解答的八特征平均激活为 0.13506 / feature / token；Answer 改为 Result 后为 0.01898，移除标题后为 0.00111。参考中的字面 Answer 占 short 激活质量 71.27%，原始生成中为 69.99%。加入非推理 Answer 注释相对 Entry 注释的配对平均激活增加 0.03517，95% bootstrap CI [0.03379,0.03653]。这些新题测量加强了词汇/格式关联解释；末尾 marker 更换不影响此前正文状态，本来就是因果注意力下的预期性质，不能单独据此证明语义解释。错误算术的配对改变幅度很小（全序列均值 −0.000785），不能与格式效应混为一谈。原工具的 bootstrap p=0 表示有限 5,000 次抽样中未观察到相反尾部，不是精确 p 值为零。

300 道 dev 完成 37,500 条实际 next-token 读回，覆盖零剂量及六方向×四剂量×五位置。每个方向从独立复制的原始前缀 cache 开始，逐分支核验完全相同的 incoming h。SAE 在 rho=0.3 的正文中点使八特征激活总和平均增加 54.91，证明干预确实改变目标编码；其 full-vocabulary KL 为 0.00665。标题前 Answer 概率平均增加 0.04409，响应末 EOS 概率增加 0.00533；这是固定前缀上的局部概率变化，不是完整输出长度、准确率或学生收益。早期 EOS 变化约 1e-12 或更小，不能因图中自动坐标缩放而解释成显著终止效应。三个随机集合及 dense/答案格式方向全部保留；dense 的内容、格式、长度混杂边界已注明。

两幅图已人工查看；新分析、方向与八个 discovery/dev 完成标记的全部 71 项直接文件绑定已复核，协议和 389 项源码绑定另已核验。L40S smoke 峰值 allocated GPU memory 为 14,882 MiB（约 14.53 GiB）。`821323` 的排队依赖曾误录为 `821318`，已在启动前更正为真实提取作业 `821322`，原提交及修正记录保留。300 道 confirmation 的特征仍未读取。后续是完整生成的剂量检查、top1/2/4/8、单特征/leave-one-out、早期/延迟/答案后窗口和独立 confirmation；当前不能把 B7 定位成已验证的高层推理特征控制。

ASC native CoT 校准 `821271` 完成：103 次生成、100 道唯一问题、100 对正确且比官方参考长的轨迹，平均 1,370.72 tokens，零 cap hit。8 步拟合 `821275` 完成，base 参数无梯度；完整 3,000 步拟合 `821276` 在 H200 运行（本次核对已到 2,000 步）。后续 H100 评估 `821277` 仍等待拟合完成；新增独立自动评分分析 `821348` 依赖评估。该新输出不属于旧统一复核队列，疑难提取还需另审计。校准、smoke 完成标记和实际拟合 GPU UUID 已复核；不能预判新配置的压缩效果。

[SAE 完整读回报告](results/phase13_baseline_expansion_v1/exploratory/sae_readback_analysis_v1/report_zh.md)、[读回执行协议](docs/experiments/phase13/phase13_sae_readback_v1.md)、[ASC 新提示自动分析配置](configs/phase13_asc_native_cot_analysis_v1.json)。TokenSkip 完整原设置 SFT/曲线和 DAP 长轨迹适配结果保持上一节的完成状态；统一 teacher 的 B0–B7 候选池与学生 SFT、MATH 划分/cap 冻结和后续扩展仍待完成。


## 2026-09-13 14:04 EDT：TokenSkip 完整曲线与统一评分完成，DAP 长轨迹通过，ASC 新提示校准启动

统一答案审计 `821261` 已完成 31,091 条输出，70 条正误标签改变。队列隐藏方法、参考答案和旧评分；项目助手逐项记录 175 个疑难案例，另 1,012 个触及 cap 且无最终答案声明的案例按格式规则处理。后者不是逐项人工验证的语义失败；整个流程也不是独立人工验证。完整答案之后重复输出的答案框被截断、答案句包含多个背景数字等问题已在该固定队列中处理，原预测和原评分保留。7 组提取/决策覆盖/评分关联检查通过；关联评分必须匹配原问题、完整预测、参考答案、token 数和 cap 状态。

TokenSkip 的 base、作者 adapter 和完整训练 replica 共 29,187 条评测及全部 23 个条件已审计完成。最新统一复核分析 `821264` 完成：base 为 82.98% / 316.83 tokens；replica 固定 512 cap 下 ratio 0.5/0.6/0.7/0.8/0.9/1.0 分别为 72.26/76.52/81.17/82.27/82.03/84.16%，长度为 167.37/196.61/243.08/269.21/287.32/323.14。比例控制已表现出明确趋势。ratio 0.7 相对 base 的准确率差为 −1.81 pp，95% CI [−3.86,+0.16]；ratio 1 为 +1.18 pp，[−0.55,+2.99]。均不能据此宣称等价或准确率提升已确立。保留单训练 seed、锁定 1,269 题和原设置适配边界；本结果不替代统一 teacher 的 B6。

ASC 六强度与提示合并分析 `821265` 已完成。Qwen 在 scale 0/0.1/0.25/0.4/0.5/1 的正确数为 64/64/62/63/63/38；所有非零强度均未缩短平均输出。R1 对应为 59/59/61/60/56/4；scale 0.25 从 417.00 降至 402.20 tokens，配对差 −14.80 tokens，95% CI [−34.92,+5.83]，没有确立压缩收益。强度 1 的退化仍存在。

同一 16 题的 R1 未干预提示检查：项目提示 458.38 tokens，native question 634.25，native + step-by-step/boxed 1,415.63，raw 无 BOS 873.81，raw 有 BOS 966.38。早期 raw 运行漏加默认 BOS，已单独标注并补跑 `821246`；旧输出不冒充上游默认输入。论文 Section 5.1 允许官方参考作为 concise 监督，因此后续继续使用该路线，同时让校准、teacher forcing 和评估采用一致的 native CoT 提示及 boxed 参考答案。

新分支 `asc_r1_native_cot_v1` 的准备 `821268` 已完成，200 个校准候选题和 64 个 dev 题无交集；264 个提示及 200 个参考末尾格式全部核验。真实 R1 校准 `821271` 在 H200 运行；8 步检查 `821275`、完整 3,000 步拟合 `821276`、64 题三强度评估 `821277` 按成功依赖排队。最后评估请求 H100 的 8 小时时限，避免 short_gpu 的 2 小时上限。此为已观察 dev 上的提示/校准格式诊断，不改变 CES/KL 目标，不等于新结果已经完成。

DAP 公开数据已按固定 revision 下载九个 parquet 并核验上游 LFS SHA-256。精确规范化问题匹配得到 74,626 个唯一长短对应题；从独立 MATH 参考核验的长轨迹中固定 64 个开发题，ID 已保留、后续学生 SFT 必须排除。GPU 首次 smoke `821253` 在生成前因模板花括号失败；修复后的独立 v2 smoke `821255` 通过，完整改写 `821256` 和审计绘图 `821272` 均完成。原长轨迹平均 3,483.34 tokens，作者完整短轨迹 616.77，本次 Qwen-7B DAP 改写 565.34（−83.77%）、63/64 正确、零 cap hit。唯一错误输出最终给出 390，属于计数错误而非提取失败。原长轨迹 64/64 正确由筛选保证；仅测得改写生成 464.59 秒，不含不可恢复的原教师生成成本。原论文使用完整 DeepSeek-R1 改写，本检查保留该模型适配差异。

SAE 四分片和合并 `821221` 已全部完成 1,000 个新问题，平均 292.695 tokens、零 cap hit；统一复核为 941/1,000，smoke 为 8/8。400 discovery / 300 dev / 300 confirmation 划分保持冻结。这里每题仅一个原始候选，尚未完成特征读回、方向/随机对照、剂量或窗口因果实验。

已复核六个新完成标记的全部直接文件绑定，并检查 TokenSkip/DAP 图。来源下载在计算节点因外部 HTTPS 超时失败的 `821245`、被完整 BOS 队列替代的待运行审计 `821240`、模板失败 `821253` 均保留记录。主目标仍未完成：继续 ASC 原设置忠实性检查、SAE 因果对照、MATH 划分/cap 冻结，再进入统一 teacher 的 B0–B7 学生 SFT 和后续规模扩展。

[统一评分](results/phase13_baseline_expansion_v1/exploratory/uniform_answer_review_v2/finalized/report.md)、[TokenSkip](results/phase13_baseline_expansion_v1/exploratory/tokenskip_author_analysis_v3/report_zh.md)、[ASC 六强度及提示](results/phase13_baseline_expansion_v1/exploratory/asc_method_analysis_v3/report_zh.md)、[DAP 长轨迹](results/phase13_baseline_expansion_v1/exploratory/dap_long_trace_analysis_v1/report_zh.md)、[ASC 新提示配置](configs/phase13_asc_r1_native_cot_v1.json)。

## 2026-09-13 13:06 EDT：ASC 留出目标诊断完成，识别强干预下的分布偏移

新增 `13_15_diagnose_asc_objective.py`，复用已验证的 ASC 损失、响应 mask 与 full-vocabulary KL，在 16 道不属于向量拟合的固定 dev 题上比较 scale 0/0.1/0.25/0.5/1。两个作业 `821222`（Qwen/L40S）、`821223`（R1/H200）均 COMPLETED，所有 160 条测量及来源/代码/结果绑定已复核；未训练新向量或生成新答案。

强度 1 时，Qwen 的 concise NLL 增加 1.8929，verbose NLL 增加 11.1023，CES 却降至 0.0021；两个目标都变得更不可能，长目标退化更多。数学响应上下文 KL=11.0808，原独立 WikiText KL 仅 0.018245。R1 concise NLL 改变 −0.2421、verbose NLL 增加 3.7516，数学 KL=3.6553，WikiText KL=0.028433。该检查支持“排序改善不等于生成分布稳定”的失败解释，不证明全部退化仅由一种原因造成。

scale 0.1/0.25 的数学 KL 较低且 concise NLL 有小幅下降，但尚未测量这两个强度的完整实际生成。下一步做已观察 dev 上的低强度生成，并区分 native chat/项目提示与上游旧版 raw-question 输入方式；不能把旧版 raw 输入或 MoD 向量直接冒充正式 CES 配方。论文主表 R1-7B 的 vanilla 长度为 GSM8K 1,080、MATH-500 3,984 tokens；不能把 3–4K 一概当作其 GSM8K 基线长度。

[Qwen 诊断](results/phase13_baseline_expansion_v1/exploratory/asc_objective_diagnostic_v1/qwen/report.md)、[R1 诊断](results/phase13_baseline_expansion_v1/exploratory/asc_objective_diagnostic_v1/r1/report.md)。TokenSkip replica 评测、SAE 四分片及后续审计仍未全部完成，主目标保持进行中。

## 2026-09-13 13:02 EDT：SAE 新题准备与 L40S 检查完成，四分片已运行

S1b 输入准备 `821215` 完成：排除 881 道历史 SAE 问题，6,592 道候选中固定 8 道 smoke 和 1,000 道机制题（400 discovery / 300 dev / 300 confirmation），没有额外近似匹配。五类格式/词汇参考各 1,008 条；999 条具备已验证算术的单数字错误对照，其余 9 条记录为缺失该对照。checkpoint、层 17 / TopK 64、历史八特征、BF16 和三个随机 SAE 集合均已冻结。

L40S `821216` 已完成 8 题检查，无 cap hit，峰值 14,618.46 MiB，GPU UUID 已登记。机器评分为 7/8；另一个最终答案实际为 11，答案句“Each of the 5 employees pays $11”被 v3 提取成 5，属于已确认的提取假阴性。[问题记录](results/phase13_baseline_expansion_v1/exploratory/sae_mechanism_inputs_v1/baseline/smoke_review_v1/review.json) 保留来源哈希、原评分和人工检查方式；通用自然语言数量提取仍需修正/审查，不能宣称 v3 消除了全部格式问题。

四个 250 题分片均已确认 RUNNING：`821217` / `821218` 使用 H200，`821219` / `821220` 使用 L40S；审计合并 `821221` 依赖四片成功。此处每题仅生成一个原始候选，用于后续机制测量；保留错误、截断及格式失败，不按当前评分筛选题目。方向学习、同状态读回及正式准确率比较前须统一处理上述提取问题。

TokenSkip replica `821086` 仍运行，原总审计 `821140` 与修正评分分析 `821211` 尚待依赖完成。ASC 当前压缩失败的结论、S1b 后续因果对照、长轨迹 DAP 与统一学生主表均继续保留为未完成工作。

## 2026-09-13 12:53 EDT：完整 TokenSkip SFT 与 ASC 扫描完成；评分修复和 SAE 新题准备

TokenSkip replica 作业 `821085` 已完成 2,238 个优化步（实际 epoch 2.996317），训练阶段 2,023.39 秒，peak allocated GPU memory 7,536.88 MiB，252 个 LoRA-B 张量非零；最终 adapter 和训练绑定已复核。作者 adapter 的 11 条件评估 `821022` 全部完成。Replica 11 条件评估 `821086` 运行中，总审计 `821140` 等待其完成，不能将部分曲线当作完整复现。

ASC 两个模型的 3,000 步拟合及 64 题 × 3 强度生成均完成。统一 GSM8K v3 提取器的 [完整分析](results/phase13_baseline_expansion_v1/exploratory/asc_method_analysis_v2/report_zh.md) 已完成（`821210`）：Qwen 强度 0/0.5/1 正确数为 63/62/34，R1 为 59/56/4；两者都没有压缩收益，R1 强度 1 平均 3,286.92 tokens、31.25% cap hit。R1 独立 WikiText KL=0.028433，高于 0.02。当前不支持声称复现了 ASC 的压缩效果，更不能直接将退化配置作为主表对手；仍需排查提示、校准文本与强度。

v2 GSM8K 提取器会漏掉行内 Answer 和包含背景数字的答案句。独立 v3 修复通过 5 组回归检查，保留全部原始预测和评分。ASC v2 报告已使用新规则；TokenSkip 独立重评分与曲线 `821211` 依赖原始总审计。分析首次启动 `821207` 因快照内相对路径失败；`821208` 是同错误的本项目待运行分析，已取消，替换为绝对路径启动，原失败记录保留。

多类型数学 grader v2 审计 `821175` 已完成并复核绑定：7,487 MATH train、500 MATH-500、1,269 GSM8K、1,269 GSM8K-Hard、254 AQuA、674 OlympiadBench，共 11,453 条参考答案全部可判分，12 组回归检查通过。v1 的 73 项失败完整保留。自比较通过只证明本参考集的解析覆盖，不能替代语义正确性检查或最终数据划分/cap 冻结。

S1b 新题输入实现已加入，4 组去重划分、算术反例及对齐测试通过，CPU 准备 `821215` 运行中：先排除历史 SAE 题和测试近似题，再固定 8 个独立 smoke 题及 400/300/300 的 1,000 题；构造格式/词汇/计算错误对照与 token/span 对齐。准备完成后先跑 GPU smoke，再按四分片生成原始轨迹。此阶段不等于完整同状态读回、方向/剂量/窗口机制验证。

当前仍需完成 TokenSkip 全曲线与审计、ASC 忠实性诊断、长轨迹 DAP 检查、S1b 因果对照和统一教师下 B0–B7 的学生 SFT。主目标保持进行中。

## 2026-09-13 12:15 EDT：TokenSkip 完整训练已启动；MATH 来源审查完成

L40S 路由再次从 Slurm 核对，并在 AGENTS.md 明确其与 H100/H200 一样属于已授权资源：`jekml_gpu / gpu_partners / short_gpu / gpu:l40s:1`，当前 QOS 时限 2 小时。在已授权项目内切换到适合的 L40S 无需再次询问机器白名单。

TokenSkip 已进入真实比例条件化 SFT。作者固定 commit 的公开数据为 6,638 道唯一 GSM8K 训练题，完整保留输入、输出和比例；审计发现零测试重合、零最终答案分歧、零截断。seed 42 划分为 5,974 train / 664 validation，最长 native-chat 序列 661 tokens。首次准备 820987 因通用 JSON reader 只接受 object 而失败；修复 array 读取后 821000 完成，失败日志保留。

- `821020`：真实 Qwen-3B LoRA smoke 训练完成，4 个优化步、252 个非零 LoRA-B 张量，峰值 allocated memory 7,533.1 MiB。
- `821057`：重新加载 smoke adapter，比例 0.5 / 1.0 各两题推理完成；这是接口检查。
- `821021`：base 的完整 1,269 题评估完成，准确率 82.90%，平均 316.83 tokens，cap-hit 4.10%。固定 512 cap、greedy BF16 SDPA；本分数属于当前 TokenSkip 设置。
- `821022`：作者公开 adapter 的 11 个比例/cap 条件运行中。附带 trainer_state 记录 step 600 / max_steps 2238，不能据此宣称它是完整三轮训练模型。
- `821085`：本项目完整 replica SFT 运行中，目标 2,238 优化步；12:15 时已超过 620 步、梯度和 loss 有限。配置 LoRA rank 8 / alpha 16 / dropout 0、有效 batch 8、3 epochs。最终实际 epoch 将单独登记。
- `821086`：replica 的 11 条件评估依赖完整训练；`821140`：CPU 总审计和准确率—长度图依赖 base、作者及 replica 评估，预期审计 29,187 条预测。均为排队任务，尚未完成。

TokenSkip 数据和训练源码冻结于 `results/phase13_baseline_expansion_v1/exploratory/tokenskip_author_sft_v1`。固定 512 cap 为主要控制曲线，`floor(512*ratio)` 为上游推理配置对照；ratio 1 两者共用一次运行。作者数据复现不替代统一 teacher / 共同问题集上的 B6，也不完成主蒸馏矩阵。

Qwen ASC 完整生成 `820736` 已完成：原始 grader 下 scale 0 / 0.5 / 1 的正确数为 63 / 62 / 34（各 64 题），平均长度 265.56 / 276.67 / 367.45 tokens，当前没有压缩收益；强度 1 出现内容退化和 5 个 cap hit。R1 完整拟合 `820759` 已完成，独立通用文本 KL=0.028433，高于 0.02 的目标预算；有限 hinge 惩罚不保证约束成立。R1 生成 `820760` 仍运行。统一 grader 的独立事后分析 `821083` 依赖该生成；原始输出和旧分数保持不变。不能据这些方法检查宣称 SAE 优于 ASC。

MATH 来源审查 `821157` 已完成，独立 [报告](results/phase13_baseline_expansion_v1/preparation/math_data_review_v1/report_zh.md) 和完成标记已落盘：排除 12 个近似评测训练题、1 个训练重复，剩余 7,487 道题供后续开发/校准划分；17 项 gold 修正逐项保留理由和原始参考。GSM8K-Hard 3 个缺失 parent 已补齐，全部 1,319 条为一对一映射，按 GSM8K parent 的锁定测试区间保留 1,269 条。仍需验证 radix/choice/multi-answer grader、GSM-Hard 浮点容差，并冻结划分和 SAE 新问题机制检查；不能把来源审查标记当成正式训练协议。

新分析入口为 `13_9_analyze_baseline_methods.py`、`13_10_analyze_tokenskip.py`、`13_11_review_math_data.py`；对应作业使用独立源码快照。目标仍未完成：完整原设置方法结果与统一教师蒸馏对照都要继续核查，不能以 smoke、已提交作业或压缩文本替代 baseline 复现。

## 2026-09-13 11:42 EDT：ASC / DAP / TokenSkip 实际复现进度

已新增正式 ASC-CES 损失、完整词表 forward-KL、解码位置一致的残差干预；七项数值/接口测试通过。DAP 附录完整难度框架已恢复，TokenSkip 保留 LLMLingua-2 和训练/推理比例条件格式。执行说明见 [baseline reproduction](docs/experiments/phase13/phase13_baseline_reproduction.md)。

`method_check_v1` 的数据、模型哈希及源码已冻结（CPU 820710 完成）：100 对训练校准题、2,000 条 WikiText 训练段落、64 条独立 WikiText KL 检查段落、64 道已观察 GSM8K dev 题。校准题与 dev 题不重叠；dev 压缩记录禁止用作学生训练。

- ASC 8 步真实 7B 检查完成（820711，H200），只训练 3,584 维向量，非零有限梯度，GPU peak 16,705.8 MiB；两题解码检查完成（820714，L40S）。这些是 smoke，不是压缩效果结论。
- TokenSkip 六比例压缩完成（820713，64 题 / 384 记录 / 64 个单比例格式样本），保留比例 0.5/0.6/0.7/0.8/0.9 的实际均值为 0.499/0.597/0.698/0.807/0.895；比例条件 SFT 与推理尚未完成。
- DAP 64 题改写完成（820712），64/64 正确、无 cap hit；完整输出平均 232.69 tokens，源 trace 平均 226.63。当前短 instruct trace 上未观察到压缩收益，不能只截取 Solution 来改变这一结果。
- Qwen-7B ASC 完整 3,000 步拟合已完成（820716，L40S）：优化用时 1,205.38 秒，peak 17,918.91 MiB，64 条独立通用文本上的平均 KL=0.018245；向量和逐步记录已封存。64 题三强度生成运行中（820736），完整准确率/长度结论尚未成立。
- R1-7B 已生成 128 条原始校准尝试（820753），覆盖 104 个候选问题。发现旧 GSM8K 提取器将 Markdown 标题、嵌套 boxed 单位及 TeX 千位分隔符误判；单独冻结 `asc_r1_method_check_v2`，重评分找回 22 条正确记录（总正确率由旧规则的 78.125% 修正为 95.3125%），未新增生成调用，按预定题序选取 100 对。v1 的 78.125% 不能当作 R1 数学能力结果。
- 修正后的 R1 真实 smoke fit / generation 完成（820756 / 820757），完整 3,000 步训练运行中（820759，H200），后续 64 题三强度生成按依赖排队（820760）。
- MATH 数据 preflight 完成（820755）：7,500 训练题和五评测集已固定 revision；与评测集零规范化精确重合，训练集内 1 个重复，12 对近似题待审查，17 个训练答案需规范化/修复，GSM8K-Hard 的 1,316 个 parent 已精确匹配、3 个待核对。结果位于 `results/phase13_baseline_expansion_v1/preparation/math_data_preflight_v1`；这不是正式训练冻结标记。

用户新增授权的 L40S 已写入 AGENTS.md，路由为 `jekml_gpu / gpu_partners / short_gpu / gpu:l40s:1`。原 H100/H200 授权继续有效。

当前仍未完成扩大后的学生蒸馏矩阵或完整 baseline 复现。后续继续核查 ASC 完整结果、原 R1 设置、MATH 数据去污染与数学 grader，并执行实际学生 SFT；不能将本节方法组件结果改称 7.5K/25K 主结果。

Qwen2.5-3B-Instruct 权重已下载并固定为 `aa8e72537993ba99e69dfaafa59ed015b17504d1`，供后续 TokenSkip 原设置/学生检查使用；尚未提交 TokenSkip SFT。新数学评分依赖位于独立 `phase13_math_grading_v1` overlay。`math_grading.py` 保持数据 preflight 绑定版本，后续 GSM8K 修复单独位于 `gsm8k_grading.py`；冻结的 R1 v2 snapshot 包含相同修复，运行时不引用可变源码。

## 2026-09-13：Phase 13 基线扩展已建计划；Answer区域诊断完成

用户要求将ASC-CES、DAP、TokenSkip纳入实际对照，分阶段推进7.5K来源池、25K目标和多模型/五评测集；随后明确批准NCSU现有账户与分区，并要求更新AGENTS.md。当前规则已改为NCSU工作区、用户youyang7、jekml账户和H200/H100路由，历史C集群规则独立归档，原实验保持不变。

S0已完成：三个官方仓库commit及源文件/论文归档；参考矩阵84格=48+12+12+12，零训练提交。发现ASC公开HEAD仍是长短末位激活差，需要另实现正式CES/KL；DAP需要按论文恢复难度感知改写；TokenSkip需保留训练/推理的比例条件。

S1a已完成：复用132题、1,036条既有test轨迹的非零事件，验证全序列与first64均值可重建，统计16特征。八个short feature在字面Answer的激活质量为52.77%–98.04%；全序列paired d为2.173–2.775，剔除答案标题及后缀后正文为−0.102至+0.129。这是事后格式区域诊断，不是新因果/语义确认。1,035条有显式标题，本支持没有标题在前64 token内出现。

下一步为新问题格式反证/同状态目标读回（S1b）与三方法实现/小规模真实验证（S2）；之后冻结新数据去污染、grader与长推理cap（S3），再进入SFT扩展。矩阵准备和CPU诊断不代表新baseline已复现或新训练已完成。

[实验计划](docs/experiments/phase13/phase13_baseline_expansion_experiment_plan_zh.md)、[机器可读计划](configs/phase13_baseline_expansion_v1.json)、[源码核对](results/phase13_baseline_expansion_v1/preparation/source_audit_v1/source_audit.json)、[参考矩阵](results/phase13_baseline_expansion_v1/preparation/reference_matrix_v1/matrix.json)、[诊断报告](results/phase13_baseline_expansion_v1/exploratory/answer_marker_diagnostic_v1/report_zh.md)、[诊断完成标记](results/phase13_baseline_expansion_v1/exploratory/answer_marker_diagnostic_v1/COMPLETE.json)。

## 2026-09-13：固定878题的多答案训练已完成

`ncsu_multi_answer_v1` 已完成：保留878道原题，53题补生成496条候选，候选池共7,520条；每题1／2／4条不同正确答案，以及最短答案重复2／4次，两种方法、三个训练seed，共30个adapter与31组各1,269题的评估。最终审计于06:33:26 EDT通过，1,539次哈希比对覆盖940个文件、156个完成标记，复核39,339条预测。

| 每题不同答案数 | 无干预准确率 | SAE准确率 | SAE−无干预（百分点，95% CI） |
| --- | ---: | ---: | ---: |
| 1 | 70.08% | 69.98% | −0.11 [−2.15, +1.89] |
| 2 | 71.39% | 70.42% | −0.97 [−2.92, +0.97] |
| 4 | 69.50% | 69.79% | +0.29 [−1.84, +2.42] |

九项登记比较区间均跨零、Holm p均为1.0000：未确立增加同题答案数扩大SAE优势，也未确立不同答案优于重复最短答案。重复2／4次时，无干预为70.76%／70.92%，SAE为70.08%／70.27%。本轮base为66.51%。SAE在不同答案三档的学生输出分别短11.44%／12.33%／13.06%；准确率收益未确立。

原统计节点启动异常，统计819781及待运行审计819782取消；同一冻结代码在健康CPU节点重提交为820079／820080并成功完成。共38个成功作业及2个保留的取消记录。两组各22题的最短答案因补生成改变，已重训k1基线；重复对照匹配记录、曝光和步数，不匹配监督token。保持探索性GSM8K、单SAE seed边界。

[完成状态Markdown](docs/experiments/phase12/ncsu_multi_answer_status_20260913.md)、[执行协议](docs/experiments/phase12/ncsu_multi_answer_v1.md)、[完整结果与图](results/ncsu_multi_answer_v1/exploratory/analysis/report_zh.md)、[同题原始训练答案](results/ncsu_multi_answer_v1/exploratory/analysis/training_examples/report_zh.md)、[最终审计](results/ncsu_multi_answer_v1/exploratory/FINAL_AUDIT.json)、[总完成标记](results/ncsu_multi_answer_v1/exploratory/EXPERIMENT_COMPLETE.json)。

## 2026-09-13：NCSU SAE 干预与学生对照已完成

独立 `ncsu_sae_intervention_v1` 已完成 dev/test 教师干预、10,572 条全量候选、878 题共同支持、24 个学生 adapter 和 25 组各 1,269 题的评估。42 个登记作业全部 `COMPLETED / 0:0`；最终分析 `817324` 于 2026-09-13 00:18:08 EDT 完成，实际 GPU 作业均使用 H200 `gpu38`。交付复核的 1,248 次哈希比对覆盖 916 个已绑定文件、89 个完成标记；31,725 条评估记录及训练步数、预算与最终 LoRA 复核通过。

Dev 固定选择 `short_rho0.3`：主 SAE layer 17 / TopK 64 的 8 个短相关特征、30% 相对 hidden norm，从首个生成 logit 开始干预。64 题后续检验缩短 15.10%，无干预与目标均 64/64 正确；全量 881 题四候选中，平均长度由 250.27 降至 217.07 tokens（−13.27%），正确候选率为 98.69% / 98.89%，差值区间跨零。

| 预算 | 无干预蒸馏 | SAE 蒸馏 | 匹配随机 | 自然 short | SAE − 无干预，百分点 [95% CI] |
| --- | ---: | ---: | ---: | ---: | ---: |
| 等样本 | 69.92% | 70.84% | 68.87% | 66.93% | +0.92 [−1.34, +3.23] |
| 等目标 token | 70.42% | 70.34% | 68.87% | 66.96% | −0.08 [−2.18, +1.97] |

本轮 Base 66.51%。SAE 相对无干预和匹配随机的准确率收益均未确立；相对自然 short 通过 Holm 校正，但自然 short 与新生成组的采样、选样规则不同，不能单独归因于 SAE。学生输出相对无干预缩短约 11.2% / 11.3%。结果保持探索性 GSM8K、单 SAE seed，`formal_claim_allowed=false`；原历史分支不变。

[最终状态 Markdown](docs/experiments/phase12/ncsu_sae_intervention_status_20260912.md)、[总完成标记](results/ncsu_sae_intervention_v1/exploratory/INTERVENTION_COMPLETE.json)、[交付审计](results/ncsu_sae_intervention_v1/exploratory/setup/final_delivery_audit_20260913.json)、[完整学生报告](results/ncsu_sae_intervention_v1/exploratory/student_followup/analysis/report_zh.md)、[执行协议](docs/experiments/phase12/ncsu_sae_intervention_v1.md)。

## 2026-09-12：NCSU Phase 1/2 新样本复现已完成

用户要求的新 teacher 样本、short/medium/long student SFT 与 SAE 分析全部完成。独立实验 `ncsu_phase12_reproduction_v1` 的最终作业 `811975` 为 `COMPLETED / 0:0`；总完成标记、104 个绑定产物及冻结源码哈希复核通过。共生成 14,096 条新轨迹，三组各 881 道训练题、三个训练 seed，10 个完整评估（base + 9 adapters），每个 1,269 条逐题预测。六组 SAE 均完成训练与特征评分；激活无重复或缺失。H100 与健康 L40S 分担工作；L40S 失败尝试保留，未修改冻结科学配置或干预其他用户进程。

平均 GSM8K 准确率：base **67.45%**、short **66.67%**、medium **66.33%**、long **64.20%**。short 比 long 高 2.47 个百分点，但三项 Holm 校正后 p=0.0852；short 低于 base，**本轮不支持“short SFT 提升原始 student”**。主 SAE（layer 17, k=64）确认 8 个 short-associated 和 8 个 long-associated 特征，不能据此推断因果收益。结果保持探索性 GSM8K 边界。

[中文完成报告](results/ncsu_phase12_reproduction_v1/exploratory/analysis/report_zh.md)、[总完成标记](results/ncsu_phase12_reproduction_v1/exploratory/EXPERIMENT_COMPLETE.json)、[交付复核](results/ncsu_phase12_reproduction_v1/exploratory/setup/final_delivery_audit.json)、[方法及调度记录](docs/experiments/phase12/ncsu_phase12_reproduction.md)。模型位于 `checkpoints/ncsu_phase12_reproduction_v1/`，历史结果保持独立。

更新时间：2026-09-11（America/New_York）；本次整理 2026-09-10 的实验与最终审计。本文件作为当前任务、阻塞原因与验收条件的统一入口；各实验的冻结协议、原始报告和历史产物保持独立。

## 2026-09-11：本地独立重跑；历史 B3/B4 输入仍缺失

**独立重跑已获明确授权并启动**：用户回复“重跑”。新实验 `phase7_local_full_sequence_rerun_v1` 全部写入 [C31 本地工作区](/mnt/local/youyang7/SAE_long_short_c31/workspaces/continuation_20260911/README.md)，在保留的 allocation `279747` 内使用四张 A6000、每卡一个进程生成 881 题 × 16 候选。冻结 [主配置](/mnt/local/youyang7/SAE_long_short_c31/workspaces/continuation_20260911/configs/phase7_local_full_sequence_rerun_v1.json) 与 [来源绑定](/mnt/local/youyang7/SAE_long_short_c31/workspaces/continuation_20260911/results/phase7_local_full_sequence_rerun_v1/exploratory/protocol/FROZEN.json) 已落盘；保留 617/132/132 划分、六 SAE 的 seeds 17/42/73、全序列 v2 筛选门槛和后续校准门槛。此处是新生成、新训练的独立实验，原 B3/B4 产物的缺失状态不变。候选池、四分片激活、六 SAE 的 1,500 步训练与共同评分均已完成；核心审计通过，逐个核对 3,507,902 个激活 token。全序列筛选得到一组稳定 long 特征；5,376 条配对校准完成，选定 prefix16 / long_suppress=0.5。该轮已完成全题集三条件生成、36 个学生 adapter、37 个完整评估及最终审计；本次重新核对总完成标记绑定的审计、报告和学生汇总哈希通过。[完成报告](/mnt/local/youyang7/SAE_long_short_c31/workspaces/continuation_20260911/results/phase7_local_full_sequence_rerun_v1/exploratory/report_zh.md)。开发集发现也取消 first64 符号与效应排序限制，仅按全序列效应选候选；这是新登记实现，不能称为原 v2 的逐字复现。[生成日志](/mnt/local/youyang7/SAE_long_short_c31/workspaces/continuation_20260911/results/phase7_local_full_sequence_rerun_v1/exploratory/logs/generation_launch.log)。

**新增 feature 数量方案，离线检查完成**：用户提出扩大 E2 干预 feature 集合。独立核对原 E2 dev 排名，full-trace-balanced 的每侧 top64 全部通过原 dev 门槛；joint64 相对 joint16 的 decoder 方向夹角为 40.46°。原 joint16 实为 16 short + 16 long。已形成 [数量扩展与学生收益方案](/mnt/local/youyang7/SAE_long_short_c31/workspaces/continuation_20260911/docs/phase8_feature_breadth_student_gain_proposal_zh.md) 和 [离线分析完成标记](/mnt/local/youyang7/SAE_long_short_c31/workspaces/continuation_20260911/results/phase8_feature_breadth_feasibility_v1/exploratory/OFFLINE_ANALYSIS_COMPLETE.json)。用户已明确要求编辑脚本并执行；独立 `phase8_feature_breadth_intervention_v1` 已冻结并在 C31 四卡启动，顺序完成 smoke、13 条件教师扫描、学生开发集选择和最终多种子两预算评估。[执行入口](/mnt/local/youyang7/SAE_long_short_c31/workspaces/continuation_20260911/docs/phase8_feature_breadth_execution_zh.md)。当前启动不代表实验完成或已获得学生收益。

**本地执行更新**：用户随后要求在 `/mnt/local` 完成实验。已建立 [C31 本地工作区](/mnt/local/youyang7/SAE_long_short_c31/workspaces/continuation_20260911/README.md)，复制代码、legacy 训练源码、三份各 881 条 rank 数据和 1,269 题评估集，并将 `/var/tmp` 保留的固定 revision 7B 教师复制到本地 HF cache。教师四个权重分片、学生权重、数据与评估集登记哈希均通过；教师和学生分别在 GPU 1、2 完成实际加载和简短生成，8 项筛选/缓存测试通过。[本地准备记录](/mnt/local/youyang7/SAE_long_short_c31/workspaces/continuation_20260911/results/local_staging/LOCAL_PREPARATION.json)。7B 路径阻塞已解决；最新本地预检仍缺 23 个筛选文件和 13 个后续生成文件。该准备阶段未启动实验；随后用户明确选择独立重跑，最新执行状态见上一段。

此前的首次续跑预检先核验了尚未完成的 B3/B4 输入。C31 allocation `279747` 内重新执行 B3 预检，结果为 **`blocked_missing_inputs`**：23 个去重后的筛选路径、16 个生成前置路径不可用；后者尚不含教师 index 恢复后才可枚举的权重分片。已安装依赖的版本信息已刷新，其中 TRL 为 0.9.6；该次预检没有完成教师加载或 GPU 实验验证。[首次预检](results/project_continuation_20260911/b3_preflight.json)、[恢复清单与历史哈希参考](results/project_continuation_20260911/recovery_manifest.json)。

C31、C32 和 C49 均未发现 BeeGFS 挂载，原 `phase1_legacy_trace_sae_distillation_v1` 路径不可访问。在已检查的 home、C31 本地目录及 C32/C49 临时目录中未找到可用的原六 SAE、原评分汇总和 mixed corpus 副本；这不代表原存储上的文件已删除。C49 只读检查作业 `280948` 已结束；C30 检查作业 `280947` 因节点不可用一直排队，已撤销该次新建检查，C30 的存储状态未知。用户原有 C31/C32 allocations 和 `gg` 均保持开放。[C31 快照](results/project_continuation_20260911/storage_c31.txt)、[C32 快照](results/project_continuation_20260911/storage_c32.txt)、[C49 快照](results/project_continuation_20260911/storage_c49_280948.txt)、[检查作业状态](results/project_continuation_20260911/storage_jobs.txt)。

**B3 尚未执行筛选，B4 尚未启动生成或训练。** 恢复条件是让原 BeeGFS 实验目录重新可读，或提供包含原评分、六 SAE 和轨迹池的备份根目录；随后按登记哈希验证并运行独立 v2 筛选，再依据词汇反证与校准门槛推进 B4。E2 的三个 SAE 和 2,643 条当前 rank 轨迹不替代 B3/B4 的历史输入。下文 9 月 9 日“输入已恢复”仅代表当日快照，不能用于当前准入。

## 2026-09-10：新增 E2，当前数据的长度控制 SAE 与加强干预

用户已要求完成相关实验并提供 docs 报告。独立实验 `phase6_local_length_controlled_strength_v1` 已从 C31 Arrow 缓存逐字节恢复原 short/medium/long 各 881 条数据，三份 SHA256 与历史审计一致。使用这 2,643 条正确 rank 轨迹重训三个 SAE，不替代历史 14,096 条原始候选池的训练或 B3/B4。

E2 已完整完成：四个提取分片、三个 SAE 的 1,500 步训练、共同位置评分、27 条件教师扫描（3,456 条）、结构匹配随机补充（384 条）、881 题四候选三条件生成（10,572 条），以及共同 873 题上四条件、两预算的 8 组学生训练和各 1,269 题评估。完成标记、201 项输入/模型哈希检查、逐题完整性审计、图表和展示复核均已落盘。使用 C31/C32 保留的 allocations `279747`、`280415`，keepalive 保持开放；新数据与最终 checkpoint 位于 home，临时激活留 C31 `/mnt/local`。

结论：按轨迹等权采样基本消除长度曝光偏差；选定 short16/long16 联合方向、30% 范数、start0 的干预在全量候选中缩短输出 20.00%，但正确候选率从 98.89% 降至 96.34%。学生等样本为 SAE 69.66% / 无干预 69.35%，等 token 为 SAE 70.37% / 无干预 70.53%；六个计划配对比较的 95% 区间均跨零，未确立学生涨点。同数据、同 seed 的随机对照重复运行仍相差 1.34 个百分点，模型权重和早期训练日志已不同；原配方未开启严格确定性，具体底层原因未隔离，不能将单次执行的微小差值解释为稳定收益。

最终交付：[E2 中文实验报告（展示复核版）](docs/experiments/phase6/phase6_local_length_controlled_strength_report_zh_v2.md)、[英文一页 PPT 文字](docs/presentations/sae_representation_one_slide_en.md)、[完成标记](results/phase6_local_length_controlled_strength_v1/exploratory/EXPERIMENT_COMPLETE.json)、[完整审计](results/phase6_local_length_controlled_strength_v1/exploratory/FINAL_AUDIT.json)、[展示复核](results/phase6_local_length_controlled_strength_v1/exploratory/PRESENTATION_REVIEW_COMPLETE.json)。保留单种子、当前已观察数据的探索性边界；E2 不替代历史 B3/B4 协议。

### E2 关键结果与证据

- **长度控制 SAE**：三个 SAE 的实际训练曝光与轨迹长度的 Spearman 相关为 0.9458 / −0.0146 / −0.0146；共同完整测试轨迹的重构 explained variance 为 0.7741 / 0.7738 / 0.6582。轨迹等权明显降低曝光偏差，prefix64 的优势限于早期位置，不能由重构指标推断学生准确率。[实际曝光审计](results/phase6_local_length_controlled_strength_v1/exploratory/training_exposure_audit.json)
- **教师干预**：dev 选定 `full_trace_balanced__joint16__rho0.3__start0`。在另外 64 题上，平均长度从 261.61 降至 198.75 tokens（−24.03%），正确题数从 63 降至 62；这不证明正确率无损。全量 881 题、每条件四候选生成中，长度从 250.07 降至 200.06 tokens（−20.00%），正确候选率下降 2.55 个百分点，按题配对 95% 区间为 [−3.52, −1.62]。[全量教师统计](results/phase6_local_length_controlled_strength_v1/exploratory/student_followup/teacher_data_summary.json)
- **学生蒸馏**：三个生成条件分别有 880 / 874 / 881 题具备正确完整候选，最终四条件共同支持为 873/881 题（99.09%）；原自然 short 也在同一支持集重训。seed 17 下完成四条件 × 两预算的 8 个 adapter，各评估 1,269 题。[共同支持与预算](results/phase6_local_length_controlled_strength_v1/exploratory/student_followup/DATA_COMPLETE.json)

| 预算 | 无干预 | SAE 联合干预 | 匹配随机 | 同题集原自然 short | SAE − 无干预，百分点 [95% CI] |
| --- | ---: | ---: | ---: | ---: | --- |
| 等样本 | 69.35% | 69.66% | 69.90% | 69.98% | +0.32 [−2.13, +2.76] |
| 等目标 token | 70.53% | 70.37% | 71.24% | 70.84% | −0.16 [−2.52, +2.21] |

六个计划配对比较的区间均跨零，Holm 校正 p 值均为 1.0000；未确立 SAE 学生准确率收益。区间只覆盖评估题目重采样，不包含训练种子或执行变异。等 token 使用完整轨迹重复，优化步数与题目权重不相等。[学生指标与全部配对比较](results/phase6_local_length_controlled_strength_v1/exploratory/student_followup/STUDENT_COMPLETE.json)

![E2 学生准确率与配对差异](figures/phase6_local_length_controlled_strength_v1/presentation_v2/05_student_accuracy.png)

**复现限制与归档状态**：同数据、同初始化、同 seed 的两次随机对照分别答对 887 与 904 题，95 题的正确状态不同；差异已出现在训练阶段，底层原因尚未隔离。两次执行不构成跨种子复现，也不能解释为预算收益。[重复对照审计](results/phase6_local_length_controlled_strength_v1/exploratory/student_followup/repeated_control_audit.json)。总完成标记时间为 2026-09-10 13:00:11（America/New_York）；`formal_claim_allowed=false`。本次文档整理重新核对了完成标记绑定的最终审计、原报告，以及展示复核绑定的完成标记和 v2 报告 SHA256，全部一致；201 项输入/模型检查引用昨日最终审计，未在本次重新执行全量模型核验。

## 当前结论

**历史长度采样消融、有效配方 SFT 复现和六个 SAE 已完成；9 月 10 日新增 E1 原 short 复查与 E2 长度控制 SAE → 干预生成 → 学生对照也均已完成。E2 实现了有正确率代价的教师压缩，尚未确立学生准确率提升。历史 B3 全序列重筛选和 B4 对应协议的干预蒸馏仍未完成。**

这里的“分数”指 GSM8K `test[50:1319]` 的 1,269 题答案正确率，不能用训练 loss 下降代替。历史 short SFT 的点估计高于 base，但相对 base 的配对区间仍跨零，不能写成已确立显著收益。

**新增 E1 已完成：本地环境使用恢复的原 881 题 short 数据训练，seed 17 达到 70.84%，相对同环境 base 68.01% 提升 2.84 个百分点。** 原四样本 smoke adapter 仍为 68.01%，没有有效权重更新。此单种子结果与上文历史三种子均值分开报告。

2026-09-09 的 BeeGFS 读写和关键产物哈希核验通过；2026-09-10 重新检查时，C31、C32 均未挂载 BeeGFS。E1 通过节点缓存恢复数据并完成训练评估；历史 SAE / 干预分支未因此自动续跑。

## 2026-09-10：本地数据与 short 涨点验证

用户希望确认 `/mnt/local` 的数据能否达到历史 short 70.03% 附近及其涨点幅度。原本本地只有 4 样本、1 步的环境测试 adapter，数据路径实际指向 home 中的 Phase-0 smoke 数据；不存在一份已经完成全量重生成与评估的本地 short 数据集。

已从 C31 `/var/tmp` 保留的 Arrow 缓存恢复原 881 题 short JSONL，SHA256 与历史文件完全一致；从 home 保留预测的 **gold 字段**恢复 1,269 题评估输入，其 SHA256 也与原 fixture 完全一致。没有使用模型预测答案作 gold。

E1 使用 `/mnt/local/youyang7/SAE_long_short_c31` 环境，在现有 C31 allocation `279747` 内并行完成 base、原本地 smoke adapter、新 short seed-17 adapter 三个条件。新训练复用原 legacy TRL 配方，实际完成 1 epoch / 221 steps；评估固定 `test[50:1319]`、batch 32、greedy、512-token cap。数据和最终 adapter 已发布到 home 项目内，新实验的本地临时目录已清理。

| 条件 | 正确题数 / 1,269 | 正确率 | 相对同环境 base |
| --- | ---: | ---: | ---: |
| Base | 863 | 68.01% | — |
| 原四样本 smoke adapter | 863 | 68.01% | 0.00 个百分点 |
| 原 881 题 short，seed 17 | 899 | **70.84%** | **+2.84 个百分点** |

Short 净多答对 36 题；配对题目 bootstrap 95% 区间为 **[+0.32, +5.36] 个百分点**，区间不包含训练种子变异。70.84% 比历史三种子均值 70.03% 高 0.81 个百分点，但不能据此声称单种子方案优于历史三种子协议。平均输出长度为 short 264.85、base 241.89 tokens，本轮涨点没有伴随相对 base 的输出缩短。

旧 smoke 的 196 个 LoRA-B 张量全部为零，其完整预测与 base 逐字节一致。源代码与单步学习率检查确认：warmup steps = ceil(0.03 × 1) = 1，唯一一次 optimizer.step 的学习率为 0，故没有产生有效 LoRA 更新；这不证明四条数据在正常训练下无效。[诊断证据](results/phase1_local_short_gain_check_v1/adapter_effect_diagnostic.json)

当前状态：**训练、三个完整评估、逐题审计和配对分析已完成**。这是一轮单种子原数据恢复复查，D1 的候选全量重生成仍未执行。

- [冻结配置](configs/phase1_local_short_gain_check_v1.json)
- [恢复数据与源码证据](results/phase1_local_short_gain_check_v1/IMPORT_COMPLETE.json)
- [运行日志](results/phase1_local_short_gain_check_v1/logs/launch.log)
- [结果与置信区间](results/phase1_local_short_gain_check_v1/analysis/report.md)
- [完成标记](results/phase1_local_short_gain_check_v1/LOCAL_SHORT_GAIN_CHECK_COMPLETE)
- [恢复的 881 题 short 数据](results/phase1_local_short_gain_check_v1/imported/sft/short.jsonl)
- [可用 adapter](checkpoints/phase1_local_short_gain_check_v1/short__seed_17)

![本地环境完整测分与配对涨幅](figures/phase1_local_short_gain_check_v1/local_short_gain.png)

## 任务看板

| ID | 任务 | 当前状态 | 已有证据 / 尚缺什么 |
| --- | --- | --- | --- |
| A1 | 控制长 trace 在 SAE 训练中的采样权重 | **已完成**，2026-09-04 | 三种采样条件，原字典加两个新字典；[采样消融审计](results/phase2_sae_sampling_ablation_v1/exploratory/audit/sae_sampling_ablation_audit.json)通过 |
| A2 | 去掉答案、末尾位置与长度分母后的特征检验 | **按门槛结束**，2026-09-05 | 清理后筛选和两版激活读回完成；随机对照覆盖率失败，独立生成与 SFT 未触发；[Phase 5 最终审计](results/phase5_sae_clean_feature_causal_v1/exploratory/final_audit.json) |
| B1 | 历史有效数据与原 SFT 配方复现 | **已完成，复现门槛通过**，2026-09-05 | 881 题、short/medium/long × seeds 17/42/73，共 9 个 adapter 和完整评估；[复现报告](results/phase1_legacy_trace_sae_distillation_v1/replication/analysis/replication_report.json) |
| B2 | 在 B1 对应原始轨迹池上做 SAE 采样消融 | **已完成**，2026-09-05 | full-trace / prefix64 trace-balanced × 3 seeds，共 6 个 SAE；[完成标记](results/phase1_legacy_trace_sae_distillation_v1/sae/SAE_TRAINING_AND_SCORING_COMPLETE) |
| B3 | 取消 first64 同向准入条件的 v2 重筛选 | **当前缺输入，实验未执行** | 9 月 11 日重新预检：23 个筛选路径不可用，BeeGFS 未挂载；[当前预检](results/project_continuation_20260911/b3_preflight.json)、[v2 协议说明](docs/experiments/phase1/phase1_full_sequence_screen_c31_v2.md) |
| B4 | 有效配方上的干预数据 vs 自然数据 SFT | **未启动** | 仍缺特征筛选、词汇反证、因果校准、全题集候选生成与共同支持集；[原停止记录](results/phase1_legacy_trace_sae_distillation_v1/intervention_calibration_v1/NO_STABLE_FEATURES.json) |
| C1 | 早期 SAE 干预蒸馏 pilot | **已完成，无稳健改善证据**，2026-09-04 | 24 个 adapter、25 组评估；不能代替 B4；[最终审计](results/phase3_sae_intervention_distillation_pilot_v1/exploratory/final_audit/final_audit.json) |
| C2 | 增大教师干预强度的 sweep | **已完成，零条件通过缩短筛选**，2026-09-05 | 768 条输出，8 个完整分片；未包含新 SFT；[分析清单](results/phase3_sae_intervention_strength_sweep_v2/exploratory/strength_analysis/strength_analysis_manifest.json) |
| D1 | 停机期间提出的全量重生成 + 单种子三个 rank adapter | **待执行，未发现产物** | 已明确三种长度、候选不足按共同有效子集继续；具体 seed 与评估范围未获得上一轮回答，不记作已冻结 |
| E1 | 本地环境复查原 881 题 short 与四样本 smoke 的涨点 | **已完成**，2026-09-10 | Short 70.84%，base / smoke 68.01%；净涨 2.84 个百分点，单种子配对区间 [+0.32, +5.36] |
| E2 | 当前正确 rank 数据的长度控制 SAE、加强干预与学生蒸馏 | **已完成，探索性**，2026-09-10 | 3 个 SAE、教师扫描及结构匹配对照、10,572 条新候选、873 题共同支持、8 个学生及完整评估；教师缩短 20.00% 并损失 2.55 个百分点正确候选率，学生涨点未确立；[最终审计](results/phase6_local_length_controlled_strength_v1/exploratory/FINAL_AUDIT.json) |

### A1：长度采样消融实际回答了什么

三种 SAE 使用相同 layer 17、28,672 features、TopK 64、seed 17、250,000 个训练 token、1,500 steps。改变的是抽样方式，没有把长度标签或显式长度惩罚放进 SAE loss。

- `full_token_uniform`：整段 token 均匀抽样，长 trace 有更多机会进入训练。
- `full_trace_balanced`：先平衡各 trace 的训练贡献，再抽取位置。
- `prefix64_trace_balanced`：平衡 trace，并只从前 64 个位置抽样。

长度与训练抽样量的 Spearman 相关从 **0.768 降到 0.006**。在共同完整测试轨迹上，重构 explained variance 分别为 **0.7765 / 0.7763 / 0.6796**；只看共同 first64 位置则为 **0.7743 / 0.7748 / 0.7943**。prefix64 的重构优势局限于早期位置，不能把不同评价分布的数值直接比较为总体优劣。

三个字典间严格稳定匹配的候选数为 **0**。因此，采样消融已完成，但其结果不能证明某个 short-associated feature 是“简洁推理能力”。[原始结果与解释](results/phase2_sae_sampling_ablation_v1/exploratory/analysis/analysis_report.md)

![SAE 抽样方式与长度暴露](figures/phase2_sae_sampling_ablation_v1/01_training_sampling_exposure.png)

### A2：Answer 混淆进一步如何处理

原观察中，答案结尾处近似相同的局部激活，除以不同的整段长度后，就可能形成 short/long 差异。这是需要验证的混淆机制，并非全部效应的既定解释。[原 token 与位置诊断](docs/tutorial/sae_training_and_feature_to_token_zh.md)

Phase 5 排除了 `Answer:` 答案段、最后 32 个 token 和格式 token；在前 128 个原始位置内，每条保留恰好 64 个合格早期 token，以固定分母评分，同时保留协变量校正前后结果。

- 2,047 条轨迹、131,008 个状态完成编码；11 个候选通过确认，按 dev 排名预定主特征 **24086**。
- 旧四个短特征的清理后效应量为 **0.079、0.091、0.223、−0.009**，比原整段指标约 2.5–2.8 小很多；由于同时改变了支持位置和清理规则，尚不能分解每个因素的独立贡献。
- 主特征的激活读回操作通过，但第二版随机对照 3739 的确认集覆盖率为 **4.15% < 5%**，未达到预设门槛。因此生成与学生蒸馏未触发。

这轮是“按协议完成并停在失败门槛”，不是“已完成生成且证明长度控制无效”。特征 24086 绑定 Phase 2 的 full-trace-balanced 字典；它与 B2 三种子新字典的 feature ID 不能互换。[完整结果](results/phase5_sae_clean_feature_causal_v1/exploratory/pilot_summary_zh.md)

![去除格式与位置混淆后的特征检验](figures/phase5_sae_clean_feature_causal_v1/exploratory/clean_feature_gate.png)

### B1–B4：使用有效训练数据的对比，做到哪一步

B1 使用逐字节复制的原始三份 881 题 SFT 数据，保留原 completion-only TRL 0.9.6 配方：Qwen2.5-1.5B-Instruct 学生、1 epoch、batch 4、lr 2e-5、rank-4 LoRA、最大输入长度 2048。三种子完整评估结果如下。

| 条件 | 平均 GSM8K 正确率 | 相对该次 C31 base |
| --- | ---: | ---: |
| C31 base | 68.01% | — |
| Short | **70.03%** | +2.02 个百分点 |
| Medium | 68.90% | +0.89 个百分点 |
| Long | 67.38% | −0.63 个百分点 |

登记的复现门槛通过；short − long 的配对 95% bootstrap 区间为 **[+0.53, +4.81] 个百分点**，short − base 为 **[−0.32, +4.31] 个百分点**。所以这里确认了预设复现门槛和 short 高于 long 的探索性证据，尚不能称相对 base 的提升显著。C49 另有 **67.14%** 的 base 回放，逐题复现旧预测；两份 base 均保留，未替换主要基线。[指标、区间和逐 run 记录](results/phase1_legacy_trace_sae_distillation_v1/replication/analysis/replication_report.json)

![历史有效数据与配方的 SFT 复现](results/phase1_legacy_trace_sae_distillation_v1/replication/analysis/replication_accuracy.png)

B2 随后使用同一 881 题、14,096 条原始 7B 轨迹池，完成 full-trace-balanced / prefix64-trace-balanced × seeds 17/42/73 六个 SAE 和跨种子评分。原组合筛选要求全序列与 first64 方向均复制，最终稳定特征数为 **0**。

其中 full-sequence long 候选跨种子对应 **10498 / 18161 / 20175**，但 seed 73 的 first64 方向未通过。已批准的 B3 将 first64 保留为诊断，不再作为准入条件；其实际重筛选仍未运行，不能凭记忆中的 feature ID 直接进入干预。[六模型汇总](results/phase1_legacy_trace_sae_distillation_v1/sae/analysis/sae_summary.json)、[筛选诊断](results/phase1_legacy_trace_sae_distillation_v1/sae/feature_screen_diagnostics/README.md)

**B4 尚无结果。** 需要通过词汇反证与配对缓存校准，再按注册设计对全部 881 题、每方法每题 16 个候选生成，选择 shortest-correct，取所有方法的正确共同题集；历史自然 rank 控制也必须在同一交集重新训练。等样本与等监督 token 结果分别报告，不能把 617 题 SAE 学习划分当作学生训练题数上限。[后续协议](docs/experiments/phase1/phase1_legacy_trace_sae_distillation.md)

### C1：为什么早期完成的 SFT 不能代替 B4

早期 pilot 在 617 道生成题中取了 613 道共同有效题，采用当时的训练实现，而非 B1 恢复的原 legacy TRL 配方。它包含无干预、short 特征增强、long 特征抑制、随机特征增强 × 两种预算 × 三种子，共 **24 次 SFT**；25 组评估包括 base。

等样本时，无干预为 **65.06%**，short 增强 **64.78%**，long 抑制 **65.30%**；均低于该 pilot 的 **67.14% base**。等监督 token 下也未发现稳健改善，目标干预相对无干预的登记配对区间均跨零。教师平均长度变化也很小。该历史 pilot 不能证明有效配方上的学生收益；新增 E2 已使用 E1 验证的有效配方完成干预学生对照，仍未确立涨点，详见上文。[pilot 分析与判定](results/phase3_sae_intervention_distillation_pilot_v1/exploratory/analysis/analysis_report.json)

## 2026-09-09 恢复与资源快照

- **BeeGFS**：已挂载为读写；4 KiB 新建文件经过 write、fsync、read-back 一致性检查，随后删除。此次关键产物读取和哈希核验通过，不代表对全部文件系统完成完整性检查。
- **证据校验**：331 次文件哈希计算，覆盖 303 个不同路径；其中 303 次比对登记哈希，覆盖 275 个不同路径，全部一致。40 项题目数量、唯一 ID / 共同支持检查全部通过。涵盖三条件采样 SAE、六个后续 SAE、9 个复现 adapter、24 个旧 pilot adapter，以及相关完整预测。[本次机器可读审计](results/project_status_audit_20260909/audit.json)
- **恢复的输入**：六 SAE 检查点、评分、mixed corpus、原 SFT 数据已可读；BeeGFS 固定 revision `a09a35458c702b33eeacc393d103063234e8bc28` 的 7B 配置、tokenizer、四个权重分片存在且可做初始读取。本次未完整加载教师或重跑 GPU 生成。
- **剩余路径问题**：B3 原配置仍指向缺失的 C31 本地 7B snapshot。筛选输入已齐，但按未修改配置做生成仍不满足输入检查。后续应在独立运行配置中绑定恢复的固定 revision，或先将其校验后暂存到节点本地。[恢复后路径审计](results/project_status_audit_20260909/recovery_prerequisites.json)
- **作业状态**：查询时只见用户 `parser` allocation：C31 `279747` 正在运行，C32 `280415` 排队；未见 SAE / SFT 实验作业。C31 GPU 快照只有用户的 `gg` keepalive，四卡剩余显存约 46.1–47.4 GiB。本次未修改任何 allocation 或 GPU 进程。[队列](results/project_status_audit_20260909/queue.txt)、[近期作业](results/project_status_audit_20260909/recent_jobs.txt)、[GPU 快照](results/project_status_audit_20260909/nvidia_smi.txt)
- **存储约束**：保留用户最新选择：新数据实际放在 `/home/youyang7/projects/SAE_long_short` 的独立结果目录；最终 adapter 优先项目目录，配额不足再用 `/mnt/local` 并登记实际位置。已存在的 BeeGFS 历史产物保持原位。本次 home 用量 29,899 MiB / soft quota 39,936 MiB，约余 9.8 GiB；环境、模型暂存、临时状态留节点本地，结束后清理不再需要的临时文件。[配额快照](results/project_status_audit_20260909/quota.txt)

## 下一步与验收条件

以下是待办与历史验收记录，不代表已提交新作业。E1/E2 已完成归档；后续优先解决 E2 已观察到的训练执行差异，再评估独立复现。历史 B3/B4 保留独立队列。

| 优先级 | 工作 | 完成条件 |
| --- | --- | --- |
| 当前阻塞，9 月 11 日已重检 | 恢复 BeeGFS 历史输入与关键证据 | C31/C32/C49 未挂载；需恢复原目录或哈希可验证备份，然后重新验证可读性和登记哈希 |
| P1，待设计 | E2 训练执行差异定位与确定性复查 | 单独冻结复查配置，固定数据、初始化和执行环境；保存逐步日志、权重哈希与完整预测，区分训练和评估变异，不覆盖原 8 次运行 |
| P1 | 执行 B3 全序列 v2 重筛选 | 校验原候选与评分来源；独立输出筛选报告、输入/配置哈希和完成标记；保留原零特征结论 |
| P1，后续启动前 | 重新验证运行环境与固定 7B 路径 | legacy TRL 依赖、教师/SAE 维度、权重与实际 GPU 准入检查通过；旧预检不作为当前通过证据 |
| P2 | 对 B3 入选特征完成词汇反证和配对因果校准 | 目标读回、独立缓存、首个续写 token 干预、正确率/长度/扰动及随机对照审计通过；没有入选特征则记录阴性结果 |
| P3 | 完成 B4 的数据生成和自然数据学生对照 | 全题集完整分片、无重复/缺失；共同支持集一致；最终 adapter、逐题预测、预算统计、配对分析及完成标记齐全 |
| 独立分支 | D1 全量重生成与三个 rank adapter | 每题 16 候选，至少三个唯一正确答案后选 shortest / lower-median / longest；不足时剔除该题并在共同子集训练。冻结单种子配置并记录新数据哈希，不称逐字节复现；建议 seed 17 和完整锁定评估，当前仅为默认建议 |
| E2 已完成；独立确认待设计 | Phase 6 长度干预后续 | E2 已完成从首个输出 logit 干预、实际范数控制、结构匹配随机对照和学生对照；下一轮需独立数据、跨训练种子与更多随机方向复现，并预先冻结正确率代价和学生收益门槛。不得把 Phase 5 的失败门槛改成通过 |

Phase 5 / Phase 6 是独立的清理特征因果验证路线；B3 / B4 是历史有效配方上的全序列特征路线。前者的“固定早期支持检验”和后者的“first64 仅作诊断”不合并成同一筛选规则。

## 更新约定

每次推进只在本文件更新任务状态、日期、作业 ID、实际输出路径、阻塞原因与证据链接。`已完成` 必须指向相应阶段的完成标记和可验证产物；`按门槛结束` 同时写清未触发的后续阶段。既有 GSM8K 旧池结果继续标为探索性，尚无“SAE 干预稳定缩短输出并提升学生”的闭环结论。

2026-09-07 的 [C31 v2 记录](docs/experiments/phase1/phase1_full_sequence_screen_c31_v2.md) 和 [Phase 6 预检记录](docs/experiments/phase6/phase6_sae_length_runtime_validation_zh.md) 中的 BeeGFS 停机描述是当时快照；2026-09-09 审计记录它曾恢复。最近资源与输入检查为本文开头的 2026-09-11 续跑预检，C31/C32/C49 未挂载 BeeGFS。历史 BeeGFS 链接需待重新挂载后访问；E1/E2 的数据、最终 adapter 与结果已存入 home。更早的 Phase 0 暂停及 teaching-utility 分支不因本次建账而恢复执行。

### 自动恢复与Luna网络隔离修订

GPU已真实启动：839093/839094在H200生成并写出候选，839095在L40S加载；839096当时Priority排队。旧controller839127在计算节点访问外部模型API超时，因此新 `controller_v2` 将恢复循环与Luna完全分离。GPU worker继续使用execution_v5，不重启、不改科学设置。控制器只授权重试该worker协议任务；原login04的monitor_v3继续每小时luna/medium审阅。新控制器在恢复锁内替换旧CPU控制器，保存superseded记录；未取消任何运行中的GPU。

### 已发生的自动恢复证据

新controller测试839197通过48项，839198在c207n01运行，heartbeat已更新为该job。实际L40S失败839095/839096已由控制器自动归档不完整输出，分别重提839212/839213并重接后继与总合并依赖；记录在 `recovery/job_839095`、`recovery/job_839096`。这是实际终态失败恢复，不仅是模拟测试。H200分片保持运行。为利用空闲H200并避开已观察到的L40S CUDA非法访问，主代理在恢复锁内迁移尚未运行的L40S分片，逐项hold/保存/重提/重接/取消旧pending，运行中的GPU保持不动，迁移记录在 `recovery/h200_pending_migration_v1`。

### 硬件故障恢复策略最终修订

15个尚未运行的L40S作业完成H200迁移，运行GPU未打断，逐项映射与依赖更新保留。控制器进一步冻结为 `controller_v3`（测试839277、服务839278）：CUDA非法访问/设备断言改GPU家族，L40S转H200、H200转H100；OOM转H200；TIMEOUT转登记H100四小时时限。确定性ValueError/文件缺失/import/syntax/hash错误停止并写NEEDS_ATTENTION。原worker仍执行execution_v5，不改变科学设置。控制器无外部模型网络调用，旧monitor_v3继续每小时Luna审阅。新增路由和确定性错误测试在本地通过，控制器完整回归预期50项。

最终验收：controller_v3的50项测试与源码/协议绑定已由主代理验证，证据为 `controller_v3/verification/VERIFIED.json`。839278控制器heartbeat已更新；当前4个GPU worker运行（839093/839094/839260为H200，839213为L40S），无新未恢复FAILED、DependencyNeverSatisfied或NEEDS_ATTENTION。Luna旧monitor_v3心跳持续更新。记录为恢复并实际运行，尚非完整steering、SFT或最终收益结果。

ABI恢复已实际验证：842428的51项测试通过，842429在H200完成4/4 TokenSkip smoke；主代理验证新源码/协议/TokenSkip及合并smoke的哈希，保存execution_v6/verification/NATIVE_FIX_VERIFIED.json。最新验收842431–842434均在gpu38 H200 RUNNING；恢复heartbeat于13:50:16 UTC更新并记录四路运行。控制器启动时与主代理恢复锁冲突的一次BlockingIOError为暂态，此后heartbeat继续更新，不是作业终态失败。尚未开始学生SFT。
