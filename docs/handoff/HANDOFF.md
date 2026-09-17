# 项目交接：Answer-free SAE 与 Phase 13 实验

更新时间：2026-09-16 22:53 EDT。项目根目录为 `/rsstu/users/m/myoon2/satc_568382/yang_ouyang/projects/SAE_long_short`。本文件是接手入口；实验状态仍以[项目执行台账](../../PROJECT_STATUS.md)、冻结协议和完成标记为准。执行前阅读仓库根目录的 [AGENTS.md](../../AGENTS.md)。

## 研究目标与当前结论

论文目标是比较 SAE 干预生成的教师解答与其他方法对 Qwen 3B 学生的效果。旧 B7 八个短特征的激活高度集中在 `Answer:` 标题，不能直接解释为正文推理特征。用户因此要求先删除 `Answer:` 及以后的 token，重训 SAE、重新筛选特征，再做干预；旧 Phase 5 的 11 个清理候选仅作为动机，特征编号不可跨字典沿用。

这一轮 `answer_free_sae_v2` 已按注册门槛**结束于同状态读回**。新 SAE 和清理后特征筛选完成，但预定主特征 F20319 的 test 激活覆盖率为 2.40%，低于 5% 门槛；随机对照之一也未满足两倍频率匹配。**没有启动教师生成或学生训练**，不能声称新特征已改变生成长度或提高学生准确率。[完整协议与执行解释](../experiments/phase13/phase13_sae_feature_separation_project_zh.md)

## 本轮可核验产物

| 阶段 | 状态和关键证据 |
| --- | --- |
| v1 失败尝试 | 准备作业 853536 因同题标签判断错误而失败，依赖作业 853537/853538 自动取消；历史冻结快照和故障记录保留，无有效 SAE。 |
| v2 输入与训练 | 853543/853544/853545 均 `COMPLETED/0:0`。14,096 条父轨迹中保留 6,769 条成对 short/long 轨迹；109 个原激活 chunk 核验后抽取 train/dev/test 250,000/50,000/50,000 token。新 layer17、TopK64、28,672 特征 SAE 完成 1,500 步；最后 dev MSE 0.2264、解释方差 0.7733。[样本 manifest](../../results/phase13_sae_feature_separation_v1/exploratory/answer_free_sae_v2/token_samples/sample_manifest.json) · [训练完成标记](../../results/phase13_sae_feature_separation_v1/exploratory/answer_free_sae_v2/training/COMPLETE.json) |
| 正文特征筛选 | 853652、853657–853661 均 `COMPLETED/0:0`。2,030 条轨迹、129,920 个固定正文 token，零重复/缺失；16 个注册候选中 4 个通过，按 dev 排名预定 F20319。[筛选结果](../../results/phase13_sae_feature_separation_v1/exploratory/answer_free_sae_v2/screen/feature_gate/selected_features.json) · [筛选审计](../../results/phase13_sae_feature_separation_v1/exploratory/answer_free_sae_v2/screen/feature_gate/feature_gate_manifest.json) |
| 同状态双向读回 | 853699–853703 均 `COMPLETED/0:0`；24,240 条逐状态记录零重复/缺失。F20319 的正负目标激活变化为 −63.13%/+56.81%，非目标/目标 L2 为 1.079/1.206，但 test 覆盖仅 2.40%。随机特征 7540/8589 各自读回通过；8589 的 dev 覆盖是主特征的 2.44 倍，超过两倍匹配门槛。[读回审计](../../results/phase13_sae_feature_separation_v1/exploratory/answer_free_sae_v2/screen/engagement_analysis_v1/engagement_analysis_manifest.json) · [停止记录](../../results/phase13_sae_feature_separation_v1/exploratory/answer_free_sae_v2/screen/final_decision.json) |

训练完成标记的 6 项文件哈希已复核；同状态读回的完整完成标记已核验。五份读回作业实际 Slurm batch 与冻结的指定 scratch 启动脚本逐字节一致，见[批处理核验](../../results/phase13_sae_feature_separation_v1/exploratory/answer_free_sae_v2/screen/engagement_launch_v1/verification/batch_audit.json)。截至本文件更新时间，`squeue -u "$(id -un)"` 无当前用户排队或运行作业；再次操作前需重新查询。

## 重要边界与接手动作

1. 本轮 dev/test 题目已在既有 NCSU 实验中被观察。筛选的“test 确认”是旧池内探索性复现，不是独立新题确认；不能据此作正式跨题泛化主张。
2. 冻结匹配实现只预筛了随机对照的 dev 覆盖，未在冻结前预筛主特征覆盖，也未在随机抽取前严格约束两倍频率/幅度。这些缺口已写入[执行解释](../experiments/phase13/phase13_sae_feature_separation_project_zh.md)；工作树中的匹配函数已加入可选的提前拒绝与严格过滤，但**本轮冻结快照及失败结果保持不变**。
3. 若继续该路线，应立独立新协议：在 dev 的实际读回位置预筛目标覆盖率≥5%及两项对照的频率/条件幅度两倍范围，按 dev 冻结目标与对照，然后用未观察的新题验证。同一批已看过的 test 结果不得重新包装为独立确认；不能为让 F20319 通过而放宽本轮 5% 门槛或事后更换对照。
4. 只有新协议的特异读回、对照及准确率边界都通过后，才冻结教师生成和 Qwen 3B 学生实验。当前 [最终决定](../../results/phase13_sae_feature_separation_v1/exploratory/answer_free_sae_v2/screen/final_decision.json) 明确不触发这两阶段。

主要实现入口：[正文支持与采样](../../src/length_budget_distill/sae_body_training.py)、[特征筛选](../../src/length_budget_distill/sae_body_screen.py)、[目标读回](../../src/length_budget_distill/sae_target_engagement.py)、[读回审计](../../src/length_budget_distill/sae_engagement_analysis.py)。本轮冻结启动源码位于对应结果目录的 `code/` 下；新的实验应另立结果目录和冻结快照，不修改历史产物。近期相关脚本为 `scripts/13_68`–`13_73` 前缀。

## 其他 Phase 13 分支

- [Qwen 3B 配方对齐](../experiments/phase13/phase13_gsm8k_qwen3b_recipe_alignment_v1.md)的 B1、SAE 0.1、SAE 0.3 各三个 seed 已完整完成；平均准确率分别 83.08%、84.48%、82.79%。SAE 0.1 相对 B1 的 +1.39 个百分点经四比较 Holm 调整后 `p=0.204`，不能宣称优于 B1。该分支是旧配方对齐 SFT，不是 KD，也不验证新 Answer-free SAE。
- [256 题 token-KD 小试 v2](../experiments/phase13/phase13_token_kd_small_v1_zh.md)已完整完成。相对匹配 SFT，两个 seed 的准确率差为 +2.00 和 0 个百分点；置信区间跨零，未过预设放大门槛，当前没有提交大规模 KD 作业。
- GSM8K SAE 学生 pilot 为单训练 seed 的探索性闭环；旧 B7 尚未证明优于 B1。其状态与其他 MATH/基线分支应分别查阅[执行台账](../../PROJECT_STATUS.md)，不要用本轮 Answer-free 结果替代。

## NCSU 运行约束

使用当前用户 `youyang7` 的 Slurm 账户和已授权 GPU/CPU 分区，提交前刷新 QOS、节点健康和资源占用。所有作业的 `TMPDIR`、`TMP`、`TEMP`、数据集缓存与中间 checkpoint 必须位于管理员指定的 `/share/jekml/youyang7/tmp` 下的作业专属目录；禁止回退到 `/var/tmp` 或系统临时目录。真实提交 batch、显式 Trainer/SAE 输出位置、scratch 容量和配额都要核对。保留历史快照、失败尝试、模型缓存和其他用户作业。详细要求见 [AGENTS.md](../../AGENTS.md) 与 [NCSU 环境说明](../environment/ncsu_sft_environment.md)。
