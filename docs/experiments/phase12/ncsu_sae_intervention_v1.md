# NCSU SAE 后续干预协议

状态：教师干预、全量候选生成、学生训练评估与最终交付复核均已完成（2026-09-13）。用户于 2026-09-12 要求执行后续干预。本轮独立于已完成的 `ncsu_phase12_reproduction_v1`，保持探索性 GSM8K 范围。

## 冻结设计

- 使用父实验第 17 层、TopK 64、seed 17 的 SAE。固定已确认的 8 个 short-associated 和 8 个 long-associated 特征，按父实验 dev discovery rank 排序。父实验的确认集已被观察，本轮不称为全新、未观察的特征确认。
- 干预位置为第 17 层 post-block residual 的最后有效位置，从预测首个生成 token 的 prompt 位置开始。短增强使用短特征 decoder 单位方向；联合干预使用短方向减去长方向后再单位化。联合方向是恒定方向干预，不是按每个 token 的长特征激活门控抑制。
- 相对实际 hidden norm 的强度固定为 0.05 / 0.15 / 0.30。短增强随机对照使用 8 个随机 decoder 列；联合随机对照使用两个互不重叠的 8 列集合，执行相同归一化与相减。随机集合排除目标特征，seed 固定。此对照匹配特征数量和扰动范数，不保证语义或激活频率匹配。
- Dev 为父实验 SAE dev 中固定哈希排序的 64 题；13 条件为无干预、两种目标模式各三种强度及其六个随机对照。每题每条件一条轨迹，四个不重叠题目分片。使用相同 prompt、逐题随机 uniforms，条件间独立 KV cache；temperature 0.7、top-p 0.95、512-token cap。
- 入选门槛：平均输出至少缩短 5%；正确率点估计下降不超过 5 个百分点；配对长度 95% bootstrap 区间上界小于零；比对应随机对照的平均输出更短。正确率点估计门槛不等于非劣检验。Dev 区间仅用于探索性筛选，不宣称多重比较后的发现。
- 多条件通过时选平均输出最短者，再按正确率较高、条件名排序打破平局。固定选择后，在 SAE test 中另取固定哈希排序的 64 题，运行无干预、目标、对应随机三条件，以相同门槛判定是否触发学生阶段。该 test 仍是已观察过的探索性题集。
- 若 dev 或 test 无条件通过，按门槛结束并发布完整阴性结果，不自动扩大强度或改换特征。

## 条件性学生阶段

教师门槛通过后，使用全部 881 道原训练题，每题每条件四候选，生成无干预、选定目标、匹配随机三组。选择去重、正确且未截断的 shortest-correct，并与父实验自然 short 数据取共同题集；四组均在同一支持集训练，不直接将原父实验 adapter 作为共同支持对照。

固定原学生、LoRA 超参数和 seeds 17/42/73，分别运行等样本与等目标 token 两种预算；完整轨迹重复用于 token 预算匹配，并报告实际 token、训练步数和题目权重。评估保持 GSM8K `test[50:1319]`、greedy、512-token cap，每个 adapter 1,269 条预测。学生阶段启动前完成其实现与输入审计，保持以上设计；不依据最终评估结果调整设置。

## 实现与执行

复用 `sae_norm_intervention` 的实测范数控制及逐题配对生成，`utility_analysis` 的配对 bootstrap，以及父实验 NCSU 的训练、评估和资源准入逻辑。新适配器位于 `src/length_budget_distill/ncsu_intervention.py`，入口为 `scripts/3_40_ncsu_sae_intervention.py`，配置为 `configs/phase12_ncsu_intervention_v1.json`。

原 Phase 6 编排硬编码了 C31 路径、不同字典集合、至少每侧 16 个特征和旧数据导入，因此本轮新增编排适配器，复用其底层干预实现；图表延续父实验配色、白底和简洁坐标轴。

新作业使用已登记的 NCSU H100 分区和账户。每次 GPU 作业启动打印 `nvidia-smi` 与物理进程清单，并仅对 Slurm 分配的可见 GPU 连续检查剩余显存。使用 `sbatch` 排队，保留原始结果和所有失败尝试。代码快照、输入与 SAE 哈希在执行前冻结，逐题输出、审计、报告和完成标记分别保存。

## 执行更新：2026-09-12 23:09 EDT

教师 dev/test 作业 `817270–817280` 全部完成。入选 `short_rho0.3` 在 test 平均长度由 246.44 降至 209.23 tokens，两者均答对 64/64；随机为 293.62 tokens、62/64。后续学生作业 `817294–817324` 已登记成功依赖。H100 满额后调度到 H200，资源变更单独记录，原科学配置保持冻结。

学生实现位于 `src/length_budget_distill/ncsu_intervention_student.py`，入口为 `scripts/4_40_ncsu_intervention_student.py`。学生阶段自身的输入和完整代码快照已经冻结，复用原 NCSU 训练与评估逻辑，以及原完整轨迹 token 预算匹配 helper。

[教师 dev 报告](../../../results/ncsu_sae_intervention_v1/exploratory/analysis/dev/report_zh.md)、[教师 test 报告](../../../results/ncsu_sae_intervention_v1/exploratory/analysis/test/report_zh.md)、[学生冻结标记](../../../results/ncsu_sae_intervention_v1/exploratory/student_followup/protocol/FROZEN.json)、[学生作业清单](../../../results/ncsu_sae_intervention_v1/exploratory/student_followup/submission.json)。

## 完成更新：2026-09-13

最终分析作业 `817324` 于 00:18:08 EDT 成功结束。全量 10,572 条候选、878 题共同支持、24 个 adapter、25 组完整评估全部完成。所有实际 GPU 作业均在 H200 `gpu38` 运行；学生作业曾在排队期间转调 L40S，但发现健康卡均已分配后，在任何学生作业启动前恢复 H200。两次调度变更均保留记录，科学配置和冻结执行源码没有改变。

全量教师输出缩短 13.27%。学生等样本为目标 70.84% / 无干预 69.92%，等 token 为目标 70.34% / 无干预 70.42%；相对无干预的准确率差区间均跨零。最终交付复核通过，原执行标记与报告保持不变。

[最终状态报告](ncsu_sae_intervention_status_20260912.md)、[总完成标记](../../../results/ncsu_sae_intervention_v1/exploratory/INTERVENTION_COMPLETE.json)、[交付复核](../../../results/ncsu_sae_intervention_v1/exploratory/setup/final_delivery_audit_20260913.json)。

补充交付审计复用现有 `select_common`、答案验证、评估汇总、文件哈希和配对 bootstrap helper，通过 `scripts/4_41_audit_ncsu_intervention.py` 调用源码中的 `audit_delivery`。它仅在完成后读回产物，不改变冻结科学配置、原生成/训练/分析源码或结果；完整教师池的配对差为事后描述性汇总。
