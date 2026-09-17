# 固定题集的多答案蒸馏协议

用户于 2026-09-13 确认执行：保留原878题，补齐每题四条不同正确答案，比较每题1/2/4条并加入最短答案重复2/4次的对照；仅等样本预算，seeds 17/42/73，共30个 adapter，另评估base。

## 数据与干预

父实验为 `ncsu_sae_intervention_v1` 的 student followup；使用其878题共同支持，不增加、删除或重排问题身份。只比较 `selected_target` 与 `no_steering`。目标固定原 layer17 / TopK64 SAE、8个已确认短特征、rho0.3、start0；教师、学生、prompt和生成参数继承已完成父实验。

先核验原四个生成分片，复用其中878题两条件各四个候选。当前825题两组均有四条不同正确答案，其余53题补生成。每轮每个待补题在两条件各生成四条，候选索引从4连续增加到最多31；随机流沿用父实验的 `generation_seed + 10000 + candidate_index` 后按题号SHA256派生，条件间共享uniforms且缓存独立。每轮结束时两组均达到四条不同正确答案则停止该题。最多累计每题每条件32个候选；仍不足则登记失败，禁止后续训练，不丢弃题目或以重复答案充数。

去重键为 `response.strip()`，保留最早候选；错误及触及生成上限的答案不用于训练。排序按生成token数、候选索引。所有档位从补齐后的同一个候选池构建，因此重新训练每题1条基线；旧adapter不替代新基线。不同文本不保证不同数学解法。

## 训练矩阵

| 方案 | 每题记录 | 每组记录 | 步数 |
| --- | ---: | ---: | ---: |
| k1_unique | 最短正确答案1条 | 878 | 220 |
| k2_unique | 最短正确答案2条 | 1,756 | 439 |
| k4_unique | 最短正确答案4条 | 3,512 | 878 |
| k2_repeat | 最短答案重复2次 | 1,756 | 439 |
| k4_repeat | 最短答案重复4次 | 3,512 | 878 |

每组各训练SAE/无干预与seeds17/42/73。继承原LoRA和completion-only TRL配方：1 epoch、batch4、gradient accumulation1、学习率2e-5、warmup0.03。固定每题曝光和同档位优化步数；不同答案与重复答案的监督token不相等，本轮没有等token分析。

全部31组评估锁定GSM8K `test[50:1319]`，每组1,269题，greedy、batch32、512-token cap。不按评估结果重新选样或调参。

## 登记统计与验收

使用10,000次训练seed×配对题目bootstrap，seed2026091304。九项比较共同作Holm校正：三个k下SAE−无干预；k2/k4的方法差距相对k1的变化；两种方法各在k2/k4的不同答案−重复答案。只有方法优势及其相对k1的扩大都满足区间下界大于零、校正p<0.05，才判定增加答案数扩大了SAE优势。

冻结配置、源代码、模型和输入哈希；补生成按分片登记、检查完整候选编号及停止规则。构建阶段验证878题都齐全、文本唯一性与嵌套关系；最终独立重建训练数据，检查步数、有效学习率、最终LoRA哈希与全部39,339条逐题评估，再发布完成标记。图表延续父实验SAE绿色、无干预灰色的风格，以答案数量曲线和配对区间为主。

## 运行方式与产物

入口 `scripts/4_42_ncsu_multi_answer.py` 提供 prepare / supplement / build / student / analyze / audit / submit；逻辑位于 `src/length_budget_distill/ncsu_multi_answer.py`，配置为 `configs/phase12_ncsu_multi_answer_v1.json`。复用既有干预、SFT、评估、统计函数，新增编排是因为原入口固定单答案和四个条件，不能直接表达本轮嵌套数据与重复控制。

使用NCSU H200的sbatch成功依赖链，一卡一进程，启动前打印nvidia-smi并连续检查显存。管理员指定的 `/share/jekml/youyang7/tmp` 下作业独立目录 用于临时数据、HF datasets cache及Trainer中间文件；最终adapter哈希验证后发布到独立项目目录。资源排队不视为完成。原实验的配置、预测、权重、报告及完成标记保持不变。

结果目录：`results/ncsu_multi_answer_v1/exploratory/`；adapter目录：`checkpoints/ncsu_multi_answer_v1/students/`。本轮保持探索性GSM8K、一个SAE seed与已观察题集的边界，`formal_claim_allowed=false`。

## 执行完成记录

2026-09-13 06:33:26 EDT通过最终审计。原878题均补齐，53题新增496候选，共7,520条；30个adapter与31组评估完整，39,339条预测已核验。九项比较均未通过Holm校正，未确立增加答案数扩大SAE优势或不同答案优于重复最短答案。详见[完成状态报告](ncsu_multi_answer_status_20260913.md)。

原CPU统计节点启动异常，仅取消819781与待运行审计819782，保留调度记录；相同冻结分析与审计在c027n01通过820079／820080完成。科学配置、训练和评估产物未修改。最终作业状态为38项成功、2项取消；[恢复记录](../../../results/ncsu_multi_answer_v1/exploratory/setup/cpu_recovery_20260913.json)与[作业状态](../../../results/ncsu_multi_answer_v1/exploratory/setup/final_job_accounting_20260913.txt)已登记。

2026-09-15 storage correction: the current launcher uses administrator-designated scratch. Historical frozen launchers and previously submitted Slurm scripts retain their original paths and require separate migration verification.
