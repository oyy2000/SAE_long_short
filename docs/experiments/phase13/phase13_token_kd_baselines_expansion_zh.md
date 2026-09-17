# Phase 13：KD 与八组 baseline 的配对扩展

## 授权、问题与边界

2026-09-16 用户要求“KD把实验做完整，大规模数据和baseline对比”。本轮据此建立独立扩展协议。此前 256 题 KD 小试的平均增益为 +1 pp，但置信区间跨零、一个种子无增益，仍保留“未通过效果放大门槛”的结论；新授权不改变旧门槛或结论。

先完成已有完整数据的 GSM8K 1,024 题矩阵，同时修复 MATH 7,500 题来源池的下游数据链。1,024 题不是 MATH 大规模实验已经完成。MATH 在去除保留题后有 6,723 道 student-pool 题，实际训练数量须以所有方法共同正确支持和去污染审计为准。

## GSM8K：48 次匹配训练

当前配置为 `configs/phase13_token_kd_baselines_gsm1024_v2.json`，结果位于 `results/phase13_baseline_expansion_v1/exploratory/token_kd_baselines_gsm1024_v2`。v1 通过 21 项检查，但重复创建已准备的 B1 文件而在 CPU 准备阶段失败，没有 GPU 训练；v2 核对并复用 B1，补充回归测试，科学设置不变。

八种监督来源：B0 随机正确、B1 无干预最短正确、B2 简洁提示、B3 dense、B4 ASC-CES、B5 DAP、B6 TokenSkip、B7 历史 SAE。沿用原开发集选点：B3/B4/B7 均为 0.1，B6 推理比例为 1.0；本轮不重新搜索强度。每组分别训练 SFT 与 token-KD，种子固定 17/42/73，共 8 × 2 × 3 = 48 个 adapter，另重评无训练 Base。

B7 使用旧字典生成的原始 trace，具有已知 Answer 格式关联，只能作为历史 SAE 对照。Answer-free SAE 是独立分支：新主特征 F20319 的 test 同状态覆盖率为 2.40%，未达到 5% 门槛，未触发生成或学生训练。不得把旧 B7 换名为清理后 SAE，或因本轮 KD 扩展而绕过其因果门槛。

教师为固定 revision 的 Qwen2.5-7B-Instruct，学生为固定 revision 的 Qwen2.5-3B-Instruct。每组 1,024 道相同题目，每个种子下 SFT/KD 的文本、token、样本顺序、LoRA、优化步数完全匹配：3 epochs、384 步、batch1/accum8、LR5e-5、cosine、warmup0.1、LoRA r8/alpha16/dropout0，七类投影。各 baseline 目标长度不同，因此跨方法是等题目/等曝光控制，不是等监督 token 控制；目标 token 和运行成本单独报告。

SFT 使用 completion-only CE；KD 使用 `0.5 CE + 0.5 T² KL(teacher || student)`，T=2。沿用修正后的整组梯度累积 token 归一化，包含 native EOS，排除 prompt/padding。教师冻结并在各方法的同一 completion prefix 上提供在线概率，不在概率教师中额外安装 SAE/dense 干预。B6 保留训练文本中的比例条件和推理比例。teacher/student 有效词表严格对齐，KL 在共同有效 token-ID 上重新归一化，记录填充输出维度的舍弃概率质量。

## 评测与证据

每个 adapter 评测 300 道 development 和 GSM8K test[50:1319] 的 1,269 道题，greedy、cap1024、repetition penalty1.0、batch8。两个 cohort 此前均被观察过，本轮为探索性扩展，不是新的独立确认，也不能与 cap512/penalty1.1 的配方对齐结果混合。

保留每题原始输出、token、评分、结束原因、硬件与耗时。主比较为各 baseline 内 KD 减 SFT，共八项；次比较为各 KD baseline 减 B1 KD，共七项。逐 cohort、逐比较族使用 Holm 校正。区间按题配对 bootstrap，三个固定训练种子一起保留，明确其条件于已训练的三个学生；同时公布每个种子的差异。双侧 p 值使用按题、对种子取均值后的配对差异 Wald 检验。

完整证据要求：48 个完成 adapter、完整预测、无缺失/重复题目、成对训练曝光和监督 token 核验、输入/代码/模型/输出哈希、聚合指标、对比图、完成标记。排队、smoke 通过、局部分片均不等于完成。两 cohort 加 Base 共预期 76,881 条预测。

## 调度与存储

提交链：CPU 测试 → 数据准备 → 最长输入 B0 的 H200 KD smoke → 四路依赖矩阵 → Base → 聚合分析。smoke 需通过显存加 8 GiB 余量和最长输入推算的训练时长门槛。H200 路由申请 2 小时，最多四路；其他硬件需先得到同设置的显存及吞吐测量，不改变科学设置。任何失败保留原日志和产物，并阻止其 afterok 下游，不盲目重试逻辑或数据错误。

所有临时文件、数据缓存、训练输出位于 `/share/jekml/youyang7/tmp/<job-specific>`；最终 adapter 经哈希核验后发布。实际 Slurm-spooled batch 与冻结启动器逐字节核对。保留 legacy overlay、libstdc++ 预加载及 frozen execution_v7 的既有运行依赖。

首次冻结在权限切换时中断，未提交作业，保存在 `token_kd_baselines_gsm1024_v1_interrupted_freeze`。v1 bootstrap 为测试 856422、准备 856423、smoke 856424、矩阵门槛 856425；准备失败后，后两项依赖自动取消。v2 为 856443 测试 → 856444 准备 → 856445 smoke → 856446 矩阵门槛；22 项测试和准备已完成，smoke 上次回读等待共享 H200 的 QOSGrpGRES。后续提交以自身 `protocol/bootstrap_jobs.json` 和 `protocol/matrix_jobs.json` 为准。

按 2026-09-16 用户补充的调度规则，排队时应换到当前可执行、且满足实测显存与时限的 H100/H200/L40S 路由；只有均不可用或科学设置固定卡型时才排队。已独立冻结科学设置相同的 `configs/phase13_token_kd_baselines_gsm1024_l40s_v1.json`，最多两路 L40S。856509 测试、856510 准备、856511 KD 最长输入 smoke、856512 矩阵门槛均完成；smoke 峰值预留 26,376 MiB，加 8,192 MiB 余量后符合卡内存门槛，训练部分 4 步约 30 秒。门槛提交 48 个训练作业 856592–856639、Base 856640、分析 856641，独立审计登记 856516 完成并提交审计 856644。首个 L40S 训练作业因共享 `QOSGrpGRES` 等待；H100 当前健康节点 gpu17 的四张卡都已分配、gpu16 不可用，H200 亦受共享配额限制，没有能立即执行该任务的替代路由。确认旧 H200 作业 856445/856446/856461 均属于 `youyang7` 且为 PENDING 后已取消，保留旧日志、冻结产物与[路由切换记录](../../../results/phase13_baseline_expansion_v1/exploratory/token_kd_baselines_gsm1024_l40s_v1/protocol/route_switch_h200_to_l40s_v1.json)。这些提交与 smoke 通过不构成完成的效果结果。

## MATH：先消除真实阻塞，再放行完整 baseline

原 raw 32 个分片均有完成标记，但旧合并 822575 因 `Candidate grading differs` 失败，下游 824506 被取消。reviewed steering 具有完整 80,676 条候选；reviewed raw、全池 DAP/TokenSkip、共同学生训练集尚未完成。

新 CPU 协议 `configs/phase13_kd_math_source_preparation_v2.json` 先归档全部 53,784 条 immutable raw 候选，对 shard、题目、候选网格、token/text 和 batch cost 做审计。归档保留历史评分，但不为其可复现性背书；随后逐条比较 stored、typed-v2 与 reviewed 评分，输出所有差异案例。该步骤不生成新的教师文本、不解锁旧作业、不直接放行训练。v1 直接重跑方案在提交前停止，历史冻结残留保留。

差异审查通过后才能冻结 reviewed raw 选择、全池 DAP/TokenSkip、八方法共同支持、长输入 KD smoke 和 MATH 训练/跨数据集评测协议。当前不得称 7,500 题 MATH KD 已提交或已完成。

MATH v2 已提交 CPU 候选归档 856437 → 评分差异审计 856438，实际 batch 已验证；作业提交不代表评分问题已解决。

独立末端审计另冻 `configs/phase13_token_kd_baselines_audit_v1.json`：在动态矩阵分析之后递归核对本实验的完成标记、adapter、逐题身份与解码配置，重算全部汇总指标，并核对 76,881 条预测和成对训练监督 token。分析完成与独立审计通过分别记录。

## 2026-09-16 MATH 续跑审计与提交

53,784 条 student-pool raw 候选和 64 条 smoke 已归档并通过 token/text、题目网格及原分片哈希核验。reviewed 评分审计 856438 COMPLETED：53,848 条中 216 条 correctness 变化，旧 typed-v2 correctness 复算零差异。216 条涉及 41 题；逐条案例检查结果为 209 条新版正确答案识别、2 条最终答案确实错误、5 条已知保守解析漏判。另有 292 条 reviewed 解析错误（含 219 条未识别完整 final box），保持不合格并公开错误分类；当前评分文件不覆盖这些漏判。完整历史评分字段诊断 856471 COMPLETED，53,784 条零差异。该诊断支持一次受控原合并重试，不能解释旧 822575 的瞬时失败原因。

据此提交原冻结代码的 raw 合并重试 856477；其成功后才允许 reviewed raw 迁移 856480（预设 216 条 correctness 变化）。其后是修订 scratch 的压缩测试 856484、准备 856485，压缩 DAG 注册 856490，全部共同学生数据注册 856494。后两个注册器只在上游成功并有完成标记时提交下游；预定压缩 DAG 为 H200 DAP、L40S TokenSkip 各两路的 smoke、32 个互斥分片和合并。完整学生数据准备完成后仍需单独冻结 MATH Qwen3B KD/SFT 训练、最大输入 smoke、五数据集评测；此处没有 MATH KD 效果结果。

上述作业中任一项 PENDING/RUNNING 仅为调度状态。原 822575/824506、旧 held 作业及其产物保留，没有对它们解除 hold、重用原旧临时路径或覆盖原结果。
