# Phase 13：修订评分后的 baseline 续跑

2026-09-13 23:16 EDT。沿用原方法、模型、采样和开发选点规则，将完成的 MATH 参考修订接入实际 baseline 流程。历史协议、预测及失败尝试保持原位。本文记录执行状态，不将排队阶段记为完成。

## 已完成

| 阶段 | 作业 | 结果 |
| --- | --- | --- |
| 校准输入重建 | `824462` | 完整来源池 395 题，重放 117 题的 180 个必要候选，恰好 100 对；07414/0 替换 06821/0，184 个历史候选全部保留，没有新生成 |
| ASC-CES 拟合 | `824469/824470` | 3,000 步逐步日志及 3,584 参数向量核验通过，H200 拟合约 571.58 秒；64 条通用文本保留集平均 KL 0.018417，新旧方向 cosine 0.925484 |
| dense 方向重算 | `824471/824473` | L40S 提取完成；dense 新旧 cosine 0.999938，SAE 八特征及三个随机 SAE 方向数值完全相同 |
| raw 开发候选迁移 | `824474` | 8 题 smoke / 64 题开发，共 576 条记录，正确性标签零变化；原生成文本、token、种子、分片及成本保留 |
| DAP/TokenSkip 新开发运行 | `824496`–`824505` | 49 项测试通过；DAP 194 条，TokenSkip 六比例共 336 条；两方法各有 56/64 题可用监督 |
| 学生接口来源校验 | `824521` | 17 项测试通过，另用真实 smoke/development 视图核验原始生成来源；拒绝混用评分后端、修改采样设置、缺失分片或篡改硬件记录 |
| H100 raw 上下文迁移 | `824526` | 64 条 smoke 记录、零标签变化；绑定全部学生问题的修订 metadata，没有完成学生池候选选择 |
| 修订 steering 开发扫描 | `824484`–`824492` | 全部八分片及合并完成，12 条件 × 64 题 × 4 候选，共 3,072 条；完整合并 35 项直接绑定通过 |
| 新开发选点 | `824527` | B3 rho=0.2 / layer17、B4 scale=0.25 / layer16、B7 rho=0.3 / layer17；各 56/64 题支持，八方法交集 54/64；26 项绑定通过 |
| 新学生 steering 准备 | `824529` | 完整 6,723 题输入，实际方向/层/强度绑定；与原 raw 的 H100 路由、采样、种子、32 分片和题序一致；26 项协议及 531 项源码绑定通过 |
| 修订原生 ASC 评测与分析 | `824535/824544` | 320 条完整开发输出；生成 5 项与分析 116 项直接绑定通过，与统一四候选扫描分开报告 |

ASC 的 KL 项仍为 soft hinge penalty，单次保留集均值低于 0.02 不等于保证所有输入满足 KL 约束。拟合完成不证明压缩或蒸馏有效。

DAP 完整开发输出平均从源轨迹的 547.40 降至 479.12 tokens，整体缩短 12.47%，194 个最终答案均通过修订评分、零 cap hit。194 条生成文本与原开发运行逐条一致。该分母是已过滤为正确的源轨迹，不是全题池的独立教师正确率；最终答案评分不认证推理主体。DAP 改写耗时合计 470.81 秒，TokenSkip 六比例压缩合计 7.34 秒，原 B1/B2 生成成本另记。TokenSkip 保留比例条件和附加源答案；其附答案正确率不能证明压缩推理质量。

执行复核见 [23 项直接绑定与 21 个完成作业](../../../results/phase13_baseline_expansion_v1/preparation/math_reference_integrity_gate_v1/calibration_rebuild_execution_v1/COMPLETE.json)，文本结果见 [开发输出复核](../../../results/phase13_baseline_expansion_v1/preparation/unified_text_compression_dev_reviewed_v1/verification_v1/summary.json)。

## 已提交，尚未完成

1. 新学生 steering：H100 smoke `824669`、smoke 合并 `824670`、32 个学生分片 `824671`–`824702`、完整合并 `824703` 已提交，最多四条学生生成通道。预期 80,676 条 B3/B4/B7 候选；最后回读 35 个作业均 PENDING，smoke 等待 H100 资源，全部成功依赖逐项相符。提交标记的 75 项直接绑定通过，见 [新生成依赖核验](../../../results/phase13_baseline_expansion_v1/preparation/unified_math_candidates_steered_reviewed_launch_v1/generation_chain_v1/verification_v1/COMPLETE.json)。这是生成提交，不是数据或 SFT 完成。
2. 全 raw 学生候选评分审计 `824506` 继续等待原合并 `822575`。当前原 raw 完成八个分片，13,464 条候选，各五项直接绑定通过；分片 08–11 仍在运行。全审计覆盖 6,723 题、53,784 条 B1/B2 候选及 64 条 smoke，变化输出须复核后才能迁移完整选择及文本/SFT 输入。

新选点保持原规则：先过滤正确候选率、正确覆盖和 cap-hit，再最小化逐题最短正确监督相对 B1 的平均 token 差。B3/B4/B7 分别为 −38.69/−19.70/−64.80 tokens，配对题数分别 54/56/54。选点不等于所有原始候选都更短，也不证明非劣。原生 ASC 的无干预 52/64、658.48 tokens 与 scale 0.5 的 49/64、623.36 tokens 是另一个单候选采样检查，不能替代此统一选点。

首次在 login 节点回读时尚未看到分析输出，稍后同一作业的完整文件可读且所有绑定通过。没有重新提交已完成分析；该现象不足以单独判定文件系统原因。

旧待运行 steering `822752`–`822779`、合并 `822780` 和文本入口 `822579` 保持 hold。历史运行中的原始生成继续保留。不能解除旧入口来替代新协议迁移。

## 后续执行顺序

新 H100 学生 steering smoke 和完整生成（开发选点/准备已完成）；完整 raw 评分影响审计 → 变化案例复核 → 全量重评分选择 → DAP/TokenSkip 学生源压缩 → B0–B7 共同支持、token 和来源审计 → 分阶段 SFT 与五评测集比较。

raw smoke 上下文仅提供原生成配方、硬件和全部问题顺序的来源绑定，不是完整学生训练数据。SFT 必须读取单独完成的全 raw 重评分选择。7.5K 来源池、25K 扩展、多学生/teacher/seed、预算比较和 SAE 稳定性仍是原目标的一部分，当前未完成全部 baseline 蒸馏比较。

原 raw 分片 07（`822550`）新增 1,680 条完成候选；旧 steering 分片 03（`822751`）完成 2,532 条，只保留为旧方向证据。上述新完成阶段的直接绑定和作业状态见 [本轮生成进展](../../../results/phase13_baseline_expansion_v1/preparation/math_reference_integrity_gate_v1/reviewed_continuation_progress_v1/COMPLETE.json)。

## 后续完整文本与 SFT 配置迁移

已准备 `configs/phase13_unified_text_compression_student_reviewed_v1.json` 与 `configs/phase13_unified_student_sft_math7_5k_reviewed_v1.json`。前者指向待建立的完整 `formal/unified_math_candidates_raw_reviewed_v1`，后者连接同一 raw 根、新 steering 根和新文本根。它们不是已冻结数据协议，也未据此提交训练；实际 raw 评分变化数仍需等待 `824506` 并逐例复核后登记，不能先假定为零。

保持原 6,723 题来源、32 分片、DAP 解码、TokenSkip 比例与分配种子、两学生模型、全部 LoRA/训练参数和首阶段 12-run 范围不变。仅迁移来源/输出路径、显式 reviewed 评分后端及 CPU 调度：移除旧 `c201n02` 固定节点，由获准 `compute` 分区按原 CPU/内存/时限请求分配。历史已冻结任务保持原样。旧 69-job DAG 不能通过放行旧入口接入新来源；待新父阶段就绪后重新冻结依赖。

学生提示和 base/合成 LoRA 接口均已完成额外执行检查，见 [统一评测接口](phase13_student_evaluation_interface_v1.md)。接口通过并不完成正式学生训练或五数据集评测。
