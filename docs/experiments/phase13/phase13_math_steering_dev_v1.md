# MATH：B3 / B4 / B7 的统一采样开发比较

本阶段给统一学生蒸馏选择教师运行强度。它保留开发队列与学生题池隔离；输出不能直接用于学生 SFT。

## 方向与复用

`math_steering_directions.py` 复用 `sae_mechanism_readback.py` 的 response hidden-state 捕获、prompt 编码和单位方向规范化。抽出的共享函数避免给 dense 控制另写一套模型前向或保存不必要的完整词表 logits。已有冻结实验的源码快照保持原位。

B3 使用 ASC 同一批 100 对 MATH calibration 长短轨迹，但在与 B7 相同的 block 17 提取状态。每条 response 内先平均 hidden states，再按题等权平均 concise−verbose 差并单位化。Verbose 使用原始采样 token IDs，核对文本及 EOS 后移除 EOS；官方 concise reference 在同一 native chat prompt 后单独编码。该控制混合了参考与模型输出的格式、内容、长度差异，不叫 ASC-CES。

B7 将历史八个 short SAE decoder rows 先取 BF16、再以 FP32 求和并规范化，复现已有数值构造；另保存三个预先抽取的随机集合方向。它是历史格式关联特征向 MATH 的迁移，未重新发现或确认语义特征。

B4 使用 MATH 主教师正式 CES/KL 拟合的向量，在原注册 block 16 做绝对向量加法。不同方法的方向范数标度不相同，所以分别登记相对 hidden norm 剂量和 ASC scale，不将二者的数值直接等同。

7 项来源、实际小模型 hidden-state、cache、BF16、窗口与读回检查 `822221` 通过，准备 `822178` 完成。L40S 提取 `822263` 完成：100 对、extraction 10.40 秒、peak allocated memory 14,840.33 MiB；dense 未规范化范数 12.3975，与 SAE 单位方向 cosine −0.02986。准备/源码/方向的 11/427/6 项绑定已验证。

## 开发矩阵

- 相同 64 道 MATH development 题，每题每条件四个真实候选。
- B3、B7 各用 rho=0.05/0.1/0.2/0.3；B4 用 scale=0.1/0.25/0.5/1。
- 总计 12 条件、3,072 条开发输出；先做 8 题、384 条 smoke，并额外逐 token 重放每个条件的首候选。
- 使用 B1 相同 prompt、4K cap、temperature 0.7、top-p 0.95、top-k 20、逐题/候选随机流和正确过滤。准备时验证 B1 的全部设置、题目记录、完成标记与源码，不能仅凭相同题号认定可比。
- B3/B7 仅在当前 live 的最后位置加入相对范数方向，B4 同位置加入绝对方向；记录实际扰动，生成时不运行 SAE encoder。

强度选择规则已在该扫描前记录：候选准确率和正确题目覆盖相对 B1 各最多下降 5 个百分点，cap-hit 不超过 1%；在可用点中，按与 B1 的共同正确题集上最短正确监督的平均 token 选择，平局取较低强度。无点合格则明确报告不可行，不以 identity 代替失败基线。这是开发用的工程筛选，不是准确率非劣检验；全部强度、失败点与完整题池覆盖都须报告。

## 调度与完成条件

准备 `822279` 已完成，17 项输入/父绑定与 429 项源码绑定验证通过。H200 smoke `822330`、CPU smoke 审计 `822331`、八个开发分片 `822332`–`822339` 以及合并 `822340` 已按成功依赖提交。开发分片最多四个同时运行，H200/L40S 分担；后四片还依赖同 lane 前一片成功。CPU 合并指定当时已核查健康的 `c201n02`。

提交不代表完成。后续必须验证实际 token/EOS、逐条答案、各题/条件/候选覆盖、范数、成本和最终共同支持，才能据扫描结果冻结学生题池的生成协议。

配置：[phase13_math_steering_directions_v1.json](../../../configs/phase13_math_steering_directions_v1.json)、[phase13_unified_steered_math_candidates_dev_v1.json](../../../configs/phase13_unified_steered_math_candidates_dev_v1.json)。


## 自动选择与派生分析

`13_33_analyze_math_steering.py` / `unified_steering_analysis.py` 已冻结，测试 `822400` 的两项反例检查通过：准确率/覆盖/cap 不合格点即使更短也不入选，平局取低强度，无共同支持不回退为 identity。分析 `822401` 依赖完整合并 `822340` 和测试成功。

“配对平均长度”落实为每道共同正确题的 method−B1 最短监督 token 差，再按题等权平均；这是当前扫描结果尚未完成前的明确化。候选准确率/长度区间先在题内平均四个候选，再按题重采样，不将 256 条候选作为 256 道独立题。图沿用既有蓝/绿/红方法比较约定及 Matplotlib 样式，显示相同候选分母的 B3/B4/B7 与 B1；DAP/TokenSkip 已经过正确源过滤，所以不把它们的附回答案分数塞入该候选准确率曲线。

分析同时关联已审计的 B0/B1/B2/B5/B6 开发支持，输出可用运行点和八方法共同题号。所有结果仍限于 development；单独的学生生成协议尚未冻结。
