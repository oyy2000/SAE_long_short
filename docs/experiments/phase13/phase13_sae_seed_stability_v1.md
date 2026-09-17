# Phase 13：SAE 随机种子复验

复用原 layer 17、TopK-64、28,672 features、batch 256、1,500 steps、相同学习率与归一化设置。原 seed 17 在训练脚本、运行依赖、样本和最终权重哈希验证后复用；新训练 seeds 为 42、73、101。固定原 250,000/50,000/50,000 train/dev/test token 样本，新的 seed 配置与 sample view 显式绑定原 manifest，没有重新抽样或改变原配置文件。

训练逻辑仍是冻结的原 `2_4_train_topk_sae.py`；新增 `sae_seed_stability.py` 负责输入核验、执行与最终权重发布。中间 checkpoints 写在`/share/jekml/youyang7/tmp` 下的作业独立目录，最终权重经有限值、decoder 单位范数与哈希检查发布至共享项目路径。协议见 `configs/phase13_sae_seed_stability_v1.json`。

## 已完成训练与原规则筛选

| SAE seed | 成功训练 job | 完成 steps | 原规则 short 数 | 原规则 long 数 | 八个 short 是否可构造 |
| --- | --- | ---: | ---: | ---: | --- |
| 17 | 历史完成证据复用 | 1,500 | 8 | 8 | 是 |
| 42 | 823190 | 1,500 | 3 | 8 | 否 |
| 73 | 823191 | 1,500 | 5 | 9 | 否 |
| 101 | 823192 | 1,500 | 7 | 6 | 否 |

三个新模型均成功保存最终权重；训练本体分别耗时 89.73、88.42、89.34 秒，不含失败/准备/评分成本。不能以训练完成代替特征与生成稳定性。

特征准备 `823168`、原规则评分 `823252/823253/823254` 均完成。复用同一 109 个 layer-17 activation chunks、原 132 道 discovery 与 132 道 held-out 问题、原 scorer 及 NumPy 1.26.4 等依赖。这里只替换每个 seed 对应的训练权重；原六-SAE 完成证据用作语料来源，不声称每个 seed 新训练了六个字典。

各 seed 按原 discovery rank 在 confirmed features 中选择，不以同编号或 cosine 对应替换。原八特征构造在三个新增 seed 上均不可行，必须作为稳定性限制报告。后续统一前三个 confirmed short 的比较属于看到数量后的探索性适配，不能声称原八特征方法已成功跨 seed 复现。

`sae_seed_analysis.py` 复用 `mutual_decoder_matches` 和 `answer_marker_diagnostic`，核对相同 held-out trace 支持、双向 decoder 最近邻、激活相关与 Answer/body/suffix 关联。沿用历史 0.5 cosine / 0.5 correlation 门槛仅作描述；保留全部数值，不改变原确认资格。图形沿用既有字体、配色与布局习惯。六项测试 `823369` 和分析 `823370` 均已完成。四个 top-3 方向的跨 seed cosine 约 0.11–0.14；六组字典配对均未找到同时满足原确认资格、双向最近邻及两个描述门槛的 short feature 对。

short features 的 Answer 激活占比范围见 [报告](../../../results/phase13_baseline_expansion_v1/exploratory/sae_seed_analysis_v1/report_zh.md)。seed 101 的 F19908 不以 Answer 为主要激活词，但其 96.25% 激活质量在冒号，96.19% 在答案标题区域；这不是摆脱格式关联的证据。区域关联仍不等于完整因果解释。原评分实际确认 seed 73 的 long features 为 9 个，选择输出只截取前 8 个；上表报告完整确认数。

## 新生成的探索性补充

`configs/phase13_sae_seed_generation_v1.json` 单独登记：沿用已观察的 64 道 confirmation 消融题、原 rho=0.3、1K cap 与逐题随机流，统一 H200、四分片，每题比较未干预、四个 seed 的前三个 confirmed short 方向、seed17 原八特征参考、Answer-format 和 dense，共 512 条；另有独立 8 题 × 8 条件 smoke。所有输出重新生成并另行答案复核，旧输出不代替新评分。

复用 `sae_generation_controls.run_registered` 的共享 rollout、实际范数检查、区域统计与完成审计，只传入已冻结的字典方向、条件和输入。方向按原 BF16 decoder 舍入后 FP32 求和归一化，此次在 CPU 构造。解码中的 SAE 激活读回仍使用历史 seed17 字典，逐条标明；它不是新增 seed 自身的 target engagement。

18 项数值/角色/规格检查 `823418`、准备 `823419` 和 H200 smoke `823475` 均已完成。协议/源码有 14/474 项直接绑定，smoke 的 7 项绑定已核验，64 条输出零 cap hit、峰值 allocated 15,309.25 MiB。准备重核了已完成 seed 分析的 75 项绑定。四个生成分片 `823476`–`823479` 已在四个不同 H200 UUID 上全部完成，各 128 条、7 项直接绑定已核验。

新答案准备 `823510` 与最终评分 `823565` 完成：576 条输出、20 个需审阅 case，项目助手阅读 15 个完全相同问题/答案句/规则组合，8 条正误标签改变，零未解决 case。审阅没有读取 gold、方法或旧标签，不声明独立人工标注或全文语义验证；本次没有需要补读全文的歧义。所有决定绑定新输出 hash，时刻的分钟保留为精确小时分数。

分析 `823610` 已完成，18 项直接绑定与图已核验。复用既有逐题完整性、token/EOS/cap、区域、实际范数和复核身份审计。四个 top-3 条件分别与未干预、原八特征、格式和 dense 做题内配对描述区间；区间不是同时检验，不提供由 bootstrap 尾部比例构成的 p 值，不证明非劣。

未干预为 63/64、287.61 tokens；top-3 seed17/42/73/101 分别 60/63/61/62 正确、270.25/305.59/279.30/278.92 tokens，全部零 cap hit。seed17 相对未干预缩短 17.36 tokens，seed42 反而增加 17.98；二者边际描述区间分别 [−30.36,−4.25]、[4.91,31.78]，其他两个 seed 的长度区间跨零。这一探索性补充未支持跨 seed 稳定缩短，也不能改变原八特征构造不足的复验结果。完整参考方向、正文长度和逐题区间在 [生成报告](../../../results/phase13_baseline_expansion_v1/exploratory/sae_seed_generation_analysis_v1/report_zh.md)。

## 失败与恢复记录

- 初次 `823088/823089` 在 gpu18 完成 1,500 步后因本地存储不足，最终权重保存失败；`823090` 在训练前因独占进程模式下父进程占用 CUDA context 而失败。三者均不算成功训练证据。共享部分输出移到 `failed_attempts/`，原日志与冻结代码保留。
- gpu18 后被 Slurm/NHC 标记为 drain，原因是 `/var` 与 `/var/log` 满。归档 `823126` 仍等待该节点恢复；尚未宣称其临时文件已归档或清理，也未重置设备、干预其他用户或关闭已有分配。
- `isolated_gpu_preflight` 在独立子进程完成 CUDA 检查并退出，随后启动原 trainer，避免父 CUDA context 与独占进程模式冲突。
- 第一轮恢复 `823154/823155/823156` 被统一 8 GiB scratch guard 拒绝，未训练。第二轮按实测 checkpoint 大小核算：五个 checkpoint 大小加 1 GiB 余量，共 5,184,879,484 bytes，并分配到 gpu19/20/21；`823190/823191/823192` 成功。新 sizing 检查与测试记录保留。
- 首次特征准备 `823096` 因 NumPy 版本与原环境不同在执行前拒绝；v2 明确使用既有 legacy overlay，未改评分规则或依赖。旧失败证据保留。

失败作业消耗属于构建成本；不得只把约 90 秒的成功 trainer 时间称为端到端成本。临时文件只可在所有权、作业终态与共享发布哈希验证后按限定路径清理。

2026-09-15 storage correction: the current launcher uses administrator-designated scratch. Historical frozen launchers and previously submitted Slurm scripts retain their original paths and require separate migration verification.
