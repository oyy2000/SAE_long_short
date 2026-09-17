# GSM8K SAE 学生收益小试

## 目标与冻结范围

用户确认第一轮回到 GSM8K：Qwen2.5-7B-Instruct 教师，Qwen2.5-3B-Instruct 学生，目标 1,024 道共同训练题，14 次 seed-17 SFT。SAE 监督须比 B0 未干预随机正确轨迹更短，但不要求比其他压缩方法更短；主要优化目标是学生开发集准确率。MATH/MATH-500 留到后续扩展。

配置：[phase13_gsm8k_qwen3b_sae_pilot_v1.json](../../../configs/phase13_gsm8k_qwen3b_sae_pilot_v1.json)。入口：`scripts/13_60_run_gsm8k_student_pilot.py`，首次冻结/提交入口：`scripts/13_62_launch_gsm8k_student_pilot.py`。结果独立存放在 `results/phase13_baseline_expansion_v1/exploratory/gsm8k_qwen3b_sae_pilot_v1`，adapter 位于对应的 checkpoints exploratory 根。历史产物和作业不变。

## 数据与方法

使用固定 revision `740312add88f781978c0658806c59bc2815b9866` 的 GSM8K train。排除已登记的历史 881 题、1,000 题机制输入和 smoke，以及全部测试问题的 exact/near 匹配；沿用 5-gram Jaccard 0.8 的保守排除规则。训练池近重复组仅保留一个代表，按固定哈希顺序抽取 2,048 题并预留至 3,072 题。开发使用既有机制实验 300 道 dev，校准来源为独立 400 道 discovery。这些开发/校准队列已观察，不能称为全流程盲测。

共同支持要求全部 14 个条件均有正确、非截断监督。首轮交集不足 1,024 时才生成预留的 1,024 题；仍不足则使用不超过目标的最大 8 的倍数，少于 768 题停止。每题一条监督，TokenSkip 六比例不产生六次问题曝光。

| 方法 | 设置 | SFT 次数 |
| --- | --- | ---: |
| B0 | 未干预池随机正确，固定选择 seed | 1 |
| B1 | 同一未干预池最短正确 | 1 |
| B2 | Concise prompt 最短正确 | 1 |
| B3 | layer17 dense，rho 0.1/0.2/0.3 | 3 |
| B4 | layer16 ASC-CES，scale 0.1/0.25/0.5 | 3 |
| B5 | 全文 DAP 难度感知改写 | 1 |
| B6 | TokenSkip 原有六档比例条件，每题一个预分配比例 | 1 |
| B7 | 原 SAE 八特征，layer17，rho 0.1/0.2/0.3 | 3 |

直接生成每题每条件四候选，配对随机流，temperature 0.7、top-p 0.95、top-k 20、1,024-token cap。B0/B1 共用原始生成成本。DAP 改写所有唯一正确未截断 B1 源，再取最短正确；保留完整 Thought/Solution。B6 从 B1 最短正确轨迹构造，附加答案来自验证过的源预测，不是 gold 偷补。

B3/B4 使用同一批至多 100 对 GSM8K 校准轨迹，重新冻结 native-chat prompt 与来源，不能直接使用 MATH 方向。ASC 保留 3,000 步 CES 与 KL 软约束；报告完整目标和保留集 KL，不能把有限 penalty 称为逐输入硬保证。SAE 固定原 checkpoint 与特征，不进行新的语义发现。

## SFT、选点与测试

复用 `unified_student_distillation.train` 和 completion-label 接口：LoRA rank8/alpha16/dropout0，七类 projection，学习率 5e-5，三轮，microbatch1/accumulation8，cosine/warmup0.1，BF16，gradient checkpointing，completion-only，关闭 packing，8,192 上限。记录真实优化步数、每题曝光和监督 tokens；超长监督报错，不能静默截断。

B3/B4 按学生 dev 准确率选点，即使最强点没有压缩也保留。B7 先满足平均教师监督 tokens 低于 B0，再按学生准确率选；平局先教师监督更短，再较低强度。B6 在开发集选六个推理比例中的最高准确率，平局选输出更短者。全部候选点保留，测试不参与选择。

统一 greedy 与 GSM8K v3 最终答案评分；新增通用评分分派，不改变旧配置默认评分。开发检查 512/1024/2048/4096 cap，要求所有模型截断率不超过 1%，相对 4096 的准确率下降不超过 1 pp，再冻结最小共同 cap。若最高 cap 仍不合格，流程停止在测试冻结之前。开发短 cap 来自相同 greedy 输出的前缀，不虚构独立采样或延迟测量。

测试固定 `test[50:1319]`，共 1,269 题；`test[:50]` 仅 smoke。所有 14 个初始运行点与 base 保存完整预测；主方法运行点在开发集先冻结。单 seed 使用题目配对区间和 exact McNemar，SAE 与七 baseline 的 p 值采用 Holm 校正。多 seed 不把同题重复输出当独立样本。

补 seed 门槛在测试前判断：SAE 比 B0 短，学生 dev 准确率高于 B0/base，且分别比 B1 与 B2–B6 的最强方法高至少 1 pp。满足后仅给 B0、B1、最强竞争条件、B7 各补 42/73，共增加 8 次，累计 22 次。补 seed 不重新选强度；若需要更大公共 cap，仅更新公共预算并保留原选点。未过门槛也完成已注册首轮测试。

## 执行和证据

复用统一候选 decoder/selector、DAP/TokenSkip 引擎、训练审计、原生学生 greedy、prefix cap 检查和统计绘图。新的独立 orchestration 只负责 GSM8K 角色、十四条件共同支持和学生选点；原 MATH 八方法完整性限制保持原样。校准可选的 sampled-text/token 保存避免重新分词改变 dense/CES 对应关系。

阶段依次为 CPU 测试和输入冻结、校准生成、ASC smoke/fit、dense/SAE 方向、raw smoke/完整池、steering smoke/完整池、文本压缩、共同支持、SFT/dev、选点/可选补 seeds、冻结测试与分析。阶段内最多两条 GPU lane，阶段间 afterok 依赖；每次提交先写 intent，结果不确定时不自动重提。每一阶段只读取已完成并 hash-bound 的父产物。

当前注册 L40S 用于生成/训练/评测，H200 用于 ASC；每次 GPU 作业先做独立短进程 admission，避免父进程占用 exclusive-process CUDA context。最低空闲显存 40,000 MiB，scratch 基础预留至少 4 GiB，SFT 还要求不少于四份实际 encoded 输入大小加 1 GiB；记录 UUID、实际硬件、显存和时间。L40S 2 小时时限需通过实际 smoke 吞吐检查，不能由 H200 合成检查推断吞吐。失败 attempt 保留，不重置 GPU 或取消其他作业。

主要产物：学生准确率图、教师长度—学生准确率散点、强度—学生 dev 准确率曲线，以及来源覆盖、成本、逐题预测、adapter 和完成标记。本轮只支持探索性 GSM8K 学生收益结论，不证明高层语义、SAE seed 稳定性或跨数据集泛化。

## 每小时小模型监控

使用 `gpt-5.6-luna`、`reasoning_effort=medium`，每 3,600 秒审阅一次只读 Slurm/完成标记/至多四个日志尾部快照。监控子代理不提交、取消、重试或修改实验；主代理负责诊断和更改。调度器完成与 marker 存在分别报告，不能把存在性检查称为完整独立 hash 审计。

入口 `scripts/13_61_monitor_gsm8k_student_pilot.py`。NCSU 当前用户 crontab 被 PAM 禁止，因此使用独立 session 的轻量元数据监控进程；每次通过非交互式、read-only Codex 调用小模型并保存输出。调用依据 [Codex 非交互模式](https://developers.openai.com/codex/noninteractive/)，小模型调用已实际 smoke 验证。

监控目录为 `results/phase13_baseline_expansion_v1/preparation/gsm8k_qwen3b_pilot_monitor_v2`，每次含 snapshot、模型事件、review 和状态。`daemon.json` 记录 PID 和首次计划时间；`latest.json` 指向最近检查。完成标记发布后最后检查并退出；可在该监控目录创建 `STOP` 仅停止监控，不影响实验。报告写入文件，不声称会自动推送聊天通知。

## 2026-09-14 启动记录

输入冻结已完成：排除与近重复代表选择后可用 5,583 题，首轮 2,048 题，累计预留 3,072 题；dev 为 300 题，locked test 为 1,269 题。这些是输入规模，尚非十四条件共同训练支持。

CPU 路由经当前 Slurm association 核验，采用 `jekml_cpu / compute_partners / short`，4 CPU、32 GiB、2 小时。原 `compute / normal` 的三个尚未运行 bootstrap 作业 830033–830035 因排队被替换，保留原快照和取消记录；不改变其他实验作业。第二次 bootstrap 的 830038 已通过 38 项检查，830039 完成输入冻结。

在 GPU 启动前修复正文长度诊断的 Answer/boxed 正则，并增加回归检查；原始输入协议和源码快照不改写。当前执行配置为结果根下 `execution_v2/protocol/frozen_config.json`，独立绑定新源码、原始输入协议和修订原因。补测 830043 成功后由 830044 启动校准链；旧的 held begin 830040 已登记为 superseded 并取消。后续阶段继续使用该执行配置。实验问题、方法矩阵与选点规则不变。

每小时监控进程 PID 1438785，首次计划时间为 2026-09-14 12:23:35 UTC（08:23:35 EDT），之后每 3,600 秒一次；监控 v1 已通过 STOP 停止，v2 跟踪当前执行链。轻量后台进程不具有主机重启后的自动恢复保证；其 PID、命令和最近报告均记录在监控目录。

39 项补测及 begin 均完成。校准 830046 在 gpu38 因本地 scratch 不足 8 GiB 提前失败；830070 保持原协议重试，记录 `df -B1 /var/tmp` 并排除该节点。830047 的 afterok 依赖已更新为 830070，830047/830048 同样临时排除 gpu38；全部原失败证据保留在 launch 目录。后续 830049–830051 仍依赖成功链。

实测 gpu39 的 `/var` 总量仅 8,522,825,728 bytes，空闲 7,263,576,064 bytes，原统一 8 GiB 门槛无法满足，因此 830070 同样在模型加载前停止。资源修订另存 `execution_v3/protocol/frozen_config.json`，复用已通过 39 项测试的 execution_v2 源码，基础空闲预留改为 4 GiB；SFT 继续取 `max(4 GiB, 4 × encoded input bytes + 1 GiB)`，且 `save_strategy=no`。原输入/科学设置不变，未修改冻结旧文件，校准 inputs 继续复用。当前执行配置应使用 execution_v3。新链 830071–830076 取代尚未运行的 830047–830051；取消和替换记录逐项保留。监控 v2 的模型/时间/结果根不变，可继续覆盖新链。

恢复启动验收：830071 在 gpu38 RUNNING，已通过 admission 并完成 Qwen 四个模型分片加载；校准生成仍在运行，无完成标记，不能记为学生结果。

## 2026-09-14 09:08 EDT：按更新后的存储规则恢复完整新题比较

用户明确继续新题池、四候选和完整 B0–B7。旧 830075 被统一 4 GiB 本地盘检查拦截；此前仅下调阈值没有解决不同节点容量差异，且报告阻塞后未及时修复。当前 `execution_v4/protocol/frozen_config.json` 替换该运行策略，保留科学设置和原失败记录，复用已完成的 100 对校准、ASC smoke/fit、方向输入，不重复生成或拟合。

新增共享 storage helper 按阶段预算：方向提取只在内存计算，输出预算来自配对数、hidden size 和真实输入文件；生成/评测按题数、候选、方法、输出 token 与 JSON 记录大小估算发布空间；训练按实际 encoded 文件的四份副本、模型尺寸计算的三个 FP32 LoRA 副本预算，不保存周期性 checkpoint，不复制模型权重。小型 tokenizer/绘图库缓存预算 16 MiB，整体加两倍安全系数。检查实际使用文件系统；本地不足则使用 `data/runtime/gsm8k_qwen3b_pilot/<job>/<stage>` 的共享临时目录，并同时预算发布产物，执行写入探针。`quota -s` 无 quota 数据，不据此声称已获得独立 quota 保证；项目导出容量检查约有 1.99 TB 空闲。真实写入失败仍会停止并保留证据。

新增 5 项存储测试在本地通过；完整回归 830248 → 恢复方向 830249 → raw begin 830250。旧阻塞的830076已登记superseded后取消。后续仍最多两条GPU lane，经 smoke、共同支持和开发选点门槛依次执行。

监控迁至 `preparation/gsm8k_qwen3b_pilot_monitor_v3`，固定 luna/medium 每小时；增加 hostname、每分钟 heartbeat 和异常记录，单次快照/调用异常不会直接结束监控循环。v2已写STOP。旧监控未记录hostname，仅在当前登录节点查不到PID不足以证明其在另一登录节点也已停止；此前关于监控停止的表述证据不足。v3启动主机login04，PID4174071；报告仍为文件，不自动改动作业或向聊天推送。

修复验收完成：830248的44项检查全部通过；830249完成100/100配对提取并发布方向文件。主代理验证方向COMPLETE和测试标记哈希，保存 `execution_v4/verification/RECOVERY_COMPLETE.json`。临时/输出预算分别33,554,432/40,147,778 bytes；本地实际空闲6,025,125,888 bytes，未触发回退；共享回退路径由边界测试验证，不能称为本次GPU实际使用共享临时目录。当前未有新学生结果。

## 2026-09-15 00:00 EDT：steering 改为32分片、四路混合GPU，并启用Slurm自动恢复

用户明确要求根据可用GPU尽快恢复、失败重试。原steering smoke 832680完成，832681根据16分片推算9,642秒超过short_gpu 2小时限制，阻塞下游。现冻结 [execution_v5](../../../results/phase13_baseline_expansion_v1/exploratory/gsm8k_qwen3b_sae_pilot_v1/execution_v5/protocol/frozen_config.json)：新steering输出根 `round_1/steered_split32`，每片64题，保留题目/四候选/种子/强度；原raw完整结果和steering smoke复用。4路轮转H200、H200、L40S、L40S；当前H200节点idle、L40S mixed，H100无idle。H200真实吞吐待新任务测量，不据此声称已测得更快。

839091回归 → 839092新smoke门槛 → 839093–839124共32个GPU分片 → 839125合并 → 839126压缩入口。旧832682–832699全为当前用户未运行的PENDING，已逐项superseded后取消；保留旧失败记录。

839127为Slurm CPU恢复控制器，每分钟检查本execution的终态GPU失败，最多额外3次重试。先保留不完整输出，再单次提交，按实时Slurm依赖重接后续；提交不确定时不盲重提，CANCELLED不重试，已完成adapter有hash标记则复用。TIMEOUT使用登记的H100四小时时限重试。只对generate/compress/train-dev/train/evaluate-dev/evaluate-test阶段自动处理；CPU、未知阶段、已有完成标记但状态异常、达到重试上限的情况写NEEDS_ATTENTION，不自动改变协议。每55分钟安排下一控制器周期（上限144），采用1CPU/2GiB；Luna/medium每小时在控制器中只读审阅。4项新增单元检查通过；模拟集成验证失败归档、一次重提、依赖更新及重复扫描不重复提交。新控制器实际启动后再停旧monitor v3。

最终监控分工：`controller_v2` 是Slurm托管的每分钟恢复控制器，只操作execution_v5的已登记终态GPU失败；Luna仍在login04的monitor_v3每小时只读审阅。计算节点外部API超时的旧控制器已替换，Luna网络请求不再阻塞重试轮询。GPU实验代码仍为execution_v5，未因控制器修订重启运行中的生成。

最终恢复服务为 `controller_v3`，worker继续execution_v5。真实L40S CUDA故障触发了自动归档、重提、依赖更新；15个未启动L40S任务由主代理迁至空闲H200。后续CUDA故障自动切换GPU家族，OOM转H200，TIMEOUT转H100四小时时限；最多额外3次，取消任务不重试，确定性数据/配置错误写NEEDS_ATTENTION。控制器与login04的Luna小时监控分离，避免计算节点外部API不可达拖慢恢复。

## 2026-09-15 09:48 EDT：TokenSkip 原生库 ABI 故障修复，恢复完整压缩阶段

本次延误不是持续计算：steering全部32分片及合并已完成，DAP smoke840408完成15条；TokenSkip840409于03:42因NLTK→sqlite3导入加载节点旧libstdc++，缺少CXXABI_1.3.15而失败，840478/840480/840483三次重试同错后达到上限，阻塞约6小时。旧报告中的CPU controller RUNNING不能当作GPU实验运行。

冻结execution_v6，通过启动器opt-in LD_PRELOAD现有sft环境的libstdc++.so.6；绑定该库及libgcc哈希，未安装/升级包，未修改旧执行快照。登录节点已验证torch→sqlite3/nltk/llmlingua导入成功；新增计算节点回归用例。ImportError明确列为确定性依赖错误，不再盲目消耗三次GPU重试。

新链842428回归 → 842429 H200 TokenSkip smoke → 842430 smoke合并 → 842431–842494 DAP/TokenSkip各32分片（4路H200） → 842495完整合并 → 842496共同支持/SFT。新控制器842497继续只读Luna之外的自动恢复，使用独立abi_recovery前缀续期，避免旧controller周期命名冲突。旧840410–840476全为本用户未运行pending，逐项保存superseded后取消；已完成教师轨迹、DAP smoke保留，失败TokenSkip目录归档，840483警报有明确RESOLUTION。主结果仍未完成SFT或收益比较。

## 2026-09-15 23:55 EDT：execution_v7 指定 scratch 与 SFT 准备恢复

本次执行使用 `execution_v7/protocol/frozen_config.json`，保留原训练/生成设置和已完成来源。项目 `tmp` 链接至 `/share/jekml/youyang7/tmp`；每个作业使用独立子目录，临时文件、HF dataset/编译缓存和 Trainer 中间输出统一落在该处。Python 存储准入核验目录真实路径、工作量预算与 GPFS 用户/组配额；空间不足报错，不回退到系统目录或项目 `data/runtime`。现有模型缓存继续复用，最终 adapter 经过哈希验证后发布到项目 checkpoints。

修复答案区域诊断对命名 `marker` 分组的错误假设，历史命名分组仍保持原语义。842496 已写出的两份 B0 文件保留在 `execution_v7/preserved_failed_842496`；已完成共同支持审计经过绑定和内容比对后复用。848103 完成 56 项计算节点测试，848123 从 support 阶段续跑；848124 恢复控制器只处理新 execution_v7。三个实际 Slurm 批处理脚本已回读验证。历史 held MATH 作业没有解除，旧冻结快照没有覆盖。最新启动与完成状态以 PROJECT_STATUS.md 及完整审计标记为准。
