# Phase 13：OpenThoughts 25K 来源可行性审计

目标是判断后续 25K 唯一数学问题是否有足够的可审查来源。当前 7.5K 来源阶段的八方法完整生成、共同支持 SFT 和成本检查仍按原计划推进；本审计没有选择或启动新的 25K 训练题池。

现有 OpenThoughts 默认配置的 parquet 只有 system/conversations，没有独立标准答案。本次复用 `13_19_stage_dap_downloads.py`，新增可配置 `file_prefix`，原默认 `data/` 行为保持不变；按 revision `bd093c3994fd54d2390985b66988ddf282a55eb6` 获取 12 个 `metadata/` 分片，共 2,469,730,242 bytes。传输标记只表示大小符合，不能替代上游 LFS SHA-256 验证。

`configs/phase13_openthoughts_inventory_v1.json`、`src/length_budget_distill/openthoughts_inventory.py` 和入口 `scripts/13_44_inventory_openthoughts.py` 执行以下步骤：

1. 核验固定 revision、全部 12 个文件的上游 LFS SHA-256/大小、所需字段类型和 113,957 行总量。
2. 仅读取 `problem`、`ground_truth_solution`、`domain`、`source`。为全部行保留原 file/row/global ID 与字段哈希，为数学行保留原问题和参考解答；不使用 `deepseek_solution` 或 `deepseek_reasoning` 作为标准答案。
3. 复用现有 balanced-box 提取、问题规范化和 DAP 明确格式前缀移除规则，统计唯一问题及重复参考 box 的词汇冲突。原文本不进行反斜杠解码或其他静默修订。
4. 对证明措辞、图像依赖、CJK 字符、缺失问题/参考/box 设置复核标记。它们是保守词汇规则，存在漏报和误报，不能作为可靠任务/语言分类。
5. 对五个完整评测来源（含 GSM8K smoke 部分）、MATH 校准/开发/DAP 保留题、SAE 新题/smoke/历史题执行精确和既有 5-gram Jaccard ≥ 0.8 匹配。每个参考输入核查原完成标记中该文件的直接绑定，不声称递归审计所有历史依赖。当前 6,723 道 MATH 学生题的重叠单独报告为信息项。
6. 输出预筛候选 ID、原文清单、重复组、精确/近似匹配和汇总。预筛候选移除词汇标记、box 冲突及精确/近似保留集匹配；近似匹配本身仍需复核。

本阶段验证的参考答案数、选定训练问题数和生成教师答案用作 gold 的数量均为零。来源内部近似重复、参考答案语义与类型、英文纯文本可回答性、实际教师正确覆盖及共同支持仍未验证；预筛数量不能直接称为 25K 可用训练样本。没有原 LiteCoT 25K 样本 ID，不能声称数据完全相同。

冻结目录为 `results/phase13_baseline_expansion_v1/preparation/openthoughts_inventory_launch_v1`；410 项输入/源码绑定。四项测试检查禁止生成答案回填 gold、证明/图像 box 不授予资格、嵌套 box 提取和保留/学生题角色隔离。测试作业 `823881`、成功依赖后的审计 `823882` 使用 NCSU CPU `jekml_cpu / compute / normal`，后者 4 CPU、24G、2 小时，指定检查过负载与容量的 `c026n04`。实验完成以最终审计、结果绑定及实际 Slurm 状态为准。

## 已执行结果

`823881` 和 `823882` 均 COMPLETED / 0:0，分别 52 秒和 2 分 16 秒。额外完整性复核 `823894` COMPLETED / 0:0（21 秒），验证来源结果的 42 项直接绑定，重新核对全部行 ID、字段哈希、数学数量、词汇标记、候选排除集合，并按 seed `2026091324` 从十个包含数学记录的原 parquet 文件各抽查四行。验证的 3 项绑定及下述重叠复核的 4 项绑定另行核验通过。

实际来源为 113,957 行，其中 89,120 行数学、19,904 行代码，其余为生物、物理、化学及谜题。89,120 个数学问题经现有规范化均唯一；89,011 条参考包含可提取 box。证明措辞标记 22,794 条、视觉依赖标记 702 条、CJK 标记 32 条、缺失/空 box 109 条，标记之间可能重叠。

690 个唯一数学问题匹配评测或其他保留题（精确或近似），其中与五个评测来源有两对精确、四对近似匹配。项目助手复读四对近似问题：三对 OlympiadBench 为同题的格式/符号变体；一对 MATH-500 共用邮票表格材料但提问不同。六个来源问题均保持保守排除，不能将四对 near 一概称为四道重复题。该复核不是独立人工标注，完整决定及输入哈希在 `preparation/openthoughts_overlap_review_v1`。

另有 5,900 对与现有 MATH 学生池的精确匹配、47 对近似匹配，作为题池关系信息记录。最终 64,946 个预筛候选意味着来源数量值得继续评估；尚无参考正确性验证、来源内部 near 去重、教师正确覆盖或八方法共同支持，不据此宣称已获得 25K 正式训练数据。见 [完整报告](../../../results/phase13_baseline_expansion_v1/preparation/openthoughts_inventory_v1/report_zh.md)。
