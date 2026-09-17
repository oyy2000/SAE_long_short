# Phase 13：八方法共同支持与首阶段学生蒸馏

配置 `configs/phase13_unified_student_sft_math7_5k_v1.json` 与入口 `scripts/13_35_distill_unified_students.py` 承接真实 MATH 学生题池。本阶段的准备代码和五项反例检查已完成；只有全部方法源数据完成后才能执行实际准备。当前没有据此产生正式学生 adapter 或性能结果。

## 数据的完整性与共同支持

准备阶段要求普通教师、steering 和文本压缩三个根目录的协议、源码、完整学生池合并标记全部有效。题池必须逐记录、逐顺序一致，且只含原来隔离的 6,723 道学生问题。原始和 steering 阶段的 32 个分片还需有实际 H100 硬件记录。

B0/B1/B2/B3/B4/B7 的选样由完整候选重新调用共用 selector 得到，并与已保存的选择逐项相等；不把 B0 随机正确选择改成 B1 最短正确选择。DAP 从全部真实改写重新选择最短正确结果。TokenSkip 的来源必须是同题 B1 最短正确 trace，完成文本必须保留由已验证源预测答案构造的后缀。

只在八种支持的共同交集上构造等样本数据，按题号排序，各方法、学生和种子共享同一题集。保存各方法完整支持数量及每道排除题缺失的方法。缺少整个方法、空交集、重复题号、越界问题、问题文本改变、错误或达到 cap 的监督都会拒绝进入训练。最终题数由真实交集决定，不重复问题凑到 6,723 或 7,500。

普通方法使用相同的数学 native-chat prompt，B2 的 concise 条件仅作用于教师。B6 保留完整 TokenSkip 比例提示、指定比例和对应 completion，每题一条，不扩成六条训练样本。保留 near-duplicate component ID 和源选择哈希；这些审计字段不会进入模型训练输入。

## 分词、训练与证据

复用已经通过两个实际学生检查的 `completion_supervision` 和 TRL wrapper。全部共同样本分别按两种 Qwen tokenizer 构造明确的 prompt 掩码、答案/EOS 标签；超出 8,192 token 直接报错，不能按方法静默截断或删样本。保存每个方法和 tokenizer 的实际输入及监督 token 数。

模型、LoRA rank 8 / alpha 16、三轮基准训练设置必须与已完成合成容量检查的基础配置一致，实际依赖版本和模型文件也要再次核对。首阶段 S4a 只启动 Qwen-1.5B 的 B1/B4/B5/B7，三个学生 seed 17、42、73，共 12 次；其余四个方法及后续模型/规模扩展仍需完成，不能用这 12 次替代全目标。

所有训练使用相同 microbatch 1、梯度累积 8、学习率 5e−5、cosine schedule、warmup 0.1 和 completion-only 标签。继承 pinned Trainer 的实际更新规则，分别记录配置轮数、实际 epoch、优化步数、每题曝光及真实监督 token，避免将相同轮数误报为完全相同 token 预算。当前训练审计只支持单进程。

训练入口拒绝覆盖旧尝试，并将缓存、中间文件和临时 adapter 放在 `/share/jekml/youyang7/tmp` 下的作业独立目录。只有预期更新数完成、最终权重有限且 LoRA 已更新，才发布 hash-verified adapter、配置和训练指标；训练完成标记分别绑定数据、运行配置、冻结源码和两个最终 LoRA 文件。训练完成与五个 benchmark 的评测完成分开标记。

## 当前验收记录

`822655` 的五项检查通过，覆盖全八方法交集、完整支持/排除、B0/B1 区别、普通学生提示、TokenSkip 条件提示、开发集/外来问题混入、重复题及错误/截断监督。两个模型的实际训练接口证据见 [合成 SFT 检查](phase13_unified_sft_interface_v1.md)。后续八项图/共同数据检查 `823697` 也已通过。

共同数据准备 `823770` 已在 [依赖流程](phase13_unified_pipeline_v1.md) 中提交，必须等待 raw `822575`、steering `822780`、text `823769` 三个完整合并。本阶段仍未绑定实际共同题数，准备尚未执行，正式 SFT 未提交；排队不算训练或数据完成。新的冻结训练源码会按编码数据大小与至少 2 GiB 的底限检查 job-local scratch，并把检查结果绑定到最终 adapter 完成标记。

源生成和文本压缩的实际成本随准备保存，B0/B1 的共同原始生成只计一次；一次性 SAE/ASC 校准/拟合成本以及后续训练/评测成本仍需分别加入，不能由数据压缩比例直接推出端到端收益。

## 修订参考后的当前入口

原配置/69-job DAG 保留为历史；当前后继配置为 `configs/phase13_unified_student_sft_math7_5k_reviewed_v1.json`，消费修订后的完整 raw、steering 和文本根。模型、训练配方和首阶段 12 次保持不变。原 raw 的 smoke-only 修订上下文不满足本阶段；完整重评分与新文本合并尚待完成，实际共同题数尚未建立。执行顺序与路径见 [修订续跑](phase13_reviewed_baseline_continuation_v1.md)。

2026-09-15 storage correction: the current launcher uses administrator-designated scratch. Historical frozen launchers and previously submitted Slurm scripts retain their original paths and require separate migration verification.
