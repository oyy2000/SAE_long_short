# Phase 13：统一学生训练接口与合成数据检查

本阶段复用 `training.py` 的 TRL SFT 和 LoRA 实现，新增显式监督标签及真实训练曝光计数。配置为 `configs/phase13_unified_sft_interface_v1.json`，入口为 `scripts/13_34_check_unified_sft.py`。它为 B0–B7 的后续统一学生比较准备训练接口，不代表主学生 SFT 已完成。

## 监督边界与成本

`completion_supervision.encode_completion` 分别渲染 native generation prompt 和完整 user–assistant 对话，要求后者包含前者作为原始文本前缀。独立分词 prompt 与 assistant suffix 后拼接，保持生成时已经固定的 prefill 边界，不因跨边界 BPE 合并改变哪些 token 被监督。保留模板的 EOS/end-of-turn 及后续空白，并把它们计入实际监督预算；正文文本 token 数单独保存。

预处理生成 `input_ids`、全 1 的 `attention_mask` 和 prompt 为 −100、完整答案后缀为原始 token ID 的 `labels`。拒绝空答案、超出上限、内部掩码缺口和模板边界不一致。`DataCollatorForSeq2Seq` 仅按长度给标签补 −100，保证 pad token 等于 EOS 时真实 EOS 标签仍被保留。答案中再次出现 assistant 分隔文字不触发重新寻找监督边界。

新 `pretokenized_completion` 路径不启用额外 completion/assistant 自动掩码或 packing。训练前逐记录验证 TRL 保留的 token 与标签，并用长短样本检查实际 collator。`TrainingExposureAudit` 在真实训练 `compute_loss` 中核查输入，原样调用 TRL 的优化逻辑，记录成功 forward 的实际输入 token、causal shift 后监督 token、padding token、microbatch 数和逐题曝光。梯度检查点重算不重复计数；当前审计仅支持单进程，不能把单 rank 计数当作多卡全局预算。

这项计数用来报告训练轮数、优化步数与问题曝光之间的差异。固定样本、固定 token 和固定步数仍是不同实验，不能同时宣称全部匹配。测得训练时间包含批次审计开销。

## 检查设置与证据

每个 Qwen2.5-1.5B/3B-Instruct 学生使用 16 条独立合成样本，交替采用普通 boxed 数学提示与真实 TokenSkip 比例提示；序列接近 512、2,048、4,096、8,192 token。它们不读取开发、校准或学生题池，合成 adapter 也不会用于主实验初始化或性能评测。

检查采用 LoRA rank 8 / alpha 16 / dropout 0，覆盖 q/k/v/o/gate/up/down projection；microbatch 1、梯度累积 8、两个更新、BF16 和 gradient checkpointing。学习率为 5e−5。正式的三轮、三 seed、统一样本支持及 TokenSkip 推理比例协议仍需随真实训练数据另行冻结。两个更新需要实际覆盖全部 16 条记录且每条一次，所有 LoRA 权重有限且至少部分 B 矩阵更新，才发布完成标记。

首个 CPU 作业 `822513` 因小模型测试夹具缺少 pinned TRL 要求的 `torch_dtype` 而失败，未进行数据准备。明确设置 CPU float32 后，独立冻结的 `822519` 通过六项检查，包括真实小型 Qwen 因果损失对照、真实 TRL/LoRA 更新、部分 epoch 的实际曝光、EOS/padding 及污染反例。准备也完成；两个学生各 16 条序列，最长 8,189 token，无截断、无真实数据。8 项协议绑定与 438 项源码绑定已复核。

H200 检查 `822531`（1.5B）和 `822532`（3B）均已 COMPLETED / 0:0。两者都完成两个更新、16 个 microbatch、每题一次曝光，共 59,326 输入 token、58,176 个 causal-shift 后监督 token。两套 adapter/训练完成标记的共 20 项直接绑定已核验；LoRA B 中分别有 196/252 个非零张量。

1.5B 的峰值 allocated/reserved 显存为 18,114.43/26,218 MiB，3B 为 21,746.97/30,038 MiB。含加载、保存和批次审计的训练阶段时间为 11.13/11.54 秒；Trainer 自身计时为 4.07/6.28 秒。这是合成序列容量检查，不代表真实数学训练吞吐或学生性能。H200 实测显存为后续硬件选择提供依据，其他型号仍需按实际显存和 walltime 准入。

1.5B 使用 GPU UUID `24738f2e-d18e-5fbf-d2ff-4c28f9f117de`，3B 使用 `eacc2725-c1f4-a3ae-4885-0041081c91d4`。原 CPU 失败与修正快照完整保留。
