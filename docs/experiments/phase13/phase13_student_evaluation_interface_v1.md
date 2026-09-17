# Phase 13：统一学生评测接口的合成检查

复用 `student_evaluation.generate_native_greedy_batch`，由 `unified_student_evaluation.py` 统一提示、精确 token 记账和五种数据集评分分派。入口为 `13_52_check_student_evaluation_interface.py`，冻结配置为 `phase13_unified_student_evaluation_interface_v1.json`。TokenSkip 原生检查也调用共同解码函数，并保留其原有模型默认 repetition penalty；统一学生比较显式设为 1.0。

CPU `824596` COMPLETED / 0:0，10 项检查通过，涵盖 EOS 与解码文本边界、上下文/forced-EOS 拒绝、显式与默认 penalty、答全多个根及大整数精确评分。Qwen-1.5B/3B 的 L40S 作业 `824597/824598` 分别在 38/47 秒内完成，各生成 30 条合成输出：base 和已完成合成 LoRA × 无比例/0.5/1.0 三种提示 × 五种合成题型。真实开发/测试答案生成数为零。

两个 base 的原始 repetition penalty 分别为 1.1 和 1.05；统一入口实际均为 1.0，避免把不同模型默认设置带入比较。两次 LoRA 加载均为实际 `PeftModelForCausalLM`，具有活动 adapter。峰值 allocated 显存：1.5B base/LoRA 3020.27/3080.86 MiB，3B 6087.41/6173.99 MiB；合计生成计时分别 20.45/28.68 秒。计时不含全部加载与来源验证成本，不作为正式吞吐比较。

256-token 接口 smoke 分别发生 7/4 次截断，均保留。该 cap 仅用于检查边界，不是正式解码预算。完整学生评测仍需完成真实八方法训练来源与 adapters、开发集预算检查及独立冻结的评测分片/合并协议。合成 adapter 不作为正式学生初始化，合成答案正确率不报告为学生质量。

[核验产物](../../../results/phase13_baseline_expansion_v1/preparation/unified_student_evaluation_interface_launch_v1/verification_v1/COMPLETE.json) 验证了启动 540 项、两个学生各 11 项直接绑定，60 条完整 student/variant/ratio/problem 网格、控制参数、summary 与 Slurm 终态。冻结 GPU worker 同时执行完整提示/token/评分重算审计；本次回读未再次加载 base 模型权重。
