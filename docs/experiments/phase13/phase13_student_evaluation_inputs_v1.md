# Phase 13：学生评测提示与上下文核验

`822715` 已完成两个测试及实际 tokenizer 核验，输入预检的 20 项直接绑定、冻结 launch 的 461 项源码/配置绑定均通过。入口为 `scripts/13_36_check_student_evaluation_inputs.py`，配置为 `configs/phase13_student_evaluation_inputs_v1.json`。

两个 Qwen 学生分别检查 GSM8K 1,269、GSM8K-Hard 1,269、MATH-500 500、AQuA-RAT 254、OlympiadBench 674 道问题。每题检查共同提示和六个 TokenSkip 比例提示，总计 55,524 个 native-chat 输入。最长输入为 1,315 tokens，加上候选 4,096 输出预算仍在 32,768 上下文范围内；没有溢出。

初版选择题统一要求最终 boxed 答案为选项字母，适用于 AQuA 的 254 题及当时 MATH-500 metadata 识别出的两道选择题。该规则同样用于 base 和所有训练条件，提示不读取参考答案。GSM8K 正式范围仍为 test[50:1319]，Hard 的父题映射保留。

这里没有生成或评分任何测试答案，也没有据此选择最终 cap。冻结实际评测仍需开发集解码预算检查、模型/adapter 绑定、完整分片和输出审计；不能把上下文容量足够视为输出不会被截断。

## 修订 cohort 后的提示复核

2026-09-13 23:02 EDT。`824587` COMPLETED / 0:0，四项检查及两个学生全部 55,524 个 native 输入重新核验通过。共享选择题判断优先读取 `reviewed_answer.kind`，只读取题型，不读取参考值。修订后 MATH-500 有三道选择题；新增识别 `math500-00296`。与初版逐键比较，恰好该题的两学生 × 七提示共 14 条改变，其余 55,510 条完全相同。

新配置为 `configs/phase13_student_evaluation_inputs_reviewed_v1.json`；最长输入仍为 1,315 tokens，没有生成测试答案或冻结最终解码 cap。启动 536 项、结果 22 项直接绑定已验证，见 [修订核验](../../../results/phase13_baseline_expansion_v1/preparation/student_evaluation_inputs_reviewed_launch_v1/verification_v1/COMPLETE.json)。实际 base/LoRA 解码接口的合成检查见 [评测接口](phase13_student_evaluation_interface_v1.md)。
