# 文档索引

文档按用途组织，实验记录再按阶段存放。当前执行状态以 [PROJECT_STATUS.md](../PROJECT_STATUS.md) 为准；目录位置不表示实验已经完成。

| 目录 | 内容 |
| --- | --- |
| [experiments/](experiments/README.md) | Phase 0–13 的实验协议、阶段记录及报告 |
| [environment/](environment/) | NCSU 环境配置和历史集群规则 |
| [latex/](latex/README.md) | 论文 LaTeX 工作稿、相关工作及研究提案 |
| [manuscripts/](manuscripts/README.md) | 精简版、扩写版论文交付包及扩写版源码 |
| [tutorial/](tutorial/README.md) | SAE 训练与 feature-to-token 中英文教程 |
| [presentations/](presentations/) | 演示材料及英文单页说明 |
| [handoff/](handoff/HANDOFF.md) | 当前实验交接、停止门槛及接手入口 |
| [archive/](archive/README.md) | 本次整理前的文档原文、旧路径和哈希映射 |

## 常用入口

- [SAE 长短特征可解释性项目](experiments/phase13/phase13_sae_feature_separation_project_zh.md)：独立机制研究计划及[历史消融回顾](experiments/phase13/phase13_sae_feature_evidence_review_20260916_zh.md)。
- [实验交接](handoff/HANDOFF.md)：Answer-free SAE 当前结论、完成产物和下一轮边界。
- [Phase 13 文档](experiments/phase13/README.md)：基线扩展、数据审计、SAE 分析、学生蒸馏及 GSM8K 小试。
  - [KD/SFT 八基线扩展与 MATH 续跑](experiments/phase13/phase13_token_kd_baselines_expansion_zh.md)。
- [Phase 12 文档](experiments/phase12/README.md)：NCSU 复现、SAE 干预及多答案蒸馏。
- [NCSU SFT 环境](environment/ncsu_sft_environment.md)。
- [历史集群规则](environment/legacy_cluster_compute_rules_20260913.md)：仅适用于文中指定的历史集群。
- [SAE 英文单页说明](presentations/sae_representation_one_slide_en.md)。

## 存放约定

新增阶段文档放在 `experiments/phaseN/`，保留描述性文件名及版本号，并更新该阶段索引。环境说明、教程、论文和演示材料分别放在对应目录。实验配置仍存放在仓库的 `configs/`，结果和完成标记仍存放在 `results/`。

2026-09-16 的整理只调整文档位置、导航和生成报告的路径。历史实验的科学设置、结果和完成标记未改写。旧文档哈希核验应使用 [归档原文与路径映射](archive/README.md)，不能用已调整相对链接的阅读版替代原文。
