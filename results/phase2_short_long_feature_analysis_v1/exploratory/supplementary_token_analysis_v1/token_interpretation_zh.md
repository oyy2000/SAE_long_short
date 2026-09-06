# Short/long SAE feature 的 token 解释

## 核心结论

Primary SAE（layer 17, k=64）有 13 个 dev 发现、test 确认的 feature：5 个 short-associated，8 个 long-associated。但是 token 分析表明两侧不是对称的潜在推理机制。

- 5 个 short-associated feature 的最高 activation-mass token 全部是 `Answer`；该 token 单独解释每个 feature 总 activation mass 的 54.2%–97.0%。
- short-associated feature 在前 80% 轨迹中几乎不激活，主要在最后 20% 突增。因为 `Answer` 基本每条轨迹只出现一次，同样一次收尾激活除以更短的序列长度，会机械地产生更高的 token-normalized trace mean。
- 8 个 long-associated feature 的最高 token mass 更分散，单一 token 占比为 3.3%–32.1%；其 token/context 多为公式换行、章节/步骤编号、Substituting/Find/Conclusion 等过程展开，以及单位和变量标签。
- long-associated feature 在首 64 token 已呈同方向（paired d 为负），说明长轨迹的展开风格在采样分岔后较早出现；但这仍然可能是格式和措辞，不足以证明 extra computation。

## 研究含义

当前结果支持的是可复现的轨迹状态/风格差异，而不是 pre-generation short intent，也不是 student utility。进入 steering 前应优先对 long-associated feature 做 lexical scrubbing、token injection 和 paraphrase invariance；short-associated `Answer` features 不应作为增强 short trace 的候选。
