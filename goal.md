初步实验计划
Phase 0：先验证 observation，暂时不要训练 SAE

你已经有四个 teacher、short/medium/long、三个训练 seed 的矩阵，可以直接做这一阶段。

先固定一个主设置：

Teacher: Qwen2.5-7B-Instruct
Student: Qwen2.5-1.5B-Instruct
Dataset: GSM8K train
Rollouts: 16 / question
Training seeds: 17, 42, 73

每个问题只保留至少有 4 条正确轨迹的情况，然后在问题内部定义：

short  = bottom 20% correct traces
medium = closest to median
long   = top 20% correct traces

主实验不要只使用 absolute shortest/longest，因为极值容易选到 answer-only 或陷入重复循环的异常样本。

必须同时做三种训练公平性设置：

Equal examples：完全相同的问题，每题一条 trace；
Equal target tokens：总监督 token 数相同；
Equal optimizer/token budget：总训练 token updates 或近似 FLOPs 相同。

另外比较：

token-normalized loss；
sequence-normalized loss；
是否只对 answer token 加权；
是否存在长 trace 稀释关键步骤梯度的问题。

主指标：

student exact-match accuracy；
student 平均输出长度；
training tokens；
paired question-level improvement；
三个 seed 的均值和置信区间。
Gate 0

只有当 short 优势在 equal-token 和不同 loss normalization 下仍然存在，才继续讲“trace content/latent behavior”；否则你的发现更可能是训练预算或优化归一化现象。

Phase 1：先定义 student teaching utility

这是最关键的一步，也应该在 SAE 之前完成。

对 student \(S_\theta\)、候选 trace \(\tau\) 和独立 validation set \(V\)，定义：

$$ U_S(\tau) = \mathcal L_V(\theta) - \mathcal L_V \left( \theta-\eta\nabla_\theta\ell(\theta;\tau) \right). $$

其中：

\(U_S(\tau)>0\)：吸收该 trace 后，student 在未见问题上的 loss 降低；
\(U_S(\tau)<0\)：该 trace 虽然正确，但更新方向损害泛化。

为了避免每条 trace 都实际更新一次，可以先使用一阶近似：

$$ \widehat U_S(\tau) \approx \eta \left\langle \nabla_\theta\mathcal L_V(\theta), \nabla_\theta\ell(\theta;\tau) \right\rangle. $$

只在 LoRA 参数上计算梯度即可。需要同时报告：

per-token normalized gradient；
per-sequence normalized gradient；
local utility：在 15–32 个相似难度问题上评估；
global utility：在 128–256 个随机验证问题上评估。

然后在同一问题内部拟合：

$$ U_{q,r} = \beta_{\text{len}}\log |\tau_{q,r}| + \beta_{\text{NLL}}\operatorname{NLL}_S(\tau_{q,r}) + b_q + \epsilon_{q,r}, $$

其中 \(b_q\) 是 question fixed effect。

这一阶段要回答
同一问题内，长度是否真的与 teaching utility 负相关？
short 的优势是否只是因为 student NLL 更低？
短 trace 是否“容易学但没信息”，即 loss 降得快但最终泛化不强？
你的 CTV 排名能否预测完整 SFT 后的排序？

把 RSR、SCAS、LARK-style score、student NLL 和 length 作为 baselines。它们已经建立了 student-specific trajectory suitability 这一研究方向，因此你的 utility 必须证明比这些表面指标更接近真实训练增益。

Phase 2：训练 SAE，但目标改成解释 utility，而不是分类长度
数据

Pilot 可以使用：

2,000 questions
16 rollouts / question
约 32,000 raw trajectories
仅在分析标签中区分 short/medium/long

训练 SAE 时必须把所有 trace 混在一起：

$$ \mathcal D_{\text{SAE}} = \mathcal D_{\text{short}} \cup \mathcal D_{\text{medium}} \cup \mathcal D_{\text{long}}. $$

不要分别训练 short SAE 和 long SAE，否则两个 dictionary 不可直接比较，而且容易引入 label-specific artifacts。

初步设置：

residual stream；
选择约 40%、65%、85% depth 的三个 layer；
TopK SAE；
expansion factor 8；
\(k\in\{32,64\}\)；
按 question 做 train/dev/test split，而不是按 trajectory；
pilot 一个 SAE seed，正式实验至少三个 seed，并匹配稳定 decoder features。

对每条 trace 提取：

feature activation frequency；
mean/max activation；
activation onset；
first 64 tokens 内的 activation；
step-level pooled activation；
whole-trace activation area。

RISE 和 SSAE 已经说明 step-level representation 能更直接地捕捉 reasoning behavior、step correctness 和长度属性，因此 step pooling 至少应当作为 token-level SAE 的对照。

不要只做这个回归
$$ \text{ShortOrLong}\leftarrow \phi_{\text{SAE}}(\tau). $$

这只能证明 SAE 能识别长度，已经不新。

应该做：

$$ U_{q,r} = \beta_{\text{len}}\log|\tau_{q,r}| + \beta_{\text{NLL}}\operatorname{NLL}_S(\tau_{q,r}) + \boldsymbol\beta^\top \phi_{\text{SAE}}(\tau_{q,r}) + b_q+\epsilon. $$

核心问题是：

SAE feature 在控制 length、student NLL 和 question difficulty 后，能否继续解释 teaching utility？

最终会得到四类 feature：

	High student utility	Low student utility
Short-associated	应增强	不能因为它短就增强
Long-associated	应保留	应抑制

这张 2×2 表本身会比“short feature / long feature”更有研究价值。

Phase 2.5：必须做 SAE falsification

这是不能省略的。最近的系统性研究发现，contrastive SAE 所谓的 reasoning features 很容易只是 “wait”“let me check”“therefore” 等 lexical/style cue；大量候选 feature 可以通过向无关文本注入几个相关 token 激活，而且其 steering 并没有改善 reasoning benchmark。

每个核心 feature 至少通过以下检查：

Token injection：把 top activating token 放进非推理文本，feature 是否仍然激活；
Paraphrase invariance：逻辑不变、措辞改变后 feature 是否稳定；
Lexical scrubbing：删除 Wait、So、Let、Therefore 等词后是否仍存在；
Position control：只比较相同 absolute token position；
Early-prefix prediction：前 32/64 tokens 的 feature 能否预测 remaining length 或 utility；
Non-reasoning counterexample：寻找高激活但完全没有相应 reasoning behavior 的句子。

否则 reviewer 会合理地认为你只是控制了 hesitation tokens，而不是“extra computation”。

还有一个理论上必须说明的问题：

同一 prompt 在生成第一个 token 之前 hidden state 完全相同，因此 natural short/long trace 并不存在一个确定的 pre-generation “short intent feature”。

真正可研究的是：

sampling 分岔后，哪些早期涌现的 trajectory-state features预测并推动后续的冗余推理和低教学价值？

Phase 3：做 causal feature intervention

选择在不同 split 和 SAE seed 下都稳定的 5–20 个 utility features。

设 SAE 表示为：

$$ z=\operatorname{Enc}(h),\qquad h\approx W_{\mathrm{dec}}z+r. $$

对 feature code 做干预：

$$ \widetilde z_j=z_j+\alpha_j, \qquad \widetilde h = h+ W_{\mathrm{dec}}(\widetilde z-z). $$

更推荐 activation-gated suppression：

$$ h_t' = h_t - \sum_{j\in \mathcal F^-} \alpha_j \mathbf 1[z_{t,j}>\delta_j] z_{t,j}w_j + \sum_{j\in \mathcal F^+} \alpha_j w_j. $$

其中：

\(\mathcal F^-\)：student-utility-negative features；
\(\mathcal F^+\)：student-utility-positive features；
不应该简单等同于 long 和 short features。
最强的 causal design

对一个问题：

先生成完全相同的 32/64-token prefix；
clone KV cache；
使用相同 sampling seed 分出：
negative intervention；
no intervention；
positive intervention；
比较后续长度、正确率和 student utility。

这样比从头独立采样三个结果强得多，因为它控制了初始 reasoning path。

Intervention baselines

至少需要：

No steering
Dense short–long difference vector
SAE shortness features
SAE student-utility features       <- Ours
Norm-matched random SAE features
Orthogonal SAE features
Explicit “answer concisely” prompt

必须报告两类结果：

Unconditional：所有生成结果的正确率、长度、重复率；
Correct-conditioned：正确 trace 中的长度和 utility。

不能只过滤正确结果后汇报，因为强 steering 可能让 80% 输出出错，而剩余 20% 看起来特别短。还应汇报：

$$ \text{Correct usable traces per generated million tokens}. $$
最关键的比较

对 SAE-shortness 和 SAE-utility 生成的 traces 进行 length matching：

$$ |\operatorname{len}(\tau_{\text{short-SAE}}) - \operatorname{len}(\tau_{\text{utility-SAE}})| <5\%. $$

如果长度相同时，utility-SAE trace 仍然具有更高 CTV 和更好的下游 SFT 效果，才能证明你控制的是 teachability，而不只是 brevity。

Phase 4：正式 student distillation

初期只做 SFT，一题一条 trace。

最小数据条件
1. Random correct trace
2. Shortest correct trace
3. Longest correct trace
4. Dense length-steered trace
5. Post-hoc compressed trace
6. SAE shortness-steered trace
7. Student-utility SAE-steered trace
8. Random-feature-steered trace

所有条件使用：

相同 question IDs；
每题一条 trace；
相同 student initialization；
seeds 17、42、73；
equal-example 和 equal-token 两套 budget；
相同验证和 checkpoint selection。

对干预后无法生成正确 trace 的问题：

主 paired comparison 使用所有方法均能生成正确 trace 的 question intersection；
同时单独报告各方法 coverage 和 generation cost。
Primary endpoint

不要把“teacher trace 变短了”作为主结果。主结果应该是：

$$ \text{Student accuracy under a fixed supervision-token budget}. $$

Secondary endpoints：

student 平均推理长度；
accuracy per generated token；
training wall-clock；
student train loss convergence；
OOD generalization；
generated-trace coverage；
teacher generation cost。
Phase 5：做一个决定性 cross-student 实验

这是最可能把论文从“SAE length steering”变成真正新工作的实验。

使用同一个 teacher，但两个 student：

Student A: Qwen2.5-1.5B
Student B: Qwen2.5-3B

分别学习：

$$ v_A^{\text{utility}},\qquad v_B^{\text{utility}}. $$

生成两套数据：

$$ \mathcal D_A=\text{Steer}(T,v_A), \qquad \mathcal D_B=\text{Steer}(T,v_B). $$

然后交叉训练：

Student	\(D_A\)	\(D_B\)
Student A	应该更好	较差
Student B	较差	应该更好

如果出现这种 diagonal advantage / student–data interaction，就能直接证明：

这些不是通用的“短输出特征”，而是面向不同 student capacity 的 pedagogical features。

这是 RISE、S³-CoT 和普通 trace compression 都没有回答的问题。