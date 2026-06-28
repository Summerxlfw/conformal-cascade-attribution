# P13 Novelty 对抗查重 — conformal-prediction 邻居深读

目标论文 P13: *Per-stage conformal inefficiency in a speech-to-intent cascade: a validated decomposition and the limits of gold-free repair targeting.*
精确贡献 = 真实 SLURP ASR→NLU 级联，**固定部署阈值（ASR-calibrated）**下，把**分类预测集大小膨胀（set-size inflation）**分解成 ASR-attributable (~31%) + NLU-intrinsic (~69%)；gold-transcript 反事实 + 第二个更强 ASR (medium.en) 双重因果验证；证明最优 deployable **gold-free** selective-repair ≈ 平凡 "largest-set-first"（ASR-internal word confidence 仅 partial Spearman +0.135）。

所有结论均回 arXiv / 官方元数据核对。2026-06-28 增补了 ConformaDecompose 与 cascaded predictive modules calibration 两个近邻，并补查了 Whisper confidence / ASR correction 作为应用背景。

---

## 1. 差异化表

| arxivID | 标题短名 | 分解对象 | per-stage? | 固定部署阈值? | 真实语音 cascade? | gold 反事实验证? | gold-free repair? | 贴近度判定 | P13 应使用的一句话差异化 |
|---|---|---|---|---|---|---|---|---|---|
| **2309.12510** | Confidence Calibration for Cascaded Predictive Modules (Gong/Yao/Lin/Divakaran/Gervasio) | 回归系统 prediction interval / system-level error | 是（module-level validation supports cascaded system interval calibration） | 否（目标是系统级区间校准） | 否（Matterport3D overlap prediction 等系统） | 否 | 否 | **新增近邻—pipeline calibration 但非 set-size attribution** | 它解决多模块系统缺少 end-to-end validation 时的区间可靠性；P13 解决固定部署阈值下分类 set-size inflation 的 stage 归因与 repair audit。|
| **2510.04406** | Decomposition-Based Modular CP for Two-Stage Modeling (Zhang/Amin/Perakis, MIT) | **回归残差宽度**（interval width），分解 R≤ΔR1+R2 | **是**（upstream ΔR1 / downstream R2） | 否（按 component 重标定 + FWER 选 scaling，外加 adaptive sliding-window 重标定） | 否（合成 + 供应链/股市回归；upstream 也可是 logistic 但 downstream 是回归 cost） | 否（用真实中间变量 x 算 R2，但无"更强上游"因果对照） | 否 | **近邻—需重定位差异化（最强威胁，但对象不同轴）** | 它分解的是**回归区间宽度**且要**逐 component 重标定**；P13 分解的是**分类预测集大小**且在**单一固定部署阈值**下做 budget-style 归因（无重标定），并加 gold 反事实+更强 ASR 因果验证，是该框架未覆盖的分类×部署×因果三重缺口。|
| **2604.27149** | ConformaDecompose (Yapicioglu et al.) | regression conformal interval reducibility under localized calibration | 否（instance-wise localization，不是 pipeline stage） | 否（解释 calibration localization 下 interval contraction/stability） | 否 | 否 | 否 | **新增近邻—conformal uncertainty decomposition 但非 pipeline/stage** | 它解释局部 calibration support 如何收缩/稳定回归区间；P13 固定阈值下分解真实 ASR→NLU 分类 set-size inflation。|
| **2602.16794** | Beyond Procedure: Substantive Fairness in CP (Liu/Yu/Belbahri/Charpentier/Asgharian/Cresswell, Layer6) | **分类预测集大小 disparity**（Δ_a,b），3-component upper bound | 否（轴=**人群/protected-attribute 组间**，非 pipeline stage） | 否（Mondrian/label-clustered 各组/各簇独立标定阈值） | 否（audio 只是 RAVDESS emotion 单分类模态；非 ASR→NLU 级联） | 否 | 否 | **近邻—需重定位差异化（分解对象同为 set-size，但轴正交）** | 它分解 set-size 的轴是**组间公平 disparity** 且 bound 的三项是 intra-cluster/cross-cluster/cross-group；P13 分解 set-size 的轴是**pipeline stage 归因份额**。两者数学对象同（set size）但**分解轴正交**，P13 若补 set-size 上界理论，引此防重造并明确"我们沿 stage 轴分解，非 group 轴"。|
| **2605.06788** | Conformal Agent Error Attribution (Feng/Sui/Hou/Wu/Cresswell, Dalhousie+Layer6) | **错误定位**：prediction set = 失败 trajectory 中含 decisive-error 的连续 step 区间 | 否（"set"是 trajectory 位置，非 class label；无 set-size 份额分解） | 否（单一阈值，但对象是定位非膨胀分解） | 否（LLM multi-agent trace，非语音） | 否 | 否（用 set 做 rollback 而非 input repair） | **不同目标—轻提即可** | 它用 CP 做**多智能体轨迹的错误步定位**（set 是连续 step 区间），P13 用 CP 做**分类 set-size 的 stage 归因**；二者唯一交集是"CP+attribution"措辞，对象与量纲完全不同。|
| **2605.27091** | MiRD: Reliable Set-Valued Prediction via Miscoverage Risk Decomposition (Hu/Wang/Jia/Fu, UESTC+Beihang) | **miscoverage 概率**分解为 sampling-failure + selection-failure 两 risk（set-size 仅作 downstream utility，明确**不**分解） | 否（"two-stage"是 CP 自身 sample→filter 两步，非真实上游模型→下游模型） | 否（Stage I marginal bound + Stage II 重新标定 selection 阈值） | 否（单 LLM open-ended QA：TriviaQA/CoQA/NQ） | 否 | 否 | **不同目标—轻提即可** | 它分解的是 **miscoverage risk**（覆盖失败概率）成 sampling/selection 两源，且 set-size 明确**不**作分解对象（"not merely uniform inflation"只是定性 utility 论断）；P13 分解的是 **set-size 本身**沿真实模型级联的 stage 份额。对象（risk vs size）与 setting（QA sampling vs 语音级联）双重不同。|

---

## 2. Novelty 总判定

**判定：(A) 存活，related work 扩写即可。**（无任何一篇实质 scoop；6 条区分维度中，没有任何一篇同时命中 ≥2 条与 P13 重合的承重维度。）

依据（逐条回源）：

- **没有任何一篇做"分类 set-size 沿 pipeline-stage 轴的归因份额"。**
  - 2309.12510 / 2510.04406 做 cascaded systems / stage 归因，但对象是**回归区间或残差宽度**，不是 classification set-size inflation。
  - 2604.27149 做 conformal uncertainty explainability，但对象是 localized calibration 下的**回归区间 reducibility**，不是 pipeline stage。
  - 2602.16794 做分类 set-size 分解，但轴是**人群 group disparity**（Thm 4.1: Δ_a,b ≤ intra-cluster label heterogeneity + cross-cluster spread + intra-label cross-group disparity）——非 stage。
  - 2605.06788 / 2605.27091 根本不分解 set size（前者定位 step、后者分解 miscoverage 概率）。
  → P13 的"set-size × per-stage"组合**未被占**。

- **没有任何一篇在"固定部署阈值"下做归因。** 四篇都涉及某种重标定：2510 按 component/sliding-window 重标 + FWER 选 scaling；2602 Mondrian/label-clustered 各组各簇独立阈值；2605.27091 Stage II 重标 selection 阈值。P13 的"single fixed ASR-calibrated deployment threshold 下做 budget-style 归因（无 per-condition recalibration）"是承重 framing，**未被占**。

- **没有任何一篇是真实语音 ASR→NLU 级联。** 最接近的"音频"是 2602 的 RAVDESS 情绪单分类（base model 是 wav2vec/XLSR，仅作特征），不是转写→意图的级联。**未被占**。

- **没有任何一篇有 gold 反事实 + 更强上游的因果验证。** 四篇都无"替换真实中间表示/更强上游模型来归因"的因果对照（2510 用真实 x 算 R2，但不做"换更强 μ̂1"的反事实）。**未被占**。

- **没有任何一篇处理 gold-free input repair 这条线。** 2605.06788 用 set 做 rollback（最接近"修复"），但那是**回退轨迹状态**，非"修上游转写以缩 set"。P13 的"gold-free selective-repair ≈ largest-set-first 平凡基线、word-confidence 仅 +0.135"是独立诚实负结果，**未被占**。

**两个真正的近邻（需在 related work 显式切割，但不威胁 novelty）：**
1. **2510.04406** 是头号近邻：同样主打"stage-wise 残差分解 + CP"。P13 已引；核完确认它低估贴近度但**未 scoop**——对象（回归宽度 vs 分类 set-size）、阈值范式（重标定 vs 固定部署）、验证（无 vs gold 反事实+更强 ASR）三处都岔开。
2. **2602.16794** 是第二近邻：唯一另一篇**分解分类 set-size 且给上界**的论文。若 P13 计划补 set-size 分解的理论/上界，**必须引它防重造**，并声明"我们沿 stage 轴、在固定阈值下分解，区别于其 group-disparity 轴"。

---

## 3. Zotero 引用清单（evidence-bearing，需用户 Zotero 锁）

| # | Title | Authors | Venue | Year | arXiv / DOI | 标注 |
|---|---|---|---|---|---|---|
| 1 | Confidence Calibration for Systems with Cascaded Predictive Modules | Yunye Gong; Yi Yao; Xiao Lin; Ajay Divakaran; Melinda Gervasio | arXiv preprint | 2023 | arXiv:2309.12510 ; DOI 10.48550/arXiv.2309.12510 | evidence-bearing，需 Zotero 锁；已入正文补近邻。 |
| 2 | Decomposition-Based Modular Conformal Prediction for Two-Stage Modeling | William Zhang; Saurabh Amin; Georgia Perakis (MIT Operations Research Center) | arXiv preprint | 2025 | arXiv:2510.04406 ; DOI 10.48550/arXiv.2510.04406 | evidence-bearing，需 Zotero 锁；2026-06-28 arXiv API 显示当前 v2 标题如左。 |
| 3 | ConformaDecompose: Explaining Uncertainty via Calibration Localization | Fatima Rabia Yapicioglu; Meltem Aksoy; Alberto Rigenti; Tuwe Löfström-Cavallin; Helena Löfström-Cavallin; Seyda Yoncaci; Luca Longo | arXiv accepted author version / World Explainable AI Conference (per arXiv comment) | 2026 | arXiv:2604.27149 ; DOI 10.48550/arXiv.2604.27149 | evidence-bearing，需最终 Springer/Zotero 锁；已入正文补 regression uncertainty-decomposition 边界。 |
| 4 | Beyond Procedure: Substantive Fairness in Conformal Prediction | Pengqi Liu; Zijun Yu; Mouloud Belbahri; Arthur Charpentier; Masoud Asgharian; Jesse C. Cresswell (Layer 6 AI 等) | arXiv preprint | 2026 | arXiv:2602.16794 ; DOI 10.48550/arXiv.2602.16794 | evidence-bearing，需 Zotero 锁。code: github.com/layer6ai-labs/llm-in-the-loop-conformal-fairness |
| 5 | Conformal Agent Error Attribution | Naihe Feng; Yi Sui; Shiyi Hou; Ga Wu; Jesse C. Cresswell (Dalhousie Univ. + Layer 6 AI) | arXiv preprint | 2026 | arXiv:2605.06788 ; DOI 10.48550/arXiv.2605.06788 | evidence-bearing，需 Zotero 锁。code: github.com/layer6ai-labs/conformal-agent-error-attribution |
| 6 | MiRD: Reliable Set-Valued Prediction for Open-Ended Question Answering via Miscoverage Risk Decomposition | Anqi Hu; Zhiyuan Wang; Zijun Jia; Bo Fu (UESTC + Beihang) | arXiv preprint | 2026 | arXiv:2605.27091 ; DOI 10.48550/arXiv.2605.27091 | evidence-bearing，需 Zotero 锁 |

**新增近邻**：本轮读完 4 篇各自 Related Work + 全部 reference list，**未发现比这 4 篇更贴近 P13 的论文**（无 per-stage classification-set 归因 / 固定部署阈值下 set-size 分解 / ASR error→conformal set 传播 的邻居）。四篇之间唯一与"cascade"沾边的是 P4 引的 *BalanceRAG: joint risk calibration for cascaded RAG*（cascade 但 RAG 非语音、且非 set-size per-stage 分解）——不构成威胁，列为可选 background。

---

## 4. 是否有贴脸威胁 novelty 的论文？

**无。** 没有任何一篇贴脸到威胁 P13 novelty。最需警惕的两篇均用原文证据句切割如下：

- **2510.04406（头号近邻）** 原文证据：
  - 对象=回归残差："R(w,y)=|y−μ̂2(μ̂1(w))| into upstream and downstream components"（§3）、"R2(x,y)=|y−μ̂2(x)|"（Def 1）。→ 是**绝对值回归残差/区间宽度**，非分类预测集大小。
  - 阈值范式=重标定+FWER 选 scaling，且 adaptive 版用 sliding window 重标："update scaling parameters based on component-wise empirical coverage"（§6）。→ 非固定部署阈值。
  - 无 gold/更强上游因果对照。
  → 同为"stage-wise decomposition + CP"但三处岔开，**不 scoop**。

- **2602.16794（第二近邻）** 原文证据：
  - 对象=分类 set-size，但轴=group disparity："Δ_a,b = |E[|C(X)| | A=a] − E[|C(X)| | A=b]|"（Eq 9）、上界三项 "(I) Intra-cluster label heterogeneity + (II) Cross-cluster spread + (III) Intra-label cross-group disparity"（Thm 4.1）。
  → set-size 分解的**轴是人群组间**，与 P13 的 **stage 轴正交**，**不 scoop**；但 P13 补 set-size 理论时须引它防重造。
