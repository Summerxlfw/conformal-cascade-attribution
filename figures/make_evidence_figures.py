#!/usr/bin/env python3
"""P13 证据图: 从真实 CSV(results/)渲染矢量 SVG+PDF. 每图同时产出 rounded summary CSV
(figure_outputs/<fig>_data.csv), 其值即图上所绘, 供 figure_text_consistency 数字 string-match.
禁位图/禁 AI 值: 所有点 trace 到 results/ 的真实 CSV(per-instance 重算或汇总行).
"""
import os, numpy as np, pandas as pd, scipy.stats as st
from scipy.stats import spearmanr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"pdf.fonttype":42,"ps.fonttype":42,"svg.fonttype":"none",
                     "font.size":9,"font.family":"sans-serif","axes.spines.top":False,"axes.spines.right":False})
HERE=os.path.dirname(os.path.abspath(__file__))
R=os.path.join(HERE,"..","results")
OUT=os.path.join(HERE,"figure_outputs")
os.makedirs(OUT,exist_ok=True)
def save(fig,name):
    fig.savefig(os.path.join(OUT,name+".svg"),bbox_inches="tight",pad_inches=0.05)
    fig.savefig(os.path.join(OUT,name+".pdf"),bbox_inches="tight",pad_inches=0.05); plt.close(fig)
r3=lambda x: float(f"{x:.3f}")

# ---------- F2: per-stage 分解 + CI(读 canonical decomposition_summary.csv, 单一数源)----------
e=pd.read_csv(os.path.join(R,"slurpE_decisive.csv")); delta=e["delta"].values; sz_asr=e["sz_asr"].values  # 供 F3 用
f2=pd.read_csv(os.path.join(R,"decomposition_summary.csv"))
f2.to_csv(os.path.join(OUT,"F2_decomposition_data.csv"),index=False)
fig,ax=plt.subplots(figsize=(3.4,2.6))
xs=[0,1]; vals=f2["inflation"].values
err=[vals-f2["ci_lo"].values, f2["ci_hi"].values-vals]
ax.bar(xs,vals,yerr=err,capsize=4,color=["#DD8452","#3E5266"],width=0.6)  # ASR=amber, NLU=slate (与 F1/F4 跨图统一)
for x,v,p in zip(xs,vals,f2["pct"].values): ax.text(x,v+0.03,f"{v:.3f}\n({p:.0f}%)",ha="center",fontsize=8)
ax.set_xticks(xs); ax.set_xticklabels(f2["stage"]); ax.set_ylabel("set-size inflation  (mean |C| - 1)")
ax.set_ylim(0,0.42); ax.set_title("Per-stage inflation (fixed deployment threshold)",fontsize=8.5)
save(fig,"F2_decomposition")

# ---------- F3: (a) partial rho(signal,Δ|sz_asr); (b) repair efficiency vs budget ----------
def prho(x,y,z):
    rx,ry,rz=st.rankdata(x),st.rankdata(y),st.rankdata(z)
    return spearmanr(rx-np.polyval(np.polyfit(rz,rx,1),rz),ry-np.polyval(np.polyfit(rz,ry,1),rz)).correlation
rho_boot=pd.read_csv(os.path.join(R,"partial_rho_bootstrap.csv")).set_index("signal")
sig_order=[
    ("word_lp_std","word_lp_std\n(gold-free)"),
    ("min_word_lp","min_word_lp\n(gold-free)"),
    ("conf_nlu_downstream","NLU-nonconf\n(gold-free)"),
    ("wer_gold_reference","WER\n(needs gold)"),
    ("random","random"),
]
f3a=pd.DataFrame([
    {"signal":label.replace("\n"," "),
     "partial_rho":float(rho_boot.loc[key,"partial_rho"]),
     "ci_lo":float(rho_boot.loc[key,"ci_lo"]),
     "ci_hi":float(rho_boot.loc[key,"ci_hi"])}
    for key,label in sig_order
])
f3a.to_csv(os.path.join(OUT,"F3_partial_rho_data.csv"),index=False)
m=pd.read_csv(os.path.join(R,"method_results.csv"))
polmap={"sz_asr(平凡强基线)":"sz_asr (trivial)","复合(可部署方法)":"combined (deployable)",
        "word_lp_std(单ASR置信)":"word_lp_std","oracle(真Δ)":"oracle","随机":"random"}
f3b=m[m["policy"].isin(polmap)].copy(); f3b["policy_en"]=f3b["policy"].map(polmap)
f3b["budget"]=f3b["budget"].round(2); f3b["ineff_reduction"]=f3b["ineff_reduction"].round(3)
f3b[["policy_en","budget","ineff_reduction"]].to_csv(os.path.join(OUT,"F3_repair_data.csv"),index=False)
repair_ci=pd.read_csv(os.path.join(R,"repair_diff_bootstrap.csv")).iloc[0]
fig,(a1,a2)=plt.subplots(1,2,figsize=(6.6,2.7))
cols=["#4C72B0","#55A868","#8172B3","#C44E52","#999999"]
prho_vals=f3a["partial_rho"].values
prho_yerr=np.vstack([prho_vals-f3a["ci_lo"].values, f3a["ci_hi"].values-prho_vals])
bars=a1.bar(range(len(sig_order)),prho_vals,yerr=prho_yerr,capsize=3,color=cols,width=0.62)
bars[3].set_hatch("///")                                   # WER 需 gold, 非可部署: 标记为天花板, 防误读为最佳可部署信号
a1.axvline(2.5,ls=":",lw=0.8,color="#888888")              # 分隔: 左=可部署 gold-free 候选; 右=需-gold/随机 参照
a1.annotate("uses gold\n(ceiling)",xy=(3,prho_vals[3]),xytext=(3,prho_vals[3]+0.05),
            ha="center",fontsize=6,color="#C44E52")
a1.axhline(0.15,ls="--",lw=1,color="k"); a1.text(1.2,0.16,"pre-reg +0.15",fontsize=7)
a1.axhline(0,lw=0.8,color="k"); a1.set_xticks(range(len(sig_order))); a1.set_xticklabels(["word_lp_std","min_word_lp","NLU-nonconf","WER","random"],fontsize=6.5,rotation=20,ha="right")  # 旋转去拥挤;gold-free/gold 区分由 x=2.5 分隔线+WER"uses gold"标注承载
a1.set_ylim(-0.22,0.31)
a1.set_ylabel(r"partial $\rho(\mathrm{signal},\Delta\,|\,\mathrm{sz\_asr})$"); a1.set_title("(a) Gold-free attribution vs trivial baseline (WER = gold ceiling)",fontsize=7.5)
markers={"oracle":"o","sz_asr (trivial)":"s","combined (deployable)":"^","word_lp_std":"D","random":"x"}
for pol in ["oracle","sz_asr (trivial)","combined (deployable)","word_lp_std","random"]:
    d=f3b[f3b["policy_en"]==pol].sort_values("budget")
    a2.plot(d["budget"],d["ineff_reduction"],marker=markers[pol],ms=3,label=pol)
a2.set_ylim(0,0.24)                                        # 纵轴从 0 起: 诚实呈现量级, 不夸大 combined/sz_asr 与 oracle 的差
a2.annotate(f"paired Δ {repair_ci['mean']:+.3f}\n95% CI [{repair_ci['ci_lo']:+.3f},{repair_ci['ci_hi']:+.3f}]",
            xy=(0.25,0.182),xytext=(0.31,0.155),
            arrowprops=dict(arrowstyle="->",lw=0.8,color="#555555"),
            fontsize=6.2,color="#333333")
a2.set_xlabel("repair budget (fraction)"); a2.set_ylabel("inefficiency recovered"); a2.set_title("(b) Selective-repair efficiency",fontsize=8)
a2.legend(fontsize=6.5,frameon=False)
save(fig,"F3_partial_rho_repair")

# ---------- F4: generality small vs medium ----------
g=pd.read_csv(os.path.join(R,"slurpF_generality.csv"))
f4=g[["tag","wer","I_ASR","I_NLU","cov","part"]].copy()
for c in ["wer","I_ASR","I_NLU","cov","part"]: f4[c]=f4[c].round(3)
f4.to_csv(os.path.join(OUT,"F4_generality_data.csv"),index=False)
fig,ax=plt.subplots(figsize=(3.6,2.7)); x=np.arange(2); w=0.35
ax.bar(x-w/2,f4["I_ASR"],w,label="I_ASR (ASR-attributable)",color="#DD8452")  # ASR=amber (跨图统一)
ax.bar(x+w/2,f4["I_NLU"],w,label="I_NLU (NLU-intrinsic)",color="#3E5266")  # NLU=slate (跨图统一)
for i,ia in enumerate(f4["I_ASR"]): ax.text(i-w/2,ia+0.008,f"{ia:.3f}",ha="center",fontsize=7)
for i,inu in enumerate(f4["I_NLU"]): ax.text(i+w/2,inu+0.008,f"{inu:.3f}",ha="center",fontsize=7)  # 四 bar 全标(I_NLU 的 0.307->0.201 也是论断)
ax.set_xticks(x); ax.set_xticklabels([f"{t}\nWER {w:.3f}" for t,w in zip(f4["tag"],f4["wer"])],fontsize=7.5)
ax.set_ylim(0,0.36); ax.set_ylabel("set-size inflation")
ax.set_title("Stronger recognizer shrinks ASR-attributable share\n(two recognizers, single corpus SLURP — not cross-dataset)",fontsize=7); ax.legend(fontsize=6.5,frameon=False)
save(fig,"F4_generality")

print("生成: F2_decomposition / F3_partial_rho_repair / F4_generality (.svg+.pdf) + *_data.csv")
print("F2 (canonical):"); print(f2.to_string(index=False))
print("F3a partial:"); print(f3a.to_string(index=False))
