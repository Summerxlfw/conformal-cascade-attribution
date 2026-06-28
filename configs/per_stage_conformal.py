#!/usr/bin/env python3
"""完整 per-stage conformal 方法 + 覆盖保证(真数据 SLURP cascade, Neurocomputing path A 步骤1).
载体: audio -> [S1 ASR Whisper small.en] -> text -> MiniLM -> [S2 NLU LR] -> split-conformal 意图集合.
四部分(每部分都出可验收数, 数字 ground truth = features npz / logprob parquet):
  P1 部署 split-conformal 预测器 + 覆盖(标准可证 1-alpha, 经验验)
  P2 精确 per-stage 分解 I = I_ASR + I_NLU(2阶段 telescoping 恒等, 无交互 gap; 残差应=0)+ bootstrap CI
  P3 no-gold 每实例 ASR-归因估计保真(ASR-internal 置信; partial ρ vs 平凡 sz_asr 基线)
  P4 覆盖保持的选择性修复策略(预算 b 修 top-b; 验覆盖不掉 + 报"修复反踢真标签" slack; 可部署 no-gold vs 平凡 vs oracle)
诚实边界: 覆盖保证是"条件 + 经验验证"(修复非增 nonconformity 时守覆盖), 非无条件定理; slack 率如实报.
"""
import os, sys, numpy as np, pandas as pd, scipy.stats as st
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import LabelEncoder, StandardScaler
from scipy.stats import spearmanr
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from conformal_primitives import split_cp_calibrate_multiclass, split_cp_predict_set_multiclass

ALPHA=0.10
HERE=os.path.dirname(os.path.abspath(__file__)); P=os.path.join(HERE,"..","data","slurp_real")
base=pd.read_parquet(f"{P}/slurp_asr.parquet")
conf_df=pd.read_parquet(f"{P}/slurp_asr_logprob.parquet").set_index("id")
CONF=["asr_avg_logprob","asr_min_word_logprob","asr_word_logprob_std"]
id2lab=dict(zip(base.id,base.intent_scenario)); le=LabelEncoder().fit(base.intent_scenario)
def load(s):
    z=np.load(f"{P}/features_{s}.npz",allow_pickle=True); ids=z["ids"]
    return z["gold_emb"],z["asr_emb"],le.transform([id2lab[i] for i in ids]),ids
Gtr,Atr,ytr,_=load("train"); Gva,Ava,yva,ids_va=load("devel"); Gte,Ate,yte,ids_te=load("test")
clf=LogisticRegression(max_iter=1000,C=10).fit(Gtr,ytr)
LOG=lambda Pm: np.log(np.clip(Pm,1e-12,1.0))      # primitives 吃 logits, softmax(log p)=p
Pva_a=clf.predict_proba(Ava); Pte_a=clf.predict_proba(Ate); Pte_g=clf.predict_proba(Gte); Pva_g=clf.predict_proba(Gva)

# ===== P1 部署 split-conformal + 覆盖 =====
t=split_cp_calibrate_multiclass(LOG(Pva_a),yva,ALPHA)      # 校准在 devel-ASR(部署口径)
C_asr,sz_asr=split_cp_predict_set_multiclass(LOG(Pte_a),t)
C_gold,sz_gold=split_cp_predict_set_multiclass(LOG(Pte_g),t)
cov_asr=C_asr[np.arange(len(yte)),yte].mean(); cov_gold=C_gold[np.arange(len(yte)),yte].mean()
sz_asr=sz_asr.astype(float); sz_gold=sz_gold.astype(float)
print("="*72)
print(f"P1 部署 split-conformal (LAC, alpha={ALPHA}, 校准 devel-ASR n={len(yva)}, t={t:.4f})")
print(f"  ASR-test 覆盖={cov_asr:.4f} (目标>={1-ALPHA:.2f})  {'守' if cov_asr>=1-ALPHA-0.01 else '破!'} | mean set-size={sz_asr.mean():.3f}")
print(f"  GOLD-test 覆盖={cov_gold:.4f} (固定同 t 反事实) | set-size={sz_gold.mean():.3f}  -> 修复单调升覆盖(P4 保证依据)")

# ===== P2 精确 per-stage 分解 =====
delta=sz_asr-sz_gold
I_total=sz_asr.mean()-1.0; I_ASR=delta.mean(); I_NLU=sz_gold.mean()-1.0
resid=I_total-I_ASR-I_NLU
def boot(fn,B=3000):
    n=len(yte); return np.percentile([fn(np.random.default_rng(b).integers(0,n,n)) for b in range(B)],[2.5,97.5])
ci_a=boot(lambda i:(sz_asr[i]-sz_gold[i]).mean()); ci_n=boot(lambda i:sz_gold[i].mean()-1.0)
print("="*72)
print("P2 精确 per-stage 分解 (I = 相对理想单点 1.0 的集合膨胀)")
print(f"  I_total={I_total:+.3f} | I_ASR(S1)={I_ASR:+.3f} CI[{ci_a[0]:+.3f},{ci_a[1]:+.3f}] | I_NLU(S2)={I_NLU:+.3f} CI[{ci_n[0]:+.3f},{ci_n[1]:+.3f}]")
print(f"  残差={resid:+.2e} (2阶段 telescoping 恒等, 应=0, 无交互 gap) | ASR 占 {100*I_ASR/I_total:.1f}% NLU 占 {100*I_NLU/I_total:.1f}%")
# canonical 分解 summary CSV(供证据图 F2 与 ledger 单一数源, 3 位小数)
r3=lambda x: float(f"{x:.3f}")
pd.DataFrame([{"stage":"ASR (S1)","inflation":r3(I_ASR),"ci_lo":r3(ci_a[0]),"ci_hi":r3(ci_a[1]),"pct":r3(100*I_ASR/I_total)},
              {"stage":"NLU-intrinsic (S2)","inflation":r3(I_NLU),"ci_lo":r3(ci_n[0]),"ci_hi":r3(ci_n[1]),"pct":r3(100*I_NLU/I_total)}]
            ).to_csv(os.path.join(HERE,"decomposition_summary.csv"),index=False)
pd.DataFrame([{"metric":"coverage_asr","value":r3(cov_asr)},{"metric":"coverage_gold","value":r3(cov_gold)},
              {"metric":"I_total","value":r3(I_total)},{"metric":"residual","value":float(f"{resid:.1e}")}]
            ).to_csv(os.path.join(HERE,"decomposition_meta.csv"),index=False)

# ===== P3 no-gold 每实例 ASR-归因估计保真 =====
Cte=conf_df.loc[ids_te,CONF].values; Cva=conf_df.loc[ids_va,CONF].values
word_std=Cte[:,CONF.index("asr_word_logprob_std")]; conf_nlu=1-Pte_a.max(1)
def prho(x,y,z):
    rx,ry,rz=st.rankdata(x),st.rankdata(y),st.rankdata(z)
    return spearmanr(rx-np.polyval(np.polyfit(rz,rx,1),rz),ry-np.polyval(np.polyfit(rz,ry,1),rz)).correlation
print("="*72)
print("P3 no-gold 每实例 ASR-归因估计保真 (控制平凡基线 sz_asr 后的增量 partial ρ vs 真Δ)")
print(f"  word_lp_std partial={prho(word_std,delta,sz_asr):+.3f} | conf_nlu(下游) partial={prho(conf_nlu,delta,sz_asr):+.3f} | sz_asr 自身 inst_ρ={spearmanr(sz_asr,delta).correlation:+.3f}(平凡最强)")

# ===== P4 覆盖保持的选择性修复策略 =====
# 可部署 no-gold 复合: devel(gold可得)训 [sz_asr_devel, ASR-internal] -> devel delta, 预测 test 修复收益, test 不碰 gold
Cva_a=split_cp_predict_set_multiclass(LOG(Pva_a),t)[1].astype(float)
Cva_g=split_cp_predict_set_multiclass(LOG(Pva_g),t)[1].astype(float)
dva=Cva_a-Cva_g
Fva=np.column_stack([Cva_a,Cva,(1-Pva_a.max(1))]); Fte=np.column_stack([sz_asr,Cte,conf_nlu])
sc=StandardScaler().fit(Fva); comb=Ridge(alpha=1.0).fit(sc.transform(Fva),dva)
score_comb=comb.predict(sc.transform(Fte))                 # 可部署方法分(无 test gold)
policies={"随机":np.random.default_rng(0).random(len(yte)),"conf_nlu(下游)":conf_nlu,
          "sz_asr(平凡强基线)":sz_asr,"word_lp_std(单ASR置信)":word_std,
          "复合(可部署方法)":score_comb,"oracle(真Δ)":delta}
def repair(score,b):
    k=int(b*len(yte)); sel=np.argsort(-score)[:k]
    Pmix=Pte_a.copy(); Pmix[sel]=Pte_g[sel]
    Cs,szs=split_cp_predict_set_multiclass(LOG(Pmix),t)
    cov=Cs[np.arange(len(yte)),yte].mean()
    was=C_asr[sel,yte[sel]]; now=C_gold[sel,yte[sel]]
    slack=(was & ~now).mean() if k>0 else 0.0   # 修复反把真标签踢出 (条件保证的 slack)
    return cov, szs.mean(), slack, k
print("="*72)
print("P4 覆盖保持的选择性修复 (修 top-b 用 gold 反事实=完美 ASR; 固定 t)")
print(f"{'策略':<20}", end="")
for b in [0.10,0.25,0.50]: print(f"  b={b:.2f}[覆盖/效率↓/slack]", end="")
print()
rows=[]
for nm,s in policies.items():
    line=f"{nm:<20}"
    for b in [0.10,0.25,0.50]:
        cov,sz,slack,k=repair(s,b); red=I_total-(sz-1.0)
        line+=f"  {cov:.3f}/{red:+.3f}/{slack:.3f}"
        rows.append((nm,b,cov,sz,red,slack))
    print(line)
rdf=pd.DataFrame(rows,columns=["policy","budget","coverage","setsize","ineff_reduction","slack"])
# 关键: 可部署复合 在 b=0.25 是否打败平凡 sz_asr? bootstrap CI on reduction diff
def red_at(score,b,idx):
    k=int(b*len(idx)); sel=idx[np.argsort(-score[idx])[:k]]
    Pmix=Pte_a.copy(); Pmix[sel]=Pte_g[sel]
    szs=split_cp_predict_set_multiclass(LOG(Pmix),t)[1]
    return (sz_asr.mean()-1.0)-(szs.mean()-1.0)   # 用全体基线近似
n=len(yte); B=2000; diffs=[]
for bb in range(B):
    idx=np.random.default_rng(bb).integers(0,n,n)
    diffs.append(red_at(score_comb,0.25,idx)-red_at(sz_asr,0.25,idx))
diffs=np.array(diffs)
print("="*72)
print(f"复合(可部署) − sz_asr(平凡) 的效率回收差 @b=0.25: {diffs.mean():+.4f} 95%CI[{np.percentile(diffs,2.5):+.4f},{np.percentile(diffs,97.5):+.4f}] P(复合更好)={(diffs>0).mean():.2f}")
# 持久化该 bootstrap 标量为一等公民产物(供 ledger/正文 +0.001 数追溯; 注意=within-resample 配对估计量, ≠ 全集点差 0.003)
r4=lambda x: float(f"{x:.4f}")
pd.DataFrame([{"metric":"combined_minus_sz_asr_reduction_diff","budget":0.25,
               "mean":r4(diffs.mean()),"ci_lo":r4(np.percentile(diffs,2.5)),"ci_hi":r4(np.percentile(diffs,97.5)),
               "p_combined_better":round(float((diffs>0).mean()),2),"n_bootstrap":B,
               "note":"within-resample paired estimator; full-set point diff at b=0.25 = +0.003"}]
            ).to_csv(os.path.join(HERE,"repair_diff_bootstrap.csv"),index=False)
print("落盘 repair_diff_bootstrap.csv (combined−sz_asr 配对 bootstrap 标量)")
cov_min=rdf.coverage.min()
print(f"\n验收: 全策略全预算最低覆盖={cov_min:.3f} ({'守 1-alpha' if cov_min>=1-ALPHA-0.01 else '破!'}); slack(修复反踢真标签)≈{rdf.slack.max():.3f} 上限 = 条件保证的诚实边界")
rdf.to_csv(os.path.join(HERE,"method_results.csv"),index=False)
print("落盘 method/method_results.csv")
