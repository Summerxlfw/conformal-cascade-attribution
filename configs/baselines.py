#!/usr/bin/env python3
"""Step2 baseline 对比(reframe: 比'各方法提供什么/覆盖效率', 非比 targeting).
诚实 baseline 集:
  B1 Vanilla split CP(我们部署的, 标准对照)
  B2 Mondrian 类条件 CP(per-scenario 阈值, 标准对照; per-class 非 per-stage)
  B3 联合覆盖三法(independent / Bonferroni / PASC joint-max) — stage1=ASR 置信(简化二元适配, 完整 n-best 版留服务器)
对照我们: 唯一做 per-stage 效率分解(精确)+ 验归因 + 覆盖保持修复.
"""
import os, sys, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from conformal_primitives import (split_cp_calibrate_multiclass as cal, split_cp_predict_set_multiclass as pred,
                                   mondrian_split_cp_calibrate as mcal, mondrian_split_cp_predict as mpred,
                                   compute_coverage_by_group)
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
ALPHA=0.10; HERE=os.path.dirname(os.path.abspath(__file__)); P=os.path.join(HERE,"..","data","slurp_real")
base=pd.read_parquet(f"{P}/slurp_asr.parquet"); conf=pd.read_parquet(f"{P}/slurp_asr_logprob.parquet").set_index("id")
id2lab=dict(zip(base.id,base.intent_scenario)); id2wer=dict(zip(base.id,base.wer)); le=LabelEncoder().fit(base.intent_scenario)
def load(s):
    z=np.load(f"{P}/features_{s}.npz",allow_pickle=True); ids=z["ids"]
    return z["gold_emb"],z["asr_emb"],le.transform([id2lab[i] for i in ids]),ids
Gtr,Atr,ytr,_=load("train"); Gva,Ava,yva,ids_va=load("devel"); Gte,Ate,yte,ids_te=load("test")
clf=LogisticRegression(max_iter=1000,C=10).fit(Gtr,ytr); LOG=lambda M:np.log(np.clip(M,1e-12,1))
Pva=clf.predict_proba(Ava); Pte=clf.predict_proba(Ate)
gva=np.array([id2lab[i] for i in ids_va]); gte=np.array([id2lab[i] for i in ids_te])
cov=lambda C: C[np.arange(len(yte)),yte].mean()

print("="*78); print("B1 Vanilla split CP (LAC, 部署=ASR 校准)")
t=cal(LOG(Pva),yva,ALPHA); C1,sz1=pred(LOG(Pte),t)
print(f"  intent 覆盖={cov(C1):.3f} | set-size={sz1.mean():.3f} | 提供: 边际覆盖. 无 per-stage 归因/修复")

print("="*78); print("B2 Mondrian 类条件 CP (per-scenario=真类 阈值, oracle-group 诊断基线)")
q=mcal(LOG(Pva),yva,yva,ALPHA); C2,sz2=mpred(LOG(Pte),yte,q)
cg=compute_coverage_by_group(C2,yte,yte); cgv=np.array(list(cg.values()))
print(f"  intent 覆盖={cov(C2):.3f} | set-size={sz2.mean():.3f} | per-group 覆盖 spread=[{cgv.min():.3f},{cgv.max():.3f}]")
print(f"  提供: per-CLASS 条件覆盖(非 per-stage). 无 ASR 阶段归因/修复")

print("="*78); print("B3 联合覆盖三法 (stage1=ASR 置信 简化二元适配; stage2=NLU; 完整 n-best 版留服务器)")
# stage2 score = NLU nonconformity of true intent; stage1 score = ASR 不确定(1-exp(avg_logprob)), 标签=ASR 正确(WER=0)
s2_va=1-Pva[np.arange(len(yva)),yva]; s2_te=1-Pte[np.arange(len(yte)),yte]
c1_va=1-np.exp(conf.loc[ids_va,"asr_avg_logprob"].values); c1_te=1-np.exp(conf.loc[ids_te,"asr_avg_logprob"].values)
asr_ok_va=np.array([id2wer[i] for i in ids_va])==0; asr_ok_te=np.array([id2wer[i] for i in ids_te])==0
def q_(s,a): n=len(s); return np.quantile(s,min(np.ceil((n+1)*(1-a))/n,1.0),method="higher")
# 联合覆盖(both stage 真标签被覆盖) = (s2<=q2) & (s1<=q1 if asr_ok else 该阶段未覆盖)
def joint_cov(q1,q2):
    stage2=s2_te<=q2; stage1=(c1_te<=q1)|(~asr_ok_te)  # asr 错时无法"覆盖正确转写"; 简化记 stage1 覆盖=置信内或本就该拒
    return (stage2 & (c1_te<=q1)).mean(), stage2.mean()  # 严格联合: 两 score 都进
# independent: 各 1-alpha
qi1=q_(c1_va,ALPHA); qi2=q_(s2_va,ALPHA); jc_i,m_i=joint_cov(qi1,qi2)
# Bonferroni: 各 1-alpha/2
qb1=q_(c1_va,ALPHA/2); qb2=q_(s2_va,ALPHA/2); jc_b,m_b=joint_cov(qb1,qb2)
# PASC joint-max: max(s1,s2) 的 1-alpha 分位
smax_va=np.maximum(c1_va,s2_va); qj=q_(smax_va,ALPHA)
stage2_p=s2_te<=qj; stage1_p=c1_te<=qj; jc_p=(stage2_p&stage1_p).mean()
# 各法下 intent set size(stage2 阈值决定)
szi=((1-Pte)<=qi2).sum(1).mean(); szb=((1-Pte)<=qb2).sum(1).mean(); szp=((1-Pte)<=qj).sum(1).mean()
print(f"  {'方法':<16}{'联合覆盖(asr+intent)':>20}{'intent set-size':>16}")
print(f"  {'independent':<16}{jc_i:>20.3f}{szi:>16.3f}  (无联合保证, 联合<目标)")
print(f"  {'Bonferroni':<16}{jc_b:>20.3f}{szb:>16.3f}  (联合保证但保守, set 大)")
print(f"  {'PASC joint-max':<16}{jc_p:>20.3f}{szp:>16.3f}  (联合保证更紧)")
print(f"  提供: 联合(转写+意图)覆盖. 不归因不修复. 目标与我们(最终意图归因)不同")

print("="*78); print("对照表(各方法提供什么)")
print(f"  {'方法':<26}{'intent cov':>11}{'set-size':>10}  提供")
print(f"  {'B1 Vanilla split CP':<26}{cov(C1):>11.3f}{sz1.mean():>10.3f}  边际覆盖")
print(f"  {'B2 Mondrian 类条件':<24}{cov(C2):>11.3f}{sz2.mean():>10.3f}  per-class 条件覆盖")
print(f"  {'B3 PASC joint-max':<25}{stage2_p.mean():>11.3f}{szp:>10.3f}  联合(转写+意图)覆盖")
print(f"  {'★Ours per-stage+repair':<25}{cov(C1):>11.3f}{sz1.mean():>10.3f}  +精确per-stage分解+验归因+覆盖保持修复")
print("\n结论: 标准/邻居 baseline 都不做'分类集合的 per-stage 效率归因+验证+无gold修复'. 我们占该点(novelty 真空, sourced).")
pd.DataFrame([["B1_vanilla",cov(C1),sz1.mean()],["B2_mondrian",cov(C2),sz2.mean()],
              ["B3_independent",jc_i,szi],["B3_bonferroni",jc_b,szb],["B3_pasc_jointmax",jc_p,szp]],
             columns=["method","coverage_or_joint","setsize"]).to_csv(os.path.join(HERE,"baselines_results.csv"),index=False)
print("落盘 method/baselines_results.csv")
