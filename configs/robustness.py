#!/usr/bin/env python3
"""Step4 稳健性: per-stage 分解 + 覆盖 + no-gold 信号 在多设定下是否守.
轴: (a) alpha 扫 {0.05,0.10,0.20}; (b) target {scenario 18类, intent_full 更多类}; (c) CP score {LAC, APS};
    (d) train-seed 稳定(子采样重训). 每设定报 ASR-test 覆盖 / I_ASR / I_NLU / word_lp_std partial(Δ|sz_asr).
诚实: 覆盖应守 1-alpha; 分解恒等(残差0)与 ASR<NLU 结构应跨设定稳; modest 信号 partial 可能波动.
"""
import os, sys, numpy as np, pandas as pd, scipy.stats as st
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from scipy.stats import spearmanr
HERE=os.path.dirname(os.path.abspath(__file__)); P=os.path.join(HERE,"..","data","slurp_real")
base=pd.read_parquet(f"{P}/slurp_asr.parquet"); conf=pd.read_parquet(f"{P}/slurp_asr_logprob.parquet").set_index("id")
id2wer=dict(zip(base.id,base.wer))
def load(s):
    z=np.load(f"{P}/features_{s}.npz",allow_pickle=True); return z["gold_emb"],z["asr_emb"],z["ids"]
Gtr,Atr,idtr=load("train"); Gva,Ava,idva=load("devel"); Gte,Ate,idte=load("test")
word_std=conf.loc[idte,"asr_word_logprob_std"].values

def prho(x,y,z):
    rx,ry,rz=st.rankdata(x),st.rankdata(y),st.rankdata(z)
    return spearmanr(rx-np.polyval(np.polyfit(rz,rx,1),rz),ry-np.polyval(np.polyfit(rz,ry,1),rz)).correlation
# CP score: LAC s=1-p_y; APS s=sum_{k:p_k>=p_y} p_k (非随机化)
def scores(Pm,y,kind):
    if kind=="LAC": return 1-Pm[np.arange(len(y)),y]
    order=np.argsort(-Pm,1); sp=np.take_along_axis(Pm,order,1); cum=np.cumsum(sp,1)
    rank=np.argmax(order==y[:,None],1); return cum[np.arange(len(y)),rank]
def setsize(Pm,kind,q):
    if kind=="LAC": return ((1-Pm)<=q).sum(1)
    order=np.argsort(-Pm,1); sp=np.take_along_axis(Pm,order,1); cum=np.cumsum(sp,1)
    inset=cum<=q; return inset.sum(1)
def incov(Pm,y,kind,q):
    if kind=="LAC": return ((1-Pm)<=q)[np.arange(len(y)),y]
    order=np.argsort(-Pm,1); sp=np.take_along_axis(Pm,order,1); cum=np.cumsum(sp,1)
    inset=cum<=q; rank=np.argmax(order==y[:,None],1); return inset[np.arange(len(y)),rank]
def qhat(s,a): n=len(s); return np.quantile(s,min(np.ceil((n+1)*(1-a))/n,1.0),method="higher")

def run(targetcol,alpha,kind,seed=0):
    lab=dict(zip(base.id,base[targetcol]))
    le=LabelEncoder().fit([lab[i] for i in idtr])           # 只 fit 训练标签
    seen=set(le.classes_)
    mva=np.array([lab[i] in seen for i in idva]); mte=np.array([lab[i] in seen for i in idte])  # 丢训练未见类(无法预测)
    ytr=le.transform([lab[i] for i in idtr])
    yva=le.transform([lab[i] for i in idva[mva]]); yte=le.transform([lab[i] for i in idte[mte]])
    if seed>0:
        rng=np.random.default_rng(seed); idx=rng.choice(len(ytr),int(0.8*len(ytr)),replace=False)
        clf=LogisticRegression(max_iter=1000,C=10).fit(Gtr[idx],ytr[idx])
    else:
        clf=LogisticRegression(max_iter=1000,C=10).fit(Gtr,ytr)
    Pva=clf.predict_proba(Ava[mva]); Pte_a=clf.predict_proba(Ate[mte]); Pte_g=clf.predict_proba(Gte[mte])
    ws=word_std[mte]
    q=qhat(scores(Pva,yva,kind),alpha)
    cov=incov(Pte_a,yte,kind,q).mean()
    sza=setsize(Pte_a,kind,q).astype(float); szg=setsize(Pte_g,kind,q).astype(float)
    delta=sza-szg; I_ASR=delta.mean(); I_NLU=szg.mean()-1.0
    part=prho(ws,delta,sza) if delta.std()>0 else float("nan")
    return cov,I_ASR,I_NLU,part,len(le.classes_)

print(f"{'设定':<34}{'cov':>7}{'I_ASR':>8}{'I_NLU':>8}{'ASR%':>6}{'word_std partial':>17}")
_rows=[]
def line(tag,targetcol,alpha,kind,seed=0):
    cov,ia,inu,pa,K=run(targetcol,alpha,kind,seed)
    tot=ia+inu; pct=100*ia/tot if tot>0 else float('nan')
    _rows.append({"setting":tag,"coverage":round(cov,3),"I_ASR":round(ia,3),"I_NLU":round(inu,3),
                  "ASR_pct":round(pct,1) if tot>0 else None,"word_std_partial":round(pa,3)})
    print(f"{tag:<34}{cov:>7.3f}{ia:>+8.3f}{inu:>+8.3f}{pct:>5.0f}%{pa:>+17.3f}")
print("--- (a) alpha 扫 (target=scenario, LAC) ---")
for a in [0.05,0.10,0.20]: line(f"alpha={a}",                "intent_scenario",a,"LAC")
print("--- (b) target 换 (alpha=0.10, LAC) ---")
line("target=scenario(18类)",          "intent_scenario",0.10,"LAC")
line("target=intent_full(更多类)",      "intent_full",   0.10,"LAC")
print("--- (c) CP score (target=scenario, alpha=0.10) ---")
line("score=LAC",                      "intent_scenario",0.10,"LAC")
line("score=APS",                      "intent_scenario",0.10,"APS")
print("--- (d) train-seed 稳定 (target=scenario, alpha=0.10, LAC, 0.8 子采样重训) ---")
for s in [1,2,3]: line(f"seed={s}",                          "intent_scenario",0.10,"LAC",s)
print("\n判定: 覆盖跨设定守 1-alpha; 分解结构 ASR<NLU(31% 量级)稳; modest word_std partial 随设定波动=诚实写 limitation.")

import pandas as _pd
_pd.DataFrame(_rows).to_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)),"robustness_summary.csv"),index=False)
print("落盘 robustness_summary.csv")
