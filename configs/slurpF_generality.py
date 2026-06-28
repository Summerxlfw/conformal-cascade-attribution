#!/usr/bin/env python3
"""Step3 generality: 更强 ASR(medium.en, WER 18%->15%)下 per-stage 分解/覆盖/no-gold 信号是否同结构.
gold_emb 不变(gold 转写不变); 本地 MiniLM(保真 cosine=1.0 已验)embed medium 转写当 asr_emb_medium.
对比 small.en vs medium.en 的 P1(覆盖)/P2(分解)/P3(modest 信号 partial).
预期: WER↓ => I_ASR↓(ASR 膨胀减), 分解结构(ASR 占少数)+ 信号 应保持 => generality.
"""
import os, sys, numpy as np, pandas as pd, scipy.stats as st, torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from scipy.stats import spearmanr
from transformers import AutoTokenizer, AutoModel
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from conformal_primitives import split_cp_calibrate_multiclass as cal, split_cp_predict_set_multiclass as pred
ALPHA=0.10; HERE=os.path.dirname(os.path.abspath(__file__)); P=os.path.join(HERE,"..","data","slurp_real")
base=pd.read_parquet(f"{P}/slurp_asr.parquet"); confS=pd.read_parquet(f"{P}/slurp_asr_logprob.parquet").set_index("id")
med=pd.read_parquet(f"{P}/slurp_asr_medium.parquet").set_index("id")
id2lab=dict(zip(base.id,base.intent_scenario)); le=LabelEncoder().fit(base.intent_scenario)
# MiniLM(保真已验 cosine=1.0)
MID='sentence-transformers/all-MiniLM-L6-v2'; tok=AutoTokenizer.from_pretrained(MID); mdl=AutoModel.from_pretrained(MID).eval()
def embed(texts,bs=64):
    out=[]
    for i in range(0,len(texts),bs):
        enc=tok([str(t) for t in texts[i:i+bs]],padding=True,truncation=True,max_length=256,return_tensors='pt')
        with torch.no_grad(): o=mdl(**enc)
        mask=enc['attention_mask'].unsqueeze(-1).float()
        mean=(o.last_hidden_state*mask).sum(1)/mask.sum(1).clamp(min=1e-9)
        out.append(torch.nn.functional.normalize(mean,p=2,dim=1).numpy())
    return np.vstack(out)
def load(s):
    z=np.load(f"{P}/features_{s}.npz",allow_pickle=True); ids=z["ids"]
    return z["gold_emb"],z["asr_emb"],le.transform([id2lab[i] for i in ids]),ids
Gtr,Atr,ytr,_=load("train"); Gva,Ava_s,yva,idva=load("devel"); Gte,Ate_s,yte,idte=load("test")
print("embed medium devel+test 转写...")
Ava_m=embed([med.loc[i,"asr_transcript_medium"] for i in idva]); Ate_m=embed([med.loc[i,"asr_transcript_medium"] for i in idte])
clf=LogisticRegression(max_iter=1000,C=10).fit(Gtr,ytr); LOG=lambda M:np.log(np.clip(M,1e-12,1))
def prho(x,y,z):
    rx,ry,rz=st.rankdata(x),st.rankdata(y),st.rankdata(z)
    return spearmanr(rx-np.polyval(np.polyfit(rz,rx,1),rz),ry-np.polyval(np.polyfit(rz,ry,1),rz)).correlation

def analyze(Ava,Ate,wordstd,wer_te,tag):
    Pva=clf.predict_proba(Ava); Pa=clf.predict_proba(Ate); Pg=clf.predict_proba(Gte)
    t=cal(LOG(Pva),yva,ALPHA)
    Ca,sza=pred(LOG(Pa),t); Cg,szg=pred(LOG(Pg),t); sza=sza.astype(float); szg=szg.astype(float)
    cov=Ca[np.arange(len(yte)),yte].mean(); covg=Cg[np.arange(len(yte)),yte].mean()
    delta=sza-szg; I_ASR=delta.mean(); I_NLU=szg.mean()-1.0; tot=sza.mean()-1.0
    part=prho(wordstd,delta,sza)
    acc=clf.score(Ate,yte)
    print(f"  [{tag}] WER={wer_te.mean():.3f} acc={acc:.3f} | cov={cov:.3f} | I_ASR={I_ASR:+.3f} I_NLU={I_NLU:+.3f} ASR={100*I_ASR/tot:.0f}% | word_std partial={part:+.3f}")
    return dict(tag=tag,wer=wer_te.mean(),acc=acc,cov=cov,I_ASR=I_ASR,I_NLU=I_NLU,tot=tot,part=part)

print("\n=== small.en vs medium.en generality (gold_emb 不变, 同 NLU/alpha) ===")
rs=analyze(Ava_s,Ate_s, confS.loc[idte,"asr_word_logprob_std"].values, base.set_index("id").loc[idte,"wer"].values, "small.en")
rm=analyze(Ava_m,Ate_m, med.loc[idte,"asr_word_logprob_std_medium"].values, med.loc[idte,"wer_medium"].values, "medium.en")
print(f"\n生成性判定:")
print(f"  WER {rs['wer']:.3f}->{rm['wer']:.3f}(↓{rs['wer']-rm['wer']:.3f}) => I_ASR {rs['I_ASR']:+.3f}->{rm['I_ASR']:+.3f} ({'↓预期成立' if rm['I_ASR']<rs['I_ASR'] else '未降!'})")
print(f"  分解结构 ASR 占少数: small {100*rs['I_ASR']/rs['tot']:.0f}% / medium {100*rm['I_ASR']/rm['tot']:.0f}% ({'保持' if rm['I_ASR']/rm['tot']<0.5 else '破'})")
print(f"  覆盖守 0.90: small {rs['cov']:.3f} / medium {rm['cov']:.3f}")
print(f"  modest 信号 partial: small {rs['part']:+.3f} / medium {rm['part']:+.3f} ({'同结构正' if rm['part']>0.05 else '形态变/失效'})")
pd.DataFrame([rs,rm]).to_csv(os.path.join(HERE,"slurpF_generality.csv"),index=False)
print("落盘 slurpF_generality.csv")
