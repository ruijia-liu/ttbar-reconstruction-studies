"""Same-event validation distribution comparison for baseline and conditional OT."""
import hashlib
import json
import sys
from pathlib import Path

import awkward as ak
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

here=Path(__file__).resolve().parent
workspace=here.parent
plan=json.loads((here/'experiment_plan.json').read_text())
metrics=json.loads((here/'results/conditional_ot/metrics.json').read_text())
canon=np.load(here/'canonical_validation.npz')
keys=['ttbar_mass','delta_abs_y','cos_K_plus','cos_phi']
truth=canon['targets']
assert truth.shape==(260190,4) and list(canon['keys'])==keys

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):
    a=ak.from_parquet(path)
    p=np.column_stack([ak.to_numpy(a.predictions[k]) for k in keys])
    t=np.column_stack([ak.to_numpy(a.targets[k]) for k in keys])
    if 'row_index' in a.fields:assert np.array_equal(ak.to_numpy(a.row_index),np.arange(len(truth)))
    return p,t

base_path=workspace/'loss_sweep_20261110/results/huber1_emd01/predictions_val.parquet'
cot_path=here/'results/conditional_ot/predictions_val.parquet'
assert sha(base_path)==plan['base_artifacts']['predictions']['sha256']
assert sha(cot_path)==metrics['predictions']['sha256']
base,base_truth=read(base_path)
cot,cot_truth=read(cot_path)
assert np.array_equal(cot_truth,truth)
assert np.all(np.max(abs(base_truth-truth),axis=0)<=[5e-4,5e-7,5e-7,5e-7])
assert np.isfinite(base).all() and np.isfinite(cot).all()

bins=[np.linspace(300,2000,86),np.linspace(-2.4,2.4,61),
      np.linspace(-1,1,51),np.linspace(-1,1,51)]
titles=[r'$m_{t\bar t}$',r'$\Delta |y|$',r'$\cos\theta_K^+$',r'$\cos\varphi$']
colors={'truth':'#30343b','base':'#d67a2b','cot':'#087f72'}
fig=plt.figure(figsize=(15.5,9.4),facecolor='white')
grid=fig.add_gridspec(4,2,height_ratios=[2.15,.95,2.15,.95],hspace=.72,wspace=.22,
                      left=.075,right=.985,top=.84,bottom=.11)
handles=None
for j in range(4):
    block=j//2;col=j%2
    top=fig.add_subplot(grid[2*block,col])
    lower=fig.add_subplot(grid[2*block+1,col],sharex=top)
    edges=bins[j];centres=(edges[:-1]+edges[1:])/2
    h_truth=np.histogram(truth[:,j],bins=edges)[0]/len(truth)
    h_base=np.histogram(base[:,j],bins=edges)[0]/len(truth)
    h_cot=np.histogram(cot[:,j],bins=edges)[0]/len(truth)
    ht=top.step(centres,h_truth,where='mid',color=colors['truth'],lw=2.0,label='Truth')
    hb=top.step(centres,h_base,where='mid',color=colors['base'],lw=1.9,label='Huber + EMD')
    hc=top.step(centres,h_cot,where='mid',color=colors['cot'],lw=2.0,label='Conditional OT')
    if handles is None:handles=[ht[0],hb[0],hc[0]]
    b_w1=plan['baseline_metrics']['observables'][keys[j]]['w1']
    c_w1=metrics['observables'][keys[j]]['w1']
    fmt='.2f' if j==0 else '.4f'
    top.set_title(f'{titles[j]}  |  W1 {b_w1:{fmt}} -> {c_w1:{fmt}}',fontsize=14,pad=8)
    top.set_ylabel('Event fraction / bin',fontsize=10)
    top.grid(axis='y',alpha=.16)
    top.tick_params(axis='x',labelbottom=False)
    for values,color,name in [(h_base-h_truth,colors['base'],'Huber + EMD - truth'),
                               (h_cot-h_truth,colors['cot'],'Conditional OT - truth')]:
        lower.step(centres,100*values,where='mid',color=color,lw=1.7,label=name)
    lower.axhline(0,color='#637487',lw=.9)
    lower.set_ylabel('Difference (pp)',fontsize=10)
    lower.set_xlabel('GeV' if j==0 else 'Value',fontsize=10)
    lower.grid(axis='y',alpha=.13)
    if j==0:
        top.set_xlim(300,1200)
    elif j==1:
        top.set_xlim(-2.4,2.4)
    else:
        top.set_xlim(-1,1)
    for ax in (top,lower):
        for side in ('top','right'):ax.spines[side].set_visible(False)
fig.suptitle('Validation distributions: original regression and conditional OT',fontsize=18,y=.975)
fig.legend(handles,['Truth','Huber + EMD','Conditional OT'],loc='upper center',
           bbox_to_anchor=(.5,.932),ncol=3,frameon=False,fontsize=11)
fig.text(.075,.025,'Lower panels show prediction minus truth in percentage points of all 260,190 events per bin. '
         'Mass axis shows 300-1200 GeV; W1 uses every event.',fontsize=10,color='#3f5368')
dest=here/'figures'/'conditional_ot_validation_distributions.png'
fig.savefig(dest,dpi=180,facecolor='white')
fig.savefig(dest.with_suffix('.pdf'),facecolor='white')
plt.close(fig)
print(dest)
