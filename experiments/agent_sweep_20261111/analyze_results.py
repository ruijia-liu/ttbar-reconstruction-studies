"""Audit paired predictions and quantify whether learned gating beats fixed controls."""
import sys
from pathlib import Path
import hashlib
import json
import zipfile
import awkward as ak
import numpy as np
from scipy.stats import wasserstein_distance
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from train_gate import diagnostics

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
OUT=HERE/'analysis'
OUT.mkdir(exist_ok=True)
KEYS=['ttbar_mass','delta_abs_y','cos_K_plus','cos_phi']
RUNS=['huber_control','fixed_half','fixed_mean','truth_partition','learned_agent']
LABELS=['Huber control','Fixed 50/50','Fixed mean (6.21% MSE)','Truth partition','Learned agent']
COLORS=['#777777','#4a78a7','#9570a8','#c57b24','#168976']
plan=json.loads((HERE/'experiment_plan.json').read_text())
audit=json.loads((HERE/'remote_status_latest.json').read_text())
for i in range(5):
    assert f'24595228_{i}|COMPLETED|' in audit['accounting']
assert hashlib.sha256((HERE/'canonical_validation.npz').read_bytes()).hexdigest()==plan['canonical_npz_sha256']
with np.load(HERE/'canonical_validation.npz',allow_pickle=False) as z:
    truth=z['targets'].copy()
    assert z['keys'].tolist()==KEYS and np.array_equal(z['row_index'],np.arange(260190))
assert truth.shape==(260190,4) and np.isfinite(truth).all()
masks={'inclusive':np.ones(len(truth),dtype=bool),'truth_mass_gt800':truth[:,0]>800,
       'truth_mass_le800':truth[:,0]<=800,'truth_mass_ge1000':truth[:,0]>=1000}
metrics={}
predictions={}
provenance={}

def evaluate(pred,target):
    result={}
    for i,key in enumerate(KEYS):
        err=pred[:,i]-target[:,i]
        result[key]=dict(n=len(pred),mae=float(np.abs(err).mean()),rmse=float(np.sqrt(np.mean(err**2))),
                        bias=float(err.mean()),w1=float(wasserstein_distance(pred[:,i],target[:,i])))
    return result

paths={run:HERE/'results'/run/'predictions_val.parquet' for run in RUNS}
old=ROOT/'本轮新结果_20261109/03_输入预测及溯源'
paths.update(historical_mse=old/'val_predictions.parquet',historical_emd=old/'val_predictions_emd01.parquet')
for run,path in paths.items():
    arr=ak.from_parquet(path)
    pred=np.column_stack([ak.to_numpy(arr.predictions[k]).astype(np.float64) for k in KEYS])
    target=np.column_stack([ak.to_numpy(arr.targets[k]).astype(np.float64) for k in KEYS])
    assert pred.shape==truth.shape and np.isfinite(pred).all()
    alignment=np.max(np.abs(target-truth),axis=0)
    assert np.all(alignment<=np.array([.01,5e-4,5e-4,5e-4]))
    predictions[run]=pred
    metrics[run]={section:evaluate(pred[mask],truth[mask]) for section,mask in masks.items()}
    provenance[run]=dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),truth_alignment_maxabs=alignment.tolist())
    if run in RUNS:
        state=json.loads((path.parent/'status.json').read_text())
        assert state['status']=='complete' and state['prediction_rows']==260190
        assert state['predictions_sha256']==provenance[run]['sha256']
        assert np.array_equal(ak.to_numpy(arr.row_index),np.arange(260190)) and np.array_equal(target,truth)
        stored=json.loads((path.parent/'metrics.json').read_text())
        slices=json.loads((path.parent/'metrics_mass_slices.json').read_text())
        for section,expected in [('inclusive',stored),('truth_mass_gt800',slices['truth_mass_gt800']['metrics']),
                                 ('truth_mass_ge1000',slices['truth_mass_1000_inf']['metrics'])]:
            for key in KEYS:
                for metric in ['mae','rmse','bias','w1']:
                    assert np.isclose(metrics[run][section][key][metric],expected[key][metric],atol=1e-12,rtol=0)
        # Validate count matrices and their denominator convention independently.
        response=json.loads((path.parent/'response_matrices.json').read_text())
        for i,key in enumerate(KEYS):
            edges=np.array(response[key]['bin_edges'])
            counts=np.histogram2d(truth[:,i],pred[:,i],bins=(edges,edges))[0]
            denom=np.histogram(truth[:,i],bins=edges)[0]
            assert np.array_equal(counts,np.array(response[key]['counts']))
            assert np.array_equal(denom,np.array(response[key]['truth_counts']))
        provenance[run]['status']=state
    print('Verified',run,flush=True)

gate_root=HERE/'results/gate'
manifest=json.loads((gate_root/'artifact_manifest.json').read_text())
for name in ['val_gate.npy','val_tail.npy','diagnostics.json','status.json']:
    assert hashlib.sha256((gate_root/name).read_bytes()).hexdigest()==manifest['artifacts'][name]['sha256']
g=np.load(gate_root/'val_gate.npy',allow_pickle=False)
tail=np.load(gate_root/'val_tail.npy',allow_pickle=False)
assert g.shape==(len(truth),) and np.array_equal(tail,masks['truth_mass_gt800'])
diag=diagnostics(tail,g)
stored_diag=json.loads((gate_root/'diagnostics.json').read_text())['validation_mean']
for key in ['bce','brier','precision','recall','average_precision','expected_calibration_error']:
    assert np.isclose(diag[key],stored_diag[key],atol=1e-12,rtol=0)
diag['mean_probability_given_true_tail']=float(g[tail].astype(float).mean())
diag['mean_probability_given_true_nontail']=float(g[~tail].astype(float).mean())
diag['constant_prevalence_brier']=float(tail.mean()*(1-tail.mean()))
previous=ROOT/'loss_sweep_20261110/results/huber05/predictions_val.parquet'
assert hashlib.sha256(previous.read_bytes()).hexdigest()==provenance['huber_control']['sha256']
contrasts={}
for candidate in ['learned_agent','truth_partition']:
    contrasts[candidate]={}
    for reference in ['huber_control','fixed_mean','fixed_half','historical_emd']:
        contrasts[candidate][reference]={section:{key:{metric:100*(metrics[reference][section][key][metric]-metrics[candidate][section][key][metric])/metrics[reference][section][key][metric]
                         for metric in ['mae','rmse','w1']} for key in KEYS} for section in masks}
report=dict(events=len(truth),tail_events=int(tail.sum()),test_evaluated=False,
            metrics=metrics,contrasts_improvement_percent=contrasts,gate_diagnostics=diag,provenance=provenance,
            previous_huber_control_reproduced_exactly=True,gate_manifest=manifest,
            notes=['Equal approximate regressor wall-time, not equal epochs: 45 except truth partition 39.',
                   'Primary checkpoint uniformly selected by standardized validation MSE; one seed, no test claims.',
                   'Fixed mean matches training OOF agent mean; learned gate remains a soft probability, not a 0.5 hard routing.',
                   'Ground-truth partition used only in supervised loss, never as an inference input.',
                   'Historical model training budgets and checkpoint selection differ.'])
(OUT/'verified_metrics.json').write_text(json.dumps(report,indent=2)+'\n')
lines=['Frozen agent gate comparison: verified full-validation metrics',
       'N=260190; truth mtt>800 GeV N='+str(int(tail.sum())),'','Run | mass MAE | mass W1 | tail MAE | tail RMSE | tail bias']
for run in paths:
    whole=metrics[run]['inclusive']['ttbar_mass'];high=metrics[run]['truth_mass_gt800']['ttbar_mass']
    lines.append(f"{run} | {whole['mae']:.6f} | {whole['w1']:.6f} | {high['mae']:.6f} | {high['rmse']:.6f} | {high['bias']:.6f}")
lines += ['',json.dumps(diag,indent=2),'','Single seed; validation-selected; no test-set performance claim.']
(OUT/'comparison.txt').write_text('\n'.join(lines),encoding='utf-8')

plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'savefig.facecolor':'white','pdf.fonttype':42,'ps.fonttype':42})
fig,axes=plt.subplots(2,2,figsize=(14,8.5))
settings=[('inclusive','mae','Overall mass MAE [GeV]'),('truth_mass_gt800','mae','Mass MAE for truth mass > 800 GeV [GeV]'),
          ('inclusive','w1','Mass Wasserstein-1 [GeV]'),('truth_mass_gt800','bias','Mass bias for truth mass > 800 GeV [GeV]')]
for ax,(section,metric,title) in zip(axes.ravel(),settings):
    values=np.array([metrics[run][section]['ttbar_mass'][metric] for run in RUNS])
    width=values.max()-values.min()
    ax.scatter(values,np.arange(5),c=COLORS,s=65,zorder=3)
    ax.axvline(values[0],color='#999',ls='--',lw=1)
    ax.set_yticks(np.arange(5),LABELS)
    ax.set_ylim(4.5,-.5)
    ax.set_xlim(values.min()-.13*width,values.max()+.35*width)
    for i,value in enumerate(values):
        ax.annotate(f'{value:.2f}',(value,i),xytext=(7,0),textcoords='offset points',va='center')
    ax.set_title(title,pad=11)
    ax.grid(axis='x',alpha=.2)
    ax.spines[['top','right']].set_visible(False)
fig.suptitle('Does an event-dependent gate improve mass reconstruction?',fontsize=17,y=.98)
fig.text(.54,.025,'Lower MAE / W1 is better; bias closer to zero is better. Dashed line: Huber control.\nFull validation N=260,190 | one seed | same network and approximate wall-time | no test evaluation',ha='center',fontsize=10)
fig.subplots_adjust(left=.21,right=.98,top=.90,bottom=.12,wspace=.70,hspace=.48)
fig.savefig(OUT/'agent_comparison.png',dpi=180,bbox_inches='tight')
fig.savefig(OUT/'agent_comparison.pdf',bbox_inches='tight')
plt.close(fig)

fig,axes=plt.subplots(2,3,figsize=(14,9),sharex=True,sharey=True)
edges=np.array([300,380,440,500,750,2000])
denom=np.histogram(truth[:,0],bins=edges)[0]
for ax,run,label in zip(axes.ravel(),RUNS,LABELS):
    counts=np.histogram2d(truth[:,0],predictions[run][:,0],bins=(edges,edges))[0]
    matrix=np.divide(counts,denom[:,None],out=np.zeros_like(counts),where=denom[:,None]>0)
    im=ax.imshow(matrix,origin='lower',vmin=0,vmax=1,cmap='Blues')
    for (i,j),value in np.ndenumerate(matrix):
        if value>=.005:ax.text(j,i,f'{value:.2f}',ha='center',va='center',fontsize=11,color='white' if value>.5 else '#111')
    ticklabels=[f'{int(a)}–{int(b)}' for a,b in zip(edges[:-1],edges[1:])]
    ax.set_xticks(range(5),ticklabels,rotation=35,ha='right')
    ax.set_yticks(range(5),ticklabels)
    ax.tick_params(labelbottom=True,labelleft=True)
    ax.set_title(label,pad=12)
    ax.set_xlabel('Predicted mass bin [GeV]');ax.set_ylabel('True mass bin [GeV]')
axes.ravel()[-1].axis('off')
fig.colorbar(im,ax=axes.ravel()[-1],fraction=.18,pad=.05,label='Fraction of all events in the truth bin')
fig.suptitle('Mass response matrices: identical bins and color scale',fontsize=17,y=.99)
fig.text(.5,.015,'Rows: truth; columns: prediction. Truth-bin denominator includes predictions outside plotted range.\nUnequal physical bin widths are displayed as equal cells. Full validation, one seed.',ha='center',fontsize=10)
fig.subplots_adjust(left=.09,right=.98,top=.93,bottom=.14,wspace=.38,hspace=.52)
fig.savefig(OUT/'mass_response_matrices.png',dpi=180,bbox_inches='tight')
fig.savefig(OUT/'mass_response_matrices.pdf',bbox_inches='tight')
plt.close(fig)
print('\n'.join(lines[:11]))
print('Gate mean probability, true tail / non-tail:',diag['mean_probability_given_true_tail'],diag['mean_probability_given_true_nontail'])
