"""Recompute ordered validation metrics and compare loss-only experiments."""
import sys
from pathlib import Path
import hashlib
import json
import zipfile
import numpy as np
import awkward as ak
from scipy.stats import wasserstein_distance
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
OUT=HERE/'analysis'
OUT.mkdir(exist_ok=True)
KEYS=['ttbar_mass','delta_abs_y','cos_K_plus','cos_phi']
LABELS=[r'$m_{t\bar t}$',r'$\Delta|y|$',r'$\cos\theta_k^+$',r'$\cos\varphi$']
RUNS=['mse','mse_emd01','huber05','huber1','logcosh1','huber1_emd01','huber1_swd01']
NAMES=['MSE','MSE + 0.1 EMD',r'Huber $\delta=0.5$',r'Huber $\delta=1$',r'Log-cosh $s=1$',r'Huber $\delta=1$ + 0.1 EMD',r'Huber $\delta=1$ + 0.1 SW1']
plan=json.loads((HERE/'experiment_plan.json').read_text(encoding='utf-8'))
audit=json.loads((HERE/'remote_status_latest.json').read_text())
assert all(f'24541197_{i}|COMPLETED|' in audit['accounting'] for i in range(7))
assert hashlib.sha256((HERE/'canonical_validation.npz').read_bytes()).hexdigest()==plan['canonical_npz_sha256']
with np.load(HERE/'canonical_validation.npz',allow_pickle=False) as f:
    truth=f['targets'].copy()
    assert f['keys'].tolist()==KEYS and np.array_equal(f['row_index'],np.arange(260190))
assert truth.shape==(260190,4) and np.isfinite(truth).all()
tail=truth[:,0]>800
metrics={}
metadata={}
predictions={}
alignments={}

def evaluate(pred,target):
    result={}
    for i,key in enumerate(KEYS):
        err=pred[:,i]-target[:,i]
        result[key]=dict(n=len(pred),mae=float(np.abs(err).mean()),mse=float(np.mean(err**2)),
                        rmse=float(np.sqrt(np.mean(err**2))),bias=float(err.mean()),
                        w1=float(wasserstein_distance(pred[:,i],target[:,i])))
    return result

paths={run:HERE/'results'/run/'predictions_val.parquet' for run in RUNS}
old=ROOT/'本轮新结果_20261109/03_输入预测及溯源'
paths.update(historical_mse=old/'val_predictions.parquet',historical_emd=old/'val_predictions_emd01.parquet')
for run,path in paths.items():
    data=ak.from_parquet(path)
    pred=np.column_stack([ak.to_numpy(data.predictions[k]).astype(np.float64) for k in KEYS])
    target=np.column_stack([ak.to_numpy(data.targets[k]).astype(np.float64) for k in KEYS])
    assert pred.shape==truth.shape and np.isfinite(pred).all()
    diff=np.max(np.abs(target-truth),axis=0)
    assert np.all(diff<=np.array([.01,5e-4,5e-4,5e-4])),(run,diff)
    alignments[run]=dict(zip(KEYS,diff.tolist()))
    predictions[run]=pred
    metrics[run]=dict(inclusive=evaluate(pred,truth),truth_mass_gt800=evaluate(pred[tail],truth[tail]))
    metadata[run]=dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    if run in RUNS:
        assert np.array_equal(ak.to_numpy(data.row_index),np.arange(len(truth)))
        state=json.loads((path.parent/'status.json').read_text())
        assert state['status']=='complete' and state['prediction_rows']==260190
        assert metadata[run]['sha256']==state['predictions_sha256']
        for split in ['train','val']:
            assert state[split]['sha256']==plan['data_sha256'][split]
        stored=json.loads((path.parent/'metrics.json').read_text())
        stored_tail=json.loads((path.parent/'metrics_mass_slices.json').read_text())['truth_mass_gt800']
        assert stored_tail['n']==int(tail.sum())
        for key in KEYS:
            for metric in ['mae','rmse','bias','w1']:
                assert np.isclose(metrics[run]['inclusive'][key][metric],stored[key][metric],atol=1e-12,rtol=0)
                assert np.isclose(metrics[run]['truth_mass_gt800'][key][metric],stored_tail['metrics'][key][metric],atol=1e-12,rtol=0)
        metadata[run]['status']=state
    print('Verified',run,flush=True)

# Hold-out projection directions assess joint shape beyond the training's 32 directions.
# All methods use the identical train-only mean/std, and no samples are clipped.
state=metadata['mse']['status']
mean=np.array(state['target_mean'])
std=np.array(state['target_std'])+1e-8
rng=np.random.default_rng(110112)
directions=rng.normal(size=(64,4))
directions/=np.linalg.norm(directions,axis=1,keepdims=True)
truth_z=(truth-mean)/std
truth_projection=np.sort(truth_z@directions.T,axis=0)
offdiag=~np.eye(4,dtype=bool)
truth_corr=np.corrcoef(truth,rowvar=False)
for run,pred in predictions.items():
    projected=np.sort(((pred-mean)/std)@directions.T,axis=0)
    corr=np.corrcoef(pred,rowvar=False)
    metrics[run]['joint']=dict(heldout_sliced_w1=float(np.abs(projected-truth_projection).mean()),
                              correlation_mae=float(np.abs(corr-truth_corr)[offdiag].mean()),correlation=corr.tolist())
    print('Joint evaluated',run,flush=True)
relative={run:{metric:{k:100*(metrics['mse']['inclusive'][k][metric]-metrics[run]['inclusive'][k][metric])/metrics['mse']['inclusive'][k][metric]
                       for k in KEYS} for metric in ['mae','w1']} for run in RUNS}
report=dict(validation_events=260190,tail_events=int(tail.sum()),train_events=4692288,test_evaluated=False,
            canonical_truth_sha256=plan['canonical_npz_sha256'],metrics=metrics,relative_improvement_percent=relative,
            truth_alignment_maxabs=alignments,provenance=metadata,joint_evaluation=dict(projections=64,seed=110112,directions=directions.tolist()),
            notes=['All prediction rows and metrics verified against common canonical truth.',
                   'One seed, validation selected; no significance or test-performance claims.',
                   'Equal approximate wall-time, not equal updates; 41-51 full epochs.',
                   'All new checkpoints selected by standardized validation MSE. Historical selection and budgets differ.',
                   'Huber/log-cosh point terms have factor two for local MSE curvature matching.',
                   'SW1 held-out directions approximate joint distribution; correlation is an additional diagnostic.'])
(OUT/'verified_metrics.json').write_text(json.dumps(report,indent=2)+'\n')
lines=['Loss-only sweep: verified full-validation results','N=260190; truth mass >800 GeV N='+str(int(tail.sum())),
       'All results use main standardized-val-MSE-selected checkpoint. No test evaluation.','']
for run in paths:
    lines.append(run)
    lines.append('  MAE (mass, delta, cosK, cosphi): '+', '.join(f"{metrics[run]['inclusive'][k]['mae']:.6f}" for k in KEYS))
    lines.append('  W1  (mass, delta, cosK, cosphi): '+', '.join(f"{metrics[run]['inclusive'][k]['w1']:.6f}" for k in KEYS))
    lines.append('  tail mass MAE/bias: '+', '.join(f"{metrics[run]['truth_mass_gt800']['ttbar_mass'][k]:.6f}" for k in ['mae','bias']))
    lines.append('  held-out joint SW1 / correlation MAE: '+', '.join(f"{metrics[run]['joint'][k]:.6f}" for k in ['heldout_sliced_w1','correlation_mae']))
    lines.append('')
(OUT/'comparison.txt').write_text('\n'.join(lines),encoding='utf-8')

plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.titlesize':14,'axes.labelsize':12,
                     'savefig.facecolor':'white','pdf.fonttype':42,'ps.fonttype':42})
fig,axes=plt.subplots(1,2,figsize=(16,7.4),sharey=True)
for ax,metric,title in zip(axes,['mae','w1'],['Event-by-event MAE','Marginal Wasserstein-1']):
    values=np.array([[relative[r][metric][k] for k in KEYS] for r in RUNS])
    bound=2.5 if metric=='mae' else 60
    im=ax.imshow(values,cmap='RdBu',norm=TwoSlopeNorm(vcenter=0,vmin=-bound,vmax=bound),aspect='auto')
    for (i,j),value in np.ndenumerate(values):
        ax.text(j,i,f'{value:+.1f}%',ha='center',va='center',fontsize=12,
                color='white' if abs(value)>bound*.60 else '#151515')
    ax.set_xticks(range(4),LABELS)
    ax.set_yticks(range(7),NAMES)
    ax.tick_params(length=0)
    ax.set_title(title,pad=14)
    for spine in ax.spines.values():spine.set_visible(False)
    fig.colorbar(im,ax=ax,orientation='horizontal',pad=.10,fraction=.045,label='Improvement relative to this-round MSE [%]')
fig.suptitle('New losses: accuracy and distribution shape',fontsize=20,y=.98)
fig.text(.60,.925,'Positive / blue = better; negative / red = worse. Color scales differ between panels.',ha='center',fontsize=11)
fig.text(.55,.025,'Full validation: 260,190 events | same architecture, approximate time budget, seed and checkpoint rule\nHuber and log-cosh terms scaled by 2. One seed; validation results only.',ha='center',fontsize=10)
fig.subplots_adjust(left=.23,right=.98,top=.86,bottom=.16,wspace=.20)
fig.savefig(OUT/'loss_comparison.png',dpi=180,bbox_inches='tight')
fig.savefig(OUT/'loss_comparison.pdf',bbox_inches='tight')
plt.close(fig)

fig,axes=plt.subplots(1,2,figsize=(13,7),sharey=True)
for ax,metric,label in zip(axes,['mae','bias'],['Tail mass MAE [GeV]; lower is better','Tail mass bias [GeV]; closer to 0 is better']):
    values=[metrics[run]['truth_mass_gt800']['ttbar_mass'][metric] for run in RUNS]
    ax.scatter(values,range(7),s=65,color=['#4d4d4d','#275d9b','#c77127','#c77127','#8c6bb1','#19856d','#4a96a6'])
    ax.axvline(values[0],color='#777',ls='--',lw=1,label='This-round MSE')
    ax.set_yticks(range(7),NAMES)
    ax.invert_yaxis()
    width=max(values)-min(values)
    ax.set_xlim(min(values)-.15*width,max(values)+.35*width)
    for y,value in enumerate(values):ax.annotate(f'{value:.2f}',(value,y),xytext=(8,0),textcoords='offset points',va='center')
    ax.set_xlabel(label);ax.grid(axis='x',alpha=.2)
    ax.spines[['top','right']].set_visible(False)
fig.suptitle(r'High-mass tail: MC truth $m_{t\bar t}>800$ GeV',fontsize=18,y=.97)
fig.text(.55,.035,f'N = {int(tail.sum()):,} | truth-defined common event selection | no prediction clipping | one seed, validation only',ha='center',fontsize=10)
fig.subplots_adjust(left=.25,right=.98,top=.88,bottom=.15,wspace=.25)
fig.savefig(OUT/'high_mass_tail.png',dpi=180,bbox_inches='tight')
fig.savefig(OUT/'high_mass_tail.pdf',bbox_inches='tight')
plt.close(fig)
with zipfile.ZipFile(HERE/'loss_results_summary.zip','w',compression=zipfile.ZIP_DEFLATED) as archive:
    for path in sorted(OUT.iterdir()):
        if path.is_file():archive.write(path,path.name)
    archive.write(HERE/'experiment_plan.json','experiment_plan.json')
    archive.write(HERE/'README.txt','README.txt')
print('\n'.join(lines))
