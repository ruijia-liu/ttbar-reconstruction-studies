"""Validation evaluation: greedy disjoint jet-lepton edges; detector leptons only."""
import sys,json,hashlib,importlib.util
from pathlib import Path
ROOT=Path(__file__).resolve().parent.parent
import numpy as np,h5py,awkward as ak
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
P=ROOT/'hepsetreg_tools'; OUT=P/'results/vyper_epoch97_validation'; OUT.mkdir(parents=True,exist_ok=True)
spec=importlib.util.spec_from_file_location('physics',P/'benchmark_20261027/part_slot_audit/slot_four_observables.py'); phys=importlib.util.module_from_spec(spec); spec.loader.exec_module(phys)
NAMES=list(phys.OBSERVABLES)
LABELS=[r'$m_{t\bar t}$ [GeV]',r'$|y_t|-|y_{\bar t}|$',r'$\cos\theta_k^+$',r'$\cos\phi$']
def digest(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def p4(a):
 pt=a['pt'].astype(float); eta=a['eta'].astype(float); phi=a['phi'].astype(float)
 return np.stack([pt*np.cos(phi),pt*np.sin(phi),pt*np.sinh(eta),a['e'].astype(float)],axis=-1)
def greedy(e,s,nj):
 eligible=np.flatnonzero((e[0]<nj)!=(e[1]<nj))
 order=eligible[np.argsort(-s[eligible],kind='stable')]; used=set(); pairs=[]
 for k in order:
  a,b=map(int,e[:,k])
  if a not in used and b not in used:
   pairs.append((a,b) if a<nj else (b,a)); used.update((a,b))
   if len(pairs)==2:return pairs
 return []
# Nontrivial tests: greedy must reject reuse of either endpoint.
assert greedy(np.array([[0,0,1,1],[2,3,2,3]]),np.array([.9,.8,.7,.1]),2)==[(0,2),(1,3)]
f=np.array([[3.,4.,5.,12.],[0,0,0,10.]])
r,ok=phys.boost_to_rest(f,f); assert ok.all() and np.allclose(r[:,:3],0,atol=1e-12)
assert np.allclose(r[:,3]**2,f[:,3]**2-(f[:,:3]**2).sum(1))
report=json.loads((P/'vyper_val_prediction_report_epoch97.json').read_text())
assert digest(P/'vyper_val_inputs.h5')==report['source_sha256']; assert digest(P/'vyper_val_predictions_epoch97.h5')==report['output_sha256']
with h5py.File(P/'vyper_val_inputs.h5') as f:
 objs={k:f['INPUTS/'+k][:] for k in ['JET','ELECTRON','MUON']}; labs={k:f['LABELS/'+k][:] for k in objs}; nut=f['LABELS/NEUTRINO'][:]; fully=f['METADATA/FullyMatched'][:].astype(bool)
n=len(fully); top=np.full((n,4),np.nan); anti=top.copy(); lp=top.copy(); lm=top.copy(); chargeok=np.zeros(n,bool); correct=np.zeros((n,2),bool); decoded=np.zeros(n,bool); pairs_saved=np.full((n,2),-1); nuerr=[]
with h5py.File(P/'vyper_val_predictions_epoch97.h5') as f:
 g=f['VyPER']
 for start in range(0,n,4096):
  edges=g['EdgeIndex'][start:start+4096]; scores=g['EdgeSoftP'][start:start+4096]; nus=g['Neutrino'][start:start+4096]
  for j in range(len(edges)):
   i=start+j; arrays=[objs[k][i][np.isfinite(objs[k][i]['pt'])] for k in objs]; nj=len(arrays[0]); nodes=np.concatenate(arrays); leps=nodes[nj:]
   assert len(leps)==2 and nj>=2
   e=np.stack(edges[j]); s=np.stack(scores[j]); assert e.min()>=0 and e.max()<len(nodes)
   assert (e[0]<e[1]).all(), 'Expected mean-reduced undirected edges'
   pairs=greedy(e,s[0],nj) # channel 0 is bl, channel 1 is background
   if not pairs:continue
   q=leps['charge']; chargeok[i]=np.sum(q>0)==1 and np.sum(q<0)==1
   if not chargeok[i]:continue
   v=p4(nodes); nu=np.stack(nus[j],axis=1).astype(float); nv=np.column_stack([nu,np.linalg.norm(nu,axis=1)])
   truthnu=np.stack([nut[x][i] for x in ['px','py','pz']],axis=1); nuerr.append(nu-truthnu)
   for jet,lep in pairs:
    li=lep-nj; sign=0 if q[li]>0 else 1
    parent=v[jet]+v[lep]+nv[li]
    if sign==0: top[i]=parent;lp[i]=v[lep]
    else: anti[i]=parent;lm[i]=v[lep]
    pairs_saved[i,sign]=jet
    jl=labs['JET'][i][jet]
    ll=np.concatenate([labs[k][i][np.isfinite(labs[k][i])] for k in ['ELECTRON','MUON']])[li]
    correct[i,sign]=(jl==(1 if sign==0 else 4)) and (ll==(1 if sign==0 else 2))
   decoded[i]=True
  print('Decoded',min(start+4096,n),'/',n,flush=True) if start%40960==0 else None
pred,valid,quality=phys.derive_observables(top,anti,lp,lm,chargeok)
ref=ak.from_parquet(ROOT/'tmp/teacher_data_audit/ttbar-2L_val_051026.parquet',columns=['targets'])['targets']
truth={k:ak.to_numpy(ref[k]).astype(float) for k in ['ttbar_mass','cos_K_plus','cos_phi']}
def rapid(prefix):
 E=ak.to_numpy(ref[prefix+'_E']).astype(float); z=ak.to_numpy(ref[prefix+'_pz']).astype(float)
 return .5*np.log((E+z)/(E-z))
truth['delta_abs_y']=abs(rapid('top'))-abs(rapid('antitop'))
methods={'VyPER (release defaults)':pred}; alignment={}
for label,file in [('MSE','val_predictions.parquet'),('MSE + 0.1 EMD','val_predictions_emd01.parquet')]:
 a=ak.from_parquet(P/file); alignment[label]={}
 for k in NAMES:
  t=ak.to_numpy(a.targets[k]); alignment[label][k]=float(np.max(abs(t-truth[k])))
  assert np.allclose(t,truth[k],rtol=0,atol=.01 if k=='ttbar_mass' else 5e-4), (label,k,'truth order mismatch')
 methods[label]={k:ak.to_numpy(a.predictions[k]).astype(float) for k in NAMES}
metrics={}
for method,values in methods.items():
 metrics[method]={}
 for k in NAMES:
  mask=valid[k]&np.isfinite(values[k])&np.isfinite(truth[k]); d=values[k][mask]-truth[k][mask]
  metrics[method][k]=dict(n=int(mask.sum()),mae=float(abs(d).mean()),rmse=float(np.sqrt(np.mean(d*d))),bias=float(d.mean()),correlation=float(np.corrcoef(values[k][mask],truth[k][mask])[0,1]))
summary=dict(events=n,decoded=int(decoded.sum()),valid={k:int(v.sum()) for k,v in valid.items()},fully_matched=int(fully.sum()),assignment_both_correct_all=float(correct.all(1).mean()),assignment_both_correct_fully_matched=float(correct[fully].all(1).mean()),assignment_per_parent_fully_matched=float(correct[fully].mean()),neutrino_component_mae_GeV=np.abs(np.array(nuerr)).mean((0,1)).tolist(),metrics=metrics,baseline_truth_alignment_maxabs=alignment,source_sha256=report['source_sha256'],prediction_sha256=report['output_sha256'],conventions={'decoder':'Greedy descending class-0 score, exactly one lepton endpoint, disjoint nodes; no extra b-tag cut. Stable ties in stored edge order.','neutrinos':'Physical px py pz, associated with detector leptons in ELECTRON then MUON node order. E=norm(p).','angles':'Sequential lab -> ttbar CM -> parent rest; K is top direction in ttbar CM (teacher target convention). Paper appendix E says beam line despite helicity notation: unresolved wording inconsistency.','truth':'MC parton truth, not ideal reco; validation only; one diffusion draw per event.','training':'Official complete release architecture defaults, not paper Table XIII hyperparameters; historical baselines differ in training configuration.'})
(OUT/'metrics.json').write_text(json.dumps(summary,indent=2))
ak.to_parquet(ak.Array({'targets':truth,'predictions':pred,'valid':valid,'row_index':np.arange(n),'selected_jet_indices':pairs_saved,'fully_matched':fully}),OUT/'four_observables.parquet')
np.savez_compressed(OUT/'reconstructed_fourvectors.npz',top=top,antitop=anti,positive_lepton=lp,negative_lepton=lm)
plt.rcParams.update({'font.family':'serif','font.size':11,'axes.grid':True,'grid.alpha':.25,'xtick.direction':'in','ytick.direction':'in'})
colors={'MSE':'#d43d3d','MSE + 0.1 EMD':'#3268af','VyPER (release defaults)':'#13866b'}
order=['MSE','MSE + 0.1 EMD','VyPER (release defaults)']
bins={'ttbar_mass':np.linspace(300,1050,31),'delta_abs_y':np.linspace(-2.4,2.4,25),'cos_K_plus':np.linspace(-1,1,21),'cos_phi':np.linspace(-1,1,21)}
leak={}
fig=plt.figure(figsize=(14,9)); outer=fig.add_gridspec(2,2,hspace=.34,wspace=.18)
for col,k in enumerate(NAMES):
 gs=outer[col//2,col%2].subgridspec(2,1,height_ratios=[3,1],hspace=.05); ax=fig.add_subplot(gs[0]); ra=fig.add_subplot(gs[1],sharex=ax); b=bins[k]; tc=np.histogram(truth[k],b)[0]; td=tc/n/np.diff(b)
 ax.stairs(td,b,fill=True,color='#e3e3e3',label='MC parton truth'); ax.stairs(td,b,color='#555555',lw=.8)
 leak[k]={'truth_outside_plot':int(((truth[k]<b[0])|(truth[k]>b[-1])).sum())}
 for m in order:
  v=methods[m][k]; counts=np.histogram(v[np.isfinite(v)],b)[0]; ax.stairs(counts/n/np.diff(b),b,color=colors[m],label=m,lw=1.3); ratio=np.divide(counts,tc,out=np.full(len(tc),np.nan),where=tc>0);ra.stairs(ratio,b,color=colors[m]);leak[k][m]=int(((v<b[0])|(v>b[-1])|~np.isfinite(v)).sum())
 ax.tick_params(labelbottom=False);ax.yaxis.set_major_locator(MaxNLocator(5,prune='lower'));ax.set_ylabel('Normalized density');ra.set_ylabel('Pred./Truth');ra.set_xlabel(LABELS[col]);ra.axhline(1,c='gray',ls='--',lw=.8);ra.set_ylim(0,2);ra.set_xlim(b[0],b[-1]);ra.yaxis.set_major_locator(MaxNLocator(3,prune='upper'))
 if k=='ttbar_mass':
  all_ratios=[np.divide(np.histogram(methods[m][k],b)[0],tc,out=np.full(len(tc),np.nan),where=tc>0) for m in order]
  ra.set_ylim(0,max(2,float(np.nanmax(all_ratios))*1.05))
handles,labels=ax.get_legend_handles_labels();fig.legend(handles,labels,loc='upper center',ncol=4);fig.suptitle(f'MC validation: N = {n:,}',y=.945,fontsize=12);fig.subplots_adjust(top=.89,bottom=.07)
fig.savefig(ROOT/'VyPER_vs_baselines_val_distributions_four.png',dpi=180);plt.close(fig)
mbins={'ttbar_mass':np.array([300,380,440,500,750,2000]),'delta_abs_y':np.array([-2.4,-1.2,-.4,.4,1.2,2.4]),'cos_K_plus':np.linspace(-1,1,5),'cos_phi':np.linspace(-1,1,5)}
for names,stem in [(('ttbar_mass','cos_phi'),'VyPER_vs_baselines_val_migration_matrices'),(('delta_abs_y','cos_K_plus'),'VyPER_vs_baselines_val_migration_matrices_other')]:
 fig,axes=plt.subplots(2,3,figsize=(12,7.6),layout='constrained')
 for row,k in enumerate(names):
  b=mbins[k]; centers=np.arange(len(b)-1)+.5
  for col,m in enumerate(order):
   ax=axes[row,col];v=methods[m][k];mask=np.isfinite(v); c=np.histogram2d(truth[k][mask],v[mask],bins=(b,b))[0];den=np.histogram(truth[k],b)[0];z=np.divide(c,den[:,None],out=np.zeros_like(c),where=den[:,None]>0)
   im=ax.imshow(z,origin='lower',vmin=0,vmax=1,cmap='Reds',extent=(0,len(b)-1,0,len(b)-1),aspect='equal');ax.grid(False)
   for iy,ix in np.ndindex(z.shape):
    if z[iy,ix]>=.005:ax.text(ix+.5,iy+.5,f'{z[iy,ix]:.2f}',ha='center',va='center',fontsize=8,color='white' if z[iy,ix]>.5 else 'black')
   ax.set_xticks(np.arange(len(b)),[f'{x:g}' for x in b],rotation=40);ax.set_yticks(np.arange(len(b)),[f'{x:g}' for x in b]);ax.set_xlabel('Predicted '+LABELS[NAMES.index(k)]);ax.set_ylabel('MC truth '+LABELS[NAMES.index(k)]);ax.set_title(m)
 fig.colorbar(im,ax=axes.ravel().tolist(),label='Probability per truth bin',shrink=.8);fig.savefig(ROOT/(stem+'.png'),dpi=180);plt.close(fig)
summary['distribution_range_exclusions']=leak;(OUT/'metrics.json').write_text(json.dumps(summary,indent=2))
print(json.dumps({'metrics':metrics,'assignment':summary['assignment_both_correct_fully_matched'],'valid':summary['valid']},indent=2))
