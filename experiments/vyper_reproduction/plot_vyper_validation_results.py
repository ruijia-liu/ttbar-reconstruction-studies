import sys,json
from pathlib import Path
ROOT=Path.cwd()
import awkward as ak,numpy as np,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
P=ROOT/'hepsetreg_tools';OUT=P/'results/vyper_epoch97_validation'
a=ak.from_parquet(OUT/'four_observables.parquet');n=len(a)
NAMES=['ttbar_mass','delta_abs_y','cos_K_plus','cos_phi']
LABELS=[r'$m_{t\bar t}$ [GeV]',r'$|y_t|-|y_{\bar t}|$',r'$\cos\theta_k^+$',r'$\cos\phi$']
truth={k:ak.to_numpy(a.targets[k]) for k in NAMES}
methods={'VyPER (release defaults)':{k:ak.to_numpy(a.predictions[k]) for k in NAMES}}
for m,f in [('MSE','val_predictions.parquet'),('MSE + 0.1 EMD','val_predictions_emd01.parquet')]:
 b=ak.from_parquet(P/f);methods[m]={k:ak.to_numpy(b.predictions[k]) for k in NAMES}
summary=json.loads((OUT/'metrics.json').read_text())
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
