"""Recompute aligned full-validation metrics and export scientific figures."""
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
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
OUT=HERE/'figures'
OUT.mkdir(exist_ok=True)
N=260190
KEYS=['ttbar_mass','delta_abs_y','cos_K_plus','cos_phi']
LABELS=[r'$m_{t\bar t}$',r'$\Delta|y|$',r'$\cos\theta_k^+$',r'$\cos\varphi$']
VARIANTS=['base','lr_1e4','lr_1e3','width64','layers6','batch1024','batch2048','heads4','dropout0','wd0','warmup_cosine','emd005','emd01']
VLAB=['Baseline',r'Learning rate $10^{-4}$',r'Learning rate $10^{-3}$',r'$d_{model}=64$','6 layers','Batch size 1024','Batch size 2048','4 attention heads','Backbone dropout = 0','Weight decay = 0','Warmup + cosine*','MSE + 0.05 EMD','MSE + 0.1 EMD']
COLORS={'regression':'#3366aa','part':'#c56828','historical_mse':'#bc3940','historical_emd':'#315faa','regression_emd01':'#9063ad','part_emd01':'#14866f'}
ARCH={'regression':'Transformer','part':'ParT'}
canonical_path=ROOT/'本轮新结果_20261109/02_评估结果/four_observables.parquet'
canonical=ak.from_parquet(canonical_path)
assert len(canonical)==N and np.array_equal(ak.to_numpy(canonical.row_index),np.arange(N))
truth={k:ak.to_numpy(canonical.targets[k]).astype(np.float64) for k in KEYS}
assert all(np.isfinite(x).all() for x in truth.values())
assert all(ak.to_numpy(canonical.valid[k]).all() for k in KEYS)
plan=json.loads((HERE/'experiment_plan.json').read_text(encoding='utf-8'))
audit=json.loads((HERE/'remote_status_latest.json').read_text(encoding='utf-8-sig'))
predictions={}
metrics={}
alignment={}
provenance={}

def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()

def evaluate(name,path):
    data=ak.from_parquet(path)
    assert len(data)==N,(name,len(data))
    values={k:ak.to_numpy(data.predictions[k]).astype(np.float64) for k in KEYS}
    alignment[name]={}
    metrics[name]={}
    for k in KEYS:
        target=ak.to_numpy(data.targets[k]).astype(np.float64)
        tolerance=0.01 if k=='ttbar_mass' else 5e-4
        assert target.shape==(N,) and np.allclose(target,truth[k],atol=tolerance,rtol=0),(name,k,'truth/order mismatch')
        assert values[k].shape==(N,) and np.isfinite(values[k]).all(),(name,k,'nonfinite prediction')
        alignment[name][k]=float(np.max(np.abs(target-truth[k])))
        error=values[k]-truth[k]
        metrics[name][k]=dict(n=N,mae=float(np.abs(error).mean()),rmse=float(np.sqrt(np.mean(error**2))),bias=float(error.mean()),w1=float(wasserstein_distance(values[k],truth[k])))
    predictions[name]=values
    provenance[name]={'path':str(path),'sha256':digest(path)}

for cfg in plan['runs']:
    name=cfg['run_id'];folder=HERE/'results'/name
    status=json.loads((folder/'status.json').read_text())
    assert status['status']=='complete' and status['train']['rows']==4692288 and status['val']['rows']==N
    assert status['val']['sha256']=='da630904b6c91a8afefe5a4fa96a0fd8a9eaac537b3e7f975efdca31eecc28ff'
    provenance[name]={'status':status,'config':cfg}
    evaluate(name,folder/'predictions_val.parquet')
    provenance[name].update(status=status,config=cfg)
    print('Evaluated',name,flush=True)
historical_folder=ROOT/'本轮新结果_20261109/03_输入预测及溯源'
evaluate('historical_mse',historical_folder/'val_predictions.parquet')
evaluate('historical_emd',historical_folder/'val_predictions_emd01.parquet')
record=dict(events=N,train_events=4692288,test_evaluated=False,truth_path=str(canonical_path),truth_sha256=digest(canonical_path),
    metrics=metrics,truth_alignment_maxabs=alignment,provenance=provenance,
    notes=['All 28 methods recalculated on identical canonical MC truth; full event order retained, no clipping.',
           'One seed; checkpoints selected with validation loss; validation results are exploratory.',
           'Sweep allocations capped at 30 min, actual successful elapsed about 22 min; historical budgets differ.',
           'Warmup/cosine reaches zero learning rate at fixed 20-epoch horizon; later epochs may not update weights.',
           'No uncertainty bands or significance claims; top-ranked values are selected on validation.'])
(OUT/'metrics_common_truth.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')

plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.titlesize':12,'axes.labelsize':11,'axes.spines.top':False,'axes.spines.right':False,
    'axes.grid':True,'grid.alpha':.2,'grid.linewidth':.6,'xtick.direction':'out','ytick.direction':'out','savefig.facecolor':'white','pdf.fonttype':42,'ps.fonttype':42})
NOTE='Validation N = 260,190 | full training sample | one seed | common truth | historical training budgets differ'

def save(fig,stem):
    fig.savefig(OUT/(stem+'.png'),dpi=180,bbox_inches='tight')
    fig.savefig(OUT/(stem+'.pdf'),bbox_inches='tight')
    plt.close(fig)

def paired(metric,stem,title):
    fig,axes=plt.subplots(2,2,figsize=(14.5,11),sharey=True)
    y=np.arange(len(VARIANTS))
    for j,ax in enumerate(axes.ravel()):
        k=KEYS[j]
        for arch,offset,marker in [('regression',-.12,'o'),('part',.12,'s')]:
            vals=[metrics[arch+'_'+s][k][metric] for s in VARIANTS]
            ax.plot(vals,y+offset,marker,ls='none',color=COLORS[arch],ms=5.5,label=ARCH[arch],zorder=3)
        for hist,label,style in [('historical_mse','Historical MSE','--'),('historical_emd','Historical MSE + 0.1 EMD',':')]:
            ax.axvline(metrics[hist][k][metric],color=COLORS[hist],ls=style,lw=1.6,label=label,zorder=2)
        for i,s in enumerate(VARIANTS):
            a,b=[metrics[arch+'_'+s][k][metric] for arch in ['regression','part']]
            ax.plot([a,b],[i-.12,i+.12],color='#c8c8c8',lw=.7,zorder=1)
        ax.set_yticks(y,VLAB);ax.set_ylim(len(VARIANTS)-.5,-.6)
        ax.set_title(LABELS[j]+(' [GeV]' if j==0 else ''))
        ax.set_xlabel(('Mean absolute error' if metric=='mae' else 'Wasserstein-1 distance')+(' [GeV]' if j==0 else ''))
        ax.grid(axis='y',visible=False);ax.tick_params(axis='y',length=0)
        if j%2:ax.tick_params(labelleft=True)
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.suptitle(title+' (lower is better)',y=.985,fontsize=16)
    fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.5,.958),ncol=4,frameon=False)
    fig.text(.5,.012,NOTE+'\n*Cosine schedule reaches zero learning rate after 20 epochs.',ha='center',fontsize=9,color='#444444')
    fig.subplots_adjust(left=.19,right=.98,bottom=.085,top=.895,wspace=.65,hspace=.25)
    save(fig,stem)

def effects(metric,stem,title):
    data={arch:np.array([[100*(metrics[arch+'_base'][k][metric]-metrics[arch+'_'+v][k][metric])/metrics[arch+'_base'][k][metric] for k in KEYS] for v in VARIANTS[1:]]) for arch in ARCH}
    bound=max(1,float(np.ceil(max(np.abs(x).max() for x in data.values()))))
    fig,axes=plt.subplots(1,2,figsize=(13.4,8.0),sharey=True)
    norm=TwoSlopeNorm(vmin=-bound,vcenter=0,vmax=bound)
    for ax,arch in zip(axes,ARCH):
        im=ax.imshow(data[arch],cmap='RdBu',norm=norm,aspect='auto')
        for (i,j),value in np.ndenumerate(data[arch]):
            ax.text(j,i,f'{value:+.1f}',ha='center',va='center',color='white' if abs(value)>bound*.60 else '#151515',fontsize=10)
        ax.set_title(ARCH[arch]);ax.set_xticks(range(4),LABELS);ax.set_yticks(range(12),VLAB[1:]);ax.tick_params(length=0);ax.grid(False)
        for spine in ax.spines.values():spine.set_visible(False)
    fig.suptitle(title,fontsize=16,y=.98)
    fig.text(.5,.923,'Improvement (%) = 100 × (architecture baseline − variant) / architecture baseline',ha='center',fontsize=10)
    cax=fig.add_axes([.92,.16,.018,.67]);fig.colorbar(im,cax=cax,label='Improvement [%]; positive = better')
    fig.text(.5,.025,NOTE+'\n*Cosine schedule reaches zero learning rate after 20 epochs.',ha='center',fontsize=9,color='#444444')
    fig.subplots_adjust(left=.23,right=.89,top=.87,bottom=.11,wspace=.12)
    save(fig,stem)

def tradeoffs():
    fig,axes=plt.subplots(2,2,figsize=(13.6,10.0))
    weight_colors={0:'#8d98a5',.05:'#dba74b',.1:'#228a70'}
    config_by_id={c['run_id']:c for c in plan['runs']}
    for j,ax in enumerate(axes.ravel()):
        k=KEYS[j]
        for cfg in plan['runs']:
            name=cfg['run_id'];m=metrics[name][k]
            ax.scatter(m['mae'],m['w1'],marker='o' if cfg['model']=='regression' else 's',c=weight_colors[cfg['emd_weight']],s=45,edgecolors='white',linewidths=.55,zorder=3)
        frontier=[]
        for name in config_by_id:
            a=metrics[name][k]
            if not any(b['mae']<=a['mae'] and b['w1']<=a['w1'] and (b['mae']<a['mae'] or b['w1']<a['w1']) for other in config_by_id if other!=name for b in [metrics[other][k]]):frontier.append(name)
        frontier.sort(key=lambda n:metrics[n][k]['mae'])
        ax.plot([metrics[n][k]['mae'] for n in frontier],[metrics[n][k]['w1'] for n in frontier],color='#333333',ls='--',lw=.9,zorder=1)
        selected={min(config_by_id,key=lambda n:metrics[n][k]['mae']),min(config_by_id,key=lambda n:metrics[n][k]['w1'])}
        for n in selected:
            cfg=config_by_id[n];label=('T' if cfg['model']=='regression' else 'ParT')+': '+{'emd01':'EMD 0.1','emd005':'EMD 0.05','dropout0':'dropout 0','lr_1e3':'LR 1e−3','batch2048':'batch 2048'}.get(n.split('_',1)[1],n.split('_',1)[1])
            ax.annotate(label,(metrics[n][k]['mae'],metrics[n][k]['w1']),xytext=(6,9),textcoords='offset points',fontsize=8.5,bbox=dict(facecolor='white',edgecolor='none',alpha=.75,pad=1.5))
        for name,marker in [('historical_mse','X'),('historical_emd','*')]:
            ax.scatter(metrics[name][k]['mae'],metrics[name][k]['w1'],c=COLORS[name],marker=marker,s=100,zorder=5)
        ax.set_title(LABELS[j]);ax.set_xlabel('MAE'+(' [GeV]' if j==0 else ''));ax.set_ylabel('W1'+(' [GeV]' if j==0 else ''))
        ax.margins(x=.20,y=.20)
    legend=[Line2D([],[],marker='o',ls='none',color='#777777',label='Transformer'),Line2D([],[],marker='s',ls='none',color='#777777',label='ParT')]
    legend += [Line2D([],[],marker='o',ls='none',color=c,label=f'EMD weight {w:g}') for w,c in weight_colors.items()]
    legend += [Line2D([],[],marker='X',ls='none',color=COLORS['historical_mse'],label='Historical MSE'),Line2D([],[],marker='*',ls='none',color=COLORS['historical_emd'],label='Historical MSE + EMD')]
    fig.suptitle('Event accuracy versus marginal distribution agreement',y=.99,fontsize=16)
    fig.legend(handles=legend,loc='upper center',bbox_to_anchor=(.5,.955),ncol=4,frameon=False,fontsize=9)
    fig.text(.5,.015,NOTE+'\nLower left is better; dashed line joins nondominated sweep configurations (historical references excluded).',ha='center',fontsize=9,color='#444444')
    fig.subplots_adjust(top=.85,bottom=.10,left=.075,right=.98,hspace=.35,wspace=.24)
    save(fig,'05_mae_vs_w1_tradeoff')

SELECTED=['historical_mse','historical_emd','regression_emd01','part_emd01']
SLAB={'historical_mse':'Historical MSE','historical_emd':'Historical MSE + 0.1 EMD','regression_emd01':'Transformer + 0.1 EMD (new)','part_emd01':'ParT + 0.1 EMD (new)'}
BINS={'ttbar_mass':np.linspace(300,1050,31),'delta_abs_y':np.linspace(-2.4,2.4,25),'cos_K_plus':np.linspace(-1,1,21),'cos_phi':np.linspace(-1,1,21)}

def distributions():
    fig=plt.figure(figsize=(14,10));grid=fig.add_gridspec(2,2,hspace=.34,wspace=.20)
    leakage={}
    for j,k in enumerate(KEYS):
        gs=grid[j//2,j%2].subgridspec(2,1,height_ratios=[3,1],hspace=.05)
        ax=fig.add_subplot(gs[0]);ratio=fig.add_subplot(gs[1],sharex=ax);bins=BINS[k]
        tc=np.histogram(truth[k],bins)[0];density=tc/N/np.diff(bins)
        ax.stairs(density,bins,fill=True,color='#e1e1e1',label='MC parton truth',zorder=1)
        ax.stairs(density,bins,color='#666666',lw=.7,zorder=2)
        leakage[k]={'truth_outside':int(np.sum((truth[k]<bins[0])|(truth[k]>bins[-1])))}
        allrat=[]
        for name in SELECTED:
            count=np.histogram(predictions[name][k],bins)[0]
            ax.stairs(count/N/np.diff(bins),bins,color=COLORS[name],lw=1.35,label=SLAB[name])
            values=np.divide(count,tc,out=np.full(len(tc),np.nan),where=tc>0)
            ratio.stairs(values,bins,color=COLORS[name],lw=1.2);allrat.append(values)
            leakage[k][name+'_outside']=int(np.sum((predictions[name][k]<bins[0])|(predictions[name][k]>bins[-1])))
        ax.set_ylabel('Density');ax.tick_params(labelbottom=False);ax.set_title(LABELS[j]+(' [GeV]' if j==0 else ''))
        ratio.set_xlabel(LABELS[j]+(' [GeV]' if j==0 else ''));ratio.set_ylabel('Pred./truth');ratio.axhline(1,color='#777777',ls='--',lw=.8)
        ratio.set_ylim(0,max(1.6,float(np.nanmax(allrat))*1.05) if j==0 else 1.6)
        ratio.yaxis.set_major_locator(MaxNLocator(3));ax.yaxis.set_major_locator(MaxNLocator(5));ratio.set_xlim(bins[0],bins[-1])
    handles,labels=ax.get_legend_handles_labels()
    fig.suptitle('Selected validation distributions on common MC truth',y=.99,fontsize=16)
    fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.5,.958),ncol=3,frameon=False,fontsize=9.5)
    fig.text(.5,.013,NOTE+'\nDensities use all events; displayed ranges omit some events. No prediction clipping or uncertainty bands.',ha='center',fontsize=9,color='#444444')
    fig.subplots_adjust(top=.85,bottom=.095,left=.075,right=.98)
    save(fig,'06_selected_validation_distributions')
    (OUT/'plot_range_counts.json').write_text(json.dumps(leakage,indent=2)+'\n')

RESPONSE_BINS={'ttbar_mass':np.array([300,380,440,500,750,2000]),'delta_abs_y':np.array([-2.4,-1,-.5,0,.5,1,2.4]),'cos_K_plus':np.array([-1,-.5,0,.5,1]),'cos_phi':np.array([-1,-.5,0,.5,1])}
def migration(keys,stem):
    fig,axes=plt.subplots(2,4,figsize=(17,9.0))
    matrices={}
    for row,k in enumerate(keys):
        b=RESPONSE_BINS[k];den=np.histogram(truth[k],b)[0]
        matrices[k]={}
        for col,name in enumerate(SELECTED):
            ax=axes[row,col];hist=np.histogram2d(truth[k],predictions[name][k],bins=(b,b))[0]
            z=np.divide(hist,den[:,None],out=np.zeros_like(hist),where=den[:,None]>0)
            matrices[k][name]=z.tolist()
            image=ax.imshow(z,origin='lower',vmin=0,vmax=1,cmap='Reds',extent=(0,len(b)-1,0,len(b)-1),aspect='auto')
            for iy,ix in np.ndindex(z.shape):
                if z[iy,ix]>=.005:ax.text(ix+.5,iy+.5,f'{z[iy,ix]:.2f}',ha='center',va='center',fontsize=8.2,color='white' if z[iy,ix]>.55 else '#222222')
            ax.set_xticks(range(len(b)),[f'{v:g}' for v in b],rotation=40);ax.set_yticks(range(len(b)),[f'{v:g}' for v in b]);ax.grid(False)
            ax.set_xlabel('Predicted '+LABELS[KEYS.index(k)]+(' [GeV]' if k=='ttbar_mass' else ''))
            if col==0:ax.set_ylabel('MC truth '+LABELS[KEYS.index(k)]+(' [GeV]' if k=='ttbar_mass' else ''))
            if row==0:ax.set_title(SLAB[name].replace(' (new)',''),fontsize=10)
    cax=fig.add_axes([.93,.16,.012,.68]);fig.colorbar(image,cax=cax,label='Probability per truth bin')
    fig.suptitle('Validation response on the same events',y=.99,fontsize=16)
    fig.text(.5,.025,NOTE+'\nRows are divided by all events in the truth bin, including predictions outside the plotted range. Unequal bin widths are shown as equal cells.',ha='center',fontsize=9,color='#444444')
    fig.subplots_adjust(top=.91,bottom=.145,left=.065,right=.91,hspace=.38,wspace=.22)
    save(fig,stem)
    (OUT/(stem+'.json')).write_text(json.dumps(matrices,indent=2)+'\n')

paired('mae','01_eventwise_mae_all26','Event-wise prediction error for all 26 configurations')
paired('w1','02_distribution_w1_all26','Marginal distribution agreement for all 26 configurations')
effects('mae','03_hyperparameter_effects_mae','Effect of one-factor changes on MAE')
effects('w1','04_hyperparameter_effects_w1','Effect of one-factor changes on W1')
tradeoffs()
distributions()
migration(['ttbar_mass','cos_phi'],'07_response_mass_cosphi')
migration(['delta_abs_y','cos_K_plus'],'08_response_rapidity_helicity')

best={k:{metric:{'run_id':min([c['run_id'] for c in plan['runs']],key=lambda n:metrics[n][k][metric]),'value':min(metrics[c['run_id']][k][metric] for c in plan['runs'])} for metric in ['mae','w1']} for k in KEYS}
(OUT/'best_common_truth.json').write_text(json.dumps(best,indent=2)+'\n')
readme='''第一轮 Transformer 与 ParT 对比图

01：26组配置的逐事件 MAE；越小越好。
02：26组配置的边缘分布 W1；越小越好。
03、04：相对各自架构基线的百分比改善；正值更好。
05：MAE与W1的权衡；越靠左下越好，虚线为本轮非支配配置。
06：选定的新旧 EMD 模型的四变量分布及预测/真值比。
07、08：同一批事件的迁移矩阵；横轴预测，纵轴真值。

每张图均提供 PNG 及矢量 PDF。全部 26 组和两个历史基线均在同一份 canonical truth 上重新评估，所有260190个事件按原顺序保留，预测不裁剪。
不同观测量的最优值可能来自不同配置；不表示一个模型同时实现全部最优。
仅验证集、单种子、按验证损失选择checkpoint。新模型每组GPU分配上限30分钟，实际约22分钟；历史训练预算不同，不能视为严格的架构单因素消融，也没有统计显著性结论。
warmup+cosine两组在固定20 epoch后学习率归零；后续执行的epoch不代表继续有效更新。

metrics_common_truth.json保存完整重新评估指标、文件SHA256和逐行truth对齐误差；plot_range_counts.json保存图外事件计数。
运行源码位于上一级plot_round1.py。原始下载为successful_predictions_26.zip，results内保留预测和溯源。
'''
(OUT/'说明.txt').write_text(readme,encoding='utf-8-sig')
with zipfile.ZipFile(HERE/'round1_figures.zip','w',zipfile.ZIP_DEFLATED) as archive:
    for path in sorted(OUT.iterdir()):
        if path.is_file():archive.write(path,path.name)
print('All figures and round1_figures.zip saved.',flush=True)
