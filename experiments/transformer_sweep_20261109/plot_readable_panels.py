"""Readable per-observable distribution and response panels from existing predictions."""
import sys
from pathlib import Path
import json
import zipfile
import numpy as np
import awkward as ak
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
OUT=HERE/'figures_readable'
OUT.mkdir(exist_ok=True)
report=json.loads((HERE/'figures/metrics_common_truth.json').read_text())
METRICS=report['metrics']
KEYS=['ttbar_mass','delta_abs_y','cos_K_plus','cos_phi']
LABELS=[r'$m_{t\bar t}$ [GeV]',r'$\Delta|y|$',r'$\cos\theta_k^+$',r'$\cos\varphi$']
TITLES=['Top-pair invariant mass','Difference in absolute rapidities','Positive-lepton helicity angle','Lepton opening angle']
STEMS=['mass','rapidity','helicity','opening_angle']
MODELS=['historical_mse','historical_emd','regression_emd01','part_emd01']
TAGS={'historical_mse':'Previous MSE','historical_emd':'Previous MSE + 0.1 EMD','regression_emd01':'New Transformer + 0.1 EMD','part_emd01':'New ParT + 0.1 EMD'}
COLORS={'historical_mse':'#bc3940','historical_emd':'#315faa','regression_emd01':'#8d5eaa','part_emd01':'#128570'}
STYLES={'historical_mse':'-','historical_emd':'--','regression_emd01':'-.','part_emd01':':'}
BINS={'ttbar_mass':np.linspace(300,1050,31),'delta_abs_y':np.linspace(-2.4,2.4,25),'cos_K_plus':np.linspace(-1,1,21),'cos_phi':np.linspace(-1,1,21)}
RBINS={'ttbar_mass':np.array([300,380,440,500,750,2000]),'delta_abs_y':np.array([-2.4,-1,-.5,0,.5,1,2.4]),'cos_K_plus':np.array([-1,-.5,0,.5,1]),'cos_phi':np.array([-1,-.5,0,.5,1])}
N=260190
canonical=ak.from_parquet(ROOT/'本轮新结果_20261109/02_评估结果/four_observables.parquet')
truth={k:ak.to_numpy(canonical.targets[k]).astype(float) for k in KEYS}
pred={}
for name in MODELS:
    path=Path(report['provenance'][name]['path'])
    data=ak.from_parquet(path)
    assert len(data)==N
    pred[name]={k:ak.to_numpy(data.predictions[k]).astype(float) for k in KEYS}

plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.labelsize':12,'axes.titlesize':13,'axes.spines.top':False,'axes.spines.right':False,
    'axes.grid':True,'grid.alpha':.17,'grid.linewidth':.6,'savefig.facecolor':'white','pdf.fonttype':42})
NOTE='Validation N = 260,190 | same MC truth | one seed | historical training budgets differ'

def save(fig,stem):
    for ext in ['png','pdf']:
        fig.savefig(OUT/(stem+'.'+ext),dpi=200,bbox_inches='tight')
    plt.close(fig)

def distribution(k,j):
    fig=plt.figure(figsize=(13.6,11.8))
    outer=fig.add_gridspec(2,2,hspace=.46,wspace=.20,left=.085,right=.975,bottom=.125,top=.855)
    b=BINS[k];width=np.diff(b);tc=np.histogram(truth[k],b)[0]
    counts={name:np.histogram(pred[name][k],b)[0] for name in MODELS}
    ymax=1.12*max([np.max(tc/N/width)]+[np.max(c/N/width) for c in counts.values()])
    ratios={name:np.divide(c,tc,out=np.full(len(tc),np.nan),where=tc>0) for name,c in counts.items()}
    ratio_max=max(1.6,max(float(np.nanmax(x)) for x in ratios.values())*1.10)
    for i,name in enumerate(MODELS):
        gs=outer[i//2,i%2].subgridspec(2,1,height_ratios=[3,1],hspace=.06)
        ax=fig.add_subplot(gs[0]);rx=fig.add_subplot(gs[1],sharex=ax)
        ax.stairs(tc/N/width,b,fill=True,color='#dfdfdf',zorder=1)
        ax.stairs(tc/N/width,b,color='#666666',lw=.8,zorder=2)
        ax.stairs(counts[name]/N/width,b,color=COLORS[name],ls=STYLES[name],lw=2.0,zorder=3)
        m=METRICS[name][k]
        fmt='.2f' if j==0 else '.4f'
        ax.set_title(TAGS[name]+'\nMAE = '+format(m['mae'],fmt)+'  |  W1 = '+format(m['w1'],fmt),pad=13,fontsize=12)
        ax.set_ylim(0,ymax);ax.set_ylabel('Density');ax.tick_params(labelbottom=False)
        ax.yaxis.set_major_locator(MaxNLocator(5))
        rx.stairs(ratios[name],b,color=COLORS[name],ls=STYLES[name],lw=1.9,baseline=None)
        rx.axhline(1,color='#555555',ls='--',lw=.9)
        rx.set_ylim(0,ratio_max);rx.set_xlim(b[0],b[-1]);rx.set_ylabel('Pred./truth',fontsize=10)
        rx.set_xlabel(LABELS[j]);rx.yaxis.set_major_locator(MaxNLocator(3))
    fig.suptitle(TITLES[j]+' — each model shown separately',y=.985,fontsize=17)
    fig.legend(handles=[Patch(facecolor='#dfdfdf',edgecolor='#666666',label='MC parton truth'),Line2D([],[],color='#444444',lw=2,label='One prediction curve per panel')],
               loc='upper center',bbox_to_anchor=(.5,.953),ncol=2,frameon=False,fontsize=11)
    unit='Mass MAE and W1 are in GeV. ' if j==0 else 'MAE and W1 are dimensionless. '
    fig.text(.5,.035,NOTE+'\n'+unit+'All metrics use the full sample; density and ratio axes are identical across panels.\nDisplayed ranges omit some events; normalization uses all events. No clipping or uncertainty bands.',ha='center',fontsize=9.5,color='#444444')
    save(fig,f'{j+1:02d}_distribution_{STEMS[j]}')

def response(k,j):
    fig,axes=plt.subplots(2,2,figsize=(12.4,11.5))
    b=RBINS[k];nb=len(b)-1;den=np.histogram(truth[k],b)[0]
    for ax,name in zip(axes.ravel(),MODELS):
        count=np.histogram2d(truth[k],pred[name][k],bins=(b,b))[0]
        z=np.divide(count,den[:,None],out=np.zeros_like(count),where=den[:,None]>0)
        image=ax.imshow(z,origin='lower',vmin=0,vmax=1,cmap='Reds',extent=(0,nb,0,nb),aspect='equal')
        for iy,ix in np.ndindex(z.shape):
            if z[iy,ix]>=.005:
                ax.text(ix+.5,iy+.5,f'{z[iy,ix]:.2f}',ha='center',va='center',fontsize=12.5,color='white' if z[iy,ix]>.55 else '#222222')
        ax.set_xticks(range(len(b)),[f'{v:g}' for v in b],rotation=35 if j==0 else 0)
        ax.set_yticks(range(len(b)),[f'{v:g}' for v in b]);ax.grid(False)
        ax.tick_params(labelsize=10.5)
        ax.set_xlabel('Predicted '+LABELS[j],labelpad=7)
        ax.set_ylabel('MC truth '+LABELS[j],labelpad=7)
        ax.set_title(TAGS[name],fontsize=12.5,pad=10)
    fig.suptitle(TITLES[j]+' — validation migration matrices',y=.985,fontsize=17)
    cax=fig.add_axes([.91,.18,.018,.65]);cb=fig.colorbar(image,cax=cax);cb.set_label('Probability per truth bin',fontsize=11)
    fig.text(.5,.025,NOTE+'\nEach row includes all events in its truth bin; predictions outside the displayed range are included in the denominator.\nUnequal physical bin widths are displayed as equal cells. Values below 0.005 are left blank.',ha='center',fontsize=9.3,color='#444444')
    fig.subplots_adjust(top=.90,bottom=.14,left=.09,right=.87,hspace=.38,wspace=.34)
    save(fig,f'{j+5:02d}_response_{STEMS[j]}')

for j,k in enumerate(KEYS):
    distribution(k,j)
    response(k,j)
    print('Saved readable distribution and response:',k,flush=True)

conclusions='''清晰版分布图与迁移矩阵

01–04：四个变量各一张分布图。四种模型分面展示，每个面板只有一条预测曲线与灰色真值，附预测/真值比；同变量四面板使用相同坐标范围和bin。
05–08：四个变量各一张迁移矩阵。四种模型按2×2排列；横轴预测，纵轴真值，颜色范围统一0–1。
各图同时提供PNG和矢量PDF。

结论（共同真值重算，所有260190个验证事件保留）：
1. 没有全面最优模型。相同参数的ParT baseline仅在质量MAE上略好于普通Transformer，其他MAE和全部W1略差。
2. EMD是改善边缘分布最明确的因素：普通Transformer加0.1EMD后，四项W1相对本轮自身baseline降低约33%、21%、30%、40%。这不等于所有逐事件误差都改善。
3. ParT+0.1EMD质量MAE为46.31923GeV，旧MSE为46.53418GeV，改善仅0.46%。其他三个变量的本轮最佳MAE仍未超过旧基线。
4. 与旧MSE+EMD相比，ParT+0.1EMD的helicity W1从0.052865降至0.051139，改善3.27%；但质量W1由5.37582升至11.34582GeV。旧MSE+EMD仍是很强的综合参照。
5. 下一轮可优先复核lr=1e-3、ParT batch1024、EMD0.05/0.1。收益有指标取舍。d_model64两架构均较差，6层收益混合。当前cosine设定在20epoch后归零，不能据此判断调度方法无用。

限制：单个随机种子、验证集选择checkpoint；尚未进行独立测试集评估。历史模型训练预算不同，小幅提升没有统计显著性结论。
'''
(OUT/'说明及结论.txt').write_text(conclusions,encoding='utf-8-sig')
with zipfile.ZipFile(HERE/'round1_readable_distributions_and_matrices.zip','w',zipfile.ZIP_DEFLATED) as archive:
    for p in sorted(OUT.iterdir()):
        if p.is_file():archive.write(p,p.name)
    archive.write(HERE/'figures/metrics_common_truth.json','metrics_common_truth.json')
print('Saved clear figures and ZIP.',flush=True)
