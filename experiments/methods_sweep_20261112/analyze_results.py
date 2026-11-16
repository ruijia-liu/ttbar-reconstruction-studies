"""Recompute full-validation metrics from downloaded predictions, without fitting."""
import argparse,json,sys
from pathlib import Path
import awkward as ak
import numpy as np
from scipy.stats import wasserstein_distance
from scipy.spatial import cKDTree
from evaluation import KEYS,BINS,digest,diagnostics,save

HERE=Path(__file__).resolve().parent

def independently_evaluate(pred,truth):
    result={}
    for j,key in enumerate(KEYS):
        edges=np.asarray(BINS[key]);n=len(edges)-1
        t=np.searchsorted(edges,truth[:,j],side='right')-1
        p=np.searchsorted(edges,pred[:,j],side='right')-1
        t[truth[:,j]==edges[-1]]=n-1;p[pred[:,j]==edges[-1]]=n-1
        valid_t=(t>=0)&(t<n);valid_p=(p>=0)&(p<n)
        counts=np.bincount(t[valid_t],minlength=n)
        mask=valid_t&valid_p
        matrix=np.bincount(t[mask]*n+p[mask],minlength=n*n).reshape(n,n)/counts[:,None]
        error=pred[:,j]-truth[:,j]
        result[key]=dict(res=float(np.diff(np.percentile(error,[15.87,84.13]))[0]),
            local_trace=float(np.diag(matrix).mean()),w1=float(wasserstein_distance(pred[:,j],truth[:,j])),
            row_probabilities=matrix.tolist(),truth_bin_counts=counts.tolist())
    return result

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--bootstrap',type=int,default=200);args=parser.parse_args()
    plan=json.loads((HERE/'experiment_plan.json').read_text());root=HERE/'results'
    assert digest(HERE/'canonical_validation.npz')==plan['canonical_npz_sha256']
    canon=np.load(HERE/'canonical_validation.npz');truth=canon['targets']
    assert np.array_equal(canon['row_index'],np.arange(len(truth))) and list(canon['keys'])==KEYS
    basefile=HERE.parent/'loss_sweep_20261110/results/huber1_emd01/predictions_val.parquet'
    assert digest(basefile)==plan['base_artifacts']['predictions']['sha256']
    baseline=ak.from_parquet(basefile)
    basepred=np.column_stack([ak.to_numpy(baseline.predictions[k]).astype(float) for k in KEYS])
    basetruth=np.column_stack([ak.to_numpy(baseline.targets[k]).astype(float) for k in KEYS])
    tolerance=np.array([5e-4,5e-7,5e-7,5e-7])
    assert np.all(np.max(abs(basetruth-truth),axis=0)<=tolerance)
    assert np.all(cKDTree(truth/tolerance).query_ball_point(basetruth/tolerance,1.,p=np.inf,return_length=True)==1)
    pred={'huber_emd_baseline':basepred};verified={};profiles={};rows=[]
    baseline_metrics=independently_evaluate(basepred,truth)
    for k in KEYS:
        for metric in ['res','local_trace','w1']:
            assert abs(baseline_metrics[k][metric]-plan['baseline_metrics']['observables'][k][metric])<1e-10
    baseline_diag=diagnostics(basepred,truth)
    for cfg in plan['runs']:
        name=cfg['run_id'];folder=root/name;m=json.loads((folder/'metrics.json').read_text())
        status=json.loads((folder/'status.json').read_text())
        assert status['status']=='complete' and status['completed_full_epochs']==200 and not status['partial_final_epoch']
        path=folder/'predictions_val.parquet';assert digest(path)==m['predictions']['sha256']==status['primary_prediction_sha256']
        data=ak.from_parquet(path);assert len(data)==len(truth)==260190
        assert np.array_equal(ak.to_numpy(data.row_index),canon['row_index'])
        target=np.column_stack([ak.to_numpy(data.targets[k]) for k in KEYS])
        assert np.array_equal(target,truth)
        output=np.column_stack([ak.to_numpy(data.predictions[k]) for k in KEYS]);assert np.isfinite(output).all()
        actual=independently_evaluate(output,truth)
        for k in KEYS:
            for metric in ['res','local_trace','w1','row_probabilities']:
                assert np.allclose(actual[k][metric],m['observables'][k][metric],rtol=0,atol=1e-10),(name,k,metric)
        pred[name]=output;verified[name]=actual
        diag=diagnostics(output,truth)
        changes={k:dict(res_improvement_pct=100*(1-actual[k]['res']/baseline_metrics[k]['res']),
                       trace_improvement_pp=100*(actual[k]['local_trace']-baseline_metrics[k]['local_trace']),
                       w1_improvement_pct=100*(1-actual[k]['w1']/baseline_metrics[k]['w1'])) for k in KEYS}
        shape_gain=100*(1-np.exp(np.mean([np.log(actual[k]['w1']/baseline_metrics[k]['w1']) for k in KEYS])))
        eligible=all(v['res_improvement_pct']>=-1e-10 and v['trace_improvement_pp']>=-1e-10 for v in changes.values())
        profiles[name]=dict(changes=changes,diagnostics=diag,illegal_cos_counts=(abs(output[:,2:])>1).sum(0).tolist(),
            illegal_cos_events=int((abs(output[:,2:])>1).any(1).sum()),
            tail_mass_res_change_pct={threshold:100*(diag['mass_tails'][threshold]['res'][0]/baseline_diag['mass_tails'][threshold]['res'][0]-1) for threshold in ['800','1000']},
            bin_variant_trace_change_pp={k:{label:100*(diag['bin_variants'][k][label]['local_trace']-baseline_diag['bin_variants'][k][label]['local_trace']) for label in ['fine','shifted']} for k in KEYS})
        rows.append(dict(run=name,eligible=eligible,selected_epoch_zero_based=m['checkpoint']['epoch'],
            mean_res_improvement_pct=float(np.mean([v['res_improvement_pct'] for v in changes.values()])),
            mean_trace_improvement_pp=float(np.mean([v['trace_improvement_pp'] for v in changes.values()])),
            geometric_w1_improvement_pct=float(shape_gain),all12_improve=all(min(v.values())>0 for v in changes.values()),
            w1_improved_observables=sum(v['w1_improvement_pct']>0 for v in changes.values())))
    ranking=sorted(rows,key=lambda r:(not r['eligible'],-r['geometric_w1_improvement_pct']))
    # Conditional, paired validation bootstrap at already selected checkpoints.
    # These intervals do not correct checkpoint/method selection or model-seed uncertainty.
    boot={name:{k:{metric:[] for metric in ['res_improvement_pct','trace_improvement_pp','w1_improvement_pct']} for k in KEYS}
          for name in ['conditional_ot','conditional_ot_proxy']}
    rng=np.random.default_rng(20261112)
    for b in range(args.bootstrap):
        idx=rng.integers(0,len(truth),len(truth));bt=truth[idx]
        ref=independently_evaluate(basepred[idx],bt)
        for name in boot:
            current=independently_evaluate(pred[name][idx],bt)
            for k in KEYS:
                boot[name][k]['res_improvement_pct'].append(100*(1-current[k]['res']/ref[k]['res']))
                boot[name][k]['trace_improvement_pp'].append(100*(current[k]['local_trace']-ref[k]['local_trace']))
                boot[name][k]['w1_improvement_pct'].append(100*(1-current[k]['w1']/ref[k]['w1']))
        if (b+1)%50==0:print('Paired validation bootstrap',b+1,'/',args.bootstrap,flush=True)
    intervals={name:{k:{metric:np.quantile(values,[.025,.975]).tolist() for metric,values in bymetric.items()}
                       for k,bymetric in bykey.items()} for name,bykey in boot.items()} if args.bootstrap else {}
    report=dict(events=len(truth),baseline=baseline_metrics,baseline_diagnostics=baseline_diag,
        verified_metrics=verified,ranked_summary=ranking,profiles=profiles,
        bootstrap=dict(replicates=args.bootstrap,seed=20261112,intervals_95=intervals,
            limitation='Paired resampling of validation events conditional on selected checkpoints; no correction for selection or seed uncertainty.'),
        scope='Full validation, single seed, frozen encoder correction heads; no independent test evaluation.',
        baseline_prediction_sha256=digest(basefile),canonical_sha256=digest(HERE/'canonical_validation.npz'))
    save(root/'comparison_verified.json',report)
    lines=['Six correction methods vs original Huber + 0.1 EMD','Full validation: 260190 events; all metric arrays independently verified.',
           'Ranking: first require all 8 Res/Trace nonworse; then minimize geometric mean W1 ratios.',
           'Resolution is Q84.13(error)-Q15.87(error); Trace is macro diagonal; smaller Res/W1 and larger Trace are better.','']
    for row in ranking:
        name=row['run'];lines.append(f"{name}: mean Res improvement {row['mean_res_improvement_pct']:.4f}%; mean Trace gain {row['mean_trace_improvement_pp']:.4f} pp; geometric W1 improvement {row['geometric_w1_improvement_pct']:.4f}%; W1 improved {row['w1_improved_observables']}/4")
        for k in KEYS:lines.append('  '+k+': '+json.dumps(profiles[name]['changes'][k]))
        lines.append('  Mass-tail Res change: '+json.dumps(profiles[name]['tail_mass_res_change_pct']))
        lines.append('  Cosine overflow counts (K, phi): '+str(profiles[name]['illegal_cos_counts']))
    lines+=['','These are validation-selected candidates, not independent test performance or superiority over the paper.']
    (root/'comparison_report.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(ranking,indent=2))
    print('Verified report:',root/'comparison_verified.json')

if __name__=='__main__':main()
