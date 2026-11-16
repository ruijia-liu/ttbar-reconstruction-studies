"""Audit same-event, shared-alpha interpolation of existing validation predictions.

This script performs no fitting, sampling, clipping, event selection or SSH.
It writes only mixture_scan_20261112/results and verifies both stored endpoints.
"""
from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = HERE/'results'
sys.path.insert(0, str(ROOT/'paper_target_20261111'))

import awkward as ak
import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import wasserstein_distance
from audit_existing import BINS, KEYS, migration, resolution, self_check

ALPHAS = [0., .1, .2, .35, .5]
ATOL = 1e-12
NROWS = 260190
METRICS = ['res', 'local_trace', 'w1', 'mae', 'rmse', 'bias']


def require(condition, message):
    if not bool(condition):
        raise ValueError(message)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def load_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write_json(path, value):
    temporary = Path(path).with_suffix('.json.tmp')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    temporary.replace(path)


def read_prediction(path, expected_hash, canonical, require_row_id):
    file_hash = digest(path)
    require(file_hash == expected_hash, 'Prediction hash differs: '+str(path))
    data = ak.from_parquet(path)
    require(len(data) == NROWS, 'Incomplete validation export: '+str(path))
    target = np.column_stack([ak.to_numpy(data.targets[key]).astype(np.float64) for key in KEYS])
    prediction = np.column_stack([ak.to_numpy(data.predictions[key]).astype(np.float64) for key in KEYS])
    require(target.shape == prediction.shape == canonical.shape, 'Array shape mismatch')
    require(np.isfinite(target).all() and np.isfinite(prediction).all(), 'Nonfinite input values')
    row_id = 'row_index' in data.fields
    require(not require_row_id or row_id, 'Flow export must have explicit row_index')
    if row_id:
        require(np.array_equal(ak.to_numpy(data.row_index), np.arange(NROWS)), 'Noncanonical row_index')
    if 'valid' in data.fields:
        require(all(bool(ak.all(data.valid[key])) for key in KEYS), 'Invalid prediction flags')
    differences = np.abs(target-canonical).max(axis=0)
    exact = np.array_equal(target, canonical)
    tolerance = np.zeros(4)
    unique_tolerant_match = None
    if not exact:
        require(not row_id, 'Row-indexed export must store canonical truth exactly')
        tolerance = np.array([5e-4,5e-7,5e-7,5e-7])
        require(np.all(differences <= tolerance), 'Legacy targets do not align row by row')
        # No reordering: every stored target must match its corresponding row
        # and exactly one canonical target within the documented old precision.
        counts = cKDTree(canonical/tolerance).query_ball_point(target/tolerance,r=1.,p=np.inf,return_length=True)
        require(np.all(counts == 1), 'Ambiguous or missing legacy coordinate identification')
        unique_tolerant_match = True
    return prediction, dict(path=str(path),sha256=file_hash,bytes=path.stat().st_size,
                            events=len(data),explicit_row_index_verified=row_id,
                            stored_truth_exact=exact,ordered_truth_maxabs=dict(zip(KEYS,differences.tolist())),
                            ordered_truth_tolerance=dict(zip(KEYS,tolerance.tolist())),
                            unique_tolerant_coordinate_match=unique_tolerant_match)


def illegal_counts(prediction):
    p = prediction
    return dict(nonfinite=int((~np.isfinite(p)).sum()),mass_nonpositive=int((p[:,0] <= 0).sum()),
                cos_K_plus_outside=int((np.abs(p[:,2]) > 1).sum()),
                cos_phi_outside=int((np.abs(p[:,3]) > 1).sum()),
                events_any_cosine_outside=int((np.abs(p[:,2:]) > 1).any(axis=1).sum()))


def evaluate(prediction, truth):
    require(prediction.dtype == np.float64 and prediction.shape == (NROWS,4), 'Expected full float64 predictions')
    require(np.isfinite(prediction).all(), 'Nonfinite mixtures cannot be omitted')
    profile = {}
    for i,key in enumerate(KEYS):
        residual = prediction[:,i]-truth[:,i]
        response = migration(truth[:,i],prediction[:,i],BINS[key])
        profile[key] = dict(n=NROWS,**resolution(residual),local_trace=response['local_trace'],
                            w1=float(wasserstein_distance(prediction[:,i],truth[:,i])),
                            mae=float(np.abs(residual).mean()),rmse=float(np.sqrt(np.mean(residual**2))),
                            bias=float(residual.mean()),migration=response)
    tails = {}
    for threshold in [800,1000]:
        mask = truth[:,0] > threshold
        profiles = {}
        for i,key in enumerate(KEYS):
            residual = prediction[mask,i]-truth[mask,i]
            profiles[key] = dict(n=int(mask.sum()),**resolution(residual),
                                w1=float(wasserstein_distance(prediction[mask,i],truth[mask,i])),
                                mae=float(np.abs(residual).mean()),rmse=float(np.sqrt(np.mean(residual**2))),
                                bias=float(residual.mean()))
        tails['truth_mass_gt_'+str(threshold)] = profiles
    return dict(events=NROWS,observables=profile,illegal_output_counts=illegal_counts(prediction),mass_slices=tails)


def compare(profile, baseline):
    differences = {};checks8 = {};checks4 = {}
    for key in KEYS:
        p,b = profile['observables'][key],baseline['observables'][key]
        differences[key] = {metric:float(p[metric]-b[metric]) for metric in METRICS}
        differences[key].update(res_percent=100*(p['res']/b['res']-1),
                                local_trace_percentage_points=100*(p['local_trace']-b['local_trace']),
                                w1_percent=100*(p['w1']/b['w1']-1))
        checks8[key+'/res'] = bool(p['res'] <= b['res']+ATOL)
        checks8[key+'/local_trace'] = bool(p['local_trace'] >= b['local_trace']-ATOL)
        checks4[key+'/w1'] = bool(p['w1'] < b['w1']-ATOL)
    return dict(delta_from_baseline=differences,res_trace_checks=checks8,w1_checks=checks4,
                strict_all8_preserved=all(checks8.values()),strict_all4_w1_improved=all(checks4.values()),
                all12_conditions_met=all(checks8.values()) and all(checks4.values()),
                preserved_res_trace_count=sum(checks8.values()),improved_w1_count=sum(checks4.values()))


def verify_profile(actual, expected, metrics, label):
    maximum = 0.
    for key in KEYS:
        for metric in metrics:
            difference = abs(actual['observables'][key][metric]-expected[key][metric])
            maximum = max(maximum,difference)
            require(difference <= 1e-10, f'{label} stored metric mismatch: {key}/{metric}, {difference}')
    return dict(verified=True,absolute_tolerance=1e-10,max_absolute_difference=maximum,metrics=metrics)


def draw_figure(candidates):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    labels = [r'$m_{t\bar t}$',r'$\Delta|y|$',r'$\cos\theta_k^+$',r'$\cos\varphi$']
    titles = ['Res change (%)\nLower is better','Local Trace change (percentage points)\nHigher is better','W1 change (%)\nLower is better']
    fields = ['res_percent','local_trace_percentage_points','w1_percent']
    colors = ['#1d4ed8','#0f766e','#7c3aed','#b45309']
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.titlesize':13,'axes.labelsize':12,
                         'xtick.labelsize':10,'ytick.labelsize':10,'savefig.facecolor':'white'})
    fig,axes = plt.subplots(4,3,figsize=(14,11),sharex=True,sharey='col')
    for row,key in enumerate(KEYS):
        for col,field in enumerate(fields):
            ax = axes[row,col]
            values = [candidate['comparison']['delta_from_baseline'][key][field] for candidate in candidates]
            ax.plot(ALPHAS,values,'o-',color=colors[row],lw=2.1,ms=5.5)
            ax.axhline(0,color='#334155',lw=1,ls='--')
            ax.grid(alpha=.2)
            ax.set_xlim(-.015,.515)
            ax.set_xticks(ALPHAS,['0','0.1','0.2','0.35','0.5'])
            if row == 0:
                ax.set_title(titles[col],pad=13)
            if col == 0:
                ax.set_ylabel(labels[row],fontsize=17,labelpad=13)
            if row == 3:
                ax.set_xlabel(r'Shared interpolation weight $\alpha$')
    for col,field in enumerate(fields):
        values = [candidate['comparison']['delta_from_baseline'][key][field] for candidate in candidates for key in KEYS]+[0]
        low,high = min(values),max(values)
        span = max(high-low,1e-3)
        axes[0,col].set_ylim(low-.12*span,high+.12*span)
    fig.suptitle('Same-event interpolation: Huber+EMD regression + flow single draw',fontsize=18,y=.988)
    fig.text(.5,.951,r'$\hat y_\alpha=(1-\alpha)\hat y_{\rm Huber+EMD}+\alpha\hat y_{\rm flow}$',ha='center',fontsize=15)
    fig.text(.5,.014,'260,190 validation events | one shared alpha for all four observables | no clipping\n'
             'Identical y-scale within each metric column. Alpha = 1 is a report-only endpoint reference. Trace uses local bins.',
             ha='center',fontsize=10,color='#475569')
    fig.tight_layout(rect=(.01,.055,.995,.925),h_pad=2.0,w_pad=2.4)
    fig.savefig(OUT/'mixture_metric_changes.png',dpi=180)
    fig.savefig(OUT/'mixture_metric_changes.pdf')
    plt.close(fig)


def write_text(report):
    candidates = report['candidates']
    lines = ['已有验证集预测的同事件混合扫描','',
             '公式：prediction_alpha = (1-alpha) * Huber+EMD + alpha * fixed flow single draw',
             '扫描 alpha = 0, 0.1, 0.2, 0.35, 0.5；alpha = 1 仅作原 Flow 端点参考。',
             '全部260,190事件；四维共享alpha；float64运算；不裁剪、不重排、不挑选事件；没有新训练。',
             'Res = Q84.13(error) - Q15.87(error)，为完整宽度；误差=预测-真值。',
             'Local Trace = 真值分箱归一化迁移矩阵的对角均值；预测越界仍留在真值分箱分母。',
             '基准为现有Huber(delta=1)+0.1EMD；严格保留指4项Res<=基准+1e-12且4项Trace>=基准-1e-12。',
             '四项W1改善指每项W1<基准-1e-12；没有通过平均分抵消失败项。','',
             'alpha  Res/Trace通过项  W1改善项  全8保留  全4W1改善  满足全部12项  余弦越界事件']
    for candidate in candidates:
        c = candidate['comparison'];p=candidate['metrics']
        label = f"{candidate['alpha']:g}"+(' [reference]' if candidate['role']=='flow_reference' else '')
        lines.append(f"{label:<15} {c['preserved_res_trace_count']}/8       {c['improved_w1_count']}/4       "
                     f"{str(c['strict_all8_preserved']):<7} {str(c['strict_all4_w1_improved']):<10} "
                     f"{str(c['all12_conditions_met']):<12} {p['illegal_output_counts']['events_any_cosine_outside']}")
    passed = [c['alpha'] for c in candidates if c['role']!='flow_reference' and c['comparison']['all12_conditions_met']]
    lines += ['', '扫描中满足全部12项的alpha：'+(str(passed) if passed else '无。'),
              '所有判断均为本次验证集上的数值比较；未计算置信区间，不能声称统计显著或测试集泛化。',
              'Trace为本地分箱，论文分箱/溢出规则尚未完全核实；这里不声明超过论文。']
    for candidate in candidates:
        lines += ['',f"alpha = {candidate['alpha']:g} ({candidate['role']})",
                  'observable           Res          Trace        W1            dRes(%)    dTrace(pp)    dW1(%)']
        for key in KEYS:
            metric = candidate['metrics']['observables'][key]
            delta = candidate['comparison']['delta_from_baseline'][key]
            lines.append(f"{key:<20} {metric['res']:11.7f} {metric['local_trace']:11.7f} {metric['w1']:12.7f} "
                         f"{delta['res_percent']:+10.5f} {delta['local_trace_percentage_points']:+12.5f} {delta['w1_percent']:+10.5f}")
        lines.append('物理越界计数：'+json.dumps(candidate['metrics']['illegal_output_counts'],ensure_ascii=False))
    lines += ['', '端点复算与原指标、输入SHA256及逐事件对齐详见 mixture_results.json。',
              'MAE、RMSE、bias及真值mass>800/1000子样本指标保存在JSON；尾部不额外定义Trace。']
    (OUT/'mixture_summary.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def main():
    checks = self_check()
    probability_root = ROOT/'probabilistic_sweep_20261111_retry2'
    flow_folder = probability_root/'results/flow'
    baseline_folder = ROOT/'loss_sweep_20261110/results/huber1_emd01'
    plan_path = probability_root/'experiment_plan.json'
    plan = load_json(plan_path)
    canonical_path = probability_root/'canonical_validation.npz'
    canonical_hash = digest(canonical_path)
    require(canonical_hash == plan['canonical_npz_sha256'], 'Canonical file hash differs from the fixed experiment plan')
    with np.load(canonical_path,allow_pickle=False) as data:
        require(data['keys'].tolist() == KEYS, 'Canonical observable order differs')
        require(np.array_equal(data['row_index'],np.arange(NROWS)), 'Canonical row order differs')
        truth = np.array(data['targets'],dtype=np.float64,copy=True)
    require(truth.shape == (NROWS,4) and np.isfinite(truth).all(), 'Invalid canonical truth')
    statuses = {name:load_json(folder/'status.json') for name,folder in [('baseline',baseline_folder),('flow',flow_folder)]}
    for name,status in statuses.items():
        require(status['status'] == 'complete' and not status.get('smoke'), 'Endpoint not a completed production run: '+name)
        require(status['prediction_rows'] == NROWS, 'Endpoint does not contain full validation predictions')
        require(status['canonical_truth_sha256'] == canonical_hash, 'Endpoint truth provenance differs')
    baseline,baseline_provenance = read_prediction(baseline_folder/'predictions_val.parquet',
                      statuses['baseline']['predictions_sha256'],truth,False)
    flow,flow_provenance = read_prediction(flow_folder/'predictions_single_draw.parquet',
                      statuses['flow']['artifacts']['predictions_single_draw.parquet']['sha256'],truth,True)
    original_flow_metrics = load_json(flow_folder/'metrics.json')
    original_baseline_metrics = load_json(baseline_folder/'metrics.json')
    baseline_profile = evaluate(baseline,truth)
    flow_profile = evaluate(flow,truth)
    endpoint_checks = dict(
        baseline_against_frozen_cache_metrics=verify_profile(baseline_profile,original_flow_metrics['baseline_point']['observables'],METRICS,'baseline/cache'),
        baseline_against_original_metrics=verify_profile(baseline_profile,original_baseline_metrics,['mae','rmse','bias','w1'],'baseline/original'),
        flow_against_original_single_draw=verify_profile(flow_profile,original_flow_metrics['single_draw']['observables'],METRICS,'flow/single_draw'))
    candidates = []
    for alpha in ALPHAS+[1.]:
        prediction = baseline.copy() if alpha==0 else flow.copy() if alpha==1 else (1-alpha)*baseline+alpha*flow
        profile = baseline_profile if alpha==0 else flow_profile if alpha==1 else evaluate(prediction,truth)
        candidate = dict(alpha=alpha,role='flow_reference' if alpha==1 else 'baseline' if alpha==0 else 'scanned_mixture',
                         prediction_float64_sha256=hashlib.sha256(prediction.astype('<f8',copy=False).tobytes()).hexdigest(),
                         metrics=profile,comparison=compare(profile,baseline_profile))
        candidates.append(candidate)
        print(json.dumps(dict(alpha=alpha,role=candidate['role'],
                              **{k:candidate['comparison'][k] for k in ['strict_all8_preserved','strict_all4_w1_improved','all12_conditions_met']})),flush=True)
    provenance = dict(baseline=baseline_provenance,flow=flow_provenance,
                      canonical=dict(path=str(canonical_path),sha256=canonical_hash,events=NROWS),
                      plan=dict(path=str(plan_path),sha256=digest(plan_path)),
                      source_script_sha256=digest(__file__),
                      shared_metric_implementation=dict(path=str(ROOT/'paper_target_20261111/audit_existing.py'),
                              sha256=digest(ROOT/'paper_target_20261111/audit_existing.py')),
                      endpoints={name:dict(status_sha256=digest(folder/'status.json'),metrics_sha256=digest(folder/'metrics.json'))
                                 for name,folder in [('baseline',baseline_folder),('flow',flow_folder)]})
    report = dict(created_utc=datetime.now(timezone.utc).isoformat(),events=NROWS,keys=KEYS,
                   formula='(1-alpha)*baseline[event,observable] + alpha*fixed_flow_draw[event,observable]',
                   scan_alphas=ALPHAS,reference_alpha=1.,dtype='float64',clipping=False,new_training=False,
                   sample_seed=statuses['flow']['single_draw_seed'],shared_alpha_for_all_observables=True,
                   comparison_absolute_tolerance=ATOL,strict_w1_criterion='candidate < baseline - 1e-12',
                   reference='Existing Huber(delta=1)+0.1EMD point prediction',local_trace_bins=BINS,
                   self_checks=checks,endpoint_metric_checks=endpoint_checks,provenance=provenance,candidates=candidates,
                   notes=['Full-width Res: quantile .8413 minus quantile .1587 of prediction minus canonical truth.',
                          'Local Trace averages row-normalized diagonal; prediction overflow remains in each truth-bin denominator.',
                          'No event filtering, reordering, truth-conditioned routing, new sampling, prediction clipping or test evaluation.',
                          'Alpha 1 is an unchanged flow endpoint reference, outside the five-alpha authorized scan and plotted range.',
                          'Numerical validation comparisons only; no bootstrap or significance claim; model and alpha selection reuse validation.',
                          'Local Trace bin/overflow equivalence to the paper is not fully established; no paper-superiority claim.',
                          'Interpolation changes the original conditional density; original flow NLL/calibration does not transfer.'])
    OUT.mkdir(parents=True,exist_ok=True)
    write_json(OUT/'mixture_results.json',report)
    write_text(report)
    draw_figure(candidates[:-1])
    print('Saved',OUT,flush=True)


if __name__ == '__main__':
    main()
