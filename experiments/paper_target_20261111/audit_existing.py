"""Read-only audit of existing validation predictions; write only results/.

Res is the full central 68.26% residual interval Q(.8413)-Q(.1587), using
the paper's rounded quantile levels. Trace uses the project's local bins:
mean diagonal of truth-row-normalized migration probabilities. Predictions
outside the displayed range remain in their truth-row denominator.
"""
from __future__ import annotations

import hashlib
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import awkward as ak
import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import wasserstein_distance

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = HERE / 'results'
KEYS = ['ttbar_mass', 'delta_abs_y', 'cos_K_plus', 'cos_phi']
BINS = dict(ttbar_mass=[300,380,440,500,750,2000],
            delta_abs_y=[-2.4,-1,-.5,0,.5,1,2.4],
            cos_K_plus=[-1,-.5,0,.5,1], cos_phi=[-1,-.5,0,.5,1])


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def load_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def matrix(array, field):
    return np.column_stack([ak.to_numpy(array[field][key]).astype(np.float64) for key in KEYS])


def resolution(residual):
    low, high = np.quantile(residual, [.1587, .8413], method='linear')
    return dict(q1587=float(low), q8413=float(high), res=float(high-low))


def migration(truth, prediction, edges):
    edges = np.asarray(edges, dtype=np.float64)
    counts = np.histogram2d(truth, prediction, bins=(edges, edges))[0].astype(np.int64)
    denominators = np.histogram(truth, bins=edges)[0]
    if np.any(denominators == 0):
        raise ValueError('Trace is undefined here because a truth bin is empty')
    response = counts / denominators[:, None]
    return dict(local_trace=float(np.diag(response).mean()),
                trace_sum=float(np.trace(response)),
                event_weighted_correct_bin_fraction=float(np.trace(counts)/denominators.sum()),
                bin_edges=edges.tolist(), truth_bin_counts=denominators.tolist(),
                counts=counts.tolist(), row_probabilities=response.tolist(),
                diagonal=np.diag(response).tolist(), row_sums=response.sum(axis=1).tolist(),
                prediction_outside_given_truth_bin=(denominators-counts.sum(axis=1)).tolist(),
                truth_outside=int(len(truth)-denominators.sum()))


def self_check():
    # Uniform grid: rounded 15.87%/84.13% quantiles are 158.7 and 841.3.
    r = resolution(np.arange(1001, dtype=float))
    assert abs(r['res']-682.6) < 1e-10 and abs(r['res']/2-341.3) < 1e-10
    m = migration(np.array([.5]*9+[1.5]), np.array([.5]*10), [0,1,2])
    assert m['local_trace'] == .5 and m['event_weighted_correct_bin_fraction'] == .9
    m = migration(np.array([.5]*4+[1.5]*2), np.array([.5,.5,3,3,1.5,3]), [0,1,2])
    assert m['local_trace'] == .5 and m['prediction_outside_given_truth_bin'] == [2,1]
    assert m['row_sums'] == [.5,.5]
    return dict(full_width_quantiles=True, trace_not_event_weighted_accuracy=True,
                overflow_retained_in_truth_denominator=True)


def main():
    checks = self_check()
    canonical_path = ROOT/'agent_sweep_20261111/canonical_validation.npz'
    plan = load_json(ROOT/'agent_sweep_20261111/experiment_plan.json')
    assert sha256(canonical_path) == plan['canonical_npz_sha256']
    with np.load(canonical_path, allow_pickle=False) as z:
        truth = z['targets'].copy()
        assert z['keys'].tolist() == KEYS
        assert np.array_equal(z['row_index'], np.arange(260190))
    assert truth.shape == (260190,4) and np.isfinite(truth).all()
    legacy_root = ROOT/'本轮新结果_20261109'
    vyper_path = legacy_root/'02_评估结果/four_observables.parquet'
    assert sha256(vyper_path) == plan['canonical_source_sha256']
    sources = [dict(id='reference/vyper_release', family='reference', path=vyper_path),
               dict(id='reference/mse', family='reference', path=legacy_root/'03_输入预测及溯源/val_predictions.parquet'),
               dict(id='reference/mse_emd01', family='reference', path=legacy_root/'03_输入预测及溯源/val_predictions_emd01.parquet')]
    for family, folder, expected_count in [('transformer','transformer_sweep_20261109',26),
                                           ('loss','loss_sweep_20261110',7),
                                           ('agent','agent_sweep_20261111',5)]:
        base = ROOT/folder
        runs = load_json(base/'experiment_plan.json')['runs']
        assert len(runs) == expected_count
        manifest_path = base/'results/download_manifest.json'
        manifest = {item['path']:item for item in load_json(manifest_path)}
        for run in runs:
            run_id = run['run_id']
            rel = run_id+'/predictions_val.parquet'
            sources.append(dict(id=family+'/'+run_id, family=family,
                                path=base/'results'/rel, expected_hash=manifest[rel]['sha256'],
                                expected_status_hash=manifest[run_id+'/status.json']['sha256'],
                                download_manifest=str(manifest_path),
                                download_manifest_sha256=sha256(manifest_path)))

    models, provenance = {}, {}
    trees, alignment_cache = {}, {}
    prediction_duplicates = defaultdict(list)
    for source in sources:
        model_id, path = source['id'], source['path']
        file_hash = sha256(path)
        if 'expected_hash' in source:
            assert file_hash == source['expected_hash'], model_id+' download hash'
        arr = ak.from_parquet(path)
        pred, stored_truth = matrix(arr,'predictions'), matrix(arr,'targets')
        assert pred.shape == truth.shape and stored_truth.shape == truth.shape, model_id
        assert np.isfinite(pred).all() and np.isfinite(stored_truth).all(), model_id
        if 'valid' in arr.fields:
            assert all(bool(ak.all(arr.valid[k])) for k in KEYS), model_id+' invalid predictions'
        has_row_id = 'row_index' in arr.fields
        if has_row_id:
            assert np.array_equal(ak.to_numpy(arr.row_index), np.arange(len(truth))), model_id
        exact = np.array_equal(stored_truth, truth)
        tolerance = np.zeros(4)
        max_difference = np.abs(stored_truth-truth).max(axis=0)
        unique_coordinate_match = None
        if not exact:
            assert not has_row_id, model_id+' row-indexed prediction should store canonical truth exactly'
            tolerance = (np.array([.01,5e-5,2e-4,1e-5]) if source['family']=='reference'
                         else np.array([5e-4,5e-7,5e-7,5e-7]))
            assert np.all(max_difference <= tolerance), (model_id, max_difference.tolist())
            # Legacy files have no event ID: check every coordinate AND that
            # each tolerant target vector identifies exactly one canonical row.
            target_hash = hashlib.sha256(stored_truth.astype('<f8').tobytes()).hexdigest()
            cache_key = (tuple(tolerance), target_hash)
            if cache_key not in alignment_cache:
                tree_key = tuple(tolerance)
                if tree_key not in trees:
                    trees[tree_key] = cKDTree(truth/tolerance)
                lengths = trees[tree_key].query_ball_point(stored_truth/tolerance, r=1., p=np.inf,
                                                           return_length=True)
                assert np.all(lengths==1), (model_id,'ambiguous or missing coordinate alignment')
                alignment_cache[cache_key] = True
            unique_coordinate_match = True
        profile = {}
        for i,key in enumerate(KEYS):
            residual = pred[:,i]-truth[:,i]
            response = migration(truth[:,i],pred[:,i],BINS[key])
            profile[key] = dict(n=len(truth), **resolution(residual),
                                mae=float(np.abs(residual).mean()),
                                rmse=float(np.sqrt(np.mean(residual**2))),
                                bias=float(residual.mean()),
                                w1=float(wasserstein_distance(pred[:,i],truth[:,i])),
                                local_trace=response['local_trace'], migration=response)
        models[model_id] = profile
        array_hash = hashlib.sha256(pred.astype('<f8').tobytes()).hexdigest()
        prediction_duplicates[array_hash].append(model_id)
        entry = dict(family=source['family'], path=str(path), file_sha256=file_hash,
                     prediction_array_sha256=array_hash, rows=len(truth), fields=arr.fields,
                     explicit_row_index_verified=has_row_id, stored_truth_exact=exact,
                     ordered_truth_maxabs=dict(zip(KEYS,max_difference.tolist())),
                     ordered_truth_tolerance=dict(zip(KEYS,tolerance.tolist())),
                     unique_tolerant_coordinate_match=unique_coordinate_match)
        if source['family']!='reference':
            status_path = path.parent/'status.json'
            assert sha256(status_path)==source['expected_status_hash']
            status = load_json(status_path)
            assert status['status']=='complete' and not status.get('smoke'), model_id
            if 'prediction_rows' in status:
                assert status['prediction_rows']==len(truth)
            if 'predictions_sha256' in status:
                assert status['predictions_sha256']==file_hash
            entry.update(status_sha256=sha256(status_path),
                         download_manifest=source['download_manifest'],
                         download_manifest_sha256=source['download_manifest_sha256'],
                         status={k:status[k] for k in ['config','train','val','source_commit','gpu','precision',
                                 'parameters','completed_full_epochs','updates','elapsed_seconds','best_checkpoint',
                                 'checkpoint_selection','best_epoch','initial_model_sha256'] if k in status})
            previous = load_json(path.parent/'metrics.json')
            for i,key in enumerate(KEYS):
                for metric in ['mae','rmse','bias','w1']:
                    assert abs(profile[key][metric]-previous[key][metric]) <= max_difference[i]+1e-10
            entry['existing_metric_consistency_verified'] = True
        provenance[model_id] = entry
        print('Verified',model_id,flush=True)

    # Validate the local VyPER and historical reference interpretation against
    # the existing four-observable audit, independently of model labels.
    old = load_json(legacy_root/'02_评估结果/metrics.json')['metrics']
    for new_id, old_id in [('reference/vyper_release','VyPER (release defaults)'),
                           ('reference/mse','MSE'),('reference/mse_emd01','MSE + 0.1 EMD')]:
        for key in KEYS:
            for metric in ['mae','rmse','bias','w1']:
                old_key = 'wasserstein_1' if metric=='w1' else metric
                assert abs(models[new_id][key][metric]-old[old_id][key][old_key]) < 1e-10

    vyper = models['reference/vyper_release']
    comparisons = {}
    for model_id, profile in models.items():
        comparisons[model_id] = dict(
            all_four_lower_res=all(profile[k]['res'] < vyper[k]['res'] for k in KEYS),
            all_four_higher_local_trace=all(profile[k]['local_trace'] > vyper[k]['local_trace'] for k in KEYS),
            all_four_lower_w1=all(profile[k]['w1'] < vyper[k]['w1'] for k in KEYS),
            observables={k:dict(res_difference=profile[k]['res']-vyper[k]['res'],
                                local_trace_difference=profile[k]['local_trace']-vyper[k]['local_trace'],
                                w1_difference=profile[k]['w1']-vyper[k]['w1']) for k in KEYS})
    notes = [
        'All results use the same 260190 validation events and canonical truth; no clipping, cuts, matching selection, or test evaluation.',
        'Res is Q84.13(error)-Q15.87(error), the FULL width, not half-width or RMSE; error=prediction-truth; NumPy linear quantiles.',
        'Trace is LOCAL: mean(diagonal(row-normalized migration)). All events in each truth bin enter its denominator, including out-of-range predictions.',
        'All migration bins/overflow handling have not been fully verified against the paper. Local Trace is not a direct published-Trace benchmark.',
        'Legacy predictions lacking row IDs passed per-row four-coordinate tolerances and a unique canonical-coordinate match; this is not an independent event-ID check.',
        'Reference VyPER is the local release-default, one-draw reconstruction; it is not the published paper checkpoint/configuration.',
        'Validation has been used for checkpoint and model selection. One seed; training budgets and selection rules differ across families.',
        'Profiles compare complete models across all four observables; no synthetic combination of per-observable winners is defined.',
        'No significance claim: repeated training seeds, untouched test evaluation, and paper protocol alignment remain necessary.'
    ]
    report = dict(created_utc=datetime.now(timezone.utc).isoformat(), events=len(truth),
                  model_entries=len(models), unique_prediction_arrays=len(prediction_duplicates),
                  canonical=dict(path=str(canonical_path),sha256=sha256(canonical_path),
                                 source_path=str(vyper_path),source_sha256=sha256(vyper_path)),
                  observables=KEYS, mass_units='GeV; all other observables dimensionless',
                  definitions=dict(res_quantiles=[.1587,.8413],res_full_width=True,
                                   quantile_method='linear', residual='prediction - canonical truth',
                                   local_trace='sum(diagonal(truth-row-normalized response)) / number_of_bins',
                                   trace_bins=BINS,paper_trace_protocol_fully_verified=False),
                  self_checks=checks,metrics=models,local_vyper_comparisons=comparisons,
                  identical_prediction_groups=[ids for ids in prediction_duplicates.values() if len(ids)>1],
                  provenance=provenance,notes=notes)
    OUT.mkdir(exist_ok=True)
    (OUT/'existing_prediction_audit.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8')

    lines=['Existing predictions: paper-defined Res and LOCAL-bin Trace',
           f'{len(models)} model entries / {len(prediction_duplicates)} distinct prediction arrays; N={len(truth)} validation events.',
           'Res=Q84.13-Q15.87 FULL WIDTH; lower is better. Local Trace=mean diagonal; higher is better. W1: lower is better.',
           'LOCAL Trace bins and overflow convention are NOT fully verified against the paper.', '']
    focus=['reference/vyper_release','reference/mse','reference/mse_emd01',
           'loss/huber05','loss/huber1_emd01','loss/mse_emd01',
           'agent/truth_partition','agent/learned_agent']
    for model_id in focus:
        lines += [model_id,'observable | Res(full width) | local Trace | W1 | MAE | RMSE | bias']
        for key in KEYS:
            p=models[model_id][key]
            lines.append(f"{key} | {p['res']:.6f} | {p['local_trace']:.6f} | {p['w1']:.6f} | {p['mae']:.6f} | {p['rmse']:.6f} | {p['bias']:.6f}")
        lines.append('')
    both=[mid for mid,c in comparisons.items() if c['all_four_lower_res'] and c['all_four_higher_local_trace']]
    all_three=[mid for mid in both if comparisons[mid]['all_four_lower_w1']]
    lines += ['Complete-model comparisons against LOCAL release-default VyPER:',
              'Lower Res and higher LOCAL Trace on all four: '+(', '.join(both) or 'none'),
              'Also lower W1 on all four: '+(', '.join(all_three) or 'none'),'',
              'Identical prediction arrays: '+json.dumps(report['identical_prediction_groups']), '']
    lines += notes
    (OUT/'audit_summary.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('\n'.join(lines[:12]),flush=True)
    print('Wrote',OUT/'existing_prediction_audit.json',flush=True)


if __name__ == '__main__':
    main()
