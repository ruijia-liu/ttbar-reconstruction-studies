import copy
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from scipy.stats import wasserstein_distance
from evaluation import BINS,KEYS,response,compute_metrics,select_metrics,write_predictions,diagnostics

class EvaluationTests(unittest.TestCase):
    def test_overflow_in_denominator_and_trace_unweighted(self):
        p=np.array([.2,3.,1.2]);t=np.array([.2,.2,1.2])
        r=response(p,t,[0,1,2])
        self.assertEqual(r['local_trace'],.75)
        self.assertEqual(r['prediction_outside_given_truth_bin'],[1,0])
        self.assertNotEqual(r['local_trace'],2/3)

    def test_fullwidth_and_sorted_w1(self):
        rng=np.random.default_rng(10)
        t=rng.normal(size=(1000,4));p=t+rng.normal(size=t.shape)
        m=compute_metrics(p,t)
        for j,k in enumerate(KEYS):
            self.assertAlmostEqual(m['observables'][k]['res'],np.diff(np.quantile(p[:,j]-t[:,j],[.1587,.8413]))[0])
            self.assertAlmostEqual(m['observables'][k]['w1'],wasserstein_distance(p[:,j],t[:,j]))

    def test_selection_does_not_trade_away_one_protected_metric(self):
        base={'observables':{k:dict(res=1.,local_trace=.5,w1=.1,empty_truth_bins=0) for k in KEYS}}
        safe=copy.deepcopy(base);bad=copy.deepcopy(base)
        for k in KEYS:
            safe['observables'][k]['w1']=.09
            bad['observables'][k]['w1']=.001
        bad['observables'][KEYS[0]]['res']=1.01
        self.assertLess(tuple(select_metrics(safe,base)['primary_rank']),tuple(select_metrics(bad,base)['primary_rank']))
        self.assertFalse(select_metrics(bad,base)['eligible'])

    def test_export_noncontiguous_roundtrip(self):
        t=np.arange(160,dtype=float).reshape(40,4)[::-2];p=t+.25
        with tempfile.TemporaryDirectory() as d:
            write_predictions(Path(d)/'pred.parquet',p[:,::-1],t)

    def test_baseline_full_validation_profile(self):
        import awkward as ak
        here=Path(__file__).resolve().parent
        plan=json.loads((here/'experiment_plan.json').read_text(encoding='utf-8'))
        local=here.parent/'loss_sweep_20261110/results/huber1_emd01/predictions_val.parquet'
        source=local if local.exists() else Path(plan['historical_baseline_artifact']['path'])
        arr=ak.from_parquet(source)
        p=np.column_stack([ak.to_numpy(arr.predictions[k]) for k in KEYS])
        t=np.column_stack([ak.to_numpy(arr.targets[k]) for k in KEYS])
        self.assertTrue(np.array_equal(ak.to_numpy(arr.row_index),np.arange(260190)))
        with np.load(here/'canonical_validation.npz',allow_pickle=False) as canonical:
            np.testing.assert_array_equal(t,canonical['targets'])
        actual=compute_metrics(p,t)
        for k in KEYS:
            for m in ['res','local_trace','w1','mae','rmse','bias']:
                self.assertAlmostEqual(actual['observables'][k][m],plan['baseline_metrics']['observables'][k][m],places=9)
        self.assertTrue(select_metrics(actual,plan['baseline_metrics'])['eligible'])
        extra=diagnostics(p,t)
        self.assertEqual(set(extra['bin_variants']),set(KEYS))
        for k in KEYS:
            for v in extra['bin_variants'][k].values():self.assertEqual(v['empty_truth_bins'],0)

if __name__=='__main__':unittest.main()
