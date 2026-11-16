"""Validate frozen cache and prepare native train-only labels/references in Slurm."""
from pathlib import Path
import json,os,time,argparse
import numpy as np
import torch
import awkward as ak
from cache_contract import load_cache
from evaluation import digest,save,compute_metrics,BINS,KEYS
from ot_proxy import build_reference

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--data',type=Path,required=True)
    args=p.parse_args();start=time.monotonic()
    if not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('Preparation requires Slurm allocation')
    here=Path(__file__).parent;plan=json.loads((here/'experiment_plan.json').read_text())
    torch.set_num_threads(4)
    args.output.mkdir(parents=True,exist_ok=False)
    save(args.output/'status.json',dict(status='running'))
    source=Path(plan['source_cache']).expanduser()
    assert digest(source/'manifest.json')==plan['source_cache_manifest_sha256']
    cache=load_cache(source);m=cache['manifest'];transforms=cache['transforms']
    assert m['data_sha256']==plan['data_sha256'] and m['frozen_checkpoint']['sha256']==plan['base_artifacts']['checkpoint']['sha256']
    assert m['canonical_truth_sha256']==plan['canonical_npz_sha256']
    mean=torch.tensor(transforms['point_z_mean'],dtype=torch.float32)
    std=torch.tensor(transforms['point_z_std'],dtype=torch.float32)+1e-8
    aux={};thresholds=None;refs=None
    for split,n in [('train',plan['train_rows']),('val',plan['val_rows'])]:
        paths=[q for q in [args.data/f'ttbar-2L_{split}_051026_augmented.parquet',args.data/f'ttbar-2L_{split}_four.parquet'] if q.is_file()]
        assert len(paths)==1 and digest(paths[0])==plan['data_sha256'][split]
        native=ak.from_parquet(paths[0],columns=['targets'])
        truth=np.column_stack([ak.to_numpy(native.targets[k]) for k in KEYS]).astype(np.float64);del native
        assert truth.shape==(n,4) and np.isfinite(truth).all()
        if split=='val':
            with np.load(here/'canonical_validation.npz') as c:
                assert np.array_equal(c['row_index'],np.arange(n))
                assert np.all(np.max(np.abs(truth-c['targets']),axis=0)<=[.01,5e-4,5e-4,5e-4])
                truth=c['targets'].copy()
            np.save(args.output/'val_truth.npy',truth)
            np.save(args.output/'val_point.npy',np.asarray(cache['arrays']['val_point']))
        target=((torch.from_numpy(truth.astype(np.float32))-mean)/std).numpy()
        np.save(args.output/(split+'_target_z.npy'),target)
        # Coordinate transform cross-check anchors the native labels to the exact cached rows.
        eps=transforms['atanh_epsilon']
        raw=np.column_stack([np.log(truth[:,0]),truth[:,1],np.arctanh(np.clip(truth[:,2],-1+eps,1-eps)),np.arctanh(np.clip(truth[:,3],-1+eps,1-eps))])
        u=((raw-np.array(transforms['mean']))/np.array(transforms['std'])).astype(np.float32)
        assert np.allclose(u,cache['arrays'][split+'_target_u'],atol=2e-6,rtol=2e-6),'Native targets and frozen rows differ'
        base_z=np.asarray(cache['arrays'][split+'_condition'][:,-4:])
        if split=='train':
            thresholds=np.quantile(base_z[:,0],plan['group_quantiles']).tolist()
            assert thresholds[0]<thresholds[1]
            edges=[((torch.tensor(BINS[k],dtype=torch.float32)-mean[j])/std[j]).tolist() for j,k in enumerate(KEYS)]
            tau=[.1*float(np.median(np.diff(BINS[k])))/float(std[j]) for j,k in enumerate(KEYS)]
            refs=build_reference(torch.from_numpy(base_z.copy()),torch.from_numpy(target),edges,tau)
        groups=np.searchsorted(thresholds,base_z[:,0],side='right').astype(np.int64)
        np.save(args.output/(split+'_groups.npy'),groups)
        aux[split]=dict(rows=n,group_counts=np.bincount(groups,minlength=3).tolist())
        del truth,target,u,raw,base_z,groups
    save(args.output/'proxy_reference.json',refs)
    manifest=dict(status='complete',source_cache=str(source),source_cache_manifest_sha256=digest(source/'manifest.json'),
        condition_sha256={s:m['artifacts'][s+'_condition.npy']['sha256'] for s in ['train','val']},
        mean=mean.tolist(),effective_std=std.tolist(),group_thresholds_z=thresholds,
        group_probabilities=(np.array(aux['train']['group_counts'])/plan['train_rows']).tolist(),
        splits=aux,baseline_metrics=plan['baseline_metrics'],data_sha256=plan['data_sha256'],
        plan_sha256=digest(here/'experiment_plan.json'),canonical_npz_sha256=plan['canonical_npz_sha256'],
        elapsed_seconds=time.monotonic()-start,artifacts={})
    baseline=compute_metrics(np.load(args.output/'val_point.npy'),np.load(args.output/'val_truth.npy'))
    for k in KEYS:
        for name in ['res','local_trace','w1']:assert abs(baseline['observables'][k][name]-plan['baseline_metrics']['observables'][k][name])<1e-9
    for path in sorted(args.output.iterdir()):
        if path.suffix=='.npy' or path.name=='proxy_reference.json':manifest['artifacts'][path.name]=dict(sha256=digest(path),bytes=path.stat().st_size)
    save(args.output/'manifest.json',manifest);save(args.output/'status.json',dict(status='complete',elapsed_seconds=manifest['elapsed_seconds']))
    print(json.dumps(dict(prepared=True,rows=aux,seconds=manifest['elapsed_seconds'])))

if __name__=='__main__':main()
