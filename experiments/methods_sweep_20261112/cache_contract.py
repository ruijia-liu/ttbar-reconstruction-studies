"""Separate single-draw, posterior-summary, and pooled-distribution evaluation."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
from scipy.stats import wasserstein_distance

KEYS = ['ttbar_mass', 'delta_abs_y', 'cos_K_plus', 'cos_phi']
BINS = dict(ttbar_mass=[300,380,440,500,750,2000],
            delta_abs_y=[-2.4,-1,-.5,0,.5,1,2.4],
            cos_K_plus=[-1,-.5,0,.5,1],cos_phi=[-1,-.5,0,.5,1])


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def save_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    temporary.replace(path)


def inverse_transform(target_u, transforms):
    """Invert standardized log/identity/atanh coordinates without clipping."""
    u = np.asarray(target_u,dtype=np.float64)
    if u.ndim < 2 or u.shape[-1] != 4 or not np.isfinite(u).all():
        raise ValueError('Expected finite transformed targets [...,4]')
    if transforms['keys'] != KEYS or transforms['transform'] != ['log','identity','atanh','atanh']:
        raise ValueError('Unknown target transform convention')
    mean, std = np.asarray(transforms['mean']),np.asarray(transforms['std'])
    if mean.shape!=(4,) or std.shape!=(4,) or not np.isfinite(mean).all() or not np.isfinite(std).all() or np.any(std<=0):
        raise ValueError('Invalid training target transform statistics')
    raw = u*std+mean
    with np.errstate(over='ignore',under='ignore',invalid='ignore'):
        result = np.stack([np.exp(raw[...,0]),raw[...,1],np.tanh(raw[...,2]),np.tanh(raw[...,3])],axis=-1)
    if not np.isfinite(result).all() or np.any(result[...,0]<=0):
        raise FloatingPointError('Inverse transform produced nonfinite/nonpositive mass; no clipping applied')
    return result


def load_cache(directory):
    """Verify frozen provenance and every array; keep arrays memory-mapped."""
    directory = Path(directory)
    manifest = json.loads((directory/'manifest.json').read_text(encoding='utf-8'))
    if manifest['status']!='complete' or not manifest['point_alignment_verified']:
        raise ValueError('Cache/baseline alignment is incomplete')
    if not manifest['encoder_frozen'] or not manifest['encoder_eval']:
        raise ValueError('Encoder must be frozen and in evaluation mode')
    if manifest['train_rows']!=4692288 or manifest['val_rows']!=260190:
        raise ValueError('This experiment requires the full fixed train/validation samples')
    arrays = {}
    for name in ['train_condition.npy','train_target_u.npy','val_condition.npy','val_target_u.npy',
                 'val_truth.npy','val_point.npy','row_index_val.npy','transforms.json']:
        entry = manifest['artifacts'][name]
        if digest(directory/name)!=entry['sha256']:
            raise ValueError('Cache artifact checksum mismatch: '+name)
        if name.endswith('.npy'):
            a = np.load(directory/name,mmap_mode='r',allow_pickle=False)
            if list(a.shape)!=entry['shape'] or str(a.dtype)!=entry['dtype'] or not np.isfinite(a).all():
                raise ValueError('Invalid cache array: '+name)
            arrays[name[:-4]]=a
    n,m = manifest['train_rows'],manifest['val_rows']
    shapes = dict(train_condition=(n,132),train_target_u=(n,4),val_condition=(m,132),
                  val_target_u=(m,4),val_truth=(m,4),val_point=(m,4),row_index_val=(m,))
    for key,shape in shapes.items():
        if arrays[key].shape!=shape:
            raise ValueError('Unexpected cache shape: '+key)
    if not np.array_equal(arrays['row_index_val'],np.arange(m)):
        raise ValueError('Validation row order differs')
    transforms = json.loads((directory/'transforms.json').read_text(encoding='utf-8'))
    # Reconstruct the recorded validation target transform independently.
    truth = np.asarray(arrays['val_truth'],dtype=np.float64)
    if np.any(truth[:,0]<=0) or np.any(np.abs(truth[:,2:])>1):
        raise ValueError('Canonical truth lies outside transform support')
    epsilon=float(transforms['atanh_epsilon'])
    transformed=np.column_stack([np.log(truth[:,0]),truth[:,1],
                                  np.arctanh(np.clip(truth[:,2],-1+epsilon,1-epsilon)),
                                  np.arctanh(np.clip(truth[:,3],-1+epsilon,1-epsilon))])
    expected=((transformed-np.asarray(transforms['mean']))/np.asarray(transforms['std'])).astype(np.float32)
    if not np.array_equal(expected,arrays['val_target_u']):
        # Cache preprocessing may round at an intermediate arithmetic step.
        if not np.allclose(expected,arrays['val_target_u'],rtol=2e-6,atol=2e-6):
            raise ValueError('Validation transformed targets are not aligned with physical truth')
    return dict(directory=directory,manifest=manifest,manifest_sha256=digest(directory/'manifest.json'),
                transforms=transforms,arrays=arrays)


def illegal_counts(prediction):
    p=np.asarray(prediction)
    return dict(nonfinite=int((~np.isfinite(p)).sum()),
                mass_nonpositive=int((p[...,0]<=0).sum()),
                cos_K_plus_outside=int((np.abs(p[...,2])>1).sum()),
                cos_phi_outside=int((np.abs(p[...,3])>1).sum()))


def compute_metrics(prediction, truth):
    p,t=np.asarray(prediction,dtype=np.float64),np.asarray(truth,dtype=np.float64)
    if p.shape!=t.shape or p.ndim!=2 or p.shape[1]!=4 or not len(p):
        raise ValueError('Prediction/truth must have the same nonempty [N,4] shape')
    counts=illegal_counts(p)
    if counts['nonfinite'] or not np.isfinite(t).all():
        raise FloatingPointError('Nonfinite values cannot be omitted from evaluation')
    result={}
    for i,key in enumerate(KEYS):
        error=p[:,i]-t[:,i]
        low,high=np.quantile(error,[.1587,.8413],method='linear')
        edges=np.asarray(BINS[key])
        response_counts=np.histogram2d(t[:,i],p[:,i],bins=(edges,edges))[0]
        denominators=np.histogram(t[:,i],bins=edges)[0]
        if np.any(denominators==0):
            raise ValueError('Cannot compute local Trace with empty truth bins')
        response=response_counts/denominators[:,None]
        result[key]=dict(n=len(p),res=float(high-low),q1587=float(low),q8413=float(high),
                         local_trace=float(np.diag(response).mean()),
                         mae=float(np.abs(error).mean()),rmse=float(np.sqrt(np.mean(error**2))),
                         bias=float(error.mean()),w1=float(wasserstein_distance(p[:,i],t[:,i])),
                         bin_edges=edges.tolist(),truth_bin_counts=denominators.tolist(),
                         row_probabilities=response.tolist(),truth_outside=int(len(t)-denominators.sum()),
                         prediction_outside_given_truth_bin=(denominators-response_counts.sum(axis=1)).astype(int).tolist())
    return dict(events=len(p),observables=result,illegal_output_counts=counts)


def selection_score(metrics, baseline, epsilon=1e-12):
    """Predeclared equal-weight log-ratio score; never a superiority claim."""
    components={}
    for key in KEYS:
        p,b=metrics['observables'][key],baseline['observables'][key]
        for metric in ['res','local_trace','w1']:
            candidate,reference=float(p[metric]),float(b[metric])
            if not np.isfinite([candidate,reference]).all() or candidate<0 or reference<=0:
                raise ValueError('Invalid metric in geometric selection score')
            ratio=(reference/max(candidate,epsilon) if metric=='local_trace' else max(candidate,epsilon)/reference)
            components[key+'/'+metric]=dict(candidate=candidate,baseline=reference,ratio=float(ratio),log_ratio=float(np.log(ratio)))
    score=float(np.mean([v['log_ratio'] for v in components.values()]))
    return dict(score=score,geometric_ratio=float(np.exp(score)),components=components,
                count=12,zero_floor=epsilon,
                interpretation='Equal-weight exploratory geometric compromise; all 12 components remain separately reported; not success criteria')


def sample_single(head, condition, transforms, device, seed=4203, batch_size=1024):
    import torch
    generator=torch.Generator(device=device).manual_seed(int(seed))
    result=np.empty((len(condition),4),dtype=np.float64)
    head.eval()
    with torch.inference_mode():
        for start in range(0,len(condition),batch_size):
            c=condition[start:start+batch_size]
            if not torch.is_tensor(c):
                c=torch.as_tensor(np.array(c,copy=True),device=device,dtype=torch.float32)
            else:
                c=c.to(device=device,dtype=torch.float32)
            u=head.sample(c,1,generator=generator)
            if u.shape!=(len(c),1,4):
                raise ValueError('Head sample shape differs from [N,K,4]')
            # The stored posterior draws use float32; use the same convention.
            with np.errstate(over='ignore',under='ignore'):
                physical=inverse_transform(u.cpu().numpy(),transforms).astype(np.float32)
            if any(illegal_counts(physical).values()):
                raise FloatingPointError('Illegal single-draw prediction; no clipping applied')
            result[start:start+len(c)]=physical[:,0,:]
    return result


def write_predictions(path,prediction,truth,row_index):
    import awkward as ak
    # A column of a C-contiguous N-by-4 matrix is itself strided. Arrow
    # requires each exported primitive buffer to be contiguous, including
    # fields backed by read-only memory maps; preserve values and dtypes.
    ak.to_parquet(ak.Array(dict(row_index=np.ascontiguousarray(row_index),
                  targets={key:np.ascontiguousarray(np.asarray(truth)[:,i]) for i,key in enumerate(KEYS)},
                  predictions={key:np.ascontiguousarray(np.asarray(prediction)[:,i]) for i,key in enumerate(KEYS)})),path)


def conditional_diagnostics(prediction, truth):
    """Cheap joint-correlation and truth-mass tail checks, without tail Trace."""
    p,t=np.asarray(prediction,dtype=np.float64),np.asarray(truth,dtype=np.float64)
    def nullable(matrix):
        return [[float(value) if np.isfinite(value) else None for value in row] for row in matrix]
    with np.errstate(divide='ignore',invalid='ignore'):
        correlation=dict(keys=KEYS,truth=nullable(np.corrcoef(t,rowvar=False)),
                         prediction=nullable(np.corrcoef(p,rowvar=False)),
                         truth_prediction=nullable(np.corrcoef(t.T,p.T)[:4,4:]),
                         undefined_value='null for zero-variance coordinates')
    slices={}
    for threshold in [800,1000]:
        mask=t[:,0]>threshold
        group={}
        for i,key in enumerate(KEYS):
            error=p[mask,i]-t[mask,i]
            if not len(error):
                group[key]=None
                continue
            low,high=np.quantile(error,[.1587,.8413],method='linear')
            group[key]=dict(n=int(mask.sum()),res=float(high-low),
                            mae=float(np.abs(error).mean()),rmse=float(np.sqrt(np.mean(error**2))),
                            bias=float(error.mean()),w1=float(wasserstein_distance(p[mask,i],t[mask,i])))
        slices['truth_mass_gt_'+str(threshold)]=dict(events=int(mask.sum()),observables=group)
    return dict(correlations=correlation,mass_slices=slices,
                note='Tail groups use physical truth mass; no empty-bin local Trace is defined for these slices')


def evaluate_full(head,cache,output,device,*,single_draw_seed=4203,posterior_seed=4204,
                  sample_batch_size=1024,num_samples=32,condition=None):
    import torch
    output=Path(output)
    if num_samples!=32:
        raise ValueError('This experiment predeclares 32 posterior samples')
    arrays=cache['arrays'];truth=np.asarray(arrays['val_truth']);n=len(truth)
    condition=arrays['val_condition'] if condition is None else condition
    baseline=compute_metrics(arrays['val_point'],truth)
    single=sample_single(head,condition,cache['transforms'],device,single_draw_seed,sample_batch_size)
    posterior_path=output/'posterior_samples32.npy'
    posterior=np.lib.format.open_memmap(posterior_path,mode='w+',dtype=np.float32,shape=(n,32,4))
    mean=np.empty((n,4),dtype=np.float64);median=np.empty_like(mean)
    generator=torch.Generator(device=device).manual_seed(int(posterior_seed))
    head.eval()
    with torch.inference_mode():
        for start in range(0,n,sample_batch_size):
            c=condition[start:start+sample_batch_size]
            c=c.to(device=device,dtype=torch.float32) if torch.is_tensor(c) else torch.as_tensor(np.array(c,copy=True),device=device,dtype=torch.float32)
            u=head.sample(c,32,generator=generator)
            if u.shape!=(len(c),32,4):
                raise ValueError('Head sample shape differs from [N,32,4]')
            with np.errstate(over='ignore',under='ignore'):
                draws=inverse_transform(u.cpu().numpy(),cache['transforms']).astype(np.float32)
            if any(illegal_counts(draws).values()):
                raise FloatingPointError('Illegal posterior prediction; no clipping applied')
            posterior[start:start+len(c)]=draws
            mean[start:start+len(c)]=draws.mean(axis=1,dtype=np.float64)
            median[start:start+len(c)]=np.median(draws.astype(np.float64),axis=1)
    posterior.flush()
    report=dict(baseline_point=baseline)
    report['baseline_point']['diagnostics']=conditional_diagnostics(arrays['val_point'],truth)
    for name,pred in [('single_draw',single),('mean32',mean),('median32',median)]:
        report[name]=compute_metrics(pred,truth)
        report[name]['diagnostics']=conditional_diagnostics(pred,truth)
        write_predictions(output/('predictions_'+name+'.parquet'),pred,truth,arrays['row_index_val'])
    # N*K equal-weight draws give each event equal total weight (K is constant).
    pool_w1={key:float(wasserstein_distance(np.asarray(posterior[:,:,i]).reshape(-1),truth[:,i])) for i,key in enumerate(KEYS)}
    report['posterior_pool32']=dict(events=n,samples_per_event=32,samples=n*32,w1=pool_w1,
                  illegal_output_counts=illegal_counts(posterior),
                  weighting='Each event total weight 1/N; each of its 32 draws weight 1/(32*N)',
                  interpretation='Marginal posterior-mixture shape only; not eventwise Res/Trace or point MAE')
    report['selection']=selection_score(report['single_draw'],baseline)
    report['sampling']=dict(single_draw_seed=single_draw_seed,posterior_seed=posterior_seed,
                            batch_size=sample_batch_size,posterior_samples=32,
                            single_draw_is_separate_from_pool=True,
                            summary_order='Inverse-transform each sample first, then mean/median each physical observable')
    report['notes']=['Validation only; one training seed; fixed-seed draws are used for checkpoint selection.',
                     'Trace uses local bins; paper bin/overflow equivalence is not fully verified.',
                     'Res is Q84.13-Q15.87 full width; all rows retained and no physical prediction clipping.',
                     'Single draw, mean32, median32 and posterior pool are different outputs and are not mixed to claim a win.']
    save_json(output/'metrics.json',report)
    del posterior
    return report


def main():
    import torch
    from density_heads import make_head
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--cache',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID') or not torch.cuda.is_available():
        raise RuntimeError('Slurm CUDA allocation required for full density evaluation')
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    checkpoint=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    head=make_head(**checkpoint['head_config']).to('cuda')
    head.load_state_dict(checkpoint['state_dict'])
    cache=load_cache(args.cache)
    if checkpoint['cache_manifest_sha256']!=cache['manifest_sha256']:
        raise ValueError('Checkpoint was fitted to a different cache')
    args.output.mkdir(parents=True,exist_ok=False)
    settings=checkpoint['evaluation_config']
    report=evaluate_full(head,cache,args.output,torch.device('cuda'),**settings)
    reference=checkpoint['local_vyper_reference']
    checks={}
    for key in KEYS:
        actual=report['single_draw']['observables'][key]
        checks[key+'/res']=actual['res']<=reference[key]['res']
        checks[key+'/local_trace']=actual['local_trace']>=reference[key]['local_trace']
    report['selection'].update(primary_eligible=bool(all(checks.values())),eligibility_checks=checks,
                               local_vyper_reference=reference)
    save_json(args.output/'metrics.json',report)


if __name__=='__main__':
    main()
