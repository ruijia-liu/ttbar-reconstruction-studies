"""One continuous prediction vector per event for every reported metric."""
from pathlib import Path
import hashlib
import json
import numpy as np

KEYS=['ttbar_mass','delta_abs_y','cos_K_plus','cos_phi']
BINS=dict(ttbar_mass=[300,380,440,500,750,2000],delta_abs_y=[-2.4,-1,-.5,0,.5,1,2.4],
          cos_K_plus=[-1,-.5,0,.5,1],cos_phi=[-1,-.5,0,.5,1])

def save(path,value):
    path=Path(path)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    tmp.replace(path)

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()

def response(p,t,edges):
    count=np.histogram2d(t,p,bins=(edges,edges))[0]
    denom=np.histogram(t,bins=edges)[0]
    # Keep empty rows explicit; main full-validation bins must all be occupied.
    matrix=np.divide(count,denom[:,None],out=np.zeros_like(count),where=denom[:,None]>0)
    return dict(bin_edges=np.asarray(edges).tolist(),truth_bin_counts=denom.tolist(),
        row_probabilities=matrix.tolist(),local_trace=float(np.diag(matrix).mean()),
        empty_truth_bins=int((denom==0).sum()),truth_outside=int(len(t)-denom.sum()),
        prediction_outside_given_truth_bin=(denom-count.sum(axis=1)).astype(int).tolist())

def compute_metrics(pred,truth):
    p,t=np.asarray(pred,dtype=np.float64),np.asarray(truth,dtype=np.float64)
    if p.shape!=t.shape or p.ndim!=2 or p.shape[1]!=4 or not len(p):raise ValueError('Bad prediction shape')
    if not np.isfinite(p).all() or not np.isfinite(t).all():raise ValueError('Nonfinite prediction/truth')
    result={}
    for j,k in enumerate(KEYS):
        e=p[:,j]-t[:,j]; lo,hi=np.quantile(e,[.1587,.8413],method='linear')
        m=response(p[:,j],t[:,j],BINS[k])
        result[k]=m|dict(res=float(hi-lo),q1587=float(lo),q8413=float(hi),
            w1=float(np.abs(np.sort(p[:,j])-np.sort(t[:,j])).mean()),
            mae=float(np.abs(e).mean()),rmse=float(np.sqrt(np.mean(e*e))),bias=float(e.mean()))
    return dict(events=len(p),observables=result,illegal_output_counts=dict(
        mass_nonpositive=int((p[:,0]<=0).sum()),cos_K_plus_outside=int((np.abs(p[:,2])>1).sum()),
        cos_phi_outside=int((np.abs(p[:,3])>1).sum())))

def select_metrics(metrics,baseline):
    values=[]; shape=[]; checks={}
    for k in KEYS:
        m,b=metrics['observables'][k],baseline['observables'][k]
        if m['empty_truth_bins']:raise ValueError('Empty primary truth bin')
        checks[k+'/res']=m['res']<=b['res']+1e-12
        checks[k+'/trace']=m['local_trace']>=b['local_trace']-1e-12
        for name in ['res','local_trace','w1']:
            ratio=(b[name]/max(m[name],1e-12) if name=='local_trace' else max(m[name],1e-12)/b[name])
            v=float(np.log(ratio));values.append(v)
            if name=='w1':shape.append(v)
    eligible=all(checks.values())
    score=float(np.mean(values));shape_score=float(np.mean(shape))
    return dict(eligible=eligible,checks=checks,joint_score=score,shape_score=shape_score,
                primary_rank=[0 if eligible else 1,shape_score if eligible else score])

def diagnostics(pred,truth):
    p,t=np.asarray(pred,dtype=float),np.asarray(truth,dtype=float)
    result=dict(bin_variants={},mass_tails={})
    for j,k in enumerate(KEYS):
        edges=np.asarray(BINS[k],dtype=float); widths=np.diff(edges)
        fine=np.sort(np.r_[edges,(edges[:-1]+edges[1:])/2])
        shifted=edges.copy();shifted[1:-1]+=.1*np.minimum(widths[:-1],widths[1:])
        result['bin_variants'][k]={label:response(p[:,j],t[:,j],bins) for label,bins in [('fine',fine),('shifted',shifted)]}
    for threshold in [800,1000]:
        mask=t[:,0]>threshold
        if not mask.any():raise ValueError('Expected mass-tail events')
        e=p[mask]-t[mask]
        result['mass_tails'][str(threshold)]=dict(n=int(mask.sum()),mae=np.abs(e).mean(0).tolist(),
            bias=e.mean(0).tolist(),res=np.diff(np.quantile(e,[.1587,.8413],axis=0),axis=0)[0].tolist())
    result['correlations']=dict(truth=np.corrcoef(t,rowvar=False).tolist(),prediction=np.corrcoef(p,rowvar=False).tolist())
    return result

def write_predictions(path,pred,truth):
    import awkward as ak
    p,t=np.asarray(pred),np.asarray(truth)
    payload=dict(row_index=np.arange(len(t),dtype=np.int64),
        predictions={k:np.ascontiguousarray(p[:,j]) for j,k in enumerate(KEYS)},
        targets={k:np.ascontiguousarray(t[:,j]) for j,k in enumerate(KEYS)})
    ak.to_parquet(ak.Array(payload),path)
    restored=ak.from_parquet(path)
    if not np.array_equal(ak.to_numpy(restored.row_index),payload['row_index']):raise ValueError('Export row mismatch')
    for field in ['predictions','targets']:
        for k in KEYS:
            if not np.array_equal(ak.to_numpy(restored[field][k]),payload[field][k]):raise ValueError('Export value mismatch')
