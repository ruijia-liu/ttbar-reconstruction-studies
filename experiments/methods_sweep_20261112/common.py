"""Shared bounded deterministic correction and verified prepared arrays."""
from pathlib import Path
import json
import hashlib
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from evaluation import digest,save,compute_metrics,select_metrics,diagnostics,write_predictions

class Correction(nn.Module):
    def __init__(self):
        super().__init__()
        self.features=nn.Sequential(nn.Linear(132,128),nn.SiLU(),nn.Linear(128,128),nn.SiLU())
        self.output=nn.Linear(128,4)
        nn.init.zeros_(self.output.weight);nn.init.zeros_(self.output.bias)
    def forward(self,condition):
        features=self.features(condition)
        delta=.5*torch.tanh(self.output(features))
        return condition[:,-4:]+delta,delta,features

def point_shape(pred,target):
    point=2*F.huber_loss(pred.float(),target.float(),delta=1.)
    shape=(pred.float().sort(0).values-target.float().sort(0).values).abs().mean()
    return point,shape

def state_hash(model):
    h=hashlib.sha256()
    for k,v in model.state_dict().items():h.update(k.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()

def checkpoint(path,model,metadata):
    path=Path(path);tmp=path.with_suffix('.tmp')
    torch.save(dict(state_dict={k:v.detach().cpu().clone() for k,v in model.state_dict().items()},**metadata),tmp)
    tmp.replace(path)

def load_data(root,device,condition=True):
    root=Path(root);manifest=json.loads((root/'manifest.json').read_text())
    if manifest['status']!='complete':raise ValueError('Preparation incomplete')
    plan_path=Path(__file__).with_name('experiment_plan.json')
    if digest(plan_path)!=manifest['plan_sha256']:raise ValueError('Preparation plan differs')
    plan=json.loads(plan_path.read_text())
    if manifest['data_sha256']!=plan['data_sha256'] or manifest['canonical_npz_sha256']!=plan['canonical_npz_sha256']:raise ValueError('Prepared data provenance differs')
    for name,entry in manifest['artifacts'].items():
        if digest(root/name)!=entry['sha256']:raise ValueError('Prepared checksum mismatch: '+name)
    arrays={}
    for name in ['train_target_z','val_target_z','train_groups','val_groups','val_truth','val_point']:
        arrays[name]=np.load(root/(name+'.npy'),mmap_mode='r',allow_pickle=False)
    cache=Path(manifest['source_cache'])
    for split in ['train','val']:
        path=cache/(split+'_condition.npy')
        if digest(path)!=manifest['condition_sha256'][split]:raise ValueError('Frozen condition cache changed')
        arrays[split+'_condition']=np.load(path,mmap_mode='r',allow_pickle=False)
        n=plan[split+'_rows']
        if arrays[split+'_condition'].shape!=(n,132) or arrays[split+'_target_z'].shape!=(n,4) or arrays[split+'_groups'].shape!=(n,):raise ValueError('Prepared array shape mismatch')
    tensors={k:torch.from_numpy(np.array(v,copy=True)).to(device) for k,v in arrays.items()
        if k in ['train_condition','val_condition','train_target_z','val_target_z','train_groups','val_groups']}
    return manifest,arrays,tensors

def predict(model,condition,base_physical,effective_std,batch_size=8192):
    model.eval();pieces=[]
    with torch.inference_mode():
        for start in range(0,len(condition),batch_size):
            pieces.append(model(condition[start:start+batch_size])[1].cpu().numpy())
    # Add only the correction to the exact original physical predictions.
    return np.asarray(base_physical,dtype=np.float64)+np.concatenate(pieces).astype(np.float64)*np.asarray(effective_std)

def evaluate(model,tensors,arrays,manifest):
    pred=predict(model,tensors['val_condition'],arrays['val_point'],manifest['effective_std'])
    metrics=compute_metrics(pred,arrays['val_truth'])
    selection=select_metrics(metrics,manifest['baseline_metrics'])
    return pred,metrics,selection
