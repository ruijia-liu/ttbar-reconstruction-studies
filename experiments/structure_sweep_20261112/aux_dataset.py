"""Row-preserving training-label attachment; model input whitelist excludes labels."""
import numpy as np
import torch
from physics_aux import load_auxiliary,fit_aux_stats,standardize_aux

class AuxiliaryDataset:
    def __init__(self,dataset,auxiliary,tail):
        self.dataset=dataset
        self.auxiliary=torch.as_tensor(auxiliary,dtype=torch.float32)
        self.tail=torch.as_tensor(tail,dtype=torch.float32)
        if self.auxiliary.shape!=(len(dataset),8) or self.tail.shape!=(len(dataset),):
            raise ValueError('Auxiliary row count mismatch')
        if not torch.isfinite(self.auxiliary).all() or not torch.isfinite(self.tail).all():raise ValueError('Nonfinite labels')
        if not ((self.tail==0)|(self.tail==1)).all():raise ValueError('Invalid tail labels')
    def __len__(self):return len(self.dataset)
    def __getattr__(self,name):return getattr(self.dataset,name)
    def __getitem__(self,index):
        result=dict(self.dataset[index]);result['aux_target']=self.auxiliary[index];result['tail_label']=self.tail[index]
        return result
    def __getitems__(self,indices):
        result=dict(self.dataset.__getitems__(indices))
        result['aux_target']=self.auxiliary[indices];result['tail_label']=self.tail[indices]
        return result

def attach_auxiliary(dm,paths,mean,std,canonical_truth):
    records={};stats=None;prevalence=None
    for split in ['train','val']:
        loaded=load_auxiliary(paths[split])
        truth=loaded['targets']
        dataset=getattr(dm,split+'_dataset')
        recovered=(dataset.target_tensor.cpu()*(std+1e-8)+mean).numpy()
        difference=np.max(np.abs(recovered-truth),axis=0)
        if truth.shape!=(len(dataset),4) or not np.all(difference<=[.01,5e-4,5e-4,5e-4]):
            raise ValueError('Auxiliary labels and dataset row order differ: '+split)
        if split=='val' and not np.all(np.max(np.abs(truth-canonical_truth),axis=0)<=[.01,5e-4,5e-4,5e-4]):
            raise ValueError('Auxiliary validation row order differs from canonical truth')
        tail=(truth[:,0]>800).astype(np.float32)
        if split=='train':
            stats=fit_aux_stats(loaded['auxiliary']);prevalence=float(tail.astype(float).mean())
            if not 0<prevalence<1:raise ValueError('Both mass regions must occur')
        auxiliary=standardize_aux(loaded['auxiliary'],stats)
        setattr(dm,split+'_dataset',AuxiliaryDataset(dataset,auxiliary,tail))
        records[split]=dict(rows=len(dataset),tail_count=int(tail.sum()),alignment_maxabs=difference.tolist(),
            source_validation=loaded['report'])
    return dict(stats=stats,tail_prevalence=prevalence,splits=records)
