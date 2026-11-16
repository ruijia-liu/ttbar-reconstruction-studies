"""One-epoch official full VyPER training smoke test; never uses TEST."""
import hashlib
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
from datetime import datetime, timezone

import h5py


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024),b''): h.update(block)
    return h.hexdigest()


def subset(source,destination,n):
    with h5py.File(source,'r') as src,h5py.File(destination,'x') as out:
        total=len(src['INPUTS/GLOBAL'])
        if total<n: raise RuntimeError('Source too short')
        def copy_group(a,b):
            for key,value in a.attrs.items(): b.attrs[key]=value
            for name,obj in a.items():
                if isinstance(obj,h5py.Group): copy_group(obj,b.create_group(name))
                else:
                    if obj.shape and obj.shape[0]!=total:
                        raise RuntimeError(f'Unexpected non-event dataset: {obj.name}')
                    data=obj[:n] if obj.shape else obj[()]
                    ds=b.create_dataset(name,data=data)
                    for key,value in obj.attrs.items(): ds.attrs[key]=value
        copy_group(src,out)
    return {'rows':n,'selection':f'first {n} rows in source order','sha256':digest(destination),'source':str(source)}


def main():
    root=Path(os.environ.get("RECO_DATA_ROOT", "external")) / 'vyper_full_20261103'
    repo=root/'source'
    commit=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()
    if commit!='c5500033c5eacd0f405b80b6d60eb2988b3dbd4a': raise RuntimeError('Unexpected source commit')
    data=root/'data/aligned_six_global'
    manifest=json.loads((data/'manifest.json').read_text())
    if not manifest.get('complete'): raise RuntimeError('Aligned data incomplete')
    import torch
    import lightning.pytorch as pl
    if not torch.cuda.is_available() or torch.cuda.device_count()!=1:
        raise RuntimeError('Exactly one allocated CUDA GPU required')
    run=root/'runs'/('full_smoke_seed42_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
    run.mkdir(parents=True,exist_ok=False)
    record={'purpose':'Full official ttbar2L architecture, small TRAIN/VAL excerpts, one epoch; not scientific performance',
            'source_commit':commit,'seed':42,'torch':torch.__version__,'cuda':torch.version.cuda,
            'gpu':torch.cuda.get_device_name(0),'slurm_job_id':os.environ.get('SLURM_JOB_ID'),
            'data_manifest_sha256':digest(data/'manifest.json'),'config_sha256':digest(repo/'configs/default.yaml'),
            'script_sha256':digest(__file__),'subsets':{},'status':'preparing'}
    for split,n in [('train',512),('val',128)]:
        record['subsets'][split]=subset(data/f'ttbar-2L_{split}.h5',run/f'{split}.h5',n)
    overrides=['--config-name=default',f'datasets.train_set={run}/train.h5',f'datasets.val_set={run}/val.h5',
               'datasets.drop_last=false','training.epochs=1','training.batch_size=32',
               'device.num_workers=2','device.num_devices=1',f'training.save_directory={run}/logs',
               f'hydra.run.dir={run}/hydra']
    record['overrides']=overrides
    report=run/'smoke_report.json'
    report.write_text(json.dumps(record,indent=2)+'\n')
    print(f'FULL VYPER SMOKE DIRECTORY: {run}',flush=True)
    print('Official architecture and both edge/diffusion tasks retained; 512 TRAIN + 128 VAL; one epoch.',flush=True)
    pl.seed_everything(42,workers=True)
    sys.path.insert(0,str(repo))
    os.chdir(repo)
    sys.argv=['VyPER.train',*overrides]
    try:
        runpy.run_module('VyPER.train',run_name='__main__')
        checkpoints=list((run/'logs').glob('**/checkpoints/*.ckpt'))
        if not checkpoints: raise RuntimeError('No checkpoint saved')
        record['checkpoints']=[str(p) for p in checkpoints]
        record['status']='training_complete'
        record['prediction_test_completed']=False
        report.write_text(json.dumps(record,indent=2)+'\n')
        print(f'FULL VYPER TRAIN SMOKE COMPLETE: {report}',flush=True)
    except BaseException as exc:
        record.update(status='failed',error=repr(exc))
        report.write_text(json.dumps(record,indent=2)+'\n')
        raise


if __name__=='__main__': main()
