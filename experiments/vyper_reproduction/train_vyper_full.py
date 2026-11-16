"""Full official VyPER training, fixed seed, independent logs, explicit resume."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
import time


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''): h.update(block)
    return h.hexdigest()


def bounded_concatenate(self):
    """Same serialized graphs/order; bounded memory and atomic cache publishing."""
    import h5py
    temporary=self.db_path+'.writing'
    with h5py.File(temporary,'w') as out:
        target=out.create_dataset('Graphs',shape=(self.size,),dtype=h5py.string_dtype('utf-8'))
        offset=0
        for filename in self.batch_files:
            with h5py.File(filename,'r') as source:
                values=source['Graphs']
                for start in range(0,len(values),4096):
                    block=values[start:start+4096]
                    if any(not x for x in block): raise RuntimeError('Incomplete worker graph cache')
                    target[offset:offset+len(block)]=block
                    offset+=len(block)
        if offset!=self.size: raise RuntimeError('Graph cache row count changed')
    os.replace(temporary,self.db_path)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    root=Path(os.environ.get("RECO_DATA_ROOT", "external")) / 'vyper_full_20261103'
    repo=root/'source'
    data=root/'data/aligned_six_global'
    run=root/'runs/full_seed42'
    commit=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()
    if commit!='c5500033c5eacd0f405b80b6d60eb2988b3dbd4a': raise RuntimeError('Unexpected source commit')
    if subprocess.check_output(['git','-C',str(repo),'diff','--name-only','HEAD'],text=True).strip():
        raise RuntimeError('Tracked official source was modified; inspect before training')
    manifest=json.loads((data/'manifest.json').read_text())
    if not manifest.get('complete'): raise RuntimeError('Incomplete data manifest')
    import h5py
    import torch
    import lightning.pytorch as pl
    if not torch.cuda.is_available() or torch.cuda.device_count()!=1: raise RuntimeError('Exactly one allocated GPU required')
    if not args.resume and run.exists(): raise RuntimeError('Run exists: use --resume after inspecting the previous log')
    if args.resume and not (run/'run.json').exists(): raise RuntimeError('No recorded full training run to resume')
    run.mkdir(parents=True,exist_ok=True)
    identity={'source_commit':commit,'config_sha256':digest(repo/'configs/default.yaml'),
              'data_manifest_sha256':digest(data/'manifest.json'),'seed':42,'batch_size':4096,
              'max_epochs':500,'early_stop_patience':10,'num_workers':2,'drop_last':False,
              'cache_change':'Chunked concatenation of identical serialized graph bytes; atomic final cache publication',
              'script_sha256':digest(__file__)}
    resume=None
    if args.resume:
        previous=json.loads((run/'run.json').read_text())
        if previous['identity']!=identity: raise RuntimeError('Run configuration/data/code identity changed')
        candidates=[]
        for path in (run/'logs').glob('version_*/checkpoints/last.ckpt'):
            checkpoint=torch.load(path,map_location='cpu',weights_only=False)
            candidates.append((checkpoint['global_step'],path))
        if candidates: resume=str(max(candidates,key=lambda item:item[0])[1])
        elif previous.get('status')=='complete': raise RuntimeError('Completed run has no last checkpoint')
    record={'identity':identity,'status':'preparing','slurm_job_id':os.environ.get('SLURM_JOB_ID'),
            'torch':torch.__version__,'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(0),'resume_checkpoint':resume}
    report=run/'run.json'
    report.write_text(json.dumps(record,indent=2)+'\n')
    for split in ('train','val'):
        path=data/manifest['splits'][split]['filename']
        if digest(path)!=manifest['splits'][split]['sha256']: raise RuntimeError(f'{split} data hash mismatch')
        with h5py.File(path) as f:
            if len(f['INPUTS/GLOBAL'])!=manifest['splits'][split]['rows']: raise RuntimeError('Unexpected row count')
        cache=path.with_suffix('.db')
        if cache.exists():
            with h5py.File(cache) as f:
                graphs=f['Graphs']
                if len(graphs)!=manifest['splits'][split]['rows']: raise RuntimeError('Incomplete cache; inspect before reuse')
                for start in range(0,len(graphs),4096):
                    if any(not x for x in graphs[start:start+4096]): raise RuntimeError('Empty entries in graph cache')
        print(f'{split}: {manifest["splits"][split]["rows"]:,} events',flush=True)
    sys.path.insert(0,str(repo))
    from VyPER.data.preprocessed import GraphDB
    GraphDB.concatenate=bounded_concatenate
    pl.seed_everything(42,workers=True)
    os.chdir(repo)
    overrides=['--config-name=default',f'datasets.train_set={data}/ttbar-2L_train.h5',
               f'datasets.val_set={data}/ttbar-2L_val.h5','datasets.drop_last=false',
               'training.batch_size=4096','training.epochs=500','device.num_workers=2',
               'device.num_devices=1',f'training.save_directory={run}/logs',
               f'hydra.run.dir={run}/hydra_'+time.strftime('%Y%m%dT%H%M%S')]
    if resume: overrides.append(f'training.continue_from_ckpt={resume}')
    record.update(status='training',overrides=overrides)
    report.write_text(json.dumps(record,indent=2)+'\n')
    sys.argv=['VyPER.train',*overrides]
    started=time.monotonic()
    try:
        runpy.run_module('VyPER.train',run_name='__main__')
        record.update(status='complete',elapsed_seconds=time.monotonic()-started)
        print(f'FULL VYPER TRAINING COMPLETE: {run}',flush=True)
    except BaseException as exc:
        record.update(status='interrupted_or_failed',error=repr(exc),elapsed_seconds=time.monotonic()-started)
        raise
    finally: report.write_text(json.dumps(record,indent=2)+'\n')


if __name__=='__main__': main()
