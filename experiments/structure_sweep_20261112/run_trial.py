"""Full-data loss comparison; all training occurs in allocated Slurm GPU jobs."""
from pathlib import Path
import argparse
import hashlib
import importlib
import json
import math
import os
import subprocess
import time
import traceback

ACTIVE_STATUS = None

def save(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def main():
    global ACTIVE_STATUS
    p = argparse.ArgumentParser()
    p.add_argument('--index', type=int, required=True)
    p.add_argument('--repo', type=Path, required=True)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--smoke', action='store_true')
    args = p.parse_args()
    started = time.monotonic()
    import sys
    sys.path.insert(0, str(args.repo / 'src'))
    import numpy as np
    import awkward as ak
    import pyarrow.parquet as pq
    import torch
    import pytorch_lightning as pl
    from scipy.stats import wasserstein_distance
    from hepsetreg.backbones.head import EventRegressor, RegressionHead
    from module import ObjectiveModule
    from aux_dataset import attach_auxiliary
    from models import build_model, summarize_details
    from evaluation import compute_metrics, select_metrics, diagnostics, write_predictions, digest
    from training_control import BudgetSelection
    from pytorch_lightning.loggers import CSVLogger

    code = Path(__file__).resolve().parent
    plan = json.loads((code / 'experiment_plan.json').read_text(encoding='utf-8'))
    cfg = plan['runs'][args.index]
    out = args.output / (('smoke_' if args.smoke else '') + cfg['run_id'])
    out.mkdir(parents=True, exist_ok=False)
    ACTIVE_STATUS = out/'status.json'
    save(out / 'config.json', cfg)
    record = dict(status='running', config=cfg, job_id=os.environ.get('SLURM_JOB_ID'), smoke=args.smoke)
    save(out / 'status.json', record)
    for name, expected in plan['verified_source_sha256'].items():
        if hashlib.sha256((args.repo/'src/hepsetreg'/name).read_bytes()).hexdigest() != expected:
            raise RuntimeError('Remote source changed since audit: '+name)
    canonical_path = code/'canonical_validation.npz'
    if hashlib.sha256(canonical_path.read_bytes()).hexdigest() != plan['canonical_npz_sha256']:
        raise RuntimeError('Common validation truth hash mismatch')
    with np.load(canonical_path, allow_pickle=False) as canonical:
        canonical_truth = canonical['targets'].copy()
        row_index = canonical['row_index'].copy()
        canonical_keys = canonical['keys'].tolist()
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision('medium')
    if not os.environ.get('SLURM_JOB_ID') or not torch.cuda.is_available():
        raise RuntimeError('CUDA required; do not train on login node')
    pl.seed_everything(cfg['seed'], workers=True)
    targets = ['ttbar_mass', 'delta_abs_y', 'cos_K_plus', 'cos_phi']
    assert canonical_keys == targets and canonical_truth.shape == (260190,4)
    assert np.array_equal(row_index, np.arange(260190)) and np.isfinite(canonical_truth).all()
    p4 = ['pt', 'eta', 'phi', 'e', 'm', 'px', 'py', 'pz']
    features = dict(global_=['global_njet', 'global_nelectron', 'global_nmuon', 'global_nbtagged', 'global_met_met', 'global_met_phi'])
    features['global'] = features.pop('global_')
    features['target'] = targets
    part = cfg['model'] == 'part'
    if part:
        features.update(jet_vector=['jet_'+f for f in p4], jet_scalar=['jet_bTag'], lepton_vector=p4, lepton_scalar=['charge','is_electron'])
    else:
        features.update(jet=['jet_'+f for f in p4]+['jet_bTag'], lepton=p4+['charge','is_electron'])
    paths = {}
    for split, rows in [('train',4692288), ('val',260190)]:
        candidates = [args.data / f'ttbar-2L_{split}_051026_augmented.parquet', args.data / f'ttbar-2L_{split}_four.parquet']
        matches = [x for x in candidates if x.is_file()]
        if len(matches) != 1:
            raise RuntimeError(f'Expected one known {split} file: {matches}')
        path = matches[0]
        if pq.ParquetFile(path).metadata.num_rows != rows:
            raise RuntimeError(f'Wrong {split} row count')
        data_hasher = hashlib.sha256()
        with path.open('rb') as f:
            for block in iter(lambda: f.read(8*1024*1024), b''):
                data_hasher.update(block)
        paths[split] = path
        record[split] = dict(path=str(path), rows=rows, sha256=data_hasher.hexdigest())
        if data_hasher.hexdigest() != plan['data_sha256'][split]:
            raise RuntimeError('Data hash changed: '+split)
    record['source_commit'] = subprocess.check_output(['git','-C',str(args.repo),'rev-parse','HEAD'], text=True).strip()
    record['source_sha256'] = {str(f.relative_to(args.repo)):hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted((args.repo/'src/hepsetreg').rglob('*.py'))}
    record.update(torch=torch.__version__, gpu=torch.cuda.get_device_name(), precision='bf16-mixed', compile=False, head_dropout=0.05,dropout_scope='backbone only; head fixed at 0.05')
    record['code_sha256'] = {f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(code.glob('*.py'))}
    record['plan_sha256'] = hashlib.sha256((code/'experiment_plan.json').read_bytes()).hexdigest()
    record['canonical_truth_sha256'] = plan['canonical_npz_sha256']
    record['checkpoint_selection'] = plan['checkpoint_selection']
    dsmod = importlib.import_module('hepsetreg.datasets.' + ('EventVectorScalarDataset' if part else 'EventDatasetBase'))
    dm = dsmod.EventDataModule(str(paths['train']), str(paths['val']), features=features, batch_size=cfg['batch_size'], keep_unscaled_data=part)
    dm.setup('fit')
    # Cache the already prepared full train/val datasets; Trainer must not reload them.
    dm.setup = lambda stage=None: None
    mean, std = [torch.as_tensor(x) for x in dm.stats['target']]
    recovered = (dm.val_dataset.target_tensor.cpu()*(std+1e-8)+mean).numpy()
    alignment = np.max(np.abs(recovered-canonical_truth), axis=0)
    if not np.all(alignment <= np.array([.01,5e-4,5e-4,5e-4])):
        raise RuntimeError('Validation target order or field mapping differs from canonical truth: '+str(alignment))
    record['truth_alignment_maxabs'] = dict(zip(targets,alignment.tolist()))
    record['target_mean'] = mean.tolist()
    record['target_std'] = std.tolist()
    backbone_module = importlib.import_module('hepsetreg.backbones.' + ('backbone_ParT' if part else 'backbone'))
    kwargs = dict(d_model=cfg['d_model'], nhead=cfg['nhead'], num_layers=cfg['num_layers'], dropout=cfg['dropout'], n_global_features=len(features['global']))
    if part:
        kwargs.update(n_jet_vector_features=8,n_jet_scalar_features=1,n_lepton_vector_features=8,n_lepton_scalar_features=2)
    else:
        kwargs.update(n_jet_features=9,n_lepton_features=10)
    backbone = getattr(backbone_module, 'TransformerBackboneParT' if part else 'TransformerBackbone')(**kwargs)
    model = EventRegressor(backbone, RegressionHead(d_model=cfg['d_model'], output_dim=4))
    initial_hash = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        initial_hash.update(name.encode())
        initial_hash.update(tensor.detach().cpu().numpy().tobytes())
    record['initial_model_sha256'] = initial_hash.hexdigest()
    assert initial_hash.hexdigest() == plan['expected_initial_model_sha256'], 'Initial model differs from audited baseline'
    assert cfg['model']=='regression' and cfg['schedule']=='constant'
    assert len(dm.train_dataset)==4692288 and len(dm.val_dataset)==260190
    assert digest(Path(plan['historical_baseline_artifact']['path']))==plan['historical_baseline_artifact']['sha256']
    auxiliary=attach_auxiliary(dm,paths,mean,std,canonical_truth)
    record['auxiliary']=auxiliary
    model=build_model(model,cfg,auxiliary['tail_prevalence'],auxiliary['stats']['mean'],
        auxiliary['stats']['std'],mean,std+1e-8)
    module=ObjectiveModule(model,cfg)
    record['parameters']=sum(x.numel() for x in model.parameters())
    record['loss_parameters']=0
    control=BudgetSelection(started,out,plan,canonical_truth,mean,std,smoke=args.smoke)
    trainer=pl.Trainer(accelerator='gpu',devices=1,precision='bf16-mixed',max_epochs=1 if args.smoke else 100,
        logger=CSVLogger(str(out),name='logs'),callbacks=[control],enable_checkpointing=False,
        num_sanity_val_steps=0,enable_progress_bar=False,log_every_n_steps=50)
    torch.save(dict(stats=dm.stats,max_jets=dm.train_dataset.max_jets,features=features,config=cfg,auxiliary=auxiliary),out/'stats.pt')
    save(out/'status.json',record)
    trainer.fit(module,datamodule=dm)
    if not control.completed:raise RuntimeError('No complete full-data training epoch within budget')
    record.update(completed_full_epochs=control.completed,partial_final_epoch=control.partial,
        updates=trainer.global_step,epoch_seconds=control.epoch_seconds,
        train_and_setup_seconds=time.monotonic()-started,checkpoints=control.best)
    # Both prediction sets carry all metrics; primary was fixed before training.
    # MSE-best is a separately labelled historical-rule diagnostic.
    profiles={}
    for label in ['primary','mse']:
        path=out/'checkpoints'/(label+'.ckpt')
        selected=torch.load(path,map_location='cpu',weights_only=False)
        module.load_state_dict(selected['state_dict'])
        module.to('cuda').eval()
        predictions=[];truth=[];detail_parts={}
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
            for batch in dm.val_dataloader():
                moved={k:v.to('cuda') if torch.is_tensor(v) else v for k,v in batch.items()}
                details=module.model.forward_details(moved)
                predictions.append(details['prediction'].float().cpu())
                for detail_key,detail_value in details.items():
                    if torch.is_tensor(detail_value):detail_parts.setdefault(detail_key,[]).append(detail_value.float().cpu())
                truth.append(batch['target'].float().cpu())
        pred=(torch.cat(predictions)*(std+1e-8)+mean).numpy()
        recovered=(torch.cat(truth)*(std+1e-8)+mean).numpy()
        if pred.shape!=(260190,4) or not np.isfinite(pred).all():raise RuntimeError('Invalid full predictions')
        if not np.all(np.max(np.abs(recovered-canonical_truth),axis=0)<=np.array([.01,5e-4,5e-4,5e-4])):
            raise RuntimeError('Final prediction event alignment mismatch')
        profile=compute_metrics(pred,canonical_truth)
        profile['selection']=select_metrics(profile,plan['baseline_metrics'])
        # The independent final pass must reproduce the selected validation metrics.
        for key in targets:
            for name in ['res','local_trace','w1','mae','rmse','bias']:
                if not np.isclose(profile['observables'][key][name],control.best[label]['metrics']['observables'][key][name],atol=1e-9,rtol=1e-10):
                    raise RuntimeError('Selected checkpoint final inference differs: '+label+'/'+key+'/'+name)
        profile['checkpoint']=dict(path=str(path),sha256=digest(path),epoch=int(selected['epoch']),selection_label=label)
        profile['diagnostics']=diagnostics(pred,canonical_truth)
        combined_details={k:torch.cat(v) for k,v in detail_parts.items()}
        profile['model_diagnostics']=summarize_details(combined_details,torch.as_tensor(canonical_truth[:,0]>800))
        if module.adaptive is not None:profile['adaptive']=module.adaptive.diagnostics()
        destination=out/('predictions_val.parquet' if label=='primary' else 'predictions_val_mse.parquet')
        write_predictions(destination,pred,canonical_truth)
        profile['predictions']=dict(path=str(destination),sha256=digest(destination),rows=len(pred),roundtrip_exact=True)
        save(out/('metrics.json' if label=='primary' else 'metrics_mse.json'),profile)
        profiles[label]=profile
    record.update(status='smoke_passed' if args.smoke else 'complete',prediction_rows=260190,
        primary_eligible=profiles['primary']['selection']['eligible'],best_epoch=profiles['primary']['checkpoint']['epoch'],
        primary_prediction_sha256=profiles['primary']['predictions']['sha256'],
        elapsed_seconds=time.monotonic()-started,independent_test_evaluated=False)
    save(out/'status.json',record)
    print(json.dumps(dict(status=record['status'],run=cfg['run_id'],full_epochs=control.completed,
        elapsed=record['elapsed_seconds'],primary_eligible=record['primary_eligible'])),flush=True)

if __name__=='__main__':
    try:main()
    except Exception as exc:
        if ACTIVE_STATUS is not None:
            state=json.loads(ACTIVE_STATUS.read_text()) if ACTIVE_STATUS.exists() else {}
            state.update(status='failed',error=str(exc),traceback=traceback.format_exc())
            save(ACTIVE_STATUS,state)
        raise
