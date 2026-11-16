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
    from module import LossRegressionModule
    from pytorch_lightning.loggers import CSVLogger

    code = Path(__file__).resolve().parent
    plan = json.loads((code / 'experiment_plan.json').read_text())
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
    if not torch.cuda.is_available():
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
        digest = hashlib.sha256()
        with path.open('rb') as f:
            for block in iter(lambda: f.read(8*1024*1024), b''):
                digest.update(block)
        paths[split] = path
        record[split] = dict(path=str(path), rows=rows, sha256=digest.hexdigest())
        if digest.hexdigest() != plan['data_sha256'][split]:
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
    module = LossRegressionModule(model, cfg)
    record['parameters'] = sum(x.numel() for x in model.parameters())
    # A fixed step schedule is explicit: cosine horizon is 20 full epochs,
    # while the wall-time budget can stop training sooner.
    if cfg['schedule'] == 'warmup_cosine':
        steps = 20 * math.ceil(4692288 / cfg['batch_size'])
        warmup = cfg['warmup_steps']
        def configure():
            optimizer = torch.optim.AdamW(module.parameters(),lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'],fused=True)
            def factor(step):
                if step < warmup:
                    return (step+1)/warmup
                progress = min(1.0,(step-warmup)/max(1,steps-warmup))
                return 0.5*(1+math.cos(math.pi*progress))
            return dict(optimizer=optimizer,lr_scheduler=dict(scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,factor),interval='step'))
        module.configure_optimizers = configure
        record['cosine_horizon_steps'] = steps

    class Budget(pl.Callback):
        def __init__(self):
            self.completed = 0
            self.epoch_start = None
            self.epoch_seconds = []
            self.cut_epoch = False
        def on_train_epoch_start(self, trainer, module):
            self.epoch_start = time.monotonic()
        def on_train_batch_end(self, trainer, module, outputs, batch, batch_idx):
            # Leave ~7 minutes for validation, prediction, checkpoint and writes.
            if time.monotonic()-started > 22*60:
                self.cut_epoch = batch_idx+1 < trainer.num_training_batches
                trainer.should_stop = True
        def on_validation_end(self, trainer, module):
            if trainer.sanity_checking:
                return
            duration = time.monotonic()-self.epoch_start
            self.epoch_seconds.append(duration)
            if not self.cut_epoch:
                self.completed += 1
            if time.monotonic()-started + duration > 22*60:
                trainer.should_stop = True

    budget = Budget()
    checkpoint = pl.callbacks.ModelCheckpoint(dirpath=out/'checkpoints/mse',monitor='val_mse',mode='min',save_top_k=1,filename='{epoch}-{val_mse:.6f}')
    objective_checkpoint = pl.callbacks.ModelCheckpoint(dirpath=out/'checkpoints/objective',monitor='val_loss',mode='min',save_top_k=1,filename='{epoch}-{val_loss:.6f}')
    trainer = pl.Trainer(accelerator='gpu',devices=1,precision='bf16-mixed',max_epochs=100,
        logger=CSVLogger(str(out),name='logs'),callbacks=[budget,checkpoint,objective_checkpoint],
        num_sanity_val_steps=0,enable_progress_bar=False,log_every_n_steps=50,
        limit_train_batches=3 if args.smoke else 1.0,limit_val_batches=2 if args.smoke else 1.0)
    if args.smoke:
        trainer.fit_loop.max_epochs = 1
    torch.save(dict(stats=dm.stats,max_jets=dm.train_dataset.max_jets,features=features,config=cfg),out/'stats.pt')
    save(out/'status.json',record)
    trainer.fit(module,datamodule=dm)
    if not checkpoint.best_model_path:
        raise RuntimeError('No checkpoint from complete validation')
    selected = torch.load(checkpoint.best_model_path,map_location='cpu',weights_only=False)
    module.load_state_dict(selected['state_dict'])
    record.update(completed_full_epochs=0 if args.smoke else budget.completed,smoke_epochs=budget.completed if args.smoke else 0,partial_final_epoch=budget.cut_epoch,updates=trainer.global_step,epoch_seconds=budget.epoch_seconds,best_checkpoint=checkpoint.best_model_path,train_and_setup_seconds=time.monotonic()-started)
    record.update(best_epoch=int(selected['epoch']), best_val_mse=float(checkpoint.best_model_score),objective_best_checkpoint=objective_checkpoint.best_model_path,objective_best_score=float(objective_checkpoint.best_model_score))
    if budget.completed == 0 and not args.smoke:
        record.update(status='budget_insufficient_no_full_epoch',elapsed_seconds=time.monotonic()-started)
        save(out/'status.json',record)
        raise RuntimeError('No complete full-data training epoch within budget')
    # Independent, full validation inference with best validation checkpoint.
    device = torch.device('cuda')
    module.to(device).eval()
    predictions, truth = [], []
    with torch.inference_mode(), torch.autocast('cuda',dtype=torch.bfloat16):
        for batch in dm.val_dataloader():
            moved = {k:v.to(device) if torch.is_tensor(v) else v for k,v in batch.items()}
            predictions.append(module(moved).float().cpu())
            truth.append(batch['target'].float().cpu())
    pred = (torch.cat(predictions)*(std+1e-8)+mean).numpy()
    target = (torch.cat(truth)*(std+1e-8)+mean).numpy()
    if pred.shape != (260190,4) or not np.isfinite(pred).all():
        raise RuntimeError('Missing or invalid predictions')
    if not np.all(np.max(np.abs(target-canonical_truth),axis=0) <= np.array([.01,5e-4,5e-4,5e-4])):
        raise RuntimeError('Full prediction pass violated event alignment')
    target = canonical_truth
    ak.to_parquet(ak.Array(dict(row_index=row_index,predictions={n:pred[:,i].copy() for i,n in enumerate(targets)},targets={n:target[:,i].copy() for i,n in enumerate(targets)})),out/'predictions_val.parquet')
    metrics = {}
    for i,name in enumerate(targets):
        error = pred[:,i].astype(float)-target[:,i]
        metrics[name] = dict(mae=float(np.abs(error).mean()),rmse=float(np.sqrt((error**2).mean())),bias=float(error.mean()),w1=float(wasserstein_distance(pred[:,i],target[:,i])))
    save(out/'metrics.json',metrics)
    # Fixed truth-defined tail and bins, with no prediction clipping.
    selections = [('truth_mass_gt800',target[:,0]>800)]
    edges = [0,380,440,500,750,1000,float('inf')]
    selections += [(f'truth_mass_{lo:g}_{hi:g}', (target[:,0]>=lo)&(target[:,0]<hi)) for lo,hi in zip(edges[:-1],edges[1:])]
    sliced = {}
    for label, mask in selections:
        sliced[label] = {'n':int(mask.sum()),'metrics':{}}
        if not mask.any():
            continue
        for i, name in enumerate(targets):
            error = pred[mask,i].astype(float)-target[mask,i]
            sliced[label]['metrics'][name] = dict(mae=float(np.abs(error).mean()),rmse=float(np.sqrt((error**2).mean())),bias=float(error.mean()),w1=float(wasserstein_distance(pred[mask,i],target[mask,i])))
    save(out/'metrics_mass_slices.json',sliced)
    save(out/'correlations.json',dict(target=np.corrcoef(target,rowvar=False).tolist(),prediction=np.corrcoef(pred,rowvar=False).tolist()))
    record.update(status='smoke_passed' if args.smoke else 'complete',elapsed_seconds=time.monotonic()-started,prediction_rows=len(pred),predictions_sha256=hashlib.sha256((out/'predictions_val.parquet').read_bytes()).hexdigest())
    save(out/'status.json',record)


if __name__ == '__main__':
    try:
        main()
    except Exception:
        # Surface failures in provenance as well as Slurm; never leave a false running status.
        try:
            if ACTIVE_STATUS is not None and ACTIVE_STATUS.is_file():
                status = json.loads(ACTIVE_STATUS.read_text())
                status.update(status='failed',error=traceback.format_exc())
                save(ACTIVE_STATUS,status)
        except Exception:
            pass
        raise
