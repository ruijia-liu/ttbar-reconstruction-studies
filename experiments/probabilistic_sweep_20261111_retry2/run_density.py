"""Full-data conditional density-head training on a verified frozen cache."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import time
import traceback
from pathlib import Path

import numpy as np
import torch

from density_heads import make_head
from evaluate_density import (BINS, KEYS, compute_metrics, digest, evaluate_full,
                              load_cache, sample_single, save_json, selection_score)

ACTIVE_STATUS = None
STATUS = {}


def eligible(metrics, reference):
    checks = {}
    for key in KEYS:
        actual = metrics['observables'][key]
        checks[key+'/res'] = actual['res'] <= reference[key]['res']
        checks[key+'/local_trace'] = actual['local_trace'] >= reference[key]['local_trace']
    return bool(all(checks.values())), checks


def state_digest(head):
    h = hashlib.sha256()
    for name, value in sorted(head.state_dict().items()):
        h.update(name.encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def write_checkpoint(path, head, metadata):
    temporary = Path(path).with_suffix('.pt.tmp')
    payload = dict(metadata, state_dict={key:value.detach().cpu().clone()
                                        for key,value in head.state_dict().items()})
    torch.save(payload, temporary)
    temporary.replace(path)


def restore(head, path, device):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    head.load_state_dict(checkpoint['state_dict'])
    return checkpoint


def validation_nll(head, condition, target, batch_size):
    head.eval()
    total = torch.zeros((), device=condition.device, dtype=torch.float64)
    with torch.inference_mode():
        for start in range(0, len(target), batch_size):
            values = -head.log_prob(target[start:start+batch_size], condition[start:start+batch_size])
            if values.shape != (len(target[start:start+batch_size]),) or not torch.isfinite(values).all().item():
                raise FloatingPointError('Nonfinite or malformed full-validation joint NLL')
            total += values.double().sum()
    return float((total/len(target)).item())


def main():
    global ACTIVE_STATUS, STATUS
    started = time.monotonic()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--kind', choices=['gaussian','flow'], required=True)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--plan', type=Path, default=Path(__file__).with_name('experiment_plan.json'))
    args = parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID') or not torch.cuda.is_available():
        raise RuntimeError('Full density training requires a Slurm GPU allocation')
    plan = json.loads(args.plan.read_text(encoding='utf-8'))
    output = args.output / (('smoke_' if args.smoke else '')+args.kind)
    output.mkdir(parents=True, exist_ok=False)
    ACTIVE_STATUS = output/'status.json'  # Set only after ownership is acquired.
    STATUS = dict(status='preparing',kind=args.kind,smoke=args.smoke,
                  slurm_job_id=os.environ['SLURM_JOB_ID'],host=os.environ.get('HOSTNAME'),
                  plan_sha256=digest(args.plan),completed_full_epochs=0,
                  training_events_subsampled=False,test_evaluation=False,
                  source_sha256={p.name:digest(p) for p in Path(__file__).parent.glob('*.py')})
    save_json(ACTIVE_STATUS, STATUS)
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    random.seed(plan['seed']);np.random.seed(plan['seed']);torch.manual_seed(plan['seed'])
    torch.cuda.manual_seed_all(plan['seed'])
    device = torch.device('cuda')
    deadline = started+60*plan['training_and_setup_budget_minutes']
    cache = load_cache(args.cache)
    manifest = cache['manifest']
    expected = [(manifest['data_sha256'],plan['data_sha256'],'dataset hashes'),
                (manifest['canonical_truth_sha256'],plan['canonical_npz_sha256'],'canonical truth hash'),
                (manifest['frozen_checkpoint']['sha256'],plan['base_artifacts']['checkpoint']['sha256'],'encoder checkpoint'),
                (manifest['source_prediction_sha256'],plan['base_artifacts']['predictions']['sha256'],'baseline predictions'),
                (manifest['train_rows'],plan['train_rows'],'training rows'),
                (manifest['val_rows'],plan['val_rows'],'validation rows'),
                (plan['trace_bins'],BINS,'local Trace bins')]
    for actual,wanted,label in expected:
        if actual != wanted:
            raise ValueError('Cache/plan mismatch: '+label)
    arrays = cache['arrays']
    tensors = {name:torch.from_numpy(np.array(arrays[name],dtype=np.float32,copy=True)).to(device)
               for name in ['train_condition','train_target_u','val_condition','val_target_u']}
    torch.cuda.synchronize()
    head_config = dict(kind=args.kind,condition_dim=plan['condition_dim'],hidden_dim=plan['hidden_dim'],
                       num_layers=plan['num_layers'],anchor_last=plan['anchor_last'])
    if head_config['anchor_last']:
        raise ValueError('The original point z-scores are conditioning only, never transformed-space anchors')
    head = make_head(**head_config).to(device=device,dtype=torch.float32)
    initial_hash = state_digest(head)
    optimizer = torch.optim.AdamW(head.parameters(),lr=plan['learning_rate'],weight_decay=plan['weight_decay'])
    train_condition,train_target = tensors['train_condition'],tensors['train_target_u']
    val_condition,val_target = tensors['val_condition'],tensors['val_target_u']
    n = len(train_target);batch_size=plan['batch_size'];steps_per_epoch=math.ceil(n/batch_size)
    baseline = compute_metrics(arrays['val_point'],arrays['val_truth'])
    evaluation_config = dict(single_draw_seed=plan['single_draw_seed'],posterior_seed=plan['posterior_seed'],
                             sample_batch_size=int(plan.get('sample_batch_size',1024)),num_samples=plan['posterior_samples'])
    metadata = dict(head_config=head_config,evaluation_config=evaluation_config,
                    cache_manifest_sha256=cache['manifest_sha256'],plan_sha256=digest(args.plan),
                    frozen_checkpoint_sha256=manifest['frozen_checkpoint']['sha256'],
                    local_vyper_reference=plan['local_vyper_reference'])
    STATUS.update(status='training',train_rows=n,val_rows=len(val_target),condition_dim=plan['condition_dim'],
                  cache_manifest_sha256=cache['manifest_sha256'],data_sha256=manifest['data_sha256'],
                  canonical_truth_sha256=manifest['canonical_truth_sha256'],
                  frozen_checkpoint_sha256=manifest['frozen_checkpoint']['sha256'],
                  initial_state_sha256=initial_hash,head_parameters=sum(p.numel() for p in head.parameters()),
                  config=dict(head_config,learning_rate=plan['learning_rate'],weight_decay=plan['weight_decay'],
                              batch_size=batch_size,seed=plan['seed'],gradient_clip_norm=plan['gradient_clip_norm']),
                  gpu=torch.cuda.get_device_name(device),density_precision='float32',tf32_enabled=False,
                  shuffle_rule='CUDA randperm of every train row; independent generator seed=training_seed+epoch',
                  checkpoint_selection_rule=plan['checkpoint_selection'],
                  setup_seconds=time.monotonic()-started,training_and_setup_budget_minutes=plan['training_and_setup_budget_minutes'],
                  shared_cache_seconds=manifest.get('elapsed_seconds'),
                  cost_note='Head wall time excludes the separately reported shared cache and prior encoder training')
    save_json(ACTIVE_STATUS,STATUS)
    history=[];completed=0;committed_updates=0;discarded_updates=0
    best_nll=float('inf');best_key=(2,float('inf'));best_global=float('inf')
    last_metric_epoch=0;last_epoch_seconds=None;last_metric_seconds=0.0

    def evaluate_candidate(epoch, nll):
        nonlocal best_key,best_global,last_metric_epoch,last_metric_seconds
        metric_started=time.monotonic()
        prediction=sample_single(head,val_condition,cache['transforms'],device,
                                 evaluation_config['single_draw_seed'],evaluation_config['sample_batch_size'])
        metrics=compute_metrics(prediction,arrays['val_truth'])
        selection=selection_score(metrics,baseline)
        qualifies,checks=eligible(metrics,plan['local_vyper_reference'])
        selection.update(primary_eligible=qualifies,eligibility_checks=checks)
        checkpoint_meta=dict(metadata,epoch=epoch,val_nll=nll,selection=selection)
        score=selection['score'];key=(0 if qualifies else 1,score)
        if score<best_global:
            best_global=score
            write_checkpoint(output/'checkpoint_global_score.pt',head,checkpoint_meta)
        if key<best_key:
            best_key=key
            write_checkpoint(output/'checkpoint_score.pt',head,checkpoint_meta)
        last_metric_epoch=epoch;last_metric_seconds=time.monotonic()-metric_started
        history[epoch-1].update(single_draw_metrics=metrics,selection=selection,
                                metric_seconds=last_metric_seconds)
        save_json(output/'epoch_history.json',history)

    epochs_limit=1 if args.smoke else int(plan['max_epochs'])
    stop_reason='maximum_full_epochs'
    for epoch in range(1,epochs_limit+1):
        elapsed=time.monotonic()
        due=(epoch==1 or epoch%plan['metric_every_epochs']==0)
        if last_epoch_seconds is not None:
            reserve=1.20*last_epoch_seconds+(1.20*last_metric_seconds if due else 0)+2
            if elapsed+reserve>=deadline:
                stop_reason='next_full_epoch_would_exceed_budget';break
        elif elapsed>=deadline:
            raise RuntimeError('Setup consumed the full training budget before any complete epoch')
        epoch_started=time.monotonic();head.train()
        # Separate epoch RNG makes row order independent of head initialization and sampling.
        order_generator=torch.Generator(device=device).manual_seed(int(plan['seed'])+epoch)
        order=torch.randperm(n,device=device,generator=order_generator)
        epoch_total=torch.zeros((),device=device,dtype=torch.float64)
        processed=0;updates=0;partial=False
        for start in range(0,n,batch_size):
            if time.monotonic()>=deadline:
                partial=True;break
            indices=order[start:start+batch_size]
            optimizer.zero_grad(set_to_none=True)
            values=-head.log_prob(train_target[indices],train_condition[indices])
            if values.shape!=(len(indices),) or not torch.isfinite(values).all().item():
                raise FloatingPointError('Nonfinite or malformed joint training NLL')
            loss=values.mean();loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(),plan['gradient_clip_norm'],error_if_nonfinite=True)
            optimizer.step()
            epoch_total+=values.detach().double().sum();processed+=len(indices);updates+=1
        torch.cuda.synchronize()
        del order
        if partial:
            discarded_updates+=updates
            if not completed:
                raise RuntimeError('Budget expired before one full training epoch; no subset result accepted')
            restore(head,output/'checkpoint_last_full.pt',device)
            stop_reason='unexpected_partial_epoch_discarded';break
        if processed!=n or updates!=steps_per_epoch:
            raise RuntimeError('Incomplete training traversal cannot produce a checkpoint')
        nll=validation_nll(head,val_condition,val_target,batch_size)
        completed=epoch;committed_updates+=updates
        record=dict(epoch=epoch,train_rows_processed=processed,optimizer_updates=updates,
                    train_nll=float((epoch_total/n).item()),val_nll=nll)
        write_checkpoint(output/'checkpoint_last_full.pt',head,dict(metadata,epoch=epoch,val_nll=nll))
        if nll<best_nll:
            best_nll=nll
            write_checkpoint(output/'checkpoint_nll.pt',head,dict(metadata,epoch=epoch,val_nll=nll))
        last_epoch_seconds=time.monotonic()-epoch_started
        record['epoch_train_validation_checkpoint_seconds']=last_epoch_seconds
        history.append(record)
        if due:
            evaluate_candidate(epoch,nll)
        save_json(output/'epoch_history.json',history)
        STATUS.update(completed_full_epochs=completed,committed_optimizer_updates=committed_updates,
                      last_val_nll=nll,elapsed_seconds=time.monotonic()-started)
        save_json(ACTIVE_STATUS,STATUS)
        print(json.dumps(dict(epoch=epoch,val_nll=nll,seconds=time.monotonic()-started,
                              sampled_metrics=due)),flush=True)

    if not completed:
        raise RuntimeError('No complete full-data epoch')
    # The final complete epoch is always eligible for selection, even off the five-epoch cadence.
    if last_metric_epoch!=completed:
        last=restore(head,output/'checkpoint_last_full.pt',device)
        evaluate_candidate(completed,last['val_nll'])
    training_seconds=time.monotonic()-started
    chosen=restore(head,output/'checkpoint_score.pt',device)
    STATUS.update(status='evaluating',stop_reason=stop_reason,completed_full_epochs=completed,
                  committed_optimizer_updates=committed_updates,discarded_partial_optimizer_updates=discarded_updates,
                  training_and_setup_seconds=training_seconds,selected_epoch=chosen['epoch'],
                  selected_checkpoint_kind='balanced_score',primary_eligible=chosen['selection']['primary_eligible'])
    save_json(ACTIVE_STATUS,STATUS)
    report=evaluate_full(head,cache,output,device,condition=val_condition,**evaluation_config)
    qualifies,checks=eligible(report['single_draw'],plan['local_vyper_reference'])
    report['selection'].update(primary_eligible=qualifies,eligibility_checks=checks,
                               local_vyper_reference=plan['local_vyper_reference'])
    if qualifies!=chosen['selection']['primary_eligible'] or not math.isclose(report['selection']['score'],chosen['selection']['score'],abs_tol=1e-9,rel_tol=1e-9):
        raise RuntimeError('Final fixed-seed single-draw evaluation does not reproduce checkpoint selection')
    report['checkpoint']=dict(epoch=chosen['epoch'],val_nll=chosen['val_nll'],
                              selection_kind='balanced_score',nll_best=best_nll)
    save_json(output/'metrics.json',report)
    artifacts={p.name:dict(sha256=digest(p),bytes=p.stat().st_size)
               for p in output.iterdir() if p.is_file() and p.name!='status.json' and not p.name.endswith('.tmp')}
    STATUS.update(status='smoke_passed' if args.smoke else 'complete',prediction_rows=len(val_target),
                  sample_draws_per_event=32,single_draw_seed=evaluation_config['single_draw_seed'],
                  posterior_seed=evaluation_config['posterior_seed'],sample_batch_size=evaluation_config['sample_batch_size'],
                  best_checkpoint=str(output/'checkpoint_score.pt'),
                  selected_checkpoint_sha256=artifacts['checkpoint_score.pt']['sha256'],
                  selected_score=report['selection']['score'],primary_eligible=qualifies,
                  best_nll=best_nll,artifacts=artifacts,elapsed_seconds=time.monotonic()-started,
                  final_evaluation_seconds=time.monotonic()-started-training_seconds,
                  peak_cuda_memory_bytes=torch.cuda.max_memory_allocated(device))
    save_json(ACTIVE_STATUS,STATUS)
    print(json.dumps(STATUS),flush=True)


if __name__=='__main__':
    try:
        main()
    except BaseException as exc:
        if ACTIVE_STATUS is not None:
            STATUS.update(status='failed',error=repr(exc),traceback=traceback.format_exc())
            save_json(ACTIVE_STATUS,STATUS)
        raise
