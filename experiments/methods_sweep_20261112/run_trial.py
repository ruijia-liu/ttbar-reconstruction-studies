"""Six controlled correction methods, complete full-data epochs and exact metrics."""
import argparse,json,os,time,traceback
from pathlib import Path
import numpy as np
import torch
from common import Correction,load_data,point_shape,checkpoint,state_hash,evaluate
from evaluation import save,digest,diagnostics,write_predictions,KEYS
from ot_proxy import ProxyConstraint,conditional_ot_loss,make_directions
from ranksim import rank_similarity_loss
from flowalign import VelocityField,path_consistency_loss

def main():
    p=argparse.ArgumentParser();p.add_argument('--index',type=int,required=True);p.add_argument('--prepared',type=Path,required=True)
    p.add_argument('--field',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--smoke',action='store_true')
    args=p.parse_args();started=time.monotonic();here=Path(__file__).parent
    if not os.environ.get('SLURM_JOB_ID') or not torch.cuda.is_available():raise RuntimeError('Slurm GPU required')
    plan=json.loads((here/'experiment_plan.json').read_text());cfg=plan['runs'][args.index]
    out=args.output/(('smoke_' if args.smoke else '')+cfg['run_id']);out.mkdir(parents=True,exist_ok=False)
    (out/'checkpoints').mkdir()
    status=dict(status='preparing',config=cfg,smoke=args.smoke,slurm_job_id=os.environ['SLURM_JOB_ID'],
        plan_sha256=digest(here/'experiment_plan.json'),code_sha256={f.name:digest(f) for f in here.glob('*.py')})
    save(out/'status.json',status)
    try:
        torch.set_num_threads(4);torch.set_float32_matmul_precision('highest')
        torch.manual_seed(42);torch.cuda.manual_seed_all(42)
        manifest,arrays,t=load_data(args.prepared,'cuda')
        model=Correction().cuda();initial=state_hash(model)
        proxy=ProxyConstraint(json.loads((args.prepared/'proxy_reference.json').read_text()),
            residual_temperature=plan['proxy']['residual_temperature'],dual_lr=plan['proxy']['dual_lr'],dual_cap=100.).cuda()
        directions=make_directions().cuda();group_prob=torch.tensor(manifest['group_probabilities'],device='cuda')
        field=None
        if cfg['flow']:
            fs=json.loads((args.field/'status.json').read_text())
            if fs['status']!='complete' or digest(args.field/'best.pt')!=fs['best_checkpoint_sha256']:raise ValueError('Field stage incomplete or changed')
            fc=torch.load(args.field/'best.pt',map_location='cpu',weights_only=False)
            if fc['prepared_manifest_sha256']!=digest(args.prepared/'manifest.json'):raise ValueError('Field data mismatch')
            # Preserve common model initialization and all explicit sampling streams.
            with torch.random.fork_rng(devices=[]):field=VelocityField(**fc['field_config'])
            field.load_state_dict(fc['state_dict']);field=field.cuda().eval().requires_grad_(False)
            status['field_checkpoint_sha256']=fs['best_checkpoint_sha256']
            status['extra_field_seconds']=fs['elapsed_seconds']
        optimizer=torch.optim.AdamW(model.parameters(),lr=plan['learning_rate'],weight_decay=plan['weight_decay'])
        rank_generator=torch.Generator(device='cuda').manual_seed(4211)
        flow_generator=torch.Generator(device='cuda').manual_seed(4212)
        zero_pred,zero_metrics,zero_selection=evaluate(model,t,arrays,manifest)
        if not np.array_equal(zero_pred,np.asarray(arrays['val_point'],dtype=np.float64)):raise RuntimeError('Zero correction changed original baseline')
        save(out/'zero_reference.json',dict(metrics=zero_metrics,selection=zero_selection))
        status.update(status='running',initial_model_sha256=initial,parameters=sum(p.numel() for p in model.parameters()),
            train_rows=plan['train_rows'],val_rows=plan['val_rows'],prepared_manifest_sha256=digest(args.prepared/'manifest.json'),
            encoder_frozen=True,zero_matches_baseline_exactly=True)
        save(out/'status.json',status)
        deadline=started+22*60;best={};history=[];completed=0;duration=0;partial=False;updates=0
        for epoch in range(1 if args.smoke else plan['max_epochs']):
            if epoch and time.monotonic()+duration*1.1>deadline:break
            es=time.monotonic();model.train();seen=0;loss_sum=0.;hard_sum=torch.zeros(12,device='cuda',dtype=torch.float64)
            component_sum=torch.zeros(7,device='cuda',dtype=torch.float64)
            gen=torch.Generator(device='cuda').manual_seed(42+epoch)
            order=torch.randperm(plan['train_rows'],device='cuda',generator=gen)
            for start in range(0,len(order),plan['batch_size']):
                idx=order[start:start+plan['batch_size']];c=t['train_condition'][idx];y=t['train_target_z'][idx]
                pred,delta,features=model(c);point,shape=point_shape(pred,y);loss=point+.1*shape
                component_sum[:2]+=torch.stack([point.detach(),shape.detach()]).double()*len(idx)
                if cfg['ot']:
                    term=conditional_ot_loss(pred,y,t['train_groups'][idx],group_prob,directions)
                    loss=loss+plan['cot']['weight']*term;component_sum[2]+=term.detach().double()*len(idx)
                if cfg['proxy']:
                    penalty,hard=proxy(pred,y);loss=loss+penalty;hard_sum+=hard.double()*len(idx);component_sum[3]+=penalty.detach().double()*len(idx)
                if cfg['rank']:
                    subset=torch.randperm(len(idx),device='cuda',generator=rank_generator)[:min(256,len(idx))]
                    term=rank_similarity_loss(features[subset],y[subset,0])
                    loss=loss+plan['ranksim']['weight']*term;component_sum[4]+=term.detach().double()*len(idx)
                if cfg['flow']:
                    term=path_consistency_loss(field,c,c[:,-4:],pred,flow_generator)
                    loss=loss+plan['flowalign']['weight']*term;component_sum[5]+=term.detach().double()*len(idx)
                if not torch.isfinite(loss):raise FloatingPointError('Nonfinite correction objective')
                optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.);optimizer.step()
                seen+=len(idx);updates+=1;loss_sum+=float(loss.detach())*len(idx)
                if time.monotonic()>deadline and seen<len(order):partial=True;break
            if partial:break
            assert seen==plan['train_rows']
            if cfg['proxy']:proxy.update((hard_sum/seen).float())
            pred_phys,metrics,selection=evaluate(model,t,arrays,manifest)
            error=(pred_phys-np.asarray(arrays['val_truth']))/np.asarray(manifest['effective_std'])
            mse=float(np.mean(error**2))
            info=dict(epoch=epoch,train_rows=seen,val_rows=plan['val_rows'],train_loss=loss_sum/seen,
                val_mse=mse,metrics=metrics,selection=selection,
                raw_loss_components=dict(zip(['point','emd','conditional_ot','proxy_penalty','ranksim','flowalign'],(component_sum[:6]/seen).cpu().tolist())),
                proxy=proxy.diagnostics() if cfg['proxy'] else None)
            for label,rank in [('primary',tuple(selection['primary_rank'])),('score',(selection['joint_score'],)),('mse',(mse,))]:
                if label not in best or rank<tuple(best[label]['rank']):
                    best[label]=info|dict(rank=list(rank))
                    checkpoint(out/'checkpoints'/(label+'.pt'),model,dict(epoch=epoch,config=cfg,prepared_manifest_sha256=digest(args.prepared/'manifest.json')))
            completed+=1;duration=time.monotonic()-es;info['seconds']=duration;history.append(info)
            save(out/'history.json',history);save(out/'checkpoint_selection.json',best)
            print(json.dumps(dict(run=cfg['run_id'],epoch=epoch,seconds=duration,selection=selection,proxy=info['proxy'])),flush=True)
        if not completed:raise RuntimeError('No complete full-data correction epoch')
        for label in ['primary','mse']:
            cp=out/'checkpoints'/(label+'.pt');payload=torch.load(cp,map_location='cpu',weights_only=False);model.load_state_dict(payload['state_dict'])
            pred,metrics,selection=evaluate(model,t,arrays,manifest)
            for k in KEYS:
                for metric in ['res','local_trace','w1']:
                    if abs(metrics['observables'][k][metric]-best[label]['metrics']['observables'][k][metric])>1e-10:raise RuntimeError('Final inference mismatch')
            dest=out/('predictions_val.parquet' if label=='primary' else 'predictions_val_mse.parquet')
            write_predictions(dest,pred,arrays['val_truth'])
            metrics.update(selection=selection,checkpoint=dict(epoch=payload['epoch'],sha256=digest(cp),path=str(cp)),
                diagnostics=diagnostics(pred,arrays['val_truth']),
                correction=dict(mean_delta=(pred-arrays['val_point']).mean(0).tolist(),mean_abs_delta=np.abs(pred-arrays['val_point']).mean(0).tolist()),
                predictions=dict(rows=len(pred),path=str(dest),sha256=digest(dest),roundtrip_exact=True))
            save(out/('metrics.json' if label=='primary' else 'metrics_mse.json'),metrics)
        status.update(status='smoke_passed' if args.smoke else 'complete',completed_full_epochs=completed,
            partial_final_epoch=partial,updates=updates,primary_eligible=best['primary']['selection']['eligible'],
            best_epoch=best['primary']['epoch'],elapsed_seconds=time.monotonic()-started,
            primary_prediction_sha256=digest(out/'predictions_val.parquet'),prediction_rows=plan['val_rows'])
        save(out/'status.json',status)
    except Exception:
        status.update(status='failed',error=traceback.format_exc());save(out/'status.json',status);raise

if __name__=='__main__':main()
