"""Full-data FlowAlign velocity-field stage; no test data."""
import argparse,json,os,time,traceback
from pathlib import Path
import torch
from flowalign import VelocityField,field_training_loss
from common import load_data,checkpoint
from evaluation import save,digest

def main():
    p=argparse.ArgumentParser();p.add_argument('--prepared',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();started=time.monotonic()
    if not os.environ.get('SLURM_JOB_ID') or not torch.cuda.is_available():raise RuntimeError('Slurm CUDA required')
    plan=json.loads(Path(__file__).with_name('experiment_plan.json').read_text())
    args.output.mkdir(parents=True,exist_ok=False)
    status=dict(status='running',slurm_job_id=os.environ['SLURM_JOB_ID']);save(args.output/'status.json',status)
    try:
        torch.set_num_threads(4);torch.set_float32_matmul_precision('highest')
        torch.manual_seed(plan['flowalign']['seed']);torch.cuda.manual_seed_all(plan['flowalign']['seed'])
        manifest,arrays,t=load_data(args.prepared,'cuda')
        field=VelocityField(condition_dim=132,hidden=128).cuda()
        optimizer=torch.optim.AdamW(field.parameters(),lr=plan['learning_rate'],weight_decay=plan['weight_decay'])
        generator=torch.Generator(device='cuda').manual_seed(4209)
        deadline=started+22*60;best=float('inf');completed=0;history=[];duration=0;partial=False
        for epoch in range(plan['flowalign']['max_epochs']):
            if epoch and time.monotonic()+duration*1.1>deadline:break
            es=time.monotonic();field.train()
            shuffle=torch.Generator(device='cuda').manual_seed(42+epoch)
            order=torch.randperm(plan['train_rows'],device='cuda',generator=shuffle)
            seen=0;total=0.
            for start in range(0,len(order),plan['batch_size']):
                idx=order[start:start+plan['batch_size']];c=t['train_condition'][idx];y=t['train_target_z'][idx]
                optimizer.zero_grad(set_to_none=True)
                loss=field_training_loss(field,c,c[:,-4:],y,generator)
                if not torch.isfinite(loss):raise FloatingPointError('Nonfinite field loss')
                loss.backward();torch.nn.utils.clip_grad_norm_(field.parameters(),5.);optimizer.step()
                seen+=len(idx);total+=float(loss.detach())*len(idx)
                if time.monotonic()>deadline and seen<len(order):partial=True;break
            if partial:break
            assert seen==plan['train_rows'];field.eval();vsum=0.;vseen=0
            vg=torch.Generator(device='cuda').manual_seed(4210)
            with torch.inference_mode():
                for start in range(0,plan['val_rows'],plan['batch_size']):
                    c=t['val_condition'][start:start+plan['batch_size']];y=t['val_target_z'][start:start+plan['batch_size']]
                    vl=field_training_loss(field,c,c[:,-4:],y,vg)
                    if not torch.isfinite(vl):raise FloatingPointError('Nonfinite field validation')
                    vsum+=float(vl)*len(c);vseen+=len(c)
            assert vseen==plan['val_rows'];value=vsum/vseen;completed+=1
            info=dict(epoch=epoch,train_rows=seen,val_rows=vseen,train_loss=total/seen,val_velocity_mse=value)
            if value<best:
                best=value;checkpoint(args.output/'best.pt',field,dict(**info,prepared_manifest_sha256=digest(args.prepared/'manifest.json'),field_config=dict(condition_dim=132,hidden=128)))
            duration=time.monotonic()-es;info['seconds']=duration;history.append(info)
            save(args.output/'history.json',history);print(json.dumps(info),flush=True)
        if not completed:raise RuntimeError('No complete field training epoch')
        status.update(status='complete',completed_full_epochs=completed,partial_final_epoch=partial,
            best_val_velocity_mse=best,best_checkpoint_sha256=digest(args.output/'best.pt'),
            prepared_manifest_sha256=digest(args.prepared/'manifest.json'),elapsed_seconds=time.monotonic()-started,
            train_rows=plan['train_rows'],val_rows=plan['val_rows'],test_evaluation=False)
        save(args.output/'status.json',status)
    except Exception:
        status.update(status='failed',error=traceback.format_exc());save(args.output/'status.json',status);raise

if __name__=='__main__':main()
