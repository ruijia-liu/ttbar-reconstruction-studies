"""Reject partial epochs; predeclared selection on full validation metrics."""
import json
import time
import numpy as np
import torch
import pytorch_lightning as pl
from evaluation import compute_metrics,select_metrics,save

def complete_epoch(seen_rows,expected_rows,seen_batches,expected_batches):
    return seen_rows==expected_rows and seen_batches==expected_batches

class BudgetSelection(pl.Callback):
    def __init__(self,started,out,plan,truth,mean,std,smoke=False):
        self.started=started;self.out=out;self.plan=plan;self.truth=truth
        self.mean=mean.cpu();self.std=std.cpu();self.smoke=smoke
        self.completed=0;self.partial=False;self.epoch_seconds=[];self.best={};self.history=[]
        (out/'checkpoints').mkdir()

    def on_train_epoch_start(self,trainer,module):
        self.epoch_start=time.monotonic();self.seen_rows=0;self.seen_batches=0

    def on_train_batch_end(self,trainer,module,outputs,batch,batch_idx):
        self.seen_rows+=len(batch['target']);self.seen_batches+=1
        if time.monotonic()-self.started>self.plan['training_and_setup_budget_minutes']*60:
            trainer.should_stop=True

    def on_validation_end(self,trainer,module):
        if trainer.sanity_checking:return
        full=complete_epoch(self.seen_rows,self.plan['train_rows'],self.seen_batches,trainer.num_training_batches)
        if not full:
            self.partial=True
            module.validation_predictions.clear();module.validation_targets.clear()
            print('Partial epoch excluded from checkpoint selection',flush=True)
            return
        pz=torch.cat(module.validation_predictions);tz=torch.cat(module.validation_targets)
        module.validation_predictions.clear();module.validation_targets.clear()
        pred=(pz*(self.std+1e-8)+self.mean).numpy()
        target=(tz*(self.std+1e-8)+self.mean).numpy()
        if pred.shape!=(self.plan['val_rows'],4):raise RuntimeError('Incomplete validation pass')
        if not np.all(np.max(np.abs(target-self.truth),axis=0)<=np.array([.01,5e-4,5e-4,5e-4])):
            raise RuntimeError('Per-epoch validation row alignment mismatch')
        metrics=compute_metrics(pred,self.truth)
        selection=select_metrics(metrics,self.plan['baseline_metrics'])
        mse=float((pz-tz).square().mean())
        entry=dict(epoch=int(trainer.current_epoch),global_step=int(trainer.global_step),
            train_rows=self.seen_rows,val_rows=len(pred),mse=mse,metrics=metrics,selection=selection,
            adaptive=module.adaptive.diagnostics() if module.adaptive is not None else None)
        for label,rank in [('primary',tuple(selection['primary_rank'])),('score',(selection['joint_score'],)),('mse',(mse,))]:
            if label not in self.best or rank<tuple(self.best[label]['rank']):
                dest=self.out/'checkpoints'/(label+'.ckpt');tmp=dest.with_suffix('.tmp')
                trainer.save_checkpoint(tmp)
                tmp.replace(dest)
                self.best[label]=entry|dict(rank=list(rank))
        self.completed+=1;self.history.append(entry)
        save(self.out/'validation_history.json',self.history)
        save(self.out/'checkpoint_selection.json',self.best)
        duration=time.monotonic()-self.epoch_start;self.epoch_seconds.append(duration)
        print(json.dumps(dict(epoch=trainer.current_epoch,full_epochs=self.completed,seconds=round(duration,2),
            selection=selection,adaptive=entry['adaptive'])),flush=True)
        elapsed=time.monotonic()-self.started
        # Reserve a full recent epoch estimate plus margin; 8 minutes remain for export.
        estimate=max(self.epoch_seconds[-3:])*1.1
        if elapsed+estimate>self.plan['training_and_setup_budget_minutes']*60:trainer.should_stop=True
