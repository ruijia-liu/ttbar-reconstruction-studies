"""Shared network and EMD; only the declared paired/auxiliary loss changes."""
import torch
from hepsetreg.training.lightning_module import EventRegressionModule
from losses import point_loss,emd_loss
from adaptive_loss import AdaptiveRobustLoss

class ObjectiveModule(EventRegressionModule):
    def __init__(self,model,config,migration):
        super().__init__(model,learning_rate=config['learning_rate'],weight_decay=config['weight_decay'],loss_kind='mse')
        self.config=config
        self.migration=migration
        self.adaptive=AdaptiveRobustLoss(dimensions=4) if config['adaptive'] else None
        self.validation_predictions=[]
        self.validation_targets=[]

    def _shared_step(self,batch,stage):
        pred=self.model(batch)
        with torch.autocast(device_type=pred.device.type,enabled=False):
            pred,target=pred.float(),batch['target'].float()
            paired=self.adaptive(pred,target) if self.adaptive is not None else 2*point_loss(pred,target,'huber',delta=1.)
            shape=emd_loss(pred,target)
            migration=self.migration(pred,target) if self.config['migration_weight'] else pred.new_zeros(())
            total=paired+.1*shape+self.config['migration_weight']*migration
            mse=(pred-target).square().mean()
        if not torch.isfinite(total):raise FloatingPointError('Nonfinite training/validation loss')
        for name,value in [('loss',total),('point',paired),('emd',shape),('migration',migration),('mse',mse)]:
            self.log(stage+'_'+name,value,on_step=False,on_epoch=True,batch_size=len(target),prog_bar=False)
        if stage=='val':
            self.validation_predictions.append(pred.detach().cpu())
            self.validation_targets.append(target.detach().cpu())
        return total

    def on_validation_epoch_start(self):
        self.validation_predictions.clear();self.validation_targets.clear()

    def configure_optimizers(self):
        groups=[dict(params=self.model.parameters(),lr=self.config['learning_rate'],weight_decay=self.config['weight_decay'])]
        if self.adaptive is not None:
            groups.append(dict(params=self.adaptive.parameters(),lr=.001,weight_decay=0.))
        return torch.optim.AdamW(groups,fused=True)
