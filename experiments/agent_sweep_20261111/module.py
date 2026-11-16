"""A fixed four-output regressor trained with frozen per-event mass-loss gates."""
import torch
from hepsetreg.training.lightning_module import EventRegressionModule
from gated_loss import gated_loss


class GatedRegressionModule(EventRegressionModule):
    def __init__(self,model,config):
        super().__init__(model,learning_rate=config['learning_rate'],
                         weight_decay=config['weight_decay'],loss_kind='mse')
        self.config=config

    def forward(self,batch):
        # Targets and truth-derived loss weights are never model inputs.
        inputs={key:batch[key] for key in ['global','jet','jet_mask','lepton','lepton_mask']}
        return self.model(inputs)

    def _shared_step(self,batch,stage):
        pred=self(batch)
        with torch.autocast(device_type=pred.device.type,enabled=False):
            pred,target=pred.float(),batch['target'].float()
            gate=batch['mass_gate'].float().detach()
            loss=gated_loss(pred,target,gate,delta=self.config['delta'])
            mse=(pred-target).square().mean()
            mass_mse=(pred[:,0]-target[:,0]).square().mean()
        for name,value in [('loss',loss),('mse',mse),('mass_mse',mass_mse),('gate_mean',gate.mean())]:
            self.log(f'{stage}_{name}',value,on_step=False,on_epoch=True,
                     batch_size=target.shape[0],prog_bar=False)
        return loss
