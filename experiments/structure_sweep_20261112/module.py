"""Same full-data optimizer and primary four-observable loss in every arm."""
import torch
from hepsetreg.training.lightning_module import EventRegressionModule
from models import objective_components


class ObjectiveModule(EventRegressionModule):
    def __init__(self,model,config):
        super().__init__(model,learning_rate=config['learning_rate'],weight_decay=config['weight_decay'],loss_kind='mse')
        self.config=config
        self.adaptive=None  # shared complete-epoch callback interface
        self.validation_predictions=[]
        self.validation_targets=[]

    def _shared_step(self,batch,stage):
        details=self.model.forward_details(batch)
        components=objective_components(self.model,details,batch)
        for name,value in components.items():
            self.log(stage+'_'+name,value,on_step=False,on_epoch=True,
                     batch_size=len(batch['target']),prog_bar=False)
        if stage=='val':
            self.validation_predictions.append(details['prediction'].detach().float().cpu())
            self.validation_targets.append(batch['target'].detach().float().cpu())
        return components['loss']

    def on_validation_epoch_start(self):
        self.validation_predictions.clear();self.validation_targets.clear()

    def configure_optimizers(self):
        return torch.optim.AdamW(self.model.parameters(),lr=self.config['learning_rate'],
                                 weight_decay=self.config['weight_decay'],fused=True)
