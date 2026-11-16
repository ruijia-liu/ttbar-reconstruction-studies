"""Same four-output network; only training objective varies between trials."""
import torch
from hepsetreg.training.lightning_module import EventRegressionModule
from losses import point_loss, emd_loss, sliced_wasserstein_loss, make_directions


class LossRegressionModule(EventRegressionModule):
    def __init__(self, model, config):
        super().__init__(model, learning_rate=config['learning_rate'],
                         weight_decay=config['weight_decay'], loss_kind='mse')
        self.config = config
        self.register_buffer('directions', make_directions(count=32, dimensions=4, seed=4201))

    def _shared_step(self, batch, stage):
        pred = self.model(batch)
        # Autocast must also be disabled for projection matmul and reductions.
        with torch.autocast(device_type=pred.device.type, enabled=False):
            pred, target = pred.float(), batch['target'].float()
            cfg = self.config
            paired = cfg['point_weight'] * point_loss(pred, target, cfg['loss_kind'],
                                                      delta=cfg['delta'], scale=cfg['scale'])
            shape = pred.new_zeros(())
            if cfg['shape_kind'] == 'emd':
                shape = emd_loss(pred, target)
            elif cfg['shape_kind'] == 'swd':
                shape = sliced_wasserstein_loss(pred, target, self.directions)
            loss = paired + cfg['shape_weight'] * shape
            mse = (pred-target).square().mean()
        for name, value in [('loss', loss), ('point_loss', paired), ('shape_loss', shape), ('mse', mse)]:
            self.log(f'{stage}_{name}', value, on_step=False, on_epoch=True,
                     batch_size=target.shape[0], prog_bar=False)
        return loss
