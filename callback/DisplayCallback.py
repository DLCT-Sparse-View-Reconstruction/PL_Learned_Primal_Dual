import torch
from torch import nn

from pytorch_lightning import Callback

from util.utils import *

class DisplayCallback(Callback):
    def __init__(self, every_n_epochs=5):
        super(DisplayCallback, self).__init__()
        self.every_n_epochs = every_n_epochs

        self.epoch_save_dic = "results/epochs/"
        if not os.path.exists(self.epoch_save_dic):
            os.makedirs(self.epoch_save_dic)

    def on_train_epoch_end(self, trainer, pl_module, *args):
        val_dataloader = trainer.val_dataloaders[0]
        val_dataset = val_dataloader.dataset

        y_true = torch.rot90(torch.squeeze(next(iter(val_dataset))), -1)
        step = 8
        d = pl_module.get_sparce_sinogram(y_true, step)
                
        # Reconstruct images
        d = d.to(pl_module.device)
        with torch.no_grad():
            pl_module.eval()
            y = pl_module(d[None, ...])
            pl_module.train()
                
            display_list = [step, d, y_true[None, None, ...], y]
            display_func(display_list, self.epoch_save_dic, pl_module.current_epoch)
