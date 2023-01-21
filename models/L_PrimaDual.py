import numpy as np
import random

import torch
from torch import nn

from pytorch_lightning import LightningModule

from torchvision.utils import save_image

import odl
from odl.contrib import torch as odl_torch

from util.utils import *

from models.LearnedPrimalDual import LearnedPrimalDual

from torchmetrics import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure, MeanSquaredError

class L_PrimalDual(LightningModule):
    
    STEPS_CONFIGS = [4, 8]
    
    def __init__(self, image_size, n_its, n_primal, n_dual, lr):
        super(L_PrimalDual, self).__init__()
        
        self.save_hyperparameters()
        
        self.automatic_optimization = False
        
        self.n = image_size
        self.space = odl.uniform_discr([-128, -128], [128, 128], [self.n, self.n], dtype='float32', weighting=1.0)
        
        angle_partition = odl.uniform_partition(0, 2 * np.pi, 512)
        detector_partition = odl.uniform_partition(-360, 360, 512)
        geometry = odl.tomo.geometry.conebeam.FanBeamGeometry(angle_partition, detector_partition, src_radius=256, det_radius=256)
        
        operator = odl.tomo.RayTransform(self.space, geometry)
        self.fp_operator = odl_torch.OperatorModule(operator)
        self.fbp_operator = odl_torch.OperatorModule(odl.tomo.fbp_op(operator))

        self.lpd_model = LearnedPrimalDual(self.fp_operator, self.fbp_operator, n_iter=n_its, n_primal=n_primal, n_dual=n_dual)

        self.criterion = nn.MSELoss()

    def forward(self, d):
        return self.lpd_model(d)
    
    def LPD_loss(self, y, y_true):
        return self.criterion(y, y_true)
    
    def step_random_choose(self):
        return random.choice(self.STEPS_CONFIGS)    
    
    def creat_mask(self, proj_data, step):
        theta = np.linspace(0, 512, 512, endpoint=False).astype(int)
        theta_lack = np.linspace(0, 512, int(512/step), endpoint=False).astype(int)
    
        mask_array = np.setdiff1d(theta, theta_lack).astype(int)
    
        mask = np.array(proj_data) != 0
    
        mask[mask_array, :] = 0
    
        return torch.Tensor(mask)    
    
    def fp_ct(self, y_true):
        yy_true = self.space.element(y_true.cpu())
        phantom = torch.Tensor(yy_true)[None, ...]
        
        proj_data = self.fp_operator(phantom)
        
        return proj_data
        
    def get_sparce_sinogram(self, y_true, step):
        proj_data = self.fp_ct(y_true)
        mask = self.creat_mask(torch.squeeze(proj_data), step)
        
        return mask * proj_data
    
    def training_step(self, batch, batch_idx):
        opt_lpd = self.optimizers()
        
        y_true = torch.rot90(torch.squeeze(batch), -1)
        
        step = self.step_random_choose()
        d_sparce = self.get_sparce_sinogram(y_true, step).type_as(y_true)
            
        y = self(d_sparce)

        ######################
        # Optimize LEARN   #
        ######################
        # compute losses
        lpd_loss = self.LPD_loss(y, y_true[None, None, ...])
        
        opt_lpd.zero_grad()
        self.manual_backward(lpd_loss)
        opt_lpd.step()
            
        self.log_dict({"lpd_loss": lpd_loss}, prog_bar=True, sync_dist=True)
            
    def validation_step(self, batch, batch_idx):
        y_true = torch.rot90(torch.squeeze(batch), -1)
        
        step = self.step_random_choose()
        d_sparce = self.get_sparce_sinogram(y_true, step).type_as(y_true)
        
        y = self.lpd_model(d_sparce)
        val_lpd_loss = self.LPD_loss(y, y_true[None, None, ...])
        
        self.log_dict({"val_lpd_loss": val_lpd_loss}, prog_bar=True, sync_dist=True)
        
    def test_step(self, batch, batch_idx):
        psnr, ssim, mse = PeakSignalNoiseRatio().to(device=batch.device), StructuralSimilarityIndexMeasure(data_range=1.0).to(device=batch.device), MeanSquaredError().to(device=batch.device)        
        y_true = torch.rot90(torch.squeeze(batch), -1)
        
        step = self.step_random_choose()
        
        d_sparce = self.get_sparce_sinogram(y_true, step).type_as(y_true)
        
        with torch.no_grad():
            self.eval()
            y = self(d_sparce)
            y_true = y_true[None, None, ...]
            self.train()
            
            test_lpd_loss = self.LPD_loss(y, y_true)
            
            self.log_dict({"test_lpd_loss": test_lpd_loss}, prog_bar=True)
            
            psnr_p, ssim_p, rmse_p = psnr(y, y_true).cpu(), ssim(y, y_true).cpu(), torch.sqrt(mse(y, y_true).cpu()) 
            file_name = "results/test/LPD/{}/idx_{}_psnr={}_ssim={}_rmse={}.png".format(step, batch_idx, 
                                                                                    psnr_p.numpy(), ssim_p.numpy(), rmse_p.numpy())
            
            y = torch.rot90(torch.squeeze(y), 1)
            save_image(y, file_name)             

    def configure_optimizers(self):
        lr = self.hparams.lr

        opt_lpd = torch.optim.Adam(self.learn_model.parameters(), lr=lr) 

        return opt_lpd
