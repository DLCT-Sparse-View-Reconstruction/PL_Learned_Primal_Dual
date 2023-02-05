import torch
from torch import nn

import odl.contrib.torch as odl_torch

from models.PrimalDualNet import *

class LearnedPrimalDual(nn.Module):
    def __init__(self,
                forward_op, backward_op, n_iter, n_primal, n_dual,
                primal_architecture = PrimalNet,
                dual_architecture = DualNet):

        super(LearnedPrimalDual, self).__init__()
        
        self.forward_op = forward_op
        self.primal_architecture = primal_architecture
        self.dual_architecture = dual_architecture
        self.n_iter = n_iter
        self.n_primal = n_primal
        self.n_dual = n_dual
        
        self.primal_shape = (n_primal,) + forward_op.domain.shape
        self.dual_shape = (n_dual,) + forward_op.range.shape
        
        self.primal_op_layer = odl_torch.OperatorModule(forward_op)
        self.dual_op_layer = odl_torch.OperatorModule(backward_op)
        
        self.primal_nets = nn.ModuleList()
        self.dual_nets = nn.ModuleList()
        
        for i in range(n_iter):
            self.primal_nets.append(
                primal_architecture(n_primal)
            )
            self.dual_nets.append(
                dual_architecture(n_dual)
            )
        
    def forward(self, g, intermediate_values = False):
        h = torch.zeros(g.shape[0:1] + (self.dual_shape), device=g.device)
        f = torch.zeros(g.shape[0:1] + (self.primal_shape), device=g.device)
        
        if intermediate_values:
            h_values = []
            f_values = []
            
        for i in range(self.n_iter):
            ## Dual
            # Apply forward operator to f^(2)
            f_2 = f[:,1:2]
            if intermediate_values:
                f_values.append(f)
            Op_f = self.primal_op_layer(f_2)
            # Apply dual network
            h = self.dual_nets[i](h, Op_f, g)
            
            ## Primal
            # Apply adjoint operator to h^(1)
            h_1 = h[:,0:1]
            if intermediate_values:
                h_values.append(h)
            OpAdj_h = self.dual_op_layer(h_1)
            # Apply primal network
            f = self.primal_nets[i](f, OpAdj_h)
        
        if intermediate_values:
            return f[:,0:1], f_values, h_values
        return f[:,0:1]
