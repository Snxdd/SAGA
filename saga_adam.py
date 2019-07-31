import torch
import copy
import numpy as np
import math
from torch.optim.optimizer import Optimizer, required

class SAGA_adam(Optimizer):
    """
    See saga.py for arguments
    eps : float
        numerical stability term used in Adam
    """
    
    def __init__(self, params,model, n_classes = required, lr=required, betas=(0.9, 0.999),eps = 1e-8, momentum = 1,
                 class_proba = None, initial_gradients = None, compute_var = False, lr_schedule = False):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        defaults = dict(lr=lr, initial_gradients = initial_gradients,betas=betas,eps=eps)
        super(SAGA_adam, self).__init__(params, defaults)
        if initial_gradients is None:
            for group in self.param_groups:
                self.avg_gradients = (self.init_params(copy.deepcopy(group['params'])))
                self.past_gradients = [self.init_params(copy.deepcopy(group['params'])) for i in range(n_classes)]
        else:
            pass
        self.n_classes = n_classes
        self.momentum = momentum
        self.class_proba = class_proba
        self.compute_var = compute_var
        self.lr_schedule = lr_schedule

    def __setstate__(self, state):
        super(SAGA_adam, self).__setstate__(state)
        

    def init_params(self,params):
        for param in params:
            param.data.fill_(0)
        return params

    def step(self, closure=None, epoch = None):
        loss = None
        if closure is None:
            raise ValueError("Need index of class")

        idx = closure
        grad_var = 0
        for group in self.param_groups:
            if (self.lr_schedule):
                if (epoch % 1000 == 0 and epoch != 0):
                    #group['lr'] = group['lr']/np.sqrt(epoch/2000+1)
                    group['lr'] = group['lr']/np.sqrt(epoch/1000)
            for (p,past_grad,avg_grad) in zip(group['params'],self.past_gradients[idx],self.avg_gradients):
                if p.grad is None:
                    continue
                
                bias_corr = 1
                if self.class_proba is not None:
                    bias_corr = 1/(self.n_classes*self.class_proba[idx])
                d_p = p.grad.data - bias_corr*past_grad.data + avg_grad.data
                avg_grad.data -= (past_grad.data)/self.n_classes
                past_grad.data = self.momentum*p.grad.data.clone() + (1-self.momentum)*past_grad.data.clone()
                avg_grad.data += past_grad.data/self.n_classes
                if (self.compute_var):
                    grad_var += (d_p**2).sum()
                
                #ADAM, same code as in pytorch adam optimizer
                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']
                state['step'] += 1                   
                
                exp_avg.mul_(beta1).add_(1 - beta1, d_p)
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, d_p, d_p)
                denom = exp_avg_sq.sqrt().add_(group['eps'])
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1
                p.data.addcdiv_(-step_size, exp_avg, denom)
                
                
        return loss, grad_var