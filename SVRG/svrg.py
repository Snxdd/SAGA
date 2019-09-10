import torch
import copy
import math
from torch.optim.optimizer import Optimizer, required

class SVRG(Optimizer):
    """
    compute_var : boolean
        if we want the optimizer to return the gradient variance
    use_adam : boolean
        will use adam if set to True
    """
    
    def __init__(self, params, lr=required, beta1 = 0,  betas=(0.9, 0.999),eps = 1e-8 ,weight_decay=0,use_adam = False,compute_var = True):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))     
            
            
        defaults = dict(lr=lr, 
                        beta1=beta1,betas=betas,eps=eps,weight_decay=weight_decay)
        super(SVRG, self).__init__(params, defaults)
        
        
        #as attributes for convenience, but could be added to defaults
        self.beta1 = beta1 #ignore beta1
        self.compute_var = compute_var
        self.use_adam = use_adam
        

    def __setstate__(self, state):
        super(SVRG, self).__setstate__(state)
        
   
    def step(self, snap_grad, snap_avg_grad, closure=None):
        loss = None
 
        grad_var = 0
        #param_groups assumed to have only 1 group for now
        for group in self.param_groups:
            weight_decay = group['weight_decay']
            for (p,s_grad,s_avg_grad) in zip(group['params'],snap_grad,snap_avg_grad):
                if p.grad is None:
                    continue
                
                #variance reduction
                d_p = p.grad.data - s_grad.grad.data + s_avg_grad.grad.data

                if (self.compute_var):
                    grad_var += (d_p**2).sum()
                    
                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)
                
                if (self.use_adam):
                    #Adam, same code as in pytorch Adam optimizer and modified SAGA
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
                else:
                    #momentum, same code as in pytorch SGD optimizer
                    if (self.beta1 != 0):
                        param_state = self.state[p]
                        if 'momentum_buffer' not in param_state:
                            buf = param_state['momentum_buffer'] = torch.zeros_like(p.data)
                            buf.mul_(self.beta1).add_(d_p)
                        else:
                            buf = param_state['momentum_buffer']
                            buf.mul_(self.beta1).add_(1, d_p)
                        d_p = buf
                    p.data.add_(-group['lr'], d_p)
 
        return loss, grad_var