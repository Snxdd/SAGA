import torch
import copy
import numpy as np
from torch.optim.optimizer import Optimizer, required

class SAGA(Optimizer):
    """
    n_classes : int
        the number of gradients that must be stored
    beta1 : float
        the "usual" sgd momentum value used
    momentum : float
        momentum on stored gradients (1 means no momentum; the lower the smaller the update)
    class_proba : list (floats)
        the probabilities of each table entry, used to remove bias if each entry has a different number of samples
    initial_gradients : list 
        if we want to initialize gradient table other than to 0; most likely not used
    compute_var : boolean
        if we want the optimizer to return the gradient variance
    """
    
    def __init__(self, params, n_classes = required, lr=required, beta1 = 0, momentum = 1,
                 class_proba = None, initial_gradients = None, compute_var = False, lr_schedule = False):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        defaults = dict(lr=lr, initial_gradients = initial_gradients,beta1=beta1,)
        super(SAGA, self).__init__(params, defaults)
        
        #initialize avg and past gradients to 0
        if initial_gradients is None:
            #param_groups assumed to have only 1 group
            for group in self.param_groups:
                self.avg_gradients = (self.init_params(copy.deepcopy(group['params'])))
                self.past_gradients = [self.init_params(copy.deepcopy(group['params'])) for i in range(n_classes)]
        else:
            #If want to initialize the gradient table, probably not the case
            pass
        
        #as attributes for convenience, but should be added to defaults
        self.n_classes = n_classes 
        self.momentum = momentum
        self.class_proba = class_proba
        self.beta1 = beta1
        self.compute_var = compute_var
        self.lr_schedule = lr_schedule

    def __setstate__(self, state):
        super(SAGA, self).__setstate__(state)
        
    #set all params to 0
    def init_params(self,params):
        for param in params:
            param.data.fill_(0)
        return params

    #use closure to get entry index of example
    def step(self, closure=None, epoch = None):
        loss = None
        if closure is None:
            raise ValueError("Need index of example")

        idx = closure
        grad_var = 0
        #param_groups assumed to have only 1 group for now
        for group in self.param_groups:
            #to test different lr schedulings, ignore for algorithm correctness
            if (self.lr_schedule):
                if (epoch == 5000 or epoch == 10000):
                    group['lr'] = group['lr']/3
            
            for (p,past_grad,avg_grad) in zip(group['params'],self.past_gradients[idx],self.avg_gradients):
                if p.grad is None:
                    continue
                
                #Bias correction
                bias_corr = 1
                if self.class_proba is not None:
                    bias_corr = 1/(self.n_classes*self.class_proba[idx])
                
                #variance reduction and update table
                d_p = p.grad.data - bias_corr*past_grad.data + avg_grad.data
                avg_grad.data -= (past_grad.data)/self.n_classes
                past_grad.data = self.momentum*p.grad.data.clone() + (1-self.momentum)*past_grad.data.clone()
                avg_grad.data += past_grad.data/self.n_classes

                if (self.compute_var):
                    grad_var += (d_p**2).sum()
                
                #momentum, same code as in pytorch SGD optimizer
                if self.beta1 != 0:
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