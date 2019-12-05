import torch
import copy
import math
from torch.optim.optimizer import Optimizer, required

class SAGA(Optimizer):
    """
    n_classes : int
        the number of gradients that must be stored
    beta1 : float
        the "usual" sgd momentum value used
    betas : float
        betas used for Adam
    momentum : float
        momentum on stored gradients (1 means no momentum)
    class_proba : list (floats)
        the probabilities of each table entry, used to remove bias if each entry has a different number of samples
    initial_gradients : list 
        if we want to initialize gradient table other than to 0; most likely not used
    compute_var : boolean
        if we want the optimizer to return the gradient variance
    use_adam : boolean
        will use Adam update if set to True
    """
    
    def __init__(self, params, n_classes = required, lr=required, beta1 = 0, momentum = 0.1, betas=(0.9, 0.999),eps = 1e-8,
                 class_proba = None, initial_gradients = None, compute_var = False ,weight_decay=0,use_adam = False):
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
            
            
        defaults = dict(lr=lr, initial_gradients = initial_gradients,
                        beta1=beta1,betas=betas,eps=eps,weight_decay=weight_decay)
        super(SAGA, self).__init__(params, defaults)
        
        #initialize avg and past gradients to 0
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if initial_gradients is None:
#             param_groups assumed to have only 1 group
            for group in self.param_groups:
                self.avg_gradients = (self._init_params(copy.deepcopy(group['params'])))
                self.past_gradients = [self._init_params(copy.deepcopy(group['params'])) for i in range(n_classes)]
        else:
            #if want to initialize the gradient table, probably not the case
            pass
        
        #as attributes for convenience, but could be added to defaults
        self.n_classes = n_classes 
        self.momentum = momentum
        self.class_proba = class_proba
        self.beta1 = beta1
        self.compute_var = compute_var
        self.use_adam = use_adam
        

    def __setstate__(self, state):
        super(SAGA, self).__setstate__(state)
        
    def _init_params(self,params):
        for param in params:
            param.data.fill_(0)
        return params
    
   
    def step(self, idx, closure=None):
        loss = None
        if idx is None:
            raise ValueError("Need index of example")

        grad_var = 0
        #param_groups assumed to have only 1 group for now
        for group in self.param_groups:
            weight_decay = group['weight_decay']
            for (p,past_grad,avg_grad) in zip(group['params'],self.past_gradients[idx],self.avg_gradients):
                if p.grad is None:
                    continue
                
                #bias correction
                bias_corr = 1
                if self.class_proba is not None:
                    bias_corr = 1/(self.n_classes*self.class_proba[idx])
                
                #variance reduction and update table
                d_p = p.grad.data - bias_corr*past_grad.data + avg_grad.data
                avg_grad.data -= (past_grad.data)/self.n_classes
                past_grad.data = self.momentum*p.grad.data + (1-self.momentum)*past_grad.data
                avg_grad.data += past_grad.data/self.n_classes

                if (self.compute_var):
                    grad_var += (d_p**2).sum()
                    
                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)
                
                if (self.use_adam):
                    #Adam, same code as in pytorch Adam optimizer
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