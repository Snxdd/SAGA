import torch
import copy
from torch.optim.optimizer import Optimizer, required

class SAGA(Optimizer):
    def __init__(self, params,model, n_samples=required, lr=required, initial_gradients = None):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        defaults = dict(lr=lr, initial_gradients = initial_gradients)
        super(SAGA, self).__init__(params, defaults)
        #initialize avg and past gradients to 0
        if initial_gradients is None:
            #copy entire model for now
            self.avg_gradients = self.init_params(copy.deepcopy(model))
            self.past_gradients = [self.init_params(copy.deepcopy(model)) for i in range(n_samples)]
        else:
            #to implement
            pass
        self.n_samples = n_samples #to add to defaults

    def __setstate__(self, state):
        super(SAGA, self).__setstate__(state)
        
    def init_params(self,model):
        for param in model.parameters():
            param.data.fill_(0)
        return model

    #use closer to get index of random example for now
    def step(self, closure=None):
        loss = None
        if closure is None:
            raise ValueError("Need index of example")

        idx = closure
        for group in self.param_groups:
            for (p,past_grad,avg_grad) in zip(group['params'],self.past_gradients[idx].parameters(),self.avg_gradients
                                              .parameters()):
                if p.grad is None:
                    continue
                d_p = p.grad.data - past_grad.data + avg_grad.data
                avg_grad.data += (p.grad.data - past_grad.data)/self.n_samples
                past_grad.data = p.grad.data.clone()
                p.data.add_(-group['lr'], d_p)

        return loss