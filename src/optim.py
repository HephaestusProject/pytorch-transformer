import numpy as np
from torch.optim.lr_scheduler import _LRScheduler


class WarmUpAdam(_LRScheduler):

    def __init__(self, optimizer, hidden_dim, warmup_steps):
        self.init_lr = np.power(hidden_dim, -0.5)
        self.optimizer = optimizer
        self.current_steps = 0
        self.warmup_steps = warmup_steps

    def step(self):
        self.current_steps += 1
        lr = self.init_lr * self.get_scale()
        for p in self.optimizer.param_groups():
            p['lr'] = lr
        self.optimizer.step()

    def zero_grad(self):
        self.optimizer.zero_grad()

    def get_scale(self):
        return np.min([np.power(self.current_steps, -0.5), self.current_steps * np.power(self.warmup_steps, -0.5)])

    def get_lr(self):
        return self.init_lr * self.get_scale()

    def get_current_steps(self):
        return self.current_steps
