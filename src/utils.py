import torch
import numpy as np
import os

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def save_checkpoint(state, is_best=False, checkpoint_dir='checkpoints', name='checkpoint', filename=None):
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    
    if filename:
        torch.save(state, filename)
        print(f"Saved model to {filename}")
    else:
        filepath = os.path.join(checkpoint_dir, f'{name}_last.pth')
        torch.save(state, filepath)
        
        if is_best:
            best_filename = os.path.join(checkpoint_dir, f'{name}_best.pth')
            torch.save(state, best_filename)
            print(f"Saved new best model to {best_filename}")
