"""
Learning Rate Scheduler

This module provides a function to adjust the learning rate during the training process using a half-cycle cosine annealing schedule with warmup.

The `adjust_learning_rate` function takes an optimizer, the current epoch, and some arguments as input. 
It calculates the learning rate based on the specified schedule and updates the learning rate for each parameter group in the optimizer.

The learning rate schedule consists of three phases:

1. Warmup phase: The learning rate is linearly increased from `args.lr * 0.5 / args.warmup_epochs` to `args.lr` over the first `args.warmup_epochs` epochs.

2. Cosine annealing phase: After the warmup phase, the learning rate follows a half-cycle cosine annealing schedule, decaying from `args.lr` to `args.min_lr` over the remaining epochs.

3. Minimum learning rate: After the cosine annealing phase, the learning rate remains at `args.min_lr`.

This learning rate schedule is useful for training deep learning models, as it provides a gradual warmup phase to prevent divergence, 
followed by a cosine annealing phase that gradually reduces the learning rate, which can improve convergence and generalization.

Args:
    optimizer (torch.optim.Optimizer): The optimizer for which the learning rate should be adjusted.
    epoch (int): The current epoch.
    args (argparse.Namespace): The command-line arguments, containing `lr`, `warmup_epochs`, `epochs`, and `min_lr`.

Returns:
    float: The adjusted learning rate.

Note: This function is copied from "https://github.com/Sense-X/MixMIM/tree/master/util".
"""

import math

def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate with half-cycle cosine after warmup"""
    if epoch == 0:
        lr = args.lr * 0.5 / args.warmup_epochs 
    elif epoch < args.warmup_epochs:
        lr = args.lr * epoch / args.warmup_epochs 
    else:
        lr = args.min_lr + (args.lr - args.min_lr) * 0.5 * \
            (1. + math.cos(math.pi * (epoch - args.warmup_epochs) / (args.epochs - args.warmup_epochs)))
    for param_group in optimizer.param_groups:
        if "lr_scale" in param_group:
            param_group["lr"] = lr * param_group["lr_scale"]
        else:
            param_group["lr"] = lr
    return lr
