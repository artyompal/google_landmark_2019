''' Learning rate schedulers. '''

import torch.optim.lr_scheduler as lr_scheduler


def step(optimizer, last_epoch, step_size=10, gamma=0.1, **_):
    return lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma,
                               last_epoch=last_epoch)

def multi_step(optimizer, last_epoch, milestones=[500, 5000], gamma=0.1, **_):
    if isinstance(milestones, str):
        milestones = eval(milestones)
    return lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma,
                                    last_epoch=last_epoch)

def exponential(optimizer, last_epoch, gamma=0.995, **_):
    return lr_scheduler.ExponentialLR(optimizer, gamma=gamma, last_epoch=last_epoch)

def none(optimizer, last_epoch, **_):
    return lr_scheduler.StepLR(optimizer, step_size=10000000, last_epoch=last_epoch)

def reduce_lr_on_plateau(optimizer, last_epoch, mode='max', factor=0.1,
                         patience=10, threshold=0.0001, threshold_mode='rel',
                         cooldown=0, min_lr=0, **_):
    return lr_scheduler.ReduceLROnPlateau(optimizer, mode=mode, factor=factor,
                                          patience=patience, threshold=threshold,
                                          threshold_mode=threshold_mode,
                                          cooldown=cooldown, min_lr=min_lr)

def cosine(optimizer, last_epoch, T_max=50, eta_min=0.00001, **_):
    return lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min,
                                          last_epoch=last_epoch)

def cyclic_lr(optimizer, last_epoch, base_lr=0.001, max_lr=0.01,
              step_size_up=2000, step_size_down=None, mode='triangular',
              gamma=1.0, scale_fn=None, scale_mode='cycle', cycle_momentum=True,
              base_momentum=0.8, max_momentum=0.9, **_):
    return lr_scheduler.CyclicLR(optimizer, base_lr=base_lr, max_lr=max_lr,
                                 step_size_up=step_size_up, step_size_down=
                                 step_size_down, mode=mode, gamma=gamma,
                                 scale_mode=scale_mode, cycle_momentum=
                                 cycle_momentum, base_momentum=base_momentum,
                                 max_momentum=max_momentum, last_epoch=last_epoch)

def get_scheduler(config, optimizer, last_epoch=-1):
    func = globals().get(config.scheduler.name)
    return func(optimizer, last_epoch, **config.scheduler.params)

def is_scheduler_continuous(name) -> bool:
    return name in ['exponential', 'cosine', 'cyclic_lr']
