
import torch.optim as optim 
import torch.optim.lr_scheduler as lr_scheduler


def get_optimizer(config, params):
    '''
    Initializes an optimizer with provided configuration.
    '''
    name = config.get('name', 'sgd')
    if name == 'sgd':
        return optim.SGD(params=params, lr=config['lr'], weight_decay=config['weight_decay'], momentum=0.9, nesterov=True)
    elif name == 'adam':
        return optim.Adam(params=params, lr=config['lr'], weight_decay=config['weight_decay'])
    else:
        raise NotImplementedError(f'Invalid optimizer {name}')


def get_scheduler(config, optimizer):
    '''
    Initializes a scheduler with provided configuration.
    '''
    name = config.get('name', None)
    warmup_epochs = config.get('warmup_epochs', 0)

    if warmup_epochs > 0:
        for group in optimizer.param_groups:
            group['lr'] = 1e-12/warmup_epochs * group['lr']

    if name is not None:
        if name == 'cosine':
            scheduler = lr_scheduler.CosineAnnealingLR(optimizer, config['epochs']-warmup_epochs, eta_min=0.0, last_epoch=-1)
        else:
            raise NotImplementedError(f'Invalid scheduler {name}')
        return scheduler, warmup_epochs

    else:
        return None, warmup_epochs


COLORS = {
    'yellow' : '\x1b[33m',
    'blue' : '\x1b[94m',    
    'green' : '\x1b[32m',
    'end' : '\033[0m'
}


def progress_bar(progress = 0, status = '', bar_len = 20):
    
    status = status.ljust(30)
    if progress == 1:
        status = '{}'.format('Done...'.ljust(30))
    block = int(round(bar_len*progress))
    text = '\rProgress: [{}] {}% {}'.format(COLORS['green'] + '#'*block + COLORS['end'] + '-'*(bar_len-block), round(progress*100,2), status)
    print(text, end='')


class AverageMeter:

    def __init__(self):
        self.reset()

    def reset(self):
        self.metrics = {}

    def add(self, metrics):
        if len(self.metrics) == 0:
            self.metrics = {key: [value] for key, value in metrics.items()}
        else:
            for key, value in metrics.items():
                if key in self.metrics:
                    self.metrics[key].append(value)
                else:
                    raise ValueError('Incorrect metric key')

    def return_metrics(self):
        metrics = {key: np.mean(values) for key, values in self.metrics.items()}
        return metrics
        
    def return_msg(self):
        metrics = self.return_metrics()
        msg = ''.join([f'{key}: {round(value, 3)} ' for key, value in metrics.items()])
        return msg       