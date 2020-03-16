from torch.optim import Adam, SGD


def select_optim(optim, model, lr, weight_decay, momentum=None, amsgrad=False):
    if optim == 'sgd':
        return SGD(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=momentum)
    elif optim == 'adam':
        return Adam(model.parameters(), lr=lr, weight_decay=weight_decay, amsgrad=amsgrad)
    else:
        pass
