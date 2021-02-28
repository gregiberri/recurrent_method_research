from torch.optim import Adam, SGD, Adagrad, RMSprop


def get_optimizer(optimizer_config, model_params):
    if optimizer_config.name == 'sgd':
        return SGD(params=model_params, **optimizer_config.params.dict())
    elif optimizer_config.name == 'adam':
        return Adam(params=model_params, **optimizer_config.params.dict())
    # elif optimizer_config.name == 'radam':
    #     return RAdam(params=model_params, **optimizer_config.params.dict())
    elif optimizer_config.name == 'adagrad':
        return Adagrad(params=model_params, **optimizer_config.params.dict())
    elif optimizer_config.name == 'rmsprop':
        return RMSprop(params=model_params, **optimizer_config.params.dict())
    else:
        raise ValueError(f'Wrong optimizer name: {optimizer_config.name}')
