from ml.models.base_convnet import BaseConvAutoencodeModel


def get_model(model_config):
    """
    Select the model according to the model config name and its parameters

    :param model_config: model_config namespace, containing the name and the params

    :return: model
    """

    if model_config.name == 'base_conv_autoencode_model':
        return BaseConvAutoencodeModel(model_config.params)
    else:
        raise ValueError(f'Wrong model name in model configs: {model_config.name}')