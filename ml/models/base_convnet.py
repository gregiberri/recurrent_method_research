import torch
import torch.nn as nn

from ml.criterias import Criteria


class BaseConvAutoencodeModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        kernel_size = self.config.kernel_size
        padding = kernel_size // 2
        stride = kernel_size // 2 + 1

        self.module_list = torch.nn.ModuleList()
        for layer_index in range(len(config.encoder_filters)):
            if layer_index == 0:
                layer = torch.nn.Sequential(torch.nn.Conv1d(1,
                                                            config.encoder_filters[layer_index],
                                                            kernel_size=kernel_size,
                                                            stride=config.encoder_strides[layer_index],
                                                            padding=padding),
                                            torch.nn.LeakyReLU())
            else:
                layer = torch.nn.Sequential(torch.nn.Conv1d(config.encoder_filters[layer_index - 1],
                                                            config.encoder_filters[layer_index],
                                                            kernel_size=kernel_size,
                                                            stride=config.encoder_strides[layer_index],
                                                            padding=padding),
                                            torch.nn.LeakyReLU())
            self.module_list.append(layer)
        # self.conv1 =
        # self.downscale1 = torch.nn.Sequential(torch.nn.Conv1d(config.encoder_filters[0],
        #                                                       config.encoder_filters[0],
        #                                                       kernel_size=kernel_size,
        #                                                       stride=config.strides[1],
        #                                                       padding=padding),
        #                                       torch.nn.LeakyReLU())
        #
        # self.conv2 = torch.nn.Sequential(torch.nn.Conv1d(config.encoder_filters[0],
        #                                                  config.encoder_filters[1],
        #                                                  kernel_size=kernel_size,
        #                                                  stride=config.strides[2],
        #                                                  padding=padding),
        #                                  torch.nn.LeakyReLU(),
        #                                  torch.nn.Conv1d(config.encoder_filters[1],
        #                                                  config.encoder_filters[1],
        #                                                  kernel_size=kernel_size,
        #                                                  stride=config.strides[3],
        #                                                  padding=padding),
        #                                  torch.nn.LeakyReLU())
        #
        # self.conv3 = torch.nn.Sequential(torch.nn.Conv1d(config.encoder_filters[1],
        #                                                  config.encoder_filters[2],
        #                                                  kernel_size=kernel_size,
        #                                                  stride=1,
        #                                                  padding=padding),
        #                                  torch.nn.LeakyReLU(),
        #                                  torch.nn.Conv1d(config.encoder_filters[2],
        #                                                  config.encoder_filters[2],
        #                                                  kernel_size=kernel_size,
        #                                                  stride=stride,
        #                                                  padding=padding),
        #                                  torch.nn.LeakyReLU())

        self.convt3 = torch.nn.Sequential(torch.nn.Conv1d(config.decoder_filters[0],
                                                          config.decoder_filters[0],
                                                          kernel_size=kernel_size,
                                                          stride=1,
                                                          padding=padding),
                                          torch.nn.LeakyReLU(),
                                          torch.nn.ConvTranspose1d(config.decoder_filters[0],
                                                                   config.decoder_filters[1],
                                                                   kernel_size=kernel_size,
                                                                   stride=stride,
                                                                   padding=padding,
                                                                   output_padding=padding),
                                          torch.nn.LeakyReLU())

        self.convt2 = torch.nn.Sequential(torch.nn.Conv1d(config.decoder_filters[1],
                                                          config.decoder_filters[1],
                                                          kernel_size=kernel_size,
                                                          stride=1,
                                                          padding=padding),
                                          torch.nn.LeakyReLU(),
                                          torch.nn.ConvTranspose1d(config.decoder_filters[1],
                                                                   config.decoder_filters[2],
                                                                   kernel_size=kernel_size,
                                                                   stride=stride,
                                                                   padding=padding,
                                                                   output_padding=padding),
                                          torch.nn.LeakyReLU())

        self.convt1 = torch.nn.Sequential(torch.nn.ConvTranspose1d(config.decoder_filters[2],
                                                                   config.decoder_filters[2],
                                                                   kernel_size=kernel_size,
                                                                   stride=stride,
                                                                   padding=padding,
                                                                   output_padding=padding),
                                          torch.nn.LeakyReLU())

        self.output = torch.nn.Conv1d(config.encoder_filters[0],
                                      out_channels=1,
                                      kernel_size=1)

        self.criteria = Criteria(config.loss)

    def forward(self, inputs, target=None):
        x = inputs
        for layer in self.module_list:
            x = layer(x)
        x3 = x
        # x1 = self.conv1(inputs)
        # x1ds = self.downscale1(x1)
        # x2 = self.conv2(x1ds)
        # x3 = self.conv3(x2)
        x4 = self.convt3(x3)
        x5 = self.convt2(x4)
        x6 = self.convt1(x5)
        output = self.output(x6)

        output_dict = self.get_prediction_and_loss([output, x3], inputs)

        return output_dict

    def get_prediction_and_loss(self, feat, target):
        # predicion
        prediction, bottleneck = feat

        # loss
        loss = self.criteria(prediction, target)

        return {'pred': prediction, 'bottleneck': bottleneck, 'loss': loss}
