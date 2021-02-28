import os
import time
from scipy.io import wavfile

import numpy as np

import torch
from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt
import matplotlib

from ml.metrics.average_meter import AverageMeter

matplotlib.use('Agg')


class IOHandler:
    def __init__(self, config):
        self.config = config
        # if no id is provided, then use the current timestamp
        if self.config.id is not None:
            exp_name = self.config.id
        else:
            exp_name = time.strftime('%Y_%m_%d-%H_%M_%S', time.localtime())

        # path for the results
        self.result_dir = os.path.join('../results/recurrent_method_research', exp_name)
        if not os.path.exists(self.result_dir):
            os.makedirs(self.result_dir)

        # save config
        self.config.save(os.path.join(self.result_dir, 'config.yaml'))

        # tensorboard writer
        self.tensorboard_dir = os.path.join(self.result_dir, 'tensorboard')
        if not os.path.exists(self.tensorboard_dir):
            os.makedirs(self.tensorboard_dir)
        self.writer = SummaryWriter(self.tensorboard_dir)

        # model dir
        self.models_dir = os.path.join(self.result_dir, 'models')
        if not os.path.exists(self.models_dir):
            os.makedirs(self.models_dir)

        # metric
        self.train_loss_meter = AverageMeter(self.result_dir, 'train')
        self.val_loss_meter = AverageMeter(self.result_dir, 'val')

    def save_tensorboard(self, model, batch, output_dict, iteration, activations, current_mode):
        # loss
        if current_mode == 'train':
            self.writer.add_scalar(f"{current_mode}/loss", self.train_loss_meter.val, iteration)
            # weights and grads
            for weight_and_grad in self.get_weights_and_grads(model):
                self.writer.add_histogram(f'bottleneck_activations', output_dict['bottleneck'][0], global_step=iteration)
                self.writer.add_histogram(f'weights/{weight_and_grad["name"]}', weight_and_grad['weight'],
                                          global_step=iteration)
                self.writer.add_histogram(f'grads/{weight_and_grad["name"]}', weight_and_grad['grad'],
                                          global_step=iteration)
            for name, activation in activations.items():
                self.writer.add_histogram(f'activations/{name}', activation,
                                          global_step=iteration)
        elif current_mode == 'val':
            self.writer.add_scalar(f"{current_mode}/loss", self.val_loss_meter.val, iteration)

        # input
        fig = plt.figure()
        plt.ylim(-1, 1)
        plt.plot(batch['inputs'][0][0].cpu().detach().numpy())
        plt.close(fig)
        self.writer.add_figure(f"{current_mode}/input", fig, global_step=iteration)

        # prediction
        fig = plt.figure()
        plt.ylim(-1, 1)
        plt.plot(output_dict['pred'][0][0].cpu().detach().numpy())
        plt.close(fig)
        self.writer.add_figure(f"{current_mode}/pred", fig, global_step=iteration)

    def get_weights_and_grads(self, model):
        weights_and_grads = []
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Conv1d) or isinstance(module, torch.nn.Linear):
                weights_and_grads.append({'name': name,
                                          'weight': module.weight.reshape(-1).detach().cpu().numpy(),
                                          'grad': module.weight.grad.reshape(-1).detach().cpu().numpy()})

        return weights_and_grads

    def update_metrics(self, loss, current_mode):
        if current_mode == 'train':
            self.train_loss_meter.update(float(loss))
        elif current_mode == 'val':
            self.val_loss_meter.update(float(loss))
        else:
            raise ValueError(f'Wrong mode: {current_mode}')

    def get_metric_mean(self, current_mode):
        if current_mode == 'train':
            return self.train_loss_meter.mean()
        elif current_mode == 'val':
            return self.val_loss_meter.mean()
        else:
            raise ValueError(f'Wrong mode: {current_mode}')

    def save_model(self):
        ...

    def load_model(self, model_name):
        ...

    def after_epoch(self, epoch, current_mode):
        if current_mode == 'train':
            self.train_loss_meter.save_metrics(epoch)
            self.train_loss_meter.reset()
        elif current_mode == 'val':
            self.val_loss_meter.save_metrics(epoch)
            self.val_loss_meter.reset()
        else:
            raise ValueError()

        self.save_model()

    def save_sound(self, np_sound, folder_name, example_name):
        scaled = np.int16(np_sound * 2 ** 15)

        file_dir = os.path.join(self.result_dir, 'examples', folder_name)
        if not os.path.exists(file_dir):
            os.makedirs(file_dir)

        file_path = os.path.join(file_dir, example_name + '.wav')
        wavfile.write(file_path, 44100, scaled)

        # input
        fig = plt.figure()
        plt.ylim(-2, 2)
        plt.plot(np_sound[:1000])
        plt.savefig(os.path.join(file_dir, example_name + '.png'))
        plt.close(fig)


