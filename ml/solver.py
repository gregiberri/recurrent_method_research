import gc
import logging
import os
import random
import sys
import time
import numpy as np
import torch
from tqdm import tqdm

from config import ConfigNameSpace
from data.datasets import get_dataloader
from ml.io_handler.io_handler import IOHandler
from ml.models import get_model
from ml.optimizers import get_optimizer

RANDOM_SEED = 5


class Solver:
    def __init__(self, args):
        self.args = args
        self.config = ConfigNameSpace(self.args.config)

        self._set_seed()

        self.io_handler = IOHandler(self.config)

        # data loaders
        self.train_loader, self.train_niter = get_dataloader(self.config.data, True)
        self.val_loader, self.val_niter = get_dataloader(self.config.data, False)

        # get model
        self.init_model_and_optimizer()

    def _set_seed(self):
        """
        Set the random seeds
        """
        torch.manual_seed(RANDOM_SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(RANDOM_SEED)
        np.random.seed(RANDOM_SEED)
        random.seed(RANDOM_SEED)

    def init_model_and_optimizer(self):
        if self.args.mode in ['train', 'autoencode']:
            self.epoch = 0
            self.model = get_model(self.config.model).cuda()
            self.optimizer = get_optimizer(self.config.optimizer, model_params=self.model.parameters())
        else:
            self.model, self.optimizer = self.io_handler.load_model(self.config.model_to_load)

        # set hook for actications
        self.activations = {}  # dictionary to store the activation of a layer
        def create_hook(name):
            def hook(m, i, o):
                # copy the output of the given layer
                self.activations[name] = o
            return hook

        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.Sequential) or isinstance(module, torch.nn.Conv1d):
                module.register_forward_hook(create_hook(name))

    def run(self):
        if self.args.mode == 'train':
            self.train()
        elif self.args.mode == 'eval':
            self.eval()
        elif self.args.mode == 'hyperopt':
            self.hyperopt()
        else:
            raise ValueError(f'Wrong mode argument: {self.args.mode}')

    def batch_to_cuda(self, minibatch):
        minibatch = {k: v.cuda() for k, v in minibatch.items() if isinstance(v, torch.Tensor)}
        return minibatch

    def get_loader(self, mode):
        if mode in ['train', 'autoencode']:
            return self.train_loader, self.train_niter
        elif mode == 'val':
            return self.val_loader, self.val_niter
        else:
            raise ValueError(f'Wrong solver mode: {mode}')

    def before_epoch(self):
        self.iteration = 0
        self.epoch = self.epoch
        torch.cuda.empty_cache()

    def after_epoch(self):
        self.io_handler.after_epoch(self.epoch, self.current_mode)
        gc.collect()
        torch.cuda.empty_cache()

    def train(self):
        for self.epoch in range(self.epoch, self.config.env.epochs):
            self.current_mode = 'train'
            self.before_epoch()
            self.run_epoch()
            self.after_epoch()

            self.current_mode = 'val'
            self.before_epoch()
            self.run_epoch()
            self.after_epoch()

        if self.config.env.vis_filters:
            self.vis_filters()

    def hyperopt(self):
        raise NotImplementedError()

    def eval(self):
        ...

    def vis_filters(self):
        length = self.config.model.params.kernel_size
        for layer_idx in range(len(self.config.model.params.encoder_filters)):
            for filter_idx in range(self.config.model.params.encoder_filters[layer_idx]):
                print(f'Visualizing layer {layer_idx} filter {filter_idx}')
                model = torch.nn.Sequential(*list(self.model.children())[0][:layer_idx+1])
                if filter_idx == 0:
                    length *= list(model.named_modules())[-2][1].stride[0]
                input_image = torch.zeros([1, 1, length], device='cuda', requires_grad=True)
                optimizer = torch.optim.Adam([input_image], lr=0.0001)

                step = 0
                while torch.max(torch.abs(input_image)) < 0.5 and step < 100000:
                    out_dict = model(input_image)
                    optimizer.zero_grad()
                    loss = -out_dict[0, filter_idx, self.config.model.params.kernel_size//2]
                    loss.backward()
                    optimizer.step()

                    step += 1

                self.io_handler.save_sound(input_image[0][0].cpu().detach().numpy(),
                                           f'filter_vis/layer_{layer_idx}',
                                           f'filter_{filter_idx}')


    def run_epoch(self):
        if self.current_mode == 'train':
            self.model.train()
        elif self.current_mode == 'val':
            self.model.eval()
        else:
            raise ValueError(f'Wrong current mode: {self.current_mode}')
        loader, niter = self.get_loader(self.current_mode)
        # metric = self.get_metric(mode, niter=niter)
        epoch_iterator = iter(loader)

        # set loading bar
        bar_format = '{desc}[{elapsed}<{remaining},{rate_fmt}]'
        pbar = tqdm(range(niter), file=sys.stdout, bar_format=bar_format)

        for idx in pbar:
            iteration = self.epoch * niter + idx

            # get the minibatch and filter out to the input and gt elements
            minibatch = epoch_iterator.next()
            model_input_dict = self.batch_to_cuda(minibatch)

            # train
            output_dict = self.step(self.current_mode, model_input_dict)

            # save results
            self.io_handler.update_metrics(output_dict['loss'], current_mode=self.current_mode)
            if iteration % self.config.env.save_frequency == 0:
                self.io_handler.save_tensorboard(self.model, model_input_dict, output_dict, iteration,
                                                 self.activations, self.current_mode)

            print_str = f'[{self.current_mode}] Epoch {self.epoch:4d}/{self.config.env.epochs:4d} ' \
                        + f'Iter{idx + 1:4d}/{niter:4d}: ' \
                        + f'losses={output_dict["loss"].item():.4e} ' \
                          f'({self.io_handler.get_metric_mean(self.current_mode):.4e})'  # '\

            pbar.set_description(print_str, refresh=False)

            if self.config.env.save_val_examples and self.current_mode == 'val':
                self.io_handler.save_sound(minibatch['inputs'][0][0].cpu().detach().numpy(),
                                           f'inputs/epoch_{self.epoch}',
                                           minibatch['filename'][0])
                self.io_handler.save_sound(output_dict['pred'][0][0].cpu().detach().numpy(),
                                           f'preds/epoch_{self.epoch}',
                                           minibatch['filename'][0])

    def step(self, mode='train', model_inputs={}):
        """
        :param model_inputs:
        :return:
        """
        if mode == 'val':
            with torch.no_grad():
                output_dict = self.model(**model_inputs)
            return output_dict

        elif mode == 'train' or mode == 'scene_retrain':
            self.iteration += 1
            output_dict = self.model(**model_inputs)

            loss = output_dict['loss']

            # backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            return output_dict

    def save_best_checkpoint(self, epoch_results):
        if not min(epoch_results['irmse']) == epoch_results['irmse'][-1]:
            return

        path = os.path.join(self.result_dir, 'model_best.pth')

        t_start = time.time()

        state_dict = {}

        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in self.model.state_dict().items():
            key = k
            if k.split('.')[0] == 'module':
                key = k[7:]
            new_state_dict[key] = v

        state_dict['config'] = self.config
        state_dict['model'] = new_state_dict
        state_dict['optimizer'] = self.optimizer.state_dict()
        state_dict['epoch'] = self.epoch
        state_dict['iteration'] = self.iteration

        t_iobegin = time.time()
        torch.save(state_dict, path)
        del state_dict
        del new_state_dict
        t_end = time.time()
        logging.info(
            "Save checkpoint to file {}, "
            "Time usage:\n\tprepare snapshot: {}, IO: {}".format(
                path, t_iobegin - t_start, t_end - t_iobegin))
