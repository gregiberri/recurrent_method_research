from torch.utils.data import DataLoader

import numpy as np

from data.datasets.irmas_dataset import IrmasDataset


def get_dataloader(data_config, is_train):
    # get the iterator object
    if data_config.name == 'irmas':
        dataset = IrmasDataset(data_config.params, is_train=is_train)
    else:
        raise ValueError(f'Wrong dataset name: {data_config.name}')

    # calculate the iteration number for the tqdm
    batch_size = data_config.params.batch_size if is_train else 1
    niters_per_epoch = int(np.ceil(dataset.__len__() // batch_size))
    shuffle = is_train

    # make the torch dataloader object
    loader = DataLoader(dataset,
                        batch_size=batch_size,
                        num_workers=data_config.params.workers,
                        drop_last=False,
                        shuffle=shuffle,
                        pin_memory=False)

    return loader, niters_per_epoch
