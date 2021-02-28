import audioop
import glob
import os

import torch
import numpy as np
import wave

from data.datasets.base_dataset import BaseDataset

CLASS_NUMBER = {'cel': 0, 'cla': 1, 'flu': 2, 'gac': 3, 'gel': 4, 'org': 5,
                'pia': 6, 'sax': 7, 'tru': 8, 'vio': 9, 'voi': 10}


class IrmasDataset(BaseDataset):
    def __init__(self, config, is_train):
        super(IrmasDataset, self).__init__(config, is_train)
        if self.split == 'train':
            self.data_dir_name = 'IRMAS-TrainingData'
        elif self.split == 'val':
            self.data_dir_name = 'IRMAS-TestingData-Part1'

        self.instrument = '*' if self.config.instruments == 'all' else self.config.instruments

        self.filenames = self._get_filepaths()

    def _get_filepaths(self):
        if self.is_train:
            return self._get_train_filepaths()
        else:
            return self._get_val_filepaths()

    def _get_train_filepaths(self):
        data_path_template = os.path.join(self.config.path, self.data_dir_name, self.instrument, '*.wav')
        data_paths = glob.glob(data_path_template)

        if self.config.few_train_input and len(data_paths) > 0:
            data_paths = data_paths[:10]

        return data_paths

    def _get_val_filepaths(self):
        data_path_template = os.path.join(self.config.path, self.data_dir_name, 'Part1', '*.wav')
        data_paths = glob.glob(data_path_template)
        label_paths = [data_path.replace('.wav', '.txt') for data_path in data_paths]

        # filter to instrument
        instrument_data_paths = []
        if self.instrument != '*':
            for data_path, label_path in zip(data_paths, label_paths):
                with open(label_path) as f:
                    instruments_in_data = f.readlines()
                    instruments_in_data = [instrument.rstrip('\n').rstrip('\t') for instrument in instruments_in_data]
                    if self.instrument in instruments_in_data:
                        instrument_data_paths.append(data_path)
        else:
            instrument_data_paths = data_paths

        if self.config.few_val_input and len(data_paths) > 0:
            data_paths = instrument_data_paths[:10]

        return data_paths

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        # Read file to get buffer
        ifile = wave.open(self.filenames[index])
        samples = ifile.getnframes()

        # Read and make mono
        audio = audioop.tomono(ifile.readframes(samples), ifile.getsampwidth(), 1, 1)

        # Convert buffer to float32 using NumPy
        audio_as_np_int16 = np.frombuffer(audio, dtype=np.int16)
        audio_as_np_float32 = audio_as_np_int16.astype(np.float32)

        # Normalise float32 array so that values are between -1.0 and +1.0
        max_int16 = 2 ** 15
        audio_normalised = audio_as_np_float32 / max_int16

        # cut out the random length for the training
        if self.is_train:
            input_size = self.config.train_input_size
        else:
            input_size = self.config.val_input_size

        start = np.random.randint(0, len(audio_normalised) - input_size) \
            if not self.config.single_input else 0
        audio_snippet = audio_normalised[start:start + input_size]
        audio_snippet = np.expand_dims(audio_snippet, axis=0)

        # convert to tensor
        audio_snippet_tensor = torch.as_tensor(audio_snippet, device=torch.device('cpu'))

        # get the label # TODO solve using labels for both train and val
        instrument_class_number = 0  # CLASS_NUMBER[self.filenames[index].split('/')[-2]]
        instrument_onehot = ((instrument_class_number ==
                              torch.arange(len(CLASS_NUMBER), device=torch.device('cpu'), dtype=torch.float32))).float()

        # get the name of the read file
        filename = self.filenames[index].split('/')[-1].split('.')[0]

        return {'inputs': audio_snippet_tensor, 'target': instrument_onehot, 'filename': filename}
