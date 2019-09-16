import datetime
import json
import logging
import os
import sys

import h5py
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm


config_path = sys.argv[1]
config = json.load(open(config_path, 'r'))

featurized_data_path = config['task_file_paths']['featurized_data_path']


class ImageCaptionsDataset(Dataset):
    def __init__(self, path_to_data, train=True):

        hf = h5py.File(path_to_data, 'r')
        captions = hf.get('captions')

        if train:
            inputs = captions.get('train')[()]
        else:
            inputs = captions.get('valid')[()]

        hf.close()
        self.inputs = torch.tensor(inputs, dtype=torch.float32)

    def __getitem__(self, idx):
        inputs_sample = self.inputs[idx]
        return inputs_sample

    def __len__(self):
        return len(self.inputs)


class EncoderAnn(nn.Module):
    def __init__(self, *args, **kwargs):
        super(EncoderAnn, self).__init__()
        self.input_dim = args[0]
        self.hidden_dim = kwargs.get('hidden_dim', 4096)
        self.latent_dim = kwargs.get('latent_dim', 1024)

        self.fc1 = nn.Linear(self.input_dim, self.hidden_dim)
        self.leaky_relu = nn.LeakyReLU()
        self.fc2 = nn.Linear(self.hidden_dim, self.latent_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.leaky_relu(x)
        x = self.fc2(x)
        return x


class DecoderAnn(nn.Module):
    def __init__(self, *args, **kwargs):
        super(DecoderAnn, self).__init__()
        self.latent_dim = args[1]
        self.hidden_dim = kwargs.get('hidden_dim', 4096)
        self.output_dim = args[0]

        self.fc1 = nn.Linear(self.latent_dim, self.hidden_dim)
        self.leaky_relu = nn.LeakyReLU()
        self.fc2 = nn.Linear(self.hidden_dim, self.output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.leaky_relu(x)
        x = self.fc2(x)
        return x


dataset_train = ImageCaptionsDataset(config['task_file_paths']['featurized_data_path'])
dataset_valid = ImageCaptionsDataset(config['task_file_paths']['featurized_data_path'], train=False)

train_loader = DataLoader(dataset_train,
                          batch_size=config['train']['trainer']['batch_size'],
                          shuffle=True)
valid_loader = DataLoader(dataset_valid,
                          batch_size=config['train']['trainer']['batch_size'])

input_dim = dataset_train[0].shape[0]
latent_dim = config['train']['encoder']['args']['latent_dim']
encoder = EncoderAnn(input_dim, **config['train']['encoder']['args'])
decoder = DecoderAnn(input_dim, latent_dim, **config['train']['decoder']['args'])
enc_optimizer = torch.optim.Adam(encoder.parameters(), **config['train']['optimizer']['args'])
dec_optimizer = torch.optim.Adam(decoder.parameters(), **config['train']['optimizer']['args'])
criterion = nn.MSELoss()

exp_name = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
model_dir = os.path.join('experiments', exp_name)
os.makedirs(model_dir, exist_ok=True)
json.dump(config, open(f'{model_dir}/config.json', 'w'))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s\n  %(message)s\n",
    handlers=[
        logging.FileHandler(f'{model_dir}/train.log', mode='w'),
        logging.StreamHandler(sys.stdout)
    ])

logger = logging.getLogger()

best_epoch_loss_valid = float('inf')
for epoch in range(1, config['train']['trainer']['epoch']+1):
    batch_losses_train = []
    batch_losses_valid = []
    for inputs in tqdm(train_loader):
        enc_optimizer.zero_grad()
        dec_optimizer.zero_grad()

        z = encoder(inputs)
        outputs = decoder(z)

        batch_loss_train = criterion(outputs, inputs)
        batch_loss_train.backward()
        enc_optimizer.step()
        dec_optimizer.step()
        batch_losses_train.append(batch_loss_train.item())

    for inputs in tqdm(valid_loader):

        z = encoder(inputs)
        outputs = decoder(z)

        batch_loss_valid = criterion(outputs, inputs)
        batch_losses_valid.append(batch_loss_valid.item())

    epoch_loss_train = np.mean(batch_losses_train)
    epoch_loss_valid = np.mean(batch_losses_valid)

    if epoch_loss_valid < best_epoch_loss_valid:
        # Save best Loss
        best_epoch_loss_valid = epoch_loss_valid
        torch.save({'encoder_state_dict': encoder.state_dict(),
                    'decoder_state_dict': decoder.state_dict()},
                   f'{model_dir}/best_model.pt')
        save_status = 'Best model saved.'
    else:
        # Save latest Loss
        torch.save({'encoder_state_dict': encoder.state_dict(),
                    'decoder_state_dict': decoder.state_dict()},
                   f'{model_dir}/latest_model.pt')
        save_status = 'Latest model saved.'

    logger.info(f'Epoch {epoch}\n  '
                f'Train Loss:      {epoch_loss_train}\n  '
                f'Validation Loss: {epoch_loss_valid}\n  '
                f'{save_status}')
