import torch
import numpy as np
from statsmodels.tsa.stattools import acf
from scipy.signal import argrelextrema
import os
import pandas as pd


def read_file(path, filename):
    file_path = os.path.join(path, filename)

    df = pd.read_csv(file_path).dropna()
    data = df.iloc[:, 0:-1].values.astype(float)
    label = df['Label'].astype(int).to_numpy()


    # normalize data globally
    data_mean = data.mean(axis=0)
    data_std = data.std(axis=0)
    # # Avoid division by zero
    data_std = np.where(data_std == 0, 1e-8, data_std)

    data = (data - data_mean) / data_std

    train_index = filename.split('.')[0].split('_')[-3]
    data_train = data[:int(train_index), :]
    data_test = data

    return data_train, data_test, label#[int(train_index):]


class ReconstructDataset(torch.utils.data.Dataset):
    def __init__(self, data, window_size, stride=1, normalize=True, add_last_partial=False):
        super().__init__()
        self.window_size = window_size
        self.stride = stride
        self.data = self._normalize_data(data) if normalize else data
        self.add_last_partial = add_last_partial

        self.univariate = self.data.shape[1] == 1
        self.sample_num = max(
            0, (self.data.shape[0] - window_size) // stride + 1)
        self.samples, self.targets = self._generate_samples()

    def _normalize_data(self, data, epsilon=1e-8):
        mean, std = np.mean(data, axis=0), np.std(data, axis=0)
        std = np.where(std == 0, epsilon, std)  # Avoid division by zero
        return (data - mean) / std

    def _generate_samples(self):
        data = torch.tensor(self.data, dtype=torch.float32)

        if self.univariate:
            data = data.squeeze()
            X = torch.stack([data[i * self.stride: i * self.stride +
                            self.window_size] for i in range(self.sample_num)])

            # for the case of stride > 1, add the last partial window as data[-window_size:]
            if self.add_last_partial and self.stride > 1:
                X = torch.cat([X, data[-self.window_size:].unsqueeze(0)], dim=0)
                self.sample_num += 1
            X = X.unsqueeze(-1)

        else:
            X = torch.stack([data[i * self.stride: i * self.stride +
                            self.window_size, :] for i in range(self.sample_num)])

        return X, X

    def __len__(self):
        return self.sample_num

    def __getitem__(self, index):
        return self.samples[index]




def find_length_rank(data, rank=1):
    data = data.squeeze()
    if len(data.shape) > 1:
        return 0
    if rank == 0:
        return 1
    data = data[:min(20000, len(data))]

    base = 3
    auto_corr = acf(data, nlags=400, fft=True)[base:]

    local_max = argrelextrema(auto_corr, np.greater)[0]

    try:
        sorted_local_max = np.argsort([auto_corr[lcm] for lcm in local_max])[
            ::-1]    # Ascending order
        max_local_max = sorted_local_max[0]     # Default
        if rank == 1:
            max_local_max = sorted_local_max[0]
        if rank == 2:
            for i in sorted_local_max[1:]:
                if i > sorted_local_max[0]:
                    max_local_max = i
                    break
        if rank == 3:
            for i in sorted_local_max[1:]:
                if i > sorted_local_max[0]:
                    id_tmp = i
                    break
            for i in sorted_local_max[id_tmp:]:
                if i > sorted_local_max[id_tmp]:
                    max_local_max = i
                    break

        if local_max[max_local_max] < 3 or local_max[max_local_max] > 300:
            return 125
        return local_max[max_local_max]+base
    except:
        return 125


def my_kl_loss(p, q):
    res = p * (torch.log(p + 0.0001) - torch.log(q + 0.0001))
    return torch.mean(torch.sum(res, dim=-1), dim=1)
