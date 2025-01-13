""" Module containing the dataset class."""
import torch
from torch.utils.data import Dataset

import flyvision


class GroomingDatasetClass(Dataset):
    def __init__(self, input_dataset, output_dataset, mask):
        """
        Initialize a Reaching task
        Args:
            task_params: dictionary containing task parameters
        """
        #  set input - outputs
        self.input_data = torch.from_numpy(input_dataset).to(
            dtype=torch.float32,
            device=flyvision.device
        )
        self.output_data = torch.from_numpy(output_dataset).to(
            dtype=torch.float32,
            device=flyvision.device
        )
        self.mask = torch.from_numpy(mask).to(
            dtype=torch.float32,
            device=flyvision.device
        )

    def __len__(self):
        """Arbitrary number of trials, as they are randomly generated anyway"""
        return self.input_data.shape[0]

    def __getitem__(self, idx):
        """
        Returns a trial

        Args:
            idx, trial index

        Returns:
            input, Tensor of size [seq_len, n_inp]
            target, Tensor of size [seq_len, n_inp]
            mask, Tensor of size [seq_len, n_inp]
        """

        return self.input_data[idx, ...], self.output_data[idx, ...], self.mask[idx, ...]


class DatasetClass(Dataset):
    def __init__(self, input_dataset, output_dataset):
        """
        Initialize a Reaching task
        Args:
            task_params: dictionary containing task parameters
        """
        #  set input - outputs
        self.input_data = torch.from_numpy(input_dataset).to(
            dtype=torch.float32,
            device=flyvision.device
        )
        self.output_data = torch.from_numpy(output_dataset).to(
            dtype=torch.float32,
            device=flyvision.device
        )

    def __len__(self):
        """Arbitrary number of trials, as they are randomly generated anyway"""
        return self.input_data.shape[0]

    def __getitem__(self, idx):
        """
        Returns a trial

        Args:
            idx, trial index

        Returns:
            input, Tensor of size [seq_len, n_inp]
            target, Tensor of size [seq_len, n_inp]
            mask, Tensor of size [seq_len, n_inp]
        """

        return self.input_data[idx, ...], self.output_data[idx, ...]


def random_split(dataset, train_ratio, random_seed=0):
    """ Randomly split the dataset as the given ratio. """
    generator = torch.Generator(device=flyvision.device)
    generator.manual_seed(random_seed)

    n_train_samples = int(len(dataset) * train_ratio)
    n_test_samples = int(len(dataset) - n_train_samples)

    train_ds, test_ds = torch.utils.data.random_split(
        dataset,
        [n_train_samples, n_test_samples],
        generator=generator,
    )

    return train_ds, test_ds


def deterministic_split(input_dataset, output_dataset, train_ratio):
    """
        Deterministically split the dataset as the given ratio.
        The train will be the first N, test will be the rest.
    """
    n_train_samples = int(input_dataset.shape[0] * train_ratio)

    train_ds = DatasetClass(
        input_dataset[:n_train_samples, ...],
        output_dataset[:n_train_samples, ...]
    )
    test_ds = DatasetClass(
        input_dataset[n_train_samples:, ...],
        output_dataset[n_train_samples:, ...]
    )

    return train_ds, test_ds
