"""HECKTOR2022 Dataloader. Tensorflow version."""
from typing import Dict, Tuple
from etils import epath
# import tensorflow as tf
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from projects.hecktor2022 import terminology as term

_TENSOR_FILE_FORMAT = '.pt'


def read_tensors(input_folder: epath.Path, id: str) -> Dict[str, torch.Tensor]:
    """Load tensors into dictionary."""
    input_file = input_folder / 'images' / (id + '_input' + _TENSOR_FILE_FORMAT)
    label_file = input_folder / 'images' / (id + '_label' + _TENSOR_FILE_FORMAT)
    input_data = torch.load(input_file)
    label_data = torch.load(label_file)
    return {'input': input_data, 'label': label_data}


def _create_transform() -> transforms.Compose:
    """Creates a transform for different phases."""
    transform_comp = transforms.Compose([
        transforms.RandomAffine(180, 0.2),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.GaussianBlur((0.01, 0.5))
    ])
    return transform_comp


# class HECKTORDataset(tf.keras.utils.Sequence):
class HECKTORDataset(Dataset):
    """Defines dataset."""
    def __init__(
            self, data_folder: epath.Path,
            batch_size: int,
            num_samples_per_epoch: int, phase: term.Phase) -> None:

        self.num_samples_per_epoch = num_samples_per_epoch
        self.datafolder = data_folder / phase.value
        self.phase = phase
        self.batch_size = batch_size
        example_files = (self.datafolder / 'labels').glob(_TENSOR_FILE_FORMAT)
        self.samples = [(example.stem)[:-6] for example in example_files]
        self.nsamples = len(self.samples)

        if phase == term.Phase.TRAIN:
            self.transform = _create_transform()
        else:
            self.transform = transforms.Compose([])

        print(f'Available data size for {self.phase.value}: {self.nsamples}')

    def __getitem__(self, index_: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if index_ <= self.nsamples - 1:
            case_id = self.samples[index_]
        else:
            new_index_ = index_ - (index_ // self.nsamples) * self.nsamples
            case_id = self.samples[new_index_]

        dict_images = read_tensors(self.datafolder, case_id)
        input_tensor = self.transform(dict_images['input'])
        label_tensor = self.transform(dict_images['label'])
        return input_tensor, label_tensor

    def __len__(self) -> int:
        return self.num_samples_per_epoch


def get_loader(datafolder: epath.Path, 
               train_bs: int = 1, val_bs: int = 1,
               train_num_samples_per_epoch: int = 1,
               val_num_samples_per_epoch: int = 1,
               num_works: int = 0):
    """Gets dataset."""
    train_dataset = HECKTORDataset(
        datafolder, train_bs,
        train_num_samples_per_epoch, phase=term.Phase.TRAIN)
    valid_dataset = HECKTORDataset(
        datafolder, val_bs,
        val_num_samples_per_epoch, phase=term.Phase.VALID)

    train_loader = DataLoader(
        dataset=train_dataset, batch_size=train_bs,
        shuffle=True, num_workers=num_works, pin_memory=False)
    valid_loader = DataLoader(
        dataset=valid_dataset, batch_size=val_bs,
        shuffle=False, num_workers=num_works, pin_memory=False)

    return train_loader, valid_loader
