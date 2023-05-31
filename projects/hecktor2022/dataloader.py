"""HECKTOR2022 Dataloader. Tensorflow version."""
from typing import Tuple
from pathlib import Path
import tensorflow as tf

from projects.hecktor2022 import terminology as term
from projects.hecktor2022 import preprocessor


class HECKTORDataset(tf.keras.utils.Sequence):
    """Defines dataset."""
    def __init__(
            self, data_folder: Path,
            batch_size: int,
            input_size: Tuple[int, int, int],
            ref_img: term.Modality,
            num_samples_per_epoch: int, phase: term.Phase):

        self.num_samples_per_epoch = num_samples_per_epoch
        self.datafolder = data_folder / phase.value
        self.batch_size = batch_size
        self.input_size = input_size
        self.preprocessor = preprocessor.HecktorProcessor(
            data_folder=self.datafolder,
            reference=ref_img,
        )
        self.patients = self.preprocessor.get_patient_list(phase)
        self.nsamples = len(self.patients)

        # if phase == term.Phase.TRAIN:
        #     self.transform = train_transform
        # elif phase == term.Phase.VALID:
        #     self.datafolder = data_folder / 'valid'
        #     self.transform = val_transform
        # elif phase == term.Phase.TEST:
        #     self.datafolder = data_folder / 'test'
        #     self.transform = val_transform

        print(f'Available data size for {self.phase}: {self.nsamples}')

    def __getitem__(self, index_: int) -> List[torch.Tensor]:
        if index_ <= self.sum_case - 1:
            case_id = self.examples[index_]
        else:
            new_index_ = index_ - (index_ // self.sum_case) * self.sum_case
            case_id = self.examples[new_index_]

        dict_images = read_data(case_id)
        list_images = pre_processing(dict_images)

        list_tensors = self.transform(list_images)
        return list_tensors

    def __len__(self):
        return self.num_samples_per_epoch