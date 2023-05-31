"""Evaluation Module for inference and for metrics calculation"""

import pathlib
from typing import Sequence
import pandas as pd
import torch

from LET_translate.metrics import metrics

_DF_COLUMNS = ('plan', 'field1', 'field2', 'field3', 'field4', 'field5', 'field6')
_FILE_TYPE = '.nii.gz'
_INPUT_FILE_CT = 'CT'
_INPUT_FILE_DOSE = 'MCWdose_p'
_LABEL_FILE = 'MCLdose_p'


class EvaluationModule():
    """Evaluation Module
    Results for different metrics are stored in a list of dataframes.
    """

    def __init__(self, model: torch.nn.Module, state_dict_path: pathlib.Path,
                 data_path: pathlib.Path) -> None:
        if not state_dict_path.exists():
            raise ValueError(f'State_dict is not found at {state_dict_path}')
        if not data_path.exists():
            raise ValueError(f'Data not found at {data_path}')

        self.model = model
        self.model.load_state_dict(torch.load(state_dict_path))

        # idntify all the patients in test
        self.data_path = data_path
        plan_ids = [x.stem for x in data_path.iterdir()]

        self.weight = 1

        self.metrics = {}
        self.results = {}
        for metric in metrics.MetricType:
            self.metrics[metric.value] = metrics.metric_factory(metric)
            self.results[metric.value] = pd.DataFrame(index=plan_ids, columns=_DF_COLUMNS)

    def __post_init__(self):
        """Initilize metrics weights here."""
        self.weight = 1

    def _prepare_input(self, filenames: Sequence) -> torch.Tensor:
        """Convert input files into input tensor."""

    def _prepare_label(self, filenames: Sequence) -> torch.Tensor:
        """Convert label file into label tensor."""

    def evaluate_field(self, metric: str, plan_id: str, field_id: str) -> float:
        """Calculates the metrics of a single field.
        Args:
            metric: name of metric
            plan_id: name of the plan or patient's id
            field_id: name of the field

        Returns:
            a single metric value

        Excpetions:
            ValueError: required files can't be found
        """
        ct_name = self.data_path / plan_id / field_id / (_INPUT_FILE_CT + _FILE_TYPE)
        input_dose_name = self.data_path / plan_id / field_id / (_INPUT_FILE_DOSE + _FILE_TYPE)
        label_file_name = self.data_path / plan_id / field_id / _LABEL_FILE
        if not ct_name.exists():
            raise ValueError(f'CT not found at {ct_name}')
        if not input_dose_name.exists():
            raise ValueError(f'Input dose not found at {input_dose_name}')
        if not label_file_name.exists():
            raise ValueError(f'Label not found at {label_file_name}')

        input = self._prepare_input([ct_name, input_dose_name])
        prediction = self.model.predict(input)
        label = self._prepare_label(label_file_name)

        result = self.metrics[metric].calc_reduce(
            prediction, label, self.weight)
        return result

    def evaluate_plan(self, metric: str, plan_id: str):
        pass

    def evaluate_cohort(self):
        pass