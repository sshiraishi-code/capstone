"""Defines the terminology used in HECKTOR project."""
from enum import Enum


class Phase(Enum):
    TRAIN = 'train'
    VALID = 'valid'
    TEST = 'test'
    PRED = 'predict'


class Modality(Enum):
    CT = 'CT'
    PET = 'PT'
    T1 = 'T1'
    T2 = 'T2'
    FLAIR = 'FLAIR'
    DWI = 'DWI'
    DTI = 'DTI'
    DSC = 'DSC'
    DCE = 'DCE'
