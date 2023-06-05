"""Defines the terminology used in HECKTOR project."""
from enum import Enum


class Phase(Enum):
    TRAIN = 'train'
    VALID = 'valid'
    TEST = 'test'
    PRED = 'predict'


class Modality(Enum):
    CT = 'ct'
    PET = 'pt'


class Target(Enum):
    GTVn = 'GTVn'
    GTVp = 'GTVp'
