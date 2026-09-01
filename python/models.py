from dataclasses import dataclass
import numpy as np


@dataclass
class Signals_Time:
    time: np.ndarray
    signals: list[np.ndarray]


@dataclass
class ResampledSignals:
    signals: list[np.ndarray]
    time: np.ndarray
    #codes: list[np.ndarray]
    fs: float

@dataclass
class SpreadSignals:
    original: list[np.ndarray]
    signals: list[np.ndarray]
    time: np.ndarray
    codes: list[np.ndarray]
    fs: float 

@dataclass
class DespreadSignals:
    original: list[np.ndarray]
    signals: list[np.ndarray]
    time: np.ndarray
    codes: list[np.ndarray]
    fs: float 


@dataclass
class Specan:
    f: np.ndarray
    Pxx_dB: np.ndarray

@dataclass
class CodeWeights: 
    node: list[np.ndarray]
    edge: list[np.ndarray]

