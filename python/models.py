from dataclasses import dataclass
import numpy as np


@dataclass
class DecorrOutput:
    signals: list[np.ndarray]
    time: np.ndarray
    fs: float

@dataclass
class SpreadOutput:
    signals: list[np.ndarray]
    time: np.ndarray
    codes: list[np.ndarray]
    fs: float

@dataclass
class Specan:
    f: np.ndarray
    Pxx_dB: np.ndarray
    bw: float

@dataclass
class CodeWeights: 
    node: list[np.ndarray]
    edge: list[np.ndarray]

