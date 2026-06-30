from dataclasses import dataclass
import numpy as np

@dataclass
class ADCOutput:
    signals: list[np.ndarray]
    code: list[np.ndarray]
    time: np.ndarray
    fss: float



