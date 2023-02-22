import pandas as pd
import numpy as np
import torch


class Preprocessor:
    def __init__(self):
        pass

    @staticmethod
    def process(x: pd.DataFrame) -> torch.Tensor:
        x = torch.tensor(x.values[:, 3:].astype(np.float32))
        x = x / x.max(0, keepdim=True)[0]
        return x
