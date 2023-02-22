import pandas as pd

from src.demo_model import DemoModel
from src.preprocessor import Preprocessor

import torch


def test_demo_model():
    batch_x = torch.tensor(
        [
            [1, 5, 100, 9000, 674826784567294],
            [4, 5, 768, 2734, 253752398475922],
            [2, 3, 234, 3245, 324523452345234],
            [2, 3, 234, 3245, 324523452345234],
        ],
        dtype=torch.float,
    )
    model = DemoModel(5)
    print(model(batch_x))


def test_preprocessor():
    df = pd.DataFrame(
        [
            [0, 1, 2, 3, 4],
            [5, 6, 7, 8, 9],
        ]
    )
    preprocessor = Preprocessor()
    print(preprocessor.process(df))


if __name__ == '__main__':
    test_demo_model()
    test_preprocessor()
