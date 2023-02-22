import torch

import src

import pandas as pd

if __name__ == '__main__':
    raw_data = pd.read_csv('tickdata_20221229.csv')

    # segment data frame into 500-length chunks
    chunk_indices = [
        (i, i + 500)
        for i in range(0, len(raw_data), 500)
    ]
    chunk_indices[-1] = (chunk_indices[-1][0], chunk_indices[-1][0] + len(raw_data) % 500)

    # prepare preprocessor and model
    preprocessor = src.Preprocessor()
    model = src.DemoModel(52)

    # bind optimizer
    optimizer = torch.optim.Adam(
        params=model.parameters(),
        lr=1e-3,
    )

    # training loops
    for i in range(len(chunk_indices) - 1):
        optimizer.zero_grad()
        src_idx, tgt_idx = chunk_indices[i], chunk_indices[i + 1]

        src_frame = raw_data[src_idx[0]: src_idx[1]]
        tgt_frame = raw_data[tgt_idx[0]: tgt_idx[1]]

        src_tensor = preprocessor.process(src_frame)
        tgt_tensor = preprocessor.process(tgt_frame)

        prediction = model(src_tensor)
        loss: torch.Tensor = torch.norm(tgt_tensor - prediction, 2)
        loss.backward()
        optimizer.step()

        print(f'current loss: {loss.item()}')
