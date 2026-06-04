import torch
import numpy as np
import numpy.typing as npt


def load_tokenized_data(path: str, dtype: np.dtype = np.uint16) -> np.memmap:
    return np.memmap(path, dtype=dtype, mode="r")


def get_batch(
        dataset: npt.NDArray,
        batch_size: int,
        context_length: int,
        device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:

    # 起始点是随机的，范围是 [0, len(dataset) - context_length)
    # 因为我们需要保证在取出 context_length 长度的序列时不会越界。
    start_indices = torch.randint(
        low = 0,
        high = len(dataset) - context_length,
        size = (batch_size,),
    )

    # 对于每个起始点，我们取出一个长度为 context_length 的序列作为输入 x，
    # torch.stack 用于将这些序列堆叠成一个批次的张量。
    x = torch.stack([
        torch.from_numpy(np.array(dataset[i : i + context_length], dtype=np.int64, copy=True))
        for i in start_indices.tolist()
    ])

    # label 的起始点是输入起始点的下一个位置
    # 因此我们取出从 i + 1 开始的长度为 context_length 的序列作为标签 y。
    y = torch.stack([
        torch.from_numpy(np.array(dataset[i + 1 : i + context_length + 1], dtype=np.int64, copy=True))
        for i in start_indices.tolist()
    ])

    x = x.to(device)
    y = y.to(device)

    return x, y
