from torch.utils.data import Dataset
from typing import (
    Sequence,
    TypeVar,
)

T_co = TypeVar('T_co', covariant=True)
T = TypeVar('T')

class PromptSubset(Dataset[T_co]):
    dataset: Dataset[T_co]
    indices: Sequence[int]

    def __init__(self, dataset: Dataset[T_co], indices: Sequence[int], prompt_indices: Sequence[int]) -> None:
        self.dataset = dataset
        self.indices = indices
        self.prompt_indices = prompt_indices

    def __getitem__(self, idx):
        if isinstance(idx, list):
            return self.dataset[[self.indices[i] for i in idx]], self.dataset[[self.prompt_indices[i] for i in idx]]
        return self.dataset[self.indices[idx]], self.dataset[self.prompt_indices[idx]]

    def __len__(self):
        return len(self.indices)

class TopKPromptSubset(Dataset[T_co]):
    dataset: Dataset[T_co]
    indices: Sequence[int]

    def __init__(self, dataset, indices, prompt_indices, topK) -> None:
        self.dataset = dataset
        self.indices = indices
        self.prompt_indices = prompt_indices
        self.topK = topK

    def __getitem__(self, idx):

        prompt_1st_indices = self.prompt_indices[:, -1].numpy().tolist()
        prompt_2nd_indices = self.prompt_indices[:, -2].numpy().tolist()
        prompt_3rd_indices = self.prompt_indices[:, -3].numpy().tolist()
        assert self.topK in [1,2,3], "only support top{1-3}"
        if self.topK == 1:
            return self.dataset[self.indices[idx]], self.dataset[prompt_1st_indices[idx]]
        elif self.topK == 2:
            return self.dataset[self.indices[idx]], self.dataset[prompt_1st_indices[idx]], self.dataset[prompt_2nd_indices[idx]]
        else:
            return self.dataset[self.indices[idx]], self.dataset[prompt_1st_indices[idx]], self.dataset[prompt_2nd_indices[idx]], self.dataset[prompt_3rd_indices[idx]]

    def __len__(self):
        return len(self.indices)