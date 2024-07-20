import torch
from torch.utils.data import Subset
import pickle
from .mySubset import TopKPromptSubset, PromptSubset
from .pl_pair_dataset import PocketLigandPairDataset

def get_dataset(config, *args, **kwargs):
    name = config.name
    root = config.path
    if name == 'pl':
        dataset = PocketLigandPairDataset(root, *args, **kwargs)
    else:
        raise NotImplementedError('Unknown dataset: %s' % name)

    if 'split' in config:
        split = torch.load(config.split)
        subsets = {k: Subset(dataset, indices=v) for k, v in split.items()}
        return dataset, subsets
    else:
        return dataset

def get_promt_dataset(config, *args, **kwargs):
    name = config.name
    root = config.path

    train_prompt_indices = torch.load(config.train_prompt_path)
    test_prompt_indices = torch.load(config.val_prompt_path)

    assert train_prompt_indices.shape[1] == test_prompt_indices.shape[1]

    assert 'split' in config
    split_indices_dict = torch.load(config.split)
    train_split_indices = split_indices_dict['train']
    test_split_indices = split_indices_dict['test']  # the 'key' of val_dataset is 'test'

    assert name == 'pl'
    dataset = PocketLigandPairDataset(root, *args, **kwargs)

    train_dataset = PromptSubset(dataset, indices=train_split_indices, prompt_indices=train_prompt_indices[:, -1].numpy().tolist())
    test_dataset = PromptSubset(dataset, indices=test_split_indices, prompt_indices=test_prompt_indices[:, -1].numpy().tolist())

    subsets = {'train': train_dataset, 'test': test_dataset}
    return subsets

def get_topk_promt_dataset(config, *args, **kwargs):
    name = config.name
    root = config.path

    train_prompt_indices = torch.load(config.train_prompt_path)
    test_prompt_indices = torch.load(config.val_prompt_path)

    assert train_prompt_indices.shape[1] == test_prompt_indices.shape[1], "the size of retrieval database is different"

    assert 'split' in config
    split_indices_dict = torch.load(config.split)
    train_split_indices = split_indices_dict['train']
    test_split_indices = split_indices_dict['test']  # the 'key' of val_dataset is 'test'

    assert name == 'pl'
    dataset = PocketLigandPairDataset(root, *args, **kwargs)

    topk_prompt = config.topk_prompt
    assert topk_prompt in [1,2,3]

    train_dataset = TopKPromptSubset(dataset, indices=train_split_indices, prompt_indices=train_prompt_indices, topK=topk_prompt)
    test_dataset = TopKPromptSubset(dataset, indices=test_split_indices, prompt_indices=test_prompt_indices, topK=topk_prompt)

    subsets = {'train': train_dataset, 'test': test_dataset}
    return subsets