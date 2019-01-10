from random import shuffle
from typing import List

import numpy as np

from python_research.experiments.sota_models.utils.list_dataset import ListDataset
from python_research.experiments.utils.datasets.hyperspectral_dataset import HyperspectralDataset


def attention_selection(data, args):
    """
    Selection of bands.
    """
    content = None
    with open(args.cont) as f:
        content = f.readlines()
        content = [int(x.rstrip('\n')) for x in content]
        content.sort()
        content = np.asarray(content, dtype=int)
    if content is not None:
        print('Reducing bands from: {}'.format(args.cont))
        data = [x[..., content] for x in data]
        print(content)
    return data


def generate_samples(args):
    samples = HyperspectralDataset(dataset=args.data_set, ground_truth=args.labels,
                                   neighbourhood_size=args.neighbourhood_size)
    samples.normalize_min_max()
    samples.normalize_labels()
    data = samples.get_data()
    if args.cont is not None:
        data = attention_selection(data=data, args=args)
    labels = samples.get_labels()
    samples = [x for x in zip(data, labels)]
    return samples


def prep_dataset(train_set: List, val_set: List, test_set: List):
    """
    Stores data sets as objects of type ListDataset, which are used in DataLoader.
    """
    shuffle(train_set), shuffle(val_set), shuffle(test_set)
    samples, labels = zip(*train_set)
    train_dataset = ListDataset(samples=samples, labels=labels)
    samples, labels = zip(*val_set)
    val_dataset = ListDataset(samples=samples, labels=labels)
    samples, labels = zip(*test_set)
    test_dataset = ListDataset(samples=samples, labels=labels)
    return train_dataset, val_dataset, test_dataset


def unravel_dataset(train_set: List[List[object]], val_set: List[List], test_set: List[List[object]]):
    """
    Unravels a data set from nested lists into one single list containing all items inside.
    """
    train_set = [item for sublist in train_set for item in sublist]
    val_set = [item for sublist in val_set for item in sublist]
    test_set = [item for sublist in test_set for item in sublist]
    return train_set, val_set, test_set