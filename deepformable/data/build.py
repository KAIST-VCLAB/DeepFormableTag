# Copyright (c) Mustafa B. Yaldiz (VCLAB, KAIST) All Rights Reserved.
import logging
import random

import torch
import torch.utils.data as torchdata

from detectron2.config import configurable
from detectron2.data.build import _train_loader_from_config, build_batch_data_loader
from detectron2.data.common import _MapIterableDataset, DatasetFromList
from detectron2.data.samplers import TrainingSampler
from detectron2.utils.serialize import PicklableWrapper


class MapDataset(torchdata.Dataset):
    """
    This method tries several times to map given data, if cannot picks
    another data. It is modified from original MapDataset located at Detectron2.    
    """
    def __init__(self, dataset, map_func, retry_count=10):
        self._dataset = dataset
        self._map_func = PicklableWrapper(map_func)  # wrap so that a lambda will work
        self._rng = random.Random(42)
        self.retry_count = retry_count

    def __new__(cls, dataset, map_func):
        is_iterable = isinstance(dataset, torchdata.IterableDataset)
        if is_iterable:
            return _MapIterableDataset(dataset, map_func)
        else:
            return super().__new__(cls)

    def __getnewargs__(self):
        return self._dataset, self._map_func

    def __len__(self):
        return len(self._dataset)

    def __getitem__(self, idx):
        cur_idx = int(idx)
        d = self._dataset[cur_idx]

        for _ in range(self.retry_count):
            data = self._map_func(d)
            if data is not None:
                return data
        
        logger = logging.getLogger(__name__)
        # This id should be same as image_id
        warn_str = "Failed to apply `_map_func` for idx: {}".format(idx)    
        logger.warning(warn_str)
            
        return self.__getitem__(self._rng.randint(0, self.__len__()-1))
        

@configurable(from_config=_train_loader_from_config)
def build_detection_train_loader(
    dataset, *, mapper, sampler=None, total_batch_size, aspect_ratio_grouping=True, num_workers=0
):
    """
    This method is modified to use our MapDataset implementation.
    """
    if isinstance(dataset, list):
        dataset = DatasetFromList(dataset, copy=False)
    if mapper is not None:
        dataset = MapDataset(dataset, mapper)

    """
    TODO: Can get output class id's and thing classes 
    from mapper and change metadata to eliminate problems here.
    """

    if isinstance(dataset, torchdata.IterableDataset):
        assert sampler is None, "sampler must be None if dataset is IterableDataset"
    else:
        if sampler is None:
            sampler = TrainingSampler(len(dataset))
        assert isinstance(sampler, torchdata.Sampler), f"Expect a Sampler but got {type(sampler)}"
    return build_batch_data_loader(
        dataset,
        sampler,
        total_batch_size,
        aspect_ratio_grouping=aspect_ratio_grouping,
        num_workers=num_workers,
    )
