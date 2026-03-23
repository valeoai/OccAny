# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# Random sampling under a constraint
# --------------------------------------------------------
import numpy as np
import torch


class BatchedRandomSampler:
    """ Random sampling under a constraint: each sample in the batch has the same feature, 
    which is chosen randomly from a known pool of 'features' for each batch.

    For instance, the 'feature' could be the image aspect-ratio.

    The index returned is a tuple (sample_idx, feat_idx).
    This sampler ensures that each series of `batch_size` indices has the same `feat_idx`.
    """

    def __init__(self, dataset, batch_size, pool_size, world_size=1, rank=0, drop_last=True):
        self.batch_size = batch_size
        self.pool_size = pool_size

        self.len_dataset = N = len(dataset)
        self.total_size = round_by(N, batch_size*world_size) if drop_last else N
        assert world_size == 1 or drop_last, 'must drop the last batch in distributed mode'

        # distributed sampler
        self.world_size = world_size
        self.rank = rank
        self.epoch = None

    def __len__(self):
        return self.total_size // self.world_size

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __iter__(self):
        # prepare RNG
        if self.epoch is None:
            assert self.world_size == 1 and self.rank == 0, 'use set_epoch() if distributed mode is used'
            seed = int(torch.empty((), dtype=torch.int64).random_().item())
        else:
            seed = self.epoch + 777
        rng = np.random.default_rng(seed=seed)

        # random indices (will restart from 0 if not drop_last)
        sample_idxs = np.arange(self.total_size)
        rng.shuffle(sample_idxs)

        # random feat_idxs (same across each batch)
        n_batches = (self.total_size+self.batch_size-1) // self.batch_size
        feat_idxs = rng.integers(self.pool_size, size=n_batches)
        feat_idxs = np.broadcast_to(feat_idxs[:, None], (n_batches, self.batch_size))
        feat_idxs = feat_idxs.ravel()[:self.total_size]

        # put them together
        idxs = np.c_[sample_idxs, feat_idxs]  # shape = (total_size, 2)

        # Distributed sampler: we select a subset of batches
        # make sure the slice for each node is aligned with batch_size
        size_per_proc = self.batch_size * ((self.total_size + self.world_size *
                                           self.batch_size-1) // (self.world_size * self.batch_size))
        idxs = idxs[self.rank*size_per_proc: (self.rank+1)*size_per_proc]

        yield from (tuple(idx) for idx in idxs)


def round_by(total, multiple, up=False):
    if up:
        total = total + multiple-1
    return (total//multiple) * multiple


class BatchedRandomSampleOccAny(BatchedRandomSampler):
    def __init__(self, dataset, 
                 batch_size,
                 num_of_aspect_ratios,
                 min_memory_num_views=2, 
                 max_memory_num_views=10, 
                 ray_map_prob=0.0, 
                 ray_map_idx=[],
                 world_size=1, rank=0, drop_last=True):
        super().__init__(dataset, batch_size, pool_size=None, world_size=world_size, rank=rank, drop_last=drop_last)
        self.num_of_aspect_ratios = num_of_aspect_ratios
        self.min_memory_num_views = min_memory_num_views
        self.max_memory_num_views = max_memory_num_views
        self.ray_map_prob = ray_map_prob
        self.ray_map_idx = ray_map_idx

    def __iter__(self):
        # prepare RNG
        if self.epoch is None:
            assert self.world_size == 1 and self.rank == 0, 'use set_epoch() if distributed mode is used'
            seed = int(torch.empty((), dtype=torch.int64).random_().item())
        else:
            seed = self.epoch + 777
        rng = np.random.default_rng(seed=seed)

        # random indices (will restart from 0 if not drop_last)
        sample_idxs = np.arange(self.total_size)
        rng.shuffle(sample_idxs)

 
        n_batches = (self.total_size + self.batch_size - 1) // self.batch_size
        resolution_idxs = rng.integers(self.num_of_aspect_ratios, size=n_batches)
        resolution_idxs = np.broadcast_to(resolution_idxs[:, None], (n_batches, self.batch_size))
        resolution_idxs = resolution_idxs.ravel()[:self.total_size]  

        memory_num_views = rng.integers(self.min_memory_num_views, self.max_memory_num_views+1, size=n_batches)
         
        
        list_ray_map_idx = []
        for i in range(n_batches):
            if len(self.ray_map_idx) > 0:
                ray_map_idx = self.ray_map_idx
            else:
                ray_map_idx = []
                for j in range(memory_num_views[i]):
                    # if (j != 0 and j != memory_num_views[i]-1) and rng.random() < self.ray_map_prob:
                    if j != 0 and rng.random() < self.ray_map_prob:                        
                        ray_map_idx.append(j)
            if len(ray_map_idx) == 0:
                ray_map_idx.append(rng.integers(1, memory_num_views[i]))
            
            if len(ray_map_idx) > 5:
                # Limit the number of gen views to 3
                ray_map_idx = sorted(rng.choice(ray_map_idx, size=5, replace=False).tolist())

            # Repeat the ray_map_idx for each item in the batch
            list_ray_map_idx.extend([ray_map_idx] * self.batch_size)

        
        memory_num_views = np.broadcast_to(memory_num_views[:, None], (n_batches, self.batch_size))
        memory_num_views = memory_num_views.ravel()[:self.total_size] 

        list_ray_map_idx = list_ray_map_idx[:self.total_size] 


        
        
        # random feat_idxs (same across each batch)
        # pool_size = self.pool_size if isinstance(self.pool_size, list) else [self.pool_size]
        # idxs = []
        # ray_map_idx = []
        # for pool_size in pool_size:
        #     n_batches = (self.total_size + self.batch_size - 1) // self.batch_size
        #     if isinstance(pool_size, tuple):
        #         feat_idxs = rng.integers(*pool_size, size=n_batches)
        #         for i in range(1, feat_idxs):
        #             if rng.random() < self.ray_map_prob:
        #                 ray_map_idx.append(i)
        #     else:
        #         feat_idxs = rng.integers(pool_size, size=n_batches)
        #     feat_idxs = np.broadcast_to(feat_idxs[:, None], (n_batches, self.batch_size))
        #     feat_idxs = feat_idxs.ravel()[:self.total_size]
        #     idxs.append(feat_idxs)

        # ray_map_idx.append(ray_map_idx)
        # put them together
        # breakpoint()
        # idxs = np.c_[sample_idxs, resolution_idxs, memory_num_views]  # shape = (total_size, n_feats)
        # print(idxs)
        # Distributed sampler: we select a subset of batches
        # make sure the slice for each node is aligned with batch_size
        size_per_proc = self.batch_size * ((self.total_size + self.world_size *
                                           self.batch_size - 1) // (self.world_size * self.batch_size))
        # idxs = idxs[self.rank * size_per_proc: (self.rank + 1) * size_per_proc]
        idxs = np.arange(self.rank * size_per_proc, (self.rank + 1) * size_per_proc)
        

        for i in idxs:
            yield (sample_idxs[i], resolution_idxs[i], memory_num_views[i], np.array(list_ray_map_idx[i]))


class DatasetAwareBatchSamplerOccAny(BatchedRandomSampler):
    def __init__(self, dataset,
                 batch_size,
                 dataset_configs,
                 world_size=1, rank=0, drop_last=True):
        super().__init__(dataset, batch_size, pool_size=None, world_size=world_size, rank=rank, drop_last=drop_last)
        self.dataset_configs, self.cum_sizes = dataset_configs

    def __iter__(self):
        if self.epoch is None:
            assert self.world_size == 1 and self.rank == 0, 'use set_epoch() if distributed mode is used'
            seed = int(torch.empty((), dtype=torch.int64).random_().item())
        else:
            seed = self.epoch + 777
        rng = np.random.default_rng(seed=seed)

        start_idx = 0
        all_batches = []

        for config, end_idx in zip(self.dataset_configs, self.cum_sizes):
            dataset_idxs = np.arange(start_idx, end_idx)
            rng.shuffle(dataset_idxs)

            num_samples = len(dataset_idxs)
            n_dataset_batches = num_samples // self.batch_size
            if n_dataset_batches == 0:
                start_idx = end_idx
                continue

            dataset_idxs = dataset_idxs[:n_dataset_batches * self.batch_size]
            dataset_idxs = dataset_idxs.reshape(n_dataset_batches, self.batch_size)

            num_of_aspect_ratios = config['num_of_aspect_ratios']
            min_memory_num_views = config['min_memory_num_views']
            max_memory_num_views = config['max_memory_num_views']
            for b_idx in range(n_dataset_batches):
                batch_sample_idxs = dataset_idxs[b_idx]

                res_idx = rng.integers(num_of_aspect_ratios)

                mem_views = rng.integers(min_memory_num_views, max_memory_num_views + 1)

                explicit_view_idx = config.get('ray_map_idx', [])
                view_prob = config.get('ray_map_prob', 0.0)

                if len(explicit_view_idx) > 0:
                    ray_map_idx = list(explicit_view_idx)
                else:
                    ray_map_idx = []
                    for j in range(mem_views):
                        if j != 0 and rng.random() < view_prob:
                            ray_map_idx.append(j)
                    if len(ray_map_idx) == 0:
                        ray_map_idx.append(rng.integers(1, mem_views))

                if len(ray_map_idx) > 5:
                    ray_map_idx = sorted(rng.choice(ray_map_idx, size=5, replace=False).tolist())

                batch_tuples = []
                for s_idx in batch_sample_idxs:
                    batch_tuples.append((s_idx, res_idx, mem_views, np.array(ray_map_idx)))
                all_batches.append(batch_tuples)

            start_idx = end_idx

        rng.shuffle(all_batches)

        total_batches = len(all_batches)
        size_per_proc = total_batches // self.world_size
        my_batches = all_batches[self.rank * size_per_proc: (self.rank + 1) * size_per_proc]

        for batch in my_batches:
            yield from batch


class BatchedRandomSamplerMust3r(BatchedRandomSampler):
    def __init__(self, dataset, batch_size, pool_size, ray_map_prob=0.0, world_size=1, rank=0, drop_last=True):
        super().__init__(dataset, batch_size, pool_size, world_size=world_size, rank=rank, drop_last=drop_last)
        self.ray_map_prob = ray_map_prob

    def __iter__(self):
        # prepare RNG
        if self.epoch is None:
            assert self.world_size == 1 and self.rank == 0, 'use set_epoch() if distributed mode is used'
            seed = int(torch.empty((), dtype=torch.int64).random_().item())
        else:
            seed = self.epoch + 777
        rng = np.random.default_rng(seed=seed)

        # random indices (will restart from 0 if not drop_last)
        sample_idxs = np.arange(self.total_size)
        rng.shuffle(sample_idxs)

        # random feat_idxs (same across each batch)
        pool_size = self.pool_size if isinstance(self.pool_size, list) else [self.pool_size]
        idxs = []
        ray_map_idx = []
        for pool_size in pool_size:
            n_batches = (self.total_size + self.batch_size - 1) // self.batch_size
            if isinstance(pool_size, tuple):
                feat_idxs = rng.integers(*pool_size, size=n_batches)
                for i in range(1, feat_idxs):
                    if rng.random() < self.ray_map_prob:
                        ray_map_idx.append(i)
            else:
                feat_idxs = rng.integers(pool_size, size=n_batches)
            feat_idxs = np.broadcast_to(feat_idxs[:, None], (n_batches, self.batch_size))
            feat_idxs = feat_idxs.ravel()[:self.total_size]
            idxs.append(feat_idxs)

        idxs.append(ray_map_idx)
        # put them together
        idxs = np.c_[sample_idxs, *idxs]  # shape = (total_size, n_feats)
        print(idxs)
        # Distributed sampler: we select a subset of batches
        # make sure the slice for each node is aligned with batch_size
        size_per_proc = self.batch_size * ((self.total_size + self.world_size *
                                           self.batch_size - 1) // (self.world_size * self.batch_size))
        idxs = idxs[self.rank * size_per_proc: (self.rank + 1) * size_per_proc]

        yield from (tuple(idx) for idx in idxs)

