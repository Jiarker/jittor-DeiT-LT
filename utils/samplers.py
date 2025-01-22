import jittor as jt
import numpy as np

class RASampler(jt.dataset.Sampler):
    def __init__(self, dataset, num_replicas=1, rank=0, shuffle=True, num_repeats=3):
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.num_repeats = num_repeats
        self.epoch = 0
        self.shuffle = shuffle
        self.num_samples = int(np.ceil(len(self.dataset) * self.num_repeats / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas
        self.num_selected_samples = int(np.floor(len(self.dataset) // 256 * 256 / self.num_replicas))

    def __iter__(self):
        if self.shuffle:
            # deterministically shuffle based on epoch
            rng = np.random.default_rng(self.epoch)
            indices = rng.permutation(len(self.dataset)).tolist()
        else:
            indices = list(range(len(self.dataset)))

        # add extra samples to make it evenly divisible
        indices = np.repeat(indices, repeats=self.num_repeats).tolist()
        padding_size = self.total_size - len(indices)
        if padding_size > 0:
            indices += indices[:padding_size]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices[:self.num_selected_samples])

    def __len__(self):
        return self.num_selected_samples

    def set_epoch(self, epoch):
        self.epoch = epoch

# Example usage:
# dataset = YourDataset()  # Replace with your dataset
# sampler = RASampler(dataset, num_replicas=4, rank=0, shuffle=True, num_repeats=3)
# for indices in sampler:
#     # Load data at indices
#     pass