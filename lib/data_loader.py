import numpy as np

class DataLoader(object):
    def __init__(self, config):
        self.batch_size = config.batch_size

    def next_batch(self):
        return (np.ones((self.batch_size, 500, 500, 3), dtype=np.float), np.ones((self.batch_size, 50), dtype=np.int32))
