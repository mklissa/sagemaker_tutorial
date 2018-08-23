
class ContinuousBatchSampler():
    def __init__(self, sampler, batch_size, datalen, max_iters=None):
        self._sampler = sampler
        self._batch_size = batch_size
        self._max_iters = max_iters
        self._counter = 0
        iters_per_epoch = int(datalen/self._batch_size)
        self.last_batch = datalen - iters_per_epoch * batch_size

    def __iter__(self):
        batch = []
        epoch = 0
        batch_size = self._batch_size
        while True:
            for i in self._sampler:
                batch.append(i)
                if len(batch) == batch_size:
                    yield batch
                    self._counter += 1
                    if self._counter > self._max_iters:
                        raise StopIteration
                    batch = []
            yield batch
            self._counter += 1
            epoch +=1
            batch = []

