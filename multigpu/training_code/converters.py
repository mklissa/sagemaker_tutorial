import mxnet as mx


class DataIterLoader():
    def __init__(self, data_iter):
        self.data_iter = data_iter

    def __iter__(self):
        self.data_iter.reset()
        return self

    def __next__(self):
        batch = self.data_iter.__next__()
        assert len(batch.data) == len(batch.label) == 1
        data = batch.data[0]
        label = batch.label[0]
        return data, label

    def next(self):
        return self.__next__() # for Python 2


class DataLoaderIter():
    def __init__(self, data_loader):
        self.data_loader = data_loader

    def __iter__(self):
        self.open_iter = self.data_loader.__iter__()
        return self

    def __next__(self):
        data, label = self.open_iter.__next__()
        data_desc = mx.io.DataDesc(name='data', shape=data.shape, dtype=data.dtype)
        label_desc = mx.io.DataDesc(name='label', shape=label.shape, dtype=label.dtype)
        batch = mx.io.DataBatch(data=[data], label=[label], provide_data=[data_desc], provide_label=[label_desc])
        return batch

    def next(self):
        return self.__next__() # for Python 2